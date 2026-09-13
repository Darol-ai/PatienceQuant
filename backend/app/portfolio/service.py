from __future__ import annotations

from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.data.service import MarketDataService
from app.db.models import DailyAccount, DailyProfit, Order, Portfolio, Position, Signal, Strategy, Trade
from app.quant_v3.csi300_strategies import BUILDERS as CSI300_STRATEGY_BUILDERS
from app.quant_v3.csi300_universe import csi300_stocks
from app.factors.engine import FactorEngine
from app.strategies.base import StrategyConfig
from app.strategies.multifactor import MultiFactorStrategy


class PaperTradingService:
    def __init__(self, db: Session, data: MarketDataService):
        self.db = db
        self.data = data

    def get_or_create_portfolio(self) -> Portfolio:
        portfolio = self.db.scalar(select(Portfolio).order_by(Portfolio.id).limit(1))
        if not portfolio:
            portfolio = Portfolio(name="默认模拟组合", initial_capital=1_000_000, cash=1_000_000)
            self.db.add(portfolio)
            self.db.commit()
            self.db.refresh(portfolio)
        return portfolio

    def reset(self) -> Portfolio:
        portfolio = self.get_or_create_portfolio()
        self.db.execute(delete(Position).where(Position.portfolio_id == portfolio.id))
        self.db.execute(delete(Order).where(Order.portfolio_id == portfolio.id))
        self.db.execute(delete(Trade).where(Trade.portfolio_id == portfolio.id))
        self.db.execute(delete(DailyAccount).where(DailyAccount.portfolio_id == portfolio.id))
        self.db.execute(delete(DailyProfit).where(DailyProfit.portfolio_id == portfolio.id))
        portfolio.cash = portfolio.initial_capital
        portfolio.strategy_id = None
        # Resetting the account also clears a backtest-specific execution
        # scope.  A subsequent manual rebalance will use the selected
        # strategy's named universe unless a new backtest is applied.
        if hasattr(portfolio, "execution_universe"):
            portfolio.execution_universe = "large_cap"
        if hasattr(portfolio, "execution_symbols"):
            portfolio.execution_symbols = []
        if hasattr(portfolio, "source_backtest_run_id"):
            portfolio.source_backtest_run_id = None
        portfolio.last_rebalance_at = None
        portfolio.auto_rebalance_enabled = False
        portfolio.next_rebalance_date = None
        portfolio.last_auto_run_at = None
        self.db.commit()
        return portfolio

    @staticmethod
    def next_rebalance_date(as_of: date, frequency: str) -> date:
        if frequency == "weekly":
            return as_of + timedelta(days=7)
        if frequency == "quarterly":
            month = ((as_of.month - 1) // 3 + 1) * 3 + 1
            year = as_of.year + (1 if month > 12 else 0)
            month = month if month <= 12 else month - 12
            return date(year, month, min(as_of.day, monthrange(year, month)[1]))
        month = as_of.month + 1
        year = as_of.year + (1 if month == 13 else 0)
        month = 1 if month == 13 else month
        return date(year, month, min(as_of.day, monthrange(year, month)[1]))

    def automation_status(self) -> Dict[str, object]:
        portfolio = self.get_or_create_portfolio()
        return {
            "enabled": bool(portfolio.auto_rebalance_enabled),
            "strategy_id": portfolio.strategy_id,
            "frequency": portfolio.auto_rebalance_frequency,
            "next_rebalance_date": portfolio.next_rebalance_date,
            "last_auto_run_at": portfolio.last_auto_run_at,
            "due": bool(portfolio.auto_rebalance_enabled and portfolio.next_rebalance_date and portfolio.next_rebalance_date <= self.data.demo.as_of),
        }

    def configure_automation(self, strategy: Strategy, enabled: bool, frequency: str) -> Dict[str, object]:
        portfolio = self.get_or_create_portfolio()
        portfolio.strategy_id = strategy.id
        portfolio.auto_rebalance_enabled = enabled
        portfolio.auto_rebalance_frequency = frequency
        # Enabling automation schedules an initial run for the current demo
        # date; subsequent runs advance by the selected frequency.
        portfolio.next_rebalance_date = self.data.demo.as_of if enabled else None
        self.db.commit()
        return self.automation_status()

    def run_due_automation(self, as_of: Optional[date] = None) -> Dict[str, object]:
        as_of = as_of or self.data.demo.as_of
        portfolio = self.get_or_create_portfolio()
        if not portfolio.auto_rebalance_enabled or not portfolio.strategy_id or not portfolio.next_rebalance_date or portfolio.next_rebalance_date > as_of:
            return {"executed": False, "reason": "not_due", "status": self.automation_status()}
        strategy = self.db.get(Strategy, portfolio.strategy_id)
        if not strategy:
            return {"executed": False, "reason": "strategy_missing", "status": self.automation_status()}
        result = self.rebalance(strategy, as_of)
        portfolio = self.get_or_create_portfolio()
        portfolio.last_auto_run_at = datetime.utcnow()
        portfolio.next_rebalance_date = self.next_rebalance_date(as_of, portfolio.auto_rebalance_frequency)
        self.db.commit()
        result["executed"] = True
        result["automation"] = self.automation_status()
        return result

    def _resolve_paper_data(self, strategy: Optional[Strategy]):
        """自由探索阶段最终确定的LightGBM策略(kind="quant_v3_regression")
        和沪深300 universe版本的三个策略(kind="csi300_*")都走独立的
        parquet数据管道，不是SQLite/MarketDataService那套通用股票目录——
        它们的持仓symbol是"600519.SH"这种带交易所后缀的格式，SQLite目录
        里存的是不带后缀的"600519"，两边对不上，模拟盘查当前价格/公司名
        必须按持仓所属的策略切到对应的数据源，不能一直用构造函数传进来
        的那个。
        """
        if strategy is None:
            return self.data
        if strategy.kind == "quant_v3_regression":
            from app.quant_v3.a_phase_data_service import APhaseDataService
            from app.quant_v3.final_strategy import final_strategy_history
            return APhaseDataService(final_strategy_history())
        if strategy.kind in CSI300_STRATEGY_BUILDERS:
            from app.quant_v3.a_phase_data_service import APhaseDataService
            from app.quant_v3.csi300_strategies import csi300_history
            return APhaseDataService(csi300_history())
        return self.data

    def snapshot(self, as_of: Optional[date] = None) -> Dict[str, object]:
        portfolio = self.get_or_create_portfolio()
        strategy = self.db.get(Strategy, portfolio.strategy_id) if portfolio.strategy_id else None
        data = self._resolve_paper_data(strategy)
        is_quant_v3 = data is not self.data
        if is_quant_v3:
            from app.quant_v3.final_strategy import latest_trading_day_on_or_before
            as_of = as_of or latest_trading_day_on_or_before(date.today())
        else:
            as_of = as_of or self.data.demo.as_of
        positions = self.db.scalars(select(Position).where(Position.portfolio_id == portfolio.id)).all()
        catalog = data.stocks().set_index("symbol")
        symbols = [position.symbol for position in positions]
        # Paper positions are an explicit user-visible market-data request,
        # so selected AKShare symbols may refresh their current price. The
        # service still falls back to deterministic Demo prices on failure.
        price_df = data.prices(symbols, date(2018, 1, 1), as_of, allow_network=True) if symbols else pd.DataFrame()
        latest = price_df.sort_values("trade_date").groupby("symbol").tail(1).set_index("symbol") if not price_df.empty else pd.DataFrame()
        items = []
        market_value = 0.0
        for position in positions:
            current_price = float(latest.loc[position.symbol].adj_close) if position.symbol in latest.index else position.avg_cost
            value = position.quantity * current_price
            pnl = (current_price - position.avg_cost) * position.quantity
            market_value += value
            # catalog列(name/industry)只有MarketDataService的通用目录才有，
            # 我们自己的parquet数据源只有symbol这一列——按列是否存在兜底，
            # 不能假设这两列一定在。
            name = catalog.loc[position.symbol]["name"] if position.symbol in catalog.index and "name" in catalog.columns else position.symbol
            industry = catalog.loc[position.symbol]["industry"] if position.symbol in catalog.index and "industry" in catalog.columns else ""
            items.append({"symbol": position.symbol, "name": name, "industry": industry, "quantity": position.quantity, "avg_cost": position.avg_cost, "current_price": current_price, "market_value": value, "pnl": pnl, "pnl_pct": pnl / max(position.avg_cost * position.quantity, .01)})
        total_assets = portfolio.cash + market_value
        initial = portfolio.initial_capital
        daily = self.db.scalar(select(DailyAccount).where(DailyAccount.portfolio_id == portfolio.id).order_by(DailyAccount.trade_date.desc()).limit(1))
        previous = daily.total_assets if daily else initial
        execution_symbols = getattr(portfolio, "execution_symbols", None) or []
        if isinstance(execution_symbols, str):
            try:
                import json
                execution_symbols = json.loads(execution_symbols)
            except (TypeError, ValueError):
                execution_symbols = []
        return {
            "portfolio_id": portfolio.id,
            "cash": portfolio.cash,
            "market_value": market_value,
            "total_assets": total_assets,
            "initial_capital": initial,
            "cumulative_return": total_assets / initial - 1,
            "daily_return": total_assets / previous - 1 if previous else 0,
            "positions": items,
            "last_rebalance_at": portfolio.last_rebalance_at,
            "strategy_id": portfolio.strategy_id,
            "strategy_name": strategy.name if strategy else None,
            "strategy_kind": strategy.kind if strategy else None,
            "execution_universe": getattr(portfolio, "execution_universe", None) or (strategy.universe if strategy else "large_cap"),
            "execution_symbols": execution_symbols,
            "source_backtest_run_id": getattr(portfolio, "source_backtest_run_id", None),
            "risk_controls": (
                {
                    "target_volatility": float(strategy.target_volatility if strategy and strategy.target_volatility is not None else .22),
                    "max_drawdown_budget": float(strategy.max_drawdown_budget if strategy and strategy.max_drawdown_budget is not None else .15),
                    "drawdown_brake_exposure": float(strategy.drawdown_brake_exposure if strategy and strategy.drawdown_brake_exposure is not None else .50),
                }
                if strategy
                else None
            ),
        }

    def rebalance(
        self,
        strategy: Strategy,
        as_of: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        universe: Optional[str] = None,
        source_backtest_run_id: Optional[int] = None,
    ) -> Dict[str, object]:
        portfolio = self.get_or_create_portfolio()
        data = self._resolve_paper_data(strategy)
        is_quant_v3 = data is not self.data
        if is_quant_v3:
            # 我们自己的策略(30支候选池的LightGBM，或沪深300 universe的
            # 三个策略)都固定用各自的universe，不接受universe/自定义股票
            # 池这些通用参数——和 /api/backtests 的分流逻辑保持一致。
            from app.quant_v3.csi300_strategies import validate_csi300_date_range
            from app.quant_v3.final_strategy import latest_trading_day_on_or_before, validate_backtest_date_range
            as_of = as_of or date.today()
            try:
                if strategy.kind in CSI300_STRATEGY_BUILDERS:
                    validate_csi300_date_range(as_of, as_of)
                else:
                    validate_backtest_date_range(as_of, as_of)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            # `as_of` (today's wall-clock date by default) isn't necessarily
            # a trading day — the signal sources need an exact bar match and
            # have no fallback, so a weekend/holiday date silently produces
            # zero scores for every symbol instead of an error. Snap to the
            # most recent real trading day first.
            as_of = latest_trading_day_on_or_before(as_of)
            if strategy.kind in CSI300_STRATEGY_BUILDERS:
                symbols = [stock["symbol"] for stock in csi300_stocks()]
            else:
                from app.quant_v3.broad_universe import BROAD_STOCKS
                symbols = [stock["symbol"] for stock in BROAD_STOCKS]
            portfolio.execution_symbols = []
            portfolio.source_backtest_run_id = None
            portfolio.execution_universe = "custom"
        else:
            as_of = as_of or self.data.demo.as_of
            # Follow the saved strategy universe so a Pink Sheets strategy is
            # executed against the same market it was researched on.  The default
            # strategy remains the liquid large-cap A-share basket.
            previous_strategy_id = portfolio.strategy_id
            requested_symbols = list(dict.fromkeys(str(symbol).strip() for symbol in (symbols or []) if str(symbol).strip()))
            saved_symbols = getattr(portfolio, "execution_symbols", None) or []
            if isinstance(saved_symbols, str):
                try:
                    import json
                    saved_symbols = json.loads(saved_symbols)
                except (TypeError, ValueError):
                    saved_symbols = []
            if requested_symbols:
                symbols = requested_symbols
            elif saved_symbols and previous_strategy_id == strategy.id and not universe:
                symbols = list(dict.fromkeys(str(symbol) for symbol in saved_symbols))
            else:
                symbols = self.data.universe_symbols(universe or strategy.universe or "large_cap")
                # Switching to another strategy without an explicit custom scope
                # must not accidentally inherit the previous backtest's symbols.
                if not requested_symbols:
                    portfolio.execution_symbols = []
                    portfolio.source_backtest_run_id = None
                    portfolio.execution_universe = universe or strategy.universe or "large_cap"
        if len(symbols) < strategy.holdings_count:
            # A manually selected pool can be smaller than a legacy strategy
            # setting.  Keep the strategy executable while preserving the
            # hard ten-stock minimum at the API boundary.
            strategy_holdings = min(strategy.holdings_count, len(symbols))
        else:
            strategy_holdings = strategy.holdings_count
        if len(symbols) < 10:
            raise ValueError("模拟盘执行股票池至少需要 10 只股票")
        if strategy_holdings * strategy.max_weight < 1:
            raise ValueError("当前模拟盘股票池与单股权重上限无法构建组合")
        if not is_quant_v3:
            # is_quant_v3 分支已经在上面把这几个字段设成固定值了，这里的
            # "沿用上次自定义股票池"逻辑是给通用策略用的，对我们的策略
            # 不适用（也没有 requested_symbols/saved_symbols 这些变量）。
            if requested_symbols or universe:
                portfolio.execution_symbols = requested_symbols
                portfolio.execution_universe = universe or strategy.universe or "large_cap"
                portfolio.source_backtest_run_id = source_backtest_run_id
            elif not getattr(portfolio, "execution_universe", None):
                portfolio.execution_universe = strategy.universe or "large_cap"
            if requested_symbols:
                portfolio.execution_symbols = requested_symbols
            elif not saved_symbols:
                portfolio.execution_symbols = []
        config = StrategyConfig(
            name=strategy.name,
            weights=strategy.weights,
            holdings_count=strategy.holdings_count,
            max_weight=strategy.max_weight,
            rebalance_frequency=strategy.rebalance_frequency,
            cash_buffer=float(strategy.cash_buffer if strategy.cash_buffer is not None else .02),
            trend_filter=bool(strategy.trend_filter if strategy.trend_filter is not None else True),
            risk_off_exposure=float(strategy.risk_off_exposure if strategy.risk_off_exposure is not None else .75),
            turnover_band=float(strategy.turnover_band if strategy.turnover_band is not None else .03),
            stop_loss=float(strategy.stop_loss if strategy.stop_loss is not None else .18),
            benchmark_enhancement=bool(strategy.benchmark_enhancement if strategy.benchmark_enhancement is not None else True),
            dip_buy_strength=float(strategy.dip_buy_strength if strategy.dip_buy_strength is not None else .06),
            profit_take_strength=float(strategy.profit_take_strength if strategy.profit_take_strength is not None else .05),
            target_volatility=float(strategy.target_volatility if strategy.target_volatility is not None else .22),
            max_drawdown_budget=float(strategy.max_drawdown_budget if strategy.max_drawdown_budget is not None else .15),
            drawdown_brake_exposure=float(strategy.drawdown_brake_exposure if strategy.drawdown_brake_exposure is not None else .50),
        )
        config.holdings_count = strategy_holdings
        if is_quant_v3:
            if strategy.kind in CSI300_STRATEGY_BUILDERS:
                strategy_result = CSI300_STRATEGY_BUILDERS[strategy.kind]().generate_weights(as_of, symbols)
            else:
                from app.quant_v3.final_strategy import build_final_strategy
                strategy_result = build_final_strategy().generate_weights(as_of, symbols)
        else:
            strategy_result = MultiFactorStrategy(FactorEngine(self.data), config).generate_weights(as_of, symbols)
        snapshot = self.snapshot(as_of)
        total_assets = float(snapshot["total_assets"])
        # Apply the same portfolio-level risk budget used by the backtest.
        # Paper trading only uses snapshots already recorded in this account,
        # so the scale is auditable and never forecasts with future prices.
        history_rows = self.db.scalars(
            select(DailyAccount)
            .where(DailyAccount.portfolio_id == portfolio.id)
            .order_by(DailyAccount.trade_date)
        ).all()
        history_values = [float(row.total_assets) for row in history_rows if row.total_assets and row.total_assets > 0]
        prior_peak = max([float(portfolio.initial_capital), *history_values], default=float(portfolio.initial_capital))
        current_drawdown = total_assets / max(prior_peak, 1.0) - 1.0
        realized_vol = 0.0
        if config.target_volatility > 0 and len(history_values) >= 21:
            realized_vol = float(pd.Series(history_values, dtype=float).pct_change().dropna().tail(63).std() * (252 ** .5))
        risk_scale = 1.0
        drawdown_triggered = (
            config.max_drawdown_budget > 0
            and current_drawdown <= -config.max_drawdown_budget
        )
        if drawdown_triggered:
            risk_scale = min(risk_scale, config.drawdown_brake_exposure)
        volatility_triggered = bool(
            config.target_volatility > 0
            and realized_vol > config.target_volatility
        )
        if volatility_triggered:
            risk_scale = min(risk_scale, config.target_volatility / realized_vol)
        if risk_scale < 1.0:
            strategy_result.weights = {
                symbol: float(weight) * risk_scale
                for symbol, weight in strategy_result.weights.items()
            }
            strategy_result.target_exposure = float(strategy_result.target_exposure * risk_scale)
            strategy_result.ranking["target_weight"] = (
                strategy_result.ranking.symbol.map(strategy_result.weights).fillna(0.0)
            )
            strategy_result.ranking["action"] = strategy_result.ranking.target_weight.map(
                lambda weight: "BUY" if weight > 0 else "WATCH"
            )
        current_positions = {
            p.symbol: p
            for p in self.db.scalars(select(Position).where(Position.portfolio_id == portfolio.id)).all()
        }
        price_symbols = list(dict.fromkeys(symbols + list(current_positions)))
        prices = data.prices(price_symbols, as_of, as_of, allow_network=True)
        if prices.empty:
            prices = data.prices(price_symbols, date(2018, 1, 1), as_of, allow_network=True)
        latest = prices.sort_values("trade_date").groupby("symbol").tail(1).set_index("symbol")
        orders: List[Dict[str, object]] = []
        # Sell positions that are no longer in the target, then trim overweight positions.
        desired_qty: Dict[str, int] = {}
        stop_out_symbols = set()
        for symbol, weight in strategy_result.weights.items():
            if symbol not in latest.index:
                continue
            price = float(latest.loc[symbol].adj_close)
            position = current_positions.get(symbol)
            current_qty = position.quantity if position else 0
            if (
                position
                and current_qty > 0
                and config.stop_loss > 0
                and price <= position.avg_cost * (1 - config.stop_loss)
            ):
                desired_qty[symbol] = 0
                stop_out_symbols.add(symbol)
                continue
            current_weight = current_qty * price / max(total_assets, 1.0)
            if (
                position
                and current_qty > 0
                and config.turnover_band > 0
                and abs(float(weight) - current_weight) < config.turnover_band
            ):
                desired_qty[symbol] = current_qty
            else:
                desired_qty[symbol] = int((total_assets * weight) / max(price, .01) / 100) * 100
        for symbol, position in current_positions.items():
            desired_qty.setdefault(symbol, 0)
        for symbol, position in current_positions.items():
            target_qty = desired_qty.get(symbol, 0)
            delta = target_qty - position.quantity
            if delta >= 0:
                continue
            price = float(latest.loc[symbol].adj_close)
            quantity = min(position.quantity, abs(delta))
            amount = quantity * price
            fee = amount * .001
            portfolio.cash += amount - fee
            position.quantity -= quantity
            position.updated_at = datetime.utcnow()
            if position.quantity == 0:
                self.db.delete(position)
            reason = (
                "保护性止损 %.0f%%" % (config.stop_loss * 100)
                if symbol in stop_out_symbols
                else "综合评分跌出 Top %s" % strategy.holdings_count
            )
            if drawdown_triggered:
                reason += " · 回撤刹车"
            if volatility_triggered:
                reason += " · 波动率缩放"
            order = Order(portfolio_id=portfolio.id, symbol=symbol, side="SELL", quantity=quantity, price=price, amount=amount, fee=fee, strategy_name=strategy.name, reason=reason)
            self.db.add(order)
            self.db.add(Trade(portfolio_id=portfolio.id, symbol=symbol, trade_date=as_of, side="SELL", quantity=quantity, price=price, amount=amount, fee=fee, reason=order.reason))
            orders.append({"symbol": symbol, "side": "SELL", "quantity": quantity, "price": price, "amount": amount, "reason": order.reason})
        self.db.flush()
        for symbol, target_qty in desired_qty.items():
            position = self.db.scalar(select(Position).where(Position.portfolio_id == portfolio.id, Position.symbol == symbol))
            existing_qty = position.quantity if position else 0
            delta = target_qty - existing_qty
            if delta <= 0:
                continue
            if symbol in stop_out_symbols:
                continue
            price = float(latest.loc[symbol].adj_close)
            affordable = int(portfolio.cash / (price * 1.001))
            quantity = min(delta, affordable)
            if quantity <= 0:
                continue
            amount = quantity * price
            fee = amount * .001
            portfolio.cash -= amount + fee
            if position:
                position.avg_cost = (position.avg_cost * position.quantity + amount) / (position.quantity + quantity)
                position.quantity += quantity
                position.updated_at = datetime.utcnow()
            else:
                self.db.add(Position(portfolio_id=portfolio.id, symbol=symbol, quantity=quantity, avg_cost=price))
            reason = "综合评分进入 Top %s" % strategy.holdings_count
            if drawdown_triggered:
                reason += " · 回撤刹车"
            if volatility_triggered:
                reason += " · 波动率缩放"
            order = Order(portfolio_id=portfolio.id, symbol=symbol, side="BUY", quantity=quantity, price=price, amount=amount, fee=fee, strategy_name=strategy.name, reason=reason)
            self.db.add(order)
            self.db.add(Trade(portfolio_id=portfolio.id, symbol=symbol, trade_date=as_of, side="BUY", quantity=quantity, price=price, amount=amount, fee=fee, reason=reason))
            orders.append({"symbol": symbol, "side": "BUY", "quantity": quantity, "price": price, "amount": amount, "reason": reason})
        portfolio.strategy_id = strategy.id
        portfolio.last_rebalance_at = datetime.utcnow()
        if portfolio.auto_rebalance_enabled:
            portfolio.auto_rebalance_frequency = strategy.rebalance_frequency
            portfolio.next_rebalance_date = self.next_rebalance_date(as_of, portfolio.auto_rebalance_frequency)
        held_after = {position.symbol for position in self.db.scalars(select(Position).where(Position.portfolio_id == portfolio.id)).all()}
        for row in strategy_result.ranking.head(30).to_dict(orient="records"):
            # RegressionRotationStrategy的ranking没有"action"列(那是
            # MultiFactorStrategy特有的)，也可能给出score=None(信号源
            # 缺数据时的诚实占位，不是bug)——按列/值是否存在兜底，不能
            # 直接假设都在。
            default_action = "BUY" if row.get("target_weight", 0) > 0 else "WATCH"
            action = "HOLD" if row["symbol"] in held_after and row["target_weight"] > 0 else row.get("action", default_action)
            self.db.add(Signal(strategy_id=strategy.id, symbol=row["symbol"], signal_date=as_of, action=action,
                               score=float(row.get("score") or 0.0), target_weight=float(row["target_weight"]),
                               reason="综合评分第 %s 名，目标权重 %.1f%%" % (row["rank"], row["target_weight"] * 100)))
        self.db.commit()
        snapshot = self.snapshot(as_of)
        account = self.db.scalar(select(DailyAccount).where(DailyAccount.portfolio_id == portfolio.id, DailyAccount.trade_date == as_of))
        if account is None:
            account = DailyAccount(portfolio_id=portfolio.id, trade_date=as_of)
            self.db.add(account)
        account.total_assets = float(snapshot["total_assets"])
        account.cash = float(snapshot["cash"])
        account.market_value = float(snapshot["market_value"])
        account.daily_return = float(snapshot["daily_return"])
        account.cumulative_return = float(snapshot["cumulative_return"])
        profit = self.db.scalar(select(DailyProfit).where(DailyProfit.portfolio_id == portfolio.id, DailyProfit.trade_date == as_of))
        if profit is None:
            profit = DailyProfit(portfolio_id=portfolio.id, trade_date=as_of)
            self.db.add(profit)
        profit.realized_profit = 0.0
        profit.unrealized_profit = sum(float(position["pnl"]) for position in snapshot["positions"])
        self.db.commit()
        return {
            "snapshot": self.snapshot(as_of),
            "orders": orders,
            "signals": strategy_result.ranking.head(20).to_dict(orient="records"),
            "notes": strategy_result.data_quality_notes,
            "execution_scope": {
                "universe": getattr(portfolio, "execution_universe", None) or strategy.universe or "large_cap",
                "symbols": symbols,
                "source_backtest_run_id": getattr(portfolio, "source_backtest_run_id", None),
            },
            "risk_controls": {
                "target_volatility": config.target_volatility,
                "max_drawdown_budget": config.max_drawdown_budget,
                "drawdown_brake_exposure": config.drawdown_brake_exposure,
                "current_drawdown": current_drawdown,
                "realized_volatility": realized_vol,
                "risk_scale": risk_scale,
                "drawdown_brake_triggered": drawdown_triggered,
                "volatility_scaled": volatility_triggered,
            },
        }
