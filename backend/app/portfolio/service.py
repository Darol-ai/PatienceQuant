from __future__ import annotations

from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.data.service import MarketDataService
from app.db.models import DailyAccount, DailyProfit, Order, Portfolio, Position, Signal, Strategy, Trade
from app.pipeline.library import FIXED_UNIVERSES, fixed_pool, default_universe, is_model_strategy, latest_market_day, strategy_spec, validate_date_range
from app.pipeline.strategy import build_pipeline_strategy


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
            portfolio.execution_universe = "csi300"
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
        if portfolio.auto_rebalance_enabled and portfolio.strategy_id and portfolio.next_rebalance_date and portfolio.next_rebalance_date > as_of:
            timing_result = self._timing_adjustment(portfolio, as_of)
            if timing_result is not None:
                return timing_result
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

    def _timing_adjustment(self, portfolio: Portfolio, as_of: date) -> Optional[Dict[str, object]]:
        """不是调仓日，但择时信号给的仓位和上次调仓时不同：按当前策略重新调仓一次
        （ADR-0052：择时每天生效）。下次定期调仓日不变。"""
        strategy = self.db.get(Strategy, portfolio.strategy_id)
        if strategy is None or strategy_spec(strategy).timing.type == "none":
            return None
        data = self._resolve_paper_data(strategy)
        day = latest_market_day(as_of) if data is not self.data else as_of
        exposure = build_pipeline_strategy(strategy_spec(strategy), data).daily_exposure(day)
        if exposure is None or portfolio.last_exposure is None or abs(exposure[0] - portfolio.last_exposure) <= 1e-9:
            return None
        next_date = portfolio.next_rebalance_date
        result = self.rebalance(strategy, as_of)
        portfolio = self.get_or_create_portfolio()
        portfolio.next_rebalance_date = next_date
        self.db.commit()
        result.update(executed=True, reason="timing_changed", automation=self.automation_status())
        return result

    def _resolve_paper_data(self, strategy: Optional[Strategy]) -> MarketDataService:
        """模型选股策略只在真实行情上有意义：不管 DATA_MODE 怎么设，都读本地行情库。"""
        if strategy is not None and is_model_strategy(strategy):
            return MarketDataService(self.db, real_market_data=True)
        return self.data

    def snapshot(self, as_of: Optional[date] = None) -> Dict[str, object]:
        portfolio = self.get_or_create_portfolio()
        strategy = self.db.get(Strategy, portfolio.strategy_id) if portfolio.strategy_id else None
        data = self._resolve_paper_data(strategy)
        if data is not self.data:
            as_of = as_of or latest_market_day(date.today())
        else:
            as_of = as_of or self.data.demo.as_of
        positions = self.db.scalars(select(Position).where(Position.portfolio_id == portfolio.id)).all()
        catalog = data.stocks().set_index("symbol")
        symbols = [position.symbol for position in positions]
        # Paper positions are an explicit user-visible market-data request,
        # so selected tushare symbols may refresh their current price. The
        # service still falls back to deterministic Demo prices on failure.
        # 只要最新价：取最近一个月就够（前复权锚定在 as_of，最后一天就是真实收盘价）
        price_df = data.prices(symbols, as_of - timedelta(days=30), as_of) if symbols else pd.DataFrame()
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
            "execution_universe": getattr(portfolio, "execution_universe", None) or (strategy.universe if strategy else "csi300"),
            "execution_symbols": execution_symbols,
            "source_backtest_run_id": getattr(portfolio, "source_backtest_run_id", None),
            "strategy_spec": strategy_spec(strategy).model_dump(mode="json") if strategy else None,
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
        spec = strategy_spec(strategy)
        if data is not self.data:
            # 周末/节假日/当天行情还没发布时落到本地行情库里最近的交易日
            as_of = latest_market_day(as_of or date.today())
        else:
            as_of = as_of or self.data.demo.as_of
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
            portfolio.execution_symbols = requested_symbols
            portfolio.execution_universe = universe or "custom"
            portfolio.source_backtest_run_id = source_backtest_run_id
        elif saved_symbols and previous_strategy_id == strategy.id and not universe:
            symbols = list(dict.fromkeys(str(symbol) for symbol in saved_symbols))
        else:
            # 换策略又没指定股票池时，不能沿用上一个回测的股票池
            universe_name = universe or default_universe(strategy)
            # 沪深300 用调仓当天的成分股（历史成分股，ADR-0052）
            symbols = (fixed_pool(universe_name, as_of, as_of).symbols if universe_name in FIXED_UNIVERSES
                       else data.universe_symbols(universe_name))
            portfolio.execution_symbols = []
            portfolio.source_backtest_run_id = None
            portfolio.execution_universe = universe_name
        if len(symbols) < 10:
            raise ValueError("模拟盘执行股票池至少需要 10 只股票")
        pipeline = build_pipeline_strategy(spec, data)
        validate_date_range(pipeline, as_of, as_of)
        pipeline.prepare(symbols, as_of, as_of)
        strategy_result = pipeline.generate_weights(as_of, symbols)
        holdings = spec.selection.n if spec.selection.type == "top_n" else max(1, round(len(symbols) * spec.selection.pct))
        turnover_band = spec.rebalance.turnover_band
        snapshot = self.snapshot(as_of)
        total_assets = float(snapshot["total_assets"])
        current_positions = {
            p.symbol: p
            for p in self.db.scalars(select(Position).where(Position.portfolio_id == portfolio.id)).all()
        }
        price_symbols = list(dict.fromkeys(symbols + list(current_positions)))
        prices = data.prices(price_symbols, as_of, as_of, allow_network=True)
        if prices.empty:
            prices = data.prices(price_symbols, as_of - timedelta(days=30), as_of)
        latest = prices.sort_values("trade_date").groupby("symbol").tail(1).set_index("symbol")
        orders: List[Dict[str, object]] = []
        # Sell positions that are no longer in the target, then trim overweight positions.
        desired_qty: Dict[str, int] = {}
        for symbol, weight in strategy_result.weights.items():
            if symbol not in latest.index:
                continue
            price = float(latest.loc[symbol].adj_close)
            if pd.isna(price):
                # 真实数据里偶尔会有一行存在但adj_close是NaN的情况——
                # Python的max(NaN, .01)行为是"未定义"的坑：会原样返回
                # NaN而不是.01，下面int((...)/max(price,.01)/100)就会
                # 变成int(NaN)直接崩掉整个回测。这支股票今天没有可用的
                # 真实价格，诚实地跳过这次调仓，不用假价格凑数，也不让
                # 它拖垮其它股票正常的调仓(ADR-0045/0046同一个原则)。
                continue
            position = current_positions.get(symbol)
            current_qty = position.quantity if position else 0
            current_weight = current_qty * price / max(total_assets, 1.0)
            if (
                position
                and current_qty > 0
                and turnover_band > 0
                and abs(float(weight) - current_weight) < turnover_band
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
            # 持仓股票今天可能压根没有价格行（停牌/数据缺失），或者有行
            # 但adj_close是NaN——两种都不能硬卖：NaN价格算出来的amount/
            # cash会悄悄把整个账户的现金污染成NaN(不会立刻报错，是更隐蔽
            # 的一种坏)。这次调仓先跳过这支股票，仓位保持不变，等下次
            # 有真实价格的时候再处理(ADR-0045/0046同一个原则)。
            if symbol not in latest.index or pd.isna(latest.loc[symbol].adj_close):
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
            reason = "目标权重下降" if target_qty > 0 else "跌出前 %s 名" % holdings
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
            reason = "分数进入前 %s 名 · 仓位 %.0f%%" % (holdings, strategy_result.target_exposure * 100)
            order = Order(portfolio_id=portfolio.id, symbol=symbol, side="BUY", quantity=quantity, price=price, amount=amount, fee=fee, strategy_name=strategy.name, reason=reason)
            self.db.add(order)
            self.db.add(Trade(portfolio_id=portfolio.id, symbol=symbol, trade_date=as_of, side="BUY", quantity=quantity, price=price, amount=amount, fee=fee, reason=reason))
            orders.append({"symbol": symbol, "side": "BUY", "quantity": quantity, "price": price, "amount": amount, "reason": reason})
        portfolio.strategy_id = strategy.id
        portfolio.last_rebalance_at = datetime.utcnow()
        portfolio.last_exposure = float(strategy_result.target_exposure)
        if portfolio.auto_rebalance_enabled:
            portfolio.auto_rebalance_frequency = spec.rebalance.frequency
            portfolio.next_rebalance_date = self.next_rebalance_date(as_of, portfolio.auto_rebalance_frequency)
        held_after = {position.symbol for position in self.db.scalars(select(Position).where(Position.portfolio_id == portfolio.id)).all()}
        for row in strategy_result.ranking.head(30).to_dict(orient="records"):
            # 没有分数的股票 score 是 NaN（历史不够/当天无行情），如实记 0 分、不给名次
            if pd.isna(row.get("score")):
                continue
            default_action = "BUY" if row.get("target_weight", 0) > 0 else "WATCH"
            action = "HOLD" if row["symbol"] in held_after and row["target_weight"] > 0 else row.get("action", default_action)
            self.db.add(Signal(strategy_id=strategy.id, symbol=row["symbol"], signal_date=as_of, action=action,
                               score=float(row["score"]), target_weight=float(row["target_weight"]),
                               reason="分数第 %d 名，目标权重 %.1f%%" % (row["rank"], row["target_weight"] * 100)))
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
                "universe": getattr(portfolio, "execution_universe", None) or strategy.universe or "csi300",
                "symbols": symbols,
                "source_backtest_run_id": getattr(portfolio, "source_backtest_run_id", None),
            },
            "strategy_spec": spec.model_dump(mode="json"),
            "target_exposure": strategy_result.target_exposure,
        }
