from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.data.service import MarketDataService
from app.strategies.base import StrategyConfig
from app.strategies.multifactor import MultiFactorStrategy
from app.factors.engine import FactorEngine


@dataclass
class BacktestConfig:
    start_date: date
    end_date: date
    initial_capital: float = 1_000_000
    rebalance_frequency: str = "monthly"
    holdings_count: int = 10
    max_weight: float = .15
    commission: float = .001
    slippage: float = .0005


@dataclass
class BacktestResult:
    metrics: Dict[str, float]
    equity: pd.DataFrame
    trades: pd.DataFrame
    ranking: pd.DataFrame
    notes: List[str]
    data_mode: str = "unknown"
    risk_summary: Optional[Dict[str, object]] = None


class BacktestEngine:
    def __init__(self, data: MarketDataService):
        self.data = data

    @staticmethod
    def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> List[date]:
        frame = pd.DataFrame(index=index)
        if frequency == "quarterly":
            periods = index.to_period("Q")
        elif frequency == "weekly":
            periods = index.to_period("W")
        else:
            periods = index.to_period("M")
        return [stamp.date() for _, group in frame.groupby(periods) for stamp in [group.index[-1]]]

    @staticmethod
    def _portfolio_value(cash: float, quantities: Dict[str, int], price_row: pd.Series) -> float:
        """Mark cash and every holding to the current close price."""
        market_value = 0.0
        for symbol, quantity in quantities.items():
            raw_price = price_row.get(symbol)
            try:
                price = float(raw_price)
            except (TypeError, ValueError):
                price = 0.0
            if not np.isfinite(price):
                price = 0.0
            market_value += quantity * price
        return float(cash + market_value)

    @staticmethod
    def _metrics(equity: pd.Series, benchmark: pd.Series, trades: pd.DataFrame, initial: float) -> Dict[str, float]:
        equity = pd.to_numeric(equity, errors="coerce").dropna()
        daily = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        years = max(len(equity) / 252, 1 / 252)
        final_assets = float(equity.iloc[-1]) if len(equity) else float(initial)
        # There are no external cash flows in a backtest.  Therefore the
        # compounded daily return and the final-assets / initial-capital
        # return must agree, while making the compounding rule explicit.
        compounded_return = float((1 + daily).prod() - 1) if len(daily) else final_assets / initial - 1
        total_return = float(final_assets / initial - 1)
        annual_return = float((final_assets / initial) ** (1 / years) - 1)
        volatility = float(daily.std() * np.sqrt(252)) if len(daily) > 1 else 0.0
        sharpe = float((daily.mean() / daily.std()) * np.sqrt(252)) if daily.std() > 0 else 0.0
        running_max = equity.cummax()
        drawdown = equity / running_max - 1
        max_drawdown = float(drawdown.min())
        win_rate = float((daily > 0).mean()) if len(daily) else 0.0
        turnover = float(trades.amount.abs().sum() / (initial * 2)) if not trades.empty else 0.0
        benchmark_return = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1) if len(benchmark) else 0.0
        return {
            "total_return": total_return,
            "overall_return": total_return,
            "compounded_return": compounded_return,
            "final_assets": final_assets,
            "total_profit": final_assets - float(initial),
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "trade_count": float(len(trades)),
            "turnover": turnover,
            "benchmark_return": benchmark_return,
            "excess_return": total_return - benchmark_return,
            "volatility": volatility,
        }

    def run(
        self,
        config: BacktestConfig,
        strategy_config: StrategyConfig,
        symbols: Optional[List[str]] = None,
        allow_network: bool = False,
    ) -> BacktestResult:
        catalog = self.data.stocks()
        symbols = symbols or catalog.symbol.tolist()
        factor_prices = self.data.prices(
            symbols,
            date(2018, 1, 1),
            config.end_date,
            allow_network=allow_network,
        )
        prices = factor_prices[
            (factor_prices.trade_date >= config.start_date) & (factor_prices.trade_date <= config.end_date)
        ].copy()
        if prices.empty:
            raise ValueError("指定区间没有可用行情数据")
        prices["trade_date"] = pd.to_datetime(prices.trade_date)
        wide = prices.pivot_table(index="trade_date", columns="symbol", values="adj_close", aggfunc="last").ffill().dropna(how="all")
        dates = wide.index
        rebalance_dates = self._rebalance_dates(dates, config.rebalance_frequency)
        factor_engine = FactorEngine(self.data)
        factor_engine.prime(
            factor_prices,
            self.data.fundamentals(symbols, config.end_date),
            self.data.benchmark(date(2018, 1, 1), config.end_date),
        )
        strategy = MultiFactorStrategy(factor_engine, strategy_config)
        prices_by_date = wide.ffill()
        cash = config.initial_capital
        quantities: Dict[str, int] = {symbol: 0 for symbol in symbols}
        entry_prices: Dict[str, float] = {}
        trades: List[Dict[str, object]] = []
        target_by_execution: Dict[date, Dict[str, float]] = {}
        execution_context: Dict[date, Dict[str, object]] = {}
        ranking_frames = []
        notes: List[str] = []
        for signal_date in rebalance_dates:
            signal_index = dates.get_loc(pd.Timestamp(signal_date))
            if signal_index + 1 >= len(dates):
                continue
            execution_date = dates[signal_index + 1].date()
            result = strategy.generate_weights(signal_date, symbols)
            target_by_execution[execution_date] = result.weights
            execution_context[execution_date] = {
                "regime": result.market_regime,
                "target_exposure": result.target_exposure,
            }
            notes.extend(result.data_quality_notes)
            ranking_frames.append(result.ranking.assign(signal_date=signal_date))

        equity_values = []
        equity_history: List[float] = []
        equity_peak = float(config.initial_capital)
        target_exposures = []
        regime_counts: Dict[str, int] = {}
        drawdown_brake_rebalances = 0
        volatility_scaled_rebalances = 0
        realized_volatilities: List[float] = []
        benchmark_df = self.data.benchmark(config.start_date, config.end_date)
        benchmark_series = benchmark_df.set_index(pd.to_datetime(benchmark_df.trade_date)).adj_close.reindex(dates).ffill().bfill()
        initial_benchmark = float(benchmark_series.iloc[0]) if len(benchmark_series) else 1.0
        for current_date in dates:
            current_day = current_date.date()
            price_row = prices_by_date.loc[current_date]
            # This is the compounding capital base: all realized and
            # unrealized gains remain in total assets and fund later trades.
            portfolio_value = self._portfolio_value(cash, quantities, price_row)
            if current_day in target_by_execution:
                target = target_by_execution[current_day]
                context = execution_context.get(current_day, {})
                regime = str(context.get("regime", "unknown"))
                base_target_exposure = float(context.get("target_exposure", 1.0))
                # Risk budget is evaluated only from observations available
                # before this execution date.  The signal itself was created
                # on the prior trading day, so this cannot leak future prices.
                risk_scale = 1.0
                # Use the last completed observation (the signal date), not
                # the execution day's close, when deciding whether to brake
                # exposure.  This keeps the next-day execution path free of
                # same-day look-ahead.
                risk_reference_value = equity_history[-1] if equity_history else portfolio_value
                current_drawdown = risk_reference_value / max(equity_peak, 1.0) - 1.0
                drawdown_triggered = (
                    strategy_config.max_drawdown_budget > 0
                    and current_drawdown <= -strategy_config.max_drawdown_budget
                )
                if drawdown_triggered:
                    risk_scale = min(risk_scale, strategy_config.drawdown_brake_exposure)
                    drawdown_brake_rebalances += 1
                realized_vol = 0.0
                if strategy_config.target_volatility > 0 and len(equity_history) >= 21:
                    history = pd.Series(equity_history, dtype=float)
                    realized_vol = float(history.pct_change().dropna().tail(63).std() * np.sqrt(252))
                    if np.isfinite(realized_vol) and realized_vol > strategy_config.target_volatility:
                        risk_scale = min(risk_scale, strategy_config.target_volatility / realized_vol)
                        volatility_scaled_rebalances += 1
                if np.isfinite(realized_vol) and realized_vol > 0:
                    realized_volatilities.append(realized_vol)
                target_exposure = base_target_exposure * risk_scale
                target = {symbol: float(weight) * risk_scale for symbol, weight in target.items()}
                target_exposures.append(target_exposure)
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                if drawdown_triggered:
                    notes.append(
                        "组合回撤达到 %.0f%% 风险预算，执行暴露刹车至 %.0f%%"
                        % (strategy_config.max_drawdown_budget * 100, strategy_config.drawdown_brake_exposure * 100)
                    )
                if realized_vol > strategy_config.target_volatility > 0:
                    notes.append(
                        "滚动实现波动率 %.1f%% 高于目标 %.1f%%，按比例缩小股票暴露"
                        % (realized_vol * 100, strategy_config.target_volatility * 100)
                    )
                # Sell first, then buy. Slippage is adverse to the portfolio.
                current_value = max(portfolio_value, 1.0)
                desired: Dict[str, int] = {}
                stop_out_symbols = set()
                for symbol, weight in target.items():
                    mark_price = float(price_row.get(symbol, 0) or 0)
                    if mark_price <= 0:
                        continue
                    current_qty = quantities.get(symbol, 0)
                    current_weight = current_qty * mark_price / current_value
                    target_qty = int((current_value * weight) / max(mark_price, .01) / 100) * 100
                    entry_price = entry_prices.get(symbol)
                    if (
                        current_qty > 0
                        and entry_price
                        and strategy_config.stop_loss > 0
                        and mark_price <= entry_price * (1 - strategy_config.stop_loss)
                    ):
                        desired[symbol] = 0
                        stop_out_symbols.add(symbol)
                    elif (
                        current_qty > 0
                        and strategy_config.turnover_band > 0
                        and abs(float(weight) - current_weight) < strategy_config.turnover_band
                    ):
                        # Ignore small monthly weight noise. This is the
                        # low-frequency turnover guard that keeps costs from
                        # eroding an otherwise sound signal.
                        desired[symbol] = current_qty
                    else:
                        desired[symbol] = target_qty
                for symbol in quantities:
                    desired.setdefault(symbol, 0)
                for symbol, qty in list(quantities.items()):
                    delta = desired.get(symbol, 0) - qty
                    if delta >= 0:
                        continue
                    price = float(price_row.get(symbol, 0)) * (1 - config.slippage)
                    sell_qty = min(qty, abs(delta))
                    amount = sell_qty * price
                    fee = amount * config.commission
                    cash += amount - fee
                    quantities[symbol] -= sell_qty
                    if quantities[symbol] <= 0:
                        quantities[symbol] = 0
                        entry_prices.pop(symbol, None)
                    reason = (
                        "保护性止损 %.0f%% · %s" % (strategy_config.stop_loss * 100, regime)
                        if symbol in stop_out_symbols
                        else "目标权重下降 · %s" % regime
                    )
                    trades.append({"trade_date": current_day, "symbol": symbol, "side": "SELL", "quantity": sell_qty, "price": price, "amount": amount, "fee": fee, "reason": reason})
                for symbol, target_qty in desired.items():
                    delta = target_qty - quantities.get(symbol, 0)
                    if delta <= 0:
                        continue
                    if symbol in stop_out_symbols:
                        continue
                    price = float(price_row.get(symbol, 0)) * (1 + config.slippage)
                    affordable = int(cash / max(price * (1 + config.commission), .01))
                    buy_qty = min(delta, affordable)
                    if buy_qty <= 0:
                        continue
                    amount = buy_qty * price
                    fee = amount * config.commission
                    cash -= amount + fee
                    previous_qty = quantities.get(symbol, 0)
                    previous_entry = entry_prices.get(symbol, price)
                    quantities[symbol] = quantities.get(symbol, 0) + buy_qty
                    entry_prices[symbol] = (
                        (previous_entry * previous_qty + amount) / quantities[symbol]
                        if quantities[symbol] > 0
                        else price
                    )
                    reason_suffix = []
                    if drawdown_triggered:
                        reason_suffix.append("回撤刹车")
                    if realized_vol > strategy_config.target_volatility > 0:
                        reason_suffix.append("波动率缩放")
                    reason_suffix_text = (" · " + " / ".join(reason_suffix)) if reason_suffix else ""
                    trades.append({"trade_date": current_day, "symbol": symbol, "side": "BUY", "quantity": buy_qty, "price": price, "amount": amount, "fee": fee, "reason": "综合评分进入 Top %s · %s · 股票暴露 %.0f%%%s" % (strategy_config.holdings_count, regime, target_exposure * 100, reason_suffix_text)})
                # Always re-mark after the complete rebalance.  This matters
                # when a rebalance only sells: fees and adverse slippage must
                # reduce the compounded total assets as well.
                portfolio_value = self._portfolio_value(cash, quantities, price_row)
            equity_values.append({"trade_date": current_day, "equity": portfolio_value, "benchmark": float(benchmark_series.loc[current_date] / initial_benchmark * config.initial_capital)})
            equity_history.append(float(portfolio_value))
            equity_peak = max(equity_peak, float(portfolio_value))
        equity = pd.DataFrame(equity_values)
        equity["drawdown"] = equity.equity / equity.equity.cummax() - 1
        trade_df = pd.DataFrame(trades, columns=["trade_date", "symbol", "side", "quantity", "price", "amount", "fee", "reason"])
        metrics = self._metrics(equity.equity, equity.benchmark, trade_df, config.initial_capital)
        ranking = pd.concat(ranking_frames, ignore_index=True) if ranking_frames else pd.DataFrame()
        return BacktestResult(
            metrics=metrics,
            equity=equity,
            trades=trade_df,
            ranking=ranking,
            notes=list(dict.fromkeys(notes)),
            data_mode=self.data.frame_data_mode(factor_prices),
            risk_summary={
                "regime_counts": regime_counts,
                "average_target_exposure": float(np.mean(target_exposures)) if target_exposures else 1.0,
                "risk_gate_rebalances": int(sum(count for regime, count in regime_counts.items() if regime in {"risk_off", "caution"})),
                "cash_buffer": strategy_config.cash_buffer,
                "trend_filter": strategy_config.trend_filter,
                "risk_off_exposure": strategy_config.risk_off_exposure,
                "target_volatility": strategy_config.target_volatility,
                "max_drawdown_budget": strategy_config.max_drawdown_budget,
                "drawdown_brake_exposure": strategy_config.drawdown_brake_exposure,
                "drawdown_brake_rebalances": drawdown_brake_rebalances,
                "volatility_scaled_rebalances": volatility_scaled_rebalances,
                "average_realized_volatility": float(np.mean(realized_volatilities)) if realized_volatilities else 0.0,
            },
        )
