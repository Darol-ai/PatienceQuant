"""BacktestEngine 每日回调钩子：on_daily_close 强制退出、notify_fill 通知。

用一个完全自定义的最小数据服务和假策略，不依赖 DemoDataProvider/DB，
只验证引擎会不会真的按钩子的指示强制平仓——这是给 V3Strategy 接进来打地基。
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List

import pandas as pd
import pytest

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.strategies.base import BaseStrategy, DailyRiskResult, StrategyConfig, StrategyResult


class _FakeDataService:
    """只实现 BacktestEngine.run() 用到的四个方法，两只股票、十个交易日。"""

    def __init__(self):
        self.dates = pd.date_range("2024-01-01", periods=10, freq="D")
        self.close = {
            "AAA": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "BBB": [20.0] * 10,
        }

    def stocks(self) -> pd.DataFrame:
        return pd.DataFrame({"symbol": ["AAA", "BBB"]})

    def prices(self, symbols, start, end, allow_network=False) -> pd.DataFrame:
        rows = []
        for symbol in symbols:
            for i, d in enumerate(self.dates):
                rows.append({"trade_date": d.date(), "symbol": symbol, "adj_close": self.close[symbol][i]})
        return pd.DataFrame(rows)

    def fundamentals(self, symbols, as_of) -> pd.DataFrame:
        return pd.DataFrame()

    def benchmark(self, start, end) -> pd.DataFrame:
        return pd.DataFrame({"trade_date": [d.date() for d in self.dates], "adj_close": [100.0] * len(self.dates)})

    def frame_data_mode(self, frame) -> str:
        return "test"


class _FakeStrategy(BaseStrategy):
    """第一次调仓把 100% 权重给 AAA；2024-01-04 无论如何都强制清仓 AAA。"""

    def __init__(self):
        self.fills: List[Dict] = []

    def generate_weights(self, as_of: date, symbols: List[str]) -> StrategyResult:
        return StrategyResult(
            as_of=as_of,
            weights={"AAA": 1.0},
            ranking=pd.DataFrame({"symbol": ["AAA"], "score": [100.0], "rank": [1], "target_weight": [1.0], "action": ["BUY"]}),
            data_quality_notes=[],
        )

    def on_daily_close(self, current_date: date, price_row: pd.Series) -> DailyRiskResult:
        # 第一次调仓信号日是 1/7（周日结束的那周最后一个交易日），下一交易日
        # 1/8 才实际建仓，所以强制退出要选在 1/8 之后才有仓位可平。
        if current_date == date(2024, 1, 9):
            return DailyRiskResult(exits={"AAA": "TEST_FORCED_EXIT"})
        return DailyRiskResult()

    def notify_fill(self, symbol, side, price, quantity, trade_date) -> None:
        self.fills.append({"symbol": symbol, "side": side, "price": price, "quantity": quantity, "trade_date": trade_date})


def test_on_daily_close_forces_an_exit_even_outside_a_rebalance_date():
    engine = BacktestEngine(_FakeDataService())
    strategy = _FakeStrategy()
    config = BacktestConfig(
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 10),
        rebalance_frequency="weekly", commission=0, slippage=0,
    )
    result = engine.run(config, StrategyConfig(holdings_count=1, max_weight=1.0), symbols=["AAA", "BBB"], strategy=strategy)

    forced_exit = result.trades[(result.trades.symbol == "AAA") & (result.trades.reason == "TEST_FORCED_EXIT")]
    assert len(forced_exit) == 1
    assert forced_exit.iloc[0]["trade_date"] == date(2024, 1, 9)
    # 2024-01-09 之后 AAA 应该已经清仓，不再持有；无手续费/滑点，总资产应
    # 该原样保留（AAA 一直没涨跌）。
    assert result.equity[result.equity.trade_date == date(2024, 1, 10)].iloc[0]["equity"] == pytest.approx(
        config.initial_capital
    )


def test_notify_fill_is_called_for_both_the_buy_and_the_forced_sell():
    engine = BacktestEngine(_FakeDataService())
    strategy = _FakeStrategy()
    config = BacktestConfig(
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 10),
        rebalance_frequency="weekly", commission=0, slippage=0,
    )
    engine.run(config, StrategyConfig(holdings_count=1, max_weight=1.0), symbols=["AAA", "BBB"], strategy=strategy)

    sides = [fill["side"] for fill in strategy.fills if fill["symbol"] == "AAA"]
    assert "BUY" in sides
    assert "SELL" in sides
