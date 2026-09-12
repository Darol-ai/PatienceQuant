from datetime import date

import pandas as pd
import pytest

from app.quant_v3.momentum_signal import MomentumSignalSource


def _history_with_flat_then_up(symbol: str) -> pd.DataFrame:
    # 总长 280 天，前 250 天走平(100)，第 250 天起涨到 130 并保持到最后。
    # lookback=252/skip=21 时，"skip 天前"(下标 -22=258，已经涨完)相对
    # "skip+lookback 天前"(下标 -274=6，还在走平阶段)的收益就是 30%。
    dates = pd.bdate_range("2020-01-01", periods=280)
    closes = [100.0] * 250 + [130.0] * 30
    return pd.DataFrame({"date": dates, "symbol": symbol, "close": closes})


def test_returns_momentum_score_when_enough_history():
    history = _history_with_flat_then_up("601288.SH")
    source = MomentumSignalSource(history, lookback=252, skip=21)

    score = source("601288.SH", history["date"].iloc[-1].date())

    assert score == pytest.approx(0.30)


def test_returns_none_when_not_enough_history():
    history = _history_with_flat_then_up("601288.SH").iloc[:100]
    source = MomentumSignalSource(history, lookback=252, skip=21)

    assert source("601288.SH", history["date"].iloc[-1].date()) is None


def test_returns_none_for_unknown_symbol_or_date():
    history = _history_with_flat_then_up("601288.SH")
    source = MomentumSignalSource(history, lookback=252, skip=21)

    assert source("999999.SH", history["date"].iloc[-1].date()) is None
    assert source("601288.SH", date(2099, 1, 1)) is None
