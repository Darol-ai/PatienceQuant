import numpy as np
import pandas as pd
import pytest

from app.pipeline import index_signals as sig
from app.pipeline.price_factors import ideal_amplitude, ubl
from app.pipeline.spec import StrategySpec


def _ohlc(close):
    close = pd.Series(close, index=pd.bdate_range("2020-01-01", periods=len(close)), dtype=float)
    return pd.DataFrame({"open": close.shift(1).fillna(close), "high": close * 1.01, "low": close * 0.99, "close": close})


def test_hysteresis_holds_until_lower_band():
    value = pd.Series([0.0, 0.8, 0.2, -0.5, -0.8, 0.5, 0.9, np.nan])
    assert sig._hysteresis(value, 0.7, -0.7).tolist() == [0, 1, 1, 1, 0, 0, 1, 0]


def test_rsrs_zscore_uses_only_past_data():
    rng = np.random.default_rng(0)
    frame = _ohlc(100 + np.cumsum(rng.normal(size=800)))
    frame["high"] = frame["close"] * (1 + rng.uniform(0, .02, 800))
    frame["low"] = frame["close"] * (1 - rng.uniform(0, .02, 800))
    full = sig.rsrs_zscore(frame, n=18, m=200)
    cut = sig.rsrs_zscore(frame.iloc[:500], n=18, m=200)
    # 后面的数据改变不了前面的分数
    pd.testing.assert_series_equal(full.iloc[:500], cut)
    assert full.iloc[:216].isna().all() and full.iloc[216:].notna().all()  # 17 天后才有斜率，再攒 200 个


def test_icu_ma_follows_trend_and_state_flips():
    rising = list(np.linspace(10, 20, 30)) + [20 - 0.03 * i ** 2 for i in range(1, 21)]  # 加速下跌
    state = sig.icu_ma_state(_ohlc(rising), n=5)
    assert state.iloc[:4].eq(0).all()  # 均线还没算出来之前空仓
    assert state.iloc[35:].eq(0).all()  # 下跌段收盘价在稳健回归线下方
    line = sig.icu_ma(pd.Series(np.arange(10.0)), n=5)
    assert line.iloc[-1] == pytest.approx(9.0)  # 直线上的稳健回归就是原值


def test_alligator_enters_on_uptrend_and_exits_on_downtrend():
    path = list(np.linspace(100, 100, 60)) + list(np.linspace(100, 160, 80)) + list(np.linspace(160, 110, 60))
    rng = np.random.default_rng(1)
    close = np.array(path) * (1 + rng.normal(0, .004, len(path)))
    frame = _ohlc(close)
    frame["high"] = frame[["open", "close"]].max(axis=1) * (1 + rng.uniform(0, .01, len(path)))
    frame["low"] = frame[["open", "close"]].min(axis=1) * (1 - rng.uniform(0, .01, len(path)))
    state = sig.alligator_state(frame)
    assert set(state.unique()) <= {0.0, 1.0}
    assert state.iloc[60:140].max() == 1.0  # 上涨段里某个时点开仓
    assert state.iloc[-10:].eq(0).all()  # 下跌段末尾已经空仓
    # 同样只用过去的数据
    pd.testing.assert_series_equal(sig.alligator_state(frame.iloc[:150]), state.iloc[:150])


def _panel(n_days=40, symbols=("a", "b", "c", "d")):
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    close = pd.DataFrame(10 + rng.normal(0, .2, (n_days, len(symbols))).cumsum(axis=0), index=idx, columns=list(symbols))
    open_ = close.shift(1).fillna(close)
    spread = pd.DataFrame(rng.uniform(.005, .03, close.shape), index=idx, columns=close.columns)
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    return {"open": open_, "high": high, "low": low, "close": close}


def test_price_factors_return_one_value_per_stock():
    panel = _panel()
    for factor in (ubl, ideal_amplitude):
        values = factor(panel)
        assert list(values.index) == ["a", "b", "c", "d"]
        assert values.notna().all()


def test_ideal_amplitude_is_high_day_minus_low_day_amplitude():
    idx = pd.bdate_range("2024-01-01", periods=21)
    close = pd.DataFrame({"x": np.arange(21, dtype=float) + 10})  # 单调上涨：20 天里分位 ≥0.2 的后 17 天是"高价日"
    close.index = idx
    high = close * 1.02
    low = close.copy()
    high.iloc[-20:-17] = close.iloc[-20:-17] * 1.10  # 3 个低价日振幅 10%
    panel = {"open": close, "high": high, "low": low, "close": close}
    value = ideal_amplitude(panel, window=20, lamb=0.2)["x"]
    assert value == pytest.approx((17 * .02 - 3 * .10) / 20)


def test_new_timing_specs_validate():
    for timing in ({"type": "rsrs"}, {"type": "icu_ma", "n": 20}, {"type": "alligator"}):
        spec = StrategySpec.model_validate({"scorer": {"type": "factor_weights", "weights": {"ubl": 1}},
                                            "timing": timing, "selection": {"type": "top_n", "n": 30}})
        assert spec.timing.type == timing["type"]
