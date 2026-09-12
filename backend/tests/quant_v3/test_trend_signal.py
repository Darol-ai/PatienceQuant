from datetime import date, timedelta

import pytest

from app.quant_v3.trend_filter import TrendSignal


def _daily_series(n: int, prices: list[float]) -> list[tuple]:
    start = date(2020, 1, 1)
    return [(start + timedelta(days=i), prices[i]) for i in range(n)]


def test_exposure_uses_trailing_window_up_to_as_of_date():
    prices = [100.0] * 199 + [110.0] + [100.0] * 50  # 第200天(index199)在均线上方，之后又跌回
    series = _daily_series(len(prices), prices)

    signal = TrendSignal(series, ma_window=200, risk_off_exposure=0.3)

    as_of_high = series[199][0]
    assert signal.exposure(as_of_high) == pytest.approx(1.0)


def test_exposure_defaults_to_full_when_date_not_found():
    series = _daily_series(10, [100.0] * 10)
    signal = TrendSignal(series, ma_window=200, risk_off_exposure=0.3)

    assert signal.exposure(date(2099, 1, 1)) == pytest.approx(1.0)
