import pytest

from app.quant_v3.trend_filter import market_exposure_multiplier


def test_full_exposure_when_index_above_its_moving_average():
    prices = [100.0] * 199 + [110.0]  # MA(200)含当天 ≈ 100.05，当天110 在均线上方

    multiplier = market_exposure_multiplier(prices, ma_window=200, risk_off_exposure=0.3)

    assert multiplier == pytest.approx(1.0)


def test_reduced_exposure_when_index_below_its_moving_average():
    prices = [100.0] * 199 + [80.0]  # 当天大幅低于均线，进入风险关闭

    multiplier = market_exposure_multiplier(prices, ma_window=200, risk_off_exposure=0.3)

    assert multiplier == pytest.approx(0.3)


def test_full_exposure_when_not_enough_history_to_judge():
    prices = [100.0] * 50

    multiplier = market_exposure_multiplier(prices, ma_window=200, risk_off_exposure=0.3)

    assert multiplier == pytest.approx(1.0)
