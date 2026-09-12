import pytest

from app.quant_v3.labels import first_touch_label


def test_first_touch_label_is_up_when_close_first_breaches_upper_boundary():
    """O*=100，upper_t=5%，lower_t=-3%：第 1 天涨 1% 不触发，第 2 天涨 6% 触及上边界。"""
    closes = [101.0, 106.0, 90.0]

    label = first_touch_label(closes, o_star=100.0, upper_t=0.05, lower_t=-0.03)

    assert label == "UP"


def test_first_touch_label_is_down_when_it_happens_before_a_later_up_touch():
    """第 2 天先跌破下边界标 DOWN，即使第 3 天本会触及上边界也不再看——先触及的赢。"""
    closes = [101.0, 96.0, 110.0]

    label = first_touch_label(closes, o_star=100.0, upper_t=0.05, lower_t=-0.03)

    assert label == "DOWN"


def test_first_touch_label_is_neutral_when_no_day_touches_either_boundary():
    closes = [101.0, 99.0, 102.0]

    label = first_touch_label(closes, o_star=100.0, upper_t=0.05, lower_t=-0.03)

    assert label == "NEUTRAL"
