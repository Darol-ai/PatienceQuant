import pytest

from app.quant_v3.risk_exits import hard_stop_loss_triggered, trailing_stop_triggered


def test_hard_stop_loss_triggers_at_exactly_minus_15_percent():
    """V3 方案 4 节：有效收盘 Q/B-1 <= -15% 触发固定风险退出，边界含等号。"""
    assert hard_stop_loss_triggered(current=85.0, b=100.0) is True


def test_hard_stop_loss_does_not_trigger_above_minus_15_percent():
    assert hard_stop_loss_triggered(current=86.0, b=100.0) is False


def test_trailing_stop_triggers_after_25_percent_peak_and_10_percent_drawdown():
    """本轮曾涨到 B 的 130%（>=25%门槛），此后从最高点 H 回撤超过 10%。"""
    assert trailing_stop_triggered(current=115.0, b=100.0, h=130.0) is True


def test_trailing_stop_does_not_trigger_with_less_than_10_percent_drawdown():
    assert trailing_stop_triggered(current=120.0, b=100.0, h=130.0) is False


def test_trailing_stop_never_arms_if_high_never_reached_25_percent():
    """H 只涨到 118%（未达 25% 门槛），即使从 H 大幅回撤也不算跟踪止损——
    跟踪止损从未被"激活"。"""
    assert trailing_stop_triggered(current=100.0, b=100.0, h=118.0) is False


def test_hard_stop_loss_threshold_is_overridable_for_sensitivity_experiments():
    """V3 原文写死 -15%，但要做"止损参数是否对趋势行情过敏"的敏感性实验
    就得能覆盖它——默认值不变，保证已有调用方/测试不受影响。"""
    assert hard_stop_loss_triggered(current=82.0, b=100.0, threshold=-0.15) is True
    assert hard_stop_loss_triggered(current=82.0, b=100.0, threshold=-0.20) is False


def test_trailing_stop_thresholds_are_overridable_for_sensitivity_experiments():
    assert trailing_stop_triggered(current=112.0, b=100.0, h=130.0, arm_threshold=0.25, drawdown_threshold=-0.10) is True
    assert trailing_stop_triggered(current=112.0, b=100.0, h=130.0, arm_threshold=0.25, drawdown_threshold=-0.15) is False
    assert trailing_stop_triggered(current=112.0, b=100.0, h=126.0, arm_threshold=0.30, drawdown_threshold=-0.10) is False
