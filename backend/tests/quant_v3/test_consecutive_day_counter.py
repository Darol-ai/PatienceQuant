from app.quant_v3.position_state import ConsecutiveDayCounter


def test_single_day_is_not_yet_confirmed():
    counter = ConsecutiveDayCounter()

    confirmed = counter.update(condition_met=True)

    assert confirmed is False


def test_two_consecutive_days_confirm():
    counter = ConsecutiveDayCounter()
    counter.update(condition_met=True)

    confirmed = counter.update(condition_met=True)

    assert confirmed is True


def test_a_failed_day_resets_the_streak():
    """V3 方案 4.1 节：条件失败中断计数。"""
    counter = ConsecutiveDayCounter()
    counter.update(condition_met=True)
    counter.update(condition_met=False)

    confirmed = counter.update(condition_met=True)

    assert confirmed is False  # 又只是重新开始的第 1 天，还没到第 2 天


def test_explicit_reset_clears_the_streak():
    """V3 方案 4.1 节：模型或校准器版本切换时计数器归零。"""
    counter = ConsecutiveDayCounter()
    counter.update(condition_met=True)
    counter.update(condition_met=True)  # 已经连续两天确认

    counter.reset()
    confirmed = counter.update(condition_met=True)

    assert confirmed is False
