from app.quant_v3.position_state import CooldownTracker


def test_not_locked_out_before_any_exit():
    tracker = CooldownTracker()

    assert tracker.is_locked_out() is False


def test_locked_out_immediately_after_start():
    tracker = CooldownTracker()

    tracker.start()

    assert tracker.is_locked_out() is True


def test_still_locked_out_after_19_days():
    """V3 方案 4.1 节：经过 20 个完整交易日才解除禁买，第 19 天还没满。"""
    tracker = CooldownTracker()
    tracker.start()

    for _ in range(19):
        tracker.advance_day()

    assert tracker.is_locked_out() is True


def test_unlocked_after_20_full_days():
    tracker = CooldownTracker()
    tracker.start()

    for _ in range(20):
        tracker.advance_day()

    assert tracker.is_locked_out() is False
