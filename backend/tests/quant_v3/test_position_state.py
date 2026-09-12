from app.quant_v3.position_state import PositionState


def test_position_state_initializes_high_not_below_entry_price():
    """V3 方案第 4 节：H 初始化不低于 B。"""
    state = PositionState(b=100.0)

    assert state.h == 100.0
    assert state.holding_days == 0


def test_record_close_raises_high_when_price_makes_a_new_peak():
    state = PositionState(b=100.0)

    state.record_close(130.0)

    assert state.h == 130.0
    assert state.holding_days == 1


def test_record_close_does_not_lower_high_on_a_pullback():
    state = PositionState(b=100.0)
    state.record_close(130.0)

    state.record_close(115.0)

    assert state.h == 130.0
    assert state.holding_days == 2
