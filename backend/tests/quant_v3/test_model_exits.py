from app.quant_v3.risk_exits import model_risk_exit_triggered, model_take_profit_triggered


def test_model_risk_exit_triggers_when_p_down_at_least_65_percent_two_days_in_a_row():
    """V3 方案 4 节：连续两个相邻交易日收盘 p_down >= 0.65。"""
    assert model_risk_exit_triggered(p_down_today=0.70, p_down_yesterday=0.66) is True


def test_model_risk_exit_does_not_trigger_on_a_single_day():
    assert model_risk_exit_triggered(p_down_today=0.70, p_down_yesterday=0.50) is False


def test_model_take_profit_triggers_when_all_three_conditions_hold_two_days_running():
    """V3 方案 4 节：连续两日 Q/B-1>=10% 且 p_down>=0.50 且 p_up<=0.35。"""
    assert (
        model_take_profit_triggered(
            q_over_b_today=0.12, p_down_today=0.55, p_up_today=0.30,
            q_over_b_yesterday=0.11, p_down_yesterday=0.52, p_up_yesterday=0.33,
        )
        is True
    )


def test_model_take_profit_does_not_trigger_if_only_today_qualifies():
    assert (
        model_take_profit_triggered(
            q_over_b_today=0.12, p_down_today=0.55, p_up_today=0.30,
            q_over_b_yesterday=0.05, p_down_yesterday=0.40, p_up_yesterday=0.50,
        )
        is False
    )
