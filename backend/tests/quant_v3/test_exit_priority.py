from app.quant_v3.risk_exits import resolve_exit_reason


def test_top_tier_exit_beats_model_exit_when_both_trigger():
    """V3 方案 4.1 节：资格退出/固定止损/跟踪止损 > 模型退出。"""
    reason = resolve_exit_reason(
        qualification_exit=False,
        hard_stop_loss=True,
        trailing_stop=False,
        model_risk_exit=True,
        model_take_profit=False,
    )

    assert reason == "HARD_STOP_LOSS"


def test_no_reason_when_nothing_triggers():
    reason = resolve_exit_reason(
        qualification_exit=False,
        hard_stop_loss=False,
        trailing_stop=False,
        model_risk_exit=False,
        model_take_profit=False,
    )

    assert reason is None


def test_qualification_exit_wins_within_top_tier():
    """顶层三种谁在前，V3 文档未写死，本项目约定：资格退出(行政性、最终) >
    固定止损 > 跟踪止损。"""
    reason = resolve_exit_reason(
        qualification_exit=True,
        hard_stop_loss=True,
        trailing_stop=True,
        model_risk_exit=False,
        model_take_profit=False,
    )

    assert reason == "QUALIFICATION_EXIT"


def test_model_risk_exit_wins_within_model_tier():
    """模型退出内部谁在前，V3 文档未写死，本项目约定：风险规避优先于止盈——
    模型风险退出 > 模型主动止盈。"""
    reason = resolve_exit_reason(
        qualification_exit=False,
        hard_stop_loss=False,
        trailing_stop=False,
        model_risk_exit=True,
        model_take_profit=True,
    )

    assert reason == "MODEL_RISK_EXIT"
