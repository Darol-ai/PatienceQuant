"""V3 方案第 4 节：独立于模型的硬风险退出规则。

B = 本轮建仓基准价，H = 本轮区间最高有效收盘价（初始化不低于 B）。
"""


def hard_stop_loss_triggered(current: float, b: float, threshold: float = -0.15) -> bool:
    """固定风险退出：有效收盘 Q/B-1 <= threshold（V3 原文 -15%）。
    threshold 可覆盖，仅用于研究"止损参数是否对趋势行情过敏"这类敏感性
    实验，默认值不变，不影响 V3 原文行为。"""
    return current / b - 1 <= threshold


def trailing_stop_triggered(
    current: float, b: float, h: float, arm_threshold: float = 0.25, drawdown_threshold: float = -0.10
) -> bool:
    """跟踪风险退出：本轮曾涨到 H/B-1 >= arm_threshold（V3 原文 25%，跟踪
    止损被激活），此后从最高点回撤 Q/H-1 <= drawdown_threshold（V3 原文
    -10%）。未曾涨过 arm_threshold 则跟踪止损从未激活。两个阈值同样只为
    敏感性实验开放覆盖，默认值不变。
    """
    armed = h / b - 1 >= arm_threshold
    drawdown_from_high = current / h - 1 <= drawdown_threshold
    return armed and drawdown_from_high


def model_risk_exit_triggered(p_down_today: float, p_down_yesterday: float) -> bool:
    """模型风险退出：连续两个相邻交易日收盘 p_down >= 0.65。"""
    return p_down_today >= 0.65 and p_down_yesterday >= 0.65


def model_take_profit_triggered(
    q_over_b_today: float,
    p_down_today: float,
    p_up_today: float,
    q_over_b_yesterday: float,
    p_down_yesterday: float,
    p_up_yesterday: float,
) -> bool:
    """模型主动止盈：连续两日均满足 Q/B-1>=10% 且 p_down>=0.50 且 p_up<=0.35。"""

    def qualifies(q_over_b: float, p_down: float, p_up: float) -> bool:
        return q_over_b >= 0.10 and p_down >= 0.50 and p_up <= 0.35

    return qualifies(q_over_b_today, p_down_today, p_up_today) and qualifies(
        q_over_b_yesterday, p_down_yesterday, p_up_yesterday
    )


def resolve_exit_reason(
    *,
    qualification_exit: bool,
    hard_stop_loss: bool,
    trailing_stop: bool,
    model_risk_exit: bool,
    model_take_profit: bool,
) -> str | None:
    """按 V3 方案 4.1 节的优先级挑出"主原因"：
    资格退出/固定止损/跟踪止损 > 模型退出。

    顶层三种、模型两种各自内部的先后顺序文档未写死，本项目约定为：
    资格退出(行政性、最终) > 固定止损 > 跟踪止损；
    模型风险退出(规避风险) > 模型主动止盈。
    """
    ordered = [
        ("QUALIFICATION_EXIT", qualification_exit),
        ("HARD_STOP_LOSS", hard_stop_loss),
        ("TRAILING_STOP", trailing_stop),
        ("MODEL_RISK_EXIT", model_risk_exit),
        ("MODEL_TAKE_PROFIT", model_take_profit),
    ]
    for reason, triggered in ordered:
        if triggered:
            return reason
    return None
