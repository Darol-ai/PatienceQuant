from app.quant_v3.qualification import is_liquidity_qualified

_WINDOW = 60


def _bars(amounts, suspended=None, decision_day_suspended=False):
    n = len(amounts)
    suspended = suspended or [False] * n
    bars = [
        {"date": f"2024-01-{i + 1:02d}", "amount": amounts[i], "volume": 0 if suspended[i] else 1000, "is_suspended": suspended[i]}
        for i in range(n)
    ]
    if decision_day_suspended:
        bars[-1]["is_suspended"] = True
        bars[-1]["volume"] = 0
    return bars


def test_qualified_when_liquid_and_actively_traded():
    """V3 方案 2.2 节沿用的资格条件（流动性子集）：决策日正常交易；最近60
    个交易日日均成交额≥1亿；最近60日至少50日有成交。"""
    bars = _bars([2e8] * _WINDOW)

    assert is_liquidity_qualified(bars, t_index=_WINDOW - 1) is True


def test_not_qualified_when_average_amount_below_threshold():
    bars = _bars([5e7] * _WINDOW)  # 5000万，低于1亿门槛

    assert is_liquidity_qualified(bars, t_index=_WINDOW - 1) is False


def test_not_qualified_when_too_many_suspended_days_in_window():
    """60日内只有40日有成交(<50日门槛)，即使日均成交额达标也不合格。"""
    amounts = [2e8] * 40 + [0.0] * 20
    suspended = [False] * 40 + [True] * 20
    bars = _bars(amounts, suspended=suspended)

    assert is_liquidity_qualified(bars, t_index=_WINDOW - 1) is False


def test_not_qualified_when_decision_day_itself_is_suspended():
    bars = _bars([2e8] * _WINDOW, decision_day_suspended=True)

    assert is_liquidity_qualified(bars, t_index=_WINDOW - 1) is False


def test_not_qualified_when_not_enough_history_to_verify():
    """不知道就不能记为合格——历史不足60天时无法验证，按不合格处理，
    不能因为"没数据"就悄悄放行。"""
    bars = _bars([2e8] * 30)

    assert is_liquidity_qualified(bars, t_index=29) is False
