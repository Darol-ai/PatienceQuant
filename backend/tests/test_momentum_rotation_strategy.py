from datetime import date

import pytest

from app.strategies.momentum_rotation_strategy import MomentumRotationStrategy

_UNIVERSE = [
    {"symbol": "A1", "group": "红利"}, {"symbol": "A2", "group": "红利"},
    {"symbol": "A3", "group": "红利"}, {"symbol": "A4", "group": "红利"},
    {"symbol": "B1", "group": "成长"}, {"symbol": "B2", "group": "成长"}, {"symbol": "B3", "group": "成长"},
    {"symbol": "C1", "group": "周期"}, {"symbol": "C2", "group": "周期"}, {"symbol": "C3", "group": "周期"},
]
_GROUP_BUDGETS = {"红利": 0.40, "成长": 0.30, "周期": 0.30}


def test_absolute_momentum_filter_excludes_negative_scores_from_ranking():
    """Dual Momentum 的绝对动量过滤：自身动量为负的股票直接不参与组内排序，
    即便它在组内"矮子里拔将军"仍是相对最高分。红利组全负 -> 整组空仓。"""
    scores = {"A1": -0.05, "A2": -0.10, "A3": -0.20, "A4": -0.30}
    strategy = MomentumRotationStrategy(
        signal_source=lambda symbol, as_of: scores.get(symbol),
        universe=_UNIVERSE, group_budgets=_GROUP_BUDGETS, top_k_ratio=0.5,
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    dividend_group_weights = {s: w for s, w in result.weights.items() if s in scores}
    assert dividend_group_weights == {}


def test_positive_scores_get_ranked_by_relative_momentum_within_group():
    scores = {
        "A1": 0.30, "A2": 0.20, "A3": 0.10, "A4": -0.05,  # 红利4选2（正且最高的两个）
        "B1": 0.15, "B2": 0.05, "B3": -0.01,               # 成长3选2
        "C1": 0.40, "C2": 0.35, "C3": 0.30,                # 周期3选2
    }
    strategy = MomentumRotationStrategy(
        signal_source=lambda symbol, as_of: scores.get(symbol),
        universe=_UNIVERSE, group_budgets=_GROUP_BUDGETS, top_k_ratio=0.5,
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert set(result.weights) == {"A1", "A2", "B1", "B2", "C1", "C2"}
    assert result.weights["A1"] == pytest.approx(0.10)  # 红利预算40%/4只候选=10%


def test_missing_score_is_treated_as_not_investable():
    strategy = MomentumRotationStrategy(
        signal_source=lambda symbol, as_of: None,
        universe=_UNIVERSE, group_budgets=_GROUP_BUDGETS, top_k_ratio=0.5,
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert result.weights == {}


def test_no_forced_daily_exits_pure_periodic_rebalance():
    """核心机制变化：不设止损/跟踪止损，只靠月度调仓时动量转负被踢出重新
    分配——验证 on_daily_close 用的是 BaseStrategy 默认的空实现。"""
    strategy = MomentumRotationStrategy(
        signal_source=lambda symbol, as_of: 0.5,
        universe=_UNIVERSE, group_budgets=_GROUP_BUDGETS, top_k_ratio=0.5,
    )

    risk = strategy.on_daily_close(date(2024, 1, 2), price_row=None)

    assert risk.exits == {}
    assert risk.blocked_symbols == set()
