import pytest

from app.quant_v3.budget import allocate_score_weighted_budget


def test_higher_score_gets_proportionally_more_weight():
    """自由探索阶段（docs/adr/0017）：等权重在近乎普涨的行情里会稀释掉
    模型真正看好的黑马——按分数加权，分数越高权重越大，而不是选中即
    等权。分数先平移到非负再归一化，避免负分数导致权重为负。"""
    scores = {"A": 0.30, "B": 0.10, "C": 0.05}
    weights = allocate_score_weighted_budget(["A", "B", "C", "D"], group_budget=1.0, model_approved={"A", "B", "C"}, scores=scores)

    assert weights["A"] > weights["B"] > weights["C"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert "D" not in weights


def test_negative_scores_are_shifted_non_negative_before_weighting():
    scores = {"A": -0.05, "B": -0.10, "C": -0.20}
    weights = allocate_score_weighted_budget(["A", "B", "C"], group_budget=1.0, model_approved={"A", "B", "C"}, scores=scores)

    assert all(w >= 0 for w in weights.values())
    assert weights["A"] > weights["C"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_equal_weight_fallback_when_all_scores_identical():
    scores = {"A": 0.1, "B": 0.1}
    weights = allocate_score_weighted_budget(["A", "B"], group_budget=1.0, model_approved={"A", "B"}, scores=scores)

    assert weights == {"A": pytest.approx(0.5), "B": pytest.approx(0.5)}


def test_empty_when_nothing_approved():
    assert allocate_score_weighted_budget(["A", "B"], group_budget=1.0, model_approved=set(), scores={}) == {}
