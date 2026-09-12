import pytest

from app.quant_v3.budget import allocate_flat_group_budget, allocate_full_weight_to_approved


def test_allocate_flat_group_budget_does_not_redistribute_rejected_slots():
    """V3 官方规则：模型拒绝的名额空出来，不给别的股票——engineering_only
    范围下故意保守。回归验证这条规则没被意外改掉。"""
    weights = allocate_flat_group_budget(["A", "B", "C", "D"], group_budget=1.0, model_approved={"A", "B"})

    assert weights == {"A": pytest.approx(0.25), "B": pytest.approx(0.25)}  # 总仓位只有50%


def test_allocate_full_weight_to_approved_redistributes_everything_to_selected():
    """自由探索阶段（docs/adr/0014）：选中的股票才是真正想持有的仓位，
    没选中的名额不该白白留成现金——按选中的股票数均分全部预算，选中几只
    就把预算全投进这几只，不是投进候选池全部数量。"""
    weights = allocate_full_weight_to_approved(["A", "B", "C", "D"], group_budget=1.0, model_approved={"A", "B"})

    assert weights == {"A": pytest.approx(0.50), "B": pytest.approx(0.50)}  # 总仓位100%


def test_allocate_full_weight_to_approved_empty_when_nothing_approved():
    assert allocate_full_weight_to_approved(["A", "B"], group_budget=1.0, model_approved=set()) == {}
