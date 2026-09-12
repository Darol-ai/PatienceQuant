import pytest

from app.quant_v3.budget import allocate_full_weight_with_boost


def test_boosted_symbols_get_multiplier_times_the_normal_slot_weight():
    """自由探索阶段（docs/adr/0025）：动量兜底选出的黑马和模型选出的股票
    等权重摊薄在一起，稀释了黑马本该有的贡献（诊断见ADR-0016/0022）——
    给"被动量兜底强制纳入"的股票单独乘一个权重倍数，不是简单等权。
    3只入选，1只(B)加倍：单位数=1+2+1=4，B权重=budget*2/4=0.5*budget，
    A/C各budget*1/4=0.25*budget。"""
    weights = allocate_full_weight_with_boost(
        ["A", "B", "C", "D"], group_budget=1.0,
        model_approved={"A", "B", "C"}, boosted_symbols={"B"}, boost_multiplier=2.0,
    )

    assert weights == {"A": pytest.approx(0.25), "B": pytest.approx(0.50), "C": pytest.approx(0.25)}
    assert sum(weights.values()) == pytest.approx(1.0)


def test_boost_multiplier_of_one_is_equivalent_to_plain_equal_weight():
    weights = allocate_full_weight_with_boost(
        ["A", "B", "C"], group_budget=1.0,
        model_approved={"A", "B", "C"}, boosted_symbols={"B"}, boost_multiplier=1.0,
    )

    assert weights == {"A": pytest.approx(1 / 3), "B": pytest.approx(1 / 3), "C": pytest.approx(1 / 3)}


def test_boosted_symbol_not_in_approved_set_has_no_effect():
    weights = allocate_full_weight_with_boost(
        ["A", "B"], group_budget=1.0,
        model_approved={"A"}, boosted_symbols={"B"}, boost_multiplier=5.0,
    )

    assert weights == {"A": pytest.approx(1.0)}


def test_empty_when_nothing_approved():
    assert allocate_full_weight_with_boost(["A"], 1.0, set(), {"A"}, 2.0) == {}
