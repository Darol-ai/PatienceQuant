import pandas as pd
import pytest

from app.quant_v3.dataset import cross_sectional_rank


def test_cross_sectional_rank_converts_raw_values_to_percentile_within_each_date():
    """同一天 3 支股票的 factor_a 原始值 [10, 30, 20] 应该被换成组内百分位排名
    [0, 1, 0.5]（pandas rank(pct=True) 的标准定义：最小值排名 1/N，最大值排名
    N/N）——具体数值用 pandas 自己的 rank(pct=True) 独立验证过。"""
    frame = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "symbol": ["A", "B", "C"],
        "factor_a": [10.0, 30.0, 20.0],
    })

    result = cross_sectional_rank(frame, ["factor_a"])

    assert result.set_index("symbol")["factor_a"].to_dict() == pytest.approx({"A": 1 / 3, "B": 1.0, "C": 2 / 3})


def test_cross_sectional_rank_does_not_mix_values_across_different_dates():
    frame = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "symbol": ["A", "B", "A", "B"],
        "factor_a": [10.0, 20.0, 100.0, 200.0],
    })

    result = cross_sectional_rank(frame, ["factor_a"])

    # 同一天内部排名一样（都是"较小的那个排前面"），跟另一天的绝对数值差异无关
    day1 = result[result.date == "2024-01-01"].set_index("symbol")["factor_a"]
    day2 = result[result.date == "2024-01-02"].set_index("symbol")["factor_a"]
    assert day1.to_dict() == pytest.approx({"A": 0.5, "B": 1.0})
    assert day2.to_dict() == pytest.approx({"A": 0.5, "B": 1.0})


def test_cross_sectional_rank_leaves_other_columns_untouched():
    frame = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01"],
        "symbol": ["A", "B"],
        "factor_a": [10.0, 20.0],
        "label": ["UP", "DOWN"],
    })

    result = cross_sectional_rank(frame, ["factor_a"])

    assert result["label"].tolist() == ["UP", "DOWN"]
