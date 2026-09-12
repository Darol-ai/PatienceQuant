from app.quant_v3.enriched_dataset import ENRICHED_FEATURE_COLUMNS, build_enriched_sample, compute_enriched_features


def _oscillating_closes(n: int) -> list[float]:
    """和 test_dataset.py 一样的有真实（非零）波动、但完全确定的价格序列。"""
    pattern = [100.0, 101.0, 99.0]
    return [pattern[i % 3] for i in range(n)]


def _make_bars(days: int, pb: float = 1.0) -> list[dict]:
    closes = _oscillating_closes(days)
    return [
        {
            "date": f"2020-01-{i + 1:02d}", "open": closes[i], "high": closes[i] * 1.01, "low": closes[i] * 0.99,
            "close": closes[i], "volume": 1000, "amount": closes[i] * 1000, "is_suspended": False,
            "turnover_rate": 1.0, "pb_mrq": pb + (i % 5) * 0.1,
        }
        for i in range(days)
    ]


def test_enriched_features_include_base_v3_features_plus_turnover_and_valuation():
    bars = _make_bars(400)

    features = compute_enriched_features(bars, t_index=390)

    assert features is not None
    assert set(ENRICHED_FEATURE_COLUMNS) <= set(features)
    assert "avg_turnover_60d" in features
    assert "valuation_percentile_252d" in features


def test_enriched_features_none_when_base_history_too_short():
    bars = _make_bars(100)

    assert compute_enriched_features(bars, t_index=50) is None


def test_enriched_features_none_when_valuation_history_too_short():
    """基础14项特征只需要120天历史，估值分位要252天——历史够14项特征但
    不够估值分位时，整条样本仍然不制作（缺特征也是缺，不能拿 None/占位
    值蒙混过关）。"""
    bars = _make_bars(200)

    assert compute_enriched_features(bars, t_index=199) is None


def test_build_enriched_sample_reuses_same_l1_label_as_official_dataset():
    bars = _make_bars(300)

    sample = build_enriched_sample(bars, t_index=270, group="综合")

    assert sample is not None
    assert set(ENRICHED_FEATURE_COLUMNS) <= set(sample)
    assert "label" in sample and "group" in sample
