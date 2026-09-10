from datetime import date
import sys
import types

import pandas as pd

from app.data.akshare_provider import AKShareDataProvider
from app.data.demo import DemoDataProvider, STOCK_SPECS
from app.strategies.multifactor import MultiFactorStrategy


def test_demo_catalog_has_five_groups_and_broad_coverage():
    provider = DemoDataProvider(seed=42, as_of=date(2026, 1, 2))
    catalog = provider.stock_catalog()
    assert len(catalog) >= 1000
    counts = catalog.groupby("group").size().to_dict()
    assert set(counts) == {"消费", "科技", "新能源", "金融", "红利/央国企"}
    assert min(counts.values()) >= 10
    assert len(STOCK_SPECS) >= 1000
    assert len(catalog.industry.unique()) >= 100


def test_demo_generation_is_reproducible():
    left = DemoDataProvider(seed=7, as_of=date(2025, 1, 3)).generate_prices().head(20)
    right = DemoDataProvider(seed=7, as_of=date(2025, 1, 3)).generate_prices().head(20)
    pd.testing.assert_frame_equal(left, right)


def test_capped_weights_sum_to_one_and_respect_cap():
    weights = MultiFactorStrategy.capped_weights({"A": .4, "B": .3, "C": .2, "D": .1}, .3)
    assert abs(sum(weights.values()) - 1) < 1e-8
    assert max(weights.values()) <= .3 + 1e-8


def test_akshare_search_normalizes_chinese_names(monkeypatch):
    fake_akshare = types.SimpleNamespace(
        stock_info_a_code_name=lambda: pd.DataFrame({"code": ["000002"], "name": ["万  科Ａ"]})
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)
    provider = AKShareDataProvider(DemoDataProvider(seed=42, as_of=date(2026, 1, 2)))
    result = provider.search_stocks("万科")
    assert len(result) == 1
    assert result.iloc[0].symbol == "000002"
    assert result.iloc[0]["name"] == "万科"
