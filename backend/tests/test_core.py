import sys
import types
from datetime import date

import pandas as pd
import pytest

from app.backtest.engine import BacktestEngine
from app.data.akshare_provider import AKShareDataProvider
from app.data.demo import STOCK_SPECS, DemoDataProvider
from app.pipeline.portfolio import weigh
from app.pipeline.spec import WeightingSpec


def test_demo_catalog_has_five_groups_and_broad_coverage():
    """Demo目录只保留手工维护的真实公司(不再用前缀+序号规则生成900多支
    虚构代码凑数)——这里断言的是这份小而真实的列表本身，不是"越大越好"。
    """
    provider = DemoDataProvider(seed=42, as_of=date(2026, 1, 2))
    catalog = provider.stock_catalog()
    assert len(catalog) == len(STOCK_SPECS) == 50
    counts = catalog.groupby("group").size().to_dict()
    assert set(counts) == {"消费", "科技", "新能源", "金融", "红利/央国企"}
    assert min(counts.values()) >= 10
    assert len(catalog.industry.unique()) >= 15


def test_demo_generation_is_reproducible():
    left = DemoDataProvider(seed=7, as_of=date(2025, 1, 3)).generate_prices().head(20)
    right = DemoDataProvider(seed=7, as_of=date(2025, 1, 3)).generate_prices().head(20)
    pd.testing.assert_frame_equal(left, right)


def test_capped_weights_sum_to_one_and_respect_cap():
    scores = pd.Series({"A": 40.0, "B": 30.0, "C": 20.0, "D": 10.0})
    weights = weigh(scores, scores, WeightingSpec(type="score", max_weight=.3))
    assert abs(sum(weights.values()) - 1) < 1e-8
    assert max(weights.values()) <= .3 + 1e-8


def test_backtest_metrics_use_compounded_total_assets():
    equity = pd.Series([100.0, 110.0, 99.0, 120.0])
    benchmark = pd.Series([100.0, 101.0, 102.0, 103.0])

    metrics = BacktestEngine._metrics(equity, benchmark, pd.DataFrame(), 100.0)

    assert metrics["final_assets"] == pytest.approx(120.0)
    assert metrics["total_profit"] == pytest.approx(20.0)
    assert metrics["overall_return"] == pytest.approx(.20)
    assert metrics["total_return"] == pytest.approx(.20)
    assert metrics["compounded_return"] == pytest.approx(.20)


def test_backtest_metrics_include_sortino_calmar_and_information_ratio():
    equity = pd.Series([100.0, 110.0, 95.0, 120.0, 90.0, 130.0])
    benchmark = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])

    metrics = BacktestEngine._metrics(equity, benchmark, pd.DataFrame(), 100.0)

    # Sortino only penalizes downside days, so with the same numerator it
    # must be at least as large as Sharpe (which penalizes all volatility).
    assert metrics["sortino"] >= metrics["sharpe"]
    assert metrics["calmar"] == pytest.approx(metrics["annual_return"] / abs(metrics["max_drawdown"]))
    # Strategy has real up-and-down swings the flat-climbing benchmark
    # doesn't share, so tracking error (and thus IR) must be nonzero.
    assert metrics["information_ratio"] != 0.0


def test_backtest_metrics_profit_loss_ratio_and_holding_days_from_round_trips():
    trades = pd.DataFrame([
        {"trade_date": date(2024, 1, 2), "symbol": "A", "side": "BUY", "quantity": 100, "price": 10.0, "amount": 1000.0, "fee": 1.0, "reason": ""},
        {"trade_date": date(2024, 1, 12), "symbol": "A", "side": "SELL", "quantity": 100, "price": 12.0, "amount": 1200.0, "fee": 1.0, "reason": ""},
        {"trade_date": date(2024, 2, 1), "symbol": "B", "side": "BUY", "quantity": 100, "price": 20.0, "amount": 2000.0, "fee": 1.0, "reason": ""},
        {"trade_date": date(2024, 2, 6), "symbol": "B", "side": "SELL", "quantity": 100, "price": 18.0, "amount": 1800.0, "fee": 1.0, "reason": ""},
    ])
    equity = pd.Series([100.0, 105.0, 102.0, 108.0])
    benchmark = pd.Series([100.0, 101.0, 102.0, 103.0])

    metrics = BacktestEngine._metrics(equity, benchmark, trades, 100.0)

    # Round trip A: bought 100@10 (+1 fee), sold 100@12 (-1 fee) -> pnl = 1200-1-1001 = 198, held 10 days.
    # Round trip B: bought 100@20 (+1 fee), sold 100@18 (-1 fee) -> pnl = 1800-1-2001 = -202, held 5 days.
    assert metrics["profit_loss_ratio"] == pytest.approx(198.0 / 202.0)
    assert metrics["avg_holding_days"] == pytest.approx((10 + 5) / 2)


def test_round_trip_trades_matches_fifo_lots_across_partial_sells():
    trades = pd.DataFrame([
        {"trade_date": date(2024, 1, 1), "symbol": "A", "side": "BUY", "quantity": 100, "price": 10.0, "amount": 1000.0, "fee": 0.0, "reason": ""},
        {"trade_date": date(2024, 1, 5), "symbol": "A", "side": "BUY", "quantity": 100, "price": 12.0, "amount": 1200.0, "fee": 0.0, "reason": ""},
        {"trade_date": date(2024, 1, 20), "symbol": "A", "side": "SELL", "quantity": 150, "price": 15.0, "amount": 2250.0, "fee": 0.0, "reason": ""},
    ])

    round_trips = BacktestEngine._round_trip_trades(trades)

    assert len(round_trips) == 2
    assert round_trips.quantity.tolist() == [100, 50]
    assert round_trips.holding_days.tolist() == [19, 15]
    assert round_trips.pnl.iloc[0] == pytest.approx(100 * (15.0 - 10.0))
    assert round_trips.pnl.iloc[1] == pytest.approx(50 * (15.0 - 12.0))


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
