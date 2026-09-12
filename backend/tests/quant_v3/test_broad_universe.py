from app.quant_v3.broad_universe import BROAD_STOCKS


def test_broad_universe_has_at_least_ten_stocks():
    """用户放开的唯一硬约束：至少10支股票。"""
    assert len(BROAD_STOCKS) >= 10


def test_broad_universe_has_unique_symbols():
    symbols = [s["symbol"] for s in BROAD_STOCKS]
    assert len(symbols) == len(set(symbols))


def test_broad_universe_all_in_a_single_group():
    assert {s["group"] for s in BROAD_STOCKS} == {"综合"}
