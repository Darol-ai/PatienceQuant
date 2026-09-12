from collections import Counter

from app.quant_v3.a_phase_universe import A_PHASE_STOCKS, EXPANDED_A_PHASE_STOCKS


def test_a_phase_universe_has_exactly_ten_stocks():
    assert len(A_PHASE_STOCKS) == 10


def test_a_phase_universe_matches_v3_group_composition_of_4_3_3():
    """V3 方案 9.1 节固定研究池：红利 4、成长 3、周期 3。"""
    counts = Counter(stock["group"] for stock in A_PHASE_STOCKS)

    assert counts == {"红利": 4, "成长": 3, "周期": 3}


def test_every_stock_has_a_unique_symbol():
    symbols = [stock["symbol"] for stock in A_PHASE_STOCKS]

    assert len(symbols) == len(set(symbols))


def test_expanded_universe_is_a_strict_superset_of_the_official_pool():
    """扩大候选池实验(见相关 ADR)不能悄悄改掉官方 A 阶段 10 支股票——只
    能往每组追加，且追加后每组都比官方多，验证"扩大"确实发生了。"""
    assert set(A_PHASE_STOCKS[0].keys()) <= set(EXPANDED_A_PHASE_STOCKS[0].keys())
    official_symbols = {s["symbol"] for s in A_PHASE_STOCKS}
    expanded_symbols = {s["symbol"] for s in EXPANDED_A_PHASE_STOCKS}
    assert official_symbols < expanded_symbols

    official_counts = Counter(s["group"] for s in A_PHASE_STOCKS)
    expanded_counts = Counter(s["group"] for s in EXPANDED_A_PHASE_STOCKS)
    for group, count in official_counts.items():
        assert expanded_counts[group] > count


def test_expanded_universe_has_unique_symbols():
    symbols = [stock["symbol"] for stock in EXPANDED_A_PHASE_STOCKS]

    assert len(symbols) == len(set(symbols))
