from app.quant_v3.broad_universe import BROAD_STOCKS
from app.quant_v3.research_notes import quant_v3_research_universe


def test_every_stock_in_the_final_universe_has_a_real_research_note():
    """课题要求"把自己研究的股票和板块都标注出来，最起码十个，要标注
    清楚"——不能有漏标注的股票，也不能是空字符串占位。"""
    rows = quant_v3_research_universe()

    assert len(rows) == len(BROAD_STOCKS)
    assert len(rows) >= 10
    for row in rows:
        assert row["thesis"], f"{row['symbol']} 缺少研究依据"
        assert row["risk"], f"{row['symbol']} 缺少风险标注"
        assert row["industry"]
        assert row["group"]


def test_research_universe_matches_the_strategy_symbols():
    rows = quant_v3_research_universe()
    strategy_symbols = {stock["symbol"] for stock in BROAD_STOCKS}

    assert {row["symbol"] for row in rows} == strategy_symbols
