from app.quant_v3.csi300_universe import CSI300_GROUP_BUDGETS, csi300_stocks


def test_csi300_stocks_has_exactly_300_real_constituents():
    stocks = csi300_stocks()
    assert len(stocks) == 300
    symbols = {s["symbol"] for s in stocks}
    assert len(symbols) == 300  # 全部去重后仍是300，没有重复代码
    for stock in stocks:
        assert stock["symbol"].endswith((".SH", ".SZ"))
        assert stock["name"]
        assert stock["group"] == "沪深300"


def test_csi300_group_budgets_sum_to_one():
    assert sum(CSI300_GROUP_BUDGETS.values()) == 1.0
