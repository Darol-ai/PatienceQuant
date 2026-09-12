import pandas as pd
import pytest

from app.quant_v3.a_phase_data_service import APhaseDataService


def test_benchmark_is_equal_dollar_weighted_not_raw_price_average():
    """自由探索阶段发现的真实bug（docs/adr/0019）：原实现直接对收盘价取
    算术平均，价格量级差异巨大的股票池里（比如30支候选池里贵州茅台
    ¥500+ vs 其他多数¥2~30）会让高价股在"基准"里被隐性赋予远超其他股票
    的权重，根本不是"每支股票投入等量资金买入持有"的真实等权重基准。
    正确做法：每支股票先按各自起始价格归一化(除以T0收盘价)，再取均值——
    这样每支股票不论绝对价格高低，对基准的贡献都是"等量资金买入后的
    涨跌幅"，不是"等量股数"。

    这里构造两支股票：A从100涨到200(+100%)，B从1000跌到500(-50%)。
    等权重(等资金)基准的真实收益应该是 (100%+(-50%))/2=+25%，跟"谁的
    绝对价格更高"无关。如果实现是错误的"直接平均收盘价"，结果会被
    B的高绝对价格level主导（(200+500)/2 相对 (100+1000)/2 只有约-22%），
    完全是另一个方向。"""
    history = pd.DataFrame({
        "symbol": ["A", "A", "B", "B"],
        "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        "close": [100.0, 200.0, 1000.0, 500.0],
    })
    service = APhaseDataService(history)

    result = service.benchmark(start=pd.Timestamp("2024-01-01").date(), end=pd.Timestamp("2024-01-02").date())

    start_value = result.loc[result.trade_date == pd.Timestamp("2024-01-01").date(), "adj_close"].iloc[0]
    end_value = result.loc[result.trade_date == pd.Timestamp("2024-01-02").date(), "adj_close"].iloc[0]
    total_return = end_value / start_value - 1

    assert total_return == pytest.approx(0.25)
