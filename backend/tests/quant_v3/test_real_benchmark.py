from datetime import date

from app.quant_v3.real_benchmark import csi300_return


def test_csi300_return_over_a_known_real_window():
    # 2019年沪深300真实是普涨大年，全年应该是显著正收益(公开可查约+36%)。
    value = csi300_return(date(2019, 1, 1), date(2019, 12, 31))
    assert value is not None
    assert 0.2 < value < 0.5


def test_csi300_return_is_none_outside_available_history():
    value = csi300_return(date(1990, 1, 1), date(1990, 12, 31))
    assert value is None


def test_csi300_return_handles_non_trading_day_bounds():
    # 起止日期不要求正好是交易日——用区间内实际存在的第一/最后一条记录。
    a_weekend = date(2024, 1, 1)  # 元旦，非交易日
    value = csi300_return(a_weekend, date(2024, 1, 31))
    assert value is not None
