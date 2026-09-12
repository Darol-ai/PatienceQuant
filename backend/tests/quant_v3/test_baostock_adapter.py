import pytest

from app.quant_v3.baostock_adapter import normalize_daily_bars

# 与 bs.query_history_k_data_plus(fields='date,code,open,high,low,close,
# volume,amount,turn,pctChg,tradestatus') 实际返回的行格式完全一致的 fixture
# （BaoStock 所有数值字段都是字符串；tradestatus=0 表示停牌）。
NORMAL_ROW = [
    "2016-01-04", "sh.601288", "1.8225453600", "1.8225453600", "1.7494185400",
    "1.7550436800", "191312700", "609918496.0000", "0.065060", "-3.405600", "1",
]
SUSPENDED_ROW = [
    "2016-03-03", "sh.600900", "7.9487783200", "7.9487783200", "7.9487783200",
    "7.9487783200", "0", "0.0000", "", "0.000000", "0",
]


def test_normalize_daily_bars_parses_strings_and_builds_vt_symbol():
    bars = normalize_daily_bars([NORMAL_ROW])

    assert bars[0]["symbol"] == "601288.SH"
    assert bars[0]["date"] == "2016-01-04"
    assert bars[0]["open"] == pytest.approx(1.82254536, abs=1e-6)
    assert bars[0]["volume"] == 191312700
    assert bars[0]["amount"] == pytest.approx(609918496.0, abs=1e-3)
    assert bars[0]["turnover_rate"] == pytest.approx(0.065060, abs=1e-6)
    assert bars[0]["is_suspended"] is False


def test_normalize_daily_bars_maps_shenzhen_prefix_too():
    row = ["2020-01-02", "sz.300308", "10.0", "10.5", "9.8", "10.2", "1000", "10000.0", "0.5", "1.0", "1"]

    bars = normalize_daily_bars([row])

    assert bars[0]["symbol"] == "300308.SZ"


def test_normalize_daily_bars_preserves_row_order():
    bars = normalize_daily_bars([NORMAL_ROW, SUSPENDED_ROW])

    assert [b["date"] for b in bars] == ["2016-01-04", "2016-03-03"]


def test_suspended_day_gets_zero_turnover_not_a_crash():
    """V3 方案 2.2 节：停牌日成交额记零。BaoStock 对停牌日的 turn 字段是空字符串，
    不能直接 float('')；由 tradestatus=0 判定这是"已知的零"，不是"未知缺失"。"""
    bars = normalize_daily_bars([SUSPENDED_ROW])

    assert bars[0]["is_suspended"] is True
    assert bars[0]["turnover_rate"] == 0.0
    assert bars[0]["volume"] == 0
    assert bars[0]["amount"] == 0.0


def test_suspended_day_can_leave_the_whole_row_blank_not_just_turn():
    """真实数据发现的坑：BaoStock 对停牌日的留空方式并不统一——2016 年那批
    数据填 volume='0'/amount='0.0000'，但 2021 年之后的一批数据整行
    volume/amount/turn 全部留空。两种都是 tradestatus=0，都要按"记零"处理，
    不能因为格式不同就报错或者留 None。"""
    fully_blank_row = [
        "2021-11-29", "sh.600900", "16.55", "16.55", "16.55", "16.55", "", "", "", "", "0",
    ]

    bars = normalize_daily_bars([fully_blank_row])

    assert bars[0]["is_suspended"] is True
    assert bars[0]["volume"] == 0
    assert bars[0]["amount"] == 0.0
    assert bars[0]["turnover_rate"] == 0.0


def test_empty_turnover_on_a_non_suspended_day_is_a_real_error():
    """V3 方案 2.2 节：未知缺失不能记零——tradestatus=1 却缺 turn 是真正的
    数据异常，必须报错而不是悄悄归零。"""
    anomalous_row = [
        "2016-01-04", "sh.601288", "1.0", "1.0", "1.0", "1.0", "100", "100.0", "", "0.0", "1",
    ]

    with pytest.raises(ValueError):
        normalize_daily_bars([anomalous_row])
