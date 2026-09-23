import pandas as pd
import pytest

from app.quant_v3.tushare_adapter import bars_from_merged_frame

# 跟 pro.daily()+pro.adj_factor()+pro.daily_basic() merge 之后的行形状一致
# （tushare统一返回float/NaN，不像baostock那样所有数值字段都是字符串）。
MERGED = pd.DataFrame([
    {"ts_code": "601288.SH", "trade_date": "20160104", "open": 1.8, "high": 1.85, "low": 1.75, "close": 1.78,
     "vol": 191312.7, "amount": 60991.8, "adj_factor": 100.0, "turnover_rate": 0.065060},
    {"ts_code": "601288.SH", "trade_date": "20160105", "open": 1.79, "high": 1.9, "low": 1.78, "close": 1.88,
     "vol": 200000.0, "amount": 65000.0, "adj_factor": 101.0, "turnover_rate": 0.07},
])


def test_bars_from_merged_frame_builds_symbol_and_qfq_price():
    bars = bars_from_merged_frame(MERGED, "601288.SH")

    assert bars[0]["symbol"] == "601288.SH"
    assert bars[0]["date"] == "20160104"
    # 前复权：锚定在这个merged frame自己最后一行(101.0)的复权因子，
    # 不是锚定"今天"——同一批数据内部相对收益不受锚点选择影响。
    assert bars[0]["close"] == pytest.approx(1.78 * 100.0 / 101.0, abs=1e-6)
    assert bars[1]["close"] == pytest.approx(1.88, abs=1e-6)  # 最后一行本身不缩放


def test_bars_from_merged_frame_preserves_row_order():
    bars = bars_from_merged_frame(MERGED, "601288.SH")

    assert [b["date"] for b in bars] == ["20160104", "20160105"]


def test_bars_from_merged_frame_carries_volume_amount_turnover():
    bars = bars_from_merged_frame(MERGED, "601288.SH")

    assert bars[0]["volume"] == pytest.approx(191312.7, abs=1e-3)
    assert bars[0]["amount"] == pytest.approx(60991.8, abs=1e-3)
    assert bars[0]["turnover_rate"] == pytest.approx(0.065060, abs=1e-6)
    assert bars[0]["is_suspended"] is False


def test_bars_from_merged_frame_missing_turnover_defaults_to_zero_not_a_crash():
    """daily_basic()是left join——某天没有基本面快照时turnover_rate/pb
    会是NaN，不能直接float(nan)传下去，要归零(不是tradestatus=0那种"已知
    停牌"，只是这个接口这天没有数据，跟baostock时代的语义不完全一样，
    但同样不能让NaN流进下游因子计算)。"""
    frame = MERGED.copy()
    frame.loc[0, "turnover_rate"] = float("nan")

    bars = bars_from_merged_frame(frame, "601288.SH")

    assert bars[0]["turnover_rate"] == 0.0


def test_bars_from_merged_frame_can_include_pb_mrq_when_requested():
    frame = MERGED.copy()
    frame["pb"] = [1.5, None]

    bars = bars_from_merged_frame(frame, "601288.SH", need_pb=True)

    assert bars[0]["pb_mrq"] == pytest.approx(1.5, abs=1e-6)
    assert bars[1]["pb_mrq"] is None


def test_bars_from_merged_frame_empty_input_returns_empty_list():
    assert bars_from_merged_frame(MERGED.iloc[0:0], "601288.SH") == []
