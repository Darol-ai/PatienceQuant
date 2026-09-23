"""把tushare daily()+adj_factor()+daily_basic()的原始返回，转换成本项目
沿用的规范字段——延续baostock时代（ADR-0045）同一份下游契约的字段名，
改成从tushare的数据形状转换过来（ADR-0046）。

跟baostock时代有一个已知的、故意不掩盖的语义差异：baostock的
query_history_k_data_plus对停牌日照样返回一行（成交量/额留空或记0），
tushare的daily()对停牌日则完全不返回那一行——这里没有做"用trade_cal()
反查完整交易日历、给停牌日插一行零成交"这种重建，因为这些脚本目前不会
被运行（ADR-0046：只换代码依赖，不重新抓训练数据），没有真实数据能
验证这段重建逻辑对不对，与其写一段没验证过的代码不如老实留空。如果将来
真的要重新跑这批脚本产出新的parquet，落库前需要重新确认这一点是否
要紧——is_suspended字段目前对tushare路径下所有返回的行恒为False，只是
为了保留跟旧字段兼容的列，不是"tushare不会停牌"。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def _build_bar(row, ts_code: str, latest_factor: float, need_pb: bool) -> dict:
    """把daily+adj_factor+daily_basic合并后的一行，转换成规范字段。拆成
    单独的纯函数是为了能用fixture数据单元测试(不用真的连tushare)——见
    tests/quant_v3/test_tushare_adapter.py。
    """
    ratio = row.adj_factor / latest_factor
    bar: dict = {
        "symbol": ts_code,
        "date": row.trade_date,
        "open": float(row.open) * ratio,
        "high": float(row.high) * ratio,
        "low": float(row.low) * ratio,
        "close": float(row.close) * ratio,
        "volume": float(row.vol) if pd.notna(getattr(row, "vol", None)) else 0.0,
        "amount": float(row.amount) if pd.notna(getattr(row, "amount", None)) else 0.0,
        "turnover_rate": float(getattr(row, "turnover_rate", 0.0) or 0.0) if pd.notna(getattr(row, "turnover_rate", None)) else 0.0,
        "is_suspended": False,
    }
    if need_pb:
        pb_value: Optional[float] = getattr(row, "pb", None)
        bar["pb_mrq"] = float(pb_value) if pb_value is not None and pd.notna(pb_value) else None
    return bar


def bars_from_merged_frame(merged: pd.DataFrame, ts_code: str, need_pb: bool = False) -> list[dict]:
    """merged是daily+adj_factor(+daily_basic)已经按trade_date合并好、按
    trade_date升序排列的DataFrame——拆出这一层是为了让"怎么合并/请求
    哪些接口"(fetch_daily_bars)和"怎么把一行数据转换成规范字段"
    (这个函数)可以分开测试。"""
    if merged.empty:
        return []
    latest_factor = merged["adj_factor"].iloc[-1]
    return [_build_bar(row, ts_code, latest_factor, need_pb) for row in merged.itertuples()]


def fetch_daily_bars(pro, ts_code: str, start: str, end: str, need_pb: bool = False) -> list[dict]:
    """start/end 用tushare约定的YYYYMMDD字符串格式。返回字段跟
    baostock时代的normalize_daily_bars同名：symbol/date/open/high/low/
    close/volume/amount/turnover_rate/is_suspended[/pb_mrq]，方便下游
    消费代码不用为了换数据源而跟着改字段名。
    """
    daily = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
    if daily is None or daily.empty:
        return []
    factor = pro.adj_factor(ts_code=ts_code, start_date=start, end_date=end)
    if factor is None or factor.empty:
        raise RuntimeError(f"{ts_code} adj_factor返回空表，无法计算前复权价")
    merged = daily.merge(factor[["trade_date", "adj_factor"]], on="trade_date", how="inner")
    if merged.empty:
        raise RuntimeError(f"{ts_code} daily/adj_factor日期对不上，无法计算前复权价")

    basic_fields = "ts_code,trade_date,turnover_rate" + (",pb" if need_pb else "")
    basic = pro.daily_basic(ts_code=ts_code, start_date=start, end_date=end, fields=basic_fields)
    if basic is not None and not basic.empty:
        merged = merged.merge(basic, on=["ts_code", "trade_date"], how="left")

    merged = merged.sort_values("trade_date").reset_index(drop=True)
    return bars_from_merged_frame(merged, ts_code, need_pb)
