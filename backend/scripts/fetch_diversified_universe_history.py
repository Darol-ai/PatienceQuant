"""拉取分散化候选池（跨13个不同行业，见 app/quant_v3/diversified_universe.py）
的历史行情，额外带 pbMRQ（估值分位特征需要），落到独立的
data/diversified_universe_history.parquet，不影响官方 A 阶段数据管线。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/fetch_diversified_universe_history.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_var, None)

import baostock as bs
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.diversified_universe import DIVERSIFIED_STOCKS
from scripts.fetch_a_phase_history import to_bao_code

FIELDS = "date,code,open,high,low,close,volume,amount,turn,pctChg,tradestatus,pbMRQ"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "diversified_universe_history.parquet"


def _zero_if_suspended(value: str, is_suspended: bool, parser, field_name: str, code: str, date: str):
    if value == "":
        if not is_suspended:
            raise ValueError(f"{code} {date}：正常交易日却缺 {field_name}，这是未知缺失，不能记零")
        return 0
    return parser(value)


def fetch_one(symbol: str, start: str, end: str) -> list[dict]:
    rs = bs.query_history_k_data_plus(
        to_bao_code(symbol), FIELDS, start_date=start, end_date=end,
        frequency="d", adjustflag="2",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"{symbol} 查询失败: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    bars = []
    last_pb = None
    for date, code, open_, high, low, close, volume, amount, turn, _pct, tradestatus, pb_mrq in rows:
        is_suspended = tradestatus == "0"
        # 停牌日 pbMRQ 有时也留空——用最近一个有效估值顺延，不能记 0（PB=0
        # 会被 valuation_percentile 误判成"历史最便宜"）。
        if pb_mrq != "":
            last_pb = float(pb_mrq)
        bars.append({
            "symbol": symbol, "date": date,
            "open": float(open_), "high": float(high), "low": float(low), "close": float(close),
            "volume": _zero_if_suspended(volume, is_suspended, int, "volume", code, date),
            "amount": _zero_if_suspended(amount, is_suspended, float, "amount", code, date),
            "turnover_rate": _zero_if_suspended(turn, is_suspended, float, "turn", code, date),
            "is_suspended": is_suspended,
            "pb_mrq": last_pb,
        })
    return bars


def main() -> None:
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login.error_msg}")

    all_bars: list[dict] = []
    try:
        for stock in DIVERSIFIED_STOCKS:
            print(f"拉取 {stock['symbol']} {stock['name']} ...", end=" ", flush=True)
            bars = fetch_one(stock["symbol"], "2016-01-01", "2026-09-11")
            for bar in bars:
                bar["group"] = stock["group"]
                bar["name"] = stock["name"]
            all_bars.extend(bars)
            valid_pb = [b["pb_mrq"] for b in bars if b["pb_mrq"] is not None]
            print(f"{len(bars)} 条，pbMRQ有效 {len(valid_pb)} 条" if bars else "无数据")
    finally:
        bs.logout()

    frame = pd.DataFrame(all_bars)
    frame = frame.dropna(subset=["pb_mrq"])  # 上市初期可能还没有 pbMRQ，丢弃这些行不硬凑
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    print(f"\n落库完成：{OUT_PATH}，共 {len(frame)} 行")


if __name__ == "__main__":
    main()
