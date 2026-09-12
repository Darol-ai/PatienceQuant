"""拉取 V3 方案 A 阶段 10 支固定股票的前复权日线，落到本地 parquet。

用 BaoStock，不用 AKShare——这台机器上 AKShare 走的 eastmoney 接口会被限流/
连接重置（见仓库根目录 docs/setup.md），BaoStock 验证过稳定。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/fetch_a_phase_history.py [--start 2016-01-01] [--end 2026-09-11]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

# 这台机器默认把出网流量走本地代理，但代理对国内数据站点的路由不稳定，
# 直连反而更可靠（同样的坑记在仓库根目录 docs/setup.md）。
for _var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_var, None)

import baostock as bs
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.a_phase_universe import A_PHASE_STOCKS
from app.quant_v3.baostock_adapter import normalize_daily_bars

FIELDS = "date,code,open,high,low,close,volume,amount,turn,pctChg,tradestatus"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "a_phase_history.parquet"


def to_bao_code(symbol: str) -> str:
    code, suffix = symbol.split(".")
    return f"{'sh' if suffix == 'SH' else 'sz'}.{code}"


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
    return normalize_daily_bars(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login.error_msg}")

    all_bars: list[dict] = []
    try:
        for stock in A_PHASE_STOCKS:
            print(f"拉取 {stock['symbol']} {stock['name']} ...", end=" ", flush=True)
            bars = fetch_one(stock["symbol"], args.start, args.end)
            for bar in bars:
                bar["group"] = stock["group"]
                bar["name"] = stock["name"]
            all_bars.extend(bars)
            print(f"{len(bars)} 条，{bars[0]['date']} ~ {bars[-1]['date']}" if bars else "无数据")
    finally:
        bs.logout()

    frame = pd.DataFrame(all_bars)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    print(f"\n落库完成：{OUT_PATH}，共 {len(frame)} 行")


if __name__ == "__main__":
    main()
