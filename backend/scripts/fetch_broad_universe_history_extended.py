"""自由探索阶段（docs/adr/0016）：诊断出2019年模型训练样本远少于其他
年份（只有2.5年历史）是2019年表现差的一个真实原因（不是策略机制问题），
往前重新抓全部30支股票从2010年开始的历史（部分股票上市更晚，各自从
实际上市日起算），落到独立的 data/broad_universe_history_extended.parquet，
不覆盖原有 data/broad_universe_history.parquet。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/fetch_broad_universe_history_extended.py
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

from app.quant_v3.baostock_adapter import normalize_daily_bars
from app.quant_v3.broad_universe import BROAD_STOCKS
from scripts.fetch_a_phase_history import FIELDS, to_bao_code

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "broad_universe_history_extended.parquet"


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
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login.error_msg}")

    all_bars: list[dict] = []
    try:
        for stock in BROAD_STOCKS:
            print(f"拉取 {stock['symbol']} {stock['name']} ...", end=" ", flush=True)
            bars = fetch_one(stock["symbol"], "2010-01-01", "2026-09-11")
            all_bars.extend(bars)
            print(f"{len(bars)} 条，{bars[0]['date']} ~ {bars[-1]['date']}" if bars else "无数据")
    finally:
        bs.logout()

    frame = pd.DataFrame(all_bars)
    frame["group"] = "综合"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    print(f"\n落库完成：{OUT_PATH}，共 {len(frame)} 行，{frame['symbol'].nunique()} 支股票")
    print(frame.groupby("symbol")["date"].min().to_string())


if __name__ == "__main__":
    main()
