"""自由探索阶段（docs/adr/0013）：拉取30支广泛分散候选池的历史行情，合并
官方10支(a_phase_history.parquet)+分散化12支(diversified_universe_history.
parquet，只取OHLCV，不需要pbMRQ——回归策略这版先用官方14项特征)+新增8支，
落到 data/broad_universe_history.parquet。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/fetch_broad_universe_history.py
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

from app.quant_v3.a_phase_universe import A_PHASE_STOCKS
from app.quant_v3.baostock_adapter import normalize_daily_bars
from app.quant_v3.broad_universe import BROAD_STOCKS
from app.quant_v3.diversified_universe import DIVERSIFIED_STOCKS
from scripts.fetch_a_phase_history import FIELDS, to_bao_code

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "broad_universe_history.parquet"


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
    existing_symbols = {s["symbol"] for s in A_PHASE_STOCKS} | {s["symbol"] for s in DIVERSIFIED_STOCKS}
    new_stocks = [s for s in BROAD_STOCKS if s["symbol"] not in existing_symbols]

    official = pd.read_parquet(DATA_DIR / "a_phase_history.parquet")
    diversified = pd.read_parquet(DATA_DIR / "diversified_universe_history.parquet")
    common_cols = ["symbol", "date", "open", "high", "low", "close", "volume", "amount", "turnover_rate", "is_suspended"]
    combined = pd.concat([official[common_cols], diversified[common_cols]], ignore_index=True)

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login.error_msg}")

    new_bars: list[dict] = []
    try:
        for stock in new_stocks:
            print(f"拉取 {stock['symbol']} {stock['name']} ...", end=" ", flush=True)
            bars = fetch_one(stock["symbol"], "2016-01-01", "2026-09-11")
            new_bars.extend(bars)
            print(f"{len(bars)} 条，{bars[0]['date']} ~ {bars[-1]['date']}" if bars else "无数据")
    finally:
        bs.logout()

    new_frame = pd.DataFrame(new_bars)[common_cols]
    combined = pd.concat([combined, new_frame], ignore_index=True)
    # 统一加上单一"综合"组标签——broad universe 不用官方三组结构
    combined["group"] = "综合"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    print(f"\n落库完成：{OUT_PATH}，共 {len(combined)} 行，{combined['symbol'].nunique()} 支股票")


if __name__ == "__main__":
    main()
