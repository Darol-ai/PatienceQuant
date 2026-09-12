"""扩大候选池实验专用：拉取 EXPANDED_A_PHASE_STOCKS 里官方 10 支之外新增的
9 支股票历史行情，和已有的 `data/a_phase_history.parquet` 合并成研究性专用
的 `data/expanded_universe_history.parquet`——不覆盖官方 A 阶段的 parquet，
两条数据管线分开，互不影响。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/fetch_expanded_universe_history.py
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

from app.quant_v3.a_phase_universe import A_PHASE_STOCKS, EXPANDED_A_PHASE_STOCKS
from app.quant_v3.baostock_adapter import normalize_daily_bars
from scripts.fetch_a_phase_history import FIELDS, fetch_one

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "expanded_universe_history.parquet"


def main() -> None:
    existing = pd.read_parquet(DATA_DIR / "a_phase_history.parquet")
    start = str(pd.Timestamp(existing["date"].min()).date())
    end = str(pd.Timestamp(existing["date"].max()).date())

    official_symbols = {s["symbol"] for s in A_PHASE_STOCKS}
    new_stocks = [s for s in EXPANDED_A_PHASE_STOCKS if s["symbol"] not in official_symbols]

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login.error_msg}")

    new_bars: list[dict] = []
    try:
        for stock in new_stocks:
            print(f"拉取 {stock['symbol']} {stock['name']} ...", end=" ", flush=True)
            bars = fetch_one(stock["symbol"], start, end)
            for bar in bars:
                bar["group"] = stock["group"]
                bar["name"] = stock["name"]
            new_bars.extend(bars)
            print(f"{len(bars)} 条，{bars[0]['date']} ~ {bars[-1]['date']}" if bars else "无数据")
    finally:
        bs.logout()

    combined = pd.concat([existing, pd.DataFrame(new_bars)], ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    print(f"\n落库完成：{OUT_PATH}，共 {len(combined)} 行（官方10支 + 新增{len(new_stocks)}支）")


if __name__ == "__main__":
    main()
