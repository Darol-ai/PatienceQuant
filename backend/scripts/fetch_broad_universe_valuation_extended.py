"""自由探索阶段（docs/adr/0034）：给30支广泛候选池补抓pbMRQ(2010年起，
和broad_universe_history_extended.parquet同样的历史深度)，用来在当前
最终框架(回归+集成+动量兜底)下重新验证时间序列估值特征是否有增量信息。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/fetch_broad_universe_valuation_extended.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import baostock as bs
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.broad_universe import BROAD_STOCKS
from scripts.fetch_diversified_universe_history import fetch_one

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "broad_universe_valuation_extended.parquet"


def main() -> None:
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login.error_msg}")

    all_bars: list[dict] = []
    try:
        for stock in BROAD_STOCKS:
            print(f"拉取 {stock['symbol']} {stock['name']} ...", end=" ", flush=True)
            bars = fetch_one(stock["symbol"], "2010-01-01", "2026-09-11")
            for bar in bars:
                bar["group"] = "综合"
                bar["name"] = stock["name"]
            all_bars.extend(bars)
            valid_pb = [b["pb_mrq"] for b in bars if b["pb_mrq"] is not None]
            print(f"{len(bars)} 条，pbMRQ有效 {len(valid_pb)} 条" if bars else "无数据")
    finally:
        bs.logout()

    frame = pd.DataFrame(all_bars)
    frame = frame.dropna(subset=["pb_mrq"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    print(f"\n落库完成：{OUT_PATH}，共 {len(frame)} 行")


if __name__ == "__main__":
    main()
