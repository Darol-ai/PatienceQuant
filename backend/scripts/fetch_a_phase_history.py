"""拉取 V3 方案 A 阶段 10 支固定股票的前复权日线，落到本地 parquet。

连接层（限速/超时/失败重试）统一走 app.data.tushare_client，见
docs/adr/0046——这份脚本目前不会被运行（现有parquet保持不动），只是把
baostock时代的代码依赖换成tushare，代码结构和字段名尽量保持一致。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/fetch_a_phase_history.py [--start 2016-01-01] [--end 2026-09-11]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.tushare_client import TushareQueryFailed, run as run_tushare
from app.quant_v3.a_phase_universe import A_PHASE_STOCKS
from app.quant_v3.tushare_adapter import fetch_daily_bars

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "a_phase_history.parquet"


def to_ts_code(symbol: str) -> str:
    return symbol  # A_PHASE_STOCKS里的symbol已经是xxxxxx.SH/.SZ格式，跟tushare约定一致


def fetch_one(symbol: str, start: str, end: str) -> list[dict]:
    ts_code = to_ts_code(symbol)
    start_compact = start.replace("-", "")
    end_compact = end.replace("-", "")

    def _fetch(pro):
        return fetch_daily_bars(pro, ts_code, start_compact, end_compact)

    return run_tushare(_fetch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    all_bars: list[dict] = []
    failed: list[str] = []
    for stock in A_PHASE_STOCKS:
        print(f"拉取 {stock['symbol']} {stock['name']} ...", end=" ", flush=True)
        try:
            bars = fetch_one(stock["symbol"], args.start, args.end)
        except TushareQueryFailed as exc:
            print(f"重试3次后仍失败: {exc}")
            failed.append(stock["symbol"])
            continue
        for bar in bars:
            bar["group"] = stock["group"]
            bar["name"] = stock["name"]
        all_bars.extend(bars)
        print(f"{len(bars)} 条，{bars[0]['date']} ~ {bars[-1]['date']}" if bars else "无数据")

    if failed:
        print(f"\n以下{len(failed)}支股票重试3次后仍未拿到真实数据，需要重跑脚本补齐：{failed}")

    frame = pd.DataFrame(all_bars)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    print(f"\n落库完成：{OUT_PATH}，共 {len(frame)} 行")


if __name__ == "__main__":
    main()
