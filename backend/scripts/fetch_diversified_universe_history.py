"""拉取分散化候选池（跨13个不同行业，见 app/quant_v3/diversified_universe.py）
的历史行情，额外带 pbMRQ（估值分位特征需要），落到独立的
data/diversified_universe_history.parquet，不影响官方 A 阶段数据管线。

连接层（限速/超时/失败重试）统一走 app.data.tushare_client，见
docs/adr/0046——这份脚本目前不会被运行（现有parquet保持不动），只是把
baostock时代的代码依赖换成tushare。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/fetch_diversified_universe_history.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.tushare_client import TushareQueryFailed, run as run_tushare
from app.quant_v3.diversified_universe import DIVERSIFIED_STOCKS
from app.quant_v3.tushare_adapter import fetch_daily_bars

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "diversified_universe_history.parquet"


def fetch_one(symbol: str, start: str, end: str) -> list[dict]:
    start_compact = start.replace("-", "")
    end_compact = end.replace("-", "")

    def _fetch(pro):
        return fetch_daily_bars(pro, symbol, start_compact, end_compact, need_pb=True)

    return run_tushare(_fetch)


def main() -> None:
    all_bars: list[dict] = []
    failed: list[str] = []
    for stock in DIVERSIFIED_STOCKS:
        print(f"拉取 {stock['symbol']} {stock['name']} ...", end=" ", flush=True)
        try:
            bars = fetch_one(stock["symbol"], "2016-01-01", "2026-09-11")
        except TushareQueryFailed as exc:
            print(f"重试3次后仍失败: {exc}")
            failed.append(stock["symbol"])
            continue
        for bar in bars:
            bar["group"] = stock["group"]
            bar["name"] = stock["name"]
        all_bars.extend(bars)
        valid_pb = [b["pb_mrq"] for b in bars if b["pb_mrq"] is not None]
        print(f"{len(bars)} 条，pbMRQ有效 {len(valid_pb)} 条" if bars else "无数据")

    if failed:
        print(f"\n以下{len(failed)}支股票重试3次后仍未拿到真实数据，需要重跑脚本补齐：{failed}")

    frame = pd.DataFrame(all_bars)
    frame = frame.dropna(subset=["pb_mrq"])  # 上市初期可能还没有 pbMRQ，丢弃这些行不硬凑
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    print(f"\n落库完成：{OUT_PATH}，共 {len(frame)} 行")


if __name__ == "__main__":
    main()
