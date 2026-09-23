"""自由探索阶段（docs/adr/0016）：诊断出2019年模型训练样本远少于其他
年份（只有2.5年历史）是2019年表现差的一个真实原因（不是策略机制问题），
往前重新抓全部30支股票从2010年开始的历史（部分股票上市更晚，各自从
实际上市日起算），落到独立的 data/broad_universe_history_extended.parquet，
不覆盖原有 data/broad_universe_history.parquet。

连接层（限速/超时/失败重试）统一走 app.data.tushare_client，见
docs/adr/0046——这份脚本目前不会被运行（现有parquet保持不动），只是把
baostock时代的代码依赖换成tushare。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/fetch_broad_universe_history_extended.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.tushare_client import TushareQueryFailed, run as run_tushare
from app.quant_v3.broad_universe import BROAD_STOCKS
from app.quant_v3.tushare_adapter import fetch_daily_bars

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "broad_universe_history_extended.parquet"


def fetch_one(symbol: str, start: str, end: str) -> list[dict]:
    start_compact = start.replace("-", "")
    end_compact = end.replace("-", "")

    def _fetch(pro):
        return fetch_daily_bars(pro, symbol, start_compact, end_compact)

    return run_tushare(_fetch)


def main() -> None:
    all_bars: list[dict] = []
    failed: list[str] = []
    for stock in BROAD_STOCKS:
        print(f"拉取 {stock['symbol']} {stock['name']} ...", end=" ", flush=True)
        try:
            bars = fetch_one(stock["symbol"], "2010-01-01", "2026-09-11")
        except TushareQueryFailed as exc:
            # 已经在连接层重试过3次(每次重新登录)——如实记录失败、继续
            # 处理剩下的股票，不拿demo/编造数据顶替，也不让一支股票拖垮
            # 整批(ADR-0045/0046)。
            print(f"重试3次后仍失败: {exc}")
            failed.append(stock["symbol"])
            continue
        all_bars.extend(bars)
        print(f"{len(bars)} 条，{bars[0]['date']} ~ {bars[-1]['date']}" if bars else "无数据")

    if failed:
        print(f"\n以下{len(failed)}支股票重试3次后仍未拿到真实数据，需要重跑脚本补齐：{failed}")

    frame = pd.DataFrame(all_bars)
    frame["group"] = "综合"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    print(f"\n落库完成：{OUT_PATH}，共 {len(frame)} 行，{frame['symbol'].nunique()} 支股票")
    print(frame.groupby("symbol")["date"].min().to_string())


if __name__ == "__main__":
    main()
