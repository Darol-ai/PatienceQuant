"""消融实验专用（docs/adr/0012）：给 V3 官方 A 阶段 10 支股票补抓 pbMRQ，
落到独立的 data/a_phase_valuation_history.parquet，不覆盖官方
a_phase_history.parquet——用来单独测"只加特征、不换股票池"这一个变量。

复用 fetch_diversified_universe_history.py 里已经写好的 fetch_one（按
symbol/start/end 拉带 pbMRQ 的日线，和股票池无关，可以直接复用）。连接层
（限速/超时/失败重试）统一走 app.data.tushare_client，见 docs/adr/0046——
这份脚本目前不会被运行（现有parquet保持不动），只是把baostock时代的
代码依赖换成tushare。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/fetch_a_phase_valuation_history.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.tushare_client import TushareQueryFailed
from app.quant_v3.a_phase_universe import A_PHASE_STOCKS
from scripts.fetch_diversified_universe_history import fetch_one

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "a_phase_valuation_history.parquet"


def main() -> None:
    all_bars: list[dict] = []
    failed: list[str] = []
    for stock in A_PHASE_STOCKS:
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
    frame = frame.dropna(subset=["pb_mrq"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    print(f"\n落库完成：{OUT_PATH}，共 {len(frame)} 行")


if __name__ == "__main__":
    main()
