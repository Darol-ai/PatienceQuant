"""自由探索阶段（docs/adr/0015）：回归目标 horizon 网格测试之一——预测
未来90日收益（约一个季度）。测过 20/60/90/120 四个值，90最好，
见 ADR-0015。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/build_broad_regression_h90_training_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.regression_dataset import build_regression_sample

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "broad_universe_history.parquet"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "broad_regression_h90_training_samples.parquet"


def main() -> None:
    raw = pd.read_parquet(IN_PATH)
    raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)

    rows: list[dict] = []
    for symbol, group_df in raw.groupby("symbol", sort=False):
        group_name = group_df["group"].iloc[0]
        bars = group_df.to_dict(orient="records")
        for t_index in range(len(bars)):
            sample = build_regression_sample(bars, t_index, group=group_name, horizon=90)
            if sample is None:
                continue
            sample["symbol"] = symbol
            rows.append(sample)
        print(f"{symbol}: 已处理")

    frame = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    print(f"\n共 {len(frame)} 条样本，落库：{OUT_PATH}")


if __name__ == "__main__":
    main()
