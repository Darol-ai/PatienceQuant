"""消融实验专用（docs/adr/0012）：分散化候选池 + 只用官方14项特征（不加
换手率/估值），用来单独测"只换股票池、不加特征"这一个变量。复用官方
`dataset.build_sample`，输入是分散化候选池的历史行情。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/build_diversified_base_training_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.dataset import build_sample

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "diversified_universe_history.parquet"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "diversified_base_training_samples.parquet"


def main() -> None:
    raw = pd.read_parquet(IN_PATH)
    raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)

    rows: list[dict] = []
    skipped_none = 0
    for symbol, group_df in raw.groupby("symbol", sort=False):
        group_name = group_df["group"].iloc[0]
        bars = group_df.to_dict(orient="records")
        for t_index in range(len(bars)):
            sample = build_sample(bars, t_index, group=group_name)
            if sample is None:
                skipped_none += 1
                continue
            sample["symbol"] = symbol
            rows.append(sample)
        print(f"{symbol}: {len(bars)} 天原始数据 -> 已处理")

    frame = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)

    print(f"\n共 {len(frame)} 条可用样本（跳过 {skipped_none} 个）")
    print("标签分布：")
    print(frame["label"].value_counts())
    print(f"落库：{OUT_PATH}")


if __name__ == "__main__":
    main()
