"""扩大候选池实验专用：把 data/expanded_universe_history.parquet（官方10支
+新增9支）组装成训练样本表，逻辑和 build_training_dataset.py 完全一致，只是
输入/输出换成扩大候选池的路径，官方数据管线不受影响。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/build_expanded_training_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.dataset import build_sample

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "expanded_universe_history.parquet"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "expanded_universe_training_samples.parquet"


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

    print(f"\n共 {len(frame)} 条可用样本（跳过 {skipped_none} 个历史不够/标签未成熟/波动率为零的位置）")
    print("标签分布：")
    print(frame["label"].value_counts())
    print(f"落库：{OUT_PATH}")


if __name__ == "__main__":
    main()
