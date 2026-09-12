"""自由探索阶段（docs/adr/0013）：30支广泛候选池的回归训练样本表（未来
20日收益，不是三分类事件）。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/build_broad_regression_training_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.regression_dataset import build_regression_sample

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "broad_universe_history.parquet"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "broad_regression_training_samples.parquet"


def main() -> None:
    raw = pd.read_parquet(IN_PATH)
    raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)

    rows: list[dict] = []
    skipped_none = 0
    for symbol, group_df in raw.groupby("symbol", sort=False):
        group_name = group_df["group"].iloc[0]
        bars = group_df.to_dict(orient="records")
        for t_index in range(len(bars)):
            sample = build_regression_sample(bars, t_index, group=group_name, horizon=20)
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
    print(f"forward_return 分布：mean={frame['forward_return'].mean():.4f}, std={frame['forward_return'].std():.4f}")
    print(f"落库：{OUT_PATH}")


if __name__ == "__main__":
    main()
