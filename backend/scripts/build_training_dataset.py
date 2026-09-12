"""把 data/a_phase_history.parquet 里的原始日线，按股票逐日组装成训练样本表。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/build_training_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.dataset import build_sample, cross_sectional_rank
from app.quant_v3.model_training import FEATURE_COLUMNS

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "a_phase_history.parquet"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "a_phase_training_samples.parquet"
USE_CROSS_SECTIONAL_RANK = False


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
    # 截面标准化实测在这个 10 支股票的小样本上没有改善（甚至更差，见相关
    # ADR）——只有 10 支股票时百分位排名把连续特征压成 10 个离散档位，
    # 信息损失可能大于"跨期更稳定"带来的好处。默认关闭，保留开关方便对比。
    if USE_CROSS_SECTIONAL_RANK:
        frame = cross_sectional_rank(frame, FEATURE_COLUMNS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)

    print(f"\n共 {len(frame)} 条可用样本（跳过 {skipped_none} 个历史不够/标签未成熟/波动率为零的位置）")
    print("标签分布：")
    print(frame["label"].value_counts())
    print(f"落库：{OUT_PATH}")


if __name__ == "__main__":
    main()
