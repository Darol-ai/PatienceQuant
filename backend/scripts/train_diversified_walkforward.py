"""分散化候选池+扩充特征(16项)的逐年滚动训练，复用同一个 train_fold，只是
feature_columns 换成 ENRICHED_FEATURE_COLUMNS。产出落到独立目录
data/diversified_model_walkforward/。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/train_diversified_walkforward.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.enriched_dataset import ENRICHED_FEATURE_COLUMNS
from app.quant_v3.model_training import train_fold

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "diversified_training_samples.parquet"
OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "diversified_model_walkforward"

TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]


def main() -> None:
    frame = pd.read_parquet(DATA_PATH)

    for year in TEST_YEARS:
        train_end = f"{year - 1}-07-01"
        calib_start = f"{year - 1}-07-01"
        calib_end = f"{year - 1}-12-31"
        print(f"=== 测试年度 {year}：训练至 {train_end}，校准 {calib_start}~{calib_end} ===")

        fold = train_fold(
            frame, train_end=train_end, calib_start=calib_start, calib_end=calib_end,
            feature_columns=ENRICHED_FEATURE_COLUMNS, use_cross_sectional_rank=False,
        )
        print(
            f"训练样本 {fold.metadata['train_samples']}，校准样本 {fold.metadata['calib_samples']}，"
            f"早停轮数 {fold.metadata['best_iteration']}，tau={fold.metadata['temperature']:.2f}"
        )

        out_dir = OUT_ROOT / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        fold.booster.save_model(str(out_dir / "lgbm_model.txt"))
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(fold.metadata, f, ensure_ascii=False, indent=2)
        print(f"落库：{out_dir}\n")

    print(f"全部完成：{OUT_ROOT}")


if __name__ == "__main__":
    main()
