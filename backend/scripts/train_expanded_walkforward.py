"""扩大候选池实验专用逐年滚动训练，逻辑和 train_v3_walkforward.py 完全一样
（复用 model_training.train_fold），只是训练样本表换成
expanded_universe_training_samples.parquet（19支股票），产出模型落到独立
目录 data/expanded_model_walkforward/，不影响官方 A 阶段的模型。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/train_expanded_walkforward.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.model_training import train_fold

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "expanded_universe_training_samples.parquet"
OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "expanded_model_walkforward"

TEST_YEARS = [2024, 2025]
USE_CROSS_SECTIONAL_RANK = False


def main() -> None:
    frame = pd.read_parquet(DATA_PATH)

    for year in TEST_YEARS:
        train_end = f"{year - 1}-07-01"
        calib_start = f"{year - 1}-07-01"
        calib_end = f"{year - 1}-12-31"
        print(f"=== 测试年度 {year}：训练至 {train_end}，校准 {calib_start}~{calib_end} ===")

        fold = train_fold(
            frame, train_end=train_end, calib_start=calib_start, calib_end=calib_end,
            use_cross_sectional_rank=USE_CROSS_SECTIONAL_RANK,
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
