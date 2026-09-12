"""V3 方案 7.2 节的逐年滚动训练：对每个测试年度 Y，用"训练截止 Y-1年7月、
Y-1年下半年校准、预测年度 Y"这套流程单独训一个模型。

替代之前的单次静态切分（train_v3_model.py，训练一刀切到 2023 底、只校准一次
就要管 2024-2025 两年）——这正是诊断出"样本外 IC≈0"之后要补的那条 V3 原始
设计，不是又一次调参数。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/train_v3_walkforward.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.model_training import train_fold

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "a_phase_training_samples.parquet"
OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "v3_model_walkforward"

# 原本只覆盖回测窗口 2024-2025 两年；后来为了检验"策略跑输基准是不是
# 2024-2025 这种强普涨行情特有的现象"这个假设（见 docs/adr/0009），扩到
# 2019-2025，覆盖更多不同行情。2019 是能保证有 ~2.5 年训练数据的最早年份
# （训练截止 Y-1 年 7 月，数据最早到 2016 年初）。
TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

# 必须和 build_training_dataset.py 里的 USE_CROSS_SECTIONAL_RANK 保持一致——
# 训练样本表本身有没有做过截面标准化，这里只是把这个事实记进模型的
# metadata，供预测阶段（model_signal.py）判断要不要对查询特征做同样的变换。
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
            f"早停轮数 {fold.metadata['best_iteration']}，tau={fold.metadata['temperature']:.2f}，"
            f"门槛 p_up>={fold.metadata['p_up_threshold']:.4f} / p_down<={fold.metadata['p_down_threshold']:.4f}"
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
