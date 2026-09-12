"""给模拟盘补一折2026年模型——`train_broad_regression_h90_ensemble_walkforward.py`
训练时（docs/adr/0028）只覆盖到2025年，模拟盘默认按"今天"驱动（今天已经
是2026年），需要按同样的滚动训练规则（训练截止Y-1年7月、Y-1下半年校准、
预测年度Y）补一折，不是新方法论，只是把已有规则套到下一年。

不重跑其余年份（那些已经训练过、结果已经写进ADR，不需要重新生成），
只加2026这一折。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/train_broad_regression_h90_ensemble_2026.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.regression_dataset import REGRESSION_FEATURE_COLUMNS

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "broad_regression_h90_extended_training_samples.parquet"
OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "broad_regression_h90_ensemble_model_walkforward"
SEEDS = [42, 7, 123, 2024, 999]
YEAR = 2026


def main() -> None:
    frame = pd.read_parquet(DATA_PATH)
    frame["group"] = frame["group"].astype("category")

    train_end = f"{YEAR - 1}-07-01"
    calib_start = f"{YEAR - 1}-07-01"
    calib_end = f"{YEAR - 1}-12-31"
    train = frame[frame.label_end < train_end]
    calib = frame[(frame.date >= calib_start) & (frame.date <= calib_end) & (frame.label_end < calib_end)]
    x_train = train[REGRESSION_FEATURE_COLUMNS + ["group"]]
    y_train = train["forward_return"]
    x_calib = calib[REGRESSION_FEATURE_COLUMNS + ["group"]]
    y_calib = calib["forward_return"]
    print(f"训练样本 {len(train)}，校准样本 {len(calib)}（训练截止{train_end}，校准{calib_start}~{calib_end}）")

    for seed_idx, seed in enumerate(SEEDS):
        model = lgb.LGBMRegressor(
            objective="regression", num_leaves=15, max_depth=4, learning_rate=0.05,
            min_child_samples=100, n_estimators=300, random_state=seed, n_jobs=4, verbosity=-1,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        )
        fit_kwargs = dict(callbacks=[lgb.log_evaluation(0)])
        if len(calib) >= 60:
            fit_kwargs["eval_set"] = [(x_calib, y_calib)]
            fit_kwargs["callbacks"] = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
        model.fit(x_train, y_train, categorical_feature=["group"], **fit_kwargs)

        out_dir = OUT_ROOT / str(YEAR) / f"seed{seed_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(out_dir / "lgbm_model.txt"))
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {"feature_columns": REGRESSION_FEATURE_COLUMNS, "group_categories": list(x_train["group"].cat.categories)},
                f, ensure_ascii=False, indent=2,
            )
    print(f"{YEAR}年折训练完成：{OUT_ROOT / str(YEAR)}")


if __name__ == "__main__":
    main()
