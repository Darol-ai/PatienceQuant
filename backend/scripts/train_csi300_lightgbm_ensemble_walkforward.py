"""沪深300universe版本的LightGBM 5种子集成滚动训练(和broad版本方法论
完全一致，见docs/adr/0028，只是universe换成了沪深300真实成分股)。
产出到 data/csi300_lightgbm_ensemble_model_walkforward/<year>/seed<i>/。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/train_csi300_lightgbm_ensemble_walkforward.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant_v3.regression_dataset import REGRESSION_FEATURE_COLUMNS

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi300_regression_h90_training_samples.parquet"
OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "csi300_lightgbm_ensemble_model_walkforward"
TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
SEEDS = [42, 7, 123, 2024, 999]


def main() -> None:
    frame = pd.read_parquet(DATA_PATH)
    frame["group"] = frame["group"].astype("category")

    for year in TEST_YEARS:
        train_end = f"{year - 1}-07-01"
        calib_start = f"{year - 1}-07-01"
        calib_end = f"{year - 1}-12-31"
        train = frame[frame.label_end < train_end]
        calib = frame[(frame.date >= calib_start) & (frame.date <= calib_end) & (frame.label_end < calib_end)]
        x_train = train[REGRESSION_FEATURE_COLUMNS + ["group"]]
        y_train = train["forward_return"]
        x_calib = calib[REGRESSION_FEATURE_COLUMNS + ["group"]]
        y_calib = calib["forward_return"]

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

            out_dir = OUT_ROOT / str(year) / f"seed{seed_idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            model.booster_.save_model(str(out_dir / "lgbm_model.txt"))
            with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"feature_columns": REGRESSION_FEATURE_COLUMNS, "group_categories": list(x_train["group"].cat.categories)},
                    f, ensure_ascii=False, indent=2,
                )
        print(f"{year}: {len(SEEDS)}个种子训练完成，训练样本{len(train)}")

    print(f"全部完成：{OUT_ROOT}")


if __name__ == "__main__":
    main()
