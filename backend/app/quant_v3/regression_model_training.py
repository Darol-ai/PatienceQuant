"""自由探索阶段（docs/adr/0013）：LightGBM 回归版单折训练——预测未来20日
收益（连续值），不是三分类概率，所以不需要 model_training.py 里的标签
编码/温度校准那一套，直接用校准窗口做早停。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import lightgbm as lgb
import pandas as pd

from app.quant_v3.regression_dataset import REGRESSION_FEATURE_COLUMNS


@dataclass
class TrainedRegressionFold:
    booster: lgb.Booster
    metadata: dict = field(default_factory=dict)


def train_regression_fold(
    frame: pd.DataFrame,
    train_end: str,
    calib_start: str,
    calib_end: str,
    feature_columns: List[str] = REGRESSION_FEATURE_COLUMNS,
) -> TrainedRegressionFold:
    frame = frame.copy()
    frame["group"] = frame["group"].astype("category")

    train = frame[frame.label_end < train_end]
    calib = frame[(frame.date >= calib_start) & (frame.date <= calib_end) & (frame.label_end < calib_end)]

    x_train = train[feature_columns + ["group"]]
    y_train = train["forward_return"]
    x_calib = calib[feature_columns + ["group"]]
    y_calib = calib["forward_return"]

    model = lgb.LGBMRegressor(
        objective="regression",
        num_leaves=15, max_depth=4, learning_rate=0.05, min_child_samples=100,
        n_estimators=300, random_state=42, n_jobs=4, verbosity=-1,
    )
    fit_kwargs = dict(callbacks=[lgb.log_evaluation(0)])
    if len(calib) >= 60:
        fit_kwargs["eval_set"] = [(x_calib, y_calib)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
    model.fit(x_train, y_train, categorical_feature=["group"], **fit_kwargs)

    metadata = {
        "feature_columns": feature_columns,
        "group_categories": list(x_train["group"].cat.categories),
        "train_end": train_end,
        "calib_window": [calib_start, calib_end],
        "train_samples": int(len(train)),
        "calib_samples": int(len(calib)),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or model.n_estimators),
    }
    return TrainedRegressionFold(booster=model.booster_, metadata=metadata)
