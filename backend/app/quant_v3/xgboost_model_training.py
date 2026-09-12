"""对齐主流做法（Qlib在CSI300上的公开基准：XGBoost AR≈7.8%/IR≈0.91，和
LightGBM同属"树模型里持续有效"的一档，见调研记录）——同一套特征/标签
流水线(regression_dataset.py)，只换模型算法，作为第二个独立策略的
信号来源，不是LightGBM策略的变体。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd
import xgboost as xgb

from app.quant_v3.regression_dataset import REGRESSION_FEATURE_COLUMNS


@dataclass
class TrainedXGBFold:
    booster: xgb.Booster
    metadata: dict = field(default_factory=dict)


def train_xgboost_fold(
    frame: pd.DataFrame,
    train_end: str,
    calib_start: str,
    calib_end: str,
    feature_columns: List[str] = REGRESSION_FEATURE_COLUMNS,
    random_state: int = 42,
) -> TrainedXGBFold:
    frame = frame.copy()
    frame["group"] = frame["group"].astype("category")

    train = frame[frame.label_end < train_end]
    calib = frame[(frame.date >= calib_start) & (frame.date <= calib_end) & (frame.label_end < calib_end)]

    x_train = train[feature_columns + ["group"]]
    y_train = train["forward_return"]
    x_calib = calib[feature_columns + ["group"]]
    y_calib = calib["forward_return"]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=4, learning_rate=0.05, min_child_weight=100,
        n_estimators=300, subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, n_jobs=4, enable_categorical=True, tree_method="hist",
        verbosity=0,
    )
    fit_kwargs = {}
    if len(calib) >= 60:
        fit_kwargs["eval_set"] = [(x_calib, y_calib)]
        fit_kwargs["verbose"] = False
        model.set_params(early_stopping_rounds=30)
    model.fit(x_train, y_train, **fit_kwargs)

    metadata = {
        "feature_columns": feature_columns,
        "group_categories": list(x_train["group"].cat.categories),
        "train_end": train_end,
        "calib_window": [calib_start, calib_end],
        "train_samples": int(len(train)),
        "calib_samples": int(len(calib)),
        "best_iteration": int(getattr(model, "best_iteration", 0) or model.n_estimators),
    }
    return TrainedXGBFold(booster=model.get_booster(), metadata=metadata)
