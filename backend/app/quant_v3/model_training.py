"""训练 + 校准一个 V3 三分类 LightGBM 模型（可复用的单折逻辑）。

抽出来是为了给"逐年滚动"（V3 方案 7.2 节）复用：每个测试年度 Y 都要走一遍
"训练截止 Y-1 年 7 月 → Y-1 下半年校准 → 预测年度 Y"，逻辑完全一样，只是
日期窗口不同。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = [
    "return_5d", "return_20d", "return_60d", "return_120d",
    "volatility_20d", "volatility_60d", "max_drawdown_60d",
    "ma_deviation_20d", "ma_deviation_60d", "ma_deviation_120d",
    "atr_ratio_20d", "volume_ratio_20d", "avg_amount_60d", "valid_trading_ratio_60d",
]
LABEL_ORDER = ["DOWN", "NEUTRAL", "UP"]


@dataclass
class TrainedFold:
    booster: lgb.Booster
    metadata: dict = field(default_factory=dict)


def _apply_temperature(raw_proba: np.ndarray, tau: float) -> np.ndarray:
    clipped = np.clip(raw_proba, 1e-12, None)
    q = np.exp(np.log(clipped) / tau)
    return q / q.sum(axis=1, keepdims=True)


def _fit_temperature(raw_proba: np.ndarray, y_true: np.ndarray) -> float:
    """V3 方案 7.3 节：q_k = exp(log(max(p_k,1e-12))/tau)，p_calibrated =
    q/sum(q)，tau 在校准集上最小化多分类 log loss。网格搜索，不引入额外依赖。
    """
    clipped = np.clip(raw_proba, 1e-12, None)

    def calibrated_loss(tau: float) -> float:
        q = np.exp(np.log(clipped) / tau)
        p = q / q.sum(axis=1, keepdims=True)
        return log_loss(y_true, p, labels=[0, 1, 2])

    grid = np.arange(0.2, 5.01, 0.02)
    losses = [calibrated_loss(tau) for tau in grid]
    return float(grid[int(np.argmin(losses))])


def train_fold(
    frame: pd.DataFrame,
    train_end: str,
    calib_start: str,
    calib_end: str,
    feature_columns: List[str] = FEATURE_COLUMNS,
    use_cross_sectional_rank: bool = False,
) -> TrainedFold:
    """在 frame 上按日期切分训练/校准，训练一个共享 LightGBM 三分类模型 +
    温度校准 + 训练集分位数入场门槛（见 docs/adr/0005，V3 原文的绝对阈值
    0.60/0.25 在这套模型约束下摸不到）。frame 需含 quant_v3.dataset.build_sample
    产出的列（14 项特征 + group + label + date + label_end）。
    """
    frame = frame.copy()
    encoder = LabelEncoder()
    encoder.fit(LABEL_ORDER)
    frame["label_id"] = encoder.transform(frame["label"])
    frame["group"] = frame["group"].astype("category")

    train = frame[frame.label_end < train_end]
    calib = frame[(frame.date >= calib_start) & (frame.date <= calib_end) & (frame.label_end < calib_end)]

    x_train = train[feature_columns + ["group"]]
    y_train = train["label_id"]
    x_calib = calib[feature_columns + ["group"]]
    y_calib = calib["label_id"]

    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        num_leaves=7, max_depth=3, learning_rate=0.05, min_child_samples=100,
        n_estimators=300, random_state=42, n_jobs=4, verbosity=-1,
    )
    model.fit(
        x_train, y_train,
        eval_set=[(x_calib, y_calib)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        categorical_feature=["group"],
    )

    raw_calib_proba = model.predict_proba(x_calib)
    calib_ok = (
        len(calib) >= 60
        and y_calib.nunique() == 3
        and calib["label"].value_counts().min() >= 50
    )
    tau = _fit_temperature(raw_calib_proba, y_calib.to_numpy()) if calib_ok else 1.0

    raw_train_proba = model.predict_proba(x_train)
    calibrated_train = _apply_temperature(raw_train_proba, tau)
    up_idx, down_idx = encoder.transform(["UP"])[0], encoder.transform(["DOWN"])[0]
    p_up_threshold = float(np.percentile(calibrated_train[:, up_idx], 90))
    p_down_threshold = float(np.percentile(calibrated_train[:, down_idx], 25))

    metadata = {
        "feature_columns": feature_columns,
        "group_categories": list(x_train["group"].cat.categories),
        "label_order": LABEL_ORDER,
        "temperature": tau,
        "calibrated": calib_ok,
        "p_up_threshold": p_up_threshold,
        "p_down_threshold": p_down_threshold,
        "threshold_methodology": (
            "训练集(校准后)分位数：p_up>=90分位、p_down<=25分位，"
            "训练前定死，不看回测结果反推；V3 原文的 0.60/0.25 在本模型下不可达"
        ),
        "use_cross_sectional_rank": use_cross_sectional_rank,
        "train_end": train_end,
        "calib_window": [calib_start, calib_end],
        "best_iteration": int(model.best_iteration_),
        "train_samples": int(len(train)),
        "calib_samples": int(len(calib)),
    }
    return TrainedFold(booster=model.booster_, metadata=metadata)
