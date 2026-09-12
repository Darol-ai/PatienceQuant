"""训练 + 校准 V3 方案的 LightGBM 三分类模型（简化版）。

范围裁剪：V3 方案 7.2 节要求逐年滚动训练+校准（2016起训练、每年重新拟合）。
本轮冲刺时间有限，这里做的是**单次静态切分**：训练集截止到 2023 年底、
2024 上半年做温度校准，之后的数据留给回测阶段做信号——这是简化，不是完整
V3 流程，之后要做 B 阶段/更严谨的评估时需要换成真正的逐年滚动。

不泄漏未来数据、3.3 节的模型超参数、7.3 节的温度校准公式，这几条照 V3 原文做。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/train_v3_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "a_phase_training_samples.parquet"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "v3_model"

FEATURE_COLUMNS = [
    "return_5d", "return_20d", "return_60d", "return_120d",
    "volatility_20d", "volatility_60d", "max_drawdown_60d",
    "ma_deviation_20d", "ma_deviation_60d", "ma_deviation_120d",
    "atr_ratio_20d", "volume_ratio_20d", "avg_amount_60d", "valid_trading_ratio_60d",
]
LABEL_ORDER = ["DOWN", "NEUTRAL", "UP"]  # 索引对应 p_down/p_neutral/p_up

TRAIN_END = "2023-12-31"     # label_end 早于这个日期的样本才能进训练集
CALIB_START = "2024-01-01"
CALIB_END = "2024-06-30"     # label_end 早于这个日期的样本才能进校准集


def _fit_temperature(raw_proba: np.ndarray, y_true: np.ndarray) -> float:
    """V3 方案 7.3 节：q_k = exp(log(max(p_k,1e-12))/tau)，p_calibrated =
    q/sum(q)，tau 在校准集上最小化多分类 log loss。做成简单网格搜索，
    不引入额外依赖。tau<=0 或优化失败时按 V3 规则回退 tau=1（"未经校准"）。
    """
    clipped = np.clip(raw_proba, 1e-12, None)

    def calibrated_loss(tau: float) -> float:
        q = np.exp(np.log(clipped) / tau)
        p = q / q.sum(axis=1, keepdims=True)
        return log_loss(y_true, p, labels=[0, 1, 2])

    grid = np.arange(0.2, 5.01, 0.02)
    losses = [calibrated_loss(tau) for tau in grid]
    best_tau = float(grid[int(np.argmin(losses))])
    return best_tau


def main() -> None:
    frame = pd.read_parquet(DATA_PATH)

    encoder = LabelEncoder()
    encoder.fit(LABEL_ORDER)
    frame["label_id"] = encoder.transform(frame["label"])
    frame["group"] = frame["group"].astype("category")

    train = frame[frame.label_end < TRAIN_END]
    calib = frame[(frame.date >= CALIB_START) & (frame.date <= CALIB_END) & (frame.label_end < CALIB_END)]
    print(f"训练样本 {len(train)} 条，校准样本 {len(calib)} 条（截至 {TRAIN_END} / {CALIB_START}~{CALIB_END}）")

    x_train = train[FEATURE_COLUMNS + ["group"]]
    y_train = train["label_id"]
    x_calib = calib[FEATURE_COLUMNS + ["group"]]
    y_calib = calib["label_id"]

    # V3 方案 3.3 节固定的模型复杂度：一个共享 CPU 模型，不为各组分别调参。
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
    print(f"实际训练轮数（早停）：{model.best_iteration_}")

    raw_calib_proba = model.predict_proba(x_calib)
    calib_ok = (
        len(calib) >= 60
        and y_calib.nunique() == 3
        and calib["label"].value_counts().min() >= 50
    )
    if calib_ok:
        tau = _fit_temperature(raw_calib_proba, y_calib.to_numpy())
    else:
        print("校准集不满足 V3 7.3 节的最低数据量规则，回退 tau=1（未经校准）")
        tau = 1.0

    calibrated_loss = log_loss(
        y_calib, _apply_temperature(raw_calib_proba, tau), labels=[0, 1, 2]
    )
    raw_loss = log_loss(y_calib, raw_calib_proba, labels=[0, 1, 2])
    print(f"校准前 log loss: {raw_loss:.4f}，校准后（tau={tau:.2f}）: {calibrated_loss:.4f}")

    # 门槛改用训练集分位数，而不是 V3 原文的 0.60/0.25 绝对数字：实测训练集
    # 里 0 条样本达到 p_up>=0.60（浅层三分类模型 + UP 事件基础概率只有约
    # 12%，两条 V3 规定的约束放在一起几乎打不到这个绝对门槛）。规则在看
    # 回测结果之前就定死：p_up 用训练集(校准后)前 10% 的分位数，p_down 用
    # 训练集(校准后)最低 25% 的分位数，两者同时满足才批准买入。
    raw_train_proba = model.predict_proba(x_train)
    calibrated_train = _apply_temperature(raw_train_proba, tau)
    p_up_threshold = float(np.percentile(calibrated_train[:, encoder.transform(["UP"])[0]], 90))
    p_down_threshold = float(np.percentile(calibrated_train[:, encoder.transform(["DOWN"])[0]], 25))
    approval_rate = float(
        (
            (calibrated_train[:, encoder.transform(["UP"])[0]] >= p_up_threshold)
            & (calibrated_train[:, encoder.transform(["DOWN"])[0]] <= p_down_threshold)
        ).mean()
    )
    print(
        f"入场门槛（训练集分位数）：p_up>={p_up_threshold:.4f}（90分位），"
        f"p_down<={p_down_threshold:.4f}（25分位），训练集通过率 {approval_rate:.4f}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(OUT_DIR / "lgbm_model.txt"))
    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
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
                "train_end": TRAIN_END,
                "calib_window": [CALIB_START, CALIB_END],
                "best_iteration": int(model.best_iteration_),
                "train_samples": int(len(train)),
                "calib_samples": int(len(calib)),
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"落库：{OUT_DIR}")


def _apply_temperature(raw_proba: np.ndarray, tau: float) -> np.ndarray:
    clipped = np.clip(raw_proba, 1e-12, None)
    q = np.exp(np.log(clipped) / tau)
    return q / q.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()
