"""加载训练好的 LightGBM 模型 + 温度校准器，包装成 V3Strategy 需要的
`(symbol, as_of) -> (p_up, p_down)` signal_source。

训练集用了截面标准化（同一天在股票池内把原始特征值换成百分位排名，见
quant_v3.dataset.cross_sectional_rank 和相关 ADR）——预测阶段必须用同一套
处理，所以这里每次查询都会先把"当天全部股票"的原始特征都算出来、排好名，
再取查询的那只股票的排名结果去预测，不能只算查询股票自己（截面排名离不开
"和谁比"）。

历史不足 120 天、当天不在历史范围内、或模型/元数据没找到时，返回中性占位
信号（p_up=p_down=0，永不触发买入或模型退出）——不伪造预测，同
V3Strategy 默认 neutral_signal 的原则一致（V3 方案 4.2 节："某股模型输入
缺失或模型不可用时，暂停该股新增模型指令"）。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.quant_v3.dataset import compute_features

NEUTRAL = (0.0, 0.0)


class TrainedSignalSource:
    def __init__(self, model_dir: Path, history: pd.DataFrame):
        with open(model_dir / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.booster = lgb.Booster(model_file=str(model_dir / "lgbm_model.txt"))
        self.feature_columns = self.metadata["feature_columns"]
        self.group_categories = self.metadata["group_categories"]
        self.tau = float(self.metadata["temperature"])
        # 训练样本表当时有没有做截面标准化，预测阶段必须用同一套处理，
        # 否则会有训练/预测不一致的隐藏 bug（见 build_training_dataset.py /
        # train_v3_walkforward.py 的 USE_CROSS_SECTIONAL_RANK 开关）。
        self.use_cross_sectional_rank = bool(self.metadata.get("use_cross_sectional_rank", False))
        # label_order 里 DOWN/NEUTRAL/UP 对应模型输出概率的列顺序
        self.down_index = self.metadata["label_order"].index("DOWN")
        self.up_index = self.metadata["label_order"].index("UP")

        self._bars_by_symbol: Dict[str, List[dict]] = {}
        self._index_by_symbol: Dict[str, Dict[date, int]] = {}
        for symbol, group_df in history.sort_values("date").groupby("symbol"):
            bars = group_df.to_dict(orient="records")
            self._bars_by_symbol[symbol] = bars
            self._index_by_symbol[symbol] = {
                pd.Timestamp(bar["date"]).date(): i for i, bar in enumerate(bars)
            }

        self._ranked_cache: Dict[date, pd.DataFrame] = {}

    def _calibrate(self, raw_proba: np.ndarray) -> np.ndarray:
        clipped = np.clip(raw_proba, 1e-12, None)
        q = np.exp(np.log(clipped) / self.tau)
        return q / q.sum()

    def _cross_sectional_features_for_day(self, as_of: date) -> pd.DataFrame:
        """算出 as_of 这天全部股票的原始特征，再做截面百分位排名。按天缓存，
        避免同一天被查询 10 次就重算 10 次（V3Strategy 每只股票查一次）。"""
        cached = self._ranked_cache.get(as_of)
        if cached is not None:
            return cached

        rows = []
        for symbol, bars in self._bars_by_symbol.items():
            t_index = self._index_by_symbol[symbol].get(as_of)
            if t_index is None:
                continue
            features = compute_features(bars, t_index)
            if features is None:
                continue
            rows.append({"symbol": symbol, "group": bars[t_index]["group"], **features})

        frame = pd.DataFrame(rows)
        if not frame.empty and self.use_cross_sectional_rank:
            for column in self.feature_columns:
                frame[column] = frame[column].rank(pct=True)
        self._ranked_cache[as_of] = frame
        return frame

    def __call__(self, symbol: str, as_of: date) -> Tuple[float, float]:
        frame = self._cross_sectional_features_for_day(as_of)
        if frame.empty or symbol not in frame["symbol"].values:
            return NEUTRAL

        row = frame[frame.symbol == symbol][self.feature_columns + ["group"]].copy()
        row["group"] = pd.Categorical(row["group"], categories=self.group_categories)

        raw_proba = self.booster.predict(row)[0]
        calibrated = self._calibrate(raw_proba)
        return float(calibrated[self.up_index]), float(calibrated[self.down_index])


def load_signal_source(model_dir: Path, history_parquet: Path) -> TrainedSignalSource:
    history = pd.read_parquet(history_parquet)
    return TrainedSignalSource(model_dir, history)
