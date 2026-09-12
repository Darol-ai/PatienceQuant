"""加载分散化候选池+扩充特征(16项)训练出的 LightGBM 模型，包装成
`(symbol, as_of) -> (p_up, p_down)` signal_source，供 V3Strategy 复用。
和 model_signal.TrainedSignalSource 结构一致，只是特征换成
`enriched_dataset.compute_enriched_features`（多了换手率均值/估值分位，
需要 bars 里有 turnover_rate/pb_mrq 字段），且不支持截面标准化——
ADR-0008 已经验证截面标准化在这个样本规模下没有帮助，不重复引入这个
变量。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.quant_v3.enriched_dataset import compute_enriched_features

NEUTRAL = (0.0, 0.0)


class EnrichedSignalSource:
    def __init__(self, model_dir: Path, history: pd.DataFrame):
        with open(model_dir / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.booster = lgb.Booster(model_file=str(model_dir / "lgbm_model.txt"))
        self.feature_columns = self.metadata["feature_columns"]
        self.group_categories = self.metadata["group_categories"]
        self.tau = float(self.metadata["temperature"])
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

    def _calibrate(self, raw_proba: np.ndarray) -> np.ndarray:
        clipped = np.clip(raw_proba, 1e-12, None)
        q = np.exp(np.log(clipped) / self.tau)
        return q / q.sum()

    def __call__(self, symbol: str, as_of: date) -> Tuple[float, float]:
        bars = self._bars_by_symbol.get(symbol)
        if bars is None:
            return NEUTRAL
        t_index = self._index_by_symbol[symbol].get(as_of)
        if t_index is None:
            return NEUTRAL

        features = compute_enriched_features(bars, t_index)
        if features is None:
            return NEUTRAL

        row = pd.DataFrame([{**{c: features[c] for c in self.feature_columns}, "group": bars[t_index]["group"]}])
        row["group"] = pd.Categorical(row["group"], categories=self.group_categories)

        raw_proba = self.booster.predict(row)[0]
        calibrated = self._calibrate(raw_proba)
        return float(calibrated[self.up_index]), float(calibrated[self.down_index])


class EnrichedWalkForwardSignalSource:
    def __init__(self, model_root: Path, history_parquet: Path):
        history = pd.read_parquet(history_parquet)
        self._by_year: Dict[int, EnrichedSignalSource] = {}
        for year_dir in sorted(model_root.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue
            self._by_year[year] = EnrichedSignalSource(year_dir, history)

    def __call__(self, symbol: str, as_of: date) -> Tuple[float, float]:
        source = self._by_year.get(as_of.year)
        if source is None:
            return NEUTRAL
        return source(symbol, as_of)
