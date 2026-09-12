"""包装逐年滚动训练的 LightGBM 回归模型成 `(symbol, as_of) -> float | None`
信号源，直接把预测的未来收益当打分喂给 RegressionRotationStrategy 的
Top-K 排序，不需要再做概率差/校准这些三分类特有的步骤。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import pandas as pd

from app.quant_v3.dataset import compute_features


class RegressionSignalSource:
    def __init__(self, model_dir: Path, history: pd.DataFrame):
        with open(model_dir / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.booster = lgb.Booster(model_file=str(model_dir / "lgbm_model.txt"))
        self.feature_columns = self.metadata["feature_columns"]
        self.group_categories = self.metadata["group_categories"]

        self._bars_by_symbol: Dict[str, List[dict]] = {}
        self._index_by_symbol: Dict[str, Dict[date, int]] = {}
        for symbol, group_df in history.sort_values("date").groupby("symbol"):
            bars = group_df.to_dict(orient="records")
            self._bars_by_symbol[symbol] = bars
            self._index_by_symbol[symbol] = {
                pd.Timestamp(bar["date"]).date(): i for i, bar in enumerate(bars)
            }

    def __call__(self, symbol: str, as_of: date) -> Optional[float]:
        bars = self._bars_by_symbol.get(symbol)
        if bars is None:
            return None
        t_index = self._index_by_symbol[symbol].get(as_of)
        if t_index is None:
            return None

        features = compute_features(bars, t_index)
        if features is None:
            return None

        row = pd.DataFrame([{**{c: features[c] for c in self.feature_columns}, "group": bars[t_index]["group"]}])
        row["group"] = pd.Categorical(row["group"], categories=self.group_categories)
        return float(self.booster.predict(row)[0])


class RegressionWalkForwardSignalSource:
    def __init__(self, model_root: Path, history_parquet: Path):
        history = pd.read_parquet(history_parquet)
        self._by_year: Dict[int, RegressionSignalSource] = {}
        for year_dir in sorted(model_root.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue
            self._by_year[year] = RegressionSignalSource(year_dir, history)

    def __call__(self, symbol: str, as_of: date) -> Optional[float]:
        source = self._by_year.get(as_of.year)
        if source is None:
            return None
        return source(symbol, as_of)
