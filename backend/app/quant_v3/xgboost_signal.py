"""XGBoost版本的按天缓存批量预测信号源，结构和 ensemble_regression_signal.py
的LightGBM版本完全对称——同一套特征(regression_dataset.py)，只是预测
调用方式不同：xgboost的原生Booster要过DMatrix，不能像LightGBM那样直接
predict(DataFrame)。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import xgboost as xgb

from app.quant_v3.dataset import compute_features


class XGBoostEnsembleSignalSource:
    def __init__(self, year_dir: Path, history: pd.DataFrame):
        self._boosters: List[xgb.Booster] = []
        self.feature_columns: Optional[List[str]] = None
        self.group_categories: Optional[List[str]] = None
        for seed_dir in sorted(year_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            with open(seed_dir / "metadata.json", encoding="utf-8") as f:
                metadata = json.load(f)
            if self.feature_columns is None:
                self.feature_columns = metadata["feature_columns"]
                self.group_categories = metadata["group_categories"]
            booster = xgb.Booster()
            booster.load_model(str(seed_dir / "xgb_model.json"))
            self._boosters.append(booster)

        self._bars_by_symbol: Dict[str, List[dict]] = {}
        self._index_by_symbol: Dict[str, Dict[date, int]] = {}
        for symbol, group_df in history.sort_values("date").groupby("symbol"):
            bars = group_df.to_dict(orient="records")
            self._bars_by_symbol[symbol] = bars
            self._index_by_symbol[symbol] = {
                pd.Timestamp(bar["date"]).date(): i for i, bar in enumerate(bars)
            }
        self._predictions_by_day: Dict[date, Dict[str, float]] = {}

    def _predict_all_for_day(self, as_of: date) -> Dict[str, float]:
        cached = self._predictions_by_day.get(as_of)
        if cached is not None:
            return cached

        rows = []
        symbols = []
        for symbol, bars in self._bars_by_symbol.items():
            t_index = self._index_by_symbol[symbol].get(as_of)
            if t_index is None:
                continue
            features = compute_features(bars, t_index)
            if features is None:
                continue
            rows.append({**{c: features[c] for c in self.feature_columns}, "group": bars[t_index]["group"]})
            symbols.append(symbol)

        result: Dict[str, float] = {}
        if rows and self._boosters:
            frame = pd.DataFrame(rows)
            frame["group"] = pd.Categorical(frame["group"], categories=self.group_categories)
            dmatrix = xgb.DMatrix(frame, enable_categorical=True)
            booster_predictions = [booster.predict(dmatrix) for booster in self._boosters]
            averaged = sum(booster_predictions) / len(booster_predictions)
            result = {symbol: float(value) for symbol, value in zip(symbols, averaged)}

        self._predictions_by_day[as_of] = result
        return result

    def __call__(self, symbol: str, as_of: date) -> Optional[float]:
        return self._predict_all_for_day(as_of).get(symbol)


class XGBoostWalkForwardSignalSource:
    def __init__(self, model_root: Path, history_parquet: Path):
        history = pd.read_parquet(history_parquet)
        self._by_year: Dict[int, XGBoostEnsembleSignalSource] = {}
        for year_dir in sorted(model_root.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue
            self._by_year[year] = XGBoostEnsembleSignalSource(year_dir, history)

    def __call__(self, symbol: str, as_of: date) -> Optional[float]:
        source = self._by_year.get(as_of.year)
        if source is None:
            return None
        return source(symbol, as_of)
