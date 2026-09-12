"""自由探索阶段（docs/adr/0028）：多个不同随机种子训练出的LightGBM回归
模型平均预测——降低单模型对训练噪声的敏感度，对所有年份统一应用，不是
只为某个问题年份定制。每年对应一个子目录，子目录下每个种子一个模型
（见 scripts/train_broad_regression_h90_ensemble_walkforward.py 的产出
结构：<year>/seed<i>/{lgbm_model.txt,metadata.json}）。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import pandas as pd

from app.quant_v3.dataset import compute_features


class EnsembleRegressionSignalSource:
    def __init__(self, year_dir: Path, history: pd.DataFrame):
        self._boosters: List[lgb.Booster] = []
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
            self._boosters.append(lgb.Booster(model_file=str(seed_dir / "lgbm_model.txt")))

        self._bars_by_symbol: Dict[str, List[dict]] = {}
        self._index_by_symbol: Dict[str, Dict[date, int]] = {}
        for symbol, group_df in history.sort_values("date").groupby("symbol"):
            bars = group_df.to_dict(orient="records")
            self._bars_by_symbol[symbol] = bars
            self._index_by_symbol[symbol] = {
                pd.Timestamp(bar["date"]).date(): i for i, bar in enumerate(bars)
            }
        # 性能优化：`RegressionRotationStrategy` 每次调仓会对同一个 as_of
        # 日期把候选池里全部股票各查一遍，逐支单独构造1行DataFrame+调用
        # 5个模型的predict，30支股票就是150次predict调用。改成按天缓存：
        # 第一次查某天时，把当天有数据的全部股票一次性批量predict(每个
        # 模型1次，共5次)，之后同一天的其余股票直接查缓存，不重复计算。
        # 和 model_signal.py 的 `_cross_sectional_features_for_day` 是
        # 同一个按天缓存的思路。
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
            booster_predictions = [booster.predict(frame) for booster in self._boosters]
            averaged = sum(booster_predictions) / len(booster_predictions)
            result = {symbol: float(value) for symbol, value in zip(symbols, averaged)}

        self._predictions_by_day[as_of] = result
        return result

    def __call__(self, symbol: str, as_of: date) -> Optional[float]:
        return self._predict_all_for_day(as_of).get(symbol)


class EnsembleRegressionWalkForwardSignalSource:
    def __init__(self, model_root: Path, history_parquet: Path):
        history = pd.read_parquet(history_parquet)
        self._by_year: Dict[int, EnsembleRegressionSignalSource] = {}
        for year_dir in sorted(model_root.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue
            self._by_year[year] = EnsembleRegressionSignalSource(year_dir, history)

    def __call__(self, symbol: str, as_of: date) -> Optional[float]:
        source = self._by_year.get(as_of.year)
        if source is None:
            return None
        return source(symbol, as_of)
