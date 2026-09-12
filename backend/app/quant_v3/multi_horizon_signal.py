"""自由探索阶段（docs/adr/0030）：跨多个预测窗口(horizon)集成——每个
horizon各自是一组5种子集成(ensemble_regression_signal.py)，这里再对
多个horizon的预测取平均。和"同一horizon多个随机种子"是不同的多样性
来源（训练目标本身不同，不是采样噪声不同），可能抵消单一horizon对
特定年份的偏差。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional

from app.quant_v3.ensemble_regression_signal import EnsembleRegressionWalkForwardSignalSource


class MultiHorizonSignalSource:
    def __init__(self, model_roots: List[Path], history_parquet: Path):
        self._sources = [
            EnsembleRegressionWalkForwardSignalSource(root, history_parquet) for root in model_roots
        ]

    def __call__(self, symbol: str, as_of: date) -> Optional[float]:
        predictions = [
            value for source in self._sources if (value := source(symbol, as_of)) is not None
        ]
        if not predictions:
            return None
        return sum(predictions) / len(predictions)
