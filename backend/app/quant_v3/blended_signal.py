"""调研佐证（Qlib CSI300公开基准：DoubleEnsemble——用LGBM做base model
再加权重组——是所有基线里表现最强的一档），第三个候选策略换成"LightGBM+
XGBoost两个独立训练的模型，预测分数取平均"这个更简单版本的模型集成
思路，而不是照搬DoubleEnsemble完整算法（那是一整套独立的重加权训练
流程，这里只是取用"集成多个树模型的预测"这个已经被验证有效的核心
思路）。两个子信号source都缺失/都没有分数时诚实返回None，不编数字。
"""
from __future__ import annotations

from datetime import date
from typing import Callable, List, Optional

Signal = Callable[[str, date], Optional[float]]


class BlendedSignalSource:
    def __init__(self, sources: List[Signal]):
        self._sources = sources

    def __call__(self, symbol: str, as_of: date) -> Optional[float]:
        scores = [value for source in self._sources if (value := source(symbol, as_of)) is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)
