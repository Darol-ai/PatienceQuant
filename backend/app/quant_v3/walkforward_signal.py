"""按 V3 方案 7.2 节的逐年滚动模型包装成 signal_source：某一天用哪个模型，
取决于那天所在的年份对应哪一折训练产物（见 scripts/train_v3_walkforward.py）。

没有对应年份的模型时返回中性占位信号，不外推用别的年份的模型硬凑——那样就
不是滚动训练了，是在假装有覆盖。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from app.quant_v3.model_signal import NEUTRAL, TrainedSignalSource


class WalkForwardSignalSource:
    def __init__(self, model_root: Path, history_parquet: Path):
        history = pd.read_parquet(history_parquet)
        self._by_year: Dict[int, TrainedSignalSource] = {}
        for year_dir in sorted(model_root.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue
            self._by_year[year] = TrainedSignalSource(year_dir, history)

    def __call__(self, symbol: str, as_of: date) -> Tuple[float, float]:
        source = self._by_year.get(as_of.year)
        if source is None:
            return NEUTRAL
        return source(symbol, as_of)

    def threshold_for(self, year: int) -> Tuple[float, float]:
        """给回测脚本读某一年折对应的入场门槛用。"""
        source = self._by_year[year]
        return source.metadata["p_up_threshold"], source.metadata["p_down_threshold"]
