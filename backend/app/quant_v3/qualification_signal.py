"""包装 `is_liquidity_qualified` 成 `(symbol, as_of) -> bool` 的信号源，
给策略在每次调仓时判断某支股票当天是否满足资格条件（V3 方案 2.2 节的
流动性子集）。历史不足/股票或日期不在历史范围内时判定不合格——不知道
就不能记为合格。
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List

import pandas as pd

from app.quant_v3.qualification import is_liquidity_qualified


class QualificationSignalSource:
    def __init__(self, history: pd.DataFrame):
        self._bars_by_symbol: Dict[str, List[dict]] = {}
        self._index_by_symbol: Dict[str, Dict[date, int]] = {}
        for symbol, group_df in history.sort_values("date").groupby("symbol"):
            bars = group_df.to_dict(orient="records")
            self._bars_by_symbol[symbol] = bars
            self._index_by_symbol[symbol] = {
                pd.Timestamp(bar["date"]).date(): i for i, bar in enumerate(bars)
            }

    def __call__(self, symbol: str, as_of: date) -> bool:
        bars = self._bars_by_symbol.get(symbol)
        if bars is None:
            return False
        t_index = self._index_by_symbol[symbol].get(as_of)
        if t_index is None:
            return False
        return is_liquidity_qualified(bars, t_index)
