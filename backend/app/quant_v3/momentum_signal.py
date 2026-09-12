"""包装 `trailing_momentum` 成 `(symbol, as_of) -> float | None` 的信号源，
给 MomentumRotationStrategy 用（见 docs/adr/0010）。历史不足/股票或日期
不在历史范围内时返回 None，不伪造分数。
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from app.quant_v3.momentum import trailing_momentum


class MomentumSignalSource:
    def __init__(self, history: pd.DataFrame, lookback: int = 252, skip: int = 21):
        self.lookback = lookback
        self.skip = skip
        self._closes_by_symbol: Dict[str, List[float]] = {}
        self._index_by_symbol: Dict[str, Dict[date, int]] = {}
        for symbol, group_df in history.sort_values("date").groupby("symbol"):
            closes = group_df["close"].tolist()
            self._closes_by_symbol[symbol] = closes
            self._index_by_symbol[symbol] = {
                pd.Timestamp(d).date(): i for i, d in enumerate(group_df["date"])
            }

    def __call__(self, symbol: str, as_of: date) -> Optional[float]:
        closes = self._closes_by_symbol.get(symbol)
        if closes is None:
            return None
        t_index = self._index_by_symbol[symbol].get(as_of)
        if t_index is None:
            return None
        return trailing_momentum(closes[: t_index + 1], lookback=self.lookback, skip=self.skip)
