from __future__ import annotations

from datetime import date
from typing import List, Protocol, Tuple

import pandas as pd


class MarketDataProvider(Protocol):
    mode: str

    def bootstrap(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ...

    def fetch_prices(self, symbols: List[str], start: date, end: date) -> pd.DataFrame: ...

    def fetch_benchmark(self, start: date, end: date) -> pd.DataFrame: ...
