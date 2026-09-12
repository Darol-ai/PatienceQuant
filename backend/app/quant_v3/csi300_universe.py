"""对齐主流做法：universe从自选30支候选池换成沪深300全部真实成分股
（baostock `bs.query_hs300_stocks()`真实返回，300支，代码/名字保证
正确对应，不是Demo那种程序生成的虚构填充数据）。

`data/csi300_constituents.parquet`由`scripts/fetch_csi300_universe_history.py`
生成，这里只是转换成`RegressionRotationStrategy`/`MomentumRotationStrategy`
需要的 {symbol, name, group} 列表格式——不分子行业组，统一放"沪深300"
一个组，预算不分组直接给1.0（和broad_universe.py的"综合"单组思路一致）。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd

_CONSTITUENTS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "csi300_constituents.parquet"

CSI300_GROUP_BUDGETS: Dict[str, float] = {"沪深300": 1.0}


@lru_cache(maxsize=1)
def csi300_stocks() -> List[dict]:
    frame = pd.read_parquet(_CONSTITUENTS_FILE)
    return [
        {"symbol": row.symbol, "name": row.name, "group": "沪深300"}
        for row in frame.itertuples()
    ]
