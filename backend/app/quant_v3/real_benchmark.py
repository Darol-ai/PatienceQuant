"""真实沪深300指数收益对照——`APhaseDataService.benchmark()`那条"沪深300"
线其实是候选池自己的等权买入持有曲线（用来隔离出"策略选股/择时"的独立
增量，见 docs/adr/0037 附近的说明），从来不是真的查过官方指数点位。这个
命名容易让人以为策略真的在对比沪深300指数，这里补一个真正查真实指数
（baostock `sh.000300`）算出来的收益率，作为额外的、独立的对照，不是
替换那条等权线——两者回答的是不同问题：等权线回答"选股有没有增量"，
这条真实指数线回答"整体是不是真的跑赢了大盘"。

`data/real_csi300_index.parquet` 是 `scripts/fetch_real_universe_fundamentals.py`
调查真实数据覆盖范围问题时顺带确认可行、又单独重新抓的一份干净数据——
指数本身没有"股票代码对应错公司"这类问题，跟那次批量抓取里发现的
数据质量问题无关。
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

_INDEX_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "real_csi300_index.parquet"


@lru_cache(maxsize=1)
def _load_index() -> pd.DataFrame:
    frame = pd.read_parquet(_INDEX_FILE)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame.sort_values("date")


def csi300_return(start: date, end: date) -> Optional[float]:
    """真实沪深300指数在[start, end]区间的收益率。

    取区间内实际有交易记录的第一个和最后一个收盘点位——不假设 start/end
    本身是交易日。区间内没有任何数据（比如整段都在指数历史范围之外）
    时返回 None，调用方需要能接受"这次没有真实对照"，不能编一个数字。
    """
    frame = _load_index()
    window = frame[(frame.date >= start) & (frame.date <= end)]
    if len(window) < 2:
        return None
    return float(window.close.iloc[-1] / window.close.iloc[0] - 1)
