"""把本地行情库的日线整理成模型特征计算用的逐股日线序列。

迁移前模型读的是离线脚本从 baostock 抓的 parquet，口径和 tushare 有三处
不同，这里统一换成旧口径，保证旧模型看到的特征和训练时一致：

- 成交量：tushare 单位是"手"，换成"股"（×100）；
- 成交额：tushare 单位是"千元"，换成"元"（×1000）；
- 停牌日：tushare 当天没有记录，旧数据里是一行"价格沿用前收、成交量 0、
  is_suspended=True"。这里按交易日历补上，否则"60 日有效交易日占比"永远
  是 100%，而且各种 N 日窗口会跨过停牌期、比真实的交易日数更长。
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Iterable, List

import pandas as pd

from app.data.market_store import AShareMarketStore
from app.data.tushare_provider import _to_ts_code


def load_bars(store: AShareMarketStore, symbols: Iterable[str], start: date, end: date) -> Dict[str, List[dict]]:
    """返回 {不带后缀的代码: [按日期升序的日线 dict]}，价格前复权。

    start 之前不额外多取历史——调用方自己把特征需要的预热期算进 start。
    """
    symbols = list(dict.fromkeys(symbols))
    code_to_symbol = {_to_ts_code(symbol): symbol for symbol in symbols}
    daily = store.load_daily(start, end, list(code_to_symbol), adjust="qfq")
    if daily.empty:
        return {}
    calendar = pd.DatetimeIndex(sorted(d for d in store.stored_days() if start <= d <= end))
    bars: Dict[str, List[dict]] = {}
    for code, frame in daily.groupby("ts_code", sort=False):
        frame = frame.set_index("trade_date")
        # 从这支股票在区间内第一次出现开始对齐交易日历（之前是还没上市或
        # 区间没取到，不是停牌），之后缺的日子都是停牌。
        days = calendar[(calendar >= frame.index.min()) & (calendar <= frame.index.max())]
        frame = frame.reindex(days)
        suspended = frame["close"].isna()
        frame["close"] = frame["close"].ffill()
        for column in ("open", "high", "low"):
            frame[column] = frame[column].fillna(frame["close"])
        frame["vol"] = frame["vol"].fillna(0.0)
        frame["amount"] = frame["amount"].fillna(0.0)
        out = pd.DataFrame({
            "date": frame.index.date,
            "open": frame["open"].to_numpy(dtype=float),
            "high": frame["high"].to_numpy(dtype=float),
            "low": frame["low"].to_numpy(dtype=float),
            "close": frame["close"].to_numpy(dtype=float),
            "volume": frame["vol"].to_numpy(dtype=float) * 100,
            "amount": frame["amount"].to_numpy(dtype=float) * 1000,
            "is_suspended": suspended.to_numpy(),
        })
        bars[code_to_symbol[code]] = out.to_dict(orient="records")
    return bars


def warmup_start(start: date, trading_days: int) -> date:
    """往前推够 trading_days 个交易日的起始日期（按一年约 242 个交易日粗估，多留余量）。"""
    return start - timedelta(days=int(trading_days * 1.6) + 30)
