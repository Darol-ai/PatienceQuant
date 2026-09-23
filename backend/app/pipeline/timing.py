"""择时插槽：输入指数数据，输出 0～100% 的整体持仓比例（ADR-0047 第 2 条）。

决定买多少，不决定买哪些。只用 as_of 当天及之前的指数数据。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional

import pandas as pd

from app.data.service import MarketDataService
from app.pipeline.bars import warmup_start
from app.pipeline.spec import IndexTrendTimingSpec, NoTimingSpec


@dataclass
class TimingResult:
    exposure: float
    label: str
    note: Optional[str] = None


class TimingSignal:
    warmup_days: int = 0

    def prepare(self, start: date, end: date) -> None:
        pass

    def exposure(self, as_of: date) -> TimingResult:
        raise NotImplementedError


class NoTiming(TimingSignal):
    def exposure(self, as_of: date) -> TimingResult:
        return TimingResult(1.0, "full")


class IndexTrendTiming(TimingSignal):
    """沪深300 跌破 200 日均线 → risk_off_exposure；跌破 50 日均线 → 满仓和
    risk_off_exposure 的中间值；否则满仓。沿用多因子 V3 原有的趋势过滤规则。"""

    warmup_days = 200

    def __init__(self, spec: IndexTrendTimingSpec, data: MarketDataService):
        self.risk_off = spec.risk_off_exposure
        self.data = data
        self._close: Optional[pd.Series] = None

    def prepare(self, start: date, end: date) -> None:
        bench = self.data.benchmark(warmup_start(start, self.warmup_days), end)
        self._close = pd.Series(bench["adj_close"].to_numpy(dtype=float), index=pd.to_datetime(bench["trade_date"])).sort_index()

    def exposure(self, as_of: date) -> TimingResult:
        if self._close is None:
            self.prepare(as_of, as_of)
        close = self._close[self._close.index <= pd.Timestamp(as_of)]
        if len(close) < 200:
            return TimingResult(1.0, "early_sample", "指数历史不足 200 个交易日，择时不生效，满仓")
        last, ma50, ma200 = float(close.iloc[-1]), float(close.tail(50).mean()), float(close.tail(200).mean())
        if last < ma200:
            return TimingResult(self.risk_off, "risk_off", "沪深300低于200日均线，仓位降至 %.0f%%" % (self.risk_off * 100))
        if last < ma50:
            mid = (1.0 + self.risk_off) / 2
            return TimingResult(mid, "caution", "沪深300低于50日均线，仓位降至 %.0f%%" % (mid * 100))
        return TimingResult(1.0, "risk_on")


def build_timing(spec, data: MarketDataService) -> TimingSignal:
    if isinstance(spec, NoTimingSpec):
        return NoTiming()
    if isinstance(spec, IndexTrendTimingSpec):
        return IndexTrendTiming(spec, data)
    raise ValueError(f"未知的择时信号：{spec}")
