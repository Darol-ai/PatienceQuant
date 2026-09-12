"""A 阶段固定 10 支股票的数据服务：只实现 BacktestEngine.run() 需要的四个
方法，数据直接来自本地 parquet（data/a_phase_history.parquet），不经过
SQLite/MarketDataService——A 阶段不需要那一整套目录/搜索能力。

从 scripts/run_v3_backtest.py 挪过来，给"研报生成因子回测"这类新功能复用，
避免每处都内联重写一遍。
"""
from __future__ import annotations

import pandas as pd


class APhaseDataService:
    def __init__(self, history: pd.DataFrame):
        self._history = history

    def stocks(self) -> pd.DataFrame:
        return pd.DataFrame({"symbol": self._history["symbol"].unique()})

    def prices(self, symbols, start, end, allow_network=False) -> pd.DataFrame:
        frame = self._history[self._history.symbol.isin(symbols)].copy()
        frame["trade_date"] = pd.to_datetime(frame["date"]).dt.date
        frame["adj_close"] = frame["close"]
        return frame[(frame.trade_date >= start) & (frame.trade_date <= end)][
            ["trade_date", "symbol", "adj_close"]
        ]

    def fundamentals(self, symbols, as_of) -> pd.DataFrame:
        return pd.DataFrame()

    def benchmark(self, start, end) -> pd.DataFrame:
        # A 阶段没有单独接指数数据，用等权（等资金）买入持有当基准参照，
        # 只用于图表对比，不参与策略计算。
        #
        # 自由探索阶段发现的真实bug（docs/adr/0019）：原实现直接对收盘价
        # 取算术平均，价格量级差异巨大的股票池（比如贵州茅台¥500+ vs
        # 其他多数¥2~30）会让高价股在"基准"里被隐性赋予远超其他股票的
        # 权重——不是"每支股票投入等量资金买入持有"的真实等权重基准，是
        # 被最贵的几支股票主导的假基准。正确做法：每支股票先按各自区间
        # 起始价格归一化(除以T0收盘价)，再取均值——这样每支股票不论绝对
        # 价格高低，对基准的贡献都是"等量资金买入后的涨跌幅"。
        frame = self._history.copy()
        frame["trade_date"] = pd.to_datetime(frame["date"]).dt.date
        frame = frame[(frame.trade_date >= start) & (frame.trade_date <= end)]
        wide = frame.pivot_table(index="trade_date", columns="symbol", values="close").ffill()
        normalized = wide / wide.bfill().iloc[0]
        mean_index = normalized.mean(axis=1)
        return pd.DataFrame({"trade_date": mean_index.index, "adj_close": mean_index.to_numpy()})

    def frame_data_mode(self, frame) -> str:
        return "real"
