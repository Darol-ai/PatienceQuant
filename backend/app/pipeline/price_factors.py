"""只用日线开高低收就能算的单个选股因子（ADR-0047 第 9 条：来自 QuantsPlaybook，见仓库 NOTICE）。

每个因子输入"截至 as_of 的开高低收宽表"（行=交易日，列=股票，前复权），
输出当天每支股票一个因子值。direction=-1 表示因子值越低越好。
打分时和其他因子一样换成股票池内的百分位（0～100，方向统一成越高越好）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd

Panel = Dict[str, pd.DataFrame]  # "open"/"high"/"low"/"close" → 宽表


def _zscore(values: pd.Series) -> pd.Series:
    clean = values.replace([np.inf, -np.inf], np.nan)
    std = clean.std()
    return (clean - clean.mean()) / std if std and np.isfinite(std) else clean * 0


def ubl(panel: Panel, window: int = 20) -> pd.Series:
    """上下影线综合因子 UBL（东吴证券《上下影线，蜡烛好还是威廉好？》，2020）。

    - 蜡烛上影线 = 最高 − max(开, 收)；威廉下影线 = 收 − 最低；
    - 各自除以前 5 天的均值做标准化；
    - 过去 20 天：蜡烛上影线的标准差、威廉下影线的均值；
    - 两者各做横截面标准化后相加。

    和研报的差异：研报先做市值中性化，市值数据还没接入本地行情库，这里跳过这一步。
    """
    high, low, open_, close = panel["high"], panel["low"], panel["open"], panel["close"]
    upper = high - np.maximum(close, open_)
    williams_lower = close - low
    std_upper = upper / upper.rolling(5).mean().shift(1)
    std_williams_lower = williams_lower / williams_lower.rolling(5).mean().shift(1)
    tail_upper = std_upper.replace([np.inf, -np.inf], np.nan).tail(window)
    tail_lower = std_williams_lower.replace([np.inf, -np.inf], np.nan).tail(window)
    upper_std = tail_upper.std().where(tail_upper.count() >= window // 2)
    lower_mean = tail_lower.mean().where(tail_lower.count() >= window // 2)
    return _zscore(upper_std) + _zscore(lower_mean)


def ideal_amplitude(panel: Panel, window: int = 20, lamb: float = 0.2) -> pd.Series:
    """理想振幅因子 V(λ)（开源证券《振幅因子的隐藏结构》，2020）。

    过去 20 天里，按每支股票自己的收盘价在这 20 天中的分位切成高价日（分位 ≥ λ）
    和低价日；V = 高价日平均振幅 − 低价日平均振幅，振幅 = 最高/最低 − 1。
    一字跌停的次日不计入（研报还剔除停牌次日；停牌日本地行情库本来就没有记录）。
    """
    high, low, close = panel["high"].tail(window + 1), panel["low"].tail(window + 1), panel["close"].tail(window + 1)
    limit_down_flat = (close / close.shift(1) - 1 < -0.09) & (high == low)
    usable = ~limit_down_flat.shift(1, fill_value=False)
    high, low, close, usable = high.tail(window), low.tail(window), close.tail(window), usable.tail(window)
    amplitude = high / low - 1
    rank = close.rank(pct=True)
    high_days = (rank >= lamb) & usable
    low_days = (rank < lamb) & usable
    v_high = (amplitude * high_days).sum() / window
    v_low = (amplitude * low_days).sum() / window
    enough = close.count() >= window
    return (v_high - v_low).where(enough)


@dataclass(frozen=True)
class PriceFactor:
    key: str
    label: str
    direction: int  # 1：越高越好；-1：越低越好
    lookback_days: int
    compute: Callable[[Panel], pd.Series]
    description: str


PRICE_FACTORS: Dict[str, PriceFactor] = {
    f.key: f
    for f in [
        PriceFactor("ubl", "上下影线（UBL）", -1, 30, ubl,
                    "蜡烛上影线波动 + 威廉下影线均值，越低越好（东吴证券 2020）"),
        PriceFactor("ideal_amplitude", "理想振幅", -1, 25, ideal_amplitude,
                    "高价日振幅 − 低价日振幅，越低越好（开源证券 2020）"),
    ]
}
