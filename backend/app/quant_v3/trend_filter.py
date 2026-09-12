"""自由探索阶段（docs/adr/0013）：大盘趋势过滤，参考 PatienceQuant 自带
`MultiFactorStrategy` 已经在用的思路（沪深300 跌破200日均线就降仓位）。
这里不接外部指数，用股票池自己的等权合成指数相对自身200日均线的位置
判断"市场是不是在趋势向下"——ADR-0010 已经验证 Dual Momentum 的绝对
动量过滤在真正下跌年份（2022）确实能起到保护作用，这是同一个思路的
更简单版本：不是对每支股票单独判断，而是对整体仓位暴露做一次性调节。

历史不足 ma_window 天时判断不了趋势，默认给满仓（不能不知道就假装知道，
凭空进入防御状态）。
"""
from collections.abc import Sequence
from datetime import date
from typing import Dict, List, Tuple


def market_exposure_multiplier(
    index_prices: Sequence[float], ma_window: int = 200, risk_off_exposure: float = 0.3
) -> float:
    if len(index_prices) < ma_window:
        return 1.0
    recent = index_prices[-ma_window:]
    ma = sum(recent) / ma_window
    return 1.0 if recent[-1] >= ma else risk_off_exposure


class TrendSignal:
    """把 (date, price) 的时间序列包装成 `exposure(as_of) -> float`，供策略
    在每次调仓时查询当天该给多少整体仓位暴露。日期不在序列里（比如还没
    到当天的数据）时给满仓——不知道就不假装知道要防御。"""

    def __init__(self, series: List[Tuple[date, float]], ma_window: int = 200, risk_off_exposure: float = 0.3):
        ordered = sorted(series, key=lambda item: item[0])
        self._dates: List[date] = [d for d, _ in ordered]
        self._prices: List[float] = [p for _, p in ordered]
        self._index_by_date: Dict[date, int] = {d: i for i, d in enumerate(self._dates)}
        self.ma_window = ma_window
        self.risk_off_exposure = risk_off_exposure

    def exposure(self, as_of: date) -> float:
        idx = self._index_by_date.get(as_of)
        if idx is None:
            return 1.0
        return market_exposure_multiplier(
            self._prices[: idx + 1], ma_window=self.ma_window, risk_off_exposure=self.risk_off_exposure
        )
