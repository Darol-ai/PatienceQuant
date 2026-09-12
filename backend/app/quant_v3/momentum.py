"""公开策略调研新方向（见 docs/adr/0010）：不再用 LightGBM 分类模型预测
涨跌概率，换成学术界最经典、最常被公开策略引用的动量因子——
Jegadeesh-Titman 的"12-1 动量"：过去 12 个月(lookback)的收益，跳过最近
1 个月(skip)，避免短期反转噪声污染趋势信号。
"""
from collections.abc import Sequence


def trailing_momentum(prices: Sequence[float], lookback: int = 252, skip: int = 21) -> float | None:
    """skip 天前收盘 相对 skip+lookback 天前收盘 的收益。价格不足则返回 None。"""
    if len(prices) < lookback + skip + 1:
        return None
    recent_price = prices[-(skip + 1)]
    past_price = prices[-(skip + 1 + lookback)]
    return recent_price / past_price - 1
