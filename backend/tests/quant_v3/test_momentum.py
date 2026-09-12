import pytest

from app.quant_v3.momentum import trailing_momentum


def test_trailing_momentum_is_return_from_lookback_ago_to_skip_ago():
    """经典 12-1 动量（Jegadeesh-Titman）：跳过最近 skip 天（避免短期反转
    噪声），看"skip 天前"相对"skip+lookback 天前"的收益。用等差价格序列
    直接按下标核对，不靠脑补切片。"""
    prices = [float(i) for i in range(300)]  # prices[i] == i，价格=下标
    skip, lookback = 21, 252
    past_price = prices[-(skip + 1 + lookback)]  # 距今 skip+lookback 天前
    recent_price = prices[-(skip + 1)]  # 距今 skip 天前

    result = trailing_momentum(prices, lookback=lookback, skip=skip)

    assert result == pytest.approx(recent_price / past_price - 1)


def test_trailing_momentum_returns_none_when_not_enough_history():
    prices = [100.0] * 100
    assert trailing_momentum(prices, lookback=252, skip=21) is None
