"""V3 方案 3.2 节：波动率缩放的标签事件边界。"""

import math
import statistics
from collections.abc import Sequence


def label_boundaries(v_t: float) -> tuple[float, float]:
    """给定 20 日尺度波动率 v_t，返回 (upper_t, lower_t) 两条事件边界。

    upper_t = exp(1.6 * v_t) - 1
    lower_t = exp(-v_t) - 1
    """
    upper_t = math.exp(1.6 * v_t) - 1
    lower_t = math.exp(-v_t) - 1
    return upper_t, lower_t


def sample_volatility(log_returns: Sequence[float]) -> float:
    """对数收益率的样本标准差（ddof=1）。"""
    return statistics.stdev(log_returns)


def scale_to_20_day_horizon(sigma_t: float) -> float:
    """v_t = sigma_t * sqrt(20)：把日频波动率缩放到 20 日尺度。"""
    return sigma_t * math.sqrt(20)


def log_returns_over_window(prices: Sequence[float], window: int) -> list[float] | None:
    """取最近 window+1 个有效价格，算出 window 个相邻对数收益。

    价格不足 window+1 个时返回 None——不足就不出样本，不能用现有数据凑数
    （V3 方案 3.2 节）。
    """
    if len(prices) < window + 1:
        return None

    recent = prices[-(window + 1):]
    return [math.log(recent[i] / recent[i - 1]) for i in range(1, len(recent))]


def compute_sigma_t(prices: Sequence[float], window: int = 60) -> float | None:
    """从价格序列算 sigma_t：取最近 window 个对数收益的样本标准差。

    价格不足、标准差为零或非有限值时返回 None——不制作该样本，不强行赋值
    （V3 方案 3.2 节）。
    """
    returns = log_returns_over_window(prices, window)
    if returns is None:
        return None

    sigma = sample_volatility(returns)
    if sigma == 0 or not math.isfinite(sigma):
        return None

    return sigma


def first_touch_label(
    closes: Sequence[float], o_star: float, upper_t: float, lower_t: float
) -> str:
    """按收盘价逐日检查，谁先触及边界谁定标签；全程未触及记 NEUTRAL。

    只用收盘确认，不用日线高低点推断盘中先后（V3 方案 3.2 节）。
    """
    for close in closes:
        change = close / o_star - 1
        if change >= upper_t:
            return "UP"
        if change <= lower_t:
            return "DOWN"
    return "NEUTRAL"
