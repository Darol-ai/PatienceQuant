"""V3 方案 3.5 节：首版 15 项特征里，价格/量能相关的那几个。

配置组是类别特征，装配训练样本时直接用 a_phase_universe 里的 group 字段，
不需要单独的计算函数；有效交易日比例、60日日均成交额留到装配训练集时再算
（跨多只股票、多个窗口对齐，放这里的纯函数意义不大）。
"""
from collections.abc import Sequence

from app.quant_v3.labels import log_returns_over_window, sample_volatility


def return_over_window(prices: Sequence[float], window: int) -> float | None:
    """window 日收益：prices[-1]/prices[-1-window]-1。价格不足则返回 None。"""
    if len(prices) < window + 1:
        return None
    return prices[-1] / prices[-1 - window] - 1


def close_vs_ma_deviation(prices: Sequence[float], window: int) -> float | None:
    """收盘相对 MA(window) 的偏离，MA 含当天。价格不足则返回 None。"""
    if len(prices) < window:
        return None
    recent = prices[-window:]
    ma = sum(recent) / window
    return prices[-1] / ma - 1


def max_drawdown_over(prices: Sequence[float], window: int) -> float | None:
    """window 日内相对滚动峰值的最大回撤（非正数）。价格不足则返回 None。"""
    if len(prices) < window:
        return None
    recent = prices[-window:]
    peak = recent[0]
    max_dd = 0.0
    for price in recent:
        peak = max(peak, price)
        max_dd = min(max_dd, price / peak - 1)
    return max_dd


def volume_ratio(volumes: Sequence[float], window: int) -> float | None:
    """最新成交量 / 最近 window 日均量（含当天）。成交量不足则返回 None。"""
    if len(volumes) < window:
        return None
    recent = volumes[-window:]
    average = sum(recent) / window
    if average == 0:
        return None
    return volumes[-1] / average


def atr_ratio(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], window: int
) -> float | None:
    """ATR(window) / 最新收盘价。真实波幅 TR = max(高-低, |高-昨收|, |低-昨收|)。
    高低收三个序列必须同尺度（同为前复权），需要 window+1 天数据才能算出
    window 个 TR。数据不足则返回 None。
    """
    if len(highs) < window + 1 or len(lows) < window + 1 or len(closes) < window + 1:
        return None
    true_ranges = []
    for i in range(-window, 0):
        high, low, prev_close = highs[i], lows[i], closes[i - 1]
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    atr = sum(true_ranges) / window
    return atr / closes[-1]


def volatility_over(prices: Sequence[float], window: int) -> float | None:
    """window 日对数收益的样本标准差（ddof=1），复用 labels.py 已测过的
    窗口提取 + 样本标准差。价格不足 window+1 个则返回 None。
    """
    returns = log_returns_over_window(prices, window)
    if returns is None:
        return None
    return sample_volatility(returns)


def average_amount(amounts: Sequence[float], window: int) -> float | None:
    """window 日成交额的简单均值。数据不足则返回 None。"""
    if len(amounts) < window:
        return None
    recent = amounts[-window:]
    return sum(recent) / window


def valid_trading_ratio(is_suspended_flags: Sequence[bool], window: int) -> float | None:
    """window 日内非停牌天数占比。数据不足则返回 None。"""
    if len(is_suspended_flags) < window:
        return None
    recent = is_suspended_flags[-window:]
    valid_days = sum(1 for suspended in recent if not suspended)
    return valid_days / window


def average_turnover(turnover_rates: Sequence[float], window: int) -> float | None:
    """window 日换手率均值——资金关注度/流动性信号，和价格/收益类特征是
    不同的信息来源（见 docs/adr/0011"特征太窄"这条限制）。数据不足则
    返回 None。"""
    if len(turnover_rates) < window:
        return None
    recent = turnover_rates[-window:]
    return sum(recent) / window


def valuation_percentile(pb_values: Sequence[float], window: int) -> float | None:
    """当前(序列最后一个) PB 在自己过去 window 天历史里的分位（0~1，用
    "严格小于"和"小于等于"两种计数的均值，中位数正好落在 0.5，不受并列
    值影响）。跨行业 PB 绝对水平不可比（银行 PB~0.5，消费股 PB~10+），
    只能看"相对自己历史贵还是便宜"，不是横截面排名。数据不足则返回 None。
    """
    if len(pb_values) < window:
        return None
    recent = pb_values[-window:]
    current = recent[-1]
    less = sum(1 for v in recent if v < current)
    equal = sum(1 for v in recent if v == current)
    return (less + equal / 2) / len(recent)
