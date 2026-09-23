"""指数择时信号：输入沪深300指数日线（开高低收），输出每天的持仓状态（1 满仓 / 0 空仓）。

来自 QuantsPlaybook（见仓库 NOTICE），原代码从聚宽/本地文件取数、用 talib 和
vectorbt，这里改成只依赖 pandas/numpy/scipy 的纯函数。每一天的状态只用当天及
之前的数据；T 日收盘算出的状态由回测引擎在 T+1 日执行。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _hysteresis(value: pd.Series, upper: float, lower: float) -> pd.Series:
    """高于 upper 开仓；持仓中只要高于 lower 就继续持有；否则空仓。"""
    state = np.zeros(len(value), dtype=float)
    holding = False
    for i, v in enumerate(value.to_numpy()):
        if np.isnan(v):
            holding = False
        elif v > upper:
            holding = True
        elif not (holding and v > lower):
            holding = False
        state[i] = 1.0 if holding else 0.0
    return pd.Series(state, index=value.index)


def rsrs_zscore(ohlc: pd.DataFrame, n: int = 18, m: int = 600) -> pd.Series:
    """RSRS 标准分（光大证券《基于阻力支撑相对强度（RSRS）的市场择时》，2017）。

    斜率 = 最近 n 天最高价对最低价的 OLS 斜率；标准分 = 斜率减去最近 m 天斜率
    均值、除以标准差。m 天斜率不够时为 NaN。
    """
    high, low = ohlc["high"].astype(float), ohlc["low"].astype(float)
    beta = high.rolling(n).cov(low) / low.rolling(n).var()
    return (beta - beta.rolling(m).mean()) / beta.rolling(m).std()


def rsrs_state(ohlc: pd.DataFrame, n: int = 18, m: int = 600, threshold: float = 0.7) -> pd.Series:
    """标准分高于 0.7 买入，持有到跌破 −0.7（研报默认参数 N=18、M=600、S=0.7）。"""
    return _hysteresis(rsrs_zscore(ohlc, n, m), threshold, -threshold)


def icu_ma(close: pd.Series, n: int = 5) -> pd.Series:
    """ICU 均线（中泰证券《"均线"才是绝对收益利器》，2023）：最近 n 天收盘价做
    重复中位数稳健回归（Siegel 1982），取回归线在当天的值。"""
    from scipy import stats

    def last_fit(values: np.ndarray) -> float:
        res = stats.siegelslopes(values, np.arange(len(values)), method="hierarchical")
        return res.intercept + res.slope * (len(values) - 1)

    return close.astype(float).rolling(n).apply(last_fit, raw=True)


def icu_ma_state(ohlc: pd.DataFrame, n: int = 5) -> pd.Series:
    """收盘价上穿 ICU 均线买入、下穿卖出，即收盘价在均线之上就持有（研报默认 n=5）。"""
    close = ohlc["close"].astype(float)
    line = icu_ma(close, n)
    return (close > line).astype(float).where(line.notna(), 0.0)


def _alignment_events(lines: np.ndarray) -> np.ndarray:
    """lines 每行是 [下颚, 牙齿, 上唇]。多头排列（下颚<牙齿<上唇）刚形成那天记 1，
    空头排列刚形成那天记 −1，其余 NaN。"""
    diffs = np.diff(lines, axis=1)
    bull = np.all(diffs > 0, axis=1)
    bear = np.all(diffs < 0, axis=1)
    out = np.full(len(lines), np.nan)
    out[1:][bull[1:] & ~bull[:-1]] = 1
    out[1:][bear[1:] & ~bear[:-1]] = -1
    return out


def alligator_state(ohlc: pd.DataFrame) -> pd.Series:
    """鳄鱼线组合择时（招商证券《基于鳄鱼线的指数择时及轮动策略》，2024），按研报的合并方法：

    - 看多：鳄鱼线=1 且（分形=1 或 MACD=1）→ 满仓；
    - 看空：鳄鱼线、AO、分形、MACD 任一 = −1 → 空仓（同一天同时出现时以空仓为准）；
    - 其他：维持前一天。

    研报里的北向资金信号需要额外数据，不包含。
    """
    close, high, low = (ohlc[c].astype(float) for c in ("close", "high", "low"))
    idx = close.index

    # 鳄鱼线：下颚 SMA13 后移 8 天、牙齿 SMA8 后移 5 天、上唇 SMA5 后移 3 天
    lines = np.column_stack([close.rolling(p).mean().shift(lag).to_numpy() for p, lag in ((13, 8), (8, 5), (5, 3))])
    alligator = pd.Series(_alignment_events(lines), index=idx).ffill().fillna(0)

    # AO（研报口径：(最高−最低)/2 的 5 日均值 − 34 日均值），连续 3 天上行=1、下行=−1
    mid = (high - low) * 0.5
    ao = mid.rolling(5).mean() - mid.rolling(34).mean()
    step = ao.diff()
    ao_signal = pd.Series(np.where((step > 0) & (step.shift(1) > 0), 1.0,
                                   np.where((step < 0) & (step.shift(1) < 0), -1.0, np.nan)), index=idx)
    ao_signal = ao_signal.ffill().fillna(0)

    # 分形：前一天是顶分形且收盘高于 3 天前 → 1；前一天是底分形且收盘低于 3 天前 → −1（当天事件，不延续）
    top = (high.shift(2) < high.shift(1)) & (high.shift(1) > high) & (low.shift(2) < low.shift(1)) & (low.shift(1) > low)
    bottom = (low.shift(2) > low.shift(1)) & (low.shift(1) < low) & (high.shift(2) > high.shift(1)) & (high.shift(1) < high)
    fractal = top.astype(int) - bottom.astype(int)
    fractal_prev = fractal.shift(1)
    fractal_signal = ((close > close.shift(3)) & (fractal_prev == 1)).astype(float) \
        - ((close < close.shift(3)) & (fractal_prev == -1)).astype(float)

    # MACD(12,26,9)：零轴上方金叉且柱子翻红=1，零轴下方死叉且柱子翻绿=−1，之后延续
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    bullish = (dif > dea) & (dif.shift(1) < dea.shift(1)) & (hist > 0) & (hist.shift(1) < 0) & (dif >= 0) & (dea >= 0)
    bearish = (dif < dea) & (dif.shift(1) > dea.shift(1)) & (hist < 0) & (hist.shift(1) > 0) & (dif < 0) & (dea < 0)
    macd = (bullish.astype(float) - bearish.astype(float)).replace(0, np.nan).ffill().fillna(0)

    entries = (alligator == 1) & ((fractal_signal == 1) | (macd == 1))
    exits = (alligator == -1) | (ao_signal == -1) | (fractal_signal == -1) | (macd == -1)
    state = np.zeros(len(idx))
    holding = False
    for i in range(len(idx)):
        if exits.iat[i]:
            holding = False
        elif entries.iat[i]:
            holding = True
        state[i] = 1.0 if holding else 0.0
    return pd.Series(state, index=idx)
