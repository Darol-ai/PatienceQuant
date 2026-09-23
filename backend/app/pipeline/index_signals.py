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


# ---- ADR-0052 第三步：第二批择时信号（都只用沪深300 指数日线）----

def _wma(values: pd.Series, n: int) -> pd.Series:
    """线性加权均线（同 talib.WMA）：最近一天权重 n，最早一天权重 1。"""
    weights = np.arange(1, n + 1, dtype=float)
    return values.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _hma(values: pd.Series, n: int) -> pd.Series:
    """Hull 均线：WMA(2·WMA(n/2) − WMA(n), √n)。"""
    return _wma(2 * _wma(values, int(n / 2)) - _wma(values, n), int(np.sqrt(n)))


def _cross_state(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """entry 当天开仓、exit 当天平仓（同一天都出现时以平仓为准），其余日子维持前一天。"""
    state = np.zeros(len(entry))
    holding = False
    for i, (buy, sell) in enumerate(zip(entry.to_numpy(), exit_.to_numpy())):
        if sell:
            holding = False
        elif buy:
            holding = True
        state[i] = 1.0 if holding else 0.0
    return pd.Series(state, index=entry.index)


def llt(close: pd.Series, alpha: float) -> pd.Series:
    """低延迟趋势线 LLT（广发证券《低延迟趋势线与交易择时》，2017）：二阶滤波，前两天取收盘价。"""
    price = close.astype(float).to_numpy()
    out = np.full(len(price), np.nan)
    if len(price) >= 2:
        out[:2] = price[:2]
    a = alpha
    for i in range(2, len(price)):
        out[i] = ((a - a * a / 4) * price[i] + (a * a / 2) * price[i - 1] - (a - 3 * a * a / 4) * price[i - 2]
                  + 2 * (1 - a) * out[i - 1] - (1 - a) ** 2 * out[i - 2])
    return pd.Series(out, index=close.index)


def llt_state(ohlc: pd.DataFrame, d: int = 30) -> pd.Series:
    """切线法：LLT 当天比前一天高就持有（α = 2/(d+1)，研报默认 d=30）。"""
    line = llt(ohlc["close"], 2 / (d + 1))
    return (line.diff() > 0).astype(float)


def ma_channel_state(ohlc: pd.DataFrame, short: int = 9, long: int = 18, n: int = 3) -> pd.Series:
    """均线交叉结合通道突破（申万宏源《均线交叉结合通道突破择时研究》，2018）：

    最近 n 天内出现过短均线上穿长均线、且收盘价创 n 日新高 → 开仓；
    最近 n 天内出现过死叉 → 平仓（原代码的平仓条件"收盘 ≥ n 日最低"几乎恒成立，照原样保留）。
    研报默认 SMA 9/18、n=3。
    """
    close = ohlc["close"].astype(float)
    s_ma, l_ma = close.rolling(short).mean(), close.rolling(long).mean()
    golden = (s_ma > l_ma) & (s_ma.shift(1) < l_ma.shift(1))
    dead = (s_ma < l_ma) & (s_ma.shift(1) > l_ma.shift(1))
    entry = (close >= close.rolling(n).max()) & (golden.astype(int).rolling(n).sum() > 0)
    exit_ = (close >= close.rolling(n).min()) & (dead.astype(int).rolling(n).sum() > 0)
    return _cross_state(entry, exit_)


def one_way_vol_state(ohlc: pd.DataFrame, window: int = 60) -> pd.Series:
    """单向波动差（国信证券《市场波动率研究：基于相对强弱下单向波动差值应用》，2015，策略一）：
    (最高/开盘−1) − (1−最低/开盘) 的 window 日均值大于 0 就持有（研报 60 日）。"""
    open_ = ohlc["open"].astype(float)
    diff = (ohlc["high"] / open_ - 1) - (1 - ohlc["low"] / open_)
    mean = diff.rolling(window).mean()
    return (mean > 0).astype(float).where(mean.notna(), 0.0)


def rps_vol_state(ohlc: pd.DataFrame, period: int = 13) -> pd.Series:
    """相对强弱 RPS 下的单向波动差（同一研报结合 RPS 的版本）：RPS = 收盘在过去 250 天高低区间的位置，
    取 period 日均值；上涨日记 +RPS、下跌日记 −RPS，再取 period 日均值，大于 0 就持有。
    period=13 是原 notebook 在 2006–2016 年样本内搜出来收益最高的参数。"""
    close = ohlc["close"].astype(float)
    low, high = close.rolling(250, min_periods=1).min(), close.rolling(250, min_periods=1).max()
    rps = ((close - low) / (high - low)).rolling(period, min_periods=1).mean()
    ret = close.pct_change()
    signed = rps.where(ret > 0, -rps)
    diff = signed.rolling(period, min_periods=1).mean()
    return (diff > 0).astype(float).where(ret.notna(), 0.0)


def high_moment_state(ohlc: pd.DataFrame, order: int = 5, window: int = 20, ema: int = 90) -> pd.Series:
    """指数高阶矩择时（广发证券《交易性择时策略研究之八：指数高阶矩择时策略》，2015，第一种方法）：近 20 天日收益的 5 阶原点矩，
    取 90 日 EMA（α=2/91）；EMA 上升时持有，当天指数跌超 10% 则次日空仓（研报的止损条件）。
    研报后续的自适应 α、多空双向版本需要做空，不包含。"""
    close = ohlc["close"].astype(float)
    ret = close.pct_change()
    moment = ret.rolling(window).apply(lambda x: np.mean(x ** order), raw=True)
    smooth = moment.ewm(alpha=2 / (ema + 1), adjust=False).mean().where(moment.rolling(ema).count() >= ema)
    rising = smooth > smooth.shift(1)
    return (rising & (ret >= -0.1)).astype(float)


def volume_resonance_state(ohlc: pd.DataFrame, bma: int = 50, ama: int = 100, lag: int = 3,
                           fast: int = 5, slow: int = 90, bull_threshold: float = 1.125, bear_threshold: float = 1.275) -> pd.Series:
    """价量共振（华创证券《成交量的奥秘：另类价量共振指标的择时》，2019）：

    价能 = HMA50 / 3 天前的 HMA50，量能 = HMA5(成交量) / HMA100(成交量)，指标 = 价能 × 量能；
    5 日均线在 90 日均线之上（多头市场）时指标大于 1.125 持有，否则要大于 1.275。参数同研报。
    """
    close, volume = ohlc["close"].astype(float), ohlc["vol"].astype(float)
    line = _hma(close, bma)
    indicator = (line / line.shift(lag)) * (_hma(volume, 5) / _hma(volume, ama))
    bull = close.rolling(fast).mean() > close.rolling(slow).mean()
    threshold = pd.Series(np.where(bull, bull_threshold, bear_threshold), index=close.index)
    return (indicator > threshold).astype(float)


def qrs_score(ohlc: pd.DataFrame, n: int = 18, m: int = 600) -> pd.Series:
    """QRS（中金公司《量化择时系列（1）：金融工程视角下的技术择时艺术》，2021：RSRS 标准分 × R²）：R² 为最高价对最低价回归的决定系数。"""
    high, low = ohlc["high"].astype(float), ohlc["low"].astype(float)
    r2 = high.rolling(n).corr(low) ** 2
    return rsrs_zscore(ohlc, n, m) * r2


def qrs_state(ohlc: pd.DataFrame, n: int = 18, m: int = 600, threshold: float = 0.7) -> pd.Series:
    """QRS 上穿 S 开仓、下穿 −S 平仓（研报 N=18、M=600、S=0.7）。"""
    return _hysteresis(qrs_score(ohlc, n, m), threshold, -threshold)
