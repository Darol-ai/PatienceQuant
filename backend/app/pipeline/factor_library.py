"""因子库（ADR-0052）：系统里全部因子的唯一目录。

每个因子写明名称、所属因子组、方向（1 越大越好 / -1 越小越好）、所需数据，以及一个
"面板函数"：输入一段区间的日线宽表（行=交易日，列=股票），一次算出每一天每支股票的
因子值（只用当天及之前的数据）。模型训练和模型打分都用面板函数，保证训练和预测口径一致。

旧模型的 14 个输入因子（原来叫"14 项特征"）和 app/quant_v3/dataset.compute_features
逐条计算的结果一致（tests/test_factor_library.py 核对），并且沿用它的约定：一支股票
上市（或进入行情库）满 121 个交易日之前，这 14 个因子都不给值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from app.pipeline import playbook_factors, price_factors

Panels = Dict[str, pd.DataFrame]  # open/high/low/close/volume/amount/suspended/bars/benchmark

LEGACY_MIN_BARS = 121  # compute_features 要求当天之前至少 120 个交易日


@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    group: str
    direction: int
    description: str
    requires: str = "daily"  # daily：只要日线；financial：要财务数据（还没接入）
    compute: Optional[Callable[[Panels], pd.DataFrame]] = field(default=None, compare=False)
    min_bars: int = 0


# ---- 面板构建 ----

def panels_from_store(store, symbols: List[str], start: date, end: date, benchmark: Optional[pd.Series] = None) -> Panels:
    """从本地行情库取日线，整理成宽表。口径和 app/pipeline/bars.load_bars 相同：
    成交量换成股、成交额换成元；停牌日（首次出现之后、最后出现之前缺的交易日）
    价格沿用前收、成交量 0、记为停牌。"""
    from app.data.tushare_provider import _to_ts_code

    code_to_symbol = {_to_ts_code(s): s for s in dict.fromkeys(symbols)}
    daily = store.load_daily(start, end, list(code_to_symbol), adjust="qfq")
    calendar = pd.DatetimeIndex(sorted(d for d in store.stored_days() if start <= d <= end))
    if daily.empty:
        empty = pd.DataFrame(index=calendar)
        return {k: empty.copy() for k in ("open", "high", "low", "close", "volume", "amount", "suspended", "bars")}
    daily = daily.assign(symbol=daily["ts_code"].map(code_to_symbol))
    wide = {col: daily.pivot_table(index="trade_date", columns="symbol", values=col, aggfunc="last").reindex(calendar)
            for col in ("open", "high", "low", "close", "vol", "amount", "adj_factor")}
    panels = _finish_panels(wide["open"], wide["high"], wide["low"], wide["close"], wide["vol"] * 100, wide["amount"] * 1000,
                            traded=wide["close"].notna(), benchmark=benchmark)
    # 前复权比例（当天复权因子 / 区间最后一天的），给需要复权成交量、vwap 的因子用
    adj = wide["adj_factor"].ffill()
    panels["adj_ratio"] = (adj / adj.iloc[-1]).where(panels["close"].notna())
    return panels


def panels_from_history(history: pd.DataFrame) -> Panels:
    """从旧的离线历史表（symbol/date/open/high/low/close/volume/amount/is_suspended，
    停牌日已经是一行）整理成宽表——只用于核对和旧流程的等价性。"""
    frame = history.assign(date=pd.to_datetime(history["date"]))
    wide = {col: frame.pivot_table(index="date", columns="symbol", values=col, aggfunc="last").sort_index()
            for col in ("open", "high", "low", "close", "volume", "amount")}
    suspended = frame.pivot_table(index="date", columns="symbol", values="is_suspended", aggfunc="last").sort_index()
    present = wide["close"].notna()
    panels = {k: v.where(present) for k, v in wide.items()}
    panels["suspended"] = suspended.astype(float).where(present)
    panels["bars"] = present.cumsum().where(present)
    return panels


def _finish_panels(open_, high, low, close, volume, amount, traded: pd.DataFrame, benchmark=None) -> Panels:
    # 每支股票从第一次出现到最后一次出现之间算"在行情里"，中间缺的日子是停牌
    first = traded.cummax()
    last = traded[::-1].cummax()[::-1]
    alive = first & last
    close = close.ffill().where(alive)
    panels = {
        "close": close,
        "open": open_.where(traded, close).where(alive),
        "high": high.where(traded, close).where(alive),
        "low": low.where(traded, close).where(alive),
        "volume": volume.where(traded, 0.0).where(alive),
        "amount": amount.where(traded, 0.0).where(alive),
        "suspended": (~traded).astype(float).where(alive),
    }
    panels["bars"] = alive.cumsum().where(alive)
    if benchmark is not None:
        panels["benchmark"] = benchmark.reindex(close.index).ffill()
    return panels


# ---- 旧模型的 14 个因子（向量化，结果与 compute_features 一致）----

def _legacy(compute):
    def wrapped(panels: Panels) -> pd.DataFrame:
        return compute(panels).where(panels["bars"] >= LEGACY_MIN_BARS)
    return wrapped


def _ret(window):
    return _legacy(lambda p: p["close"] / p["close"].shift(window) - 1)


def _vol(window):
    def compute(p):
        log_ret = np.log(p["close"] / p["close"].shift(1))
        return log_ret.rolling(window).std(ddof=1)
    return _legacy(compute)


def _ma_dev(window):
    return _legacy(lambda p: p["close"] / p["close"].rolling(window).mean() - 1)


def _max_drawdown(window):
    def compute(p):
        close = p["close"].to_numpy(dtype=float)
        out = np.full(close.shape, np.nan)
        if len(close) >= window:
            view = np.lib.stride_tricks.sliding_window_view(close, window, axis=0)  # (T-w+1, N, w)
            peak = np.maximum.accumulate(view, axis=2)
            out[window - 1:] = np.min(view / peak - 1, axis=2).clip(max=0.0)
        return pd.DataFrame(out, index=p["close"].index, columns=p["close"].columns)
    return _legacy(compute)


def _atr_ratio(window):
    def compute(p):
        prev_close = p["close"].shift(1).to_numpy(dtype=float)
        high, low = p["high"].to_numpy(dtype=float), p["low"].to_numpy(dtype=float)
        # np.maximum 遇到 NaN 返回 NaN：没有前收的那天没有真实波幅，和逐条计算一致
        tr = np.maximum(np.maximum(high - low, np.abs(high - prev_close)), np.abs(low - prev_close))
        tr = pd.DataFrame(tr, index=p["close"].index, columns=p["close"].columns)
        return tr.rolling(window).mean() / p["close"]
    return _legacy(compute)


def _volume_ratio(window):
    def compute(p):
        mean = p["volume"].rolling(window).mean()
        return (p["volume"] / mean).where(mean != 0)
    return _legacy(compute)


def _avg_amount(window):
    return _legacy(lambda p: p["amount"].rolling(window).mean())


def _valid_ratio(window):
    return _legacy(lambda p: 1 - p["suspended"].rolling(window).mean())


# ---- 因子权重策略原有的动量组、风险组（按一年约 252 个交易日）----

def _period_return(days):
    return lambda p: p["close"] / p["close"].shift(days) - 1


def _annual_vol(p):
    return p["close"].pct_change(fill_method=None).rolling(252, min_periods=60).std() * np.sqrt(252)


def _drawdown_1y(p):
    peak = p["close"].rolling(252, min_periods=60).max()
    return (1 - p["close"] / peak).rolling(252, min_periods=60).max()


def _beta_1y(p):
    if "benchmark" not in p:
        return p["close"] * np.nan
    stock = p["close"].pct_change(fill_method=None)
    bench = p["benchmark"].pct_change()
    cov = stock.rolling(252, min_periods=60).cov(bench)
    return (cov.div(bench.rolling(252, min_periods=60).var(), axis=0)).abs()


def _single_day(fn, lookback):
    """把"算最后一天"的截面函数（上下影线、理想振幅）扩成逐日面板。只在需要时用（训练）。"""
    def compute(p):
        out = pd.DataFrame(np.nan, index=p["close"].index, columns=p["close"].columns)
        cols = {k: p[k] for k in ("open", "high", "low", "close")}
        for i in range(lookback, len(out)):
            window = {k: v.iloc[i - lookback: i + 1] for k, v in cols.items()}
            out.iloc[i] = fn(window).reindex(out.columns).to_numpy()
        return out
    return compute


_LEGACY_14 = [
    Factor("return_5d", "5 日收益", "价格动量", 1, "最近 5 个交易日涨跌幅", compute=_ret(5)),
    Factor("return_20d", "20 日收益", "价格动量", 1, "最近 20 个交易日涨跌幅", compute=_ret(20)),
    Factor("return_60d", "60 日收益", "价格动量", 1, "最近 60 个交易日涨跌幅", compute=_ret(60)),
    Factor("return_120d", "120 日收益", "价格动量", 1, "最近 120 个交易日涨跌幅", compute=_ret(120)),
    Factor("volatility_20d", "20 日波动率", "波动", -1, "最近 20 个交易日对数收益的标准差", compute=_vol(20)),
    Factor("volatility_60d", "60 日波动率", "波动", -1, "最近 60 个交易日对数收益的标准差", compute=_vol(60)),
    Factor("max_drawdown_60d", "60 日最大回撤", "波动", 1, "最近 60 个交易日内相对高点的最大跌幅（负数，越接近 0 越好）", compute=_max_drawdown(60)),
    Factor("ma_deviation_20d", "20 日均线偏离", "均线", 1, "收盘价相对 20 日均线的偏离", compute=_ma_dev(20)),
    Factor("ma_deviation_60d", "60 日均线偏离", "均线", 1, "收盘价相对 60 日均线的偏离", compute=_ma_dev(60)),
    Factor("ma_deviation_120d", "120 日均线偏离", "均线", 1, "收盘价相对 120 日均线的偏离", compute=_ma_dev(120)),
    Factor("atr_ratio_20d", "20 日 ATR 比率", "波动", -1, "20 日平均真实波幅除以收盘价", compute=_atr_ratio(20)),
    Factor("volume_ratio_20d", "20 日量比", "量能", 1, "当天成交量除以 20 日平均成交量", compute=_volume_ratio(20)),
    Factor("avg_amount_60d", "60 日平均成交额", "量能", 1, "最近 60 个交易日平均成交额（元），反映流动性", compute=_avg_amount(60)),
    Factor("valid_trading_ratio_60d", "60 日有效交易占比", "量能", 1, "最近 60 个交易日里没有停牌的比例", compute=_valid_ratio(60)),
]
for _f in _LEGACY_14:
    object.__setattr__(_f, "min_bars", LEGACY_MIN_BARS)

FACTOR_LIBRARY: Dict[str, Factor] = {f.key: f for f in [
    *_LEGACY_14,
    Factor("return_3m", "3 个月收益", "动量组", 1, "最近 63 个交易日涨跌幅", compute=_period_return(63)),
    Factor("return_6m", "6 个月收益", "动量组", 1, "最近 126 个交易日涨跌幅", compute=_period_return(126)),
    Factor("return_12m", "12 个月收益", "动量组", 1, "最近 252 个交易日涨跌幅", compute=_period_return(252)),
    Factor("volatility", "年化波动率", "风险组", -1, "最近一年日收益的年化标准差", compute=_annual_vol),
    Factor("max_drawdown", "一年最大回撤", "风险组", -1, "最近一年相对高点的最大跌幅（正数，越小越好）", compute=_drawdown_1y),
    Factor("beta", "Beta", "风险组", -1, "最近一年相对沪深300 的 Beta", compute=_beta_1y),
    Factor("ubl", "上下影线（UBL）", "形态", -1, price_factors.PRICE_FACTORS["ubl"].description,
           compute=_single_day(price_factors.ubl, 30)),
    Factor("ideal_amplitude", "理想振幅", "形态", -1, price_factors.PRICE_FACTORS["ideal_amplitude"].description,
           compute=_single_day(price_factors.ideal_amplitude, 25)),
    Factor("salience_str", "凸显理论 STR", "行为金融", -1,
           "近 20 天按凸显度加权的收益协方差，越高越容易回落（招商证券 2022）", compute=playbook_factors.salience_str),
    Factor("terrified_score", "惊恐度", "行为金融", -1,
           "近 20 天相对沪深300 的惊恐度加权收益（均值与波动的平均），越低越好（方正证券 2022）", compute=playbook_factors.terrified_score),
    Factor("apb_20d", "买卖压力 APB", "量价", 1,
           "近 20 天 vwap 算术平均相对成交量加权平均的对数偏差，越高买压越大（东方证券 2019）", compute=playbook_factors.apb),
    *[Factor(key, label, group, direction, description, requires="financial") for key, label, group, direction, description in [
        ("roe", "ROE", "基本面组", 1, "净资产收益率"), ("roa", "ROA", "基本面组", 1, "总资产收益率"),
        ("revenue_growth", "营收增长", "基本面组", 1, "营业收入同比增长"), ("profit_growth", "利润增长", "基本面组", 1, "净利润同比增长"),
        ("operating_cashflow", "经营现金流", "基本面组", 1, "经营活动现金流"),
        ("pe", "市盈率", "估值组", -1, "PE"), ("pb", "市净率", "估值组", -1, "PB"), ("ps", "市销率", "估值组", -1, "PS"),
        ("dividend_yield", "股息率", "估值组", 1, "股息率"),
        ("roe_stability", "ROE 稳定性", "盈利质量组", 1, "ROE 的稳定程度"), ("gross_margin", "毛利率", "盈利质量组", 1, "毛利率"),
        ("net_margin", "净利率", "盈利质量组", 1, "净利率"), ("cashflow_profit_ratio", "现金流/利润", "盈利质量组", 1, "经营现金流与净利润之比"),
    ]],
]}

LEGACY_MODEL_FACTORS: List[str] = [f.key for f in _LEGACY_14]

# 因子权重策略的因子组（FactorEngine 的五组）→ 组内因子
FACTOR_GROUPS: Dict[str, List[str]] = {
    "fundamental": ["roe", "roa", "revenue_growth", "profit_growth", "operating_cashflow"],
    "valuation": ["pe", "pb", "ps", "dividend_yield"],
    "quality": ["roe_stability", "gross_margin", "net_margin", "cashflow_profit_ratio"],
    "momentum": ["return_3m", "return_6m", "return_12m"],
    "risk": ["volatility", "max_drawdown", "beta"],
}


def trainable(key: str) -> bool:
    factor = FACTOR_LIBRARY.get(key)
    return factor is not None and factor.requires == "daily" and factor.compute is not None


def compute_factors(panels: Panels, keys: List[str]) -> Dict[str, pd.DataFrame]:
    missing = [k for k in keys if not trainable(k)]
    if missing:
        raise ValueError("这些因子不能用来训练（没有真实数据或没有计算方法）：%s" % "、".join(missing))
    return {key: FACTOR_LIBRARY[key].compute(panels) for key in keys}
