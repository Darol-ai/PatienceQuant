"""来自 QuantsPlaybook 的日线选股因子（ADR-0052 第三步，见仓库 NOTICE），写成面板函数：
输入日线宽表（行=交易日，列=股票，价格前复权），输出每天每支股票的因子值，只用当天及之前的数据。

横截面类因子（STR）的"截面"就是传入面板里的股票，即这次训练或回测的股票池。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Panels = dict


def _daily_return(p: Panels) -> pd.DataFrame:
    """日收益（今收/昨收−1）；停牌日没有成交，不当作 0 收益。"""
    ret = p["close"].pct_change(fill_method=None)
    return ret.where(p["suspended"] == 0) if "suspended" in p else ret


def salience_str(p: Panels, window: int = 20, delta: float = 0.7) -> pd.DataFrame:
    """凸显理论 STR 因子（招商证券《行为金融新视角，"凸显性收益"因子STR》，2022；源自 Cosemans & Frehen 2021）。

    - 凸显度 σ = |r − r̄| / (|r| + |r̄| + 0.1)，r̄ 为当天截面平均收益；
    - 当天按 σ 从大到小排名 k，权重 = δ^k 再除以截面均值（δ = 0.7）；
    - STR = 过去 20 天权重与日收益的协方差。越高表示近期越"凸显"的上涨越多，未来越容易回落。
    """
    ret = _daily_return(p)
    bench = ret.mean(axis=1)
    sigma = ret.sub(bench, axis=0).abs().div(ret.abs().add(bench.abs(), axis=0) + 0.1)
    rank = sigma.rank(axis=1, ascending=False)
    weight = np.power(delta, rank)
    weight = weight.div(weight.mean(axis=1), axis=0)
    return weight.rolling(window, min_periods=window // 2).cov(ret)


def terrified_score(p: Panels, window: int = 20) -> pd.DataFrame:
    """惊恐度因子（方正证券《显著效应、极端收益扭曲决策权重和"草木皆兵"因子》，2022）。

    惊恐度 = |r − r_m| / (|r| + |r_m| + 0.1)，r_m 为沪深300 日收益（研报用中证全指，本地只有沪深300）；
    用惊恐度加权日收益，取过去 20 天的均值与标准差的平均。越高表示近期被"守株待兔"式追捧越多，方向为负。
    """
    if "benchmark" not in p:
        return p["close"] * np.nan
    ret = _daily_return(p)
    market = p["benchmark"].pct_change()
    sigma = ret.sub(market, axis=0).abs().div(ret.abs().add(market.abs(), axis=0) + 0.1)
    weighted = sigma * ret
    roll = weighted.rolling(window, min_periods=window // 2)
    return (roll.mean() + roll.std()) * 0.5


def apb(p: Panels, window: int = 20) -> pd.DataFrame:
    """均价偏差 APB（东方证券《基于量价关系度量股票的买卖压力》，2019，月度版）。

    APB = ln( 过去 20 天日 vwap 的算术平均 / 按成交量加权的平均 )，vwap = 成交额/成交量，
    价格和成交量都按复权比例调整。低价位成交更多（买压大）时 APB > 0，方向为正。
    研报的日度版 apb_1d 需要 30 分钟 K 线，不包含。
    """
    ratio = p.get("adj_ratio")
    ratio = ratio if ratio is not None else 1.0
    volume = p["volume"].where(p["volume"] > 0)
    vwap = p["amount"] / volume * ratio
    adj_volume = volume / ratio
    valid = vwap.notna() & adj_volume.notna()
    vwap, adj_volume = vwap.where(valid), adj_volume.where(valid)
    min_days = window // 2
    simple = vwap.rolling(window, min_periods=min_days).mean()
    weighted = (vwap * adj_volume).rolling(window, min_periods=min_days).sum() / adj_volume.rolling(window, min_periods=min_days).sum()
    return np.log(simple / weighted)
