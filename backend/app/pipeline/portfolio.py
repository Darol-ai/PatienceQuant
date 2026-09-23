"""① 选股规则 和 ② 权重方案：只认分数，不关心分数是因子还是模型算的。"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from app.pipeline.spec import SelectionSpec, WeightingSpec


def select(scores: pd.Series, rule: SelectionSpec, universe_size: int) -> pd.Series:
    """按分数从高到低取前 N 名 / 前 x%，返回入选股票的分数（已排序）。

    前 x% 的名额按股票池大小算（不是按"有分数的股票数"）：和迁移前沪深300
    模型策略的口径一致（300 支取 10% = 30 支），这样池子里一部分股票当天
    没分数（新股、停牌、历史不够）时名额不会跟着缩水。
    """
    scored = scores.dropna().sort_values(ascending=False, kind="mergesort")
    if rule.type == "top_n":
        k = rule.n
    else:
        k = max(1, round(universe_size * rule.pct))
    return scored.head(k)


def _cap(raw: Dict[str, float], max_weight: float) -> Dict[str, float]:
    """按比例分配，超过单股上限的截到上限、多出来的再按比例分给没到上限的；
    所有股票都到上限后剩下的留作现金，不强行满仓。"""
    result = {symbol: 0.0 for symbol in raw}
    active = {symbol for symbol, value in raw.items() if value > 0}
    remaining = 1.0
    while active and remaining > 1e-12:
        total = sum(raw[s] for s in active)
        capped = {s for s in active if result[s] + remaining * raw[s] / total >= max_weight - 1e-12}
        if not capped:
            for s in active:
                result[s] += remaining * raw[s] / total
            break
        for s in capped:
            remaining -= max_weight - result[s]
            result[s] = max_weight
        active -= capped
    return {s: w for s, w in result.items() if w > 1e-9}


def weigh(selected: pd.Series, all_scores: pd.Series, rule: WeightingSpec) -> Dict[str, float]:
    if selected.empty:
        return {}
    if rule.type == "equal":
        raw = {symbol: 1.0 for symbol in selected.index}
    else:
        # 因子分数是 0～100 的百分位，直接按分数比例；模型分数（预测收益）
        # 可能是负数，这时按整个截面的最低分平移到非负，入选股票分数越高
        # 权重越大。入选股票分数全是 0 时退化成等权。
        floor = min(float(all_scores.dropna().min()), 0.0)
        shifted = selected - floor
        if float(shifted.sum()) <= 0:
            raw = {symbol: 1.0 for symbol in selected.index}
        else:
            raw = {symbol: float(value) for symbol, value in shifted.items()}
    return _cap(raw, rule.max_weight)
