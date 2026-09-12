"""V3 方案 2.3 / 9.1 节：A 阶段 engineering_only 模式的预算分配。

不做因子排序，组内固定预算等权；模型拒绝买入造成的空缺不再顺延，不转给其他股票。
"""

from collections.abc import Sequence


def allocate_flat_group_budget(
    target_symbols: Sequence[str],
    group_budget: float,
    model_approved: set[str],
) -> dict[str, float]:
    """按目标名单等分组预算，只给模型批准买入的股票分配权重。

    权重按 target_symbols 的原始名额数(而不是实际获批数)计算，
    被拒绝的名额空出来，不重新分给其他获批股票。
    """
    if not target_symbols:
        return {}

    weight_per_slot = group_budget / len(target_symbols)
    return {
        symbol: weight_per_slot
        for symbol in target_symbols
        if symbol in model_approved
    }


def allocate_score_weighted_budget(
    target_symbols: Sequence[str],
    group_budget: float,
    model_approved: set[str],
    scores: dict[str, float],
) -> dict[str, float]:
    """自由探索阶段（docs/adr/0017）：等权重在近乎普涨的行情里会把模型
    真正看好的黑马摊薄成和其他入选股票一样的权重——按预测分数加权，
    分数越高权重越大。分数先减去入选集合里的最小值（平移到非负）再
    归一化，避免负分数（比如回归预测的都是小额负收益）导致权重为负；
    全部分数相同时退化成等权重（没有区分度就没理由不等权）。
    """
    approved_in_pool = [symbol for symbol in target_symbols if symbol in model_approved]
    if not approved_in_pool:
        return {}
    approved_scores = {symbol: scores[symbol] for symbol in approved_in_pool}
    min_score = min(approved_scores.values())
    shifted = {symbol: value - min_score for symbol, value in approved_scores.items()}
    total = sum(shifted.values())
    if total <= 0:
        weight_per_slot = group_budget / len(approved_in_pool)
        return {symbol: weight_per_slot for symbol in approved_in_pool}
    return {symbol: group_budget * value / total for symbol, value in shifted.items()}


def allocate_full_weight_with_boost(
    target_symbols: Sequence[str],
    group_budget: float,
    model_approved: set[str],
    boosted_symbols: set[str],
    boost_multiplier: float,
) -> dict[str, float]:
    """自由探索阶段（docs/adr/0025）：等权重会把动量兜底选出的黑马和模型
    选出的股票摊薄在一起，稀释了黑马本该有的贡献（见ADR-0016/0022的
    诊断）——给"被动量兜底强制纳入"的股票单独乘一个权重倍数，其余股票
    仍然等权。按"单位数"分配：入选且被加权的股票算 boost_multiplier
    个单位，其余算1个单位，总预算按单位数比例分配，boost_multiplier=1.0
    时退化成普通等权重。
    """
    approved_in_pool = [symbol for symbol in target_symbols if symbol in model_approved]
    if not approved_in_pool:
        return {}
    units = {
        symbol: boost_multiplier if symbol in boosted_symbols else 1.0
        for symbol in approved_in_pool
    }
    total_units = sum(units.values())
    return {symbol: group_budget * unit / total_units for symbol, unit in units.items()}


def allocate_full_weight_to_approved(
    target_symbols: Sequence[str],
    group_budget: float,
    model_approved: set[str],
) -> dict[str, float]:
    """自由探索阶段（docs/adr/0014）：不再是 V3 的"名额空出不顺延"，选中
    几只就把预算全投进这几只——避免 Top-K 选出候选池一半时永远只有一半
    仓位在场内，在强趋势行情里白白留出一半现金吃不到涨幅。"""
    approved_in_pool = [symbol for symbol in target_symbols if symbol in model_approved]
    if not approved_in_pool:
        return {}
    weight_per_slot = group_budget / len(approved_in_pool)
    return {symbol: weight_per_slot for symbol in approved_in_pool}
