"""V3 方案第 4 节：月末买入/补仓的入场门槛。

门槛数字可覆盖：默认是 V3 原文的 0.60/0.25。实测发现这两个绝对数字在
"防过拟合的浅层三分类模型 + 12% 基础概率的 UP 事件" 这套约束下几乎摸不到
（训练集里 0 条样本达到 p_up>=0.60，样本外 IC 也证明模型没有虚假自信）；
改用训练集打分分布的分位数作门槛（见 scripts/train_v3_model.py 和相关
ADR），规则在看回测结果之前就定死，不是为了让它出交易而事后调阈值。
"""


def entry_approved(
    p_up: float, p_down: float, p_up_threshold: float = 0.60, p_down_threshold: float = 0.25
) -> bool:
    """入选且资格有效的前提下，p_up>=p_up_threshold 且 p_down<=p_down_threshold
    才批准买入/补仓。"""
    return p_up >= p_up_threshold and p_down <= p_down_threshold


def top_k_by_score(scores: dict, k: int) -> set:
    """相对排序版的入场门槛：不卡绝对概率，按打分从高到低选前 k 名——参照
    公开 LightGBM 选股策略普遍采用的 Qlib TopkDropoutStrategy 思路。并列按
    股票代码升序，保证结果确定、可复现。"""
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {symbol for symbol, _ in ranked[:k]}
