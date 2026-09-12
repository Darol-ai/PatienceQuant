"""自由探索阶段（docs/adr/0013）：LightGBM 回归预测未来收益 + 组内(这里
是单一"综合"组)Top-K 相对排序 + 大盘趋势过滤整体仓位暴露。三个机制各自
独立测过（Top-K：ADR-0008；趋势过滤思路：借鉴 PatienceQuant 自带
MultiFactorStrategy 和 ADR-0010 的 Dual Momentum 绝对动量过滤；分散化
候选池：ADR-0011/0012），这里第一次把它们组合到一起、配合回归预测分数
（不是三分类概率差）。

没有止损/跟踪止损/冷静期这类逐日状态机——ADR-0010 已经验证"月度调仓
自然淘汰打分变差的持仓"本身就足够，止损层是冗余的。
"""
from __future__ import annotations

from datetime import date
from typing import Callable, Dict, List, Optional

import pandas as pd

from app.quant_v3.budget import (
    allocate_full_weight_to_approved,
    allocate_full_weight_with_boost,
    allocate_score_weighted_budget,
)
from app.quant_v3.entry_signal import top_k_by_score
from app.strategies.base import BaseStrategy, StrategyResult

ScoreSource = Callable[[str, date], Optional[float]]
TrendSource = Callable[[date], float]
RallyTrigger = Callable[[date], bool]
QualificationSource = Callable[[str, date], bool]


class RegressionRotationStrategy(BaseStrategy):
    def __init__(
        self,
        signal_source: ScoreSource,
        universe: List[dict],
        group_budgets: Dict[str, float],
        top_k_ratio: float = 0.5,
        trend_signal: Optional[TrendSource] = None,
        score_weighted: bool = False,
        breadth_threshold: Optional[float] = None,
        momentum_override_source: Optional[ScoreSource] = None,
        momentum_override_count: int = 0,
        momentum_override_min_hold_months: int = 0,
        momentum_override_boost: float = 1.0,
        full_coverage_trigger: Optional[RallyTrigger] = None,
        qualification_source: Optional[QualificationSource] = None,
    ):
        """score_weighted=True 时（docs/adr/0017）入选股票按预测分数加权
        （分数越高权重越大），不是入选即等权——等权重在近乎普涨的行情里
        会把模型真正看好的黑马摊薄成和其他入选股票一样的权重。默认
        False（等权重），不改变已有行为。

        breadth_threshold（docs/adr/0018）：ADR-0016 诊断出固定
        top_k_ratio 在"几乎全员看涨"的月份会把大部分涨幅拒之门外——这里
        用模型自己的打分算"广度"（组内正分数股票占比），某个月的广度
        达到或超过这个阈值时，当月临时把该组的 top_k_ratio 拉到 1.0
        （全覆盖），低于阈值时维持原本设的 top_k_ratio。默认 None 表示
        不启用这个机制（行为和之前完全一样）。是模型自己的打分在决定
        要不要扩大覆盖面，不是外部指标，符合"以LightGBM为主体"的要求。

        momentum_override_source/momentum_override_count（docs/adr/0022）：
        诊断发现模型在近乎普涨行情里会误判个别真正的动量黑马（比如2019年
        1月给当年最大涨幅股打了很低的分）——这里加一层动量兜底，不管模型
        打分如何，每次调仓强制把每组动量最强的 `momentum_override_count`
        支也纳入候选（用独立于模型的原始动量信号，比如trailing_momentum）。
        默认 0（不启用），是模型选股之外的安全网，不是替代模型。

        momentum_override_min_hold_months（docs/adr/0024）：诊断出2019年
        动量兜底"时进时出"——某支股票靠动量兜底某月进场，下个月动量排名
        掉出前列就被踢出，没能完整拿到持续上涨的涨幅。设为>0时，一旦某
        股票靠动量兜底进场，接下来至少强制持有这么多个月（含进场当月），
        不因为后续排名下降就提前踢出，到期后如果模型/动量都不再支持才
        真正剔除。默认 0（不启用，行为和之前完全一样，进场当月之外没有
        额外粘性）。

        momentum_override_boost（docs/adr/0025）：动量兜底选出的黑马和
        模型选出的股票等权重摊薄在一起，稀释了黑马本该有的贡献——大于
        1.0时，被动量兜底强制纳入(含粘性持有期内)的股票按这个倍数单独
        加权，不是简单等权（`budget.allocate_full_weight_with_boost`）。
        默认 1.0（不启用，等价于普通等权重）。仅在 score_weighted=False
        时生效——score_weighted 已经是另一套差异化加权方式，两者不叠加。

        full_coverage_trigger（docs/adr/0031）：ADR-0018/0020已经证明"用
        模型自己的打分算广度"是基准计算bug的产物，不是真实机制——这里换
        成基于大盘指数**已实现**动量的独立信号(不依赖模型判断，比如指数
        trailing N日涨幅超过阈值)，某个月触发时直接全覆盖(不管
        top_k_ratio/breadth_threshold平时设多严格)，未触发时行为完全
        不变。默认 None（不启用）。

        qualification_source（docs/adr/0036）：V3 方案 4.1 节资格退出的
        优先级高于模型/风控退出——不合格的股票既不能被 Top-K 选中，也不能
        被动量兜底强制纳入，不管打分多高。默认 None（不启用，所有股票都
        视为合格，行为不变）。
        """
        self._signal = signal_source
        self.universe = universe
        self.group_budgets = group_budgets
        self.top_k_ratio = top_k_ratio
        self._trend_signal = trend_signal or (lambda as_of: 1.0)
        self.score_weighted = score_weighted
        self.breadth_threshold = breadth_threshold
        self._momentum_override_source = momentum_override_source
        self.momentum_override_count = momentum_override_count
        self.momentum_override_min_hold_months = momentum_override_min_hold_months
        self.momentum_override_boost = momentum_override_boost
        self._full_coverage_trigger = full_coverage_trigger
        self._qualification_source = qualification_source
        self._sticky_hold_remaining: Dict[str, int] = {}

    def _groups(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for stock in self.universe:
            groups.setdefault(stock["group"], []).append(stock["symbol"])
        return groups

    def generate_weights(self, as_of: date, symbols: List[str]) -> StrategyResult:
        rows = []
        scores_by_group: Dict[str, Dict[str, float]] = {}
        disqualified: set = set()
        for stock in self.universe:
            symbol = stock["symbol"]
            qualified = self._qualification_source is None or self._qualification_source(symbol, as_of)
            if not qualified:
                disqualified.add(symbol)
            score = self._signal(symbol, as_of)
            if score is not None and qualified:
                scores_by_group.setdefault(stock["group"], {})[symbol] = score
            rows.append({"symbol": symbol, "group": stock["group"], "score": score, "qualified": qualified})

        approved: set = set()
        boosted: set = set()
        for group, group_symbols in self._groups().items():
            group_scores = scores_by_group.get(group, {})
            if not group_scores:
                continue
            effective_ratio = self.top_k_ratio
            if self.breadth_threshold is not None:
                breadth = sum(1 for value in group_scores.values() if value > 0) / len(group_symbols)
                if breadth >= self.breadth_threshold:
                    effective_ratio = 1.0
            if self._full_coverage_trigger is not None and self._full_coverage_trigger(as_of):
                effective_ratio = 1.0
            k = max(1, round(len(group_symbols) * effective_ratio))
            approved |= top_k_by_score(group_scores, k)

            if self._momentum_override_source is not None and self.momentum_override_count > 0:
                momentum_scores = {
                    symbol: value
                    for symbol in group_symbols
                    if symbol not in disqualified
                    and (value := self._momentum_override_source(symbol, as_of)) is not None
                }
                if momentum_scores:
                    override_count = min(self.momentum_override_count, len(momentum_scores))
                    fresh_overrides = top_k_by_score(momentum_scores, override_count)
                    for symbol in fresh_overrides:
                        # 新进场(或粘性到期后重新入选)才重置持有计数，已经在粘性期
                        # 内的不重复延长，避免"每个月都续命"变相无限期持有。
                        if symbol not in self._sticky_hold_remaining:
                            self._sticky_hold_remaining[symbol] = self.momentum_override_min_hold_months
                    approved |= fresh_overrides
                    boosted |= fresh_overrides

        sticky_symbols = {
            symbol for symbol, remaining in self._sticky_hold_remaining.items()
            if remaining > 0 and symbol not in disqualified
        }
        approved |= sticky_symbols
        boosted |= sticky_symbols
        for symbol in list(self._sticky_hold_remaining):
            self._sticky_hold_remaining[symbol] -= 1
            if self._sticky_hold_remaining[symbol] <= 0:
                del self._sticky_hold_remaining[symbol]

        flat_scores = {symbol: score for group_scores in scores_by_group.values() for symbol, score in group_scores.items()}
        exposure = self._trend_signal(as_of)
        weights: Dict[str, float] = {}
        for group, group_symbols in self._groups().items():
            if self.score_weighted:
                group_weights = allocate_score_weighted_budget(
                    group_symbols, self.group_budgets[group], approved, flat_scores
                )
            elif self.momentum_override_boost != 1.0:
                group_weights = allocate_full_weight_with_boost(
                    group_symbols, self.group_budgets[group], approved, boosted, self.momentum_override_boost
                )
            else:
                group_weights = allocate_full_weight_to_approved(group_symbols, self.group_budgets[group], approved)
            weights.update({symbol: w * exposure for symbol, w in group_weights.items()})

        ranking = pd.DataFrame(rows)
        ranking["target_weight"] = ranking["symbol"].map(weights).fillna(0.0)
        ranking = ranking.sort_values(["target_weight", "score"], ascending=False).reset_index(drop=True)
        ranking["rank"] = range(1, len(ranking) + 1)

        return StrategyResult(
            as_of=as_of,
            weights=weights,
            ranking=ranking,
            data_quality_notes=[f"回归动量轮动：整体仓位暴露 {exposure:.0%}（大盘趋势过滤）"],
        )
