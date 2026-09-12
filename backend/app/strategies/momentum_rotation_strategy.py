"""公开策略调研新方向（docs/adr/0010）：ADR-0009 排除了止损参数、截面
标准化、候选池大小、回测区间四个维度后，换一类完全不同的策略机制——不再
用 LightGBM 分类模型预测涨跌概率，改用学术界最经典的 Jegadeesh-Titman
"12-1 动量"因子 + Gary Antonacci "Dual Momentum" 的绝对动量过滤（自身
动量为负就空仓，不参与组内排序，budget 直接空出不转给别的股票）。

同时验证 ADR-0009 候选方向 1：不设止损/跟踪止损，只靠月度调仓时动量转负
被踢出重新分配——`on_daily_close` 用 BaseStrategy 默认的空实现，没有
`self.positions`/`CooldownTracker` 这类逐日状态。
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Callable, Dict, List, Optional

from app.quant_v3.budget import allocate_flat_group_budget
from app.quant_v3.entry_signal import top_k_by_score
from app.strategies.base import BaseStrategy, StrategyResult

import pandas as pd

MomentumSource = Callable[[str, date], Optional[float]]


class MomentumRotationStrategy(BaseStrategy):
    def __init__(
        self,
        signal_source: MomentumSource,
        universe: List[dict],
        group_budgets: Dict[str, float],
        top_k_ratio: float = 0.5,
    ):
        self._signal = signal_source
        self.universe = universe
        self.group_budgets = group_budgets
        self.top_k_ratio = top_k_ratio

    def _groups(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for stock in self.universe:
            groups.setdefault(stock["group"], []).append(stock["symbol"])
        return groups

    def generate_weights(self, as_of: date, symbols: List[str]) -> StrategyResult:
        rows = []
        scores_by_group: Dict[str, Dict[str, float]] = defaultdict(dict)
        for stock in self.universe:
            symbol = stock["symbol"]
            score = self._signal(symbol, as_of)
            # 绝对动量过滤（Dual Momentum）：动量缺失或为负就不参与排序，
            # 这是这套机制本身的设计，不是像 V3 早期版本那样的隐性偏见。
            if score is not None and score > 0:
                scores_by_group[stock["group"]][symbol] = score
            rows.append({"symbol": symbol, "group": stock["group"], "score": score})

        approved: set = set()
        for group, group_symbols in self._groups().items():
            group_scores = scores_by_group.get(group, {})
            if not group_scores:
                continue
            k = max(1, round(len(group_symbols) * self.top_k_ratio))
            approved |= top_k_by_score(group_scores, k)

        weights: Dict[str, float] = {}
        for group, group_symbols in self._groups().items():
            weights.update(
                allocate_flat_group_budget(group_symbols, self.group_budgets[group], approved)
            )

        ranking = pd.DataFrame(rows)
        ranking["target_weight"] = ranking["symbol"].map(weights).fillna(0.0)
        ranking = ranking.sort_values(["target_weight", "score"], ascending=False).reset_index(drop=True)
        ranking["rank"] = range(1, len(ranking) + 1)

        return StrategyResult(
            as_of=as_of,
            weights=weights,
            ranking=ranking,
            data_quality_notes=["动量轮动(12-1动量+绝对动量过滤)：不设止损，纯周期调仓踢出动量转负的持仓"],
        )
