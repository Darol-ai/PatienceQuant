"""V3 方案 A 阶段（engineering_only）：固定 10 只研究股票，组内等权预算，
组内相对排序（Top-K）控制买入，risk_exits + position_state 做逐日强制退出。

入场机制不再是 V3 原文"p_up>=0.60 且 p_down<=0.25"的绝对概率门槛——实测
这两个数字在浅层三分类模型 + 低基础概率的 UP 事件下几乎摸不到（训练集里
0 条样本达标）。调研了公开的 LightGBM 选股策略（BigQuant/Qlib 等）后发现
它们普遍不卡绝对阈值，而是"打分排序选前 K 名"（Qlib TopkDropoutStrategy
的思路），于是把这里也换成组内按 (p_up - p_down) 打分的相对排序（见相关
ADR）。

一开始加过"score 必须 > 0"这道方向性门槛，后来实测发现是错的：DOWN 事件
的标签边界比 UP 松（1.0 倍 vs 1.6 倍），天生更容易触发，一个诚实校准的
模型大多数时候 p_down 本来就会略高于 p_up，"score>0"会把大量"相对而言
这批里最不差"的情况也当成"没有信号"过滤掉，这本身是一种隐性偏见，不是
公开策略的做法——Qlib/BigQuant 的 Top-K 从不看绝对分数正负，只要打分能
分出相对高低就一直满仓排序。改成看"组内打分是否完全没有区分度"：完全
相同（比如下面的占位中性信号 neutral_signal，人人都是 0）才不硬凑排名，
只要分数有任何差异，哪怕全是负的，也照样选出相对最好的一批。

还没有接入真正训练好的 LightGBM 三分类模型时，signal_source 默认返回中性
占位信号（p_up=p_down=0）。等模型训练好后，把它包装成同样的
`(symbol, as_of) -> (p_up, p_down)` 签名传进来即可。
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from app.quant_v3.a_phase_universe import A_PHASE_STOCKS
from app.quant_v3.budget import allocate_flat_group_budget
from app.quant_v3.entry_signal import top_k_by_score
from app.quant_v3.position_state import CooldownTracker, PositionState
from app.quant_v3.risk_exits import (
    hard_stop_loss_triggered,
    resolve_exit_reason,
    trailing_stop_triggered,
)
from app.strategies.base import BaseStrategy, DailyRiskResult, StrategyResult

GROUP_BUDGETS = {"红利": 0.40, "成长": 0.30, "周期": 0.30}

SignalSource = Callable[[str, date], Tuple[float, float]]


def neutral_signal(symbol: str, as_of: date) -> Tuple[float, float]:
    """占位信号源：所有股票的 score(=p_up-p_down) 都恰好是 0，组内打分完全
    没有区分度，排不出相对高低，不会被硬凑进榜单。"""
    return 0.0, 0.0


class V3Strategy(BaseStrategy):
    def __init__(
        self,
        signal_source: Optional[SignalSource] = None,
        top_k_ratio: float = 0.5,
        hard_stop_loss_threshold: float = -0.15,
        trailing_stop_arm_threshold: float = 0.25,
        trailing_stop_drawdown_threshold: float = -0.10,
        universe: Optional[List[dict]] = None,
        group_budgets: Optional[Dict[str, float]] = None,
    ):
        """top_k_ratio：每个配置组里按打分选前 ratio 比例的股票（向上取整，
        至少 1 只），参照 Qlib TopkDropoutStrategy 的相对排序思路，默认取
        "组内前一半"。三个止损阈值默认是 V3 原文数值（-15%/25%触发/-10%
        回撤），可覆盖仅用于"止损参数是否对趋势行情过敏"这类敏感性实验
        （见相关 ADR），不改变默认交付行为。universe 默认是 V3 官方 A 阶段
        固定 10 支研究池（`A_PHASE_STOCKS`），可换成 `EXPANDED_A_PHASE_STOCKS`
        之类的候选池，用来做"扩大候选池是否让 Top-K 选出更好的相对赢家"
        这类范围外的研究性实验（同样不影响默认交付行为）。group_budgets
        默认是 V3 官方 40/30/30，可覆盖配合非官方 universe 使用（比如分散化
        候选池只有一个"综合"组，见 docs/adr/0011）。"""
        self._signal = signal_source or neutral_signal
        self.top_k_ratio = top_k_ratio
        self.hard_stop_loss_threshold = hard_stop_loss_threshold
        self.trailing_stop_arm_threshold = trailing_stop_arm_threshold
        self.trailing_stop_drawdown_threshold = trailing_stop_drawdown_threshold
        self.universe = universe if universe is not None else A_PHASE_STOCKS
        self.group_budgets = group_budgets if group_budgets is not None else GROUP_BUDGETS
        self.positions: Dict[str, PositionState] = {}
        self.cooldowns: Dict[str, CooldownTracker] = {}

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
            cooldown = self.cooldowns.get(symbol)
            if cooldown and cooldown.is_locked_out():
                rows.append({"symbol": symbol, "group": stock["group"], "p_up": None, "p_down": None, "action": "COOLDOWN"})
                continue
            p_up, p_down = self._signal(symbol, as_of)
            score = p_up - p_down
            scores_by_group[stock["group"]][symbol] = score
            rows.append({"symbol": symbol, "group": stock["group"], "p_up": p_up, "p_down": p_down, "action": "WATCH"})

        approved: set = set()
        for group, group_symbols in self._groups().items():
            group_scores = scores_by_group.get(group, {})
            if len(set(group_scores.values())) <= 1:
                continue  # 组内打分完全没有区分度（比如占位中性信号），不硬凑排名
            k = max(1, round(len(group_symbols) * self.top_k_ratio))
            approved |= top_k_by_score(group_scores, k)
        for row in rows:
            if row["symbol"] in approved:
                row["action"] = "BUY"

        weights: Dict[str, float] = {}
        for group, group_symbols in self._groups().items():
            weights.update(allocate_flat_group_budget(group_symbols, self.group_budgets[group], approved))

        # V3 方案第 4 节："月末未达买入条件也不因此自动清仓，不补仓"——入场
        # 门槛只管新买入/补仓，已经持有的仓位只能被 on_daily_close 的真正
        # 退出条件平掉，这里的普通调仓不能把它清零。
        stock_by_symbol = {stock["symbol"]: stock for stock in self.universe}
        groups = self._groups()
        for symbol in self.positions:
            if symbol in weights:
                continue
            stock = stock_by_symbol.get(symbol)
            if stock is None:
                continue
            group_symbols = groups[stock["group"]]
            weights[symbol] = self.group_budgets[stock["group"]] / len(group_symbols)

        ranking = pd.DataFrame(rows)
        ranking["score"] = ranking["p_up"].fillna(0.0) * 100
        ranking["target_weight"] = ranking["symbol"].map(weights).fillna(0.0)
        ranking = ranking.sort_values(["target_weight", "score"], ascending=False).reset_index(drop=True)
        ranking["rank"] = range(1, len(ranking) + 1)

        return StrategyResult(
            as_of=as_of,
            weights=weights,
            ranking=ranking,
            data_quality_notes=["V3 A 阶段 engineering_only：组内等权预算，模型拒绝买入的名额不顺延"],
        )

    def on_daily_close(self, current_date: date, price_row: pd.Series) -> DailyRiskResult:
        blocked: set = set()
        for symbol, tracker in self.cooldowns.items():
            tracker.advance_day()
            if tracker.is_locked_out():
                blocked.add(symbol)

        exits: Dict[str, str] = {}
        for symbol, state in list(self.positions.items()):
            raw_price = price_row.get(symbol)
            if raw_price is None or pd.isna(raw_price):
                continue
            price = float(raw_price)
            state.record_close(price)

            p_up, p_down = self._signal(symbol, current_date)
            risk_confirmed = state.risk_exit_counter.update(p_down >= 0.65)
            profit_confirmed = state.take_profit_counter.update(
                price / state.b - 1 >= 0.10 and p_down >= 0.50 and p_up <= 0.35
            )

            reason = resolve_exit_reason(
                qualification_exit=False,  # A 阶段固定研究池，本期不做资格判断
                hard_stop_loss=hard_stop_loss_triggered(price, state.b, threshold=self.hard_stop_loss_threshold),
                trailing_stop=trailing_stop_triggered(
                    price, state.b, state.h,
                    arm_threshold=self.trailing_stop_arm_threshold,
                    drawdown_threshold=self.trailing_stop_drawdown_threshold,
                ),
                model_risk_exit=risk_confirmed,
                model_take_profit=profit_confirmed,
            )
            if reason:
                exits[symbol] = reason

        return DailyRiskResult(exits=exits, blocked_symbols=blocked)

    def notify_fill(self, symbol: str, side: str, price: float, quantity: int, trade_date: date) -> None:
        if side == "BUY":
            self.positions.setdefault(symbol, PositionState(b=price))
        elif side == "SELL":
            if symbol in self.positions:
                del self.positions[symbol]
                self.cooldowns.setdefault(symbol, CooldownTracker()).start()
