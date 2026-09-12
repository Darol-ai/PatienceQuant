from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Set

import pandas as pd


@dataclass
class StrategyConfig:
    name: str = "沪深300增强趋势价值成长策略 V3"
    weights: Dict[str, float] = field(default_factory=lambda: {
        "fundamental": .12, "valuation": .12, "quality": .16, "momentum": .40, "risk": .20
    })
    holdings_count: int = 10
    max_weight: float = .15
    rebalance_frequency: str = "monthly"
    cash_buffer: float = .02
    trend_filter: bool = True
    risk_off_exposure: float = .75
    turnover_band: float = .03
    stop_loss: float = .18
    benchmark_enhancement: bool = True
    dip_buy_strength: float = .06
    profit_take_strength: float = .05
    # Portfolio-level risk budget.  These controls scale exposure; they do
    # not predict returns or guarantee a loss limit.
    target_volatility: float = .22
    max_drawdown_budget: float = .15
    drawdown_brake_exposure: float = .50
    model_enabled: bool = True
    model_buy_threshold: float = .60
    model_down_threshold: float = .25


@dataclass
class StrategyResult:
    as_of: date
    weights: Dict[str, float]
    ranking: pd.DataFrame
    data_quality_notes: List[str]
    market_regime: str = "unknown"
    target_exposure: float = 1.0


@dataclass
class DailyRiskResult:
    """BaseStrategy.on_daily_close 的返回值。

    exits: 今天必须强制清仓的 symbol -> 原因，引擎会在当天以收盘价（含滑点）
    卖出，不管这天是不是调仓日、也不管策略当次 generate_weights 给的目标权重。
    blocked_symbols: 当前处于冷静期、这次调仓不能新买入的 symbol。
    """

    exits: Dict[str, str] = field(default_factory=dict)
    blocked_symbols: Set[str] = field(default_factory=set)


class BaseStrategy:
    def generate_weights(self, as_of: date, symbols: List[str]) -> StrategyResult:
        raise NotImplementedError

    def on_daily_close(self, current_date: date, price_row: pd.Series) -> DailyRiskResult:
        """引擎每个交易日收盘后都会调用一次（不止调仓日），用于逐日强制退出
        判断和冷静期维护。默认策略没有逐日状态，不强制退出、不封锁任何股票。
        """
        return DailyRiskResult()

    def notify_fill(self, symbol: str, side: str, price: float, quantity: int, trade_date: date) -> None:
        """引擎每次实际成交（买/卖，含强制退出）后回调一次，供策略更新自己的
        持仓状态（比如 V3 的建仓基准价 B、区间最高价 H）。默认不做任何事。
        """
        return None
