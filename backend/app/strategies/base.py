from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List

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


@dataclass
class StrategyResult:
    as_of: date
    weights: Dict[str, float]
    ranking: pd.DataFrame
    data_quality_notes: List[str]
    market_regime: str = "unknown"
    target_exposure: float = 1.0


class BaseStrategy:
    def generate_weights(self, as_of: date, symbols: List[str]) -> StrategyResult:
        raise NotImplementedError
