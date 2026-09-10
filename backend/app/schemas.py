from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


DEFAULT_WEIGHTS = {"fundamental": .12, "valuation": .12, "quality": .16, "momentum": .40, "risk": .20}


class StrategyPayload(BaseModel):
    name: str = "沪深300增强趋势价值成长策略 V3"
    description: str = "以跑赢或贴近沪深300为目标，并用目标波动率与回撤预算控制组合暴露"
    weights: Dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    holdings_count: int = Field(10, ge=10, le=100)
    max_weight: float = Field(.15, gt=0, le=1)
    rebalance_frequency: str = "monthly"
    universe: str = "large_cap"
    research_start_date: date = date(2018, 1, 1)
    research_end_date: date = date(2025, 12, 31)
    cash_buffer: float = Field(.02, ge=0, lt=.5)
    trend_filter: bool = True
    risk_off_exposure: float = Field(.75, gt=0, le=1)
    turnover_band: float = Field(.03, ge=0, le=.25)
    stop_loss: float = Field(.18, gt=0, le=.8)
    benchmark_enhancement: bool = True
    dip_buy_strength: float = Field(.06, ge=0, le=.5)
    profit_take_strength: float = Field(.05, ge=0, le=.5)
    target_volatility: float = Field(.22, gt=0, le=1)
    max_drawdown_budget: float = Field(.15, gt=0, le=.95)
    drawdown_brake_exposure: float = Field(.50, gt=0, le=1)

    @field_validator("weights")
    @classmethod
    def valid_weights(cls, value: Dict[str, float]) -> Dict[str, float]:
        required = set(DEFAULT_WEIGHTS)
        if set(value) != required or any(weight < 0 for weight in value.values()) or sum(value.values()) <= 0:
            raise ValueError("策略必须包含五个非负因子组且权重之和大于 0")
        total = sum(value.values())
        return {key: weight / total for key, weight in value.items()}

    @field_validator("rebalance_frequency")
    @classmethod
    def valid_frequency(cls, value: str) -> str:
        if value not in {"weekly", "monthly", "quarterly"}:
            raise ValueError("调仓周期仅支持 weekly/monthly/quarterly")
        return value

    @field_validator("universe")
    @classmethod
    def valid_strategy_universe(cls, value: str) -> str:
        if value not in {"a_share", "large_cap", "pink_sheets", "all_assets"} and not value.startswith("custom:"):
            raise ValueError("策略股票池仅支持大盘股、A股、Pink Sheets 或自定义股票池")
        return value

    @model_validator(mode="after")
    def valid_portfolio_constraints(self):
        if self.max_weight * self.holdings_count < 1:
            raise ValueError("持仓数量与单股最大权重无法构建满仓组合")
        if self.research_start_date >= self.research_end_date:
            raise ValueError("策略研究结束日期必须晚于开始日期")
        if self.cash_buffer + self.risk_off_exposure > 1:
            raise ValueError("现金缓冲与风险关闭时的组合暴露之和不能超过 100%")
        if self.drawdown_brake_exposure > 1 - self.cash_buffer:
            raise ValueError("回撤刹车暴露不能高于扣除现金缓冲后的正常暴露")
        return self


class BacktestRequest(BaseModel):
    strategy_id: int = 1
    universe: str = "a_share"
    custom_symbols: List[str] = Field(default_factory=list)
    start_date: date = date(2018, 1, 1)
    end_date: date = date(2025, 12, 31)
    initial_capital: float = Field(1_000_000, gt=0)
    rebalance_frequency: Optional[str] = None
    holdings_count: Optional[int] = Field(None, ge=10, le=100)
    max_weight: Optional[float] = Field(None, gt=0, le=1)
    commission: float = Field(.001, ge=0, le=.05)
    slippage: float = Field(.0005, ge=0, le=.05)

    @field_validator("universe")
    @classmethod
    def valid_universe(cls, value: str) -> str:
        if value not in {"a_share", "large_cap", "hs300", "csi_a500", "pink_sheets", "all_assets", "custom"} and not value.startswith("custom:"):
            raise ValueError("股票池仅支持 a_share/large_cap/hs300/csi_a500/pink_sheets/all_assets/custom")
        return value


class PaperRebalanceRequest(BaseModel):
    # When omitted, the Paper Broker reuses the strategy currently attached
    # to the account.  This is important after applying a custom backtest:
    # the account's frozen Top-N and manually selected universe must remain
    # the default execution scope for subsequent rebalance calls.
    strategy_id: Optional[int] = None
    as_of: Optional[date] = None


class ApplyBacktestPaperRequest(BaseModel):
    reset_account: bool = False
    execute: bool = True
    enable_automation: bool = False
    initial_capital: Optional[float] = Field(None, gt=0)


class PaperAutomationPayload(BaseModel):
    strategy_id: int = 1
    enabled: bool = False
    frequency: str = "monthly"

    @field_validator("frequency")
    @classmethod
    def valid_frequency(cls, value: str) -> str:
        if value not in {"weekly", "monthly", "quarterly"}:
            raise ValueError("自动调仓周期仅支持 weekly/monthly/quarterly")
        return value


class PaperResetRequest(BaseModel):
    initial_capital: float = Field(1_000_000, gt=0)


class ExplainRequest(BaseModel):
    symbol: str
    action: str = "HOLD"
    score: float = 0
    rank: Optional[int] = None
    industry: str = ""
    volatility: float = 0
    max_drawdown: float = 0
    target_weight: float = 0
    use_llm: bool = False


class SyncRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    start_date: date = date(2025, 1, 1)
    end_date: date = Field(default_factory=date.today)
