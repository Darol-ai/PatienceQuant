from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


DEFAULT_WEIGHTS = {"fundamental": .12, "valuation": .12, "quality": .16, "momentum": .40, "risk": .20}


class StrategyPayload(BaseModel):
    name: str = "我的因子策略"
    description: str = "以跑赢或贴近沪深300为目标，并用目标波动率与回撤预算控制组合暴露"
    weights: Dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    holdings_count: int = Field(10, ge=10, le=100)
    max_weight: float = Field(.15, gt=0, le=1)
    rebalance_frequency: str = "monthly"
    universe: str = "csi300"
    research_start_date: date = date(2018, 1, 1)
    research_end_date: date = date(2025, 12, 31)
    cash_buffer: float = Field(.02, ge=0, lt=.5)
    trend_filter: bool = True
    risk_off_exposure: float = Field(.75, gt=0, le=1)
    turnover_band: float = Field(.03, ge=0, le=.25)
    # stop_loss/target_volatility/max_drawdown_budget 允许 0——0 是"关闭
    # 这条引擎级风控"的合法取值（ADR-0037：这层叠加对LightGBM策略是净
    # 拖累，特意存的就是0），不是"忘了设置"，schema不能把它当非法输入拒绝。
    stop_loss: float = Field(.18, ge=0, le=.8)
    benchmark_enhancement: bool = True
    dip_buy_strength: float = Field(.06, ge=0, le=.5)
    profit_take_strength: float = Field(.05, ge=0, le=.5)
    target_volatility: float = Field(.22, ge=0, le=1)
    max_drawdown_budget: float = Field(.15, ge=0, le=.95)
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
        from app.pipeline.library import is_valid_universe, normalize_universe

        value = normalize_universe(value)
        if not is_valid_universe(value) or value == "custom":
            raise ValueError("策略股票池仅支持 a_share/csi300/broad30/custom:<编号>")
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


class StrategySpecPayload(BaseModel):
    """策略制定表单：名称、说明、回测时默认用的股票池，以及策略规格（app/pipeline/spec.py）。"""

    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    default_universe: str = "csi300"
    spec: Dict[str, Any]

    @field_validator("default_universe")
    @classmethod
    def valid_default_universe(cls, value: str) -> str:
        from app.pipeline.library import is_valid_universe, normalize_universe

        value = normalize_universe(value)
        if not is_valid_universe(value) or value == "custom":
            raise ValueError("默认股票池仅支持 a_share/csi300/broad30/custom:<编号>")
        return value


class BacktestRequest(BaseModel):
    strategy_id: int = 1
    # 不传时用策略自己的默认股票池（模型策略是训练时的股票池）
    universe: Optional[str] = None
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
    def valid_universe(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        from app.pipeline.library import is_valid_universe, normalize_universe

        value = normalize_universe(value)
        if not is_valid_universe(value):
            raise ValueError("股票池仅支持 a_share/csi300/broad30/custom/custom:<编号>")
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


class RecommendRequest(BaseModel):
    """智能体推荐的偏好：市场、能接受的最大回撤（0.25 表示 −25%，不填表示不限）、持仓周期。"""

    market: str = "a_share"
    max_drawdown: Optional[float] = Field(None, gt=0, le=1)
    holding: str = "any"
    question: str = Field("", max_length=1000)

    @field_validator("holding")
    @classmethod
    def valid_holding(cls, value: str) -> str:
        if value not in {"short", "medium", "long", "any"}:
            raise ValueError("持仓周期只能是 short/medium/long/any")
        return value


class AgentPrefs(BaseModel):
    market: str = "a_share"
    max_drawdown: Optional[float] = Field(None, gt=0, le=1)
    holding: str = "any"

    @field_validator("holding")
    @classmethod
    def valid_holding(cls, value: str) -> str:
        if value not in {"short", "medium", "long", "any"}:
            raise ValueError("持仓周期只能是 short/medium/long/any")
        return value


class AgentAction(BaseModel):
    type: str
    max_drawdown: Optional[float] = None
    holding: Optional[str] = None
    strategy_id: Optional[int] = None
    strategy_ids: Optional[List[int]] = None
    term: Optional[str] = None


class AgentTurnRequest(BaseModel):
    """智能体对话的一轮：当前偏好 + 一个按钮动作或一句话（二选一）。"""

    prefs: AgentPrefs = Field(default_factory=AgentPrefs)
    action: Optional[AgentAction] = None
    text: str = Field("", max_length=1000)


class ScorecardRefreshRequest(BaseModel):
    strategy_ids: Optional[List[int]] = None
    force: bool = False


class AIDecisionRequest(BaseModel):
    adopted: Optional[bool] = None
    rolled_back: Optional[bool] = None


class AISettingsRequest(BaseModel):
    """留空(None)的字段不修改，只更新显式传入的字段。"""
    api_key: Optional[str] = Field(None, min_length=1, max_length=300)
    base_url: Optional[str] = Field(None, min_length=1, max_length=300)
    model: Optional[str] = Field(None, min_length=1, max_length=100)
