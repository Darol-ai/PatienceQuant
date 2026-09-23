from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Industry(Base):
    __tablename__ = "industries"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    color: Mapped[str] = mapped_column(String(20), default="#31d0aa")


class ResearchGroup(Base):
    __tablename__ = "research_groups"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    description: Mapped[str] = mapped_column(String(240), default="")
    color: Mapped[str] = mapped_column(String(20), default="#31d0aa")


class Stock(Base):
    __tablename__ = "stocks"
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(80), index=True)
    exchange: Mapped[str] = mapped_column(String(20), default="A股")
    industry_id: Mapped[int] = mapped_column(ForeignKey("industries.id"))
    research_group_id: Mapped[int] = mapped_column(ForeignKey("research_groups.id"))
    sector: Mapped[str] = mapped_column(String(80))
    tags: Mapped[List[str]] = mapped_column(JSON, default=list)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    industry: Mapped[Industry] = relationship()
    research_group: Mapped[ResearchGroup] = relationship()


class Fundamental(Base):
    __tablename__ = "fundamentals"
    __table_args__ = (UniqueConstraint("symbol", "report_date", name="uq_fundamental"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    report_date: Mapped[date] = mapped_column(Date, index=True)
    roe: Mapped[Optional[float]] = mapped_column(Float)
    roa: Mapped[Optional[float]] = mapped_column(Float)
    revenue_growth: Mapped[Optional[float]] = mapped_column(Float)
    profit_growth: Mapped[Optional[float]] = mapped_column(Float)
    operating_cashflow: Mapped[Optional[float]] = mapped_column(Float)
    pe: Mapped[Optional[float]] = mapped_column(Float)
    pb: Mapped[Optional[float]] = mapped_column(Float)
    ps: Mapped[Optional[float]] = mapped_column(Float)
    dividend_yield: Mapped[Optional[float]] = mapped_column(Float)
    roe_stability: Mapped[Optional[float]] = mapped_column(Float)
    gross_margin: Mapped[Optional[float]] = mapped_column(Float)
    net_margin: Mapped[Optional[float]] = mapped_column(Float)
    cashflow_profit_ratio: Mapped[Optional[float]] = mapped_column(Float)
    data_mode: Mapped[str] = mapped_column(String(20), default="demo")


class Watchlist(Base):
    __tablename__ = "watchlists"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    symbols: Mapped[List[str]] = mapped_column(JSON, default=list)


class ResearchAnnotation(Base):
    __tablename__ = "research_annotations"
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    bucket: Mapped[str] = mapped_column(String(40), default="大盘核心")
    thesis: Mapped[str] = mapped_column(Text, default="")
    risk: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Strategy(Base):
    __tablename__ = "strategies"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    # "multifactor"（默认，原有通用因子策略）或 "quant_v3_regression"
    # （自由探索阶段的最终LightGBM量化策略，见 docs/adr/0036）——按 kind
    # 分流到不同的数据源/策略对象，不影响原有策略的行为。
    kind: Mapped[str] = mapped_column(String(30), default="multifactor")
    version: Mapped[int] = mapped_column(Integer, default=1)
    description: Mapped[str] = mapped_column(Text, default="")
    weights: Mapped[Dict[str, float]] = mapped_column(JSON)
    holdings_count: Mapped[int] = mapped_column(Integer, default=10)
    max_weight: Mapped[float] = mapped_column(Float, default=0.15)
    rebalance_frequency: Mapped[str] = mapped_column(String(20), default="monthly")
    universe: Mapped[str] = mapped_column(String(30), default="large_cap")
    research_start_date: Mapped[date] = mapped_column(Date, default=date(2018, 1, 1))
    research_end_date: Mapped[date] = mapped_column(Date, default=date(2025, 12, 31))
    # Risk controls are persisted with the strategy so the same rules are
    # used by research, backtest, paper trading, and scheduled rebalancing.
    cash_buffer: Mapped[float] = mapped_column(Float, default=.02)
    trend_filter: Mapped[bool] = mapped_column(Boolean, default=True)
    risk_off_exposure: Mapped[float] = mapped_column(Float, default=.75)
    turnover_band: Mapped[float] = mapped_column(Float, default=.03)
    stop_loss: Mapped[float] = mapped_column(Float, default=.18)
    benchmark_enhancement: Mapped[bool] = mapped_column(Boolean, default=True)
    dip_buy_strength: Mapped[float] = mapped_column(Float, default=.06)
    profit_take_strength: Mapped[float] = mapped_column(Float, default=.05)
    target_volatility: Mapped[float] = mapped_column(Float, default=.22)
    max_drawdown_budget: Mapped[float] = mapped_column(Float, default=.15)
    drawdown_brake_exposure: Mapped[float] = mapped_column(Float, default=.50)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    # 策略规格（app/pipeline/spec.py 的 StrategySpec，ADR-0047/0048）：打分、择时、
    # 选股、权重、调仓。回测和模拟盘只按它执行；上面那些旧字段只给旧表单用。
    spec: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class StrategyFactor(Base):
    __tablename__ = "strategy_factors"
    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategies.id"), index=True)
    group_name: Mapped[str] = mapped_column(String(40))
    factor_name: Mapped[str] = mapped_column(String(80))
    weight: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(12), default="positive")


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategies.id"))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    initial_capital: Mapped[float] = mapped_column(Float)
    config: Mapped[Dict[str, Any]] = mapped_column(JSON)
    data_mode: Mapped[str] = mapped_column(String(20), default="demo")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class BacktestMetric(Base):
    __tablename__ = "backtest_metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    backtest_run_id: Mapped[int] = mapped_column(ForeignKey("backtest_runs.id"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    value: Mapped[float] = mapped_column(Float)


class BacktestEquity(Base):
    __tablename__ = "backtest_equity"
    id: Mapped[int] = mapped_column(primary_key=True)
    backtest_run_id: Mapped[int] = mapped_column(ForeignKey("backtest_runs.id"), index=True)
    trade_date: Mapped[date] = mapped_column(Date)
    equity: Mapped[float] = mapped_column(Float)
    benchmark: Mapped[float] = mapped_column(Float)
    drawdown: Mapped[float] = mapped_column(Float)


class Portfolio(Base):
    __tablename__ = "portfolio"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), default="默认模拟组合")
    initial_capital: Mapped[float] = mapped_column(Float, default=1_000_000)
    cash: Mapped[float] = mapped_column(Float, default=1_000_000)
    strategy_id: Mapped[Optional[int]] = mapped_column(ForeignKey("strategies.id"), nullable=True)
    last_rebalance_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    auto_rebalance_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    auto_rebalance_frequency: Mapped[str] = mapped_column(String(20), default="monthly")
    next_rebalance_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    last_auto_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # The execution scope may come from a completed custom backtest rather
    # than the strategy's named universe.  Persist it so subsequent manual
    # and scheduled paper rebalances use the same selected symbols.
    execution_universe: Mapped[str] = mapped_column(String(50), default="large_cap")
    execution_symbols: Mapped[List[str]] = mapped_column(JSON, default=list)
    source_backtest_run_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("portfolio_id", "symbol", name="uq_position"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolio.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    quantity: Mapped[int] = mapped_column(Integer)
    avg_cost: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolio.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(10))
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    amount: Mapped[float] = mapped_column(Float)
    fee: Mapped[float] = mapped_column(Float)
    strategy_name: Mapped[str] = mapped_column(String(120))
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="filled")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Trade(Base):
    __tablename__ = "trades"
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[Optional[int]] = mapped_column(ForeignKey("portfolio.id"), nullable=True, index=True)
    backtest_run_id: Mapped[Optional[int]] = mapped_column(ForeignKey("backtest_runs.id"), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    side: Mapped[str] = mapped_column(String(10))
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    amount: Mapped[float] = mapped_column(Float)
    fee: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)


class DailyAccount(Base):
    __tablename__ = "daily_account"
    __table_args__ = (UniqueConstraint("portfolio_id", "trade_date", name="uq_daily_account"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolio.id"), index=True)
    trade_date: Mapped[date] = mapped_column(Date)
    total_assets: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    market_value: Mapped[float] = mapped_column(Float)
    daily_return: Mapped[float] = mapped_column(Float, default=0)
    cumulative_return: Mapped[float] = mapped_column(Float, default=0)


class DailyProfit(Base):
    __tablename__ = "daily_profit"
    __table_args__ = (UniqueConstraint("portfolio_id", "trade_date", name="uq_daily_profit"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolio.id"), index=True)
    trade_date: Mapped[date] = mapped_column(Date)
    realized_profit: Mapped[float] = mapped_column(Float, default=0)
    unrealized_profit: Mapped[float] = mapped_column(Float, default=0)


class Signal(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategies.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    signal_date: Mapped[date] = mapped_column(Date, index=True)
    action: Mapped[str] = mapped_column(String(10))
    score: Mapped[float] = mapped_column(Float)
    target_weight: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)


class AppSetting(Base):
    """运行时可改的配置覆盖——目前只用来存OpenAI key/base_url/model。

    这些原本只能通过.env在部署时写死、改完要重启进程才生效；有了这张表，
    前端可以直接调POST /api/settings/ai写入，立即对后续AI调用生效，不用
    重启Docker容器。key按`key`列取值，value统一存成字符串(即使本质是
    简单标量)，避免为每种设置类型单独建列。
    """
    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(String(60), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AIExplanation(Base):
    """AI 助手的审计留痕：每次调用记模型版本、输入、输出、时间、置信度、
    是否被人工采纳、是否被回滚（PRD §19.1 第 5 条）。confidence 允许为空——
    规则解释器给不出真实置信度时，宁可留空也不伪造一个数字。
    """
    __tablename__ = "ai_explanations"
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    explanation_type: Mapped[str] = mapped_column(String(40))
    content: Mapped[str] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(String(30), default="rules")
    model_version: Mapped[str] = mapped_column(String(60), default="rules")
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    context: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    adopted: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    rolled_back: Mapped[bool] = mapped_column(Boolean, default=False)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
