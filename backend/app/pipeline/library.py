"""策略库：数据库里每一行策略 ↔ 策略规格；以及回测可选的股票池。

迁移前的 5 个策略（多因子 V3、30 支候选池回归、沪深300 LightGBM/XGBoost/
集成）没有规格，第一次启动时按下面的对应关系补上并存回数据库。迁移时去掉
的东西都在 docs/adr/0048 里逐条记录。
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.service import MarketDataService
from app.db.models import Strategy
from app.pipeline.spec import StrategySpec
from app.pipeline.strategy import PipelineStrategy, build_pipeline_strategy

# 旧 kind → 规格（不含多因子，多因子按行里的参数生成）
_MODEL_SPECS: Dict[str, dict] = {
    "csi300_lightgbm": {"scorer": {"type": "model", "models": ["legacy/csi300_lightgbm"]},
                        "selection": {"type": "top_pct", "pct": 0.1}},
    "csi300_xgboost": {"scorer": {"type": "model", "models": ["legacy/csi300_xgboost"]},
                       "selection": {"type": "top_pct", "pct": 0.1}},
    "csi300_ensemble": {"scorer": {"type": "model", "models": ["legacy/csi300_lightgbm", "legacy/csi300_xgboost"]},
                        "selection": {"type": "top_pct", "pct": 0.1}},
    "quant_v3_regression": {"scorer": {"type": "model", "models": ["legacy/broad30_lightgbm"]},
                            "selection": {"type": "top_pct", "pct": 0.4}},
}

# 旧 kind → 回测默认股票池
_DEFAULT_UNIVERSE = {
    "csi300_lightgbm": "csi300",
    "csi300_xgboost": "csi300",
    "csi300_ensemble": "csi300",
    "quant_v3_regression": "broad30",
}


def spec_from_legacy_row(row: Strategy) -> StrategySpec:
    if row.kind in _MODEL_SPECS:
        return StrategySpec.model_validate(_MODEL_SPECS[row.kind])
    timing = ({"type": "index_trend", "risk_off_exposure": float(row.risk_off_exposure if row.risk_off_exposure is not None else .75)}
              if row.trend_filter else {"type": "none"})
    return StrategySpec.model_validate({
        "scorer": {"type": "factor_weights", "weights": row.weights or {}},
        "timing": timing,
        "selection": {"type": "top_n", "n": int(row.holdings_count or 10)},
        "weighting": {"type": "score", "max_weight": float(row.max_weight or 1.0)},
        "rebalance": {"frequency": row.rebalance_frequency or "monthly",
                      "turnover_band": float(row.turnover_band or 0.0)},
    })


def strategy_spec(row: Strategy) -> StrategySpec:
    if row.spec:
        return StrategySpec.model_validate(row.spec)
    return spec_from_legacy_row(row)


def is_model_strategy(row: Strategy) -> bool:
    return strategy_spec(row).scorer.type == "model"


def default_universe(row: Strategy) -> str:
    return _DEFAULT_UNIVERSE.get(row.kind or "", row.universe or "large_cap")


def ensure_specs(db: Session) -> int:
    """给还没有规格的策略补上规格，返回补了几个。"""
    rows = db.scalars(select(Strategy).where(Strategy.spec.is_(None))).all()
    for row in rows:
        row.spec = spec_from_legacy_row(row).model_dump(mode="json")
        if row.kind in _DEFAULT_UNIVERSE:
            row.universe = _DEFAULT_UNIVERSE[row.kind]
    if rows:
        db.commit()
    return len(rows)


def data_service_for(db: Session, row: Strategy) -> MarketDataService:
    """模型选股策略只在真实行情上有意义，不管 DATA_MODE 怎么设都读本地行情库。"""
    return MarketDataService(db, real_market_data=is_model_strategy(row))


def build_strategy(db: Session, row: Strategy, data: Optional[MarketDataService] = None) -> PipelineStrategy:
    return build_pipeline_strategy(strategy_spec(row), data or data_service_for(db, row))


def validate_date_range(strategy: PipelineStrategy, start: date, end: date) -> None:
    earliest, latest = strategy.earliest_start(), strategy.latest_end()
    if earliest and start < earliest:
        raise ValueError(f"该策略用到的模型最早只能给 {earliest.year} 年打分，回测开始日期不能早于 {earliest}")
    if latest and end > latest:
        raise ValueError(f"该策略用到的模型最晚只能给 {latest.year} 年打分，回测结束日期不能晚于 {latest}")


# ---- 股票池 ----

def csi300_symbols() -> List[str]:
    """沪深300 成分股（离线准备时的最新名单，不是每个历史时点的名单——
    回测早年会有幸存者偏差，这一点和迁移前一致）。"""
    from app.quant_v3.csi300_universe import csi300_stocks

    return [stock["symbol"].split(".")[0] for stock in csi300_stocks()]


def broad30_symbols() -> List[str]:
    from app.quant_v3.broad_universe import BROAD_STOCKS

    return [stock["symbol"].split(".")[0] for stock in BROAD_STOCKS]


FIXED_UNIVERSES = {
    "csi300": ("沪深300成分股", csi300_symbols,
               "沪深300 真实成分股名单（最新一期，早年回测有幸存者偏差）"),
    "broad30": ("30支跨行业候选池", broad30_symbols,
                "研究阶段手工挑选的 30 支跨行业股票"),
}


def latest_market_day(today: date) -> date:
    """本地行情库里不晚于 today 的最后一个交易日（周末/节假日/当天行情
    还没发布时落到上一个交易日）。"""
    from app.data.market_refresh import get_market_store

    days = [d for d in get_market_store().stored_days() if d <= today]
    if not days:
        raise ValueError("本地行情库为空，先补齐行情")
    return days[-1]


# 迁移前逐年回测验证过、跑赢真实沪深300指数的策略（见 csi300 策略描述）
VALIDATED_KINDS = ("csi300_lightgbm", "csi300_xgboost", "csi300_ensemble")


def migrate_paper_positions(db: Session) -> int:
    """迁移前模型策略的模拟盘持仓代码带 .SH/.SZ 后缀（读的是离线 parquet），
    现在全平台统一用不带后缀的代码。历史订单/成交记录保持原样。"""
    from app.db.models import Position

    rows = db.scalars(select(Position).where(Position.symbol.like("%.%"))).all()
    for row in rows:
        row.symbol = row.symbol.split(".")[0]
    if rows:
        db.commit()
    return len(rows)
