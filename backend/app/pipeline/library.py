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


# ---- 内置策略（ADR-0047 第 9 条第一批）----

_ENSEMBLE = {"type": "model", "models": ["legacy/csi300_lightgbm", "legacy/csi300_xgboost"]}
_PLAYBOOK = "来源：QuantsPlaybook（hugo2046，经作者同意使用，见仓库 NOTICE）。"
_WEEKLY_NOTE = "择时信号只在调仓日判断，研报是每天判断，所以这里每周调仓。"

BUILTIN_STRATEGIES: List[dict] = [
    {"kind": "pb_ens_rsrs", "name": "集成模型 + RSRS择时",
     "description": "沪深300集成模型选前10%等权，整体仓位由RSRS标准分择时决定（光大证券2017：18日最高价对最低价的斜率，"
                    "600日标准分高于0.7满仓、跌破−0.7空仓）。" + _WEEKLY_NOTE + _PLAYBOOK,
     "spec": {"scorer": _ENSEMBLE, "timing": {"type": "rsrs"}, "selection": {"type": "top_pct", "pct": 0.1},
              "rebalance": {"frequency": "weekly"}}},
    {"kind": "pb_ens_alligator", "name": "集成模型 + 鳄鱼线择时",
     "description": "沪深300集成模型选前10%等权，整体仓位由鳄鱼线组合择时决定（招商证券2024：鳄鱼线+AO+分形+MACD，"
                    "不含研报里的北向资金信号）。" + _WEEKLY_NOTE + _PLAYBOOK,
     "spec": {"scorer": _ENSEMBLE, "timing": {"type": "alligator"}, "selection": {"type": "top_pct", "pct": 0.1},
              "rebalance": {"frequency": "weekly"}}},
    {"kind": "pb_ens_icu_ma", "name": "集成模型 + ICU均线择时",
     "description": "沪深300集成模型选前10%等权，整体仓位由ICU均线择时决定（中泰证券2023：5日重复中位数稳健回归均线，"
                    "收盘价在均线之上满仓、之下空仓）。" + _WEEKLY_NOTE + _PLAYBOOK,
     "spec": {"scorer": _ENSEMBLE, "timing": {"type": "icu_ma"}, "selection": {"type": "top_pct", "pct": 0.1},
              "rebalance": {"frequency": "weekly"}}},
    {"kind": "pb_ubl", "name": "上下影线因子选股",
     "description": "按上下影线综合因子UBL从低到高选前30名等权，每月调仓（东吴证券2020：蜡烛上影线波动+威廉下影线均值；"
                    "研报先做市值中性化，市值数据还没接入，这里省略）。" + _PLAYBOOK,
     "spec": {"scorer": {"type": "factor_weights", "weights": {"ubl": 1.0}}, "selection": {"type": "top_n", "n": 30}}},
    {"kind": "pb_ideal_amplitude", "name": "理想振幅因子选股",
     "description": "按理想振幅因子（高价日振幅−低价日振幅，λ=20%）从低到高选前30名等权，每月调仓（开源证券2020）。" + _PLAYBOOK,
     "spec": {"scorer": {"type": "factor_weights", "weights": {"ideal_amplitude": 1.0}}, "selection": {"type": "top_n", "n": 30}}},
]


# 迁移前的内置策略描述里带着迁移前、口径不同的成绩数字（如"7年复合+611.9%"），
# 和现在回测历史里的结果对不上；统一换成只讲方法的说明，成绩以回测历史/成绩卡为准。
_LEGACY_DESCRIPTIONS = {
    "csi300_lightgbm": "沪深300成分股，用 LightGBM 旧模型（14 项价格/量能特征，预测未来 90 个交易日收益，5 个种子平均，逐年滚动训练）打分，选前 10% 等权，每月调仓，不择时。",
    "csi300_xgboost": "沪深300成分股，用 XGBoost 旧模型（与 LightGBM 旧模型同一套特征和训练规则）打分，选前 10% 等权，每月调仓，不择时。",
    "csi300_ensemble": "沪深300成分股，LightGBM 和 XGBoost 两个旧模型的预测分数取平均，选前 10% 等权，每月调仓，不择时。",
    "quant_v3_regression": "30 支跨行业候选池，用 LightGBM 旧模型打分，选前 40% 等权，每月调仓，不择时。迁移时去掉了原来的动量兜底和流动性资格两层（见 ADR-0048）。",
    "multifactor": "因子权重策略：基本面/估值/盈利质量/动量/风险五组因子按权重合成综合分，选前 10 名按分数加权（单股 ≤15%），沪深300 均线趋势择时，每月调仓。真实行情下只有动量、风险两组有真实数据。",
}


def ensure_builtin_library(db: Session) -> None:
    """登记内置策略，并给已有策略标上来源（内置 / 用户 / 模拟盘快照）。"""
    from datetime import date as _date

    for item in BUILTIN_STRATEGIES:
        if db.scalar(select(Strategy).where(Strategy.kind == item["kind"]).limit(1)):
            continue
        spec = StrategySpec.model_validate(item["spec"])
        db.add(Strategy(
            name=item["name"], kind=item["kind"], version=1, description=item["description"],
            spec=spec.model_dump(mode="json"), origin="builtin", universe="csi300",
            weights=dict(getattr(spec.scorer, "weights", {}) or {}),
            holdings_count=spec.selection.n or 30, max_weight=spec.weighting.max_weight,
            rebalance_frequency=spec.rebalance.frequency, turnover_band=spec.rebalance.turnover_band,
            trend_filter=False, stop_loss=0, target_volatility=0, max_drawdown_budget=0, cash_buffer=0,
            research_start_date=_date(2019, 1, 1), research_end_date=_date(2025, 12, 31), is_default=False,
        ))
    builtin_kinds = set(_MODEL_SPECS) | {item["kind"] for item in BUILTIN_STRATEGIES}
    rows = db.scalars(select(Strategy).order_by(Strategy.id)).all()
    first_v3 = next((r for r in rows if r.kind == "multifactor" and r.name == "沪深300增强趋势价值成长策略 V3"), None)
    for row in rows:
        if row.kind in builtin_kinds or row is first_v3:
            row.origin = "builtin"
            if row.kind in _LEGACY_DESCRIPTIONS:
                row.description = _LEGACY_DESCRIPTIONS[row.kind]
        elif "回测 #" in (row.name or ""):
            row.origin = "paper_snapshot"
        elif not row.origin:
            row.origin = "user"
    db.commit()


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
