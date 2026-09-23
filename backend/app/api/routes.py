from __future__ import annotations

import csv
import json
import time
from functools import lru_cache
from datetime import date, datetime
from io import StringIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai.credentials import clear_ai_credentials, get_ai_credentials, set_ai_credentials
from app.ai.report_analysis import ReportAnalysisService
from app.ai.report_extraction import extract_text
from app.backtest.engine import BacktestConfig, BacktestEngine
from app.config import get_settings
from app.data.market_refresh import market_status, start_refresh
from app.data.service import MarketDataService
from app.quant_v3.research_notes import quant_v3_research_universe
from app.db.models import AIExplanation, Industry, BacktestEquity, BacktestMetric, BacktestRun, DailyAccount, Order, Portfolio, Position, ResearchAnnotation, ResearchGroup, Stock, Strategy, StrategyFactor, Trade, Watchlist
from app.db.session import get_db
from app.portfolio.service import PaperTradingService
from app.schemas import AIDecisionRequest, AISettingsRequest, ApplyBacktestPaperRequest, BacktestRequest, PaperAutomationPayload, PaperRebalanceRequest, PaperResetRequest, StrategyPayload, StrategySpecPayload, RecommendRequest, ScorecardRefreshRequest, AgentTurnRequest, ModelStrategyPayload
from app.db.seed import DEFAULT_WEIGHTS
from app.pipeline.library import FIXED_UNIVERSES, Pool, fixed_pool, normalize_universe, VALIDATED_KINDS, build_strategy, spec_from_legacy_row, data_service_for, default_universe, is_model_strategy, latest_market_day, strategy_spec, validate_date_range
from app.pipeline.model_library import MODEL_LIBRARY, model_years
from app.pipeline.price_factors import PRICE_FACTORS
from app.pipeline.scorers import FUNDAMENTAL_FACTOR_GROUPS
from app.pipeline.spec import StrategySpec
from pydantic import BaseModel, Field, ValidationError
from app.pipeline.strategy import build_pipeline_strategy
from app.strategies.base import StrategyConfig, StrategyResult


router = APIRouter()


RESEARCH_ANNOTATIONS: Dict[str, Dict[str, str]] = {
    "600519": {"bucket": "大盘核心", "thesis": "品牌壁垒与高 ROE，观察现金流和估值中枢", "risk": "消费复苏与估值收缩"},
    "000333": {"bucket": "大盘核心", "thesis": "制造效率、全球化与稳定现金流", "risk": "海外需求和原材料波动"},
    "600036": {"bucket": "大盘核心", "thesis": "零售银行质量与低频价值重估", "risk": "净息差和资产质量"},
    "300750": {"bucket": "大盘核心", "thesis": "动力电池龙头，跟踪全球份额与储能增量", "risk": "价格战和技术路线变化"},
    "002594": {"bucket": "大盘核心", "thesis": "整车、电池和出海协同增长", "risk": "竞争加剧和资本开支"},
    "601318": {"bucket": "大盘核心", "thesis": "保险价值修复与综合金融协同", "risk": "权益市场和新业务价值"},
    "688981": {"bucket": "大盘核心", "thesis": "国产晶圆制造核心资产，关注产能利用率", "risk": "周期、制程和外部限制"},
    "601088": {"bucket": "大盘核心", "thesis": "高股息、低估值与一体化能源现金流", "risk": "煤价下行和周期波动"},
    "600900": {"bucket": "大盘核心", "thesis": "水电现金流稳定，适合作为低频防守底仓", "risk": "来水和利率变化"},
    "601006": {"bucket": "大盘核心", "thesis": "铁路资产、稳定分红与低波动特征", "risk": "运量和运价变化"},
}


def _clean(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    if frame.empty:
        return []
    return [_clean(row) for row in frame.to_dict(orient="records")]


def _latest_selection(ranking: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    """Return the final rebalance's Top-N target portfolio for product display."""
    if ranking.empty:
        return []
    frame = ranking.copy()
    if "signal_date" in frame.columns:
        frame = frame[frame.signal_date == frame.signal_date.max()]
    frame = frame[frame.target_weight > 0].sort_values("rank").head(limit)
    columns = [
        column
        for column in [
            "symbol",
            "name",
            "exchange",
            "group",
            "sector",
            "industry",
            "rank",
            "score",
            "target_weight",
            "action",
            "signal_date",
            "market_regime",
            "target_exposure",
        ]
        if column in frame.columns
    ]
    return _records(frame[columns])


def _default_multifactor_strategy(db: Session) -> Optional[Strategy]:
    """展示型端点(/api/stocks、/api/research/coverage、/api/signals、
    /api/analytics/universe、Dashboard的analytics兜底)要的是一个因子权重
    策略——`Strategy.is_default` 可能落在模型选股策略上(比如csi300_ensemble，
    见set_default_strategy)，那类策略没有"因子"可以展示。
    这里优先找同时是is_default又是multifactor的，找不到就退到随便一个
    multifactor策略——不能因为平台默认换成了训练好的模型，这些通用
    展示页面就跟着悄悄坏掉。"""
    strategy = db.scalar(select(Strategy).where(Strategy.is_default.is_(True), Strategy.kind == "multifactor").limit(1))
    if strategy:
        return strategy
    return db.scalar(select(Strategy).where(Strategy.kind == "multifactor").order_by(Strategy.created_at.desc()).limit(1))


def _factor_ranking(db: Session, data: MarketDataService, symbols: List[str], as_of: date) -> "StrategyResult":
    """展示型端点用默认因子权重策略给一批股票打分排名。"""
    row = _default_multifactor_strategy(db)
    spec = strategy_spec(row) if row else StrategySpec.model_validate(
        {"scorer": {"type": "factor_weights", "weights": dict(DEFAULT_WEIGHTS)}, "selection": {"type": "top_n", "n": 10}})
    strategy = build_pipeline_strategy(spec, data)
    strategy.prepare(symbols, as_of, as_of)
    return strategy.generate_weights(as_of, symbols)


def _spec_for_run(strategy: Strategy, request: BacktestRequest) -> StrategySpec:
    """这次回测实际执行的规格。回测表单上的持仓数量/单股上限/调仓频率
    （旧表单遗留）只覆盖这一次，不改策略本身。"""
    spec = strategy_spec(strategy).model_copy(deep=True)
    if request.holdings_count and spec.selection.type == "top_n":
        spec.selection.n = request.holdings_count
    if request.max_weight:
        spec.weighting.max_weight = request.max_weight
    if request.rebalance_frequency:
        spec.rebalance.frequency = request.rebalance_frequency
    return spec


def _engine_config(spec: StrategySpec, universe_size: int) -> StrategyConfig:
    """回测引擎只负责④调仓：换手阈值 + 按手取整。引擎里旧的止损/目标波动率/
    回撤刹车叠加层不属于流水线（ADR-0048），全部关闭。"""
    holdings = spec.selection.n if spec.selection.type == "top_n" else max(1, round(universe_size * spec.selection.pct))
    return StrategyConfig(
        holdings_count=holdings, max_weight=spec.weighting.max_weight,
        rebalance_frequency=spec.rebalance.frequency, turnover_band=spec.rebalance.turnover_band,
        cash_buffer=0, trend_filter=spec.timing.type != "none", stop_loss=0,
        target_volatility=0, max_drawdown_budget=0, drawdown_brake_exposure=1.0,
        benchmark_enhancement=False, dip_buy_strength=0, profit_take_strength=0,
    )


def _spec_summary(spec: StrategySpec) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "spec": spec.model_dump(mode="json"),
        "scorer": spec.scorer.type,
        "rebalance_frequency": spec.rebalance.frequency,
        "turnover_band": spec.rebalance.turnover_band,
        "max_weight": spec.weighting.max_weight,
        "weighting": spec.weighting.type,
        "trend_filter": spec.timing.type == "index_trend",
        "risk_off_exposure": getattr(spec.timing, "risk_off_exposure", 1.0),
    }
    if spec.scorer.type == "factor_weights":
        summary["weights"] = spec.scorer.weights
    else:
        summary["models"] = spec.scorer.models
    if spec.selection.type == "top_n":
        summary["holdings_count"] = spec.selection.n
    else:
        summary["top_pct"] = spec.selection.pct
    return summary


def _strategy_symbols(data: MarketDataService, strategy: Strategy) -> List[str]:
    """模拟盘/看板用的股票池：策略自己记的默认股票池。"""
    universe = default_universe(strategy)
    if universe in FIXED_UNIVERSES:
        return FIXED_UNIVERSES[universe][1]()
    return _universe_symbols(data, universe)


def _backtest_pool(data: MarketDataService, payload: BacktestRequest) -> Pool:
    """这次回测的股票池。沪深300 按历史成分股逐日变化（ADR-0052），其余股票池名单固定。"""
    if payload.universe in FIXED_UNIVERSES:
        return fixed_pool(payload.universe, payload.start_date, payload.end_date)
    if payload.universe == "custom":
        catalog_sync = data.ensure_symbols(payload.custom_symbols)
        if catalog_sync.get("missing"):
            raise HTTPException(422, "以下股票代码无法从本地目录或真实数据源解析: %s" % ", ".join(catalog_sync["missing"]))
    return Pool(_universe_symbols(data, payload.universe, payload.custom_symbols))


def _universe_symbols(data: MarketDataService, universe: str, custom_symbols: Optional[List[str]] = None) -> List[str]:
    universe = normalize_universe(universe)
    if universe in FIXED_UNIVERSES:
        return FIXED_UNIVERSES[universe][1]()
    asset_symbols = set(data.stocks().symbol.tolist())
    if universe.startswith("custom:"):
        try:
            watchlist = data.db.get(Watchlist, int(universe.split(":", 1)[1]))
        except (TypeError, ValueError):
            watchlist = None
        if not watchlist:
            raise HTTPException(404, "自定义股票池不存在")
        symbols = [symbol for symbol in watchlist.symbols if symbol in asset_symbols]
        if len(symbols) < 10:
            raise HTTPException(422, "自定义股票池至少需要 10 只有效股票")
        return symbols
    if universe == "custom":
        requested = list(dict.fromkeys(
            str(symbol).strip() for symbol in (custom_symbols or []) if str(symbol).strip() in asset_symbols
        ))
        if len(requested) < 10:
            raise HTTPException(422, "自定义股票池至少需要 10 只有效股票")
        return requested
    return data.universe_symbols("a_share")


@router.get("/health")
def health() -> Dict[str, Any]:
    settings = get_settings()
    return {"status": "ok", "app": settings.app_name, "data_mode": settings.data_mode, "as_of": settings.demo_as_of.isoformat(), "python": ">=3.9"}


@router.get("/research/quant-v3-universe")
def quant_v3_research_coverage() -> Dict[str, Any]:
    """本组自由探索阶段最终确定的LightGBM策略实际使用的30支候选池研究
    标注（docs/adr/0013~0039）——独立于SQLite通用目录和/api/research/coverage
    那套多因子排序展示，直接返回本组真实的研究依据，不是脚手架自带的
    demo文案。"""
    rows = quant_v3_research_universe()
    sectors = {}
    for row in rows:
        sectors.setdefault(row["group"], []).append(row["symbol"])
    return {
        "items": rows,
        "total": len(rows),
        "groups": [{"name": name, "count": len(symbols), "symbols": symbols} for name, symbols in sectors.items()],
        "industries": sorted({row["industry"] for row in rows}),
        "research_window": {"start": "2019-01-01", "end": "2025-12-31"},
        "strategy_summary": "LightGBM回归预测(90日窗口，5模型集成) + Top-K相对排序 + 动量兜底 + 流动性资格判断，2019-2025历史回测6/7年跑赢等权重买入持有基准，详见 docs/adr/0013~0039",
    }


class PoolPayload(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    symbols: List[str]


def _clean_pool_symbols(db: Session, symbols: List[str]) -> List[str]:
    data = MarketDataService(db)
    requested = list(dict.fromkeys(str(s).strip() for s in symbols if str(s).strip()))
    missing = data.ensure_symbols(requested).get("missing", [])
    if missing:
        raise HTTPException(422, "以下股票代码在股票目录里找不到：%s" % "、".join(missing[:20]))
    if len(requested) < 10:
        raise HTTPException(422, "自定义股票池至少需要 10 只股票")
    return requested


@router.get("/pools")
def list_pools(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """股票池管理（ADR-0051）：固定股票池 + A股全市场 + 用户自定义股票池。"""
    data = MarketDataService(db)
    pools = [{"id": key, "name": name, "description": description, "kind": "fixed", "count": len(fn())}
             for key, (name, fn, description) in FIXED_UNIVERSES.items()]
    pools.append({"id": "a_share", "name": "A股全市场", "description": "通用股票目录里的全部 A 股", "kind": "fixed",
                  "count": len(data.universe_symbols("a_share"))})
    pools += [{"id": "custom:%s" % row.id, "name": row.name, "description": "自定义股票池", "kind": "custom", "count": len(row.symbols)}
              for row in db.scalars(select(Watchlist).order_by(Watchlist.id)).all()]
    return pools


@router.get("/pools/{pool_id}/members")
def pool_members(pool_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    data = MarketDataService(db)
    pool_id = normalize_universe(pool_id)
    if pool_id in FIXED_UNIVERSES or pool_id == "a_share":
        symbols = data.universe_symbols(pool_id)
    elif pool_id.startswith("custom:"):
        row = db.get(Watchlist, int(pool_id.split(":", 1)[1])) if pool_id.split(":", 1)[1].isdigit() else None
        if not row:
            raise HTTPException(404, "股票池不存在")
        symbols = list(row.symbols)
    else:
        raise HTTPException(404, "股票池不存在")
    catalog = data.stocks().set_index("symbol") if not data.stocks().empty else pd.DataFrame()
    names = _fixed_universe_names()
    members = []
    for symbol in symbols:
        row = catalog.loc[symbol] if symbol in catalog.index else None
        members.append({"symbol": symbol,
                        "name": (row["name"] if row is not None else names.get(symbol)) or symbol,
                        "industry": row["industry"] if row is not None else None})
    return {"id": pool_id, "count": len(members), "members": members}


@router.post("/pools")
def create_pool(payload: PoolPayload, db: Session = Depends(get_db)) -> Dict[str, Any]:
    name = payload.name.strip()
    if db.scalar(select(Watchlist).where(Watchlist.name == name)):
        raise HTTPException(409, "股票池名称已存在")
    row = Watchlist(name=name, symbols=_clean_pool_symbols(db, payload.symbols))
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"id": "custom:%s" % row.id, "name": row.name, "count": len(row.symbols)}


def _pool_users(db: Session, pool_id: str) -> List[str]:
    users = [f"策略「{row.name}」" for row in db.scalars(select(Strategy)).all() if row.universe == pool_id]
    portfolio = db.scalar(select(Portfolio).limit(1))
    if portfolio and portfolio.execution_universe == pool_id:
        users.append("模拟盘")
    return users


@router.put("/pools/{pool_id}")
def update_pool(pool_id: str, payload: PoolPayload, db: Session = Depends(get_db)) -> Dict[str, Any]:
    row = db.get(Watchlist, int(pool_id.split(":", 1)[1])) if pool_id.startswith("custom:") and pool_id.split(":", 1)[1].isdigit() else None
    if not row:
        raise HTTPException(404, "只能修改自定义股票池")
    clash = db.scalar(select(Watchlist).where(Watchlist.name == payload.name.strip(), Watchlist.id != row.id))
    if clash:
        raise HTTPException(409, "股票池名称已存在")
    row.name = payload.name.strip()
    row.symbols = _clean_pool_symbols(db, payload.symbols)
    db.commit()
    return {"id": pool_id, "name": row.name, "count": len(row.symbols)}


@router.delete("/pools/{pool_id}")
def delete_pool(pool_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    row = db.get(Watchlist, int(pool_id.split(":", 1)[1])) if pool_id.startswith("custom:") and pool_id.split(":", 1)[1].isdigit() else None
    if not row:
        raise HTTPException(404, "只能删除自定义股票池")
    users = _pool_users(db, pool_id)
    if users:
        raise HTTPException(409, "这个股票池正在被使用：%s" % "、".join(users))
    db.delete(row)
    db.commit()
    return {"deleted": pool_id}


@router.get("/stocks/search")
def search_stocks(
    q: str = "",
    search: str = "",
    group: str = "",
    industry: str = "",
    exchange: str = "",
    source: str = "auto",
    limit: Optional[int] = None,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Fast stock-directory search for manual portfolio construction.

    Unlike ``/stocks``, this endpoint does not run the factor engine across the
    entire universe. ``source=tushare`` queries the tushare code/name table and
    caches any newly discovered symbols in SQLite so they can immediately be
    selected for a custom backtest.
    """
    data = MarketDataService(db)
    query = q or search
    frame, resolved_source = data.search_catalog(
        query=query,
        limit=limit,
        source=source,
        group=group,
        industry=industry,
        exchange=exchange,
    )
    if limit is not None:
        frame = frame.head(max(1, min(limit, 200)))
    return {
        "items": _records(frame),
        "total": len(frame),
        "query": query,
        "requested_source": source,
        "source": resolved_source,
        "data_mode": data.mode,
        "real_directory_enabled": bool(data.real.catalog_cached or resolved_source == "tushare"),
        "as_of": get_settings().demo_as_of.isoformat(),
    }


@router.get("/stocks/{symbol}")
def stock_detail(symbol: str, range: str = "all", interval: str = "monthly", db: Session = Depends(get_db)) -> Dict[str, Any]:
    data = MarketDataService(db)
    catalog = data.stocks()
    if symbol not in set(catalog.symbol):
        raise HTTPException(404, "股票不存在")
    ranking_symbols = data.analysis_symbols(catalog.symbol.tolist())
    if symbol not in ranking_symbols:
        ranking_symbols.append(symbol)
    result = _factor_ranking(db, data, ranking_symbols, get_settings().demo_as_of)
    row = result.ranking[result.ranking.symbol == symbol].iloc[0].to_dict()
    end_date = get_settings().demo_as_of
    range_years = {"1y": 1, "3y": 3, "5y": 5}
    start_date = date(2018, 1, 1) if range == "all" else max(date(2018, 1, 1), date(end_date.year - range_years.get(range, 3), end_date.month, min(end_date.day, 28)))
    prices = data.prices([symbol], start_date, end_date, allow_network=True)
    price_data_mode = data.frame_data_mode(prices)
    if prices.empty:
        return {"stock": _clean(row), "prices": [], "data_mode": data.mode, "chart": {"range": range, "interval": interval}}
    prices = prices[["trade_date", "adj_close", "volume"]].copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    if interval in {"weekly", "monthly"}:
        prices["period"] = prices["trade_date"].dt.to_period("W-FRI" if interval == "weekly" else "M")
        prices = prices.groupby("period", as_index=False).agg({"trade_date": "max", "adj_close": "last", "volume": "sum"}).drop(columns="period")
    prices["trade_date"] = prices["trade_date"].dt.date
    first_price = float(prices.adj_close.iloc[0])
    last_price = float(prices.adj_close.iloc[-1])
    return {"stock": _clean(row), "prices": _records(prices), "data_mode": price_data_mode,
            "chart": {"range": range, "interval": interval, "start_date": prices.trade_date.iloc[0], "end_date": prices.trade_date.iloc[-1],
                      "return": last_price / first_price - 1 if first_price else 0, "high": float(prices.adj_close.max()), "low": float(prices.adj_close.min()), "points": len(prices)}}


@router.get("/strategies")
def list_strategies(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    # Product screens should open on the active production/default strategy,
    # while backtest-derived paper snapshots remain available underneath.
    rows = db.scalars(select(Strategy).order_by(Strategy.is_default.desc(), Strategy.created_at.desc())).all()
    result = []
    for row in rows:
        latest = db.scalar(select(BacktestRun).where(BacktestRun.strategy_id == row.id, BacktestRun.status == "completed").order_by(BacktestRun.id.desc()).limit(1))
        metrics = {}
        if latest:
            metrics = {item.name: item.value for item in db.scalars(select(BacktestMetric).where(BacktestMetric.backtest_run_id == latest.id)).all()}
        research_start = row.research_start_date or date(2018, 1, 1)
        research_end = row.research_end_date or date(2025, 12, 31)
        result.append({"id": row.id, "name": row.name, "kind": row.kind or "multifactor", "version": row.version, "description": row.description, "weights": row.weights, "holdings_count": row.holdings_count, "max_weight": row.max_weight, "rebalance_frequency": row.rebalance_frequency, "universe": row.universe or "csi300", "research_start_date": research_start.isoformat(), "research_end_date": research_end.isoformat(),
                       "cash_buffer": float(row.cash_buffer if row.cash_buffer is not None else .02),
                       "trend_filter": bool(row.trend_filter if row.trend_filter is not None else True),
                       "risk_off_exposure": float(row.risk_off_exposure if row.risk_off_exposure is not None else .75),
                       "turnover_band": float(row.turnover_band if row.turnover_band is not None else .03),
                       "stop_loss": float(row.stop_loss if row.stop_loss is not None else .18),
                       "benchmark_enhancement": bool(row.benchmark_enhancement if row.benchmark_enhancement is not None else True),
                       "dip_buy_strength": float(row.dip_buy_strength if row.dip_buy_strength is not None else .06),
                       "profit_take_strength": float(row.profit_take_strength if row.profit_take_strength is not None else .05),
                       "target_volatility": float(row.target_volatility if row.target_volatility is not None else .22),
                       "max_drawdown_budget": float(row.max_drawdown_budget if row.max_drawdown_budget is not None else .15),
                       "drawdown_brake_exposure": float(row.drawdown_brake_exposure if row.drawdown_brake_exposure is not None else .50),
                       "is_default": row.is_default, "created_at": row.created_at.isoformat(),
                       "validated": row.kind in VALIDATED_KINDS,
                       "spec": strategy_spec(row).model_dump(mode="json"),
                       "origin": row.origin or "user",
                       "default_universe": default_universe(row),
                       "study_period": {"start": latest.start_date.isoformat(), "end": latest.end_date.isoformat()} if latest else {"start": research_start.isoformat(), "end": research_end.isoformat()},
                       "backtest_run_id": latest.id if latest else None, "backtest_metrics": metrics})
    return result


@router.post("/strategies")
def create_strategy(payload: StrategyPayload, db: Session = Depends(get_db)) -> Dict[str, Any]:
    latest = db.scalar(select(Strategy).where(Strategy.name == payload.name).order_by(Strategy.version.desc()).limit(1))
    db.query(Strategy).filter(Strategy.is_default.is_(True)).update({"is_default": False}, synchronize_session=False)
    strategy = Strategy(
        name=payload.name,
        version=(latest.version + 1 if latest else 1),
        description=payload.description,
        weights=payload.weights,
        holdings_count=payload.holdings_count,
        max_weight=payload.max_weight,
        rebalance_frequency=payload.rebalance_frequency,
        universe=payload.universe,
        research_start_date=payload.research_start_date,
        research_end_date=payload.research_end_date,
        cash_buffer=payload.cash_buffer,
        trend_filter=payload.trend_filter,
        risk_off_exposure=payload.risk_off_exposure,
        turnover_band=payload.turnover_band,
        stop_loss=payload.stop_loss,
        benchmark_enhancement=payload.benchmark_enhancement,
        dip_buy_strength=payload.dip_buy_strength,
        profit_take_strength=payload.profit_take_strength,
        target_volatility=payload.target_volatility,
        max_drawdown_budget=payload.max_drawdown_budget,
        drawdown_brake_exposure=payload.drawdown_brake_exposure,
        is_default=True,
    )
    strategy.kind = "multifactor"
    strategy.origin = "user"
    strategy.spec = spec_from_legacy_row(strategy).model_dump(mode="json")
    db.add(strategy)
    db.commit()
    db.refresh(strategy)
    return {"id": strategy.id, "version": strategy.version, "message": "策略版本已保存"}


@router.get("/pipeline/options")
def pipeline_options(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """策略制定表单的选项：每个插槽能选什么、各因子在当前数据模式下有没有真实数据。"""
    real = get_settings().data_mode.lower() == "real"
    group_labels = {"fundamental": ("基本面", "ROE · 成长 · 现金流"), "valuation": ("估值", "PE · PB · PS · 股息率"),
                    "quality": ("盈利质量", "稳定性 · 毛利 · 净利"), "momentum": ("动量", "3/6/12 个月收益"),
                    "risk": ("风险", "波动 · 回撤 · Beta，越低越好")}
    factors = [{"key": key, "label": label, "description": hint, "kind": "group",
                "available": not (real and key in FUNDAMENTAL_FACTOR_GROUPS),
                "unavailable_reason": "需要财务数据，本地行情库还没有" if real and key in FUNDAMENTAL_FACTOR_GROUPS else None}
               for key, (label, hint) in group_labels.items()]
    factors += [{"key": f.key, "label": f.label, "description": f.description, "kind": "price", "available": True,
                 "unavailable_reason": None} for f in PRICE_FACTORS.values()]
    from app.pipeline.factor_library import FACTOR_GROUPS, FACTOR_LIBRARY, LEGACY_MODEL_FACTORS, trainable
    from app.pipeline.model_library import all_models

    models = [m for m in all_models() if m["status"] == "ready"]
    training_factors = [{"key": f.key, "label": f.label, "group": f.group, "description": f.description}
                        for f in FACTOR_LIBRARY.values() if trainable(f.key)]
    training_defaults = {"framework": "lightgbm", "factors": LEGACY_MODEL_FACTORS, "horizon": 90, "pool": "csi300",
                         "membership": "historical", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15,
                         "max_depth": 4, "seeds": 5, "cs_rank": False}
    timings = [
        {"type": "none", "label": "不择时（满仓）", "params": []},
        {"type": "index_trend", "label": "沪深300 均线趋势", "description": "跌破 50 日均线降仓，跌破 200 日均线降到设定仓位",
         "params": [{"key": "risk_off_exposure", "label": "最低仓位", "default": 0.75, "min": 0, "max": 1, "step": 0.05, "percent": True}]},
        {"type": "rsrs", "label": "RSRS 择时", "description": "光大证券 2017：阻力支撑相对强度标准分",
         "params": [{"key": "n", "label": "斜率窗口 N", "default": 18, "min": 5, "max": 60, "step": 1},
                    {"key": "m", "label": "标准分窗口 M", "default": 600, "min": 100, "max": 1200, "step": 50},
                    {"key": "threshold", "label": "阈值 S", "default": 0.7, "min": 0.1, "max": 3, "step": 0.1}]},
        {"type": "icu_ma", "label": "ICU 均线择时", "description": "中泰证券 2023：稳健回归均线",
         "params": [{"key": "n", "label": "均线窗口", "default": 5, "min": 3, "max": 250, "step": 1}]},
        {"type": "alligator", "label": "鳄鱼线择时", "description": "招商证券 2024：鳄鱼线 + AO + 分形 + MACD", "params": []},
    ]
    universes = [{"id": key, "name": name, "description": description} for key, (name, _, description) in FIXED_UNIVERSES.items()]
    universes += [{"id": "a_share", "name": "A股全市场", "description": "通用目录全部 A 股"}]
    universes += [{"id": "custom:%s" % row.id, "name": row.name, "description": "自定义股票池"}
                  for row in db.scalars(select(Watchlist).order_by(Watchlist.id)).all()]
    return {"data_mode": get_settings().data_mode, "factors": factors, "models": models, "timings": timings,
            "training_factors": training_factors, "training_defaults": training_defaults, "factor_groups": FACTOR_GROUPS,
            "universes": universes, "markets": [{"id": "a_share", "name": "A股"}]}


@router.get("/factors")
def factor_library(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """因子库（ADR-0052）：每个因子的说明，以及哪些策略、模型在用它。"""
    from app.pipeline.factor_library import FACTOR_GROUPS, FACTOR_LIBRARY, trainable
    from app.pipeline.model_library import all_models

    used_by: Dict[str, List[str]] = {key: [] for key in FACTOR_LIBRARY}
    for row in db.scalars(select(Strategy)).all():
        if (row.origin or "user") == "paper_snapshot":
            continue
        spec = strategy_spec(row)
        if spec.scorer.type == "factor_weights":
            keys = [k for group, weight in spec.scorer.weights.items() if weight > 0
                    for k in (FACTOR_GROUPS.get(group) or [group])]
        else:
            from app.pipeline.model_library import model_factors
            keys = [k for m in spec.scorer.models for k in model_factors(m)]
        for key in dict.fromkeys(keys):
            used_by.setdefault(key, []).append(f"策略「{row.name}」")
    for model in all_models():
        for key in model.get("factors", []):
            used_by.setdefault(key, []).append(f"模型「{model['name']}」")
    real = get_settings().data_mode.lower() == "real"
    return [{"key": f.key, "label": f.label, "group": f.group, "direction": f.direction, "description": f.description,
             "requires": f.requires, "trainable": trainable(f.key),
             "available": f.requires == "daily" or not real, "used_by": used_by.get(f.key, [])}
            for f in FACTOR_LIBRARY.values()]


@router.get("/models")
def list_models() -> List[Dict[str, Any]]:
    from app.pipeline.model_library import all_models
    from app.training import jobs

    running = jobs.current()
    return _clean([{**m, "running": m["id"] == running} for m in all_models()])


@router.get("/models/{model_id:path}/detail")
def model_detail(model_id: str) -> Dict[str, Any]:
    from app.pipeline.model_library import all_models

    model = next((m for m in all_models() if m["id"] == model_id), None)
    if model is None:
        raise HTTPException(404, "模型不存在")
    return _clean(model)


@router.post("/models/{model_id:path}/cancel")
def cancel_model(model_id: str) -> Dict[str, Any]:
    from app.training import jobs

    if not jobs.cancel(model_id):
        raise HTTPException(422, "这个模型不在排队或训练中")
    return {"cancelled": model_id}


@router.post("/strategies/model")
def create_model_strategy(payload: ModelStrategyPayload, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """制定模型选股策略（ADR-0052）：按训练设置找已有的同设置模型，没有就排队训练一个新模型；
    策略立即保存，模型训练好后自动算成绩卡。"""
    from app.pipeline.library import FIXED_UNIVERSES as _pools
    from app.training import jobs
    from app.training.trainer import TrainingConfig, find_by_fingerprint, write_record

    try:
        config = TrainingConfig(**payload.training).validate()
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    fingerprint = config.fingerprint()
    existing = find_by_fingerprint(fingerprint)
    if existing:
        model_id, reused = existing["id"], True
    else:
        model_id, reused = f"trained/{fingerprint}", False
        pool_label = _pools[config.pool][0] if config.pool in _pools else config.pool
        write_record({
            "id": model_id, "fingerprint": fingerprint, "status": "queued", "created_at": datetime.now().isoformat(timespec="seconds"),
            "name": f"{'LightGBM' if config.framework == 'lightgbm' else 'XGBoost'} · {pool_label} · {len(config.factors)} 个因子 · {config.horizon} 日",
            "pool_label": pool_label + ("（今天的名单）" if config.pool == "csi300" and config.membership == "latest" else ""),
            "description": f"为策略「{payload.name}」训练", "config": config.__dict__,
        })
    try:
        spec = StrategySpec.model_validate({**payload.spec, "scorer": {"type": "model", "models": [model_id]}})
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    latest = db.scalar(select(Strategy).where(Strategy.name == payload.name).order_by(Strategy.version.desc()).limit(1))
    strategy = Strategy(
        name=payload.name, version=(latest.version + 1 if latest else 1), description=payload.description,
        kind="custom", origin="user", spec=spec.model_dump(mode="json"), universe=payload.default_universe, weights={},
        holdings_count=spec.selection.n or 0, max_weight=spec.weighting.max_weight,
        rebalance_frequency=spec.rebalance.frequency, turnover_band=spec.rebalance.turnover_band,
        trend_filter=spec.timing.type != "none", stop_loss=0, target_volatility=0, max_drawdown_budget=0, cash_buffer=0,
        benchmark_enhancement=False, dip_buy_strength=0, profit_take_strength=0, is_default=False,
    )
    db.add(strategy)
    db.commit()
    db.refresh(strategy)
    if not reused:
        jobs.enqueue(model_id)
    elif existing.get("status") == "ready":
        from app.pipeline import scorecard
        scorecard.start_refresh([strategy.id])
    return {"id": strategy.id, "name": strategy.name, "version": strategy.version, "model_id": model_id, "reused_model": reused,
            "message": "策略已保存，" + ("沿用已有的同设置模型" if reused else "模型已排队训练，训练好后自动计算成绩卡")}


@router.post("/strategies/spec")
def create_strategy_from_spec(payload: StrategySpecPayload, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """策略制定：按规格保存一个新策略。策略保存后不再修改，改任何一项都另存为新版本（ADR-0047 第 3 条）。"""
    try:
        spec = StrategySpec.model_validate(payload.spec)
        build_pipeline_strategy(spec, MarketDataService(db, real_market_data=spec.scorer.type == "model"))
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, str(exc)) from exc
    latest = db.scalar(select(Strategy).where(Strategy.name == payload.name).order_by(Strategy.version.desc()).limit(1))
    strategy = Strategy(
        name=payload.name, version=(latest.version + 1 if latest else 1), description=payload.description,
        kind="custom", origin="user", spec=spec.model_dump(mode="json"), universe=payload.default_universe,
        weights=dict(getattr(spec.scorer, "weights", {}) or {}),
        holdings_count=spec.selection.n or 0, max_weight=spec.weighting.max_weight,
        rebalance_frequency=spec.rebalance.frequency, turnover_band=spec.rebalance.turnover_band,
        trend_filter=spec.timing.type != "none", risk_off_exposure=getattr(spec.timing, "risk_off_exposure", 1.0),
        stop_loss=0, target_volatility=0, max_drawdown_budget=0, cash_buffer=0, benchmark_enhancement=False,
        dip_buy_strength=0, profit_take_strength=0, is_default=False,
    )
    db.add(strategy)
    db.commit()
    db.refresh(strategy)
    return {"id": strategy.id, "name": strategy.name, "version": strategy.version, "message": "策略已保存到策略库"}


@router.put("/strategies/{strategy_id}")
def update_strategy(strategy_id: int, payload: StrategyPayload, db: Session = Depends(get_db)) -> Dict[str, Any]:
    strategy = db.get(Strategy, strategy_id)
    if not strategy:
        raise HTTPException(404, "策略不存在")
    if strategy.kind and strategy.kind != "multifactor":
        # 通用编辑表单/schema是给 MultiFactorStrategy 的权重、止损等参数
        # 设计的——LightGBM策略(kind="quant_v3_regression")的止损/波动率/
        # 回撤字段特意存的是0(ADR-0037：关闭引擎级风控叠加层，那层对它是
        # 净拖累)，一次无条件全量覆盖式的更新会把这些字段悄悄填回schema
        # 默认值，重新引入已经修复过的问题。这类策略的参数由代码固定
        # 管理，不通过这个通用编辑端点改。
        raise HTTPException(422, f"策略类型「{strategy.kind}」的参数由代码固定管理，不支持通过通用编辑接口修改")
    for field, value in payload.model_dump().items():
        if hasattr(strategy, field):
            setattr(strategy, field, value)
    strategy.spec = spec_from_legacy_row(strategy).model_dump(mode="json")
    db.commit()
    return {"id": strategy.id, "message": "策略已更新"}


@router.post("/strategies/{strategy_id}/set-default")
def set_default_strategy(strategy_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """把某个已有策略设为平台默认——跟create_strategy()不同，这里不新建
    版本、不改任何参数，只是把is_default标记从当前策略挪到这一个。
    is_default决定的是"没指定具体策略时，各个通用/展示型端点(策略中心
    默认加载哪个、回测中心/analytics面板没有strategy_id时的兜底)用哪个
    策略"，跟模拟盘账户实际绑定的策略(PaperTradingService单独维护)是
    两回事——账户已经在用哪个策略调仓，不受这个接口影响。"""
    strategy = db.get(Strategy, strategy_id)
    if not strategy:
        raise HTTPException(404, "策略不存在")
    db.query(Strategy).filter(Strategy.is_default.is_(True)).update({"is_default": False}, synchronize_session=False)
    strategy.is_default = True
    db.commit()
    return {"id": strategy.id, "message": f"「{strategy.name}」已设为平台默认策略"}


@router.post("/backtests")
def run_backtest(payload: BacktestRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    strategy = db.get(Strategy, payload.strategy_id)
    if not strategy:
        raise HTTPException(404, "策略不存在")
    if payload.start_date >= payload.end_date:
        raise HTTPException(422, "结束日期必须晚于开始日期")
    payload.universe = normalize_universe(payload.universe) if payload.universe else default_universe(strategy)
    spec = _spec_for_run(strategy, payload)
    run = BacktestRun(strategy_id=strategy.id, status="running", start_date=payload.start_date, end_date=payload.end_date, initial_capital=payload.initial_capital, config=payload.model_dump(mode="json"), data_mode=get_settings().data_mode)
    db.add(run)
    db.commit()
    db.refresh(run)
    try:
        market_data = data_service_for(db, strategy)
        run.data_mode = market_data.mode
        pool = _backtest_pool(market_data, payload)
        symbols = pool.symbols
        universe_size = len(pool.members_at(payload.end_date)) if pool.members_at else len(symbols)
        try:
            pipeline = build_pipeline_strategy(spec, market_data)
            pipeline.members_at = pool.members_at
            validate_date_range(pipeline, payload.start_date, payload.end_date)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        sc = _engine_config(spec, universe_size)
        result = BacktestEngine(market_data).run(
            BacktestConfig(
                start_date=payload.start_date,
                end_date=payload.end_date,
                initial_capital=payload.initial_capital,
                rebalance_frequency=spec.rebalance.frequency,
                holdings_count=sc.holdings_count,
                max_weight=spec.weighting.max_weight,
                commission=payload.commission,
                slippage=payload.slippage,
            ),
            sc,
            symbols,
            strategy=pipeline,
        )
        if pool.note:
            result.notes.insert(0, pool.note)
        if pool.members_at:
            result.notes.insert(0, "股票池按历史成分股逐日变化：区间内先后有 %d 支股票在指数里，每次调仓只从当时的成分股里选" % len(symbols))
        if market_data.last_price_fetch_failures:
            result.notes.insert(0, "本地行情库中缺少 %d 支股票在该区间的行情，已从回测中排除：%s"
                                % (len(market_data.last_price_fetch_failures), "、".join(market_data.last_price_fetch_failures[:10])))
        if result.data_mode == "real":
            # real 模式下基准就是本地行情库里的沪深300指数
            result.metrics["csi300_index_return"] = result.metrics["benchmark_return"]
            result.metrics["excess_return_vs_csi300_index"] = result.metrics["excess_return"]
        run.data_mode = result.data_mode
        selected_stocks = _latest_selection(result.ranking, sc.holdings_count)
        run.config = {
            **payload.model_dump(mode="json"),
            "risk_controls_version": 5,
            "selected_stocks": selected_stocks,
            "risk_summary": result.risk_summary or {},
            "strategy_config": _spec_summary(spec),
            "notes": result.notes,
            "universe_size": universe_size,
            "strategy_name": strategy.name,
        }
        db.add_all([BacktestMetric(backtest_run_id=run.id, name=name, value=value) for name, value in result.metrics.items()])
        for name, value in (result.risk_summary or {}).items():
            if isinstance(value, (int, float)):
                db.add(BacktestMetric(backtest_run_id=run.id, name=name, value=float(value)))
        db.bulk_insert_mappings(BacktestEquity, [{"backtest_run_id": run.id, **row} for row in result.equity.to_dict(orient="records")])
        db.bulk_insert_mappings(Trade, [{"backtest_run_id": run.id, "portfolio_id": None, **row} for row in result.trades.to_dict(orient="records")])
        run.status = "completed"
        run.completed_at = datetime.utcnow()
        db.commit()
        return _clean(_backtest_result_payload(run, db))
    except HTTPException:
        run.status = "failed"
        run.error_message = "股票池与组合约束不匹配"
        db.commit()
        raise
    except Exception as exc:
        run.status = "failed"
        run.error_message = str(exc)
        db.commit()
        raise HTTPException(500, "回测失败: %s" % exc)


def _annual_returns(equity: pd.DataFrame) -> List[Dict[str, Any]]:
    frame = equity.copy()
    frame["trade_date"] = pd.to_datetime(frame.trade_date)
    results = []
    for year, group in frame.groupby(frame.trade_date.dt.year):
        results.append({"year": int(year), "strategy": float(group.equity.iloc[-1] / group.equity.iloc[0] - 1), "benchmark": float(group.benchmark.iloc[-1] / group.benchmark.iloc[0] - 1)})
    return results


def _backtest_equity_records(equity: pd.DataFrame, initial_capital: float) -> List[Dict[str, Any]]:
    """Expose the compounded total-assets fields without changing the DB schema."""
    if equity.empty:
        return []
    frame = equity.copy()
    frame["total_assets"] = frame["equity"].astype(float)
    frame["cumulative_return"] = frame["total_assets"] / float(initial_capital) - 1
    return _records(frame)


def _chart_trade_markers(equity: pd.DataFrame, trades: pd.DataFrame) -> List[Dict[str, Any]]:
    """Attach each execution to the strategy equity value on that date."""
    if equity.empty or trades.empty:
        return []
    curve = equity.copy()
    curve["trade_date"] = pd.to_datetime(curve.trade_date).dt.date
    value_by_date = curve.set_index("trade_date").equity.to_dict()
    markers = []
    for row in trades.to_dict(orient="records"):
        trade_date = row["trade_date"]
        if isinstance(trade_date, pd.Timestamp):
            trade_date = trade_date.date()
        markers.append({
            "trade_date": trade_date.isoformat() if isinstance(trade_date, date) else str(trade_date),
            "equity": float(value_by_date.get(trade_date, 0)),
            "symbol": row["symbol"],
            "side": row["side"],
            "quantity": int(row["quantity"]),
            "price": float(row["price"]),
            "amount": float(row["amount"]),
            "reason": row["reason"],
        })
    return markers


def _with_names(db: Session, stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """模型策略的打分结果里没有目录字段（名称、行业），展示时补上。"""
    symbols = [str(stock.get("symbol", "")).split(".")[0] for stock in stocks]
    industry = dict(db.execute(select(Stock.symbol, Industry.name).join(Industry, Stock.industry_id == Industry.id)
                               .where(Stock.symbol.in_(symbols))).all()) if symbols else {}
    out = []
    for stock in stocks:
        symbol = str(stock.get("symbol", ""))
        out.append({**stock, "name": stock.get("name") or _stock_name(db, symbol) or symbol,
                    "industry": stock.get("industry") or industry.get(symbol.split(".")[0])})
    return out


def _backtest_result_payload(run: BacktestRun, db: Session) -> Dict[str, Any]:
    """一次回测的完整结果，全部从数据库读——刚跑完返回的和之后从回测历史里
    调出来的是同一份内容（ADR-0047 第 3 条：每次回测结果都保存、可随时调出）。"""
    config = run.config or {}
    metrics = {m.name: m.value for m in db.scalars(select(BacktestMetric).where(BacktestMetric.backtest_run_id == run.id)).all()}
    equity_rows = db.scalars(select(BacktestEquity).where(BacktestEquity.backtest_run_id == run.id).order_by(BacktestEquity.trade_date)).all()
    equity = pd.DataFrame([{"trade_date": r.trade_date, "equity": r.equity, "benchmark": r.benchmark, "drawdown": r.drawdown} for r in equity_rows],
                          columns=["trade_date", "equity", "benchmark", "drawdown"])
    trade_rows = db.scalars(select(Trade).where(Trade.backtest_run_id == run.id).order_by(Trade.trade_date, Trade.id)).all()
    trades = pd.DataFrame([{"id": r.id, "trade_date": r.trade_date, "symbol": r.symbol, "side": r.side, "quantity": r.quantity, "price": r.price,
                            "amount": r.amount, "fee": r.fee, "reason": r.reason} for r in trade_rows],
                          columns=["id", "trade_date", "symbol", "side", "quantity", "price", "amount", "fee", "reason"])
    initial = float(run.initial_capital)
    final_assets = float(metrics.get("final_assets", initial))
    strategy = db.get(Strategy, run.strategy_id)
    return {
        "id": run.id,
        "strategy_id": run.strategy_id,
        "status": run.status,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "start_date": run.start_date.isoformat(),
        "end_date": run.end_date.isoformat(),
        "initial_capital": initial,
        "final_assets": final_assets,
        "total_profit": float(metrics.get("total_profit", final_assets - initial)),
        "overall_return": float(metrics.get("overall_return", final_assets / initial - 1)),
        "strategy_name": config.get("strategy_name") or (strategy.name if strategy else ""),
        "metrics": metrics,
        "equity": _backtest_equity_records(equity, initial),
        "trades": _records(trades),
        "trade_markers": _chart_trade_markers(equity, trades),
        "annual_returns": _annual_returns(equity) if not equity.empty else [],
        "selected_stocks": _with_names(db, config.get("selected_stocks", [])),
        "universe_size": config.get("universe_size") or len(config.get("custom_symbols") or []),
        "custom_symbols": config.get("custom_symbols") or [],
        "notes": config.get("notes", []),
        "risk_summary": config.get("risk_summary", {}),
        "strategy_config": config.get("strategy_config", {}),
        "data_mode": run.data_mode,
        "universe": config.get("universe"),
        "commission": config.get("commission"),
        "slippage": config.get("slippage"),
        "error_message": run.error_message,
    }


_HISTORY_METRICS = ("annual_return", "total_return", "max_drawdown", "sharpe", "benchmark_return", "excess_return", "trade_count")


@router.get("/backtests")
def list_backtests(strategy_id: Optional[int] = None, limit: int = 50, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """回测历史：某个策略（或全部策略）保存下来的每一次回测，新的在前。"""
    query = select(BacktestRun).order_by(BacktestRun.id.desc()).limit(max(1, min(limit, 500)))
    if strategy_id is not None:
        query = query.where(BacktestRun.strategy_id == strategy_id)
    runs = db.scalars(query).all()
    names = {row.id: row.name for row in db.scalars(select(Strategy)).all()}
    result = []
    for run in runs:
        metrics = {m.name: m.value for m in db.scalars(select(BacktestMetric).where(
            BacktestMetric.backtest_run_id == run.id, BacktestMetric.name.in_(_HISTORY_METRICS))).all()}
        config = run.config or {}
        result.append({
            "id": run.id, "strategy_id": run.strategy_id, "strategy_name": names.get(run.strategy_id, ""),
            "status": run.status, "created_at": run.created_at.isoformat() if run.created_at else None,
            "start_date": run.start_date.isoformat(), "end_date": run.end_date.isoformat(),
            "initial_capital": run.initial_capital, "universe": config.get("universe"),
            "universe_size": config.get("universe_size"), "data_mode": run.data_mode,
            "seeded": bool(config.get("seeded")), "error_message": run.error_message, "metrics": metrics,
        })
    return result


@router.get("/backtests/{run_id}/result")
def get_backtest_result(run_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    run = db.get(BacktestRun, run_id)
    if not run:
        raise HTTPException(404, "回测不存在")
    if run.status != "completed":
        raise HTTPException(422, "这次回测没有完成：%s" % (run.error_message or run.status))
    return _clean(_backtest_result_payload(run, db))


@router.get("/backtests/{run_id}")
def get_backtest(run_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    run = db.get(BacktestRun, run_id)
    if not run:
        raise HTTPException(404, "回测不存在")
    metrics = db.scalars(select(BacktestMetric).where(BacktestMetric.backtest_run_id == run_id)).all()
    metric_values = {m.name: m.value for m in metrics}
    initial_capital = float(run.initial_capital)
    final_assets = float(metric_values.get("final_assets", initial_capital * (1 + metric_values.get("total_return", 0))))
    overall_return = float(metric_values.get("overall_return", metric_values.get("total_return", final_assets / initial_capital - 1)))
    config = run.config or {}
    saved_strategy_config = config.get("strategy_config", {})
    if not saved_strategy_config:
        saved_strategy_config = {
            key: config.get(key)
            for key in (
                "cash_buffer",
                "trend_filter",
                "risk_off_exposure",
                "turnover_band",
                "stop_loss",
                "benchmark_enhancement",
                "dip_buy_strength",
                "profit_take_strength",
                "target_volatility",
                "max_drawdown_budget",
                "drawdown_brake_exposure",
            )
            if key in config
        }
    return {
        "id": run.id,
        "status": run.status,
        "start_date": run.start_date,
        "end_date": run.end_date,
        "initial_capital": initial_capital,
        "final_assets": final_assets,
        "total_profit": float(metric_values.get("total_profit", final_assets - initial_capital)),
        "overall_return": overall_return,
        "metrics": metric_values,
        "risk_summary": config.get("risk_summary", {}),
        "strategy_config": saved_strategy_config,
        "trade_markers": _trade_markers(run_id, db),
        "error_message": run.error_message,
        "data_mode": run.data_mode,
    }


@router.get("/backtests/{run_id}/equity")
def get_backtest_equity(run_id: int, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    run = db.get(BacktestRun, run_id)
    if not run:
        raise HTTPException(404, "回测不存在")
    rows = db.scalars(select(BacktestEquity).where(BacktestEquity.backtest_run_id == run_id).order_by(BacktestEquity.trade_date)).all()
    frame = pd.DataFrame(
        [
            {"trade_date": row.trade_date, "equity": row.equity, "benchmark": row.benchmark, "drawdown": row.drawdown}
            for row in rows
        ]
    )
    return _backtest_equity_records(frame, run.initial_capital)


def _trade_markers(run_id: int, db: Session) -> List[Dict[str, Any]]:
    """Return chart-friendly BUY/SELL markers for a backtest equity curve."""
    equity_rows = db.scalars(select(BacktestEquity).where(BacktestEquity.backtest_run_id == run_id)).all()
    value_by_date = {row.trade_date.isoformat(): row.equity for row in equity_rows}
    rows = db.scalars(select(Trade).where(Trade.backtest_run_id == run_id).order_by(Trade.trade_date)).all()
    return [
        {
            "trade_date": row.trade_date.isoformat(),
            "equity": value_by_date.get(row.trade_date.isoformat(), 0),
            "symbol": row.symbol,
            "side": row.side,
            "quantity": row.quantity,
            "price": row.price,
            "amount": row.amount,
            "reason": row.reason,
        }
        for row in rows
    ]


@router.get("/backtests/{run_id}/trades")
def get_backtest_trades(run_id: int, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    run = db.get(BacktestRun, run_id)
    if not run:
        raise HTTPException(404, "回测不存在")
    rows = db.scalars(select(Trade).where(Trade.backtest_run_id == run_id).order_by(Trade.trade_date.desc())).all()
    catalog = MarketDataService(db).stocks()
    metadata = catalog.set_index("symbol").to_dict(orient="index") if not catalog.empty else {}
    strategy = db.get(Strategy, run.strategy_id)
    return [
        {
            "id": row.id,
            "backtest_id": run_id,
            "trade_date": row.trade_date.isoformat(),
            "symbol": row.symbol,
            "name": metadata.get(row.symbol, {}).get("name", row.symbol),
            "exchange": metadata.get(row.symbol, {}).get("exchange", ""),
            "research_group": metadata.get(row.symbol, {}).get("group", ""),
            "sector": metadata.get(row.symbol, {}).get("sector", ""),
            "industry": metadata.get(row.symbol, {}).get("industry", ""),
            "side": row.side,
            "quantity": row.quantity,
            "price": row.price,
            "amount": row.amount,
            "fee": row.fee,
            "reason": row.reason,
            "strategy": strategy.name if strategy else "",
            "data_mode": run.data_mode,
        }
        for row in rows
    ]


@router.get("/backtests/{run_id}/trades.csv")
def export_backtest_trades_csv(run_id: int, db: Session = Depends(get_db)) -> StreamingResponse:
    """Export the complete, auditable execution ledger for one backtest."""
    run = db.get(BacktestRun, run_id)
    if not run:
        raise HTTPException(404, "回测不存在")
    strategy = db.get(Strategy, run.strategy_id)
    rows = db.scalars(select(Trade).where(Trade.backtest_run_id == run_id).order_by(Trade.trade_date, Trade.id)).all()
    catalog = MarketDataService(db).stocks()
    metadata = catalog.set_index("symbol").to_dict(orient="index") if not catalog.empty else {}
    fields = [
        "backtest_id",
        "trade_date",
        "symbol",
        "name",
        "exchange",
        "research_group",
        "sector",
        "industry",
        "side",
        "quantity",
        "price",
        "amount",
        "fee",
        "strategy",
        "reason",
        "data_mode",
    ]
    output = StringIO()
    output.write("\ufeff")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        stock = metadata.get(row.symbol, {})
        writer.writerow(
            {
                "backtest_id": run_id,
                "trade_date": row.trade_date.isoformat(),
                "symbol": row.symbol,
                "name": stock.get("name", row.symbol),
                "exchange": stock.get("exchange", ""),
                "research_group": stock.get("group", ""),
                "sector": stock.get("sector", ""),
                "industry": stock.get("industry", ""),
                "side": row.side,
                "quantity": row.quantity,
                "price": f"{row.price:.6f}",
                "amount": f"{row.amount:.2f}",
                "fee": f"{row.fee:.2f}",
                "strategy": strategy.name if strategy else "",
                "reason": row.reason,
                "data_mode": run.data_mode,
            }
        )
    headers = {"Content-Disposition": f'attachment; filename="backtest_{run_id}_trades.csv"'}
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv; charset=utf-8", headers=headers)


@router.get("/backtests/{run_id}/stocks")
def get_backtest_stocks(run_id: int, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    run = db.get(BacktestRun, run_id)
    if not run:
        raise HTTPException(404, "回测不存在")
    stored = (run.config or {}).get("selected_stocks", [])
    if stored:
        return stored
    # 早期回测没存入选名单：按策略现在的规格在回测结束日重新打一次分
    strategy = db.get(Strategy, run.strategy_id)
    data = data_service_for(db, strategy)
    request = BacktestRequest.model_validate({**(run.config or {}), "strategy_id": strategy.id})
    pool = _backtest_pool(data, request)
    symbols = pool.symbols
    spec = _spec_for_run(strategy, request)
    pipeline = build_pipeline_strategy(spec, data)
    pipeline.members_at = pool.members_at
    pipeline.prepare(symbols, run.end_date, run.end_date)
    ranking = pipeline.generate_weights(run.end_date, symbols).ranking
    return _latest_selection(ranking.assign(signal_date=run.end_date), _engine_config(spec, len(symbols)).holdings_count)


@lru_cache(maxsize=1)
def _fixed_universe_names() -> Dict[str, str]:
    from app.quant_v3.broad_universe import BROAD_STOCKS
    from app.quant_v3.csi300_universe import csi300_stocks

    names = {}
    for stock in [*BROAD_STOCKS, *csi300_stocks()]:
        names[stock["symbol"].split(".")[0]] = stock["name"]
    return names


def _stock_name(db: Session, symbol: str) -> Optional[str]:
    """股票名称：先查通用目录，查不到再查固定股票池名单（demo 模式的目录只有 50 支）。
    迁移前的模型策略回测记录里代码带 .SH/.SZ 后缀，这里一并兼容。"""
    bare = symbol.split(".")[0]
    stock = db.scalar(select(Stock).where(Stock.symbol == bare, Stock.active.is_(True)))
    if stock:
        return stock.name
    try:
        return _fixed_universe_names().get(bare)
    except FileNotFoundError:
        return None


def _backtest_chart_prices(data: MarketDataService, symbol: str, start: date, end: date) -> pd.DataFrame:
    return data.prices([symbol.split(".")[0]], start, end)


@router.get("/backtests/{run_id}/stocks/{symbol}/chart")
def get_backtest_stock_chart(run_id: int, symbol: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    run = db.get(BacktestRun, run_id)
    if not run:
        raise HTTPException(404, "回测不存在")
    strategy = db.get(Strategy, run.strategy_id)
    # 已经调出指数或退市的股票可能不在当前目录里，没有名称就显示代码，不当作"股票不存在"
    name = _stock_name(db, symbol) or symbol.split(".")[0]
    data = data_service_for(db, strategy) if strategy else MarketDataService(db)
    prices = _backtest_chart_prices(data, symbol, run.start_date, run.end_date)
    price_data_mode = data.frame_data_mode(prices)
    if prices.empty:
        return {"symbol": symbol, "name": name, "prices": [], "trade_markers": [], "data_mode": price_data_mode}
    prices = prices.sort_values("trade_date")
    trades = db.scalars(select(Trade).where(Trade.backtest_run_id == run_id, Trade.symbol == symbol).order_by(Trade.trade_date)).all()
    markers = [{"trade_date": item.trade_date.isoformat(), "price": item.price, "side": item.side, "quantity": item.quantity, "amount": item.amount, "reason": item.reason} for item in trades]
    first_price = float(prices.adj_close.iloc[0])
    last_price = float(prices.adj_close.iloc[-1])
    return {"symbol": symbol, "name": name, "start_date": run.start_date.isoformat(), "end_date": run.end_date.isoformat(),
            "prices": _records(prices[["trade_date", "adj_close", "volume"]]), "trade_markers": markers, "data_mode": price_data_mode,
            "return": last_price / first_price - 1 if first_price else 0, "trade_count": len(markers)}


@router.get("/backtests/{run_id}/stocks/charts")
def get_backtest_stock_charts(run_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return every final Top-N price curve and its executed BUY/SELL points.

    The single-stock endpoint remains useful for a large chart. This bundle
    endpoint powers the audit grid so a user can review all selected names
    without manually opening ten separate requests.
    """
    run = db.get(BacktestRun, run_id)
    if not run:
        raise HTTPException(404, "回测不存在")
    strategy = db.get(Strategy, run.strategy_id)
    stored = (run.config or {}).get("selected_stocks", [])
    symbols = [str(item.get("symbol")) for item in stored if item.get("symbol")]
    if not symbols:
        symbols = [str(item.get("symbol")) for item in get_backtest_stocks(run_id, db) if item.get("symbol")]
    data = data_service_for(db, strategy) if strategy else MarketDataService(db)
    result: List[Dict[str, Any]] = []
    for symbol in symbols[:100]:
        name = _stock_name(db, symbol) or symbol.split(".")[0]
        prices = _backtest_chart_prices(data, symbol, run.start_date, run.end_date)
        if prices.empty:
            result.append({
                "symbol": symbol,
                "name": name,
                "start_date": run.start_date.isoformat(),
                "end_date": run.end_date.isoformat(),
                "prices": [],
                "trade_markers": [],
                "data_mode": "unknown",
                "return": 0.0,
                "trade_count": 0,
            })
            continue
        prices = prices.sort_values("trade_date")
        trades = db.scalars(
            select(Trade)
            .where(Trade.backtest_run_id == run_id, Trade.symbol == symbol)
            .order_by(Trade.trade_date)
        ).all()
        markers = [
            {
                "trade_date": item.trade_date.isoformat(),
                "price": item.price,
                "side": item.side,
                "quantity": item.quantity,
                "amount": item.amount,
                "reason": item.reason,
            }
            for item in trades
        ]
        first_price = float(prices.adj_close.iloc[0])
        last_price = float(prices.adj_close.iloc[-1])
        result.append({
            "symbol": symbol,
            "name": name,
            "start_date": run.start_date.isoformat(),
            "end_date": run.end_date.isoformat(),
            "prices": _records(prices[["trade_date", "adj_close", "volume"]]),
            "trade_markers": markers,
            "data_mode": data.frame_data_mode(prices),
            "return": last_price / first_price - 1 if first_price else 0,
            "trade_count": len(markers),
        })
    return {
        "run_id": run_id,
        "items": result,
        "selected_count": len(result),
        "data_mode": data.mode,
    }


@router.post("/backtests/{run_id}/apply-paper")
def apply_backtest_to_paper(
    run_id: int,
    payload: Optional[ApplyBacktestPaperRequest] = None,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Freeze one completed backtest as a Paper Trading strategy.

    A backtest may have runtime overrides (Top N, max weight, frequency) and
    may use a manually selected custom pool.  The normal strategy endpoint
    cannot represent those ephemeral overrides, so this endpoint creates a
    versioned, non-default strategy snapshot and, for a custom pool, a
    durable watchlist.  Paper Trading then runs the exact same factor weights
    and risk controls against that snapshot.
    """
    run = db.get(BacktestRun, run_id)
    if not run:
        raise HTTPException(404, "回测不存在")
    if run.status != "completed":
        raise HTTPException(422, "只有已完成的回测才能应用到模拟盘")

    options = payload or ApplyBacktestPaperRequest()
    reset_account = options.reset_account
    execute = options.execute
    enable_automation = options.enable_automation
    initial_capital = options.initial_capital

    source_strategy = db.get(Strategy, run.strategy_id)
    if not source_strategy:
        raise HTTPException(404, "回测对应策略不存在")
    config = run.config or {}
    universe = normalize_universe(str(config.get("universe") or "csi300"))
    custom_symbols = list(dict.fromkeys(str(symbol).strip() for symbol in (config.get("custom_symbols") or []) if str(symbol).strip()))
    data = data_service_for(db, source_strategy)

    watchlist = None
    if universe == "custom":
        if len(custom_symbols) < 10:
            raise HTTPException(422, "该回测没有可用于模拟盘的 10 只有效股票")
        ensured = data.ensure_symbols(custom_symbols)
        if ensured.get("missing"):
            raise HTTPException(422, "以下股票无法加入模拟盘: %s" % ", ".join(ensured["missing"]))
        watchlist_name = "回测 #%s · 模拟盘股票池" % run.id
        watchlist = db.scalar(select(Watchlist).where(Watchlist.name == watchlist_name))
        if watchlist is None:
            watchlist = Watchlist(name=watchlist_name, symbols=custom_symbols)
            db.add(watchlist)
            db.flush()
        else:
            watchlist.symbols = custom_symbols
        strategy_universe = "custom:%s" % watchlist.id
    else:
        strategy_universe = universe

    saved_spec = (config.get("strategy_config") or {}).get("spec")
    run_spec = (StrategySpec.model_validate(saved_spec) if saved_spec
                else _spec_for_run(source_strategy, BacktestRequest.model_validate({**config, "strategy_id": source_strategy.id})))
    strategy_name = ("%s · 回测 #%s 模拟盘" % (source_strategy.name, run.id))[:120]
    paper_strategy = db.scalar(select(Strategy).where(Strategy.name == strategy_name).order_by(Strategy.id.desc()).limit(1))
    if paper_strategy is None:
        paper_strategy = Strategy(name=strategy_name)
        db.add(paper_strategy)
    paper_strategy.version = 1
    paper_strategy.kind = source_strategy.kind
    paper_strategy.origin = "paper_snapshot"
    paper_strategy.description = "从回测 #%s 应用；冻结股票池和这次回测实际执行的策略规格" % run.id
    paper_strategy.spec = run_spec.model_dump(mode="json")
    paper_strategy.weights = dict(getattr(run_spec.scorer, "weights", {}) or {})
    paper_strategy.holdings_count = _engine_config(run_spec, len(custom_symbols) or 300).holdings_count
    paper_strategy.max_weight = run_spec.weighting.max_weight
    paper_strategy.rebalance_frequency = run_spec.rebalance.frequency
    paper_strategy.turnover_band = run_spec.rebalance.turnover_band
    paper_strategy.trend_filter = run_spec.timing.type != "none"
    paper_strategy.risk_off_exposure = getattr(run_spec.timing, "risk_off_exposure", 1.0)
    paper_strategy.universe = strategy_universe
    paper_strategy.research_start_date = run.start_date
    paper_strategy.research_end_date = run.end_date
    paper_strategy.is_default = False
    db.flush()
    if not db.scalar(select(StrategyFactor.id).where(StrategyFactor.strategy_id == paper_strategy.id).limit(1)):
        source_factors = db.scalars(select(StrategyFactor).where(StrategyFactor.strategy_id == source_strategy.id)).all()
        for factor in source_factors:
            db.add(
                StrategyFactor(
                    strategy_id=paper_strategy.id,
                    group_name=factor.group_name,
                    factor_name=factor.factor_name,
                    weight=factor.weight,
                    direction=factor.direction,
                )
            )

    service = PaperTradingService(db, data)
    portfolio = service.get_or_create_portfolio()
    if reset_account:
        portfolio = service.reset()
    if initial_capital is not None:
        portfolio.initial_capital = initial_capital
        if reset_account:
            portfolio.cash = initial_capital
    portfolio.strategy_id = paper_strategy.id
    portfolio.execution_universe = strategy_universe
    portfolio.execution_symbols = custom_symbols if universe == "custom" else []
    portfolio.source_backtest_run_id = run.id
    db.commit()

    rebalance_result = service.rebalance(paper_strategy) if execute else None
    if enable_automation:
        service.configure_automation(paper_strategy, True, paper_strategy.rebalance_frequency)
        # The initial execution already happened above; schedule the next
        # cycle from the next period so the background loop cannot duplicate
        # the same order immediately.
        portfolio = service.get_or_create_portfolio()
        portfolio.next_rebalance_date = service.next_rebalance_date(data.demo.as_of, paper_strategy.rebalance_frequency)
        db.commit()
        automation = service.automation_status()
    else:
        automation = service.automation_status()

    return _clean(
        {
            "applied": True,
            "run_id": run_id,
            "strategy": {
                "id": paper_strategy.id,
                "name": paper_strategy.name,
                "universe": paper_strategy.universe,
                "holdings_count": paper_strategy.holdings_count,
                "max_weight": paper_strategy.max_weight,
                "rebalance_frequency": paper_strategy.rebalance_frequency,
                "weights": paper_strategy.weights,
                "spec": paper_strategy.spec,
            },
            "watchlist": (
                {"id": "custom:%s" % watchlist.id, "name": watchlist.name, "count": len(watchlist.symbols), "symbols": watchlist.symbols}
                if watchlist
                else None
            ),
            "rebalance": rebalance_result,
            "automation": automation,
            "account": service.snapshot(),
        }
    )


@router.get("/paper/account")
def paper_account(db: Session = Depends(get_db)) -> Dict[str, Any]:
    return _clean(PaperTradingService(db, MarketDataService(db)).snapshot())


@router.get("/paper/equity")
def paper_equity(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return the paper account NAV history and executions for charting."""
    data = MarketDataService(db)
    portfolio = db.scalar(select(Portfolio).order_by(Portfolio.id).limit(1))
    if not portfolio:
        return {"equity": [], "trade_markers": [], "data_mode": data.mode}
    accounts = db.scalars(
        select(DailyAccount).where(DailyAccount.portfolio_id == portfolio.id).order_by(DailyAccount.trade_date)
    ).all()
    equity = [{"trade_date": row.trade_date.isoformat(), "equity": row.total_assets, "benchmark": portfolio.initial_capital, "drawdown": 0.0} for row in accounts]
    if equity:
        peak = portfolio.initial_capital
        for point in equity:
            peak = max(peak, point["equity"])
            point["drawdown"] = point["equity"] / peak - 1
    trades = db.scalars(select(Trade).where(Trade.portfolio_id == portfolio.id).order_by(Trade.trade_date)).all()
    value_by_date = {point["trade_date"]: point["equity"] for point in equity}
    markers = [{"trade_date": row.trade_date.isoformat(), "equity": value_by_date.get(row.trade_date.isoformat(), portfolio.initial_capital), "symbol": row.symbol, "side": row.side, "quantity": row.quantity, "price": row.price, "amount": row.amount, "reason": row.reason} for row in trades]
    return {"equity": equity, "trade_markers": markers, "data_mode": data.mode}


@router.get("/paper/positions")
def paper_positions(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    return _clean(PaperTradingService(db, MarketDataService(db)).snapshot()["positions"])


@router.get("/paper/orders")
def paper_orders(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    portfolio = db.scalar(select(Portfolio).order_by(Portfolio.id).limit(1))
    query = select(Order).order_by(Order.created_at.desc())
    if portfolio:
        query = query.where(Order.portfolio_id == portfolio.id)
    rows = db.scalars(query.limit(100)).all()
    names = {row.symbol: row.name for row in db.scalars(select(Stock)).all()}
    return [{"id": row.id, "time": row.created_at.isoformat(), "symbol": row.symbol, "name": names.get(row.symbol, row.symbol), "side": row.side, "quantity": row.quantity, "price": row.price, "amount": row.amount, "fee": row.fee, "strategy": row.strategy_name, "reason": row.reason, "status": row.status} for row in rows]


@router.get("/paper/orders.csv")
def export_paper_orders_csv(db: Session = Depends(get_db)) -> StreamingResponse:
    """Export the complete paper execution ledger for the current account."""
    portfolio = db.scalar(select(Portfolio).order_by(Portfolio.id).limit(1))
    query = select(Order).order_by(Order.created_at, Order.id)
    if portfolio:
        query = query.where(Order.portfolio_id == portfolio.id)
    rows = db.scalars(query).all()
    names = {row.symbol: row.name for row in db.scalars(select(Stock)).all()}
    fields = [
        "order_id",
        "time",
        "symbol",
        "name",
        "side",
        "quantity",
        "price",
        "amount",
        "fee",
        "strategy",
        "reason",
        "status",
    ]
    output = StringIO()
    output.write("\ufeff")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "order_id": row.id,
                "time": row.created_at.isoformat(),
                "symbol": row.symbol,
                "name": names.get(row.symbol, row.symbol),
                "side": row.side,
                "quantity": row.quantity,
                "price": f"{row.price:.6f}",
                "amount": f"{row.amount:.2f}",
                "fee": f"{row.fee:.2f}",
                "strategy": row.strategy_name,
                "reason": row.reason,
                "status": row.status,
            }
        )
    headers = {"Content-Disposition": 'attachment; filename="paper_orders.csv"'}
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv; charset=utf-8", headers=headers)


@router.post("/paper/reset")
def paper_reset(payload: PaperResetRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    service = PaperTradingService(db, MarketDataService(db))
    portfolio = service.reset()
    portfolio.initial_capital = payload.initial_capital
    portfolio.cash = payload.initial_capital
    db.commit()
    return _clean(service.snapshot())


@router.post("/paper/rebalance")
def paper_rebalance(payload: PaperRebalanceRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    portfolio = db.scalar(select(Portfolio).order_by(Portfolio.id).limit(1))
    strategy_id = payload.strategy_id or (portfolio.strategy_id if portfolio else None)
    if strategy_id is None:
        default_strategy = db.scalar(select(Strategy).where(Strategy.is_default.is_(True)).limit(1))
        strategy_id = default_strategy.id if default_strategy else None
    strategy = db.get(Strategy, strategy_id) if strategy_id is not None else None
    if not strategy:
        raise HTTPException(404, "策略不存在")
    try:
        return _clean(PaperTradingService(db, data_service_for(db, strategy)).rebalance(strategy, payload.as_of))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/paper/automation")
def paper_automation_status(db: Session = Depends(get_db)) -> Dict[str, Any]:
    return _clean(PaperTradingService(db, MarketDataService(db)).automation_status())


@router.post("/paper/automation")
def paper_automation(payload: PaperAutomationPayload, db: Session = Depends(get_db)) -> Dict[str, Any]:
    strategy = db.get(Strategy, payload.strategy_id)
    if not strategy:
        raise HTTPException(404, "策略不存在")
    service = PaperTradingService(db, MarketDataService(db))
    return _clean(service.configure_automation(strategy, payload.enabled, payload.frequency))


@router.post("/paper/automation/run")
def paper_automation_run(db: Session = Depends(get_db)) -> Dict[str, Any]:
    return _clean(PaperTradingService(db, MarketDataService(db)).run_due_automation())


@router.get("/settings/ai")
def get_ai_settings(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """不回传真实key——前端只需要知道"有没有配置过"和base_url/model是什么，
    没有必要（也不应该）把已保存的key明文吐回浏览器。"""
    credentials = get_ai_credentials(db)
    return {
        "has_api_key": bool(credentials.api_key),
        "base_url": credentials.base_url,
        "model": credentials.model,
        "source": credentials.source,
    }


@router.post("/settings/ai")
def update_ai_settings(payload: AISettingsRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    set_ai_credentials(db, api_key=payload.api_key, base_url=payload.base_url, model=payload.model)
    return get_ai_settings(db)


@router.delete("/settings/ai")
def reset_ai_settings(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """清掉数据库里保存的覆盖配置，回退到.env/环境变量里的默认值。"""
    clear_ai_credentials(db)
    return get_ai_settings(db)


class ExplainTradeRef(BaseModel):
    trade_id: int


def _record_trade_explanation(db: Session, symbol: str, out: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    record = AIExplanation(symbol=symbol, explanation_type="trade", content=out["text"], provider="rules",
                           model_version="rules", confidence=None, context={**context, "facts": out["facts"]})
    db.add(record)
    db.commit()
    db.refresh(record)
    return _clean({**out, "audit_id": record.id})


@router.post("/backtests/{run_id}/trades/explain")
def explain_backtest_trade(run_id: int, payload: ExplainTradeRef, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """解释回测里的一笔成交：在信号日按这次回测实际执行的规格重新打分，只陈述事实。"""
    from app.ai.trade_explainer import explain_trade, previous_trading_day

    run = db.get(BacktestRun, run_id)
    trade = db.get(Trade, payload.trade_id)
    if not run or not trade or trade.backtest_run_id != run_id:
        raise HTTPException(404, "成交不存在")
    strategy = db.get(Strategy, run.strategy_id)
    config = run.config or {}
    saved_spec = (config.get("strategy_config") or {}).get("spec")
    spec = StrategySpec.model_validate(saved_spec) if saved_spec else strategy_spec(strategy)
    data = data_service_for(db, strategy)
    pool = _backtest_pool(data, BacktestRequest.model_validate({**config, "strategy_id": strategy.id}))
    signal_date = previous_trading_day(data, trade.trade_date)
    if signal_date is None:
        raise HTTPException(422, "找不到这笔成交的信号日")
    # 和回测时一样，只给信号日当天的成分股打分
    symbols = [sym for sym in pool.symbols if sym in pool.members_at(signal_date)] if pool.members_at else pool.symbols
    record = {"symbol": trade.symbol, "side": trade.side, "quantity": trade.quantity, "price": trade.price,
              "amount": trade.amount, "trade_date": trade.trade_date.isoformat(), "reason": trade.reason}
    out = explain_trade(db, strategy, symbols, signal_date, record, _stock_name(db, trade.symbol) or trade.symbol, spec)
    return _record_trade_explanation(db, trade.symbol, out, {"run_id": run_id, "trade_id": trade.id})


@router.post("/paper/orders/{order_id}/explain")
def explain_paper_order(order_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """解释模拟盘的一笔订单：模拟盘在调仓当天收盘出信号并成交，按当天的打分陈述事实。"""
    from app.ai.trade_explainer import explain_trade

    order = db.get(Order, order_id)
    if not order:
        raise HTTPException(404, "订单不存在")
    trade = db.scalar(select(Trade).where(Trade.portfolio_id == order.portfolio_id, Trade.symbol == order.symbol,
                                          Trade.side == order.side, Trade.quantity == order.quantity)
                      .order_by(Trade.id.desc()).limit(1))
    strategy = db.scalar(select(Strategy).where(Strategy.name == order.strategy_name).order_by(Strategy.id.desc()).limit(1))
    if not trade or not strategy:
        raise HTTPException(422, "找不到这笔订单对应的成交或策略")
    portfolio = db.get(Portfolio, order.portfolio_id)
    data = data_service_for(db, strategy)
    saved = list(portfolio.execution_symbols or []) if portfolio else []
    symbols = saved or _strategy_symbols(data, strategy)
    record = {"symbol": order.symbol, "side": order.side, "quantity": order.quantity, "price": order.price,
              "amount": order.amount, "trade_date": trade.trade_date.isoformat(), "reason": order.reason}
    out = explain_trade(db, strategy, symbols, trade.trade_date, record, _stock_name(db, order.symbol) or order.symbol)
    return _record_trade_explanation(db, order.symbol, out, {"order_id": order.id})


@router.get("/scorecards")
def scorecards(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """策略库里每个策略的成绩卡（标准条件下的回测事实），以及后台计算进度。"""
    from app.pipeline import scorecard

    return _clean({"standard": {k: str(v) for k, v in scorecard.STANDARD.items()},
                   "cards": scorecard.all_cards(db), "refresh": scorecard.refresh_state()})


@router.post("/scorecards/refresh")
def refresh_scorecards(payload: Optional[ScorecardRefreshRequest] = None) -> Dict[str, Any]:
    from app.pipeline import scorecard

    options = payload or ScorecardRefreshRequest()
    started = scorecard.start_refresh(options.strategy_ids, options.force)
    return {"started": started, "refresh": scorecard.refresh_state()}


def _recommend(db: Session, market: str, max_drawdown: Optional[float], holding: str, question: str) -> Dict[str, Any]:
    from app.ai import recommender
    from app.pipeline import scorecard

    cards = scorecard.all_cards(db)
    result = recommender.rank(cards, market, max_drawdown, holding)
    prefs_text = "市场 %s，能接受的最大回撤 %s，持仓周期 %s" % (
        recommender.MARKET_LABEL.get(market, market),
        "不限" if max_drawdown is None else "−%.0f%%" % (max_drawdown * 100),
        recommender.HOLDING_LABEL[holding])
    if result["chosen"] is None:
        return {"recommendation": None, "alternatives": [], "rules": result["rules"], "notes": result["notes"],
                "preferences": prefs_text, "explanation": None}
    explanation = recommender.explain(db, result, prefs_text, question)
    chosen = result["chosen"]
    record = AIExplanation(
        symbol="-", explanation_type="strategy_recommend", content=explanation["content"],
        provider=explanation["provider"], model_version=explanation.get("model_version", "rules"), confidence=None,
        context={"preferences": {"market": market, "max_drawdown": max_drawdown, "holding": holding, "question": question},
                 "chosen_strategy_id": chosen["strategy_id"], "scorecard_run_id": chosen.get("run_id"),
                 "rules": result["rules"], "notes": result["notes"],
                 # 被数字核对拦下的大模型原文只进审计记录，不返回给页面
                 "rejected_llm_content": explanation.pop("rejected_content", None)},
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {
        "recommendation": chosen, "alternatives": result["alternatives"], "rules": result["rules"],
        "notes": result["notes"], "preferences": prefs_text, "explanation": explanation, "audit_id": record.id,
        # 智能体不能下单：只给出去策略实践回测的入口和建议的回测参数
        "next_step": {"practice_url": f"/backtest?strategy_id={chosen['strategy_id']}", "strategy_url": f"/library/{chosen['strategy_id']}",
                      "universe": chosen["universe"], "start_date": chosen["standard"]["start_date"], "end_date": chosen["standard"]["end_date"]},
    }


@router.post("/ai/recommend")
def recommend_strategy(payload: RecommendRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """智能体推荐：规则按偏好在成绩卡里选一个策略，大模型只负责解释；不下单。"""
    return _clean(_recommend(db, payload.market, payload.max_drawdown, payload.holding, payload.question))


@router.post("/ai/agent")
def agent_turn(payload: AgentTurnRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """智能体对话的一轮（ADR-0051）：按钮直接给出动作；打字时先由大模型翻译成四类意图之一
    （改偏好重新推荐 / 为什么不是某个策略 / 比较两个策略 / 解释名词），回答由规则和成绩卡生成。"""
    from app.ai import recommender
    from app.pipeline import scorecard

    prefs = dict(payload.prefs.model_dump())
    cards = scorecard.all_cards(db)
    action = payload.action.model_dump(exclude_none=True) if payload.action else None
    if action is None:
        if not payload.text.strip():
            raise HTTPException(422, "需要一个动作或一句话")
        action = recommender.parse_intent(db, payload.text, cards, prefs)
    kind = action.get("type")
    if kind == "unsupported":
        hint = ("没有配置大模型，只能点下面的按钮提问。" if action.get("reason") == "no_llm"
                else "这个我答不了。可以问我：换个回撤或调仓频率重新推荐、为什么不推荐某个策略、比较两个策略、解释某个名词（比如夏普比率）。")
        return {"prefs": prefs, "reply": {"type": "unsupported", "text": hint}}
    if kind == "recommend":
        for key in ("max_drawdown", "holding"):
            if key in action:
                prefs[key] = action[key]
        result = _recommend(db, prefs["market"], prefs.get("max_drawdown"), prefs["holding"], payload.text)
        text = result["explanation"]["content"] if result["explanation"] else "；".join(result["notes"])
        return _clean({"prefs": prefs, "reply": {"type": "recommend", "text": text, **result}})
    if kind == "why_not":
        out = recommender.why_not(cards, int(action["strategy_id"]), prefs["market"], prefs.get("max_drawdown"), prefs["holding"])
        return _clean({"prefs": prefs, "reply": {"type": "why_not", **out}})
    if kind == "compare":
        out = recommender.compare(cards, [int(i) for i in action.get("strategy_ids", [])][:2])
        return _clean({"prefs": prefs, "reply": {"type": "compare", **out}})
    if kind == "explain_term":
        term = action.get("term")
        if term not in recommender.TERMS:
            raise HTTPException(422, "不认识这个名词")
        return {"prefs": prefs, "reply": {"type": "explain_term", "text": f"{term}：{recommender.TERMS[term]}"}}
    raise HTTPException(422, "未知的动作")


@router.post("/ai/audit/{audit_id}/decision")
def record_ai_decision(audit_id: int, payload: AIDecisionRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """人工对某次 AI 输出的采纳/回滚记录——审计留痕闭环的另一半：不仅要
    记录 AI 说了什么，还要记录人有没有听、有没有事后撤销。"""
    record = db.get(AIExplanation, audit_id)
    if not record:
        raise HTTPException(404, "审计记录不存在")
    if payload.adopted is not None:
        record.adopted = payload.adopted
    if payload.rolled_back is not None:
        record.rolled_back = payload.rolled_back
    record.decided_at = datetime.utcnow()
    db.commit()
    return {"id": record.id, "adopted": record.adopted, "rolled_back": record.rolled_back, "decided_at": record.decided_at.isoformat()}


@router.get("/ai/audit")
def list_ai_audit(limit: int = 50, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    rows = db.scalars(select(AIExplanation).order_by(AIExplanation.id.desc()).limit(limit)).all()
    return [
        {
            "id": row.id, "symbol": row.symbol, "explanation_type": row.explanation_type,
            "provider": row.provider, "model_version": row.model_version, "confidence": row.confidence,
            "content": row.content, "created_at": row.created_at.isoformat(),
            "adopted": row.adopted, "rolled_back": row.rolled_back,
            "decided_at": row.decided_at.isoformat() if row.decided_at else None,
        }
        for row in rows
    ]


@router.post("/ai/report/analyze")
async def analyze_report(file: UploadFile = File(...), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """功能①：上传研报，AI 挖掘给总结和策略参考——只解读，不生成代码、
    不自动创建策略。"""
    content = await file.read()
    try:
        report_text = extract_text(file.filename or "report.txt", content)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    result = ReportAnalysisService().analyze(report_text, db)
    record = AIExplanation(
        symbol="-", explanation_type="report_analysis", content=json.dumps(
            {"summary": result["summary"], "key_points": result["key_points"], "strategy_reference": result["strategy_reference"]},
            ensure_ascii=False,
        ),
        provider=result["provider"], model_version=result.get("model_version", "rules"),
        confidence=result.get("confidence"), context={"filename": file.filename, "text_preview": report_text[:500]},
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {**result, "audit_id": record.id, "filename": file.filename}


@router.get("/data/market/status")
def market_data_status() -> Dict[str, Any]:
    """本地行情库最新日期、最近交易日、缺多少天，以及补齐任务的进度。"""
    return market_status()


@router.post("/data/market/refresh")
def market_data_refresh() -> Dict[str, Any]:
    """手动补齐本地行情库到最近交易日。在后台跑，立即返回；已有补齐在跑时不重复开始。"""
    started = start_refresh()
    return {"started": started, **market_status()}


@router.post("/data/catalog/sync")
def sync_catalog(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Explicitly refresh the local stock directory from tushare."""
    return MarketDataService(db).sync_akshare_catalog()
