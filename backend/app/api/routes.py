from __future__ import annotations

import csv
from datetime import date, datetime
from io import StringIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai.service import AIResearchService
from app.backtest.engine import BacktestConfig, BacktestEngine
from app.config import get_settings
from app.data.service import MarketDataService
from app.db.models import AIExplanation, BacktestEquity, BacktestMetric, BacktestRun, DailyAccount, Order, Portfolio, Position, ResearchAnnotation, ResearchGroup, Stock, Strategy, StrategyFactor, Trade, Watchlist
from app.db.session import get_db
from app.factors.engine import FactorEngine
from app.portfolio.service import PaperTradingService
from app.schemas import ApplyBacktestPaperRequest, BacktestRequest, ExplainRequest, PaperAutomationPayload, PaperRebalanceRequest, PaperResetRequest, StrategyPayload, SyncRequest
from app.strategies.base import StrategyConfig
from app.strategies.multifactor import MultiFactorStrategy


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
    "PINK003": {"bucket": "Pink Sheets", "thesis": "软件服务 Demo 样本，用于验证 OTC 低频筛选", "risk": "流动性和信息披露风险"},
    "PINK005": {"bucket": "Pink Sheets", "thesis": "清洁能源 Demo 样本，观察高成长与高波动平衡", "risk": "高波动和融资风险"},
    "PINK009": {"bucket": "Pink Sheets", "thesis": "金融 Demo 样本，侧重低估值和资本质量", "risk": "信用和流动性风险"},
    "PINK014": {"bucket": "Pink Sheets", "thesis": "公用事业 Demo 样本，侧重现金流与防守性", "risk": "成交深度和监管风险"},
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


def _strategy_config(strategy: Strategy, request: Optional[BacktestRequest] = None) -> StrategyConfig:
    return StrategyConfig(
        name=strategy.name,
        weights=strategy.weights,
        holdings_count=request.holdings_count if request and request.holdings_count else strategy.holdings_count,
        max_weight=request.max_weight if request and request.max_weight else strategy.max_weight,
        rebalance_frequency=request.rebalance_frequency if request and request.rebalance_frequency else strategy.rebalance_frequency,
        cash_buffer=float(strategy.cash_buffer if strategy.cash_buffer is not None else .02),
        trend_filter=bool(strategy.trend_filter if strategy.trend_filter is not None else True),
        risk_off_exposure=float(strategy.risk_off_exposure if strategy.risk_off_exposure is not None else .75),
        turnover_band=float(strategy.turnover_band if strategy.turnover_band is not None else .03),
        stop_loss=float(strategy.stop_loss if strategy.stop_loss is not None else .18),
        benchmark_enhancement=bool(strategy.benchmark_enhancement if strategy.benchmark_enhancement is not None else True),
        dip_buy_strength=float(strategy.dip_buy_strength if strategy.dip_buy_strength is not None else .06),
        profit_take_strength=float(strategy.profit_take_strength if strategy.profit_take_strength is not None else .05),
        target_volatility=float(strategy.target_volatility if strategy.target_volatility is not None else .22),
        max_drawdown_budget=float(strategy.max_drawdown_budget if strategy.max_drawdown_budget is not None else .15),
        drawdown_brake_exposure=float(strategy.drawdown_brake_exposure if strategy.drawdown_brake_exposure is not None else .50),
    )


def _universe_symbols(data: MarketDataService, universe: str, custom_symbols: Optional[List[str]] = None) -> List[str]:
    catalog = data.stocks()
    symbols = data.universe_symbols("a_share")
    asset_symbols = set(catalog.symbol.tolist())
    if universe == "hs300":
        return symbols[: min(300, len(symbols))]
    if universe == "large_cap":
        return data.universe_symbols("large_cap")
    if universe == "pink_sheets":
        return data.universe_symbols("pink_sheets")
    if universe == "all_assets":
        return data.universe_symbols("all_assets")
    if universe == "csi_a500":
        return symbols[: min(500, len(symbols))]
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
    return symbols


@router.get("/health")
def health() -> Dict[str, Any]:
    settings = get_settings()
    return {"status": "ok", "app": settings.app_name, "data_mode": settings.data_mode, "as_of": settings.demo_as_of.isoformat(), "demo_universe_size": settings.demo_universe_size, "python": ">=3.9"}


@router.get("/research-groups")
def research_groups(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    groups = db.scalars(select(ResearchGroup).order_by(ResearchGroup.id)).all()
    result = []
    for group in groups:
        count = db.scalar(select(func.count()).select_from(Stock).where(Stock.research_group_id == group.id))
        result.append({"id": group.id, "name": group.name, "description": group.description, "color": group.color, "count": count})
    return result


@router.get("/universes")
def universes(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    data = MarketDataService(db)
    catalog = data.stocks()
    a_share = data.universe_symbols("a_share")
    pink = data.universe_symbols("pink_sheets")
    watchlists = db.scalars(select(Watchlist).order_by(Watchlist.id)).all()
    return [
        {"id": "a_share", "name": "A股全市场", "count": len(a_share), "description": "Demo Provider 千级跨行业股票池"},
        {"id": "large_cap", "name": "大盘股核心池", "count": min(300, len(a_share)), "description": "低频策略优先研究的核心大盘股票"},
        {"id": "hs300", "name": "沪深300（Demo代理）", "count": min(300, len(a_share)), "description": "演示环境使用前 300 只核心权重代理"},
        {"id": "csi_a500", "name": "中证A500（Demo代理）", "count": min(500, len(a_share)), "description": "演示环境使用 500 只跨行业核心股票池"},
        {"id": "pink_sheets", "name": "OTC / Pink Sheets（Demo）", "count": len(pink), "description": "离线示例扩展市场；不代表实时 OTC 行情"},
        {"id": "all_assets", "name": "A股 + OTC扩展", "count": len(catalog), "description": "统一 Data Adapter 下的全部可研究资产"},
    ] + [{"id": "custom:%s" % row.id, "name": row.name, "count": len(row.symbols), "description": "自定义研究股票池"} for row in watchlists]


@router.get("/research/coverage")
def research_coverage(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return the user's clearly annotated, cross-market low-frequency research list."""
    data = MarketDataService(db)
    catalog = data.stocks()
    strategy_row = db.scalar(select(Strategy).where(Strategy.is_default.is_(True)).limit(1))
    ranking_symbols = data.analysis_symbols(catalog.symbol.tolist())
    ranking = MultiFactorStrategy(FactorEngine(data), _strategy_config(strategy_row)).generate_weights(
        get_settings().demo_as_of, ranking_symbols
    ).ranking
    annotations = db.scalars(select(ResearchAnnotation).order_by(ResearchAnnotation.id)).all()
    annotation_by_symbol = {item.symbol: item for item in annotations}
    rows = ranking[ranking.symbol.isin(annotation_by_symbol)].copy()
    for field in ("bucket", "thesis", "risk"):
        rows[field] = rows.symbol.map(lambda symbol: getattr(annotation_by_symbol[str(symbol)], field))
    order = {item.symbol: index for index, item in enumerate(annotations)}
    rows["research_order"] = rows.symbol.map(order)
    rows = rows.sort_values("research_order")
    sectors = [
        {"name": name, "count": int(len(group)), "symbols": group.symbol.tolist()}
        for name, group in rows.groupby("sector", sort=False)
    ]
    return {
        "items": _records(rows),
        "sectors": sectors,
        "total": len(rows),
        "large_cap_count": int((rows.bucket == "大盘核心").sum()),
        "pink_count": int((rows.bucket == "Pink Sheets").sum()),
        "as_of": get_settings().demo_as_of.isoformat(),
        "data_mode": data.mode,
    }


@router.put("/research/coverage/{symbol}")
def update_research_annotation(symbol: str, payload: Dict[str, Any], db: Session = Depends(get_db)) -> Dict[str, Any]:
    annotation = db.scalar(select(ResearchAnnotation).where(ResearchAnnotation.symbol == symbol))
    if not annotation:
        raise HTTPException(404, "研究标注不存在")
    annotation.bucket = str(payload.get("bucket", annotation.bucket)).strip()[:40]
    annotation.thesis = str(payload.get("thesis", annotation.thesis)).strip()[:500]
    annotation.risk = str(payload.get("risk", annotation.risk)).strip()[:500]
    if not annotation.thesis:
        raise HTTPException(422, "研究逻辑不能为空")
    db.commit()
    return {"symbol": annotation.symbol, "bucket": annotation.bucket, "thesis": annotation.thesis, "risk": annotation.risk}


@router.post("/watchlists")
def create_watchlist(payload: Dict[str, Any], db: Session = Depends(get_db)) -> Dict[str, Any]:
    name = str(payload.get("name", "自定义股票池")).strip()
    if db.scalar(select(Watchlist).where(Watchlist.name == name)):
        raise HTTPException(409, "自定义股票池名称已存在")
    symbols = [str(symbol) for symbol in payload.get("symbols", [])]
    valid_symbols = set(MarketDataService(db).stocks().symbol.tolist())
    symbols = list(dict.fromkeys(symbol for symbol in symbols if symbol in valid_symbols))
    if len(symbols) < 10:
        raise HTTPException(422, "自定义股票池至少需要 10 只有效股票")
    watchlist = Watchlist(name=name, symbols=symbols)
    db.add(watchlist)
    db.commit()
    db.refresh(watchlist)
    return {"id": "custom:%s" % watchlist.id, "name": watchlist.name, "count": len(symbols), "symbols": symbols}


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
    entire universe. ``source=akshare`` queries the AKShare code/name table and
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
        "akshare_enabled": bool(data.real.catalog_cached or resolved_source == "akshare"),
        "as_of": get_settings().demo_as_of.isoformat(),
    }


@router.get("/stocks")
def stocks(search: str = "", group: str = "", industry: str = "", exchange: str = "", universe: str = "all_assets", signal: str = "", sort: str = "score", order: str = "desc", db: Session = Depends(get_db)) -> Dict[str, Any]:
    data = MarketDataService(db)
    catalog = data.stocks()
    strategy_row = db.scalar(select(Strategy).where(Strategy.is_default.is_(True)).limit(1))
    strategy = MultiFactorStrategy(FactorEngine(data), _strategy_config(strategy_row))
    ranking = strategy.generate_weights(
        get_settings().demo_as_of,
        data.analysis_symbols(catalog.symbol.tolist()),
    ).ranking
    held = {position.symbol for position in db.scalars(select(Position).where(Position.portfolio_id == db.scalar(select(Portfolio.id).limit(1)))).all()}
    ranking["action"] = ranking.apply(lambda row: "HOLD" if row.symbol in held and row.target_weight > 0 else ("SELL" if row.symbol in held else row.action), axis=1)
    frame = ranking
    if universe in {"a_share", "large_cap", "pink_sheets"}:
        frame = frame[frame.symbol.isin(data.universe_symbols(universe))]
    if search:
        needle = search.lower()
        frame = frame[frame.symbol.str.lower().str.contains(needle) | frame.name.str.lower().str.contains(needle)]
    if group:
        frame = frame[frame.group == group]
    if industry:
        frame = frame[frame.industry == industry]
    if exchange:
        frame = frame[frame.exchange == exchange]
    if signal:
        frame = frame[frame.action == signal]
    sort_column = sort if sort in frame.columns else "score"
    frame = frame.sort_values(sort_column, ascending=order == "asc")
    return {"items": _records(frame), "total": len(frame), "data_mode": data.mode, "as_of": get_settings().demo_as_of.isoformat()}


@router.get("/stocks/{symbol}")
def stock_detail(symbol: str, range: str = "all", interval: str = "monthly", db: Session = Depends(get_db)) -> Dict[str, Any]:
    data = MarketDataService(db)
    catalog = data.stocks()
    if symbol not in set(catalog.symbol):
        raise HTTPException(404, "股票不存在")
    strategy = db.scalar(select(Strategy).where(Strategy.is_default.is_(True)).limit(1))
    ranking_symbols = data.analysis_symbols(catalog.symbol.tolist())
    if symbol not in ranking_symbols:
        ranking_symbols.append(symbol)
    result = MultiFactorStrategy(FactorEngine(data), _strategy_config(strategy)).generate_weights(
        get_settings().demo_as_of,
        ranking_symbols,
    )
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
        result.append({"id": row.id, "name": row.name, "version": row.version, "description": row.description, "weights": row.weights, "holdings_count": row.holdings_count, "max_weight": row.max_weight, "rebalance_frequency": row.rebalance_frequency, "universe": row.universe or "large_cap", "research_start_date": research_start.isoformat(), "research_end_date": research_end.isoformat(),
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
    db.add(strategy)
    db.commit()
    db.refresh(strategy)
    return {"id": strategy.id, "version": strategy.version, "message": "策略版本已保存"}


@router.put("/strategies/{strategy_id}")
def update_strategy(strategy_id: int, payload: StrategyPayload, db: Session = Depends(get_db)) -> Dict[str, Any]:
    strategy = db.get(Strategy, strategy_id)
    if not strategy:
        raise HTTPException(404, "策略不存在")
    for field, value in payload.model_dump().items():
        setattr(strategy, field, value)
    db.commit()
    return {"id": strategy.id, "message": "策略已更新"}


@router.post("/backtests")
def run_backtest(payload: BacktestRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    strategy = db.get(Strategy, payload.strategy_id)
    if not strategy:
        raise HTTPException(404, "策略不存在")
    if payload.start_date >= payload.end_date:
        raise HTTPException(422, "结束日期必须晚于开始日期")
    holdings_count = payload.holdings_count or strategy.holdings_count
    max_weight = payload.max_weight or strategy.max_weight
    if holdings_count * max_weight < 1:
        raise HTTPException(422, "持仓数量与单股最大权重无法构建满仓组合")
    run = BacktestRun(strategy_id=strategy.id, status="running", start_date=payload.start_date, end_date=payload.end_date, initial_capital=payload.initial_capital, config=payload.model_dump(mode="json"), data_mode=get_settings().data_mode)
    db.add(run)
    db.commit()
    db.refresh(run)
    try:
        sc = _strategy_config(strategy, payload)
        market_data = MarketDataService(db)
        run.data_mode = market_data.mode
        if payload.universe == "custom":
            catalog_sync = market_data.ensure_symbols(payload.custom_symbols)
            if catalog_sync.get("missing"):
                raise HTTPException(
                    422,
                    "以下股票代码无法从本地目录或 AKShare 解析: %s"
                    % ", ".join(catalog_sync["missing"]),
                )
        symbols = _universe_symbols(market_data, payload.universe, payload.custom_symbols)
        if sc.holdings_count > len(symbols):
            sc.holdings_count = len(symbols)
        if sc.holdings_count * sc.max_weight < 1:
            raise HTTPException(422, "当前股票池标的数量不足以满足单股最大权重限制，请增加股票或提高上限")
        result = BacktestEngine(market_data).run(
            BacktestConfig(
                start_date=payload.start_date,
                end_date=payload.end_date,
                initial_capital=payload.initial_capital,
                rebalance_frequency=sc.rebalance_frequency,
                holdings_count=sc.holdings_count,
                max_weight=sc.max_weight,
                commission=payload.commission,
                slippage=payload.slippage,
            ),
            sc,
            symbols,
            # A manually selected pool is an explicit user request for
            # current AKShare data. Named full-market universes remain
            # offline-safe unless DATA_MODE=real is enabled.
            allow_network=payload.universe == "custom",
        )
        run.data_mode = result.data_mode
        selected_stocks = _latest_selection(result.ranking, sc.holdings_count)
        run.config = {
            **payload.model_dump(mode="json"),
            "risk_controls_version": 5,
            "selected_stocks": selected_stocks,
            "risk_summary": result.risk_summary or {},
            "strategy_config": {
                "cash_buffer": sc.cash_buffer,
                "trend_filter": sc.trend_filter,
                "risk_off_exposure": sc.risk_off_exposure,
                "turnover_band": sc.turnover_band,
                "stop_loss": sc.stop_loss,
                "benchmark_enhancement": sc.benchmark_enhancement,
                "dip_buy_strength": sc.dip_buy_strength,
                "profit_take_strength": sc.profit_take_strength,
                "target_volatility": sc.target_volatility,
                "max_drawdown_budget": sc.max_drawdown_budget,
                "drawdown_brake_exposure": sc.drawdown_brake_exposure,
            },
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
        return {
            "id": run.id,
            "status": run.status,
            "start_date": payload.start_date.isoformat(),
            "end_date": payload.end_date.isoformat(),
            "initial_capital": payload.initial_capital,
            "strategy_name": strategy.name,
            "metrics": result.metrics,
            "equity": _records(result.equity),
            "trades": _records(result.trades),
            "trade_markers": _chart_trade_markers(result.equity, result.trades),
            "annual_returns": _annual_returns(result.equity),
            "selected_stocks": selected_stocks,
            "universe_size": len(symbols),
            "custom_symbols": symbols if payload.universe == "custom" else [],
            "notes": result.notes,
            "risk_summary": result.risk_summary or {},
            "strategy_config": {
                "weights": sc.weights,
                "holdings_count": sc.holdings_count,
                "max_weight": sc.max_weight,
                "rebalance_frequency": sc.rebalance_frequency,
                "cash_buffer": sc.cash_buffer,
                "trend_filter": sc.trend_filter,
                       "risk_off_exposure": sc.risk_off_exposure,
                       "turnover_band": sc.turnover_band,
                       "stop_loss": sc.stop_loss,
                       "benchmark_enhancement": sc.benchmark_enhancement,
                       "dip_buy_strength": sc.dip_buy_strength,
                       "profit_take_strength": sc.profit_take_strength,
                       "target_volatility": sc.target_volatility,
                       "max_drawdown_budget": sc.max_drawdown_budget,
                       "drawdown_brake_exposure": sc.drawdown_brake_exposure,
            },
            "data_mode": run.data_mode,
            "universe": payload.universe,
        }
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


@router.get("/backtests/{run_id}")
def get_backtest(run_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    run = db.get(BacktestRun, run_id)
    if not run:
        raise HTTPException(404, "回测不存在")
    metrics = db.scalars(select(BacktestMetric).where(BacktestMetric.backtest_run_id == run_id)).all()
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
        "metrics": {m.name: m.value for m in metrics},
        "risk_summary": config.get("risk_summary", {}),
        "strategy_config": saved_strategy_config,
        "trade_markers": _trade_markers(run_id, db),
        "error_message": run.error_message,
        "data_mode": run.data_mode,
    }


@router.get("/backtests/{run_id}/equity")
def get_backtest_equity(run_id: int, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    rows = db.scalars(select(BacktestEquity).where(BacktestEquity.backtest_run_id == run_id).order_by(BacktestEquity.trade_date)).all()
    return [{"trade_date": row.trade_date.isoformat(), "equity": row.equity, "benchmark": row.benchmark, "drawdown": row.drawdown} for row in rows]


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
    strategy = db.get(Strategy, run.strategy_id)
    data = MarketDataService(db)
    universe = (run.config or {}).get("universe", "a_share")
    symbols = _universe_symbols(data, universe, (run.config or {}).get("custom_symbols"))
    config = _strategy_config(strategy)
    config.holdings_count = int((run.config or {}).get("holdings_count") or strategy.holdings_count)
    ranking = MultiFactorStrategy(FactorEngine(data), config).generate_weights(run.end_date, symbols).ranking
    return _latest_selection(ranking.assign(signal_date=run.end_date), config.holdings_count)


@router.get("/backtests/{run_id}/stocks/{symbol}/chart")
def get_backtest_stock_chart(run_id: int, symbol: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    run = db.get(BacktestRun, run_id)
    if not run:
        raise HTTPException(404, "回测不存在")
    stock = db.scalar(select(Stock).where(Stock.symbol == symbol, Stock.active.is_(True)))
    if not stock:
        raise HTTPException(404, "股票不存在")
    data = MarketDataService(db)
    prices = data.prices([symbol], run.start_date, run.end_date, allow_network=True)
    price_data_mode = data.frame_data_mode(prices)
    if prices.empty:
        return {"symbol": symbol, "name": stock.name, "prices": [], "trade_markers": [], "data_mode": price_data_mode}
    prices = prices.sort_values("trade_date")
    trades = db.scalars(select(Trade).where(Trade.backtest_run_id == run_id, Trade.symbol == symbol).order_by(Trade.trade_date)).all()
    markers = [{"trade_date": item.trade_date.isoformat(), "price": item.price, "side": item.side, "quantity": item.quantity, "amount": item.amount, "reason": item.reason} for item in trades]
    first_price = float(prices.adj_close.iloc[0])
    last_price = float(prices.adj_close.iloc[-1])
    return {"symbol": symbol, "name": stock.name, "start_date": run.start_date.isoformat(), "end_date": run.end_date.isoformat(),
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
    stored = (run.config or {}).get("selected_stocks", [])
    symbols = [str(item.get("symbol")) for item in stored if item.get("symbol")]
    if not symbols:
        symbols = [str(item.get("symbol")) for item in get_backtest_stocks(run_id, db) if item.get("symbol")]
    data = MarketDataService(db)
    result: List[Dict[str, Any]] = []
    for symbol in symbols[:100]:
        stock = db.scalar(select(Stock).where(Stock.symbol == symbol, Stock.active.is_(True)))
        if not stock:
            continue
        prices = data.prices([symbol], run.start_date, run.end_date, allow_network=True)
        if prices.empty:
            result.append({
                "symbol": symbol,
                "name": stock.name,
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
            "name": stock.name,
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
    universe = str(config.get("universe") or "large_cap")
    custom_symbols = list(dict.fromkeys(str(symbol).strip() for symbol in (config.get("custom_symbols") or []) if str(symbol).strip()))
    data = MarketDataService(db)

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

    source_config = _strategy_config(source_strategy)
    strategy_name = ("%s · 回测 #%s 模拟盘" % (source_strategy.name, run.id))[:120]
    paper_strategy = db.scalar(select(Strategy).where(Strategy.name == strategy_name).order_by(Strategy.id.desc()).limit(1))
    if paper_strategy is None:
        paper_strategy = Strategy(name=strategy_name)
        db.add(paper_strategy)
    paper_strategy.version = 1
    paper_strategy.description = "从回测 #%s 应用；冻结候选池、Top N、风险控制与交易参数" % run.id
    paper_strategy.weights = dict(source_config.weights)
    paper_strategy.holdings_count = int(config.get("holdings_count") or source_config.holdings_count)
    paper_strategy.max_weight = float(config.get("max_weight") or source_config.max_weight)
    paper_strategy.rebalance_frequency = str(config.get("rebalance_frequency") or source_config.rebalance_frequency)
    paper_strategy.universe = strategy_universe
    paper_strategy.research_start_date = run.start_date
    paper_strategy.research_end_date = run.end_date
    paper_strategy.cash_buffer = source_config.cash_buffer
    paper_strategy.trend_filter = source_config.trend_filter
    paper_strategy.risk_off_exposure = source_config.risk_off_exposure
    paper_strategy.turnover_band = source_config.turnover_band
    paper_strategy.stop_loss = source_config.stop_loss
    paper_strategy.benchmark_enhancement = source_config.benchmark_enhancement
    paper_strategy.dip_buy_strength = source_config.dip_buy_strength
    paper_strategy.profit_take_strength = source_config.profit_take_strength
    paper_strategy.target_volatility = source_config.target_volatility
    paper_strategy.max_drawdown_budget = source_config.max_drawdown_budget
    paper_strategy.drawdown_brake_exposure = source_config.drawdown_brake_exposure
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
                "cash_buffer": paper_strategy.cash_buffer,
                "trend_filter": paper_strategy.trend_filter,
                "risk_off_exposure": paper_strategy.risk_off_exposure,
                "turnover_band": paper_strategy.turnover_band,
                "stop_loss": paper_strategy.stop_loss,
                "benchmark_enhancement": paper_strategy.benchmark_enhancement,
                "dip_buy_strength": paper_strategy.dip_buy_strength,
                "profit_take_strength": paper_strategy.profit_take_strength,
                "target_volatility": paper_strategy.target_volatility,
                "max_drawdown_budget": paper_strategy.max_drawdown_budget,
                "drawdown_brake_exposure": paper_strategy.drawdown_brake_exposure,
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
    return _clean(PaperTradingService(db, MarketDataService(db)).rebalance(strategy, payload.as_of))


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


@router.get("/signals")
def signals(db: Session = Depends(get_db)) -> Dict[str, Any]:
    data = MarketDataService(db)
    catalog = data.stocks()
    strategy = db.scalar(select(Strategy).where(Strategy.is_default.is_(True)).limit(1))
    ranking = MultiFactorStrategy(FactorEngine(data), _strategy_config(strategy)).generate_weights(
        get_settings().demo_as_of,
        data.analysis_symbols(catalog.symbol.tolist()),
    ).ranking
    return {"items": _records(ranking.head(30)), "as_of": get_settings().demo_as_of.isoformat(), "data_mode": data.mode}


@router.get("/analytics/universe")
def universe_analytics(db: Session = Depends(get_db)) -> Dict[str, Any]:
    data = MarketDataService(db)
    catalog = data.stocks()
    strategy = db.scalar(select(Strategy).where(Strategy.is_default.is_(True)).limit(1))
    ranking = MultiFactorStrategy(FactorEngine(data), _strategy_config(strategy)).generate_weights(
        get_settings().demo_as_of,
        data.analysis_symbols(catalog.symbol.tolist()),
    ).ranking
    industry = ranking.groupby("industry").agg(count=("symbol", "count"), avg_score=("score", "mean"), avg_return=("return_12m", "mean")).reset_index().sort_values("count", ascending=False)
    if len(industry) > 12:
        remainder = industry.iloc[12:]["count"].sum()
        industry = pd.concat([industry.iloc[:12], pd.DataFrame([{"industry": "其他", "count": int(remainder), "avg_score": float(ranking.score.mean()), "avg_return": float(ranking.return_12m.mean())}])], ignore_index=True)
    bins = pd.cut(ranking.score, bins=[0, 40, 50, 60, 70, 80, 100], include_lowest=True).value_counts(sort=False)
    return {"industry_distribution": _records(industry), "score_distribution": [{"range": str(key), "count": int(value)} for key, value in bins.items()], "valuation_growth": _records(ranking[["symbol", "name", "industry", "pe", "revenue_growth", "score"]]), "stock_returns": _records(ranking.sort_values("return_12m", ascending=False)[["symbol", "name", "return_12m"]].head(15)), "portfolio_weights": _records(ranking[ranking.target_weight > 0][["symbol", "name", "industry", "target_weight"]])}


@router.post("/ai/explain/trade")
def explain_trade(payload: ExplainRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    result = AIResearchService().explain(payload.model_dump(), payload.use_llm)
    db.add(AIExplanation(symbol=payload.symbol, explanation_type="trade", content=result["content"], provider=result["provider"], context=payload.model_dump()))
    db.commit()
    return result


@router.post("/data/sync")
def sync_data(payload: SyncRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    data = MarketDataService(db)
    symbols = payload.symbols or data.stocks().symbol.head(10).tolist()
    return data.sync_real_prices(symbols, payload.start_date, payload.end_date)


@router.post("/data/catalog/sync")
def sync_catalog(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Explicitly refresh the local stock directory from AKShare."""
    return MarketDataService(db).sync_akshare_catalog()


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db)) -> Dict[str, Any]:
    data = MarketDataService(db)
    paper = PaperTradingService(db, data)
    account = paper.snapshot()
    catalog = data.stocks()
    strategy_row = db.scalar(select(Strategy).where(Strategy.is_default.is_(True)).limit(1))
    strategy_result = MultiFactorStrategy(FactorEngine(data), _strategy_config(strategy_row)).generate_weights(get_settings().demo_as_of, data.universe_symbols("large_cap"))
    orders = paper_orders(db)
    latest_run = db.scalar(select(BacktestRun).where(BacktestRun.status == "completed").order_by(BacktestRun.id.desc()).limit(1))
    metrics = {}
    equity = []
    trade_markers = []
    if latest_run:
        metric_rows = db.scalars(select(BacktestMetric).where(BacktestMetric.backtest_run_id == latest_run.id)).all()
        metrics = {row.name: row.value for row in metric_rows}
        equity = get_backtest_equity(latest_run.id, db)
        trade_markers = _trade_markers(latest_run.id, db)
    return {"account": _clean(account), "backtest_metrics": metrics, "equity": equity, "trade_markers": trade_markers, "positions": _clean(account["positions"]), "recent_orders": orders[:8], "signals": _records(strategy_result.ranking.head(8)), "analytics": universe_analytics(db), "data_mode": data.mode, "as_of": get_settings().demo_as_of.isoformat(), "study_period": {"research": ["2018-01-01", "2024-12-31"], "validation": ["2025-01-01", "2025-12-31"], "paper": ["2026-01-01", get_settings().demo_as_of.isoformat()]}}
