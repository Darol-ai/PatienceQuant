from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.config import get_settings
from app.data.service import MarketDataService
from app.quant_v3.broad_universe import BROAD_STOCKS
from app.pipeline.library import VALIDATED_KINDS
from app.quant_v3.csi300_universe import csi300_stocks
from app.db.models import BacktestEquity, BacktestMetric, BacktestRun, Portfolio, ResearchAnnotation, Strategy, StrategyFactor, Trade, Watchlist


DEFAULT_WEIGHTS = {
    # The V3 default is an index-enhancement profile: stronger momentum and
    # relative-strength exposure to pursue excess return versus 沪深300, while
    # fundamental/valuation/quality/risk still keep it investable.
    "fundamental": .12,
    "valuation": .12,
    "quality": .16,
    "momentum": .40,
    "risk": .20,
}


FACTOR_GROUPS = {
    "fundamental": ["roe", "roa", "revenue_growth", "profit_growth", "operating_cashflow"],
    "valuation": ["pe", "pb", "ps", "dividend_yield"],
    "quality": ["roe_stability", "gross_margin", "net_margin", "cashflow_profit_ratio"],
    "momentum": ["return_3m", "return_6m", "return_12m"],
    "risk": ["volatility", "max_drawdown", "beta"],
}


def seed_database(db: Session) -> None:
    MarketDataService(db).seed_if_empty()
    strategy = db.scalar(select(Strategy).where(Strategy.is_default.is_(True)).limit(1))
    if not strategy:
        strategy = Strategy(
            name="沪深300增强趋势价值成长策略 V3",
            version=1,
            description="以跑赢或贴近沪深300为目标：月度低频调仓，动量和相对强度优先，叠加估值低吸、过热高抛、趋势风险闸门和基础质量约束",
            weights=DEFAULT_WEIGHTS,
            holdings_count=10,
            max_weight=.15,
            rebalance_frequency="monthly",
            universe="large_cap",
            research_start_date=date(2018, 1, 1),
            research_end_date=date(2025, 12, 31),
            cash_buffer=.02,
            trend_filter=True,
            risk_off_exposure=.75,
            turnover_band=.03,
            stop_loss=.18,
            benchmark_enhancement=True,
            dip_buy_strength=.06,
            profit_take_strength=.05,
            target_volatility=.22,
            max_drawdown_budget=.15,
            drawdown_brake_exposure=.50,
            is_default=True,
        )
        db.add(strategy)
        db.flush()
        for group, factors in FACTOR_GROUPS.items():
            for factor in factors:
                db.add(StrategyFactor(strategy_id=strategy.id, group_name=group, factor_name=factor,
                                      weight=1 / len(factors), direction="negative" if factor in {"pe", "pb", "ps", "volatility", "max_drawdown", "beta"} else "positive"))
    if not db.scalar(select(Portfolio.id).limit(1)):
        db.add(Portfolio(name="默认模拟组合", initial_capital=1_000_000, cash=1_000_000, strategy_id=strategy.id if strategy else None,
                         auto_rebalance_enabled=False, auto_rebalance_frequency="monthly"))
    if not db.scalar(select(Watchlist).where(Watchlist.name == "我的低频研究清单")):
        db.add(Watchlist(name="我的低频研究清单", symbols=[
            "600519", "000333", "600036", "300750", "002594", "601318",
            "688981", "601088", "600900", "601006",
        ]))
    from app.api.routes import RESEARCH_ANNOTATIONS
    for symbol, annotation in RESEARCH_ANNOTATIONS.items():
        if not db.scalar(select(ResearchAnnotation).where(ResearchAnnotation.symbol == symbol)):
            db.add(ResearchAnnotation(symbol=symbol, **annotation))
    # SessionLocal是autoflush=False(app/db/session.py)，上面这些db.add()在
    # commit前不会自动对后续select()可见。_ensure_group_research_annotations
    # 内部会重新select一遍"已存在的symbol"来去重——如果不在这里flush一下，
    # 它会看不到刚add的这些行，把同一个symbol(比如RESEARCH_ANNOTATIONS和
    # 通用目录里都有的000333)当成"不存在"再插一遍，commit时两条pending
    # insert撞唯一约束报错。缩小Demo目录到50支真实公司后这种重叠概率
    # 大幅上升，之前1000+虚构股票的目录几乎不会撞上，掩盖了这个问题。
    db.flush()
    _ensure_group_research_annotations(db)
    db.commit()
    # Upgrade the original MVP default in place so existing test/demo
    # databases receive the same risk-aware execution rules without losing
    # their historical runs. A fresh calculated baseline is then created for
    # the upgraded strategy.
    if strategy.name in {"低频价值成长多因子策略 V1", "稳健低频价值成长策略 V2", "沪深300增强低频价值成长策略 V3"}:
        strategy.name = "沪深300增强趋势价值成长策略 V3"
        strategy.description = "以跑赢或贴近沪深300为目标：月度低频调仓，动量和相对强度优先，叠加估值低吸、过热高抛、趋势风险闸门和基础质量约束"
        strategy.weights = DEFAULT_WEIGHTS
        strategy.cash_buffer = .02
        strategy.trend_filter = True
        strategy.risk_off_exposure = .75
        strategy.turnover_band = .03
        strategy.stop_loss = .18
        strategy.benchmark_enhancement = True
        strategy.dip_buy_strength = .06
        strategy.profit_take_strength = .05
        strategy.target_volatility = .22
        strategy.max_drawdown_budget = .15
        strategy.drawdown_brake_exposure = .50
        db.commit()
    if strategy.is_default and strategy.name == "沪深300增强趋势价值成长策略 V3" and strategy.weights != DEFAULT_WEIGHTS:
        strategy.weights = DEFAULT_WEIGHTS
        strategy.cash_buffer = .02
        strategy.risk_off_exposure = .75
        strategy.benchmark_enhancement = True
        strategy.dip_buy_strength = .06
        strategy.profit_take_strength = .05
        strategy.target_volatility = .22
        strategy.max_drawdown_budget = .15
        strategy.drawdown_brake_exposure = .50
        strategy.description = "以跑赢或贴近沪深300为目标：月度低频调仓，动量和相对强度优先，叠加估值低吸、过热高抛、趋势风险闸门和基础质量约束"
        db.commit()
    _ensure_quant_v3_strategy(db)
    _ensure_csi300_strategies(db)
    # 所有策略都按规格执行（ADR-0048）：给迁移前的策略补上规格
    from app.pipeline.library import ensure_specs, migrate_paper_positions

    ensure_specs(db)
    migrate_paper_positions(db)
    # _ensure_demo_backtest对默认多因子策略的universe(large_cap)逐支调用
    # market_data.prices()——real模式下这会变成真的顺序调用tushare查
    # 2018-2025年8年日线，几十支股票可能要几分钟。这个默认多因子
    # 策略在回测中心/自动交易的下拉框里已经不可选了(只展示已验证的
    # ACTIVE_CSI300_STRATEGIES)，不值得为它付这个代价——real模式下直接
    # 跳过这份种子回测，Dashboard对应位置会显示"运行一次回测后，净值
    # 曲线会显示在这里"，不影响页面可用性。demo模式下这条路径一直很快
    # (纯本地模拟价格)，行为不变。
    if get_settings().data_mode.lower() == "real":
        return
    _ensure_demo_backtest(db, strategy)


def _ensure_quant_v3_strategy(db: Session) -> None:
    """自由探索阶段最终确定的LightGBM量化策略（docs/adr/0013~0037），
    按策略规格执行（app/pipeline，ADR-0048），不经过
    SQLite/MarketDataService通用股票目录。stop_loss/turnover_band/
    target_volatility/max_drawdown_budget 全部置0——ADR-0037已经证明
    BacktestEngine的引擎级风控叠加层对这个策略是净拖累，这里保持和策略
    设计文档一致的行为，不叠加没被写进策略本身的额外风控。
    """
    if db.scalar(select(Strategy).where(Strategy.kind == "quant_v3_regression").limit(1)):
        return
    db.add(Strategy(
        name="LightGBM动量增强策略(30支候选池)",
        kind="quant_v3_regression",
        version=1,
        description="30支跨行业候选池 + LightGBM回归预测(90日窗口，5模型集成) + Top-K相对排序 + 独立动量兜底 + 流动性资格判断，2019-2025历史回测6/7年跑赢等权重买入持有基准，详见 docs/adr/0013~0037",
        weights={},
        holdings_count=len(BROAD_STOCKS),
        max_weight=1.0,
        rebalance_frequency="monthly",
        universe="custom",
        research_start_date=date(2019, 1, 1),
        research_end_date=date(2025, 12, 31),
        cash_buffer=0,
        trend_filter=False,
        risk_off_exposure=1.0,
        turnover_band=0,
        stop_loss=0,
        benchmark_enhancement=False,
        dip_buy_strength=0,
        profit_take_strength=0,
        target_volatility=0,
        max_drawdown_budget=0,
        drawdown_brake_exposure=1.0,
        is_default=False,
    ))
    db.commit()


_CSI300_STRATEGY_LABELS = {
    "csi300_lightgbm": (
        "LightGBM沪深300策略",
        "沪深300全部300支真实成分股 + LightGBM回归预测(90日窗口，5模型集成) + Top-30相对排序，"
        "2019-2025历史回测6/7年跑赢真实沪深300指数，7年复合+646.9% vs 指数+58.9%",
    ),
    "csi300_xgboost": (
        "XGBoost沪深300策略",
        "同一套特征/训练规则，模型换成XGBoost，验证\"树模型持续有效\"的公开研究结论，"
        "2019-2025历史回测6/7年跑赢真实沪深300指数，7年复合+620.8% vs 指数+58.9%",
    ),
    "csi300_ensemble": (
        "LightGBM+XGBoost集成沪深300策略",
        "两个独立训练的模型预测分数取平均，2019-2025历史回测6/7年跑赢真实沪深300指数，"
        "7年复合+611.9% vs 指数+58.9%（略低于单独任一模型，简单平均在两个高度相关的树模型间"
        "没有额外增益，如实记录，仍是有效策略）",
    ),
}


def _ensure_csi300_strategies(db: Session) -> None:
    """对齐主流做法：universe换成沪深300全部真实成分股，3个逐年回测验证
    过真实有效的策略（docs/adr待补），走独立的parquet数据管道
    （app/pipeline，ADR-0048），不经过SQLite/MarketDataService
    通用股票目录。同样不叠加引擎级风控叠加层（ADR-0037的结论延用）。

    `csi300_stocks()`要读`data/csi300_constituents.parquet`——这份文件
    连同同一条数据管道用到的其它parquet/模型目录都不在git里，需要按
    docs/部署-真实数据准备.md先在宿主机生成好。首次部署如果还没来得及
    生成这些文件，不能让整个服务连启动都启动不起来——这里捕获缺文件
    的情况，跳过这3个策略的注册(通用目录/AI投研等其它功能不受影响)，
    等文件准备好后重启服务即可正常注册。
    """
    try:
        holdings_count = max(1, round(len(csi300_stocks()) * 0.1))
    except FileNotFoundError:
        return
    for kind in VALIDATED_KINDS:
        if db.scalar(select(Strategy).where(Strategy.kind == kind).limit(1)):
            continue
        name, description = _CSI300_STRATEGY_LABELS[kind]
        db.add(Strategy(
            name=name,
            kind=kind,
            version=1,
            description=description,
            weights={},
            holdings_count=holdings_count,
            max_weight=1.0,
            rebalance_frequency="monthly",
            universe="custom",
            research_start_date=date(2019, 1, 1),
            research_end_date=date(2025, 12, 31),
            cash_buffer=0,
            trend_filter=False,
            risk_off_exposure=1.0,
            turnover_band=0,
            stop_loss=0,
            benchmark_enhancement=False,
            dip_buy_strength=0,
            profit_take_strength=0,
            target_volatility=0,
            max_drawdown_budget=0,
            drawdown_brake_exposure=1.0,
            is_default=False,
        ))
    db.commit()


def _ensure_group_research_annotations(db: Session) -> None:
    """Keep the visible research board at ten clearly labelled names per group."""
    catalog = MarketDataService(db).stocks()
    if catalog.empty:
        return
    existing = {row.symbol: row for row in db.scalars(select(ResearchAnnotation)).all()}
    for group, group_frame in catalog.groupby("group", sort=False):
        covered = [symbol for symbol in group_frame.symbol.tolist() if symbol in existing]
        if len(covered) >= 10:
            continue
        candidates = group_frame.copy()
        candidates["market_priority"] = candidates["exchange"].map(
            lambda value: 0 if value == "A股" else 1
        )
        candidates = candidates.sort_values(["market_priority", "symbol"])
        for row in candidates.itertuples(index=False):
            if row.symbol in existing:
                continue
            annotation = ResearchAnnotation(
                symbol=row.symbol,
                bucket="大盘核心",
                thesis="%s研究组 · %s / %s，作为长期低频观察样本，重点跟踪基本面、估值与动量变化"
                % (group, row.sector, row.industry),
                risk="关注%s行业景气、估值回撤和流动性；当前为离线 Demo 数据"
                % row.industry,
            )
            db.add(annotation)
            existing[row.symbol] = annotation
            covered.append(row.symbol)
            if len(covered) >= 10:
                break


def _ensure_demo_backtest(db: Session, strategy: Strategy) -> None:
    """Create one calculated baseline so the first Dashboard visit is useful."""
    latest = db.scalar(
        select(BacktestRun)
        .where(BacktestRun.strategy_id == strategy.id, BacktestRun.status == "completed")
        .order_by(BacktestRun.id.desc())
        .limit(1)
    )
    if latest and (latest.config or {}).get("risk_controls_version") == 6:
        return
    try:
        from app.api.routes import _engine_config, _spec_summary
        from app.pipeline.library import strategy_spec
        from app.pipeline.strategy import build_pipeline_strategy

        spec = strategy_spec(strategy)
        market_data = MarketDataService(db)
        symbols = market_data.universe_symbols(strategy.universe or "large_cap")
        config = _engine_config(spec, len(symbols))
        result = BacktestEngine(market_data).run(
            BacktestConfig(start_date=date(2018, 1, 1), end_date=date(2025, 12, 31), initial_capital=1_000_000,
                           rebalance_frequency=spec.rebalance.frequency, holdings_count=config.holdings_count,
                           max_weight=spec.weighting.max_weight), config, symbols,
            strategy=build_pipeline_strategy(spec, market_data),
        )
        run = BacktestRun(strategy_id=strategy.id, status="completed", start_date=date(2018, 1, 1), end_date=date(2025, 12, 31),
                          initial_capital=1_000_000,
                          config={
                              "seeded": True,
                              "risk_controls_version": 6,
                              "universe": strategy.universe or "large_cap",
                              "strategy_config": _spec_summary(spec),
                              "risk_summary": result.risk_summary or {},
                          },
                          data_mode=result.data_mode)
        db.add(run)
        db.flush()
        db.add_all([BacktestMetric(backtest_run_id=run.id, name=name, value=value) for name, value in result.metrics.items()])
        db.bulk_insert_mappings(BacktestEquity, [{"backtest_run_id": run.id, **row} for row in result.equity.to_dict(orient="records")])
        db.bulk_insert_mappings(Trade, [{"backtest_run_id": run.id, "portfolio_id": None, **row} for row in result.trades.to_dict(orient="records")])
        db.commit()
    except Exception:
        db.rollback()
