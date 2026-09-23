from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
import asyncio
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.db import Base, SessionLocal, engine, migrate_lightweight_schema
from app.db.seed import seed_database
from app.data.service import MarketDataService
from app.portfolio.service import PaperTradingService


def _sync_real_catalog_in_background() -> None:
    """real模式下把全市场tushare目录挪到后台刷新，不堵住启动。见
    MarketDataService.seed_if_empty 的说明：启动时已经用demo的50支
    真实公司垫底，这里跑完之后目录会原地换成真实全市场数据。

    这里不再补跑默认多因子策略的基线回测(app.db.seed.seed_database在
    real模式下会跳过这一步)——那个策略在回测中心/自动交易的下拉框里
    已经不可选了(只展示ACTIVE_CSI300_STRATEGIES里已验证的策略)，为一个
    没人会选的策略在后台预热几十支股票的真实历史，得不偿失。首次进
    Dashboard如果这个策略还没有回测记录，净值曲线区域会显示"运行一次
    回测后，净值曲线会显示在这里"，不影响页面本身可用。

    目录刷新完接着预热Dashboard真正会用到的价格缓存(见
    `_warm_dashboard_price_cache`)——两个都要用tushare连接层，放在同一个
    后台线程里顺序执行（tushare是无状态HTTP+token，不需要像baostock
    时代那样争抢一把进程级独占锁，见docs/adr/0046，这里顺序执行单纯是
    为了控制节奏、不必要地并发把两组预热同时打出去）。"""
    try:
        with SessionLocal() as db:
            MarketDataService(db).sync_catalog()
    except Exception:
        # 刷新失败就继续用demo兜底目录，不能把服务搞挂。
        pass
    _warm_dashboard_price_cache()


def _warm_dashboard_price_cache() -> None:
    """Dashboard绑定默认多因子策略时，real模式下会对最多
    ANALYSIS_SYMBOL_CAP(30)支股票发起真实tushare价格查询——不预热的话
    第一个真实访问Dashboard的用户要现付这个代价。写透缓存(prices()内部的
    _persist_real_prices)已经会把结果存进SQLite，这里只是提前把这份
    "第一次总要付的成本"挪到服务启动后台，而不是等真实用户来触发。

    必须跟dashboard()路由里实际用来选股的调用完全一致
    (analysis_symbols(universe_symbols("large_cap")))，不能图省事传demo
    目录的50支了事——analysis_symbols超过上限时按group轮询取前30支，
    这一步对输入列表的顺序敏感：demo目录的顺序和真实目录large_cap的顺序
    不一样，即使两边是同一个50支的集合，轮询结果也可能选出不同的30支，
    预热了却不是真实请求要用的那30支，缓存等于白预热(实测发现——不是
    假设)。这一步在_sync_real_catalog_in_background里已经排在
    sync_catalog()后面，真实目录这时候已经就绪，不需要再退回demo兜底。
    """
    try:
        with SessionLocal() as db:
            service = MarketDataService(db)
            symbols = service.analysis_symbols(service.universe_symbols("large_cap"))
            service.prices(symbols, date(2018, 1, 1), date.today())
            # 沪深300基准指数也要预热——FactorEngine算超额收益/beta时每次
            # 都要用它，之前benchmark()完全没缓存，是Dashboard每次打开都
            # 要再付一次真实tushare延迟的另一个根因（ADR-0045补充发现）。
            service.benchmark(date(2018, 1, 1), date.today())
    except Exception:
        # 预热失败就退化成"第一个真实请求慢"，不能把服务搞挂；real模式
        # 下prices()本身已经不会静默退回demo数据(ADR-0045)，这里的
        # try/except只是防止预热这个后台任务本身的异常影响其它启动流程。
        pass


def _warm_csi300_signal_sources() -> None:
    """沪深300策略第一次被用到时，要从磁盘读35个LightGBM+35个XGBoost
    模型文件、并各自建一份90万行历史的按symbol索引——实测冷启动要
    5分多钟。这个成本只在进程生命周期内付一次(之后靠lru_cache命中)，
    但不能让"刚好第一个点这个策略的用户"承担这5分钟——服务启动时就在
    后台预热，而不是等第一次真实请求才触发。"""
    try:
        from app.quant_v3.csi300_strategies import build_csi300_ensemble_strategy
        build_csi300_ensemble_strategy()  # 这一句同时把LightGBM+XGBoost两个信号源都建好并缓存
    except Exception:
        # 预热失败不能让整个服务起不来——退化成"第一个真实请求慢"，
        # 而不是服务直接挂掉。
        pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    migrate_lightweight_schema()
    with SessionLocal() as db:
        seed_database(db)
    # 测试套件在tests/conftest.py里把这个环境变量设成"0"——5分钟的后台
    # 预热是给手动起的开发服务器用的，pytest的TestClient每个测试都会
    # 重新进一次lifespan，不需要也不应该跟着触发。同一个开关也用来控制
    # 真实目录的后台刷新(两者都是"给手动开发服务器用，测试不需要")。
    if os.environ.get("PATIENCEQUANT_WARM_CSI300", "1") != "0":
        asyncio.create_task(asyncio.to_thread(_warm_csi300_signal_sources))
        if get_settings().data_mode.lower() == "real":
            asyncio.create_task(asyncio.to_thread(_sync_real_catalog_in_background))
    stop = asyncio.Event()

    async def automation_loop() -> None:
        while not stop.is_set():
            try:
                with SessionLocal() as db:
                    PaperTradingService(db, MarketDataService(db)).run_due_automation()
            except Exception:
                # A transient provider or SQLite lock must not take down the API.
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=settings.auto_rebalance_poll_seconds)
            except asyncio.TimeoutError:
                continue

    task = asyncio.create_task(automation_loop())
    try:
        yield
    finally:
        stop.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix=settings.api_prefix)


@app.get("/")
def root():
    return {"name": settings.app_name, "docs": "/docs", "status": "running"}
