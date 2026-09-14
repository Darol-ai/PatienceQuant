from __future__ import annotations

from contextlib import asynccontextmanager
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
    """real模式下把全市场baostock目录(40+秒)挪到后台刷新，不堵住启动。
    见 MarketDataService.seed_if_empty 的说明：启动时已经用demo的50支
    真实公司垫底，这里跑完之后目录会原地换成真实全市场数据。

    这里不再补跑默认多因子策略的基线回测(app.db.seed.seed_database在
    real模式下会跳过这一步)——那个策略在回测中心/自动交易的下拉框里
    已经不可选了(只展示ACTIVE_CSI300_STRATEGIES里已验证的策略)，为一个
    没人会选的策略在后台预热几十支股票的真实历史，只会占着baostock的
    进程级锁(见baostock_provider.py)跟真实用户请求抢，得不偿失。首次
    进Dashboard如果这个策略还没有回测记录，净值曲线区域会显示"运行一次
    回测后，净值曲线会显示在这里"，不影响页面本身可用。"""
    try:
        with SessionLocal() as db:
            MarketDataService(db).sync_catalog()
    except Exception:
        # 刷新失败就继续用demo兜底目录，不能把服务搞挂。
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
