from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.db import Base, SessionLocal, engine, migrate_lightweight_schema
from app.db.seed import seed_database
from app.data.service import MarketDataService
from app.portfolio.service import PaperTradingService


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    migrate_lightweight_schema()
    with SessionLocal() as db:
        seed_database(db)
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
