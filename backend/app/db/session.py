from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
is_sqlite = settings.database_url.startswith("sqlite")
# sqlite3自己的忙等超时默认只有5秒——real模式下一次大回测(300支股票，
# 每支拿到价格就单独commit一次)持续写库的时间远不止5秒，同一时间
# 后台的自动调仓轮询/前端的其它GET请求想读同一个库文件，很容易撞上
# "database is locked"直接报错，而不是等一等就好。两个改动一起解决：
# WAL模式让"一个写事务进行中"不再挡住普通读请求（读写基本不互斥），
# busy_timeout兜底覆盖WAL模式覆盖不到的场景(比如两个写事务真的撞上)。
connect_args = {"check_same_thread": False, "timeout": 30} if is_sqlite else {}
engine = create_engine(settings.database_url, connect_args=connect_args, future=True)
if is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def migrate_lightweight_schema() -> None:
    """Keep local demo databases forward-compatible without a migration dependency."""
    inspector = inspect(engine)
    if "research_groups" in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns("research_groups")}
        if "color" not in columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE research_groups ADD COLUMN color VARCHAR(20) DEFAULT '#31d0aa'"))
    if "portfolio" in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns("portfolio")}
        migrations = {
            "auto_rebalance_enabled": "ALTER TABLE portfolio ADD COLUMN auto_rebalance_enabled BOOLEAN DEFAULT 0",
            "auto_rebalance_frequency": "ALTER TABLE portfolio ADD COLUMN auto_rebalance_frequency VARCHAR(20) DEFAULT 'monthly'",
            "next_rebalance_date": "ALTER TABLE portfolio ADD COLUMN next_rebalance_date DATE",
            "last_auto_run_at": "ALTER TABLE portfolio ADD COLUMN last_auto_run_at DATETIME",
            "execution_universe": "ALTER TABLE portfolio ADD COLUMN execution_universe VARCHAR(50) DEFAULT 'large_cap'",
            "execution_symbols": "ALTER TABLE portfolio ADD COLUMN execution_symbols JSON DEFAULT '[]'",
            "source_backtest_run_id": "ALTER TABLE portfolio ADD COLUMN source_backtest_run_id INTEGER",
        }
        pending = [statement for name, statement in migrations.items() if name not in columns]
        if pending:
            with engine.begin() as connection:
                for statement in pending:
                    connection.execute(text(statement))
    if "ai_explanations" in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns("ai_explanations")}
        migrations = {
            "model_version": "ALTER TABLE ai_explanations ADD COLUMN model_version VARCHAR(60) DEFAULT 'rules'",
            "confidence": "ALTER TABLE ai_explanations ADD COLUMN confidence FLOAT",
            "adopted": "ALTER TABLE ai_explanations ADD COLUMN adopted BOOLEAN",
            "rolled_back": "ALTER TABLE ai_explanations ADD COLUMN rolled_back BOOLEAN DEFAULT 0",
            "decided_at": "ALTER TABLE ai_explanations ADD COLUMN decided_at DATETIME",
        }
        pending = [statement for name, statement in migrations.items() if name not in columns]
        if pending:
            with engine.begin() as connection:
                for statement in pending:
                    connection.execute(text(statement))
    if "strategies" in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns("strategies")}
        migrations = {
            "universe": "ALTER TABLE strategies ADD COLUMN universe VARCHAR(30) DEFAULT 'large_cap'",
            "research_start_date": "ALTER TABLE strategies ADD COLUMN research_start_date DATE DEFAULT '2018-01-01'",
            "research_end_date": "ALTER TABLE strategies ADD COLUMN research_end_date DATE DEFAULT '2025-12-31'",
            "cash_buffer": "ALTER TABLE strategies ADD COLUMN cash_buffer FLOAT DEFAULT 0.02",
            "trend_filter": "ALTER TABLE strategies ADD COLUMN trend_filter BOOLEAN DEFAULT 1",
            "risk_off_exposure": "ALTER TABLE strategies ADD COLUMN risk_off_exposure FLOAT DEFAULT 0.75",
            "turnover_band": "ALTER TABLE strategies ADD COLUMN turnover_band FLOAT DEFAULT 0.03",
            "stop_loss": "ALTER TABLE strategies ADD COLUMN stop_loss FLOAT DEFAULT 0.18",
            "benchmark_enhancement": "ALTER TABLE strategies ADD COLUMN benchmark_enhancement BOOLEAN DEFAULT 1",
            "dip_buy_strength": "ALTER TABLE strategies ADD COLUMN dip_buy_strength FLOAT DEFAULT 0.06",
            "profit_take_strength": "ALTER TABLE strategies ADD COLUMN profit_take_strength FLOAT DEFAULT 0.05",
            "target_volatility": "ALTER TABLE strategies ADD COLUMN target_volatility FLOAT DEFAULT 0.22",
            "max_drawdown_budget": "ALTER TABLE strategies ADD COLUMN max_drawdown_budget FLOAT DEFAULT 0.15",
            "drawdown_brake_exposure": "ALTER TABLE strategies ADD COLUMN drawdown_brake_exposure FLOAT DEFAULT 0.50",
            "kind": "ALTER TABLE strategies ADD COLUMN kind VARCHAR(30) DEFAULT 'multifactor'",
        }
        pending = [statement for name, statement in migrations.items() if name not in columns]
        if pending:
            with engine.begin() as connection:
                for statement in pending:
                    connection.execute(text(statement))
