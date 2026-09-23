"""成绩卡（ADR-0047 第 8 条）：每个策略在同一套标准条件下回测一次得到的事实数据，
供用户比较和智能体推荐。智能体引用的数字只能来自这里。

标准条件：2019-01-01～2025-12-31、初始资金 100 万、手续费 0.1%、滑点 0.05%，
股票池用策略自己的默认股票池。成绩卡本身就是一次普通回测（在回测历史里能看到），
只是 config 里多一个 scorecard 标记；策略规格或标准条件变了，旧成绩卡自动失效。
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import BacktestEquity, BacktestMetric, BacktestRun, Strategy
from app.pipeline.library import default_universe, strategy_spec

STANDARD = {
    "start_date": date(2019, 1, 1),
    "end_date": date(2025, 12, 31),
    "initial_capital": 1_000_000,
    "commission": 0.001,
    "slippage": 0.0005,
}
CARD_METRICS = ("annual_return", "total_return", "max_drawdown", "sharpe", "volatility", "calmar",
                "benchmark_return", "excess_return", "avg_holding_days", "turnover", "trade_count")


def card_key(row: Strategy) -> str:
    payload = {"spec": strategy_spec(row).model_dump(mode="json"), "universe": default_universe(row),
               "standard": {k: str(v) for k, v in STANDARD.items()}}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def library_rows(db: Session) -> List[Strategy]:
    return [row for row in db.scalars(select(Strategy).order_by(Strategy.id)).all() if (row.origin or "user") != "paper_snapshot"]


def _latest_card_run(db: Session, row: Strategy) -> Optional[BacktestRun]:
    key = card_key(row)
    runs = db.scalars(select(BacktestRun).where(BacktestRun.strategy_id == row.id, BacktestRun.status == "completed")
                      .order_by(BacktestRun.id.desc())).all()
    return next((run for run in runs if (run.config or {}).get("scorecard") == key), None)


def _yearly(db: Session, run: BacktestRun) -> List[Dict[str, Any]]:
    rows = db.scalars(select(BacktestEquity).where(BacktestEquity.backtest_run_id == run.id).order_by(BacktestEquity.trade_date)).all()
    years: Dict[int, List[BacktestEquity]] = {}
    for r in rows:
        years.setdefault(r.trade_date.year, []).append(r)
    out, prev = [], None
    for year, items in sorted(years.items()):
        start_equity = prev.equity if prev else items[0].equity
        start_bench = prev.benchmark if prev else items[0].benchmark
        strategy_return = items[-1].equity / start_equity - 1
        bench_return = items[-1].benchmark / start_bench - 1
        out.append({"year": year, "strategy": strategy_return, "benchmark": bench_return, "beat": strategy_return > bench_return})
        prev = items[-1]
    return out


def build_card(db: Session, row: Strategy) -> Dict[str, Any]:
    spec = strategy_spec(row)
    card: Dict[str, Any] = {
        "strategy_id": row.id, "name": row.name, "version": row.version, "origin": row.origin or "user",
        "kind": row.kind, "description": row.description, "market": spec.market,
        "universe": default_universe(row), "scorer": spec.scorer.type, "timing": spec.timing.type,
        "rebalance_frequency": spec.rebalance.frequency, "spec": spec.model_dump(mode="json"),
        "standard": {k: str(v) for k, v in STANDARD.items()}, "status": "missing",
    }
    run = _latest_card_run(db, row)
    if run is None:
        return card
    metrics = {m.name: m.value for m in db.scalars(select(BacktestMetric).where(BacktestMetric.backtest_run_id == run.id)).all()}
    yearly = _yearly(db, run)
    card.update({
        "status": "ready", "run_id": run.id, "computed_at": run.completed_at.isoformat() if run.completed_at else None,
        "data_mode": run.data_mode,
        "metrics": {k: metrics.get(k) for k in CARD_METRICS},
        "average_exposure": (run.config or {}).get("risk_summary", {}).get("average_target_exposure"),
        "yearly": yearly,
        "years_beating_benchmark": sum(1 for y in yearly if y["beat"]),
        "years": len(yearly),
        "worst_year": min(yearly, key=lambda y: y["strategy"]) if yearly else None,
        "best_year": max(yearly, key=lambda y: y["strategy"]) if yearly else None,
    })
    return card


def all_cards(db: Session) -> List[Dict[str, Any]]:
    return [build_card(db, row) for row in library_rows(db)]


# ---- 后台计算：一次只跑一个 ----

_lock = threading.Lock()
_state: Dict[str, Any] = {"running": False, "done": 0, "total": 0, "current": None, "errors": [], "started_at": None, "finished_at": None}


def refresh_state() -> Dict[str, Any]:
    return dict(_state, errors=list(_state["errors"]))


def _run_one(db: Session, row: Strategy) -> None:
    from app.api.routes import run_backtest
    from app.schemas import BacktestRequest

    request = BacktestRequest(strategy_id=row.id, universe=default_universe(row), **STANDARD)
    result = run_backtest(request, db)
    run = db.get(BacktestRun, result["id"])
    run.config = {**(run.config or {}), "scorecard": card_key(row)}
    db.commit()


def start_refresh(strategy_ids: Optional[List[int]] = None, force: bool = False) -> bool:
    """在后台把缺失或过期的成绩卡补上。已经在跑就不再启动，返回 False。"""
    from app.db.session import SessionLocal

    if not _lock.acquire(blocking=False):
        return False

    def worker():
        try:
            with SessionLocal() as db:
                rows = [r for r in library_rows(db) if strategy_ids is None or r.id in strategy_ids]
                todo = [r for r in rows if force or _latest_card_run(db, r) is None]
                _state.update(running=True, done=0, total=len(todo), current=None, errors=[],
                              started_at=datetime.utcnow().isoformat(), finished_at=None)
                for row in todo:
                    _state["current"] = row.name
                    try:
                        _run_one(db, row)
                    except Exception as exc:  # 一个策略失败不影响其它策略
                        db.rollback()
                        detail = getattr(exc, "detail", None) or str(exc)
                        _state["errors"].append({"strategy_id": row.id, "name": row.name, "error": str(detail)})
                    _state["done"] += 1
        finally:
            _state.update(running=False, current=None, finished_at=datetime.utcnow().isoformat())
            _lock.release()

    _state.update(running=True)
    threading.Thread(target=worker, name="scorecard-refresh", daemon=True).start()
    return True
