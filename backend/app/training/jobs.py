"""训练任务：后台单线程排队执行，一次只训练一个模型（ADR-0047 第 6 条、ADR-0052）。

状态写在模型自己的 model.json 里（queued → training → ready / failed / cancelled），
服务重启时把"训练到一半"的标成失败，不假装它还在跑。训练完成后，给用到这个模型的
策略算成绩卡。
"""
from __future__ import annotations

import threading
import traceback
from collections import deque
from datetime import datetime
from typing import Deque, Optional

from app.training.trainer import TrainingConfig, list_records, read_record, train, write_record

_lock = threading.Lock()
_queue: Deque[str] = deque()
_cancel = threading.Event()
_current: Optional[str] = None
_worker: Optional[threading.Thread] = None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _update(model_id: str, **fields) -> dict:
    record = read_record(model_id) or {"id": model_id}
    record.update(fields)
    write_record(record)
    return record


def recover_interrupted() -> None:
    """服务重启后，之前排队或训练到一半的任务都已中断，如实标成失败。"""
    for record in list_records():
        if record.get("status") in ("queued", "training"):
            _update(record["id"], status="failed", error="服务重启，训练中断，需要重新训练", finished_at=_now())


def enqueue(model_id: str) -> None:
    global _worker
    with _lock:
        if model_id not in _queue and model_id != _current:
            _queue.append(model_id)
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, name="model-training", daemon=True)
            _worker.start()


def cancel(model_id: str) -> bool:
    with _lock:
        if model_id in _queue:
            _queue.remove(model_id)
            _update(model_id, status="cancelled", finished_at=_now(), progress=None)
            return True
        if model_id == _current:
            _cancel.set()
            return True
    return False


def current() -> Optional[str]:
    return _current


def _run() -> None:
    global _current
    while True:
        with _lock:
            if not _queue:
                _current = None
                return
            _current = _queue.popleft()
            _cancel.clear()
        model_id = _current
        record = read_record(model_id) or {}
        try:
            config = TrainingConfig(**record["config"]).validate()
            _update(model_id, status="training", started_at=_now(), progress="准备数据", error=None, trace=None)
            from app.data.market_refresh import get_market_store

            until = get_market_store().latest_trading_day()
            metrics = train(config, model_id, until,
                            progress=lambda text: _update(model_id, progress=text),
                            cancelled=_cancel.is_set)
            _update(model_id, status="ready", progress=None, error=None, trace=None, metrics=metrics, finished_at=_now(), data_until=until.isoformat())
            _after_ready(model_id)
        except InterruptedError:
            _update(model_id, status="cancelled", progress=None, finished_at=_now())
        except Exception as exc:  # 训练失败要如实记录原因
            _update(model_id, status="failed", progress=None, error=str(exc), trace=traceback.format_exc()[-2000:], finished_at=_now())


def _after_ready(model_id: str) -> None:
    """模型训练好以后，给用到它的策略算成绩卡。"""
    from sqlalchemy import select

    from app.db.models import Strategy
    from app.db.session import SessionLocal
    from app.pipeline import scorecard

    with SessionLocal() as db:
        ids = [row.id for row in db.scalars(select(Strategy)).all()
               if model_id in ((row.spec or {}).get("scorer") or {}).get("models", [])]
    if not ids:
        return

    def retry():
        import time

        # 可能正好有别的成绩卡在算（一次只算一个），等它算完再开始，最多等 1 小时
        for _ in range(360):
            if scorecard.start_refresh(ids):
                return
            time.sleep(10)

    threading.Thread(target=retry, name="scorecard-after-training", daemon=True).start()


def ensure_model(config: TrainingConfig, purpose: str) -> tuple:
    """按训练设置找已有的同设置模型；没有（或上次失败/取消）就登记一个并排队训练。

    返回 (model_id, record, queued)。模型归策略所有，只有训练设置完全相同时才复用（ADR-0052）。
    """
    from app.pipeline.library import FIXED_UNIVERSES
    from app.training.trainer import DATA_START, find_by_fingerprint

    fingerprint = config.fingerprint()
    model_id = f"trained/{fingerprint}"
    existing = find_by_fingerprint(fingerprint)
    if existing and existing.get("status") not in ("failed", "cancelled"):
        return existing["id"], existing, False
    pool_label = FIXED_UNIVERSES[config.pool][0] if config.pool in FIXED_UNIVERSES else config.pool
    algorithm = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "lstm": "LSTM"}.get(config.framework, config.framework)
    record = {
        "id": model_id, "fingerprint": fingerprint, "status": "queued", "created_at": _now(),
        "name": f"{algorithm} · {pool_label} · {len(config.factors)} 个因子 · {config.horizon} 日",
        "pool_label": pool_label + ("（今天的名单）" if config.pool == "csi300" and config.membership == "latest" else ""),
        "description": purpose, "config": config.__dict__, "data_start": DATA_START.isoformat(),
    }
    write_record(record)
    enqueue(model_id)
    return model_id, record, True
