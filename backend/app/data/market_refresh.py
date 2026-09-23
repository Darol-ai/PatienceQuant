"""本地行情库的补齐任务：服务启动时和用户手动点击共用这一个入口。

同一时间只允许一个补齐在跑——行情库按年份文件整体重写，两个补齐同时写
同一年的文件会互相覆盖。补齐在后台线程里跑，接口立刻返回，进度通过
status() 查询。
"""
from __future__ import annotations

import threading
from datetime import datetime
from functools import lru_cache
from typing import Dict, Optional

from app.data.market_store import AShareMarketStore

BENCHMARK_INDEX = "000300.SH"

_lock = threading.Lock()
_state: Dict[str, object] = {"running": False}


@lru_cache
def get_market_store() -> AShareMarketStore:
    return AShareMarketStore()


def _run() -> None:
    store = get_market_store()

    def progress(day, index, total):
        _state.update(current_day=day.isoformat(), done=index, total=total)

    try:
        report = store.backfill(on_progress=progress)
        store.refresh_index(BENCHMARK_INDEX)
        store.refresh_index_members(BENCHMARK_INDEX)
        # 每日指标跟着日线补（ADR-0053）
        basic = store.backfill_basic(on_progress=progress)
        if basic.stopped_reason and not report.stopped_reason:
            report.stopped_reason = "每日指标：" + basic.stopped_reason
        _state.update(
            filled_days=len(report.filled_days),
            empty_days=[d.isoformat() for d in report.empty_days],
            stopped_reason=report.stopped_reason,
        )
    except Exception as exc:  # noqa: BLE001 - 补齐失败要如实报告，不能让后台线程静默退出
        _state.update(stopped_reason=str(exc))
    finally:
        _state.update(running=False, finished_at=datetime.now().isoformat(timespec="seconds"))


def start_refresh() -> bool:
    """开始补齐；已经有一个在跑时不重复开始，返回 False。"""
    with _lock:
        if _state.get("running"):
            return False
        _state.clear()
        _state.update(running=True, started_at=datetime.now().isoformat(timespec="seconds"), done=0, total=None)
    threading.Thread(target=_run, name="market-refresh", daemon=True).start()
    return True


def refresh_state() -> Dict[str, object]:
    return dict(_state)


def market_status() -> Dict[str, object]:
    """行情库现状 + 当前/上一次补齐任务的状态。交易日历拿不到时如实报错。"""
    try:
        status: Dict[str, Optional[object]] = get_market_store().status()
    except Exception as exc:  # noqa: BLE001
        status = {"error": str(exc)}
    status["refresh"] = refresh_state()
    return status
