"""一键复现：训练内置模型、计算全部成绩卡，并与仓库里的成绩卡快照逐项对比。

在 backend/ 目录下运行（需要先 `python scripts/fetch_data.py` 下载数据包）：

    python scripts/reproduce.py              # 复现并对比，约 1 小时（GPU）
    python scripts/reproduce.py --export     # 从当前数据库导出成绩卡快照（维护者用）

对比按策略的 kind（内置策略的固定标识）对齐。XGBoost 在 GPU 和 CPU 上的结果不逐位相同，
LSTM 在不同显卡上也可能有细微差别；树模型 LightGBM 和因子策略应当完全一致。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SNAPSHOT = Path(__file__).resolve().parents[2] / "docs" / "成绩卡快照.json"
METRICS = ("total_return", "annual_return", "max_drawdown", "sharpe", "benchmark_return")


def _prepare_database():
    from app.db.models import Base
    from app.db.seed import seed_database
    from app.db.session import SessionLocal, engine, migrate_lightweight_schema

    Base.metadata.create_all(bind=engine)
    migrate_lightweight_schema()
    with SessionLocal() as db:
        seed_database(db)


def _cards():
    from app.db.session import SessionLocal
    from app.pipeline import scorecard

    with SessionLocal() as db:
        return scorecard.all_cards(db)


def export() -> None:
    cards = [c for c in _cards() if c["status"] == "ready"]
    import lightgbm
    import xgboost

    payload = {
        "说明": "成绩卡快照：标准条件（2019-01-01 ~ 2025-12-31、初始 100 万、手续费 0.1%、滑点 0.05%、默认股票池）。按 kind 对齐。",
        "导出时间": datetime.now().isoformat(timespec="seconds"),
        "环境": {"python": platform.python_version(), "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__,
                 "行情截至": "2026-09-23"},
        "strategies": {c["kind"]: {"name": c["name"], **{k: c["metrics"][k] for k in METRICS}} for c in cards},
    }
    try:
        import torch

        payload["环境"]["torch"] = torch.__version__
        payload["环境"]["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    SNAPSHOT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已导出 {len(cards)} 张成绩卡到 {SNAPSHOT}")


def reproduce() -> int:
    from app.pipeline import scorecard
    from app.pipeline.library import TRAINED_BUILTINS, trained_builtin_config
    from app.training import jobs
    from app.training.trainer import read_record

    t0 = time.time()
    print("1/3 初始化数据库，登记内置策略……", flush=True)
    _prepare_database()
    jobs.recover_interrupted()

    print("2/3 训练内置模型（已有且可用的跳过）……", flush=True)
    for item in TRAINED_BUILTINS:
        model_id, _, queued = jobs.ensure_model(trained_builtin_config(item), f"内置策略「{item['name']}」的模型")
        print(f"   {item['name']}：{'排队训练' if queued else '已有'}（{model_id}）", flush=True)
    last = None
    while True:
        records = [read_record(f"trained/{trained_builtin_config(i).fingerprint()}") or {} for i in TRAINED_BUILTINS]
        status = [(r.get("name"), r.get("status"), r.get("progress")) for r in records]
        if status != last:
            for name, st, progress in status:
                if st in ("queued", "training"):
                    print(f"   {name}：{st} {progress or ''}", flush=True)
            last = status
        if all(r.get("status") not in ("queued", "training") for r in records):
            break
        time.sleep(15)
    failed = [r.get("name") for r in records if r.get("status") != "ready"]
    if failed:
        print(f"   这些模型没有训练成功：{failed}（见 backend/data/models/*/model.json 的 error）", flush=True)

    print("3/3 计算全部成绩卡……", flush=True)
    while not scorecard.start_refresh(None):  # 训练完成后可能已经有一次成绩卡计算在跑，等它结束
        time.sleep(10)
    while scorecard.refresh_state()["running"]:
        state = scorecard.refresh_state()
        print(f"\r   {state['done']}/{state['total']} {state['current'] or ''}".ljust(60), end="", flush=True)
        time.sleep(10)
    print()
    for error in scorecard.refresh_state()["errors"]:
        print(f"   失败：{error['name']}：{error['error']}", flush=True)

    return compare(time.time() - t0)


def compare(seconds: float) -> int:
    if not SNAPSHOT.exists():
        print(f"没有找到成绩卡快照 {SNAPSHOT}")
        return 1
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["strategies"]
    cards = {c["kind"]: c for c in _cards()}
    rows, exact, close, differ, missing = [], 0, 0, 0, 0
    for kind, expected in snapshot.items():
        card = cards.get(kind)
        if card is None or card["status"] != "ready":
            missing += 1
            rows.append((expected["name"], "—", f"{expected['annual_return']:.2%}", "没有算出成绩卡"))
            continue
        got = card["metrics"]
        gap = max(abs(got[k] - expected[k]) for k in METRICS)
        verdict = "一致" if gap < 1e-9 else ("接近（<0.5 个百分点）" if gap < 0.005 else "不同")
        exact += verdict == "一致"
        close += verdict.startswith("接近")
        differ += verdict == "不同"
        rows.append((expected["name"], f"{got['annual_return']:.2%}", f"{expected['annual_return']:.2%}", verdict))
    width = max(len(r[0]) for r in rows)
    print(f"\n{'策略'.ljust(width)}  复现年化   快照年化   结果")
    for name, got, want, verdict in rows:
        print(f"{name.ljust(width)}  {got:>8}  {want:>8}   {verdict}")
    print(f"\n共 {len(rows)} 个策略：一致 {exact}，接近 {close}，不同 {differ}，缺失 {missing}。用时 {seconds / 60:.0f} 分钟。")
    report = Path("reproduce_report.json")
    report.write_text(json.dumps({"rows": rows, "exact": exact, "close": close, "differ": differ, "missing": missing},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果另存为 backend/{report}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", action="store_true", help="从当前数据库导出成绩卡快照")
    parser.add_argument("--compare-only", action="store_true", help="不训练不计算，只和快照对比")
    args = parser.parse_args()
    if args.export:
        export()
    elif args.compare_only:
        sys.exit(compare(0))
    else:
        sys.exit(reproduce())
