"""系统内训练（ADR-0052）：按固定的滚动规则训练 LightGBM / XGBoost，产物登记进模型库。

规则写死（与旧模型相同，保证可比、防止偷看未来）：给第 Y 年打分的模型，用"标签在
Y−1 年 7 月 1 日之前已经揭晓"的样本训练，用 Y−1 年下半年的样本做验证（早停 30 轮），
在第 Y 年上预测。每个种子训练一个模型，预测时取平均。
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import os

from app.data.market_store import STORE_START

# 训练出的模型放在 data/models/<id>/；测试时用 PATIENCEQUANT_MODELS_DIR 指到临时目录
MODELS_ROOT = Path(os.environ.get("PATIENCEQUANT_MODELS_DIR") or Path(__file__).resolve().parents[2] / "data" / "models")
SEEDS = [42, 7, 123, 2024, 999]  # 与旧模型相同的种子顺序
FIRST_TEST_YEAR = 2012  # 数据 2010 年起；训练样本不足 1000 条的年份自动跳过
DATA_START = STORE_START


@dataclass
class TrainingConfig:
    framework: str  # lightgbm | xgboost
    factors: List[str]
    horizon: int = 90
    pool: str = "csi300"
    # historical：训练样本只取当天在指数里的股票；latest：用今天的名单（只用于和旧模型对比）
    membership: str = "historical"
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 15  # LightGBM
    max_depth: int = 4
    seeds: int = 5
    cs_rank: bool = False  # 因子先换成当天池内百分位

    def validate(self) -> "TrainingConfig":
        from app.pipeline.factor_library import trainable

        if self.framework not in ("lightgbm", "xgboost", "lstm"):
            raise ValueError("算法只能是 LightGBM、XGBoost 或 LSTM")
        bad = [f for f in self.factors if not trainable(f)]
        if not self.factors or bad:
            raise ValueError("输入因子不能为空，且必须是因子库里有真实数据的因子：%s" % "、".join(bad))
        if self.horizon not in (20, 60, 90):
            raise ValueError("预测周期只能是 20 / 60 / 90 个交易日")
        if self.membership not in ("historical", "latest"):
            raise ValueError("membership 只能是 historical 或 latest")
        if not (1 <= self.seeds <= 5) or not (50 <= self.n_estimators <= 2000) or not (0.001 <= self.learning_rate <= 0.5):
            raise ValueError("超参数超出允许范围")
        if not (4 <= self.num_leaves <= 255) or not (2 <= self.max_depth <= 12):
            raise ValueError("超参数超出允许范围")
        self.factors = list(dict.fromkeys(self.factors))
        return self

    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["factors"] = sorted(payload["factors"])
        payload["data"] = f"store-{DATA_START.year}"
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


# ---- 模型登记（文件即登记：data/models/<id>/model.json）----

def model_dir(model_id: str) -> Path:
    return MODELS_ROOT / model_id.split("/", 1)[-1]


def read_record(model_id: str) -> Optional[dict]:
    path = model_dir(model_id) / "model.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_record(record: dict) -> None:
    path = model_dir(record["id"]) / "model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def list_records() -> List[dict]:
    if not MODELS_ROOT.exists():
        return []
    records = []
    for path in sorted(MODELS_ROOT.glob("*/model.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return records


def find_by_fingerprint(fingerprint: str) -> Optional[dict]:
    return next((r for r in list_records() if r.get("fingerprint") == fingerprint and r.get("status") != "failed"
                 and r.get("status") != "cancelled"), None)


# ---- 样本 ----

def _pool_panels(config: TrainingConfig, until: date, progress: Callable[[str], None]):
    """训练股票池的日线面板和输入因子面板，返回 (因子面板, 日线面板, 历史成分股或 None)。"""
    from app.data.market_refresh import BENCHMARK_INDEX, get_market_store
    from app.pipeline.factor_library import compute_factors, panels_from_store
    from app.pipeline.library import broad30_symbols, csi300_membership, csi300_symbols

    store = get_market_store()
    membership = None
    if config.pool == "csi300":
        membership = csi300_membership() if config.membership == "historical" else None
        symbols = membership.union(DATA_START, until) if membership else csi300_symbols()
    elif config.pool == "broad30":
        symbols = broad30_symbols()
    elif config.pool.startswith("custom:"):
        from app.db.models import Watchlist
        from app.db.session import SessionLocal

        with SessionLocal() as db:
            row = db.get(Watchlist, int(config.pool.split(":", 1)[1]))
            symbols = list(row.symbols) if row else []
    else:
        raise ValueError("不支持的训练股票池：%s" % config.pool)
    if len(symbols) < 10:
        raise ValueError("训练股票池至少需要 10 支股票")
    progress("读取行情")
    bench = store.load_index(BENCHMARK_INDEX, DATA_START, until)
    bench_series = pd.Series(bench["close"].to_numpy(dtype=float), index=pd.to_datetime(bench["trade_date"]))
    panels = panels_from_store(store, symbols, DATA_START, until, benchmark=bench_series)
    progress("计算因子")
    return compute_factors(panels, config.factors), panels, membership


def build_samples(config: TrainingConfig, until: date, progress: Callable[[str], None] = lambda _: None) -> pd.DataFrame:
    """每个交易日、每支（当天在股票池里的）股票一行：输入因子 + 未来 horizon 个交易日的收益。"""
    values, panels, membership = _pool_panels(config, until, progress)
    close = panels["close"]
    label = close.shift(-config.horizon) / close - 1
    label_end = pd.Series(close.index, index=close.index).shift(-config.horizon)
    columns = {key: frame.stack() for key, frame in values.items()}
    columns["forward_return"] = label.stack()
    frame = pd.concat(columns, axis=1)
    frame.index.names = ["date", "symbol"]
    frame = frame.reset_index()
    frame["label_end"] = frame["date"].map(label_end)
    if membership is not None:
        keep = [symbol in membership.at(day.date()) for day, symbol in zip(frame["date"], frame["symbol"])]
        frame = frame[keep]
    frame = frame.dropna(subset=config.factors)
    # 标签已到期却算不出收益的（窗口内退市、被吸收合并，如 601989、600837），拿不到真实收益，只能丢掉；
    # 历史成分股口径下约占 0.04%。标签还没到期的行保留（label_end 为空），训练时自然不会用到。
    frame = frame[frame["label_end"].isna() | np.isfinite(frame["forward_return"])]
    if config.cs_rank:
        for key in config.factors:
            frame[key] = frame.groupby("date")[key].rank(pct=True)
    return frame.reset_index(drop=True)


# ---- 训练与评估 ----

def _fit(config: TrainingConfig, seed: int, x_train, y_train, x_calib, y_calib):
    if config.framework == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMRegressor(objective="regression", num_leaves=config.num_leaves, max_depth=config.max_depth,
                                  learning_rate=config.learning_rate, min_child_samples=100, n_estimators=config.n_estimators,
                                  random_state=seed, n_jobs=4, verbosity=-1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
        kwargs = dict(callbacks=[lgb.log_evaluation(0)])
        if len(x_calib) >= 60:
            kwargs["eval_set"] = [(x_calib, y_calib)]
            kwargs["callbacks"] = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
        model.fit(x_train, y_train, **kwargs)
        return model.booster_, "lgbm_model.txt", lambda booster, path: booster.save_model(str(path))
    import xgboost as xgb

    model = xgb.XGBRegressor(objective="reg:squarederror", max_depth=config.max_depth, learning_rate=config.learning_rate,
                             min_child_weight=100, n_estimators=config.n_estimators, subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=4, tree_method="hist", device=_xgb_device(), verbosity=0,
                             enable_categorical=True)
    kwargs = {}
    if len(x_calib) >= 60:
        kwargs = dict(eval_set=[(x_calib, y_calib)], verbose=False)
        model.set_params(early_stopping_rounds=30)
    model.fit(x_train, y_train, **kwargs)
    return model.get_booster(), "xgb_model.json", lambda booster, path: booster.save_model(str(path))


def _xgb_device() -> str:
    """有显卡就用 GPU；PATIENCEQUANT_XGB_DEVICE=cpu 可强制 CPU（GPU 和 CPU 的结果不逐位相同）。"""
    import os

    forced = os.environ.get("PATIENCEQUANT_XGB_DEVICE")
    if forced:
        return forced
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def evaluate_year(frame: pd.DataFrame, predictions: np.ndarray) -> dict:
    """样本外成绩：逐日 IC（预测与实际未来收益的秩相关）的均值和 IR，以及每天分数前 10% 相对当天全池的平均超额收益。"""
    scored = frame.assign(pred=predictions)
    ics, excess = [], []
    for _, day in scored.groupby("date"):
        if len(day) < 10:
            continue
        ic = day["pred"].rank().corr(day["forward_return"].rank())
        if np.isfinite(ic):
            ics.append(ic)
        top = day[day["pred"] >= day["pred"].quantile(0.9)]
        excess.append(top["forward_return"].mean() - day["forward_return"].mean())
    ic_mean = float(np.mean(ics)) if ics else None
    ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else None
    return {"days": len(ics), "ic": ic_mean, "icir": (ic_mean / ic_std) if ic_mean is not None and ic_std else None,
            "top10_excess": float(np.mean(excess)) if excess else None, "samples": int(len(frame))}


def train(config: TrainingConfig, model_id: str, until: date, progress: Callable[[str], None],
          cancelled: Callable[[], bool] = lambda: False, frame: Optional[pd.DataFrame] = None) -> dict:
    """训练全部年份的折，保存到 data/models/<id>/<year>/seed<i>/，返回逐年样本外成绩。"""
    if config.framework == "lstm":
        from app.training.lstm import train_lstm

        return train_lstm(config, model_id, until, progress, cancelled)
    frame = frame if frame is not None else build_samples(config, until, progress)
    root = model_dir(model_id)
    results = {}
    years = list(range(FIRST_TEST_YEAR, until.year + 1))
    for year in years:
        train_end = pd.Timestamp(f"{year - 1}-07-01")
        calib_start, calib_end = pd.Timestamp(f"{year - 1}-07-01"), pd.Timestamp(f"{year - 1}-12-31")
        train_rows = frame[frame["label_end"] < train_end]
        calib_rows = frame[(frame["date"] >= calib_start) & (frame["date"] <= calib_end) & (frame["label_end"] < calib_end)]
        test_rows = frame[(frame["date"].dt.year == year) & frame["forward_return"].notna()]
        if len(train_rows) < 1000:
            results[str(year)] = {"skipped": f"训练样本只有 {len(train_rows)} 条，不训练这一年"}
            continue
        x_train, y_train = train_rows[config.factors], train_rows["forward_return"]
        x_calib, y_calib = calib_rows[config.factors], calib_rows["forward_return"]
        predictions = []
        for index, seed in enumerate(SEEDS[:config.seeds]):
            if cancelled():
                raise InterruptedError("训练已取消")
            progress(f"{year} 年 · 种子 {index + 1}/{config.seeds}")
            booster, filename, save = _fit(config, seed, x_train, y_train, x_calib, y_calib)
            out = root / str(year) / f"seed{index}"
            out.mkdir(parents=True, exist_ok=True)
            save(booster, out / filename)
            (out / "metadata.json").write_text(json.dumps({"feature_columns": config.factors, "group_categories": []},
                                                          ensure_ascii=False), encoding="utf-8")
            if len(test_rows):
                predictions.append(_predict(config.framework, booster, test_rows[config.factors]))
        results[str(year)] = {"train_samples": int(len(train_rows)), "calib_samples": int(len(calib_rows)),
                              **(evaluate_year(test_rows, np.mean(predictions, axis=0)) if predictions else {"samples": 0})}
    return results


def _predict(framework: str, booster, x: pd.DataFrame) -> np.ndarray:
    if framework == "lightgbm":
        return booster.predict(x)
    import xgboost as xgb

    return booster.predict(xgb.DMatrix(x, enable_categorical=True))
