"""模型库（ADR-0047 第 6、7 条）。

目前只有"旧模型"：此前离线流程逐年滚动训练好的模型，原样导入、不重训，
保证已有成绩可复现。它们的特征都是 V3 方案 3.5 节那 14 项价格/量能特征
（app/quant_v3/dataset.compute_features），外加一个只有一个取值的"配置组"
类别特征。系统内训练的新模型之后也登记到这里。

每个模型目录结构：<year>/seed<i>/{lgbm_model.txt | xgb_model.json, metadata.json}，
<year> 这一折只用该年之前的数据训练，只能给该年打分。
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class ModelEntry:
    id: str
    name: str
    framework: str  # "lightgbm" | "xgboost"
    directory: Path
    origin: str  # "legacy"：旧模型
    trained_on: str
    horizon_days: int
    description: str


MODEL_LIBRARY: Dict[str, ModelEntry] = {
    entry.id: entry
    for entry in [
        ModelEntry(
            id="legacy/csi300_lightgbm",
            name="LightGBM 沪深300（旧模型）",
            framework="lightgbm",
            directory=_DATA_DIR / "csi300_lightgbm_ensemble_model_walkforward",
            origin="legacy",
            trained_on="沪深300成分股",
            horizon_days=90,
            description="14 项价格/量能特征，预测未来 90 个交易日收益，5 个随机种子平均，逐年滚动训练",
        ),
        ModelEntry(
            id="legacy/csi300_xgboost",
            name="XGBoost 沪深300（旧模型）",
            framework="xgboost",
            directory=_DATA_DIR / "csi300_xgboost_ensemble_model_walkforward",
            origin="legacy",
            trained_on="沪深300成分股",
            horizon_days=90,
            description="与 LightGBM 沪深300 旧模型同一套特征和训练规则，模型换成 XGBoost",
        ),
        ModelEntry(
            id="legacy/broad30_lightgbm",
            name="LightGBM 30支候选池（旧模型）",
            framework="lightgbm",
            directory=_DATA_DIR / "broad_regression_h90_ensemble_model_walkforward",
            origin="legacy",
            trained_on="30 支跨行业候选池",
            horizon_days=90,
            description="14 项价格/量能特征，预测未来 90 个交易日收益，5 个随机种子平均，逐年滚动训练",
        ),
    ]
}


Predictor = Callable[[pd.DataFrame], "object"]


@dataclass
class YearFold:
    feature_columns: List[str]
    group_category: Optional[str]  # 旧模型训练时带一个只有一个取值的"配置组"列；新模型没有
    predictors: List[Predictor]

    def predict(self, features: pd.DataFrame) -> pd.Series:
        frame = features[self.feature_columns].copy()
        if self.group_category is not None:
            frame["group"] = pd.Categorical([self.group_category] * len(frame), categories=[self.group_category])
        predictions = [predict(frame) for predict in self.predictors]
        return pd.Series(sum(predictions) / len(predictions), index=features.index, dtype=float)


def _load_seed(framework: str, seed_dir: Path) -> Predictor:
    if framework == "lightgbm":
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(seed_dir / "lgbm_model.txt"))
        return booster.predict
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(seed_dir / "xgb_model.json"))
    return lambda frame: booster.predict(xgb.DMatrix(frame, enable_categorical=True))


_load_lock = threading.Lock()


def _directory_and_framework(model_id: str) -> Tuple[Path, str]:
    if model_id in MODEL_LIBRARY:
        entry = MODEL_LIBRARY[model_id]
        return entry.directory, entry.framework
    from app.training.trainer import model_dir, read_record

    record = read_record(model_id)
    if record is None:
        raise KeyError(f"模型库里没有模型 {model_id}")
    if record.get("status") != "ready":
        raise ValueError(f"模型「{record.get('name', model_id)}」还不能用（状态：{STATUS_TEXT.get(record.get('status'), record.get('status'))}）")
    return model_dir(model_id), record["config"]["framework"]


@lru_cache(maxsize=None)
def _folds_cached(model_id: str) -> Tuple[Tuple[int, YearFold], ...]:
    directory, framework = _directory_and_framework(model_id)
    folds = []
    for year_dir in sorted(directory.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        feature_columns: Optional[List[str]] = None
        categories: List[str] = []
        predictors = []
        for seed_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
            metadata = json.loads((seed_dir / "metadata.json").read_text(encoding="utf-8"))
            feature_columns = feature_columns or metadata["feature_columns"]
            categories = metadata["group_categories"]
            predictors.append(_load_seed(framework, seed_dir))
        if predictors and len(categories) <= 1:
            folds.append((int(year_dir.name), YearFold(feature_columns, categories[0] if categories else None, predictors)))
    return tuple(folds)


def load_folds(model_id: str) -> Dict[int, YearFold]:
    """{年份: 该年的模型}。加载模型文件较慢，进程内只加载一次；加锁避免
    两个请求同时第一次用到同一个模型时各自重复加载一遍。"""
    with _load_lock:
        return dict(_folds_cached(model_id))


STATUS_TEXT = {"queued": "排队中", "training": "训练中", "ready": "可用", "failed": "训练失败", "cancelled": "已取消"}


def model_factors(model_id: str) -> List[str]:
    """模型的输入因子。旧模型固定是那 14 个。"""
    from app.pipeline.factor_library import LEGACY_MODEL_FACTORS

    if model_id in MODEL_LIBRARY:
        return list(LEGACY_MODEL_FACTORS)
    from app.training.trainer import read_record

    record = read_record(model_id) or {}
    return list((record.get("config") or {}).get("factors") or [])


def model_uses_cs_rank(model_id: str) -> bool:
    if model_id in MODEL_LIBRARY:
        return False
    from app.training.trainer import read_record

    return bool(((read_record(model_id) or {}).get("config") or {}).get("cs_rank"))


def model_exists(model_id: str) -> bool:
    if model_id in MODEL_LIBRARY:
        return True
    from app.training.trainer import read_record

    return read_record(model_id) is not None


def model_years(model_id: str) -> List[int]:
    try:
        directory, _ = _directory_and_framework(model_id)
    except (KeyError, ValueError):
        return []
    if not directory.exists():
        return []
    return sorted(int(p.name) for p in directory.iterdir() if p.is_dir() and p.name.isdigit())


def all_models() -> List[dict]:
    """模型库：旧模型 + 系统内训练的模型。"""
    from app.pipeline.factor_library import LEGACY_MODEL_FACTORS
    from app.training.trainer import list_records

    out = [{"id": e.id, "name": e.name, "framework": e.framework, "origin": e.origin, "trained_on": e.trained_on,
            "horizon_days": e.horizon_days, "description": e.description, "factors": list(LEGACY_MODEL_FACTORS),
            "status": "ready", "years": model_years(e.id), "data": "离线流程（baostock 2010 年起，今天的成分股名单）"}
           for e in MODEL_LIBRARY.values()]
    for record in list_records():
        config = record.get("config") or {}
        out.append({"id": record["id"], "name": record.get("name"), "framework": config.get("framework"), "origin": "trained",
                    "trained_on": record.get("pool_label"), "horizon_days": config.get("horizon"),
                    "description": record.get("description", ""), "factors": config.get("factors", []),
                    "status": record.get("status"), "progress": record.get("progress"), "error": record.get("error"),
                    "years": model_years(record["id"]) if record.get("status") == "ready" else [],
                    "metrics": record.get("metrics"), "config": config, "created_at": record.get("created_at"),
                    "finished_at": record.get("finished_at"), "data": "本地行情库（2016 年起）"})
    return out
