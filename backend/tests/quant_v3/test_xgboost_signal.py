import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.quant_v3.regression_dataset import REGRESSION_FEATURE_COLUMNS
from app.quant_v3.xgboost_model_training import train_xgboost_fold
from app.quant_v3.xgboost_signal import XGBoostEnsembleSignalSource


def _synthetic_training_frame(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    data = {column: rng.normal(size=n) for column in REGRESSION_FEATURE_COLUMNS}
    data["group"] = "沪深300"
    data["date"] = dates.strftime("%Y-%m-%d")
    data["label_end"] = dates.strftime("%Y-%m-%d")
    data["forward_return"] = rng.normal(scale=0.05, size=n)
    return pd.DataFrame(data)


def _synthetic_history(symbol: str, n: int = 260) -> pd.DataFrame:
    """一支股票足够多天的原始行情，够 compute_features 用（最长回看120日）。"""
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    price = 10 + np.cumsum(rng.normal(scale=0.1, size=n))
    return pd.DataFrame({
        "symbol": symbol,
        "date": dates.strftime("%Y-%m-%d"),
        "open": price, "high": price + 0.1, "low": price - 0.1, "close": price,
        "volume": rng.integers(1000, 5000, size=n),
        "amount": price * rng.integers(1000, 5000, size=n),
        "turnover_rate": rng.uniform(0.5, 3.0, size=n),
        "is_suspended": False,
        "group": "沪深300",
    })


def test_xgboost_ensemble_signal_source_round_trips_saved_model(tmp_path):
    frame = _synthetic_training_frame()
    fold = train_xgboost_fold(frame, "2016-06-01", "2016-01-01", "2016-06-01")

    year_dir = tmp_path / "2024"
    seed_dir = year_dir / "seed0"
    seed_dir.mkdir(parents=True)
    fold.booster.save_model(str(seed_dir / "xgb_model.json"))
    with open(seed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {"feature_columns": REGRESSION_FEATURE_COLUMNS, "group_categories": ["沪深300"]},
            f,
        )

    history = _synthetic_history("600000.SH")
    source = XGBoostEnsembleSignalSource(year_dir, history)

    as_of = date(2024, 12, 20)  # 足够晚，compute_features 有完整回看窗口
    score = source("600000.SH", as_of)
    assert score is not None
    assert isinstance(score, float)
    assert score == score  # not NaN

    # 同一天再查一次，走缓存分支，结果必须一致。
    assert source("600000.SH", as_of) == score

    # 历史上没有的股票/日期，诚实返回 None，不能编分数。
    assert source("999999.SH", as_of) is None
    assert source("600000.SH", date(2000, 1, 1)) is None
