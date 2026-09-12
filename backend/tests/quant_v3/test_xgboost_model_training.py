import numpy as np
import pandas as pd
import pytest

from app.quant_v3.regression_dataset import REGRESSION_FEATURE_COLUMNS
from app.quant_v3.xgboost_model_training import train_xgboost_fold


def _synthetic_frame(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    data = {column: rng.normal(size=n) for column in REGRESSION_FEATURE_COLUMNS}
    data["group"] = "沪深300"
    data["date"] = dates.strftime("%Y-%m-%d")
    data["label_end"] = dates.strftime("%Y-%m-%d")
    data["forward_return"] = rng.normal(scale=0.05, size=n)
    return pd.DataFrame(data)


def test_train_xgboost_fold_produces_a_usable_booster():
    frame = _synthetic_frame()
    fold = train_xgboost_fold(frame, train_end="2016-06-01", calib_start="2016-01-01", calib_end="2016-06-01")

    assert fold.booster is not None
    assert fold.metadata["feature_columns"] == REGRESSION_FEATURE_COLUMNS
    assert fold.metadata["train_samples"] > 0


def test_train_xgboost_fold_with_different_seeds_gives_different_trees():
    frame = _synthetic_frame()
    fold_a = train_xgboost_fold(frame, "2016-06-01", "2016-01-01", "2016-06-01", random_state=1)
    fold_b = train_xgboost_fold(frame, "2016-06-01", "2016-01-01", "2016-06-01", random_state=2)

    dump_a = fold_a.booster.get_dump()
    dump_b = fold_b.booster.get_dump()
    assert dump_a != dump_b
