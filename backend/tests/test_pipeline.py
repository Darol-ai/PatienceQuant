from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from app.data.market_store import AShareMarketStore
from app.pipeline.bars import load_bars
from app.pipeline.portfolio import select, weigh
from app.pipeline.scorers import ScoreResult, Scorer
from app.pipeline.spec import SelectionSpec, StrategySpec, WeightingSpec
from app.pipeline.strategy import PipelineStrategy
from app.pipeline.timing import NoTiming, TimingResult, TimingSignal


def test_spec_rejects_missing_selection_size_and_unknown_scorer():
    with pytest.raises(ValidationError):
        StrategySpec.model_validate({"scorer": {"type": "model", "models": ["x"]}, "selection": {"type": "top_n"}})
    with pytest.raises(ValidationError):
        StrategySpec.model_validate({"scorer": {"type": "magic"}, "selection": {"type": "top_n", "n": 3}})
    spec = StrategySpec.model_validate({"scorer": {"type": "factor_weights", "weights": {"momentum": 1}},
                                        "selection": {"type": "top_pct", "pct": 0.1}})
    assert spec.timing.type == "none" and spec.weighting.type == "equal" and spec.rebalance.frequency == "monthly"


def test_top_pct_counts_slots_from_universe_size_and_skips_unscored():
    scores = pd.Series({"a": 3.0, "b": None, "c": 1.0, "d": 2.0, "e": None})
    picked = select(scores, SelectionSpec(type="top_pct", pct=0.4), universe_size=5)
    assert list(picked.index) == ["a", "d"]
    assert list(select(scores, SelectionSpec(type="top_n", n=10), 5).index) == ["a", "d", "c"]


def test_equal_weights_respect_cap_and_leave_cash():
    selected = pd.Series({"a": 3.0, "b": 2.0, "c": 1.0})
    assert weigh(selected, selected, WeightingSpec(type="equal")) == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
    capped = weigh(selected, selected, WeightingSpec(type="equal", max_weight=0.2))
    assert capped == pytest.approx({"a": 0.2, "b": 0.2, "c": 0.2})


def test_score_weights_shift_negative_scores_and_redistribute_over_cap():
    all_scores = pd.Series({"a": 0.05, "b": 0.01, "c": -0.03, "d": -0.07})
    selected = all_scores.head(3)
    weights = weigh(selected, all_scores, WeightingSpec(type="score"))
    # 按全截面最低分 -0.07 平移：0.12 / 0.08 / 0.04
    assert weights == pytest.approx({"a": 0.5, "b": 1 / 3, "c": 1 / 6})
    capped = weigh(selected, all_scores, WeightingSpec(type="score", max_weight=0.4))
    assert capped["a"] == pytest.approx(0.4)
    assert sum(capped.values()) == pytest.approx(1.0)
    assert capped["b"] / capped["c"] == pytest.approx(2.0)


class FixedScorer(Scorer):
    def __init__(self, scores):
        self.scores = scores

    def score(self, as_of, symbols):
        return ScoreResult(scores=pd.Series(self.scores), notes=["scored"])


class HalfTiming(TimingSignal):
    def exposure(self, as_of):
        return TimingResult(0.5, "risk_off", "半仓")


def test_pipeline_multiplies_weights_by_timing_exposure():
    spec = StrategySpec.model_validate({"scorer": {"type": "model", "models": ["legacy/csi300_lightgbm"]},
                                        "selection": {"type": "top_n", "n": 2}})
    strategy = PipelineStrategy(spec, FixedScorer({"a": 3.0, "b": 2.0, "c": 1.0}), HalfTiming())
    result = strategy.generate_weights(date(2024, 1, 2), ["a", "b", "c", "z"])
    assert result.weights == pytest.approx({"a": 0.25, "b": 0.25})
    assert result.target_exposure == 0.5
    ranking = result.ranking.set_index("symbol")
    assert ranking.loc["a", "action"] == "BUY" and ranking.loc["c", "action"] == "WATCH"
    assert pd.isna(ranking.loc["z", "score"]) and pd.isna(ranking.loc["z", "rank"])
    assert result.data_quality_notes == ["scored", "半仓"]

    full = PipelineStrategy(spec, FixedScorer({"a": 3.0, "b": 2.0}), NoTiming())
    assert sum(full.generate_weights(date(2024, 1, 2), ["a", "b"]).weights.values()) == pytest.approx(1.0)


class FakePro:
    DAYS = ["20240102", "20240103", "20240104"]

    def trade_cal(self, exchange, start_date, end_date):
        days = pd.date_range("2024-01-01", "2024-12-31").strftime("%Y%m%d")
        return pd.DataFrame({"cal_date": days, "is_open": [1 if d in self.DAYS else 0 for d in days]})

    def daily(self, trade_date):
        codes = ["000001.SZ"] if trade_date == "20240103" else ["000001.SZ", "600000.SH"]  # 600000 在 0103 停牌
        return pd.DataFrame({"ts_code": codes, "trade_date": trade_date, "open": 10.0, "high": 11.0, "low": 9.0,
                             "close": 10.0, "vol": 5.0, "amount": 2.0})

    def adj_factor(self, trade_date):
        return pd.DataFrame({"ts_code": ["000001.SZ", "600000.SH"], "adj_factor": 1.0})


def test_bars_convert_units_and_fill_suspended_days(tmp_path):
    store = AShareMarketStore(root=tmp_path, runner=lambda fn: fn(FakePro()), start=date(2024, 1, 1))
    store.backfill(until=date(2024, 1, 4))

    bars = load_bars(store, ["600000", "000001"], date(2024, 1, 1), date(2024, 1, 4))

    suspended = bars["600000"]
    assert [b["date"] for b in suspended] == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    assert [b["is_suspended"] for b in suspended] == [False, True, False]
    assert suspended[1]["close"] == 10.0 and suspended[1]["volume"] == 0.0
    assert suspended[0]["volume"] == 500.0  # 手 → 股
    assert suspended[0]["amount"] == 2000.0  # 千元 → 元
