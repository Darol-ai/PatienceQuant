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


class _History:
    """最小的数据服务：两支股票、一段交易日，价格恒定（只看交易发生在哪天）。"""

    def __init__(self, days):
        self.days = days

    def stocks(self):
        return pd.DataFrame({"symbol": ["a", "b"]})

    def prices(self, symbols, start, end, allow_network=False):
        rows = [{"trade_date": d, "symbol": s, "adj_close": 10.0} for d in self.days for s in symbols if start <= d <= end]
        return pd.DataFrame(rows)

    def benchmark(self, start, end):
        return pd.DataFrame({"trade_date": [d for d in self.days if start <= d <= end], "adj_close": 1.0})

    def frame_data_mode(self, frame):
        return "real"


class SwitchTiming(TimingSignal):
    """1 月 15 日（含）之后看空。"""

    def exposure(self, as_of):
        return TimingResult(0.0, "risk_off") if as_of >= date(2024, 1, 15) else TimingResult(1.0, "risk_on")


def test_timing_takes_effect_on_non_rebalance_days():
    from app.backtest.engine import BacktestConfig, BacktestEngine
    from app.strategies.base import StrategyConfig

    days = [d.date() for d in pd.bdate_range("2023-12-25", "2024-02-29")]
    spec = StrategySpec.model_validate({"scorer": {"type": "model", "models": ["legacy/csi300_lightgbm"]},
                                        "timing": {"type": "rsrs"}, "selection": {"type": "top_n", "n": 2}})
    strategy = PipelineStrategy(spec, FixedScorer({"a": 2.0, "b": 1.0}), SwitchTiming())
    result = BacktestEngine(_History(days)).run(
        BacktestConfig(start_date=days[0], end_date=days[-1], rebalance_frequency="monthly", commission=0, slippage=0),
        StrategyConfig(stop_loss=0, turnover_band=0, target_volatility=0, max_drawdown_budget=0), ["a", "b"], strategy=strategy)
    trades = result.trades
    # 12 月底调仓 → 下一个交易日（bdate_range 里是 1 月 1 日）买入；
    # 1 月 15 日收盘看空 → 16 日（不是调仓日）清仓；1 月底调仓时仍看空，不买
    buys = trades[trades.side == "BUY"]
    sells = trades[trades.side == "SELL"]
    assert set(buys.trade_date) == {date(2024, 1, 1)}
    assert set(sells.trade_date) == {date(2024, 1, 16)}
    assert sells.reason.str.contains("择时降仓至 0%").all()
    assert result.risk_summary["timing_adjustments"] == 1
    assert 0 < result.risk_summary["average_target_exposure"] < 0.3


def test_membership_limits_scoring_to_members_of_the_day():
    spec = StrategySpec.model_validate({"scorer": {"type": "model", "models": ["legacy/csi300_lightgbm"]},
                                        "selection": {"type": "top_pct", "pct": 0.5}})

    class RecordingScorer(FixedScorer):
        def score(self, as_of, symbols):
            self.seen = list(symbols)
            return super().score(as_of, symbols)

    scorer = RecordingScorer({"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.5})
    strategy = PipelineStrategy(spec, scorer, NoTiming())
    strategy.members_at = lambda day: {"b", "c"} if day < date(2024, 6, 1) else {"a", "b", "c", "d"}
    early = strategy.generate_weights(date(2024, 1, 2), ["a", "b", "c", "d"])
    assert scorer.seen == ["b", "c"]
    assert early.weights == pytest.approx({"b": 1.0})  # 两支成员的前 50% = 1 支；a 分数最高但当时不在指数里
    late = strategy.generate_weights(date(2024, 7, 1), ["a", "b", "c", "d"])
    assert set(late.weights) == {"a", "b"}
