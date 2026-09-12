import pytest

from app.quant_v3.regression_dataset import REGRESSION_FEATURE_COLUMNS, build_regression_sample, forward_return


def test_forward_return_is_close_to_close_pct_change_over_horizon():
    closes = [100.0, 101.0, 99.0, 105.0, 110.0]
    # t_index=0(100.0)，horizon=3 -> bars[3]=105.0
    assert forward_return(closes, t_index=0, horizon=3) == pytest.approx(0.05)


def test_forward_return_none_when_horizon_exceeds_available_bars():
    closes = [100.0, 101.0]
    assert forward_return(closes, t_index=0, horizon=5) is None


def _oscillating_closes(n: int) -> list[float]:
    pattern = [100.0, 101.0, 99.0]
    return [pattern[i % 3] for i in range(n)]


def _make_bars(days: int) -> list[dict]:
    closes = _oscillating_closes(days)
    return [
        {
            "date": f"2020-01-{i + 1:02d}", "open": closes[i], "high": closes[i] * 1.01, "low": closes[i] * 0.99,
            "close": closes[i], "volume": 1000, "amount": closes[i] * 1000, "is_suspended": False,
        }
        for i in range(days)
    ]


def test_build_regression_sample_has_features_and_forward_return_label():
    bars = _make_bars(200)

    sample = build_regression_sample(bars, t_index=150, group="综合", horizon=20)

    assert sample is not None
    assert set(REGRESSION_FEATURE_COLUMNS) <= set(sample)
    assert "forward_return" in sample
    assert "date" in sample and "group" in sample


def test_build_regression_sample_none_when_future_window_not_available():
    bars = _make_bars(160)

    assert build_regression_sample(bars, t_index=150, group="综合", horizon=20) is None
