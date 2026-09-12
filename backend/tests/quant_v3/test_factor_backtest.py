from datetime import date, timedelta

import pandas as pd
import pytest

from app.quant_v3.factor_backtest import GeneratedFactorStrategy, top_n_equal_weight


def test_top_n_equal_weight_picks_highest_scores():
    scores = {"A": 0.5, "B": 0.9, "C": 0.1, "D": 0.7}

    weights = top_n_equal_weight(scores, top_n=2)

    assert weights == {"B": 0.5, "D": 0.5}


def test_top_n_equal_weight_breaks_ties_by_symbol_ascending():
    scores = {"600003.SH": 0.9, "600004.SH": 0.5, "600002.SH": 0.5, "600001.SH": 0.5}

    weights = top_n_equal_weight(scores, top_n=3)

    assert set(weights) == {"600003.SH", "600001.SH", "600002.SH"}


def _make_history(symbol_closes: dict) -> pd.DataFrame:
    rows = []
    for symbol, closes in symbol_closes.items():
        for i, close in enumerate(closes):
            trade_date = date(2024, 1, 1) + timedelta(days=i)
            rows.append({
                "date": trade_date.isoformat(), "symbol": symbol,
                "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                "volume": 1_000_000, "amount": close * 1_000_000, "is_suspended": False,
            })
    return pd.DataFrame(rows)


def test_generated_factor_strategy_ranks_by_weighted_feature_sum():
    """两支股票 121 天历史刚好够算特征；AAA 最后几天明显上涨，BBB 全程持平。
    只用 return_5d、权重为正时，应该选中涨得更多的 AAA。"""
    history = _make_history({
        "AAA": [100.0] * 115 + [100, 101, 102, 103, 104, 110],
        "BBB": [100.0] * 121,
    })
    strategy = GeneratedFactorStrategy(feature_weights={"return_5d": 1.0}, history=history, top_n=1)
    as_of = date(2024, 1, 1) + timedelta(days=120)

    result = strategy.generate_weights(as_of, symbols=["AAA", "BBB"])

    assert result.weights == {"AAA": 1.0}


def test_generated_factor_strategy_skips_symbols_without_enough_history():
    history = _make_history({"AAA": [100.0] * 50})  # 不够 120 天历史
    strategy = GeneratedFactorStrategy(feature_weights={"return_5d": 1.0}, history=history, top_n=1)
    as_of = date(2024, 1, 1) + timedelta(days=49)

    result = strategy.generate_weights(as_of, symbols=["AAA"])

    assert result.weights == {}
