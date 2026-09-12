from datetime import date

import pandas as pd

from app.quant_v3.qualification_signal import QualificationSignalSource


def _history(symbol: str, amounts: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(amounts))
    return pd.DataFrame({
        "date": dates, "symbol": symbol, "amount": amounts,
        "volume": [1000] * len(amounts), "is_suspended": [False] * len(amounts),
    })


def test_qualified_symbol_with_enough_liquid_history():
    history = _history("601288.SH", [2e8] * 65)
    source = QualificationSignalSource(history)

    as_of = history["date"].iloc[-1].date()
    assert source("601288.SH", as_of) is True


def test_not_qualified_below_liquidity_threshold():
    history = _history("601288.SH", [5e7] * 65)
    source = QualificationSignalSource(history)

    as_of = history["date"].iloc[-1].date()
    assert source("601288.SH", as_of) is False


def test_unknown_symbol_or_date_is_not_qualified():
    history = _history("601288.SH", [2e8] * 65)
    source = QualificationSignalSource(history)

    assert source("999999.SH", history["date"].iloc[-1].date()) is False
    assert source("601288.SH", date(2099, 1, 1)) is False
