from datetime import date

from app.quant_v3.blended_signal import BlendedSignalSource


def test_blended_signal_averages_available_sources():
    source = BlendedSignalSource([
        lambda symbol, as_of: 0.1,
        lambda symbol, as_of: 0.3,
    ])
    assert source("600000.SH", date(2024, 1, 1)) == 0.2


def test_blended_signal_ignores_missing_sources_rather_than_treating_as_zero():
    source = BlendedSignalSource([
        lambda symbol, as_of: None,
        lambda symbol, as_of: 0.4,
    ])
    assert source("600000.SH", date(2024, 1, 1)) == 0.4


def test_blended_signal_returns_none_when_all_sources_missing():
    source = BlendedSignalSource([
        lambda symbol, as_of: None,
        lambda symbol, as_of: None,
    ])
    assert source("600000.SH", date(2024, 1, 1)) is None
