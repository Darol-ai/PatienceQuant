from app.ai.report_factor import sanitize_feature_weights


def test_sanitize_feature_weights_keeps_valid_whitelisted_entries():
    raw = {"return_20d": 0.6, "volatility_60d": -0.3}

    clean = sanitize_feature_weights(raw)

    assert clean == {"return_20d": 0.6, "volatility_60d": -0.3}


def test_sanitize_feature_weights_drops_unknown_feature_names():
    raw = {"return_20d": 0.5, "made_up_feature": 0.9}

    clean = sanitize_feature_weights(raw)

    assert clean == {"return_20d": 0.5}


def test_sanitize_feature_weights_drops_out_of_range_values():
    raw = {"return_20d": 5.0, "volatility_60d": -2.0, "max_drawdown_60d": 0.4}

    clean = sanitize_feature_weights(raw)

    assert clean == {"max_drawdown_60d": 0.4}


def test_sanitize_feature_weights_drops_non_numeric_values():
    raw = {"return_20d": "high", "volatility_60d": 0.2}

    clean = sanitize_feature_weights(raw)

    assert clean == {"volatility_60d": 0.2}
