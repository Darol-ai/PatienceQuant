import pytest

from app.quant_v3.labels import (
    compute_sigma_t,
    label_boundaries,
    log_returns_over_window,
    sample_volatility,
    scale_to_20_day_horizon,
)


def test_label_boundaries_matches_v3_worked_example_at_2_percent():
    """V3 方案 3.2 节算例：20日尺度波动率 v 为 2% 时，上下边界约为 +3.25%／−1.98%。"""
    upper, lower = label_boundaries(v_t=0.02)

    assert upper == pytest.approx(0.0325, abs=1e-4)
    assert lower == pytest.approx(-0.0198, abs=1e-4)


def test_label_boundaries_matches_v3_worked_example_at_5_percent():
    """V3 方案 3.2 节算例：v 为 5% 时，上下边界约为 +8.33%／−4.88%。"""
    upper, lower = label_boundaries(v_t=0.05)

    assert upper == pytest.approx(0.0833, abs=1e-4)
    assert lower == pytest.approx(-0.0488, abs=1e-4)


def test_sample_volatility_uses_ddof_1_sample_standard_deviation():
    """V3 方案 3.2 节：标准差取样本标准差 ddof=1。
    手算：收益 [0.0, 0.02, -0.02, 0.04]，均值 0.01，
    离差平方和 = 0.0001+0.0001+0.0009+0.0009 = 0.002，
    样本方差 = 0.002 / (4-1) = 0.0006667，标准差 = sqrt(0.0006667) ≈ 0.02582。"""
    returns = [0.0, 0.02, -0.02, 0.04]

    sigma = sample_volatility(returns)

    assert sigma == pytest.approx(0.02582, abs=1e-4)


def test_scale_to_20_day_horizon_multiplies_by_sqrt_20():
    """V3 方案 3.2 节：v_t = sigma_t * sqrt(20)。手算：0.01 * sqrt(20) ≈ 0.04472。"""
    v_t = scale_to_20_day_horizon(sigma_t=0.01)

    assert v_t == pytest.approx(0.04472, abs=1e-4)


def test_log_returns_over_window_computes_consecutive_log_returns():
    """手算：100→110→121 均为 10% 涨幅，两段对数收益都是 ln(1.1) ≈ 0.09531。"""
    prices = [100.0, 110.0, 121.0]

    returns = log_returns_over_window(prices, window=2)

    assert returns == pytest.approx([0.09531, 0.09531], abs=1e-4)


def test_log_returns_over_window_returns_none_when_not_enough_prices():
    """V3 方案 3.2 节：需要 61 个相邻交易日的有效价格（window+1 个价格）；
    不足时不制作该样本，不能拿现有数据凑数。"""
    prices = [100.0, 110.0]  # window=2 需要 3 个价格，只给了 2 个

    returns = log_returns_over_window(prices, window=2)

    assert returns is None


def test_compute_sigma_t_combines_window_and_sample_volatility():
    """手算：100→110→99 的两段对数收益 ln(1.1)≈0.09531、ln(0.9)≈-0.10536，
    均值≈-0.00503，离差平方和≈0.02013，ddof=1 方差≈0.02013，标准差≈0.14189。"""
    prices = [100.0, 110.0, 99.0]

    sigma_t = compute_sigma_t(prices, window=2)

    assert sigma_t == pytest.approx(0.14189, abs=1e-4)


def test_compute_sigma_t_returns_none_when_all_returns_identical():
    """V3 方案 3.2 节：标准差为零或非有限值时不制作该样本，不强行赋 NEUTRAL。"""
    prices = [100.0, 110.0, 121.0]  # 两段对数收益都是 ln(1.1)，标准差为 0

    sigma_t = compute_sigma_t(prices, window=2)

    assert sigma_t is None


def test_compute_sigma_t_returns_none_when_not_enough_prices():
    prices = [100.0, 110.0]  # window=2 需要 3 个价格

    sigma_t = compute_sigma_t(prices, window=2)

    assert sigma_t is None
