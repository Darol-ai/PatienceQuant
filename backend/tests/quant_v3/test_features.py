import pytest

from app.quant_v3.features import (
    atr_ratio,
    average_amount,
    average_turnover,
    close_vs_ma_deviation,
    max_drawdown_over,
    return_over_window,
    valid_trading_ratio,
    valuation_percentile,
    volatility_over,
    volume_ratio,
)


def test_return_over_window_computes_simple_pct_change():
    assert return_over_window([100.0, 105.0], window=1) == pytest.approx(0.05)


def test_return_over_window_none_when_not_enough_prices():
    assert return_over_window([100.0], window=5) is None


def test_close_vs_ma_deviation_uses_trailing_window_including_today():
    """MA(4) 用最近 4 个收盘价（含今天）：[10,10,10,15] 均值 11.25，
    今天收盘 15 相对偏离 15/11.25-1 ≈ 0.3333。"""
    prices = [10.0, 10.0, 10.0, 10.0, 15.0]

    deviation = close_vs_ma_deviation(prices, window=4)

    assert deviation == pytest.approx(0.3333, abs=1e-4)


def test_max_drawdown_over_tracks_the_worst_drop_from_a_running_peak():
    """[100,120,90,110]：峰值 120，最低点 90，最大回撤 90/120-1=-0.25。"""
    prices = [100.0, 120.0, 90.0, 110.0]

    assert max_drawdown_over(prices, window=4) == pytest.approx(-0.25)


def test_volume_ratio_compares_latest_to_trailing_average():
    """[10,10,10,20] 均值 12.5，最新一天 20，比值 20/12.5=1.6。"""
    volumes = [10, 10, 10, 20]

    assert volume_ratio(volumes, window=4) == pytest.approx(1.6)


def test_atr_ratio_uses_true_range_over_close():
    """单日真实波幅 TR = max(高-低, |高-昨收|, |低-昨收|)：
    high=106, low=96, prev_close=100 → TR = max(10, 6, 4) = 10；
    ATR(1)/收盘价 = 10/101 ≈ 0.09901。"""
    highs = [105.0, 106.0]
    lows = [95.0, 96.0]
    closes = [100.0, 101.0]

    ratio = atr_ratio(highs, lows, closes, window=1)

    assert ratio == pytest.approx(0.09901, abs=1e-4)


def test_volatility_over_is_sample_stdev_of_log_returns():
    """复用 test_labels.py 里已经手算验证过的同一个例子：100→110→99 的两段
    对数收益 ln(1.1)≈0.09531、ln(0.9)≈-0.10536，ddof=1 标准差 ≈ 0.14189。"""
    prices = [100.0, 110.0, 99.0]

    sigma = volatility_over(prices, window=2)

    assert sigma == pytest.approx(0.14189, abs=1e-4)


def test_volatility_over_none_when_not_enough_prices():
    assert volatility_over([100.0, 101.0], window=60) is None


def test_average_amount_is_plain_trailing_mean():
    assert average_amount([100.0, 200.0, 300.0], window=3) == pytest.approx(200.0)


def test_valid_trading_ratio_counts_non_suspended_days():
    flags = [False, False, True, False]  # 1 天停牌 / 4 天

    ratio = valid_trading_ratio(flags, window=4)

    assert ratio == pytest.approx(0.75)


def test_average_turnover_is_plain_trailing_mean():
    """换手率均值——资金关注度/流动性信号，和价格/收益类特征是不同的
    信息来源（见 docs/adr/0011 "特征太窄"这条限制）。"""
    assert average_turnover([1.0, 2.0, 3.0], window=3) == pytest.approx(2.0)


def test_valuation_percentile_is_relative_to_own_trailing_history():
    """估值分位：当前(序列最后一个) PB 在自己过去 window 天历史里排第几
    分位，不是跨股票横截面比较（银行 PB~0.5，消费股 PB~10+，绝对值不可
    比，只能看"相对自己贵还是便宜"）。历史 [5,1,4,2,3]，当前值 3 排在
    正中间，分位 0.5。"""
    pb_history = [5.0, 1.0, 4.0, 2.0, 3.0]

    percentile = valuation_percentile(pb_history, window=5)

    assert percentile == pytest.approx(0.5)


def test_valuation_percentile_none_when_not_enough_history():
    assert valuation_percentile([1.0, 2.0], window=5) is None
