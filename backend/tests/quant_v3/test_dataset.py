import pytest

from app.quant_v3.dataset import build_sample, compute_features
from app.quant_v3.labels import compute_sigma_t, first_touch_label, label_boundaries, scale_to_20_day_horizon


def _make_bars(closes, opens=None, highs=None, lows=None, volumes=None, amounts=None, suspended=None):
    n = len(closes)
    opens = opens or closes
    highs = highs or [c * 1.01 for c in closes]
    lows = lows or [c * 0.99 for c in closes]
    volumes = volumes or [1_000_000] * n
    amounts = amounts or [c * v for c, v in zip(closes, volumes)]
    suspended = suspended or [False] * n
    return [
        {
            "date": f"2024-01-{i + 1:02d}",
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
            "volume": volumes[i],
            "amount": amounts[i],
            "is_suspended": suspended[i],
        }
        for i in range(n)
    ]


def _oscillating_closes(n: int) -> list[float]:
    """一个有真实（非零）波动、但完全确定的价格序列：100→101→99 循环。"""
    pattern = [100.0, 101.0, 99.0]
    return [pattern[i % 3] for i in range(n)]


def test_build_sample_returns_none_when_history_too_short():
    closes = _oscillating_closes(140)  # 差 1 天不够 121 天历史
    bars = _make_bars(closes)

    assert build_sample(bars, t_index=119, group="红利") is None


def test_build_sample_returns_none_when_future_window_not_mature():
    closes = _oscillating_closes(139)  # 121 天历史够了，但之后不够 20 天成熟期
    bars = _make_bars(closes)

    assert build_sample(bars, t_index=120, group="红利") is None


def test_build_sample_wires_the_right_slices_into_label_and_features():
    """用同样的价格切片，分别直接调用已经测过的 sigma_t→v_t→边界→首触判定，
    和 build_sample 内部走的应该是同一条数据，结果必须一致——这里要抓的是
    "喂给标签计算的价格切片对不对"这类接线错误，不是重新验证公式本身。"""
    closes = _oscillating_closes(160)
    opens = closes  # 简化：开盘价等于收盘价
    bars = _make_bars(closes, opens=opens)
    t_index = 130

    sample = build_sample(bars, t_index=t_index, group="成长")
    assert sample is not None

    history_closes = closes[: t_index + 1]
    sigma_t = compute_sigma_t(history_closes, window=60)
    v_t = scale_to_20_day_horizon(sigma_t)
    upper_t, lower_t = label_boundaries(v_t)
    o_star = opens[t_index + 1]
    future_closes = closes[t_index + 1 : t_index + 21]
    expected_label = first_touch_label(future_closes, o_star, upper_t, lower_t)

    assert sample["label"] == expected_label
    assert sample["date"] == bars[t_index]["date"]
    assert sample["label_end"] == bars[t_index + 20]["date"]
    assert sample["group"] == "成长"
    assert sample["return_5d"] == pytest.approx(closes[t_index] / closes[t_index - 5] - 1)


def test_compute_features_is_reusable_without_needing_future_data():
    """预测/回测阶段没有未来数据，compute_features 必须只靠历史就能算出
    build_sample 用的同一套特征——这是训练和预测复用同一份计算的保证。"""
    closes = _oscillating_closes(121)  # 刚好够 120 天历史，不需要任何未来数据
    bars = _make_bars(closes)

    features = compute_features(bars, t_index=120)

    assert features is not None
    assert set(features) == {
        "return_5d", "return_20d", "return_60d", "return_120d",
        "volatility_20d", "volatility_60d", "max_drawdown_60d",
        "ma_deviation_20d", "ma_deviation_60d", "ma_deviation_120d",
        "atr_ratio_20d", "volume_ratio_20d", "avg_amount_60d", "valid_trading_ratio_60d",
    }


def test_compute_features_none_when_history_too_short():
    closes = _oscillating_closes(100)
    bars = _make_bars(closes)

    assert compute_features(bars, t_index=99) is None


def test_build_sample_returns_none_when_history_is_perfectly_flat():
    """历史价格完全没有波动，sigma_t 为零，V3 方案 3.2 节：标准差为零时不
    制作该样本——不能悄悄给个假的边界。"""
    closes = [100.0] * 160
    bars = _make_bars(closes)

    assert build_sample(bars, t_index=130, group="红利") is None
