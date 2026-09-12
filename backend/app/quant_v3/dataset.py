"""把单只股票的日线序列，在某个样本日 t，组装成一条训练样本：
V3 方案 3.5 节的 15 项特征 + 3.2 节的 L1 标签（sigma_t → v_t → 边界 → 首触判定）。

bars 需按日期升序排列，每个元素至少含 open/high/low/close/volume/amount/
is_suspended。t_index 是样本日 t 在 bars 里的下标；需要 t 之前至少 120 天
历史（算 120 日收益等特征）、t 之后至少 20 天数据（标签成熟窗口），否则
返回 None——历史不够或标签未成熟都不制作该样本，同 sigma_t 为零/非有限值
时的处理一致，不强行凑数。
"""
import pandas as pd

from app.quant_v3.features import (
    atr_ratio,
    average_amount,
    close_vs_ma_deviation,
    max_drawdown_over,
    return_over_window,
    valid_trading_ratio,
    volatility_over,
    volume_ratio,
)
from app.quant_v3.labels import compute_sigma_t, first_touch_label, label_boundaries, scale_to_20_day_horizon

_HISTORY_WINDOW = 120
_LABEL_HORIZON = 20


def cross_sectional_rank(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """把每个特征列的原始值换成同一天在这批股票里的百分位排名（0~1）。

    公开的 LightGBM 选股策略（BigQuant/Qlib）普遍这么做：同一个原始特征值
    在牛市和熊市代表的相对强弱不一样，换成"当天在股票池里排第几"之后，
    不同市场环境下更容易跨期比较、模型更容易学到稳定的规律，而不是记住
    某个绝对数值区间（那样的规律换个市场环境就失效了）。
    """
    result = frame.copy()
    for column in feature_columns:
        result[column] = result.groupby("date")[column].rank(pct=True)
    return result


def compute_features(bars: list[dict], t_index: int) -> dict | None:
    """只算 V3 3.5 节的 14 项数值特征（配置组不在这里，由调用方附加），
    不涉及标签——训练和实盘/回测预测都用这同一份特征计算，避免两处实现
    悄悄漂移。历史不足 120 天则返回 None。
    """
    if t_index - _HISTORY_WINDOW < 0:
        return None

    history = bars[: t_index + 1]
    closes = [b["close"] for b in history]
    highs = [b["high"] for b in history]
    lows = [b["low"] for b in history]
    volumes = [b["volume"] for b in history]
    amounts = [b["amount"] for b in history]
    suspended_flags = [b["is_suspended"] for b in history]

    features = {
        "return_5d": return_over_window(closes, 5),
        "return_20d": return_over_window(closes, 20),
        "return_60d": return_over_window(closes, 60),
        "return_120d": return_over_window(closes, 120),
        "volatility_20d": volatility_over(closes, 20),
        "volatility_60d": volatility_over(closes, 60),
        "max_drawdown_60d": max_drawdown_over(closes, 60),
        "ma_deviation_20d": close_vs_ma_deviation(closes, 20),
        "ma_deviation_60d": close_vs_ma_deviation(closes, 60),
        "ma_deviation_120d": close_vs_ma_deviation(closes, 120),
        "atr_ratio_20d": atr_ratio(highs, lows, closes, 20),
        "volume_ratio_20d": volume_ratio(volumes, 20),
        "avg_amount_60d": average_amount(amounts, 60),
        "valid_trading_ratio_60d": valid_trading_ratio(suspended_flags, 60),
    }
    if any(value is None for value in features.values()):
        return None
    return features


def compute_label_l1(bars: list[dict], t_index: int) -> dict | None:
    """算 V3 3.2 节的 L1 标签（sigma_t → v_t → 边界 → 首触判定），不涉及
    特征——抽出来给 build_sample 和"扩充特征集合"的实验性数据管线
    （见 enriched_dataset.py，docs/adr/0011）复用，两边标签定义必须完全
    一致，不能各自实现一份悄悄漂移。未来窗口不够/波动率为零则返回 None。
    """
    if t_index + _LABEL_HORIZON >= len(bars):
        return None

    history_closes = [b["close"] for b in bars[: t_index + 1]]
    sigma_t = compute_sigma_t(history_closes, window=60)
    if sigma_t is None:
        return None
    upper_t, lower_t = label_boundaries(scale_to_20_day_horizon(sigma_t))

    o_star = bars[t_index + 1]["open"]
    future_closes = [b["close"] for b in bars[t_index + 1 : t_index + 1 + _LABEL_HORIZON]]
    label = first_touch_label(future_closes, o_star, upper_t, lower_t)

    return {
        "date": bars[t_index]["date"],
        "label_end": bars[t_index + _LABEL_HORIZON]["date"],
        "label": label,
    }


def build_sample(bars: list[dict], t_index: int, group: str) -> dict | None:
    features = compute_features(bars, t_index)
    if features is None:
        return None

    label_info = compute_label_l1(bars, t_index)
    if label_info is None:
        return None

    return {
        "date": label_info["date"],
        "label_end": label_info["label_end"],
        "group": group,
        **features,
        "label": label_info["label"],
    }
