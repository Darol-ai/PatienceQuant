"""调研新方向（docs/adr/0011）：在 V3 官方 14 项价量特征基础上，加两个
不同信息来源的特征——换手率均值（资金关注度/流动性）、估值分位（相对
自己历史贵还是便宜）。复用 `dataset.compute_features`/`compute_label_l1`，
不改官方 `dataset.py`/`build_sample` 的行为，只是研究性实验专用的扩充版
数据管线，两边标签定义完全一致（同一个 `compute_label_l1`）。

bars 除了官方 14 项特征需要的字段外，还需要 `turnover_rate`（BaoStock
`turn` 字段，官方 parquet 已经在抓，只是没当特征用）和 `pb_mrq`（BaoStock
K 线数据额外字段，需要新抓，见 fetch_diversified_universe_history.py）。
"""
from app.quant_v3.dataset import compute_features, compute_label_l1
from app.quant_v3.features import average_turnover, valuation_percentile

_TURNOVER_WINDOW = 60
_VALUATION_WINDOW = 252

ENRICHED_FEATURE_COLUMNS = [
    "return_5d", "return_20d", "return_60d", "return_120d",
    "volatility_20d", "volatility_60d", "max_drawdown_60d",
    "ma_deviation_20d", "ma_deviation_60d", "ma_deviation_120d",
    "atr_ratio_20d", "volume_ratio_20d", "avg_amount_60d", "valid_trading_ratio_60d",
    "avg_turnover_60d", "valuation_percentile_252d",
]


def compute_enriched_features(bars: list[dict], t_index: int) -> dict | None:
    base_features = compute_features(bars, t_index)
    if base_features is None:
        return None

    history = bars[: t_index + 1]
    turnovers = [b["turnover_rate"] for b in history]
    pb_values = [b["pb_mrq"] for b in history]

    avg_turnover_60d = average_turnover(turnovers, _TURNOVER_WINDOW)
    valuation_percentile_252d = valuation_percentile(pb_values, _VALUATION_WINDOW)
    if avg_turnover_60d is None or valuation_percentile_252d is None:
        return None

    return {
        **base_features,
        "avg_turnover_60d": avg_turnover_60d,
        "valuation_percentile_252d": valuation_percentile_252d,
    }


def build_enriched_sample(bars: list[dict], t_index: int, group: str) -> dict | None:
    features = compute_enriched_features(bars, t_index)
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
