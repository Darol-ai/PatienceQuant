"""自由探索阶段（docs/adr/0013）：把 LightGBM 从"三分类事件标签"换成
"直接回归预测未来N日收益"——三分类标签(dataset.py)里 NEUTRAL 占了近70%
样本，真正能学的 UP/DOWN 样本比看起来少得多；回归目标是连续值，每条
样本都有信息量，也更贴近 Qlib/BigQuant 公开做法里"预测收益排序选股"的
标准范式（预测分数直接拿来排序，不需要再分箱成三类）。

复用官方14项特征（`dataset.compute_features`），只换标签定义。
"""
from app.quant_v3.dataset import compute_features

REGRESSION_FEATURE_COLUMNS = [
    "return_5d", "return_20d", "return_60d", "return_120d",
    "volatility_20d", "volatility_60d", "max_drawdown_60d",
    "ma_deviation_20d", "ma_deviation_60d", "ma_deviation_120d",
    "atr_ratio_20d", "volume_ratio_20d", "avg_amount_60d", "valid_trading_ratio_60d",
]

_DEFAULT_HORIZON = 20


def forward_return(closes: list[float], t_index: int, horizon: int) -> float | None:
    """t_index 到 t_index+horizon 的简单收益。越界（未来数据不够）返回 None。"""
    if t_index + horizon >= len(closes):
        return None
    return closes[t_index + horizon] / closes[t_index] - 1


def build_regression_sample(bars: list[dict], t_index: int, group: str, horizon: int = _DEFAULT_HORIZON) -> dict | None:
    features = compute_features(bars, t_index)
    if features is None:
        return None

    closes = [b["close"] for b in bars]
    label = forward_return(closes, t_index, horizon)
    if label is None:
        return None

    return {
        "date": bars[t_index]["date"],
        "label_end": bars[t_index + horizon]["date"],
        "group": group,
        **features,
        "forward_return": label,
    }
