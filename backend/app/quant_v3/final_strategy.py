"""自由探索阶段最终确定的量化策略（docs/adr/0013~0037）——30支跨行业
候选池 + LightGBM回归(90日窗口，5模型集成) + Top-K相对排序 + 动量兜底 +
流动性资格判断，不叠加引擎级止损/波动率/回撤风控（ADR-0037已证明那层
是净拖累）。

这里把整套装配逻辑集中一处，供 API 路由复用，模型/历史行情只在进程内
加载一次（`lru_cache`）——重新构造 `EnsembleRegressionWalkForwardSignalSource`
每次都要从磁盘读 35 个 LightGBM 模型文件，不能每个请求都重来一遍。
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import pandas as pd

from app.quant_v3.broad_universe import BROAD_GROUP_BUDGETS, BROAD_STOCKS
from app.quant_v3.ensemble_regression_signal import EnsembleRegressionWalkForwardSignalSource
from app.quant_v3.momentum_signal import MomentumSignalSource
from app.quant_v3.qualification_signal import QualificationSignalSource
from app.strategies.base import StrategyConfig
from app.strategies.regression_rotation_strategy import RegressionRotationStrategy

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_HISTORY_FILE = _DATA_DIR / "broad_universe_history_extended.parquet"
_MODEL_DIR = _DATA_DIR / "broad_regression_h90_ensemble_model_walkforward"

# ADR-0037：引擎级止损/换手带/波动率目标/回撤刹车叠加层实测是净拖累，
# 显式全部关闭，让实际运行的东西和策略设计文档保持一致。
FINAL_ENGINE_STRATEGY_CONFIG = StrategyConfig(
    stop_loss=0, turnover_band=0, target_volatility=0, max_drawdown_budget=0,
)

# EnsembleRegressionWalkForwardSignalSource 按年份分发模型，超出这个范围
# 的年份没有对应折的模型，会退化成中性占位信号（永不触发买卖）——不能
# 假装能预测训练范围之外的年份，回测/展示都应该限制在这个区间内。2026
# 这一折是专门为模拟盘补的（模拟盘默认按"今天"驱动），用和2019-2025
# 完全相同的滚动训练规则（训练截止Y-1年7月、Y-1下半年校准），不是新
# 方法论，见 scripts/train_broad_regression_h90_ensemble_2026.py。
SUPPORTED_YEAR_RANGE: Tuple[int, int] = (2019, 2026)


@lru_cache(maxsize=1)
def _load_history() -> pd.DataFrame:
    return pd.read_parquet(_HISTORY_FILE)


def latest_trading_day_on_or_before(as_of: date) -> date:
    """Snap to the most recent date this pipeline actually has price bars for.

    Paper trading defaults `as_of` to the literal wall-clock date, which is
    not a trading day on weekends/holidays (and can be ahead of the data
    even on a weekday, before that day's close is ingested). The signal
    sources (`EnsembleRegressionWalkForwardSignalSource`,
    `MomentumSignalSource`, `QualificationSignalSource`) all index bars by
    an exact date match with no fallback, so calling them with a
    non-trading `as_of` doesn't error — it silently returns no score for
    every single symbol, which looks like "no trade opportunity today"
    instead of the data-alignment bug it actually is. Snapping here, once,
    before any signal source sees the date, is what prevents that.
    """
    history = _load_history()
    all_dates = pd.to_datetime(history["date"]).dt.date
    available = all_dates[all_dates <= as_of]
    if available.empty:
        raise ValueError(f"没有 {as_of} 或更早的行情数据（最早 {all_dates.min()}）")
    return available.max()


@lru_cache(maxsize=1)
def _cached_signal_sources() -> Tuple[
    EnsembleRegressionWalkForwardSignalSource, MomentumSignalSource, QualificationSignalSource
]:
    """这三个信号源加载后是纯只读查询（模型文件/历史行情一次性建好索引，
    之后每次调用都不改内部状态），可以安全地在多个请求、多个策略实例
    之间共享——真正需要按请求隔离的只有 `RegressionRotationStrategy` 自己
    的可变状态（比如粘性持仓计数器），见 `build_final_strategy` 的说明。
    """
    history = _load_history()
    signal_source = EnsembleRegressionWalkForwardSignalSource(_MODEL_DIR, _HISTORY_FILE)
    momentum_source = MomentumSignalSource(history, lookback=60, skip=0)
    qualification_source = QualificationSignalSource(history)
    return signal_source, momentum_source, qualification_source


def build_final_strategy() -> RegressionRotationStrategy:
    """每次调用都构造一个全新的策略实例——`RegressionRotationStrategy`
    内部有可变状态（`_sticky_hold_remaining`），如果像早期实现那样把整个
    策略对象也用 `lru_cache` 缓存成进程级单例，两个不相关的回测请求
    （尤其是并发请求：`/api/backtests` 的路由是同步函数，FastAPI 会丢进
    线程池执行，两个请求可能真的在不同线程里同时跑）会共享并互相污染
    这份状态。真正应该跨请求复用的是下面这几个无状态的信号源（加载模型/
    建索引很贵，构造 `RegressionRotationStrategy` 本身很便宜，只是包了
    几个引用）。
    """
    signal_source, momentum_source, qualification_source = _cached_signal_sources()
    return RegressionRotationStrategy(
        signal_source=signal_source,
        universe=BROAD_STOCKS,
        group_budgets=BROAD_GROUP_BUDGETS,
        top_k_ratio=0.4,
        trend_signal=None,
        momentum_override_source=momentum_source,
        momentum_override_count=2,
        momentum_override_boost=1.5,
        qualification_source=qualification_source,
    )


def final_strategy_history() -> pd.DataFrame:
    return _load_history()


def validate_backtest_date_range(start: date, end: date) -> None:
    min_year, max_year = SUPPORTED_YEAR_RANGE
    if start.year < min_year or end.year > max_year:
        raise ValueError(
            f"该策略只在 {min_year}-{max_year} 年之间有逐年滚动训练好的模型，"
            f"超出范围的年份会退化成占位信号（不会有任何交易），请把回测区间限制在这个范围内"
        )
