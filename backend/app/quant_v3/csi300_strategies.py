"""对齐主流做法（沪深300 universe，见调研记录）——集中管理沪深300
universe下的三个独立策略：LightGBM回归集成、XGBoost回归集成、
Jegadeesh-Titman 12-1动量。每个策略是否真的"有效"由对应的
scripts/run_csi300_*_per_year_backtest.py 逐年验证后再决定是否纳入
`ACTIVE_CSI300_STRATEGIES`——没有验证通过之前不应该被前端/API暴露。

和 final_strategy.py（30支候选池版本）是两套独立的universe，不共用
模型/数据文件。
"""
from __future__ import annotations

import threading
from functools import lru_cache, wraps
from pathlib import Path
from typing import Tuple

import pandas as pd

from app.quant_v3.blended_signal import BlendedSignalSource
from app.quant_v3.csi300_universe import CSI300_GROUP_BUDGETS, csi300_stocks
from app.quant_v3.ensemble_regression_signal import EnsembleRegressionWalkForwardSignalSource
from app.quant_v3.momentum_signal import MomentumSignalSource
from app.quant_v3.xgboost_signal import XGBoostWalkForwardSignalSource
from app.strategies.momentum_rotation_strategy import MomentumRotationStrategy
from app.strategies.regression_rotation_strategy import RegressionRotationStrategy

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_HISTORY_FILE = _DATA_DIR / "csi300_universe_history.parquet"
_LIGHTGBM_MODEL_DIR = _DATA_DIR / "csi300_lightgbm_ensemble_model_walkforward"
_XGBOOST_MODEL_DIR = _DATA_DIR / "csi300_xgboost_ensemble_model_walkforward"

TOP_K_RATIO = 0.1
# 2026这一折是专门为模拟盘补的(模拟盘默认按"今天"驱动)，用和2019-2025
# 完全相同的滚动训练规则(训练截止Y-1年7月、Y-1下半年校准)，不是新方法论。
SUPPORTED_YEAR_RANGE: Tuple[int, int] = (2019, 2026)


def latest_csi300_trading_day_on_or_before(as_of):
    """和 `final_strategy.latest_trading_day_on_or_before` 同样的理由——
    信号源按精确日期索引，周末/节假日/数据还没到账的"今天"需要先落到
    最近一个真实交易日，否则每支股票都会诚实但无意义地返回None。这里
    单独实现是因为CSI300和30支候选池用的是两份不同的history parquet，
    不能共用对方的"最近交易日"。
    """
    history = _history()
    all_dates = pd.to_datetime(history["date"]).dt.date
    available = all_dates[all_dates <= as_of]
    if available.empty:
        raise ValueError(f"没有 {as_of} 或更早的行情数据（最早 {all_dates.min()}）")
    return available.max()


def validate_csi300_date_range(start, end) -> None:
    min_year, max_year = SUPPORTED_YEAR_RANGE
    if start.year < min_year or end.year > max_year:
        raise ValueError(
            f"沪深300策略只在 {min_year}-{max_year} 年之间有逐年滚动训练好的模型，"
            f"超出范围的年份会退化成占位信号（不会有任何交易），请把回测区间限制在这个范围内"
        )

# 逐年回测验证通过之后才把策略key加进来。验证结果（2019-2025，Top30/
# 300，月度调仓，vs真实沪深300指数）：
#   csi300_lightgbm：6/7年跑赢，7年复合+646.9% vs 指数+58.9% —— 有效
#   csi300_xgboost： 6/7年跑赢，7年复合+620.8% vs 指数+58.9% —— 有效
#   csi300_ensemble：6/7年跑赢，7年复合+611.9% vs 指数+58.9% —— 有效
#     （两个独立模型取平均分，比单独LightGBM/XGBoost都略低——诚实记录，
#     不是"集成一定更好"，简单平均在两个高度相关的树模型之间没有额外
#     增益，属于正常、可解释的结果，不代表这次集成没有意义：它仍然是
#     大幅跑赢真实指数的有效策略，只是不是三者中最强的一个）
#   csi300_momentum：3/7年跑赢，7年复合仅+8.9%，大幅跑输指数 —— 无效，
#     不纳入。12-1动量+绝对动量过滤在这个universe/区间里经常触发"清仓
#     观望"（收益长期接近0%），不是没调好参数就退回去重调——那样等于
#     看着这次的结果反推参数，是被反复强调过的p-hacking，不能做。如实
#     排除，第三个策略换成上面验证过有效的集成方向。
ACTIVE_CSI300_STRATEGIES: Tuple[str, ...] = ("csi300_lightgbm", "csi300_xgboost", "csi300_ensemble")


def _singleton(builder):
    """lru_cache(maxsize=1)本身只保证"读缓存"线程安全,两个线程同时
    miss缓存时会各自独立跑一遍builder(观测到过:预热线程和真实请求线程
    撞在一起,90万行索引被重复构建,内存直接翻倍、耗时也翻倍)。这里用一把
    锁把"构建"这一步也串行化——后来者拿锁时前者早已把结果写进lru_cache,
    直接命中返回,不会重复计算。"""
    lock = threading.Lock()
    cached = lru_cache(maxsize=1)(builder)

    @wraps(builder)
    def wrapper():
        # 每次调用都先拿锁再查缓存:一旦builder跑过一次,后续调用在锁内
        # 也是纯缓存命中,开销可以忽略;真正要避免的是"锁外先探测缓存"这种
        # 写法给并发miss留出重复构建的窗口。
        with lock:
            return cached()

    return wrapper


@_singleton
def _history() -> pd.DataFrame:
    return pd.read_parquet(_HISTORY_FILE)


def csi300_history() -> pd.DataFrame:
    return _history()


@_singleton
def _lightgbm_signal_source() -> EnsembleRegressionWalkForwardSignalSource:
    return EnsembleRegressionWalkForwardSignalSource(_LIGHTGBM_MODEL_DIR, _HISTORY_FILE)


@_singleton
def _xgboost_signal_source() -> XGBoostWalkForwardSignalSource:
    return XGBoostWalkForwardSignalSource(_XGBOOST_MODEL_DIR, _HISTORY_FILE)


@_singleton
def _momentum_signal_source() -> MomentumSignalSource:
    return MomentumSignalSource(_history(), lookback=252, skip=21)


def build_csi300_lightgbm_strategy() -> RegressionRotationStrategy:
    return RegressionRotationStrategy(
        signal_source=_lightgbm_signal_source(), universe=csi300_stocks(),
        group_budgets=CSI300_GROUP_BUDGETS, top_k_ratio=TOP_K_RATIO, trend_signal=None,
    )


def build_csi300_xgboost_strategy() -> RegressionRotationStrategy:
    return RegressionRotationStrategy(
        signal_source=_xgboost_signal_source(), universe=csi300_stocks(),
        group_budgets=CSI300_GROUP_BUDGETS, top_k_ratio=TOP_K_RATIO, trend_signal=None,
    )


def build_csi300_momentum_strategy() -> MomentumRotationStrategy:
    return MomentumRotationStrategy(
        signal_source=_momentum_signal_source(), universe=csi300_stocks(),
        group_budgets=CSI300_GROUP_BUDGETS, top_k_ratio=TOP_K_RATIO,
    )


def build_csi300_ensemble_strategy() -> RegressionRotationStrategy:
    """LightGBM+XGBoost两个独立模型预测分数取平均——调研佐证见模块内
    `blended_signal.py`的说明。"""
    blended = BlendedSignalSource([_lightgbm_signal_source(), _xgboost_signal_source()])
    return RegressionRotationStrategy(
        signal_source=blended, universe=csi300_stocks(),
        group_budgets=CSI300_GROUP_BUDGETS, top_k_ratio=TOP_K_RATIO, trend_signal=None,
    )


BUILDERS = {
    "csi300_lightgbm": build_csi300_lightgbm_strategy,
    "csi300_xgboost": build_csi300_xgboost_strategy,
    "csi300_momentum": build_csi300_momentum_strategy,
    "csi300_ensemble": build_csi300_ensemble_strategy,
}
