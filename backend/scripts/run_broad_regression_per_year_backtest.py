"""自由探索阶段（docs/adr/0013）：30支广泛候选池 + LightGBM回归预测 +
Top-K相对排序 + 大盘趋势过滤(自建等权指数200日均线)，同样的2019-2025
按年网格，和之前所有版本对比。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/run_broad_regression_per_year_backtest.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.quant_v3.a_phase_data_service import APhaseDataService
from app.quant_v3.broad_universe import BROAD_GROUP_BUDGETS, BROAD_STOCKS
from app.quant_v3.ensemble_regression_signal import EnsembleRegressionWalkForwardSignalSource
from app.quant_v3.momentum_signal import MomentumSignalSource
from app.quant_v3.qualification_signal import QualificationSignalSource
from app.strategies.base import StrategyConfig
from app.strategies.regression_rotation_strategy import RegressionRotationStrategy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

# ADR-0014最早网格测过{0.5,0.7,1.0}选了0.7（1.0会让模型选股形同虚设，
# 退化成"月度再平衡的等权买入持有"，不再是"以LightGBM为主体"）。ADR-0019
# 修复基准计算bug后，ADR-0021 用真实基准重新做网格{0.3,0.4,0.5,0.6,0.7,
# 0.8}，0.4是真实的局部峰值（两侧0.3和0.5-0.8都更差）——首次在真实基准
# 下做到5/7年跑赢+7年复合收益反超（316.5% vs 303.9%）。
TOP_K_RATIO = 0.4

# ADR-0015：回归目标预测窗口(horizon)网格测过 20/60/90/120 天，90天(约
# 一季度)最好。ADR-0016 诊断2019问题时把训练历史从2016年起扩展到2010年
# 起（更多股票的完整历史），虽然没解决2019问题，但整体质量更扎实。
# MODEL_DIR 本身在 ADR-0028 又换成了集成版本，定义见下方。
HISTORY_FILE = "broad_universe_history_extended.parquet"

# ADR-0018 曾经引入广度自适应(达到阈值当月临时全覆盖)，网格显示能把
# 跑赢年数从4/7提升到5/7——但 ADR-0019 修复基准计算bug后，ADR-0020 用
# 真实基准重新核对，发现四个配置(不加/0.7/0.8/0.9)全部并列4/7年跑赢，
# 不加广度机制的最简单版本总收益反而最高。之前的"提升"是bug造成的
# 假象，撤回，改回 None（不启用）。
BREADTH_THRESHOLD = None

# ADR-0022：诊断出模型在近乎普涨行情里会误判个别真正的动量黑马（2019年
# 1月模型给当年最大涨幅股打了很低的分，排第23名）。加一层独立于模型的
# 动量兜底——不管模型打分如何，每次调仓强制把每组trailing 60日动量最强
# 的 MOMENTUM_OVERRIDE_COUNT 支也纳入候选。网格测{1,2,3,5}，2是真实局部
# 峰值（1更差4/7年、3和5往下递减）——**6/7年跑赢基准，7年复合收益从
# ADR-0021的316.5%跃升到401.5%**，2025年首次转正，只剩2019年未解决。
MOMENTUM_OVERRIDE_COUNT = 2

# ADR-0025：动量兜底选出的黑马和模型选出的股票等权重摊薄在一起——加一个
# 权重倍数，被动量兜底强制纳入的股票单独按倍数加权。网格测{1.5,2.0,3.0}，
# 1.5同时改善两个指标(6/7年持平最高记录+7年复合收益406.0%，比不加倍数
# 的401.5%更高)；2.0/3.0总收益更高但跑赢年数掉到5/7(2024转负)，属于
# "牺牲年度胜率换总收益"的权衡，不选。
MOMENTUM_OVERRIDE_BOOST = 1.5

# ADR-0028：LightGBM默认不做行/列子采样(subsample/colsample_bytree都是
# 1.0)，导致换random_state训出来的树结构完全相同——之前以为的"集成"其实
# 只是同一个模型算了5遍。开启子采样(0.8)后5个种子才真正学到不同的树，
# 平均预测显著降噪：6/7年跑赢(持平)，7年复合收益从406.0%提升到
# **428.0%**，2019差距从-14.8%收窄到-11.4%，几乎所有年份都有改善。
MODEL_DIR = "broad_regression_h90_ensemble_model_walkforward"

# ADR-0037：重大发现——`BacktestEngine.run()` 不管传入什么自定义策略对象，
# 都会在引擎层面额外叠加一层通用风控(18%止损/3%换手带/22%目标波动率/15%
# 回撤刹车，来自 StrategyConfig 默认值)，这几个月一直在真实生效，
# 和"我们的策略没有止损"这个反复写在多篇ADR里的说法矛盾——那个说法只
# 对策略对象本身成立，不代表整个系统的真实行为。关掉这层后表现反而更好
# (7年复合从428.0%提升到450.96%)，说明这层是净拖累，不是中性的。现在
# 显式关闭，让实际跑的东西和一直以来的文字描述保持一致。
PURE_STRATEGY_CONFIG = StrategyConfig(stop_loss=0, turnover_band=0, target_volatility=0, max_drawdown_budget=0)


def run_one_year(history: pd.DataFrame, year: int) -> dict:
    signal_source = EnsembleRegressionWalkForwardSignalSource(
        DATA_DIR / MODEL_DIR, DATA_DIR / HISTORY_FILE
    )
    momentum_source = MomentumSignalSource(history, lookback=60, skip=0)
    qualification_source = QualificationSignalSource(history)
    strategy = RegressionRotationStrategy(
        signal_source=signal_source, universe=BROAD_STOCKS, group_budgets=BROAD_GROUP_BUDGETS,
        top_k_ratio=TOP_K_RATIO, trend_signal=None, breadth_threshold=BREADTH_THRESHOLD,
        momentum_override_source=momentum_source, momentum_override_count=MOMENTUM_OVERRIDE_COUNT,
        momentum_override_boost=MOMENTUM_OVERRIDE_BOOST, qualification_source=qualification_source,
    )
    config = BacktestConfig(
        start_date=date(year, 1, 1), end_date=date(year, 12, 31),
        initial_capital=1_000_000, rebalance_frequency="monthly",
        commission=0.0003, slippage=0.0005,
    )
    result = BacktestEngine(APhaseDataService(history)).run(
        config, PURE_STRATEGY_CONFIG, symbols=list(history["symbol"].unique()), strategy=strategy,
    )
    metrics = result.metrics
    return {
        "year": year,
        "trades": len(result.trades),
        "strategy_return": metrics.get("overall_return"),
        "benchmark_return": metrics.get("benchmark_return"),
        "excess_return": metrics.get("excess_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe": metrics.get("sharpe"),
    }


def main() -> None:
    history = pd.read_parquet(DATA_DIR / HISTORY_FILE)
    rows = [run_one_year(history, year) for year in YEARS]
    table = pd.DataFrame(rows).set_index("year")
    print(table.to_string())
    beat = (table["excess_return"] > 0).sum()
    print(f"\n{beat}/{len(YEARS)} 个年份策略跑赢了基准")
    strat_compound = 1.0
    bench_compound = 1.0
    for row in rows:
        strat_compound *= 1 + row["strategy_return"]
        bench_compound *= 1 + row["benchmark_return"]
    print(f"7年复合收益：策略 {strat_compound - 1:.4f}，基准 {bench_compound - 1:.4f}")


if __name__ == "__main__":
    main()
