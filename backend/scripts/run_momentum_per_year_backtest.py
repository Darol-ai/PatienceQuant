"""调研公开策略新方向（docs/adr/0010）：用经典 12-1 动量 + 绝对动量过滤
换掉 LightGBM 分类信号，同时去掉止损/跟踪止损，纯靠月度调仓踢出动量转负
的持仓。用和 run_v3_per_year_backtest.py 完全一样的 2019-2025 按年网格、
同一批官方10支股票、同样的手续费/滑点，直接对比两套机制。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/run_momentum_per_year_backtest.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.quant_v3.a_phase_data_service import APhaseDataService
from app.quant_v3.a_phase_universe import A_PHASE_STOCKS
from app.quant_v3.momentum_signal import MomentumSignalSource
from app.strategies.base import StrategyConfig
from app.strategies.momentum_rotation_strategy import MomentumRotationStrategy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
GROUP_BUDGETS = {"红利": 0.40, "成长": 0.30, "周期": 0.30}


def run_one_year(history: pd.DataFrame, year: int) -> dict:
    signal_source = MomentumSignalSource(history, lookback=252, skip=21)
    strategy = MomentumRotationStrategy(
        signal_source=signal_source, universe=A_PHASE_STOCKS,
        group_budgets=GROUP_BUDGETS, top_k_ratio=0.5,
    )
    config = BacktestConfig(
        start_date=date(year, 1, 1), end_date=date(year, 12, 31),
        initial_capital=1_000_000, rebalance_frequency="monthly",
        commission=0.0003, slippage=0.0005,
    )
    result = BacktestEngine(APhaseDataService(history)).run(
        config, StrategyConfig(), symbols=list(history["symbol"].unique()), strategy=strategy,
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
    history = pd.read_parquet(DATA_DIR / "a_phase_history.parquet")
    rows = [run_one_year(history, year) for year in YEARS]
    table = pd.DataFrame(rows).set_index("year")
    print(table.to_string())
    beat = (table["excess_return"] > 0).sum()
    print(f"\n{beat}/{len(YEARS)} 个年份策略跑赢了基准")


if __name__ == "__main__":
    main()
