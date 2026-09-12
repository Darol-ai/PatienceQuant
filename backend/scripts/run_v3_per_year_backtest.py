"""检验"策略跑输基准是不是 2024-2025 这种强普涨行情特有的现象"（见
docs/adr/0009）：对 2019-2025 每一年分别单独跑一次回测（不是像
run_v3_backtest.py 那样一次跑 2024-2025 两年连续窗口），策略逻辑和参数
完全不变（官方10支股票、原始特征、修正后 Top-K、V3 原文止损参数），只
变回测年份，看不同年份/不同行情下策略相对基准的表现是否一致。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/run_v3_per_year_backtest.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.quant_v3.a_phase_data_service import APhaseDataService
from app.quant_v3.walkforward_signal import WalkForwardSignalSource
from app.strategies.base import StrategyConfig
from app.strategies.v3_strategy import V3Strategy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]


def run_one_year(history: pd.DataFrame, year: int) -> dict:
    signal_source = WalkForwardSignalSource(DATA_DIR / "v3_model_walkforward", DATA_DIR / "a_phase_history.parquet")
    strategy = V3Strategy(signal_source=signal_source, top_k_ratio=0.5)
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
