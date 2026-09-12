"""调研新方向（docs/adr/0011）：分散化候选池(12支跨行业股票) + 扩充特征
(16项，加换手率均值/估值分位) + V3Strategy 同一套 Top-K 机制，跑和
run_v3_per_year_backtest.py 一样的 2019-2025 按年网格，直接对比"股票池
分散度+特征宽度"这个变量单独的影响。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/run_diversified_per_year_backtest.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.quant_v3.a_phase_data_service import APhaseDataService
from app.quant_v3.diversified_universe import DIVERSIFIED_GROUP_BUDGETS, DIVERSIFIED_STOCKS
from app.quant_v3.enriched_model_signal import EnrichedWalkForwardSignalSource
from app.strategies.base import StrategyConfig
from app.strategies.v3_strategy import V3Strategy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]


def run_one_year(history: pd.DataFrame, year: int) -> dict:
    signal_source = EnrichedWalkForwardSignalSource(
        DATA_DIR / "diversified_model_walkforward", DATA_DIR / "diversified_universe_history.parquet"
    )
    strategy = V3Strategy(
        signal_source=signal_source, top_k_ratio=0.5,
        universe=DIVERSIFIED_STOCKS, group_budgets=DIVERSIFIED_GROUP_BUDGETS,
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
    history = pd.read_parquet(DATA_DIR / "diversified_universe_history.parquet")
    rows = [run_one_year(history, year) for year in YEARS]
    table = pd.DataFrame(rows).set_index("year")
    print(table.to_string())
    beat = (table["excess_return"] > 0).sum()
    print(f"\n{beat}/{len(YEARS)} 个年份策略跑赢了基准")


if __name__ == "__main__":
    main()
