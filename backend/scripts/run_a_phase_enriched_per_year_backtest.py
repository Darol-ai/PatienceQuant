"""消融实验专用（docs/adr/0012）：V3 官方10支股票 + 16项扩充特征，隔离
"只加特征、不换股票池"这一个变量，和 run_v3_per_year_backtest.py（官方
10支+14项特征）对比。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/run_a_phase_enriched_per_year_backtest.py
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
from app.quant_v3.enriched_model_signal import EnrichedWalkForwardSignalSource
from app.strategies.base import StrategyConfig
from app.strategies.v3_strategy import V3Strategy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]


def run_one_year(history: pd.DataFrame, year: int) -> dict:
    signal_source = EnrichedWalkForwardSignalSource(
        DATA_DIR / "a_phase_enriched_model_walkforward", DATA_DIR / "a_phase_valuation_history.parquet"
    )
    strategy = V3Strategy(signal_source=signal_source, top_k_ratio=0.5, universe=A_PHASE_STOCKS)
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
    }


def main() -> None:
    history = pd.read_parquet(DATA_DIR / "a_phase_valuation_history.parquet")
    rows = [run_one_year(history, year) for year in YEARS]
    table = pd.DataFrame(rows).set_index("year")
    print(table.to_string())
    beat = (table["excess_return"] > 0).sum()
    print(f"\n{beat}/{len(YEARS)} 个年份策略跑赢了基准")


if __name__ == "__main__":
    main()
