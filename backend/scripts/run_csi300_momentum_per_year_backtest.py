"""沪深300universe版本的经典动量策略逐年回测验证——Jegadeesh-Titman
12-1动量 + Dual Momentum绝对动量过滤(docs/adr/0010)，不是机器学习模型，
作为"第三个独立有效策略"的候选：和LightGBM/XGBoost两个基于同一套
特征工程的模型策略，在方法论上完全不同，不是同一个思路的变体。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/run_csi300_momentum_per_year_backtest.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.quant_v3.a_phase_data_service import APhaseDataService
from app.quant_v3.csi300_universe import CSI300_GROUP_BUDGETS, csi300_stocks
from app.quant_v3.momentum_signal import MomentumSignalSource
from app.quant_v3.real_benchmark import csi300_return
from app.strategies.base import StrategyConfig
from app.strategies.momentum_rotation_strategy import MomentumRotationStrategy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_FILE = "csi300_universe_history.parquet"
TOP_K_RATIO = 0.1
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
PURE_STRATEGY_CONFIG = StrategyConfig(stop_loss=0, turnover_band=0, target_volatility=0, max_drawdown_budget=0)


def run_one_year(history: pd.DataFrame, year: int) -> dict:
    signal_source = MomentumSignalSource(history, lookback=252, skip=21)
    universe = csi300_stocks()
    strategy = MomentumRotationStrategy(
        signal_source=signal_source, universe=universe, group_budgets=CSI300_GROUP_BUDGETS,
        top_k_ratio=TOP_K_RATIO,
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
    real_csi300 = csi300_return(date(year, 1, 1), date(year, 12, 31))
    return {
        "year": year,
        "trades": len(result.trades),
        "strategy_return": metrics.get("overall_return"),
        "csi300_index_return": real_csi300,
        "excess_vs_csi300": (metrics.get("overall_return") - real_csi300) if real_csi300 is not None else None,
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe": metrics.get("sharpe"),
    }


def main() -> None:
    history = pd.read_parquet(DATA_DIR / HISTORY_FILE)
    rows = [run_one_year(history, year) for year in YEARS]
    table = pd.DataFrame(rows).set_index("year")
    print(table.to_string())
    beat = (table["excess_vs_csi300"] > 0).sum()
    print(f"\n{beat}/{len(YEARS)} 个年份跑赢真实沪深300指数")
    strat_compound = 1.0
    csi300_compound = 1.0
    for row in rows:
        strat_compound *= 1 + row["strategy_return"]
        csi300_compound *= 1 + (row["csi300_index_return"] or 0)
    print(f"7年复合收益：策略 {strat_compound - 1:.4f}，真实沪深300指数 {csi300_compound - 1:.4f}")


if __name__ == "__main__":
    main()
