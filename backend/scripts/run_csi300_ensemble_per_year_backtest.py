"""沪深300universe版本的LightGBM+XGBoost集成(取平均分)策略逐年回测
验证——调研佐证(Qlib CSI300公开基准DoubleEnsemble是最强基线)，
动量候选(run_csi300_momentum_per_year_backtest.py)验证失败后换的
第三个候选策略。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/run_csi300_ensemble_per_year_backtest.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.quant_v3.a_phase_data_service import APhaseDataService
from app.quant_v3.blended_signal import BlendedSignalSource
from app.quant_v3.csi300_universe import CSI300_GROUP_BUDGETS, csi300_stocks
from app.quant_v3.ensemble_regression_signal import EnsembleRegressionWalkForwardSignalSource
from app.quant_v3.real_benchmark import csi300_return
from app.quant_v3.xgboost_signal import XGBoostWalkForwardSignalSource
from app.strategies.base import StrategyConfig
from app.strategies.regression_rotation_strategy import RegressionRotationStrategy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_FILE = "csi300_universe_history.parquet"
LIGHTGBM_MODEL_DIR = "csi300_lightgbm_ensemble_model_walkforward"
XGBOOST_MODEL_DIR = "csi300_xgboost_ensemble_model_walkforward"
TOP_K_RATIO = 0.1
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
PURE_STRATEGY_CONFIG = StrategyConfig(stop_loss=0, turnover_band=0, target_volatility=0, max_drawdown_budget=0)


def run_one_year(history: pd.DataFrame, year: int) -> dict:
    lgbm_source = EnsembleRegressionWalkForwardSignalSource(DATA_DIR / LIGHTGBM_MODEL_DIR, DATA_DIR / HISTORY_FILE)
    xgb_source = XGBoostWalkForwardSignalSource(DATA_DIR / XGBOOST_MODEL_DIR, DATA_DIR / HISTORY_FILE)
    blended = BlendedSignalSource([lgbm_source, xgb_source])
    universe = csi300_stocks()
    strategy = RegressionRotationStrategy(
        signal_source=blended, universe=universe, group_budgets=CSI300_GROUP_BUDGETS,
        top_k_ratio=TOP_K_RATIO, trend_signal=None,
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
