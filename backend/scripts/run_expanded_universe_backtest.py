"""扩大候选池实验专用端到端回测：19支股票候选池（官方10支+9支同组风格
补充），其余一切和官方 A 阶段回测(scripts/run_v3_backtest.py)保持一致——
原始特征、修正后 Top-K（dispersion check，不设方向性门槛）、top_k_ratio=0.5、
同一套 V3 原文止损参数、同样的 2024-2025 回测窗口。只对比"候选池从10支
扩到19支，Top-K 排序能不能选出更好的相对赢家"这一个变量。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/run_expanded_universe_backtest.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.quant_v3.a_phase_data_service import APhaseDataService
from app.quant_v3.a_phase_universe import EXPANDED_A_PHASE_STOCKS
from app.quant_v3.walkforward_signal import WalkForwardSignalSource
from app.strategies.base import StrategyConfig
from app.strategies.v3_strategy import V3Strategy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    history = pd.read_parquet(DATA_DIR / "expanded_universe_history.parquet")
    signal_source = WalkForwardSignalSource(
        DATA_DIR / "expanded_model_walkforward", DATA_DIR / "expanded_universe_history.parquet"
    )
    strategy = V3Strategy(
        signal_source=signal_source, top_k_ratio=0.5, universe=EXPANDED_A_PHASE_STOCKS,
    )

    config = BacktestConfig(
        start_date=date(2024, 1, 1),
        end_date=date(2025, 12, 31),
        initial_capital=1_000_000,
        rebalance_frequency="monthly",
        commission=0.0003,
        slippage=0.0005,
    )
    result = BacktestEngine(APhaseDataService(history)).run(
        config, StrategyConfig(), symbols=list(history["symbol"].unique()), strategy=strategy,
    )

    print("=== 绩效指标（19支候选池） ===")
    for key, value in result.metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    print(f"\n=== 交易记录（共 {len(result.trades)} 笔） ===")
    if not result.trades.empty:
        print(result.trades["reason"].value_counts())
        print("\n涉及股票：")
        print(result.trades["symbol"].value_counts())
    else:
        print("(无成交)")


if __name__ == "__main__":
    main()
