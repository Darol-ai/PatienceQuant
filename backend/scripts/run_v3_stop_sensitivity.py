"""止损参数敏感性实验：固定其余一切不变（原始特征、修正后 Top-K、同一套
walk-forward 模型），只改 hard_stop_loss / trailing_stop 两组阈值，看
"止损是否对 2024-2025 这种强普涨行情过敏"。

网格在跑之前就定死，全部跑完一次性报告，不根据中途结果挑着调（避免
retroactive p-hacking）：
    A 基线：V3 原文 -15% / 25%触发 / -10%回撤
    B 更宽的硬止损：-20%，跟踪止损不变
    C 更宽的跟踪止损回撤容忍：-15%，硬止损不变
    D 最宽组合：硬止损 -20%，跟踪止损 40%触发 / -15%回撤

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/run_v3_stop_sensitivity.py
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

GRID = [
    ("A 基线(V3原文)", dict(hard_stop_loss_threshold=-0.15, trailing_stop_arm_threshold=0.25, trailing_stop_drawdown_threshold=-0.10)),
    ("B 更宽硬止损-20%", dict(hard_stop_loss_threshold=-0.20, trailing_stop_arm_threshold=0.25, trailing_stop_drawdown_threshold=-0.10)),
    ("C 更宽跟踪回撤-15%", dict(hard_stop_loss_threshold=-0.15, trailing_stop_arm_threshold=0.25, trailing_stop_drawdown_threshold=-0.15)),
    ("D 最宽组合", dict(hard_stop_loss_threshold=-0.20, trailing_stop_arm_threshold=0.40, trailing_stop_drawdown_threshold=-0.15)),
]


def run_one(history: pd.DataFrame, params: dict) -> dict:
    signal_source = WalkForwardSignalSource(DATA_DIR / "v3_model_walkforward", DATA_DIR / "a_phase_history.parquet")
    strategy = V3Strategy(signal_source=signal_source, top_k_ratio=0.5, **params)
    config = BacktestConfig(
        start_date=date(2024, 1, 1), end_date=date(2025, 12, 31),
        initial_capital=1_000_000, rebalance_frequency="monthly",
        commission=0.0003, slippage=0.0005,
    )
    result = BacktestEngine(APhaseDataService(history)).run(
        config, StrategyConfig(), symbols=list(history["symbol"].unique()), strategy=strategy,
    )
    metrics = result.metrics
    reason_counts = result.trades["reason"].value_counts().to_dict() if not result.trades.empty else {}
    stop_exits = sum(v for k, v in reason_counts.items() if "STOP" in str(k))
    return {
        "trades": len(result.trades),
        "stop_exits": stop_exits,
        "total_return": metrics.get("overall_return"),
        "annual_return": metrics.get("annual_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe": metrics.get("sharpe"),
        "benchmark_return": metrics.get("benchmark_return"),
        "excess_return": metrics.get("excess_return"),
    }


def main() -> None:
    history = pd.read_parquet(DATA_DIR / "a_phase_history.parquet")
    rows = []
    for label, params in GRID:
        print(f"=== {label} {params} ===")
        row = run_one(history, params)
        row["config"] = label
        rows.append(row)
        print(row, "\n")

    table = pd.DataFrame(rows).set_index("config")
    print("=== 汇总 ===")
    print(table.to_string())


if __name__ == "__main__":
    main()
