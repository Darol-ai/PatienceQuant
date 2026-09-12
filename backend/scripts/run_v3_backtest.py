"""端到端跑一次 V3Strategy 回测：真实数据 + 训练好的 LightGBM 信号 +
BacktestEngine 的逐日退出钩子。不经过 SQLite/MarketDataService，直接从
data/a_phase_history.parquet 喂数据——A 阶段固定 10 支股票，不需要那一整套
目录/搜索能力。

用法：
    conda activate stock
    cd PatienceQuant/backend
    python scripts/run_v3_backtest.py
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


def main() -> None:
    history = pd.read_parquet(DATA_DIR / "a_phase_history.parquet")
    signal_source = WalkForwardSignalSource(
        DATA_DIR / "v3_model_walkforward", DATA_DIR / "a_phase_history.parquet"
    )
    # 入场机制已经从"绝对概率门槛"换成组内相对排序（top_k_ratio，见相关
    # ADR 和 app/strategies/v3_strategy.py 顶部说明），不再需要按年读取阈值。
    strategy = V3Strategy(signal_source=signal_source, top_k_ratio=0.5)

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

    print("=== 绩效指标 ===")
    for key, value in result.metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    print(f"\n=== 交易记录（共 {len(result.trades)} 笔） ===")
    if not result.trades.empty:
        print(result.trades.to_string(index=False))
        print("\n退出原因分布：")
        print(result.trades["reason"].value_counts())
    else:
        print("(无成交——占位信号 neutral_signal 永不批准买入是正常的；"
              "用了训练好的模型还是零成交，就要去查信号阈值/数据对不对了)")


if __name__ == "__main__":
    main()
