"""把"研报生成的因子"接进回测系统：一个简单的按打分选 top-N、等权配置的
策略，不做止损/冷静期这类 V3 才有的复杂状态机——生成的因子本来就不要求
收益好，只要求这条链路（因子 → 打分 → 选股 → 回测）是真的在跑。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from app.quant_v3.a_phase_universe import A_PHASE_STOCKS
from app.quant_v3.dataset import compute_features
from app.strategies.base import BaseStrategy, StrategyConfig, StrategyResult

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def top_n_equal_weight(scores: Dict[str, float], top_n: int) -> Dict[str, float]:
    """按打分从高到低选 top_n，等权分配；并列按代码升序，保证结果确定、
    可复现（同 quant_v3.selection 系列模块的约定）。"""
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top = ranked[:top_n]
    if not top:
        return {}
    weight = 1 / len(top)
    return {symbol: weight for symbol, _ in top}


class GeneratedFactorStrategy(BaseStrategy):
    """feature_weights：{特征名: 权重}，特征名必须来自 quant_v3.features 那
    14 项（由调用方在生成阶段就做白名单校验，这里不再重复校验）。
    每只股票的打分 = sum(权重 * 特征值)，历史不够 120 天时该股票跳过。
    """

    def __init__(self, feature_weights: Dict[str, float], history: pd.DataFrame, top_n: int = 5):
        self.feature_weights = feature_weights
        self.top_n = top_n
        self._bars_by_symbol: Dict[str, List[dict]] = {}
        self._index_by_symbol: Dict[str, Dict[date, int]] = {}
        for symbol, group_df in history.sort_values("date").groupby("symbol"):
            bars = group_df.to_dict(orient="records")
            self._bars_by_symbol[symbol] = bars
            self._index_by_symbol[symbol] = {
                pd.Timestamp(bar["date"]).date(): i for i, bar in enumerate(bars)
            }

    def _score(self, symbol: str, as_of: date) -> Optional[float]:
        bars = self._bars_by_symbol.get(symbol)
        index_map = self._index_by_symbol.get(symbol)
        if bars is None or index_map is None:
            return None
        t_index = index_map.get(as_of)
        if t_index is None:
            return None
        features = compute_features(bars, t_index)
        if features is None:
            return None
        return sum(weight * features[name] for name, weight in self.feature_weights.items() if name in features)

    def generate_weights(self, as_of: date, symbols: List[str]) -> StrategyResult:
        universe = symbols or [stock["symbol"] for stock in A_PHASE_STOCKS]
        scores = {}
        for symbol in universe:
            score = self._score(symbol, as_of)
            if score is not None:
                scores[symbol] = score

        weights = top_n_equal_weight(scores, self.top_n)
        ranking = pd.DataFrame(
            [{"symbol": symbol, "score": score, "target_weight": weights.get(symbol, 0.0),
              "action": "BUY" if symbol in weights else "WATCH"}
             for symbol, score in scores.items()]
        )
        return StrategyResult(
            as_of=as_of,
            weights=weights,
            ranking=ranking,
            data_quality_notes=["研报生成因子策略：不做止损/冷静期，只验证因子能否接入回测系统"],
        )


def run_generated_factor_backtest(
    feature_weights: Dict[str, float],
    start: date,
    end: date,
    top_n: int = 5,
    history_path: Optional[Path] = None,
):
    """把研报生成的因子真正接进 A 阶段回测系统跑一遍——功能②的落地点。
    延迟导入 BacktestEngine，避免这个模块在只用 GeneratedFactorStrategy
    做单测时也要拉全套引擎依赖。
    """
    from app.backtest.engine import BacktestConfig, BacktestEngine
    from app.quant_v3.a_phase_data_service import APhaseDataService

    history = pd.read_parquet(history_path or (DATA_DIR / "a_phase_history.parquet"))
    strategy = GeneratedFactorStrategy(feature_weights=feature_weights, history=history, top_n=top_n)
    symbols = [stock["symbol"] for stock in A_PHASE_STOCKS]
    config = BacktestConfig(start_date=start, end_date=end, initial_capital=1_000_000, rebalance_frequency="monthly")
    return BacktestEngine(APhaseDataService(history)).run(
        config, StrategyConfig(), symbols=symbols, strategy=strategy,
    )
