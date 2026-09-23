"""把打分 → ① 选股 → ② 权重 → ③ × 择时 串成一个策略；④ 调仓由回测引擎/模拟盘执行。"""
from __future__ import annotations

from datetime import date
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from app.data.market_refresh import get_market_store
from app.data.service import MarketDataService
from app.pipeline.portfolio import select, weigh
from app.pipeline.scorers import Scorer, build_scorer
from app.pipeline.spec import StrategySpec
from app.pipeline.timing import TimingSignal, build_timing
from app.strategies.base import BaseStrategy, StrategyResult


class PipelineStrategy(BaseStrategy):
    def __init__(self, spec: StrategySpec, scorer: Scorer, timing: TimingSignal):
        self.spec = spec
        self.scorer = scorer
        self.timing = timing
        # 股票池成员随时间变化时（如沪深300 历史成分股），给出某一天可以入选的股票；None 表示不变
        self.members_at: Optional[Callable[[date], set]] = None

    def earliest_start(self) -> Optional[date]:
        return self.scorer.earliest_start()

    def latest_end(self) -> Optional[date]:
        return self.scorer.latest_end()

    def prepare(self, symbols: List[str], start: date, end: date) -> None:
        self.scorer.prepare(symbols, start, end)
        self.timing.prepare(start, end)

    def generate_weights(self, as_of: date, symbols: List[str]) -> StrategyResult:
        # 股票池成员随时间变化时只给当天的成员打分：因子分是池内百分位，混进非成员会改变排序；
        # 不在指数里的股票没有分数、不能入选（已持有的会在这次调仓被卖出）
        candidates = symbols
        if self.members_at is not None:
            members = self.members_at(as_of)
            candidates = [symbol for symbol in symbols if symbol in members]
        scored = self.scorer.score(as_of, candidates)
        scores = scored.scores.reindex(symbols)
        if self.members_at is not None:
            scores = scores.where(scores.index.isin(candidates))
        selected = select(scores, self.spec.selection, len(candidates))
        base_weights = weigh(selected, scores, self.spec.weighting)
        weights = base_weights
        timing = self.timing.exposure(as_of)
        weights = {symbol: w * timing.exposure for symbol, w in weights.items()}

        ranking = scored.detail.reindex(symbols).drop(columns=["score", "rank", "target_weight"], errors="ignore")
        ranking.insert(0, "score", scores)
        ranking.index.name = "symbol"
        ranking = ranking.reset_index()
        ranking = ranking.sort_values("score", ascending=False, na_position="last", kind="mergesort").reset_index(drop=True)
        ranking["rank"] = np.where(ranking["score"].notna(), np.arange(1, len(ranking) + 1), np.nan)
        ranking["target_weight"] = ranking["symbol"].map(weights).fillna(0.0)
        ranking["action"] = np.where(ranking["target_weight"] > 0, "BUY", "WATCH")
        ranking["market_regime"] = timing.label
        ranking["target_exposure"] = timing.exposure

        notes = list(scored.notes)
        if timing.note:
            notes.append(timing.note)
        invested = sum(weights.values())
        if invested < timing.exposure - 1e-6 and weights:
            notes.append("单股上限 %.0f%% × %d 支不够满仓，剩余 %.0f%% 留作现金"
                         % (self.spec.weighting.max_weight * 100, len(weights), (timing.exposure - invested) * 100))
        return StrategyResult(
            as_of=as_of,
            weights=weights,
            ranking=ranking,
            data_quality_notes=notes,
            market_regime=timing.label,
            target_exposure=timing.exposure,
            base_weights=base_weights,
        )

    def daily_exposure(self, signal_date: date):
        if self.spec.timing.type == "none":
            return None
        timing = self.timing.exposure(signal_date)
        return timing.exposure, timing.label


def build_pipeline_strategy(spec: StrategySpec, data: MarketDataService) -> PipelineStrategy:
    store = get_market_store()
    return PipelineStrategy(spec, build_scorer(spec.scorer, data, store), build_timing(spec.timing, data))
