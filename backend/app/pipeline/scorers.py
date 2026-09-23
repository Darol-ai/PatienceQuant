"""打分插槽：输入股票池和日期，输出"每支股票一个分数"（ADR-0047 第 2 条）。

因子权重打分和模型打分都实现同一个接口，后面的选股、权重、择时只认分数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from app.data.market_store import AShareMarketStore
from app.data.service import MarketDataService
from app.factors.engine import FactorEngine, GROUP_FACTORS
from app.pipeline.bars import load_bars, warmup_start
from app.pipeline.model_library import MODEL_LIBRARY, load_folds, model_years
from app.pipeline.spec import FactorWeightScorerSpec, ModelScorerSpec
from app.quant_v3.dataset import compute_features

# 真实模式下只有日线行情是真的。财务/估值数据还没接入本地行情库，
# MarketDataService.fundamentals() 在库里没有时会用程序生成的数补上——
# 这些数不能进入真实模式的打分。
PRICE_FACTOR_GROUPS = {"momentum", "risk"}
FUNDAMENTAL_FACTOR_GROUPS = set(GROUP_FACTORS) - PRICE_FACTOR_GROUPS


@dataclass
class ScoreResult:
    scores: pd.Series  # index=symbol，没有分数的股票为 NaN
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)  # index=symbol，展示用的中间结果
    notes: List[str] = field(default_factory=list)


class Scorer:
    #: 第一次打分前需要多少个交易日的历史
    warmup_days: int = 0

    def earliest_start(self) -> Optional[date]:
        """能打分的最早日期；None 表示只受行情库起点限制。"""
        return None

    def latest_end(self) -> Optional[date]:
        return None

    def prepare(self, symbols: List[str], start: date, end: date) -> None:
        """回测开始前一次性取好 [start, end] 需要的数据。"""

    def score(self, as_of: date, symbols: List[str]) -> ScoreResult:
        raise NotImplementedError


class FactorWeightScorer(Scorer):
    """每组因子在股票池里的百分位排名（0～100）按权重加总。"""

    warmup_days = 260

    def __init__(self, spec: FactorWeightScorerSpec, data: MarketDataService):
        self.data = data
        self.engine = FactorEngine(data)
        self.requested = {k: float(v) for k, v in spec.weights.items() if k in GROUP_FACTORS and float(v) > 0}
        self.real_mode = data.mode != "demo"
        dropped = FUNDAMENTAL_FACTOR_GROUPS & set(self.requested) if self.real_mode else set()
        self.effective = {k: v for k, v in self.requested.items() if k not in dropped}
        self.dropped = sorted(dropped)
        if not self.effective:
            raise ValueError("真实模式下没有可用的因子：%s 需要财务数据，本地行情库还没有" % "、".join(self.dropped))

    def prepare(self, symbols: List[str], start: date, end: date) -> None:
        warm = warmup_start(start, self.warmup_days)
        self.engine.prime(
            self.data.prices(symbols, warm, end),
            self.data.fundamentals(symbols, end),
            self.data.benchmark(warm, end),
        )

    def score(self, as_of: date, symbols: List[str]) -> ScoreResult:
        ranking, notes = self.engine.score(as_of, symbols, self.effective)
        if self.dropped:
            notes = [n for n in notes if not any(f in n for g in self.dropped for f in GROUP_FACTORS[g])]
            notes.append("%s 因子需要财务数据，真实模式下还没有，已按 0 权重计算" % "、".join(self.dropped))
        detail = ranking.set_index("symbol")
        # 当天还没有价格的股票（还没上市）因子全是用中位数补的，不能打分入选
        no_price = detail["current_price"].isna() if "current_price" in detail.columns else pd.Series(False, index=detail.index)
        scores = detail["score"].astype(float).mask(no_price)
        if self.dropped:
            # 程序生成的财务数字不能出现在真实模式的结果里，哪怕权重是 0。
            fake = [f for g in self.dropped for f in GROUP_FACTORS[g]]
            detail = detail.drop(columns=[c for c in detail.columns
                                          if c in fake or c.removeprefix("factor_") in fake
                                          or c.removeprefix("score_") in self.dropped], errors="ignore")
        return ScoreResult(scores=scores, detail=detail, notes=notes)


class ModelScorer(Scorer):
    """模型库里一个或多个模型的预测分数取平均；某个模型当天没分数就只平均其余的。"""

    # compute_features 需要当天之前至少 120 个交易日
    warmup_days = 130

    def __init__(self, spec: ModelScorerSpec, store: AShareMarketStore):
        missing = [m for m in spec.models if m not in MODEL_LIBRARY]
        if missing:
            raise ValueError("模型库里没有这些模型：%s" % "、".join(missing))
        self.model_ids = list(spec.models)
        self.store = store
        self._bars: Dict[str, List[dict]] = {}
        self._index: Dict[str, Dict[date, int]] = {}
        self._cache: Dict[date, pd.DataFrame] = {}
        years = [y for m in self.model_ids for y in model_years(m)]
        self._years = (min(years), max(years)) if years else None

    def earliest_start(self) -> Optional[date]:
        return date(self._years[0], 1, 1) if self._years else None

    def latest_end(self) -> Optional[date]:
        return date(self._years[1], 12, 31) if self._years else None

    def prepare(self, symbols: List[str], start: date, end: date) -> None:
        self._bars = load_bars(self.store, symbols, warmup_start(start, self.warmup_days), end)
        self._index = {s: {bar["date"]: i for i, bar in enumerate(bars)} for s, bars in self._bars.items()}
        self._cache = {}

    def _features(self, as_of: date, symbols: List[str]) -> pd.DataFrame:
        rows = {}
        for symbol in symbols:
            i = self._index.get(symbol, {}).get(as_of)
            if i is None:
                continue
            window = self._bars[symbol][max(0, i - 130): i + 1]
            features = compute_features(window, len(window) - 1)
            if features is not None:
                rows[symbol] = features
        return pd.DataFrame.from_dict(rows, orient="index")

    def score(self, as_of: date, symbols: List[str]) -> ScoreResult:
        if not self._bars:
            self.prepare(symbols, as_of, as_of)
        key = as_of
        if key not in self._cache:
            features = self._features(as_of, symbols)
            predictions = pd.DataFrame(index=features.index)
            for model_id in self.model_ids:
                fold = load_folds(model_id).get(as_of.year)
                if fold is not None and not features.empty:
                    predictions["pred_" + model_id.split("/", 1)[-1]] = fold.predict(features)
            self._cache[key] = predictions
        predictions = self._cache[key]
        scores = predictions.mean(axis=1) if not predictions.empty else pd.Series(dtype=float)
        scores = scores.reindex(symbols)
        # 不带日期和数量，回测里每期一样的提示会被合并成一条
        notes = (["部分股票在某些调仓日没有分数、不参与选股（上市不足 120 个交易日、当天停牌或该年没有模型）"]
                 if scores.isna().any() else [])
        return ScoreResult(scores=scores, detail=predictions.reindex(symbols), notes=notes)


def build_scorer(spec, data: MarketDataService, store: AShareMarketStore) -> Scorer:
    if isinstance(spec, FactorWeightScorerSpec):
        return FactorWeightScorer(spec, data)
    if isinstance(spec, ModelScorerSpec):
        return ModelScorer(spec, store)
    raise ValueError(f"未知的打分方式：{spec}")
