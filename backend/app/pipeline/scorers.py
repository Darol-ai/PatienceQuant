"""打分插槽：输入股票池和日期，输出"每支股票一个分数"（ADR-0047 第 2 条）。

因子权重打分和模型打分都实现同一个接口，后面的选股、权重、择时只认分数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.data.market_store import AShareMarketStore
from app.data.service import MarketDataService
from app.factors.engine import FactorEngine, GROUP_FACTORS
from app.pipeline.bars import warmup_start
from app.pipeline.price_factors import PRICE_FACTORS
from app.pipeline.model_library import load_folds, model_exists, model_factors, model_uses_cs_rank, model_years
from app.pipeline.spec import FactorWeightScorerSpec, ModelScorerSpec

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
    """每个因子（组）在股票池里的百分位排名（0～100）按权重加总。

    权重的键可以是五个因子组（FactorEngine 的 fundamental/valuation/quality/
    momentum/risk）、price_factors 里的单个价量因子，也可以是因子库里任何只用
    日线的因子（整段区间用面板函数一次算好，和模型训练同一套代码）。
    """

    warmup_days = 260

    def __init__(self, spec: FactorWeightScorerSpec, data: MarketDataService, store: Optional[AShareMarketStore] = None):
        from app.pipeline.factor_library import FACTOR_LIBRARY, trainable

        self.data = data
        self.store = store
        self.engine = FactorEngine(data)
        library_keys = {k for k in FACTOR_LIBRARY if trainable(k) and k not in PRICE_FACTORS and k not in GROUP_FACTORS}
        unknown = [k for k in spec.weights if k not in GROUP_FACTORS and k not in PRICE_FACTORS and k not in library_keys]
        if unknown:
            raise ValueError("未知的因子：%s" % "、".join(unknown))
        self.requested = {k: float(v) for k, v in spec.weights.items() if float(v) > 0}
        self.real_mode = data.mode != "demo"
        dropped = FUNDAMENTAL_FACTOR_GROUPS & set(self.requested) if self.real_mode else set()
        self.effective = {k: v for k, v in self.requested.items() if k not in dropped}
        self.dropped = sorted(dropped)
        if not self.effective:
            raise ValueError("真实模式下没有可用的因子：%s 需要财务数据，本地行情库还没有" % "、".join(self.dropped))
        self.group_weights = {k: v for k, v in self.effective.items() if k in GROUP_FACTORS}
        self.price_weights = {k: v for k, v in self.effective.items() if k in PRICE_FACTORS}
        self.library_weights = {k: v for k, v in self.effective.items() if k in library_keys}
        if self.library_weights:
            self.warmup_days = 300  # 一年期因子（如 12 个月收益）要约 252 个交易日
        self._panel: Dict[str, pd.DataFrame] = {}
        self._library_values: Dict[str, pd.DataFrame] = {}

    def prepare(self, symbols: List[str], start: date, end: date) -> None:
        warm = warmup_start(start, self.warmup_days)
        prices = self.data.prices(symbols, warm, end)
        self.engine.prime(prices, self.data.fundamentals(symbols, end), self.data.benchmark(warm, end))
        if self.price_weights and not prices.empty:
            frame = prices.assign(trade_date=pd.to_datetime(prices["trade_date"]))
            self._panel = {
                column: frame.pivot_table(index="trade_date", columns="symbol", values=column, aggfunc="last").sort_index()
                for column in ("open", "high", "low", "close")
            }
        if self.library_weights:
            self._prepare_library(symbols, start, end)

    def _prepare_library(self, symbols: List[str], start: date, end: date) -> None:
        from app.data.market_refresh import BENCHMARK_INDEX, get_market_store
        from app.pipeline.factor_library import compute_factors, panels_from_store

        store = self.store or get_market_store()
        warm = warmup_start(start, self.warmup_days)
        bench = store.load_index(BENCHMARK_INDEX, warm, end)
        bench_series = pd.Series(bench["close"].to_numpy(dtype=float), index=pd.to_datetime(bench["trade_date"])) if not bench.empty else None
        panels = panels_from_store(store, symbols, warm, end, benchmark=bench_series)
        self._library_values = compute_factors(panels, list(self.library_weights))

    def _library_factor_scores(self, as_of: date, symbols: List[str]) -> pd.DataFrame:
        from app.pipeline.factor_library import FACTOR_LIBRARY

        result = pd.DataFrame(index=pd.Index(symbols, name="symbol"))
        day = pd.Timestamp(as_of)
        for key in self.library_weights:
            values = self._library_values.get(key)
            raw = values.loc[day].reindex(symbols) if values is not None and day in values.index else pd.Series(np.nan, index=symbols)
            raw = raw.replace([np.inf, -np.inf], np.nan)
            result["factor_" + key] = raw
            result["score_" + key] = raw.rank(pct=True, ascending=FACTOR_LIBRARY[key].direction > 0) * 100
        return result

    def _price_factor_scores(self, as_of: date, symbols: List[str]) -> pd.DataFrame:
        cut = {k: v.loc[: pd.Timestamp(as_of)].reindex(columns=symbols) for k, v in self._panel.items()}
        result = pd.DataFrame(index=pd.Index(symbols, name="symbol"))
        if not cut or cut["close"].empty:
            return result
        # 当天没有行情（停牌/还没上市）的股票不给因子值
        traded_today = cut["close"].index[-1] == pd.Timestamp(as_of)
        today = cut["close"].iloc[-1].notna() if traded_today else pd.Series(False, index=symbols)
        for key in self.price_weights:
            factor = PRICE_FACTORS[key]
            window = {k: v.tail(factor.lookback_days + 10) for k, v in cut.items()}
            raw = factor.compute(window).reindex(symbols).where(today)
            result["factor_" + key] = raw
            result["score_" + key] = raw.rank(pct=True, ascending=factor.direction > 0) * 100
        return result

    def score(self, as_of: date, symbols: List[str]) -> ScoreResult:
        notes: List[str] = []
        parts = []
        if self.group_weights:
            ranking, notes = self.engine.score(as_of, symbols, self.group_weights)
            detail = ranking.set_index("symbol")
            # 当天还没有价格的股票（还没上市）因子全是用中位数补的，不能打分入选
            if "current_price" in detail.columns:
                detail.loc[detail["current_price"].isna(), "score"] = np.nan
            parts.append((sum(self.group_weights.values()), detail["score"].astype(float)))
        else:
            detail = pd.DataFrame(index=pd.Index(symbols, name="symbol"))
        if self.price_weights:
            price_detail = self._price_factor_scores(as_of, symbols)
            detail = detail.join(price_detail.drop(columns=[c for c in price_detail.columns if c in detail.columns]), how="outer")
            for key, weight in self.price_weights.items():
                parts.append((weight, price_detail["score_" + key]))
        if self.library_weights:
            if not self._library_values:
                self._prepare_library(symbols, as_of, as_of)
            library_detail = self._library_factor_scores(as_of, symbols)
            detail = detail.join(library_detail.drop(columns=[c for c in library_detail.columns if c in detail.columns]), how="outer")
            for key, weight in self.library_weights.items():
                parts.append((weight, library_detail["score_" + key]))
        if "name" not in detail.columns:
            catalog = self.data.stocks()
            if not catalog.empty:
                detail = detail.join(catalog.set_index("symbol")[["name", "industry", "group", "exchange"]], how="left")
        total = sum(w for w, _ in parts)
        combined = sum(w * series.reindex(symbols) for w, series in parts) / total
        detail["score"] = combined
        if self.dropped:
            notes = [n for n in notes if not any(f in n for g in self.dropped for f in GROUP_FACTORS[g])]
            notes.append("%s 因子需要财务数据，真实模式下还没有，已按 0 权重计算" % "、".join(self.dropped))
            # 程序生成的财务数字不能出现在真实模式的结果里，哪怕权重是 0。
            fake = [f for g in self.dropped for f in GROUP_FACTORS[g]]
            detail = detail.drop(columns=[c for c in detail.columns
                                          if c in fake or c.removeprefix("factor_") in fake
                                          or c.removeprefix("score_") in self.dropped], errors="ignore")
        return ScoreResult(scores=combined.astype(float), detail=detail, notes=notes)


class ModelScorer(Scorer):
    """模型库里一个或多个模型的预测分数取平均；某个模型当天没分数就只平均其余的。

    输入因子用因子库的面板函数计算（和训练时同一套代码）。旧模型的 14 个因子要求
    股票在行情里满 121 个交易日，所以取数时往前多留 300 个交易日。
    """

    warmup_days = 300

    def __init__(self, spec: ModelScorerSpec, store: AShareMarketStore):
        missing = [m for m in spec.models if not model_exists(m)]
        if missing:
            raise ValueError("模型库里没有这些模型：%s" % "、".join(missing))
        self.model_ids = list(spec.models)
        self.store = store
        self.factor_keys = list(dict.fromkeys(k for m in self.model_ids for k in model_factors(m)))
        self._values: Dict[str, pd.DataFrame] = {}
        self._cache: Dict[date, pd.DataFrame] = {}
        years = [y for m in self.model_ids for y in model_years(m)]
        self._years = (min(years), max(years)) if years else None

    def earliest_start(self) -> Optional[date]:
        return date(self._years[0], 1, 1) if self._years else None

    def latest_end(self) -> Optional[date]:
        return date(self._years[1], 12, 31) if self._years else None

    def prepare(self, symbols: List[str], start: date, end: date) -> None:
        from app.data.market_refresh import BENCHMARK_INDEX
        from app.pipeline.factor_library import compute_factors, panels_from_store

        warm = warmup_start(start, self.warmup_days)
        bench = self.store.load_index(BENCHMARK_INDEX, warm, end)
        bench_series = pd.Series(bench["close"].to_numpy(dtype=float), index=pd.to_datetime(bench["trade_date"])) if not bench.empty else None
        panels = panels_from_store(self.store, symbols, warm, end, benchmark=bench_series)
        self._values = compute_factors(panels, self.factor_keys)
        self._cache = {}

    def _factors_on(self, as_of: date, symbols: List[str]) -> pd.DataFrame:
        day = pd.Timestamp(as_of)
        rows = {key: frame.loc[day].reindex(symbols) if day in frame.index else pd.Series(np.nan, index=symbols)
                for key, frame in self._values.items()}
        return pd.DataFrame(rows, index=pd.Index(symbols, name="symbol"))

    def score(self, as_of: date, symbols: List[str]) -> ScoreResult:
        if not self._values:
            self.prepare(symbols, as_of, as_of)
        if as_of not in self._cache:
            factors = self._factors_on(as_of, symbols)
            predictions = pd.DataFrame(index=factors.index)
            for model_id in self.model_ids:
                fold = load_folds(model_id).get(as_of.year)
                inputs = factors[model_factors(model_id)].dropna()
                if fold is None or inputs.empty:
                    continue
                if model_uses_cs_rank(model_id):
                    inputs = inputs.rank(pct=True)  # 和训练时一样：当天在股票池里的百分位
                predictions.loc[inputs.index, "pred_" + model_id.split("/", 1)[-1]] = fold.predict(inputs)
            self._cache[as_of] = predictions
        predictions = self._cache[as_of]
        scores = predictions.mean(axis=1) if not predictions.empty else pd.Series(dtype=float)
        scores = scores.reindex(symbols)
        # 不带日期和数量，回测里每期一样的提示会被合并成一条
        notes = (["部分股票在某些调仓日没有分数、不参与选股（上市不足 120 个交易日、当天停牌或该年没有模型）"]
                 if scores.isna().any() else [])
        return ScoreResult(scores=scores, detail=predictions.reindex(symbols), notes=notes)


def build_scorer(spec, data: MarketDataService, store: AShareMarketStore) -> Scorer:
    if isinstance(spec, FactorWeightScorerSpec):
        return FactorWeightScorer(spec, data, store)
    if isinstance(spec, ModelScorerSpec):
        return ModelScorer(spec, store)
    raise ValueError(f"未知的打分方式：{spec}")
