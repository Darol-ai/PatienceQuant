from __future__ import annotations

from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd

from app.factors.engine import FactorEngine
from app.strategies.lightgbm_signal import LightGBMSignal
from app.strategies.base import BaseStrategy, StrategyConfig, StrategyResult


class MultiFactorStrategy(BaseStrategy):
    def __init__(self, factor_engine: FactorEngine, config: StrategyConfig):
        self.factor_engine = factor_engine
        self.config = config
        self.signal_model = LightGBMSignal()

    @staticmethod
    def capped_weights(raw: Dict[str, float], max_weight: float) -> Dict[str, float]:
        if not raw:
            return {}
        if max_weight * len(raw) < 1 - 1e-9:
            raise ValueError("单股最大权重过低，无法构建满仓组合")
        remaining = 1.0
        active = set(raw)
        result = {symbol: 0.0 for symbol in raw}
        source = {symbol: max(float(weight), 0) for symbol, weight in raw.items()}
        while active and remaining > 1e-12:
            active_total = sum(source[s] for s in active)
            if active_total <= 0:
                allocation = remaining / len(active)
                for symbol in active:
                    result[symbol] += allocation
                break
            hit_cap = False
            for symbol in list(active):
                proposed = remaining * source[symbol] / active_total
                capacity = max_weight - result[symbol]
                if proposed >= capacity - 1e-12:
                    result[symbol] += capacity
                    remaining -= capacity
                    active.remove(symbol)
                    hit_cap = True
            if not hit_cap:
                for symbol in active:
                    allocation = remaining * source[symbol] / active_total
                    result[symbol] += allocation
                remaining = 0
        return {symbol: float(weight) for symbol, weight in result.items() if weight > 1e-8}

    def generate_weights(self, as_of: date, symbols: List[str]) -> StrategyResult:
        ranking, notes = self.factor_engine.score(as_of, symbols, self.config.weights)
        ranking = self._apply_benchmark_low_buy_high_sell_overlay(ranking)
        if self.config.model_enabled and not ranking.empty:
            feature_cols = [c for c in ranking.columns if c.startswith("score_") or c.startswith("factor_")]
            probs, backend = self.signal_model.probabilities(ranking[feature_cols])
            for col in probs.columns:
                ranking[col] = probs[col].to_numpy()
            ranking["model_signal"] = np.where(
                (ranking.p_up >= self.config.model_buy_threshold) &
                (ranking.p_down <= self.config.model_down_threshold), "BUY", "WATCH"
            )
            ranking["score"] = (ranking["score"] * (.75 + .5 * ranking.p_up)).clip(0, 100)
            notes.append(f"LightGBM信号层：{backend}；p_up≥{self.config.model_buy_threshold:.2f}且p_down≤{self.config.model_down_threshold:.2f}用于入场")
            ranking = ranking.sort_values(["score", "score_base"], ascending=[False, False]).reset_index(drop=True)
            ranking["rank"] = np.arange(1, len(ranking) + 1)
        top = ranking.head(self.config.holdings_count).copy()
        weight_source = top["score"].to_numpy(dtype=float)
        if {"dip_buy_score", "overheat_score"}.issubset(top.columns):
            dip_adjust = np.clip((top["dip_buy_score"].to_numpy(dtype=float) - 50.0) / 50.0, -1, 1)
            overheat_adjust = np.clip((top["overheat_score"].to_numpy(dtype=float) - 50.0) / 50.0, -1, 1)
            relative_adjust = (
                ((top.get("benchmark_relative_score", pd.Series(50, index=top.index)).to_numpy(dtype=float) - 50.0) / 50.0)
            )
            relative_adjust = np.clip(relative_adjust, -1, 1)
            multiplier = (
                1.0
                + self.config.dip_buy_strength * 0.55 * dip_adjust
                - self.config.profit_take_strength * 0.45 * np.maximum(overheat_adjust, 0.0)
                + (0.05 * relative_adjust if self.config.benchmark_enhancement else 0.0)
            )
            weight_source = weight_source * np.clip(multiplier, 0.72, 1.28)
        score = np.maximum(weight_source, 1e-6)
        raw = {symbol: float(value / score.sum()) for symbol, value in zip(top.symbol, score)}
        weights = self.capped_weights(raw, self.config.max_weight)
        regime = self.factor_engine.market_regime(as_of)
        normal_exposure = max(0.0, min(1.0, 1.0 - self.config.cash_buffer))
        regime_exposure = normal_exposure
        if self.config.trend_filter and regime.get("label") == "risk_off":
            regime_exposure = min(normal_exposure, self.config.risk_off_exposure)
            notes.append(
                "沪深300低于200日均线，组合进入风险关闭状态，目标股票暴露降至 %.0f%%"
                % (regime_exposure * 100)
            )
        elif self.config.trend_filter and regime.get("label") == "caution":
            regime_exposure = min(normal_exposure, (normal_exposure + self.config.risk_off_exposure) / 2)
            notes.append(
                "沪深300低于50日均线，组合进入谨慎状态，目标股票暴露降至 %.0f%%"
                % (regime_exposure * 100)
            )
        if regime_exposure < 1:
            weights = {symbol: weight * regime_exposure for symbol, weight in weights.items()}
        ranking["target_weight"] = ranking.symbol.map(weights).fillna(0.0)
        ranking["action"] = np.where(ranking.target_weight > 0, "BUY", "WATCH")
        ranking["market_regime"] = str(regime.get("label", "unknown"))
        ranking["target_exposure"] = regime_exposure
        return StrategyResult(
            as_of=as_of,
            weights=weights,
            ranking=ranking,
            data_quality_notes=notes,
            market_regime=str(regime.get("label", "unknown")),
            target_exposure=regime_exposure,
        )

    def _apply_benchmark_low_buy_high_sell_overlay(self, ranking: pd.DataFrame) -> pd.DataFrame:
        """Tilt the base factor score toward benchmark outperformance.

        This layer is deliberately transparent:
        - relative strength versus 沪深300 gets a modest bonus;
        - high-quality, low-valuation names after a short pullback get a
          "低吸" bonus;
        - expensive, fast-rising and high-risk names get a "高抛/过热" penalty.

        All columns come from ``FactorEngine`` and therefore only use data
        available on or before the signal date.
        """
        if ranking.empty:
            return ranking
        result = ranking.copy()

        def column(name: str, default: float = 50.0) -> pd.Series:
            if name in result.columns:
                return pd.to_numeric(result[name], errors="coerce").fillna(default)
            return pd.Series(default, index=result.index, dtype=float)

        result["score_base"] = column("score")
        valuation = column("score_valuation")
        quality = column("score_quality")
        risk = column("score_risk")
        return_3m = column("factor_return_3m")
        return_6m = column("factor_return_6m")
        return_12m = column("factor_return_12m")
        relative = column("factor_excess_return_12m", 50.0)

        # "低买" does not mean catching every falling knife.  The bonus is
        # strongest when valuation, quality and risk are healthy, while the
        # short-term price action has cooled down.
        trend_floor = np.where(return_12m >= 35, 1.0, np.where((quality >= 70) & (risk >= 70), 0.75, 0.35))
        result["dip_buy_score"] = (
            (100 - return_3m) * 0.30
            + valuation * 0.28
            + quality * 0.22
            + risk * 0.20
        ) * trend_floor

        # "高卖" trims crowded/overheated candidates: recent acceleration is
        # only penalised meaningfully when valuation or risk no longer offers
        # enough margin of safety.
        result["overheat_score"] = (
            return_3m * 0.28
            + return_6m * 0.22
            + (100 - valuation) * 0.27
            + (100 - risk) * 0.23
        )
        result["benchmark_relative_score"] = relative

        benchmark_bonus = (relative - 50.0) * 0.07 if self.config.benchmark_enhancement else 0.0
        dip_bonus = np.maximum(result["dip_buy_score"] - 52.0, 0.0) * self.config.dip_buy_strength
        overheat_penalty = np.maximum(result["overheat_score"] - 58.0, 0.0) * self.config.profit_take_strength
        result["enhancement_score"] = benchmark_bonus + dip_bonus - overheat_penalty
        result["score"] = (result["score_base"] + result["enhancement_score"]).clip(0, 100)
        result = result.sort_values(["score", "score_base"], ascending=[False, False]).reset_index(drop=True)
        result["rank"] = np.arange(1, len(result) + 1)
        return result
