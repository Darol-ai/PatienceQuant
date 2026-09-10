from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.data.service import MarketDataService


GROUP_FACTORS: Dict[str, List[str]] = {
    "fundamental": ["roe", "roa", "revenue_growth", "profit_growth", "operating_cashflow"],
    "valuation": ["pe", "pb", "ps", "dividend_yield"],
    "quality": ["roe_stability", "gross_margin", "net_margin", "cashflow_profit_ratio"],
    "momentum": ["return_3m", "return_6m", "return_12m"],
    "risk": ["volatility", "max_drawdown", "beta"],
}

LOW_IS_BETTER = {"pe", "pb", "ps", "volatility", "max_drawdown", "beta"}


class FactorEngine:
    def __init__(self, data: MarketDataService):
        self.data = data
        self._wide_price_cache: Optional[pd.DataFrame] = None
        self._fundamental_cache: Optional[pd.DataFrame] = None
        self._benchmark_cache: Optional[pd.DataFrame] = None

    def prime(self, prices: pd.DataFrame, fundamentals: pd.DataFrame, benchmark: pd.DataFrame) -> None:
        """Prime one backtest's immutable market snapshot for repeated factor dates."""
        prepared = prices.copy()
        prepared["trade_date"] = pd.to_datetime(prepared["trade_date"])
        self._wide_price_cache = prepared.pivot_table(
            index="trade_date", columns="symbol", values="adj_close", aggfunc="last"
        ).sort_index().ffill()
        self._fundamental_cache = fundamentals.copy()
        if not self._fundamental_cache.empty:
            self._fundamental_cache["report_date"] = pd.to_datetime(self._fundamental_cache["report_date"]).dt.date
        self._benchmark_cache = benchmark.copy()
        if not self._benchmark_cache.empty:
            self._benchmark_cache["trade_date"] = pd.to_datetime(self._benchmark_cache["trade_date"])

    def market_regime(self, as_of: date) -> Dict[str, object]:
        """Return a low-frequency benchmark trend regime.

        The regime is deliberately simple and auditable: it only uses
        benchmark observations available on or before ``as_of``.  A close
        below its 200-session average reduces invested exposure; a close
        below the 50-session average is a softer caution state.  This is a
        portfolio-level risk gate, not a prediction of future returns.
        """
        benchmark = self._benchmark_cache
        if benchmark is None or benchmark.empty:
            benchmark = self.data.benchmark(date(2018, 1, 1), as_of)
            if not benchmark.empty:
                benchmark = benchmark.copy()
                benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
        if benchmark is None or benchmark.empty:
            return {"label": "unknown", "exposure": 1.0, "close": 0.0, "ma50": 0.0, "ma200": 0.0}
        frame = benchmark[benchmark.trade_date <= pd.Timestamp(as_of)].sort_values("trade_date")
        if frame.empty:
            return {"label": "unknown", "exposure": 1.0, "close": 0.0, "ma50": 0.0, "ma200": 0.0}
        close_series = frame["adj_close"].astype(float)
        close = float(close_series.iloc[-1])
        ma50 = float(close_series.tail(50).mean())
        ma200 = float(close_series.tail(200).mean())
        if len(close_series) < 50:
            label, exposure = "early_sample", 1.0
        elif close < ma200:
            label, exposure = "risk_off", .55
        elif close < ma50:
            label, exposure = "caution", .75
        else:
            label, exposure = "risk_on", 1.0
        return {"label": label, "exposure": exposure, "close": close, "ma50": ma50, "ma200": ma200}

    @staticmethod
    def _safe_return(series: pd.Series, periods: int) -> float:
        clean = series.dropna()
        if len(clean) < 2:
            return 0.0
        start_index = max(0, len(clean) - periods - 1)
        base = float(clean.iloc[start_index])
        return float(clean.iloc[-1] / base - 1) if base else 0.0

    def raw_factors(self, as_of: date, symbols: List[str]) -> pd.DataFrame:
        if self._wide_price_cache is not None:
            wide = self._wide_price_cache.loc[:pd.Timestamp(as_of)].reindex(columns=symbols).ffill()
            fundamentals = self._fundamental_cache
            fundamentals = fundamentals[fundamentals.report_date <= as_of] if fundamentals is not None and not fundamentals.empty else pd.DataFrame()
            benchmark = self._benchmark_cache
            benchmark = benchmark[benchmark.trade_date <= pd.Timestamp(as_of)] if benchmark is not None and not benchmark.empty else pd.DataFrame()
        else:
            prices = self.data.prices(symbols, date(2018, 1, 1), as_of)
            fundamentals = self.data.fundamentals(symbols, as_of)
            benchmark = self.data.benchmark(date(2018, 1, 1), as_of)
            if prices.empty:
                return pd.DataFrame(index=symbols)
            prices = prices.copy()
            prices["trade_date"] = pd.to_datetime(prices["trade_date"])
            wide = prices.pivot_table(index="trade_date", columns="symbol", values="adj_close", aggfunc="last").sort_index()
            wide = wide.reindex(columns=symbols).ffill()
        if wide.empty:
            return pd.DataFrame(index=symbols)
        latest_fund = (
            fundamentals.sort_values("report_date").groupby("symbol", as_index=False).tail(1).set_index("symbol")
            if not fundamentals.empty
            else pd.DataFrame(index=symbols)
        )
        # Vectorize the historical calculations across the whole cross-section.
        # This matters once the offline universe reaches 1,000+ symbols: the
        # strategy still evaluates every symbol, but avoids a Python loop per
        # symbol at every monthly rebalance.
        current = wide.iloc[-1] if len(wide) else pd.Series(index=symbols, dtype=float)
        returns = wide.pct_change(fill_method=None)

        def period_return(periods: int) -> pd.Series:
            if len(wide) < 2:
                return pd.Series(0.0, index=symbols)
            start = max(0, len(wide) - periods - 1)
            return wide.iloc[-1].div(wide.iloc[start]).sub(1.0)

        return_3m = period_return(63)
        return_6m = period_return(126)
        return_12m = period_return(252)
        tail_returns = returns.tail(252)
        volatility = tail_returns.std() * np.sqrt(252)
        price_window = wide.tail(252)
        max_drawdown = (price_window.div(price_window.cummax()) - 1.0).min().abs()

        benchmark_returns = pd.Series(
            benchmark["adj_close"].to_numpy(dtype=float), index=pd.to_datetime(benchmark["trade_date"])
        ).pct_change()
        benchmark_prices = pd.Series(
            benchmark["adj_close"].to_numpy(dtype=float), index=pd.to_datetime(benchmark["trade_date"])
        ).dropna()
        if len(benchmark_prices) >= 2:
            start = max(0, len(benchmark_prices) - 252 - 1)
            benchmark_12m = float(benchmark_prices.iloc[-1] / benchmark_prices.iloc[start] - 1.0)
        else:
            benchmark_12m = 0.0
        benchmark_tail = benchmark_returns.reindex(tail_returns.index).ffill().bfill()
        benchmark_centered = benchmark_tail - benchmark_tail.mean()
        centered = tail_returns.sub(tail_returns.mean(axis=0), axis=1)
        denominator = float((benchmark_centered ** 2).sum())
        beta = centered.mul(benchmark_centered, axis=0).sum(axis=0).div(denominator) if denominator > 0 else pd.Series(1.0, index=symbols)
        beta = beta.abs().replace([np.inf, -np.inf], np.nan).fillna(1.0)

        result = pd.DataFrame(
            {
                "current_price": current,
                "return_3m": return_3m,
                "return_6m": return_6m,
                "return_12m": return_12m,
                "excess_return_12m": return_12m - benchmark_12m,
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "beta": beta,
            },
            index=pd.Index(symbols, name="symbol"),
        )
        if not latest_fund.empty:
            fund_values = latest_fund.reindex(symbols)
            for factor in GROUP_FACTORS["fundamental"] + GROUP_FACTORS["valuation"] + GROUP_FACTORS["quality"]:
                result[factor] = fund_values[factor]
        return result

    @staticmethod
    def _fill_missing(frame: pd.DataFrame, catalog: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        result = frame.copy()
        notes: List[str] = []
        industry_by_symbol = catalog.set_index("symbol")["industry"]
        for column in result.columns:
            if not pd.api.types.is_numeric_dtype(result[column]):
                continue
            missing = int(result[column].isna().sum())
            if not missing:
                continue
            mapped_industry = result.index.to_series().map(industry_by_symbol)
            medians = result[column].groupby(mapped_industry).transform("median")
            result[column] = result[column].fillna(medians).fillna(result[column].median()).fillna(0)
            notes.append("%s 缺失 %s 项，已使用行业/截面中位数补齐" % (column, missing))
        return result, notes

    def score(self, as_of: date, symbols: List[str], group_weights: Dict[str, float]) -> tuple[pd.DataFrame, List[str]]:
        catalog = self.data.stocks()
        raw = self.raw_factors(as_of, symbols)
        raw, notes = self._fill_missing(raw, catalog)
        score_frame = pd.DataFrame(index=raw.index)
        for group, factors in GROUP_FACTORS.items():
            factor_scores = []
            for factor in factors:
                values = raw[factor].replace([np.inf, -np.inf], np.nan).fillna(0)
                factor_score = values.rank(pct=True, ascending=factor not in LOW_IS_BETTER)
                score_frame["factor_%s" % factor] = factor_score * 100
                factor_scores.append(factor_score)
            score_frame["score_%s" % group] = pd.concat(factor_scores, axis=1).mean(axis=1) * 100
        if "excess_return_12m" in raw.columns:
            score_frame["factor_excess_return_12m"] = raw["excess_return_12m"].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0).rank(pct=True, ascending=True) * 100
        normalized_weights = {key: float(group_weights.get(key, 0)) for key in GROUP_FACTORS}
        total = sum(normalized_weights.values()) or 1
        normalized_weights = {key: value / total for key, value in normalized_weights.items()}
        score_frame["score"] = sum(score_frame["score_%s" % key] * weight for key, weight in normalized_weights.items())
        result = raw.join(score_frame).reset_index().merge(catalog, on="symbol", how="left")
        result["rank"] = result["score"].rank(method="first", ascending=False).astype(int)
        return result.sort_values("rank"), notes
