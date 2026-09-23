"""股票池分析（ADR-0053）：分析选股范围本身，不涉及策略。只读本地行情库，最近约一年。"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

LOOKBACK_TRADING_DAYS = 250


def analyze(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    from app.data.market_refresh import BENCHMARK_INDEX, get_market_store
    from app.data.tushare_provider import _to_ts_code

    store = get_market_store()
    end = store.latest_trading_day()
    days = [d for d in store.stored_days() if d <= end][-LOOKBACK_TRADING_DAYS - 1:]
    start, end = days[0], days[-1]  # 当天行情还没发布时，截止到库里最新的一天
    code_to_symbol = {_to_ts_code(m["symbol"]): m["symbol"] for m in members}
    daily = store.load_daily(start, end, list(code_to_symbol), adjust="qfq")
    close = daily.assign(symbol=daily["ts_code"].map(code_to_symbol)).pivot_table(
        index="trade_date", columns="symbol", values="close", aggfunc="last").sort_index()

    by_symbol = {m["symbol"]: m for m in members}
    points = []
    for symbol in close.columns:
        series = close[symbol].dropna()
        if len(series) < 60:
            continue  # 近一年交易不足 60 天（新股、长期停牌）不画点
        ret = series.pct_change().dropna()
        member = by_symbol[symbol]
        points.append({"symbol": symbol, "name": member["name"], "group": member.get("sector") or member.get("industry") or "未分类",
                       "return": float(series.iloc[-1] / series.iloc[0] - 1), "volatility": float(ret.std() * np.sqrt(252)),
                       "days": int(len(series))})

    # 分组：有研究标注的用"所属板块"，否则用行业
    groups = Counter((m.get("sector") or m.get("industry") or "未分类") for m in members)
    grouped_by = "板块（研究标注）" if any(m.get("sector") for m in members) else "行业"

    # 股票池等权组合：每天对当天有行情的成员等权，日收益取平均（不含交易成本）
    daily_ret = close.pct_change(fill_method=None)
    pool_ret = daily_ret.mean(axis=1, skipna=True).fillna(0)
    pool_curve = (1 + pool_ret).cumprod()
    bench = store.load_index(BENCHMARK_INDEX, start, end)
    bench_close = pd.Series(bench["close"].to_numpy(dtype=float), index=pd.to_datetime(bench["trade_date"])).reindex(pool_curve.index).ffill()
    bench_curve = bench_close / bench_close.iloc[0]
    curve = [{"date": d.date().isoformat(), "pool": float(p), "benchmark": float(b)}
             for d, p, b in zip(pool_curve.index, pool_curve, bench_curve) if np.isfinite(b)]

    returns = [p["return"] for p in points]
    return {
        "start": start.isoformat(), "end": end.isoformat(), "grouped_by": grouped_by,
        "groups": [{"name": k, "count": v} for k, v in groups.most_common()],
        "points": points, "curve": curve,
        "summary": {
            "members": len(members), "analyzed": len(points),
            "pool_return": float(pool_curve.iloc[-1] - 1) if len(pool_curve) else None,
            "benchmark_return": float(bench_curve.dropna().iloc[-1] - 1) if bench_curve.notna().any() else None,
            "median_return": float(np.median(returns)) if returns else None,
            "up_ratio": float(np.mean([r > 0 for r in returns])) if returns else None,
        },
    }
