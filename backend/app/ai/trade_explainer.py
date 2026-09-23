"""解释一笔真实交易（ADR-0051：交易解释挂在回测报告和模拟盘的具体交易上）。

只陈述事实：在信号日重新跑一次这个策略的打分，给出这支股票的名次、分数、入选门槛、
目标权重和当天的择时仓位，再加上成交本身。不编"基本面支撑"之类没有依据的理由。
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import Strategy
from app.pipeline.library import data_service_for, strategy_spec
from app.pipeline.strategy import build_pipeline_strategy


def previous_trading_day(data, day: date) -> Optional[date]:
    bench = data.benchmark(day - timedelta(days=20), day)
    days = sorted(pd.to_datetime(bench["trade_date"]).dt.date) if not bench.empty else []
    earlier = [d for d in days if d < day]
    return earlier[-1] if earlier else None


def _fmt_score(value: float, scorer: str) -> str:
    return f"{value:.4f}" if scorer == "model" else f"{value:.1f}"


def explain_trade(db: Session, strategy: Strategy, symbols: List[str], signal_date: date, trade: Dict[str, Any],
                  name: str, spec=None) -> Dict[str, Any]:
    """spec 不给时用策略当前的规格；回测交易要传这次回测实际执行的规格。"""
    spec = spec or strategy_spec(strategy)
    data = data_service_for(db, strategy)
    pipeline = build_pipeline_strategy(spec, data)
    pipeline.prepare(symbols, signal_date, signal_date)
    result = pipeline.generate_weights(signal_date, symbols)
    ranking = result.ranking.set_index("symbol")
    scored = ranking["score"].notna()
    n_scored = int(scored.sum())
    k = spec.selection.n if spec.selection.type == "top_n" else max(1, round(len(symbols) * spec.selection.pct))
    selected = ranking[ranking["target_weight"] > 0]
    cutoff = float(selected["score"].min()) if not selected.empty else None
    symbol = trade["symbol"].split(".")[0]
    row = ranking.loc[symbol] if symbol in ranking.index else None
    side = trade["side"]
    exposure = result.target_exposure
    execution = (f"{trade['trade_date']} 以 {trade['price']:.2f} 元{'买入' if side == 'BUY' else '卖出'} "
                 f"{int(trade['quantity'])} 股（成交额 {trade['amount']:,.0f} 元）。")
    facts: Dict[str, Any] = {"signal_date": signal_date.isoformat(), "scored": n_scored, "selection_size": k,
                             "timing_exposure": exposure, "timing_label": result.market_regime}
    head = f"信号日 {signal_date} 收盘后按「{strategy.name}」重新打分："
    if row is None or pd.isna(row["score"]):
        body = f"{name} 当天没有分数（停牌、上市不满所需天数或不在股票池内），不会被选入。"
    else:
        rank, score, weight = int(row["rank"]), float(row["score"]), float(row["target_weight"])
        facts.update(rank=rank, score=score, target_weight=weight, cutoff=cutoff)
        body = f"{name} 在 {n_scored} 支有分数的股票里排第 {rank}，分数 {_fmt_score(score, spec.scorer.type)}"
        if cutoff is not None:
            body += f"；这次选前 {k} 名，入选门槛分数 {_fmt_score(cutoff, spec.scorer.type)}"
        if weight > 0:
            body += f"，它入选了，目标权重 {weight:.1%}。"
        else:
            body += "，它没有入选，目标权重 0。"
    if spec.timing.type != "none":
        body += f"当天择时信号给的整体仓位是 {exposure:.0%}。"
    if side == "SELL" and row is not None and not pd.isna(row["score"]) and float(row["target_weight"]) > 0:
        body += "卖出是因为目标权重比当时的持仓比例低（换手阈值以内的差额不交易），不是清仓。"
    text = head + body + execution
    if trade.get("reason"):
        text += f"成交记录上的原因：{trade['reason']}。"
    return {"text": text, "facts": facts}
