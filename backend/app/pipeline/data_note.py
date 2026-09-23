"""数据说明（ADR-0053）：一个策略在一段回测里用了哪几年的数据。

作业要求"策略上写清楚综合了哪几年的数据"。这里按策略规格和回测区间生成几行说明，
策略详情页和回测报告共用，口径只写在这一处。
"""
from __future__ import annotations

from datetime import date
from typing import List

from app.data.market_store import STORE_START
from app.pipeline.bars import warmup_start
from app.pipeline.spec import StrategySpec

_INDEX_TIMINGS = {"rsrs", "icu_ma", "alligator", "llt", "ma_channel", "one_way_vol", "rps_vol", "high_moment",
                  "volume_resonance", "qrs"}


def _model_line(model_id: str, start: date, end: date) -> str:
    from app.pipeline.model_library import MODEL_LIBRARY
    from app.training.trainer import read_record

    if model_id in MODEL_LIBRARY:
        entry = MODEL_LIBRARY[model_id]
        return (f"{entry.name}：离线流程用 baostock 2010 年起的日线训练，训练样本取今天的成分股名单（有幸存者偏差）；"
                f"逐年滚动，给 {start.year}–{end.year} 年打分的模型分别只用到 {start.year - 1}–{end.year - 1} 年 6 月底。")
    record = read_record(model_id) or {}
    first = str(record.get("data_start") or "2016")[:4]
    pool = record.get("pool_label") or "训练股票池"
    return (f"{record.get('name', model_id)}：用本地行情库 {first} 年起的日线训练（{pool}，每天只取当天在池里的股票）；"
            f"逐年滚动，给 {start.year} 年打分的模型用 {first} 年 ~ {start.year - 1} 年 6 月底的样本，"
            f"给 {end.year} 年的用 {first} 年 ~ {end.year - 1} 年 6 月底的样本。")


def data_note(spec: StrategySpec, start: date, end: date, universe: str) -> List[str]:
    lines = [f"行情：本地行情库 A 股日线（tushare，{STORE_START.year} 年起，前复权）。回测区间 {start} ~ {end}。"]
    if spec.scorer.type == "model":
        lines += [_model_line(m, start, end) for m in spec.scorer.models]
    else:
        lines.append(f"打分：因子权重不需要训练，每个调仓日只用当天及之前约一年的日线（本次最早用到 {warmup_start(start, 300)}）。")
    if spec.timing.type in _INDEX_TIMINGS:
        lines.append("择时：用沪深300 指数日线（2005 年起），每天收盘判断，次日调整仓位。")
    elif spec.timing.type == "index_trend":
        lines.append("择时：用沪深300 指数的 50 日、200 日均线，每天收盘判断。")
    if universe == "csi300":
        lines.append(f"股票池：每个调仓日只用当时真实在沪深300 里的股票（历史成分股，{STORE_START.year} 年起）。")
    return lines
