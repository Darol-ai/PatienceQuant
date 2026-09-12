"""自由探索阶段（docs/adr/0013）：用户放开范围限制，目标是尽量多的年份
跑赢基准。ADR-0012 消融证明分散化股票池是目前最有效的杠杆，这里在
`diversified_universe.py` 12支的基础上再扩到 30 支，覆盖近30个不同行业，
不再局限于 V3 方案的三组结构，全部放进一个"综合"组。
"""
from app.quant_v3.a_phase_universe import A_PHASE_STOCKS
from app.quant_v3.diversified_universe import DIVERSIFIED_STOCKS

_EXTRA_STOCKS = [
    {"symbol": "002594.SZ", "name": "比亚迪", "industry": "新能源车", "group": "综合"},
    {"symbol": "000063.SZ", "name": "中兴通讯", "industry": "通信设备", "group": "综合"},
    {"symbol": "603288.SH", "name": "海天味业", "industry": "食品饮料", "group": "综合"},
    {"symbol": "601857.SH", "name": "中国石油", "industry": "石油天然气", "group": "综合"},
    {"symbol": "600585.SH", "name": "海螺水泥", "industry": "建材", "group": "综合"},
    {"symbol": "002714.SZ", "name": "牧原股份", "industry": "农业养殖", "group": "综合"},
    {"symbol": "600760.SH", "name": "中航沈飞", "industry": "军工", "group": "综合"},
    {"symbol": "002027.SZ", "name": "分众传媒", "industry": "传媒广告", "group": "综合"},
]

BROAD_STOCKS = [{**s, "group": "综合"} for s in A_PHASE_STOCKS] + DIVERSIFIED_STOCKS + _EXTRA_STOCKS
BROAD_GROUP_BUDGETS = {"综合": 1.0}
