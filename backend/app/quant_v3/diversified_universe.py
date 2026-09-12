"""调研新方向（docs/adr/0011）：ADR-0010 的结论是"股票池同质化、特征太窄"
可能才是限制点，不是算法机制本身。这里换一批跨行业、理论上相关性更低的
股票——不再局限于 V3 官方"红利/成长/周期"三组高相关大盘蓝筹，覆盖白酒、
家电、医药、保险、地产、银行、旅游零售、电子制造、光伏、工程机械、显示
面板、券商 12 个不同行业。研究性实验专用，不是 V3 官方研究池，不影响
`A_PHASE_STOCKS`。

不再用组内预算这套设计（没有理由假定这些不同行业该分成三组各占固定比例），
统一放进一个"综合"组，Top-K 直接在全部 12 支里选相对最好的一半。
"""

DIVERSIFIED_STOCKS = [
    {"symbol": "600519.SH", "name": "贵州茅台", "industry": "白酒", "group": "综合"},
    {"symbol": "000333.SZ", "name": "美的集团", "industry": "家电", "group": "综合"},
    {"symbol": "600276.SH", "name": "恒瑞医药", "industry": "医药", "group": "综合"},
    {"symbol": "601318.SH", "name": "中国平安", "industry": "保险", "group": "综合"},
    {"symbol": "000002.SZ", "name": "万科A", "industry": "地产", "group": "综合"},
    {"symbol": "600036.SH", "name": "招商银行", "industry": "银行", "group": "综合"},
    {"symbol": "601888.SH", "name": "中国中免", "industry": "旅游零售", "group": "综合"},
    {"symbol": "002475.SZ", "name": "立讯精密", "industry": "电子制造", "group": "综合"},
    {"symbol": "601012.SH", "name": "隆基绿能", "industry": "光伏", "group": "综合"},
    {"symbol": "600031.SH", "name": "三一重工", "industry": "工程机械", "group": "综合"},
    {"symbol": "000725.SZ", "name": "京东方A", "industry": "显示面板", "group": "综合"},
    {"symbol": "600030.SH", "name": "中信证券", "industry": "券商", "group": "综合"},
]

DIVERSIFIED_GROUP_BUDGETS = {"综合": 1.0}
