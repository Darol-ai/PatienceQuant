"""V3 方案 9.1 节：A 阶段固定研究池——10 支股票，仅限研究范围，
不是本版动态筛选的实际输出，未证明当前均满足所有资格（qualification_mode 由调用方决定）。
"""

A_PHASE_STOCKS = [
    {"symbol": "601288.SH", "ak_code": "601288", "name": "农业银行", "industry": "银行", "group": "红利"},
    {"symbol": "601398.SH", "ak_code": "601398", "name": "工商银行", "industry": "银行", "group": "红利"},
    {"symbol": "600900.SH", "ak_code": "600900", "name": "长江电力", "industry": "水电", "group": "红利"},
    {"symbol": "600377.SH", "ak_code": "600377", "name": "宁沪高速", "industry": "高速公路", "group": "红利"},
    {"symbol": "300308.SZ", "ak_code": "300308", "name": "中际旭创", "industry": "光通信", "group": "成长"},
    {"symbol": "300124.SZ", "ak_code": "300124", "name": "汇川技术", "industry": "工业自动化", "group": "成长"},
    {"symbol": "600406.SH", "ak_code": "600406", "name": "国电南瑞", "industry": "电网自动化", "group": "成长"},
    {"symbol": "601899.SH", "ak_code": "601899", "name": "紫金矿业", "industry": "金属资源", "group": "周期"},
    {"symbol": "601088.SH", "ak_code": "601088", "name": "中国神华", "industry": "煤炭与综合能源", "group": "周期"},
    {"symbol": "600309.SH", "ak_code": "600309", "name": "万华化学", "industry": "化工材料", "group": "周期"},
]

# 研究性实验专用，不是 V3 方案 9.1 节的官方 A 阶段研究池——用来验证"扩大
# 候选池是否能让组内 Top-K 排序选出更好的相对赢家"这个假设（见相关 ADR）。
# 每组在官方 4/3/3 基础上各加 3 支同组风格的流动性较好的大中盘股，分组
# 预算(GROUP_BUDGETS)和官方池保持一致语义，只是候选变多。
EXPANDED_A_PHASE_STOCKS = A_PHASE_STOCKS + [
    {"symbol": "601939.SH", "ak_code": "601939", "name": "建设银行", "industry": "银行", "group": "红利"},
    {"symbol": "601328.SH", "ak_code": "601328", "name": "交通银行", "industry": "银行", "group": "红利"},
    {"symbol": "601006.SH", "ak_code": "601006", "name": "大秦铁路", "industry": "铁路运输", "group": "红利"},
    {"symbol": "300750.SZ", "ak_code": "300750", "name": "宁德时代", "industry": "动力电池", "group": "成长"},
    {"symbol": "002415.SZ", "ak_code": "002415", "name": "海康威视", "industry": "安防电子", "group": "成长"},
    {"symbol": "300760.SZ", "ak_code": "300760", "name": "迈瑞医疗", "industry": "医疗器械", "group": "成长"},
    {"symbol": "601600.SH", "ak_code": "601600", "name": "中国铝业", "industry": "有色金属", "group": "周期"},
    {"symbol": "600019.SH", "ak_code": "600019", "name": "宝钢股份", "industry": "钢铁", "group": "周期"},
    {"symbol": "600028.SH", "ak_code": "600028", "name": "中国石化", "industry": "石油化工", "group": "周期"},
]
