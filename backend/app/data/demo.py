from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StockSpec:
    symbol: str
    name: str
    industry: str
    group: str
    sector: str
    tags: Tuple[str, ...]
    drift: float
    volatility: float
    quality: float
    value: float
    growth: float
    # The default universe is A-share.  OTC/Pink demo instruments use the
    # same factor and pricing pipeline but are clearly labelled as an
    # extension market in the catalog and UI.
    exchange: str = "A股"


GROUPS: Dict[str, Dict[str, object]] = {
    "消费": {"description": "品牌消费、食品饮料与家电龙头", "color": "#f5b94c"},
    "科技": {"description": "半导体、软件、通信和智能制造", "color": "#6c8cff"},
    "新能源": {"description": "新能源车、光伏、储能与电力设备", "color": "#31d0aa"},
    "金融": {"description": "银行、保险、券商与金融科技", "color": "#c084fc"},
    "红利/央国企": {"description": "高股息、能源、公用事业和央国企", "color": "#fb7185"},
}

BASE_STOCK_SPECS: List[StockSpec] = [
    StockSpec("600519", "贵州茅台", "白酒", "消费", "消费", ("品牌", "高ROE"), .12, .21, .90, .62, .72),
    StockSpec("000858", "五粮液", "白酒", "消费", "消费", ("品牌", "现金流"), .105, .23, .86, .65, .68),
    StockSpec("600887", "伊利股份", "食品饮料", "消费", "消费", ("必选消费", "股息"), .075, .20, .82, .66, .60),
    StockSpec("000333", "美的集团", "家电", "消费", "消费", ("制造升级", "现金流"), .11, .24, .84, .68, .73),
    StockSpec("000651", "格力电器", "家电", "消费", "消费", ("高股息", "龙头"), .07, .25, .82, .72, .52),
    StockSpec("603288", "海天味业", "食品饮料", "消费", "消费", ("调味品", "品牌"), .08, .22, .79, .64, .62),
    StockSpec("002304", "洋河股份", "白酒", "消费", "消费", ("白酒", "估值"), .06, .27, .71, .68, .48),
    StockSpec("600600", "青岛啤酒", "食品饮料", "消费", "消费", ("啤酒", "消费升级"), .085, .24, .77, .62, .65),
    StockSpec("002714", "牧原股份", "农林牧渔", "消费", "消费", ("养殖", "周期"), .09, .38, .58, .52, .76),
    StockSpec("000568", "泸州老窖", "白酒", "消费", "消费", ("品牌", "成长"), .11, .26, .83, .60, .74),
    StockSpec("688981", "中芯国际", "半导体", "科技", "科技", ("芯片", "国产替代"), .14, .38, .62, .38, .92),
    StockSpec("002415", "海康威视", "电子", "科技", "科技", ("AI视觉", "现金流"), .105, .29, .80, .55, .72),
    StockSpec("000063", "中兴通讯", "通信", "科技", "科技", ("通信", "算力"), .12, .34, .70, .45, .82),
    StockSpec("601138", "工业富联", "电子", "科技", "科技", ("服务器", "制造"), .13, .31, .76, .49, .86),
    StockSpec("688111", "金山办公", "软件", "科技", "科技", ("办公软件", "信创"), .14, .39, .73, .39, .91),
    StockSpec("688012", "中微公司", "半导体", "科技", "科技", ("设备", "高成长"), .155, .42, .68, .35, .97),
    StockSpec("002230", "科大讯飞", "软件", "科技", "科技", ("AI", "软件"), .13, .45, .60, .40, .88),
    StockSpec("300308", "中际旭创", "通信", "科技", "科技", ("光模块", "算力"), .18, .50, .63, .34, .99),
    StockSpec("600570", "恒生电子", "软件", "科技", "科技", ("金融IT", "软件"), .10, .35, .70, .48, .75),
    StockSpec("002371", "北方华创", "半导体", "科技", "科技", ("半导体设备", "国产替代"), .155, .44, .71, .39, .94),
    StockSpec("300750", "宁德时代", "新能源车", "新能源", "新能源", ("电池", "龙头"), .16, .40, .72, .43, .95),
    StockSpec("601012", "隆基绿能", "光伏设备", "新能源", "新能源", ("光伏", "制造"), .12, .43, .64, .46, .90),
    StockSpec("600438", "通威股份", "光伏设备", "新能源", "新能源", ("硅料", "周期"), .10, .45, .61, .50, .78),
    StockSpec("002594", "比亚迪", "汽车", "新能源", "新能源", ("新能源车", "出海"), .155, .35, .77, .45, .92),
    StockSpec("300014", "亿纬锂能", "电力设备", "新能源", "新能源", ("电池", "储能"), .14, .41, .65, .44, .91),
    StockSpec("002460", "赣锋锂业", "有色金属", "新能源", "新能源", ("锂资源", "周期"), .09, .53, .55, .42, .82),
    StockSpec("601865", "福莱特", "光伏设备", "新能源", "新能源", ("光伏玻璃", "制造"), .105, .41, .65, .48, .79),
    StockSpec("300274", "阳光电源", "电力设备", "新能源", "新能源", ("逆变器", "储能"), .14, .44, .75, .43, .90),
    StockSpec("600089", "特变电工", "电力设备", "新能源", "新能源", ("电网", "高股息"), .08, .31, .78, .60, .65),
    StockSpec("601985", "中国核电", "电力", "新能源", "新能源", ("核电", "央企"), .075, .25, .83, .67, .58),
    StockSpec("600036", "招商银行", "银行", "金融", "金融", ("零售银行", "高ROE"), .095, .23, .86, .66, .61),
    StockSpec("601318", "中国平安", "保险", "金融", "金融", ("保险", "综合金融"), .085, .27, .74, .60, .58),
    StockSpec("000001", "平安银行", "银行", "金融", "金融", ("银行", "零售"), .065, .29, .70, .62, .54),
    StockSpec("601166", "兴业银行", "银行", "金融", "金融", ("银行", "高股息"), .072, .25, .78, .70, .49),
    StockSpec("600030", "中信证券", "券商", "金融", "金融", ("券商", "资本市场"), .11, .36, .63, .55, .66),
    StockSpec("601688", "华泰证券", "券商", "金融", "金融", ("券商", "财富管理"), .105, .34, .65, .58, .65),
    StockSpec("601601", "中国太保", "保险", "金融", "金融", ("保险", "股息"), .08, .28, .79, .63, .55),
    StockSpec("600919", "江苏银行", "银行", "金融", "金融", ("银行", "成长"), .10, .28, .74, .65, .70),
    StockSpec("601398", "工商银行", "银行", "金融", "金融", ("四大行", "高股息"), .06, .20, .90, .79, .42),
    StockSpec("601818", "光大银行", "银行", "金融", "金融", ("银行", "估值"), .058, .27, .68, .71, .40),
    StockSpec("601857", "中国石油", "石油石化", "红利/央国企", "红利", ("能源", "央企"), .075, .28, .84, .72, .52),
    StockSpec("600028", "中国石化", "石油石化", "红利/央国企", "红利", ("能源", "高股息"), .068, .24, .86, .75, .46),
    StockSpec("601088", "中国神华", "煤炭", "红利/央国企", "红利", ("煤炭", "高股息"), .09, .27, .92, .78, .55),
    StockSpec("600900", "长江电力", "电力", "红利/央国企", "红利", ("水电", "现金流"), .075, .19, .94, .73, .48),
    StockSpec("601006", "大秦铁路", "交通运输", "红利/央国企", "红利", ("铁路", "高股息"), .06, .18, .88, .79, .40),
    StockSpec("601668", "中国建筑", "建筑装饰", "红利/央国企", "红利", ("基建", "央企"), .07, .25, .80, .69, .50),
    StockSpec("601390", "中国中铁", "建筑装饰", "红利/央国企", "红利", ("基建", "央企"), .068, .26, .78, .68, .50),
    StockSpec("600025", "华能水电", "电力", "红利/央国企", "红利", ("水电", "成长"), .082, .23, .86, .69, .62),
    StockSpec("601919", "中远海控", "交通运输", "红利/央国企", "红利", ("航运", "周期"), .11, .48, .62, .61, .74),
    StockSpec("600674", "川投能源", "电力", "红利/央国企", "红利", ("水电", "高股息"), .075, .20, .91, .75, .50),
]

# A small, clearly labelled OTC/Pink Sheets extension basket.  These are
# offline demo instruments rather than claims about live OTC listings.  They
# intentionally reuse the same five research groups so every group remains
# comparable in the factor engine while the market is visible in the UI.
PINK_SPECS: List[StockSpec] = [
    StockSpec("PINK001", "Pink Sheets示例·能源", "能源服务", "红利/央国企", "Pink Sheets", ("OTC", "高股息", "Demo"), .07, .36, .72, .68, .48, "OTC/Pink Sheets"),
    StockSpec("PINK002", "Pink Sheets示例·医疗", "医疗服务", "消费", "Pink Sheets", ("OTC", "医疗", "Demo"), .085, .43, .67, .52, .72, "OTC/Pink Sheets"),
    StockSpec("PINK003", "Pink Sheets示例·软件", "软件服务", "科技", "Pink Sheets", ("OTC", "软件", "Demo"), .11, .48, .63, .45, .84, "OTC/Pink Sheets"),
    StockSpec("PINK004", "Pink Sheets示例·半导体", "半导体", "科技", "Pink Sheets", ("OTC", "芯片", "Demo"), .13, .55, .60, .39, .91, "OTC/Pink Sheets"),
    StockSpec("PINK005", "Pink Sheets示例·新能源", "新能源设备", "新能源", "Pink Sheets", ("OTC", "清洁能源", "Demo"), .12, .50, .65, .44, .90, "OTC/Pink Sheets"),
    StockSpec("PINK006", "Pink Sheets示例·电池", "电池材料", "新能源", "Pink Sheets", ("OTC", "储能", "Demo"), .115, .53, .62, .46, .88, "OTC/Pink Sheets"),
    StockSpec("PINK007", "Pink Sheets示例·消费", "消费品", "消费", "Pink Sheets", ("OTC", "消费", "Demo"), .075, .34, .75, .61, .58, "OTC/Pink Sheets"),
    StockSpec("PINK008", "Pink Sheets示例·食品", "食品饮料", "消费", "Pink Sheets", ("OTC", "食品", "Demo"), .07, .31, .78, .64, .53, "OTC/Pink Sheets"),
    StockSpec("PINK009", "Pink Sheets示例·银行", "银行服务", "金融", "Pink Sheets", ("OTC", "金融", "Demo"), .065, .29, .79, .70, .44, "OTC/Pink Sheets"),
    StockSpec("PINK010", "Pink Sheets示例·保险", "保险服务", "金融", "Pink Sheets", ("OTC", "保险", "Demo"), .07, .32, .74, .64, .50, "OTC/Pink Sheets"),
    StockSpec("PINK011", "Pink Sheets示例·通信", "通信设备", "科技", "Pink Sheets", ("OTC", "通信", "Demo"), .10, .45, .66, .47, .82, "OTC/Pink Sheets"),
    StockSpec("PINK012", "Pink Sheets示例·云计算", "云计算", "科技", "Pink Sheets", ("OTC", "云服务", "Demo"), .12, .49, .64, .43, .89, "OTC/Pink Sheets"),
    StockSpec("PINK013", "Pink Sheets示例·光伏", "光伏设备", "新能源", "Pink Sheets", ("OTC", "光伏", "Demo"), .105, .52, .63, .48, .86, "OTC/Pink Sheets"),
    StockSpec("PINK014", "Pink Sheets示例·公用事业", "公用事业", "红利/央国企", "Pink Sheets", ("OTC", "公用事业", "Demo"), .06, .27, .84, .74, .39, "OTC/Pink Sheets"),
    StockSpec("PINK015", "Pink Sheets示例·矿业", "矿业", "红利/央国企", "Pink Sheets", ("OTC", "资源", "Demo"), .08, .46, .68, .65, .57, "OTC/Pink Sheets"),
    StockSpec("PINK016", "Pink Sheets示例·物流", "物流服务", "红利/央国企", "Pink Sheets", ("OTC", "物流", "Demo"), .072, .36, .76, .63, .52, "OTC/Pink Sheets"),
    StockSpec("PINK017", "Pink Sheets示例·AI", "人工智能", "科技", "Pink Sheets", ("OTC", "AI", "Demo"), .14, .58, .57, .36, .98, "OTC/Pink Sheets"),
    StockSpec("PINK018", "Pink Sheets示例·机器人", "机器人", "科技", "Pink Sheets", ("OTC", "机器人", "Demo"), .125, .51, .65, .42, .93, "OTC/Pink Sheets"),
    StockSpec("PINK019", "Pink Sheets示例·医药", "生物医药", "消费", "Pink Sheets", ("OTC", "创新药", "Demo"), .10, .47, .69, .43, .88, "OTC/Pink Sheets"),
    StockSpec("PINK020", "Pink Sheets示例·基础设施", "基础设施", "红利/央国企", "Pink Sheets", ("OTC", "基础设施", "Demo"), .065, .30, .81, .71, .42, "OTC/Pink Sheets"),
]


_EXTRA_INDUSTRIES = [
    ("科技", "计算机设备", "科技", "服务器与终端", .115, .34, .69, .46, .79),
    ("科技", "通信设备", "科技", "通信基础设施", .105, .32, .73, .50, .76),
    ("科技", "消费电子", "科技", "智能终端", .12, .36, .70, .47, .84),
    ("科技", "元件", "科技", "电子元件", .11, .33, .72, .51, .80),
    ("科技", "光学光电子", "科技", "显示与光学", .10, .38, .65, .49, .78),
    ("科技", "软件开发", "科技", "企业软件", .12, .35, .68, .43, .86),
    ("科技", "互联网服务", "科技", "平台经济", .095, .42, .62, .50, .75),
    ("科技", "IT服务", "科技", "数字化服务", .10, .31, .71, .56, .73),
    ("新能源", "电池", "新能源", "动力电池", .13, .42, .68, .45, .90),
    ("新能源", "风电设备", "新能源", "风电整机", .11, .39, .70, .52, .80),
    ("新能源", "电网设备", "新能源", "智能电网", .10, .30, .80, .58, .72),
    ("新能源", "电机", "新能源", "电驱系统", .11, .35, .74, .53, .81),
    ("新能源", "汽车零部件", "新能源", "汽车智能化", .105, .33, .73, .50, .84),
    ("新能源", "能源金属", "新能源", "关键金属", .09, .51, .58, .44, .78),
    ("消费", "旅游酒店", "消费", "文旅消费", .10, .34, .66, .55, .73),
    ("消费", "商业百货", "消费", "零售消费", .075, .30, .70, .62, .55),
    ("消费", "服装家纺", "消费", "品牌服饰", .08, .28, .73, .60, .61),
    ("消费", "美容护理", "消费", "美妆个护", .09, .31, .74, .54, .73),
    ("消费", "传媒", "消费", "内容与营销", .085, .42, .58, .48, .79),
    ("消费", "食品加工", "消费", "大众食品", .075, .22, .79, .66, .58),
    ("金融", "多元金融", "金融", "金融服务", .09, .35, .63, .58, .65),
    ("金融", "银行设备", "金融", "金融基础设施", .085, .25, .76, .61, .55),
    ("金融", "金融科技", "金融", "数字金融", .10, .38, .64, .49, .76),
    ("红利/央国企", "公用事业", "红利", "公共服务", .068, .18, .88, .76, .44),
    ("红利/央国企", "钢铁", "红利", "周期制造", .062, .28, .76, .67, .48),
    ("红利/央国企", "有色金属", "红利", "资源品", .075, .43, .66, .56, .68),
    ("红利/央国企", "建筑材料", "红利", "基建材料", .07, .27, .77, .65, .53),
    ("红利/央国企", "物流", "红利", "供应链物流", .08, .31, .74, .60, .65),
    ("消费", "医药商业", "医药", "医药流通", .085, .29, .75, .57, .64),
    ("消费", "化学制药", "医药", "创新药", .10, .38, .68, .47, .83),
    ("消费", "医疗器械", "医药", "医疗设备", .11, .36, .75, .48, .87),
    ("消费", "中药", "医药", "中药品牌", .08, .27, .80, .61, .60),
    ("科技", "航空装备", "军工", "航空航天", .105, .41, .67, .47, .85),
    ("科技", "航天装备", "军工", "卫星互联网", .11, .45, .64, .42, .90),
    ("科技", "地面兵装", "军工", "高端装备", .09, .36, .70, .53, .78),
    ("红利/央国企", "化学原料", "化工", "基础化工", .085, .36, .72, .58, .72),
    ("红利/央国企", "化学制品", "化工", "新材料", .095, .39, .69, .51, .82),
    ("新能源", "塑料制品", "化工", "高分子材料", .09, .35, .69, .52, .78),
    ("科技", "通用设备", "机械", "工业自动化", .10, .29, .80, .58, .75),
    ("科技", "专用设备", "机械", "高端装备", .115, .36, .75, .49, .84),
    ("科技", "工程机械", "机械", "基建装备", .09, .32, .78, .62, .69),
    ("新能源", "自动化设备", "机械", "智能制造", .12, .33, .76, .48, .87),
    ("新能源", "汽车整车", "新能源", "汽车制造", .115, .37, .78, .47, .91),
    ("消费", "汽车服务", "消费", "汽车后市场", .075, .28, .70, .59, .59),
    ("消费", "家用轻工", "消费", "家居生活", .078, .25, .76, .63, .57),
    ("红利/央国企", "房地产开发", "红利", "城市运营", .045, .35, .57, .65, .36),
    ("红利/央国企", "房地产服务", "红利", "物业服务", .06, .31, .62, .60, .48),
    ("消费", "农业综合", "消费", "现代农业", .065, .36, .60, .56, .68),
    ("消费", "种植业", "消费", "粮食安全", .058, .31, .69, .65, .50),
    ("消费", "养殖业", "消费", "生猪养殖", .07, .46, .56, .52, .74),
    ("红利/央国企", "航运港口", "红利", "港口航运", .075, .42, .73, .63, .61),
    ("红利/央国企", "铁路公路", "红利", "交通基础设施", .065, .21, .86, .75, .45),
    ("红利/央国企", "航空机场", "红利", "航空运输", .072, .35, .70, .62, .58),
]

_INDUSTRY_EXPANSION = [
    # Advanced technology and digital infrastructure
    ("科技", "集成电路", "科技", "芯片设计", .135, .41, .66, .40, .93),
    ("科技", "半导体材料", "科技", "电子材料", .12, .37, .70, .45, .87),
    ("科技", "半导体设备", "科技", "晶圆设备", .14, .43, .68, .39, .94),
    ("科技", "电子化学品", "科技", "电子化学材料", .115, .35, .71, .47, .85),
    ("科技", "印制电路板", "科技", "PCB制造", .11, .34, .74, .50, .83),
    ("科技", "LED", "科技", "显示照明", .09, .39, .64, .54, .76),
    ("科技", "计算机硬件", "科技", "终端设备", .10, .33, .73, .53, .78),
    ("科技", "云计算", "科技", "云基础设施", .13, .37, .66, .42, .90),
    ("科技", "数据中心", "科技", "算力基础设施", .125, .35, .70, .46, .88),
    ("科技", "网络安全", "科技", "安全软件", .115, .36, .72, .45, .84),
    ("科技", "人工智能", "科技", "AI应用", .15, .48, .61, .38, .98),
    ("科技", "大数据", "科技", "数据服务", .12, .39, .65, .44, .89),
    ("科技", "数字媒体", "消费", "数字内容", .10, .42, .59, .51, .83),
    ("科技", "游戏", "消费", "游戏娱乐", .095, .45, .57, .49, .86),
    ("科技", "出版", "消费", "出版传媒", .07, .29, .73, .64, .55),
    ("科技", "广播电视", "消费", "视听服务", .065, .31, .69, .58, .52),
    ("科技", "广告营销", "消费", "营销服务", .09, .34, .67, .54, .72),
    ("科技", "卫星互联网", "科技", "航天通信", .13, .47, .62, .41, .95),
    ("科技", "无人机", "科技", "智能飞行器", .14, .43, .66, .43, .93),
    ("科技", "激光设备", "科技", "激光加工", .115, .38, .73, .50, .86),
    # Industrial, materials and manufacturing
    ("科技", "工业机器人", "科技", "机器人", .13, .39, .75, .45, .90),
    ("科技", "工业软件", "科技", "工业数字化", .12, .34, .76, .48, .86),
    ("科技", "仪器仪表", "科技", "工业测量", .10, .31, .78, .58, .76),
    ("科技", "机床工具", "科技", "高端机床", .10, .36, .72, .55, .79),
    ("科技", "轨道交通设备", "红利/央国企", "轨道装备", .085, .27, .82, .65, .62),
    ("科技", "船舶制造", "红利/央国企", "海洋装备", .10, .38, .70, .57, .75),
    ("科技", "航空发动机", "科技", "航空动力", .12, .42, .67, .45, .91),
    ("红利/央国企", "水泥", "红利", "建材制造", .055, .25, .79, .73, .40),
    ("红利/央国企", "玻璃制造", "红利", "玻璃材料", .065, .34, .70, .62, .58),
    ("红利/央国企", "装配式建筑", "红利", "建筑工业化", .08, .31, .72, .61, .65),
    ("红利/央国企", "工程咨询", "红利", "工程服务", .07, .24, .80, .70, .48),
    ("红利/央国企", "环保设备", "红利", "环保工程", .075, .32, .73, .59, .68),
    ("红利/央国企", "水务", "红利", "水资源运营", .065, .18, .88, .78, .43),
    ("红利/央国企", "燃气", "红利", "城市燃气", .068, .19, .86, .77, .46),
    ("红利/央国企", "火力发电", "红利", "火电运营", .06, .22, .83, .75, .42),
    ("红利/央国企", "热力服务", "红利", "热力供应", .058, .20, .84, .76, .40),
    ("红利/央国企", "稀土", "红利", "稀土资源", .085, .47, .64, .58, .73),
    ("红利/央国企", "黄金", "红利", "贵金属", .095, .35, .70, .61, .69),
    ("红利/央国企", "铜业", "红利", "工业金属", .08, .41, .68, .59, .70),
    ("红利/央国企", "铝业", "红利", "有色冶炼", .075, .38, .70, .62, .65),
    ("红利/央国企", "焦炭", "红利", "煤化工", .055, .42, .65, .71, .52),
    ("红利/央国企", "油服工程", "红利", "能源服务", .078, .33, .75, .67, .59),
    ("红利/央国企", "炼化技术", "红利", "石化工程", .07, .31, .77, .68, .55),
    ("红利/央国企", "新型建材", "红利", "绿色建材", .085, .29, .76, .60, .66),
    ("红利/央国企", "木材加工", "消费", "家居材料", .065, .28, .72, .66, .54),
    # Energy transition, mobility and utilities
    ("新能源", "光伏辅材", "新能源", "光伏材料", .12, .43, .65, .46, .88),
    ("新能源", "光伏电池", "新能源", "光伏制造", .13, .45, .66, .44, .92),
    ("新能源", "储能系统", "新能源", "储能集成", .14, .44, .68, .42, .95),
    ("新能源", "充电桩", "新能源", "充换电", .115, .41, .69, .48, .88),
    ("新能源", "氢能", "新能源", "氢能产业", .12, .46, .61, .43, .94),
    ("新能源", "核电设备", "新能源", "核电装备", .095, .30, .82, .63, .70),
    ("新能源", "电力自动化", "新能源", "电力控制", .105, .29, .80, .57, .76),
    ("新能源", "电力电子", "新能源", "功率器件", .125, .37, .72, .47, .89),
    ("新能源", "电线电缆", "新能源", "输配电材料", .09, .27, .78, .64, .65),
    ("新能源", "燃料电池", "新能源", "氢能电池", .13, .49, .60, .41, .96),
    ("新能源", "摩托车", "新能源", "两轮出行", .09, .30, .76, .58, .72),
    ("新能源", "商用车", "新能源", "商用交通", .085, .36, .74, .57, .74),
    ("新能源", "智能座舱", "新能源", "汽车电子", .13, .38, .70, .45, .90),
    ("新能源", "轮胎", "新能源", "汽车材料", .085, .28, .78, .64, .64),
    ("新能源", "汽车内饰", "新能源", "汽车零部件", .09, .31, .73, .57, .76),
    # Consumer, healthcare and agriculture
    ("消费", "白色家电", "消费", "家电制造", .085, .22, .82, .68, .59),
    ("消费", "黑色家电", "消费", "影音设备", .075, .29, .72, .60, .64),
    ("消费", "小家电", "消费", "生活电器", .08, .28, .75, .59, .66),
    ("消费", "珠宝首饰", "消费", "可选消费", .09, .34, .68, .52, .69),
    ("消费", "乳制品", "消费", "乳业", .075, .22, .80, .66, .55),
    ("消费", "肉制品", "消费", "食品加工", .07, .25, .77, .62, .55),
    ("消费", "调味发酵品", "消费", "调味食品", .08, .23, .80, .64, .60),
    ("消费", "休闲食品", "消费", "零食饮品", .085, .31, .73, .56, .72),
    ("消费", "饮料乳品", "消费", "饮料", .09, .27, .76, .58, .70),
    ("消费", "家居用品", "消费", "家居消费", .08, .27, .75, .61, .63),
    ("消费", "纺织制造", "消费", "纺织品", .072, .26, .75, .63, .55),
    ("消费", "鞋帽服饰", "消费", "服装消费", .08, .29, .73, .60, .64),
    ("消费", "教育服务", "消费", "教育产业", .078, .33, .64, .51, .74),
    ("消费", "社会服务", "消费", "生活服务", .08, .34, .67, .55, .70),
    ("消费", "连锁零售", "消费", "零售连锁", .075, .26, .74, .62, .61),
    ("消费", "机场服务", "红利/央国企", "机场运营", .065, .25, .80, .72, .49),
    ("消费", "酒店餐饮", "消费", "酒店餐饮", .09, .36, .64, .50, .76),
    ("消费", "医疗服务", "消费", "医疗服务", .095, .32, .73, .49, .80),
    ("消费", "生物制品", "消费", "生物医药", .11, .41, .68, .44, .90),
    ("消费", "医疗研发", "消费", "医药研发", .115, .38, .69, .42, .91),
    ("消费", "医疗耗材", "消费", "医疗耗材", .10, .33, .75, .50, .84),
    ("消费", "医美服务", "消费", "医美消费", .105, .39, .67, .46, .88),
    ("消费", "动物保健", "消费", "动保产业", .10, .35, .72, .52, .82),
    ("消费", "饲料", "消费", "农业饲料", .08, .29, .76, .60, .69),
    ("消费", "农产品加工", "消费", "农产品", .07, .30, .72, .62, .60),
    ("消费", "林业", "消费", "林业资源", .06, .27, .70, .68, .48),
    ("消费", "渔业", "消费", "水产养殖", .068, .38, .60, .54, .68),
    # Financial and property services
    ("金融", "城商行", "金融", "城市银行", .075, .24, .79, .70, .50),
    ("金融", "农商行", "金融", "农村金融", .07, .26, .74, .72, .46),
    ("金融", "金融租赁", "金融", "融资租赁", .082, .30, .69, .63, .59),
    ("金融", "消费金融", "金融", "消费信贷", .09, .37, .63, .55, .73),
    ("金融", "保险服务", "金融", "保险经纪", .08, .29, .76, .65, .55),
    ("金融", "资产管理", "金融", "资管服务", .095, .32, .71, .56, .68),
    ("金融", "证券IT", "金融", "证券科技", .11, .39, .65, .49, .84),
    ("金融", "期货", "金融", "期货服务", .10, .42, .60, .52, .78),
    ("红利/央国企", "物业管理", "红利", "物业运营", .065, .28, .72, .61, .53),
    ("红利/央国企", "园区开发", "红利", "产业园区", .06, .31, .67, .66, .48),
    ("红利/央国企", "城市更新", "红利", "城市建设", .062, .33, .64, .65, .52),
]

_EXTRA_INDUSTRIES.extend(_INDUSTRY_EXPANSION)

_NAME_PREFIXES = ["华星", "启明", "远航", "新城", "中科", "天元", "恒信", "瑞泽", "盛达", "智联", "安泰", "嘉和", "宏远", "鼎新", "星河", "联创", "金瑞", "科源", "云启", "龙腾", "华辰", "睿智", "晨曦", "国泰", "东岳", "丰泽", "海纳", "汇通", "泰和", "优创", "明德", "博远", "瑞丰", "新锐", "锦程", "天启", "聚能", "恒达", "科信", "云图", "德润", "鸿远", "华誉", "卓越", "智造", "新航", "盛景", "安联", "嘉诚", "宏图", "鼎盛", "星瀚", "联达", "金桥", "科达", "云峰", "龙脉", "华源", "启航", "远景", "中锐", "天工", "恒泰", "瑞华", "盛丰", "智汇", "安澜", "嘉盛", "宏信", "鼎益", "星联", "联科", "金帆", "科泰", "云程", "龙腾", "华岳", "启辰", "远信"]
_NAME_SUFFIXES = ["科技", "股份", "智能", "集团", "新材", "装备", "电子", "发展", "信息", "能源", "产业", "制造", "医疗", "控股", "材料"]

DEFAULT_UNIVERSE_SIZE = 1000


def _build_extra_specs(target_count: int) -> List[StockSpec]:
    """Build a broad, deterministic Demo A-share universe without copying real vendor data."""
    existing = {spec.symbol for spec in BASE_STOCK_SPECS}
    existing_names = {spec.name for spec in BASE_STOCK_SPECS}
    specs: List[StockSpec] = []
    index = 0
    prefixes = ["000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688", "689"]
    while len(specs) < max(0, target_count - len(BASE_STOCK_SPECS)):
        group, industry, sector, tag, drift, volatility, quality, value, growth = _EXTRA_INDUSTRIES[index % len(_EXTRA_INDUSTRIES)]
        candidate = f"{prefixes[index % len(prefixes)]}{100 + index // len(prefixes):03d}"
        if candidate in existing:
            index += 1
            continue
        prefix = _NAME_PREFIXES[index % len(_NAME_PREFIXES)]
        suffix = _NAME_SUFFIXES[(index // len(_NAME_PREFIXES)) % len(_NAME_SUFFIXES)]
        cycle = index // (len(_NAME_PREFIXES) * len(_NAME_SUFFIXES))
        name = "%s%s%s" % (prefix, suffix, cycle + 1 if cycle else "")
        if name in existing_names:
            name = "%s%s%04d" % (prefix, suffix, index + 1)
        specs.append(StockSpec(candidate, name, industry, group, sector, (tag, "Demo扩展"),
                               drift + (index % 5 - 2) * .004, volatility + (index % 7 - 3) * .008,
                               min(.96, quality + (index % 6 - 2) * .012),
                               min(.92, value + (index % 5 - 2) * .018),
                               min(.99, growth + (index % 7 - 3) * .018)))
        existing.add(candidate)
        existing_names.add(name)
        index += 1
    return specs


def stock_specs_for_size(target_count: int = DEFAULT_UNIVERSE_SIZE) -> List[StockSpec]:
    """Return a deterministic catalog sized for the current demo environment."""
    # Keep the requested A-share count stable and append a separate OTC/Pink
    # extension basket.  This makes the standard A-share universes remain
    # 1,000+ while giving users a selectable Pink Sheets demo market.
    return BASE_STOCK_SPECS + _build_extra_specs(target_count) + PINK_SPECS


STOCK_SPECS: List[StockSpec] = stock_specs_for_size()


def trading_days(start: date, end: date) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="B")


def _stable_symbol_seed(symbol: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(str(symbol)))


class DemoDataProvider:
    mode = "demo"

    def __init__(self, seed: int = 20260909, as_of: Optional[date] = None, universe_size: int = DEFAULT_UNIVERSE_SIZE):
        self.seed = seed
        self.as_of = as_of or date.today()
        self.universe_size = max(len(BASE_STOCK_SPECS), int(universe_size))
        self.specs = stock_specs_for_size(self.universe_size)
        self._prices: Optional[pd.DataFrame] = None
        self._fundamentals: Optional[pd.DataFrame] = None

    def stock_catalog(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": s.symbol,
                    "name": s.name,
                    "industry": s.industry,
                    "group": s.group,
                    "sector": s.sector,
                    "tags": list(s.tags),
                    "exchange": s.exchange,
                }
                for s in self.specs
            ]
        )

    def generate_prices(self) -> pd.DataFrame:
        if self._prices is not None:
            return self._prices
        days = trading_days(date(2018, 1, 1), self.as_of)
        rng = np.random.default_rng(self.seed)
        market = rng.normal(0.00022, 0.012, len(days))
        symbols: List[str] = []
        dates: List[object] = []
        opens: List[np.ndarray] = []
        highs: List[np.ndarray] = []
        lows: List[np.ndarray] = []
        closes: List[np.ndarray] = []
        volumes: List[np.ndarray] = []
        for index, spec in enumerate(self.specs):
            local_rng = np.random.default_rng(self.seed + index * 101)
            cyclical = 0.00035 * np.sin(np.arange(len(days)) / (85 + index % 11))
            idio = local_rng.normal(0, spec.volatility / np.sqrt(252), len(days))
            ret = spec.drift / 252 + 0.45 * market + cyclical + idio
            ret = np.clip(ret, -0.18, 0.18)
            # Keep generated prices in a realistic, tradable range even for
            # the 1,000+ symbol catalog.  The paper broker uses 100-share
            # lots, so a monotonically increasing base price would make the
            # last symbols impossible to trade.
            close = (8 + (index % 180) * 2.4) * np.exp(np.cumsum(ret))
            close = np.maximum(close, 1.5)
            noise = np.abs(local_rng.normal(0, 0.012, len(days)))
            open_price = close * (1 + local_rng.normal(0, 0.006, len(days)))
            high = np.maximum(open_price, close) * (1 + noise)
            low = np.minimum(open_price, close) * (1 - noise)
            volume = local_rng.lognormal(mean=14.3 + (index % 4) * 0.16, sigma=0.35, size=len(days))
            symbols.extend([spec.symbol] * len(days))
            dates.extend(days.date)
            opens.append(np.round(open_price, 3).astype("float32"))
            highs.append(np.round(high, 3).astype("float32"))
            lows.append(np.round(low, 3).astype("float32"))
            closes.append(np.round(close, 3).astype("float32"))
            volumes.append(np.round(volume, 0).astype("float32"))
        close_array = np.concatenate(closes)
        self._prices = pd.DataFrame(
            {
                "symbol": symbols,
                "trade_date": dates,
                "open": np.concatenate(opens),
                "high": np.concatenate(highs),
                "low": np.concatenate(lows),
                "close": close_array,
                "adj_close": close_array.copy(),
                "volume": np.concatenate(volumes),
                "data_mode": "demo",
            }
        )
        return self._prices

    def generate_fundamentals(self) -> pd.DataFrame:
        if self._fundamentals is not None:
            return self._fundamentals
        rng = np.random.default_rng(self.seed + 7)
        years = list(range(2017, self.as_of.year + 1))
        rows = []
        for index, spec in enumerate(self.specs):
            local_rng = np.random.default_rng(self.seed + index * 17 + 3)
            for year in years:
                maturity = (year - 2017) / max(self.as_of.year - 2017, 1)
                noise = lambda scale: float(local_rng.normal(0, scale))
                rows.append(
                    {
                        "symbol": spec.symbol,
                        "report_date": date(year, 12, 31),
                        "roe": max(0.02, spec.quality * 0.16 + noise(.012)),
                        "roa": max(0.01, spec.quality * 0.075 + noise(.008)),
                        "revenue_growth": spec.growth * 0.23 + maturity * 0.015 + noise(.035),
                        "profit_growth": spec.growth * 0.28 + maturity * 0.02 + noise(.05),
                        "operating_cashflow": max(0.02, spec.quality * 0.18 + noise(.025)),
                        "pe": max(5, 42 - spec.value * 24 + noise(2.5) + spec.growth * 5),
                        "pb": max(.5, 5.4 - spec.value * 3.0 + noise(.3)),
                        "ps": max(.4, 10.5 - spec.value * 5.5 + noise(.45)),
                        "dividend_yield": max(.002, spec.value * .055 + noise(.008)),
                        "roe_stability": max(.1, spec.quality * .84 + noise(.05)),
                        "gross_margin": max(.08, .22 + spec.quality * .35 + noise(.025)),
                        "net_margin": max(.02, .06 + spec.quality * .18 + noise(.018)),
                        "cashflow_profit_ratio": max(.25, .75 + spec.quality * .3 + noise(.1)),
                        "data_mode": "demo",
                    }
                )
        self._fundamentals = pd.DataFrame(rows)
        return self._fundamentals

    def generate_benchmark(self) -> pd.DataFrame:
        days = trading_days(date(2018, 1, 1), self.as_of)
        rng = np.random.default_rng(self.seed + 99)
        ret = rng.normal(0.00018, 0.0105, len(days)) + .00025 * np.sin(np.arange(len(days)) / 130)
        close = 100 * np.exp(np.cumsum(ret))
        return pd.DataFrame({"trade_date": days.date, "close": close, "adj_close": close, "data_mode": "demo"})

    def bootstrap(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        catalog = self.stock_catalog()
        industries = catalog[["industry"]].drop_duplicates().rename(columns={"industry": "name"})
        groups = pd.DataFrame(
            [{"name": key, "description": value["description"], "color": value["color"]} for key, value in GROUPS.items()]
        )
        return catalog, self.generate_prices(), self.generate_fundamentals(), self.generate_benchmark()

    def fetch_prices(self, symbols: List[str], start: date, end: date) -> pd.DataFrame:
        df = self.generate_prices()
        known_symbols = {spec.symbol for spec in self.specs}
        known = [symbol for symbol in symbols if symbol in known_symbols]
        unknown = [symbol for symbol in symbols if symbol not in known_symbols]
        frames = []
        if known:
            frames.append(df[(df.symbol.isin(known)) & (df.trade_date >= start) & (df.trade_date <= end)].copy())
        if unknown:
            frames.append(self.synthetic_prices(unknown, start, end))
        if not frames:
            return pd.DataFrame(columns=df.columns)
        return pd.concat(frames, ignore_index=True).sort_values(["trade_date", "symbol"])

    def synthetic_prices(self, symbols: List[str], start: date, end: date) -> pd.DataFrame:
        """Deterministic demo fallback for AKShare-only symbols when prices fail."""
        all_days = trading_days(date(2018, 1, 1), end)
        if all_days.empty:
            return pd.DataFrame(columns=["symbol", "trade_date", "open", "high", "low", "close", "adj_close", "volume", "data_mode"])
        rows = []
        for symbol in symbols:
            seed = self.seed + _stable_symbol_seed(symbol) * 37
            local_rng = np.random.default_rng(seed)
            symbol_number = _stable_symbol_seed(symbol)
            drift = 0.055 + (symbol_number % 9) * 0.008
            volatility = 0.20 + (symbol_number % 11) * 0.018
            base_price = 9 + (symbol_number % 150) * 1.75
            market = local_rng.normal(0.00018, 0.010, len(all_days))
            idio = local_rng.normal(0, volatility / np.sqrt(252), len(all_days))
            cycle = 0.00028 * np.sin(np.arange(len(all_days)) / (70 + symbol_number % 23))
            close = np.maximum(1.2, base_price * np.exp(np.cumsum(drift / 252 + 0.35 * market + cycle + idio)))
            noise = np.abs(local_rng.normal(0, 0.011, len(all_days)))
            open_price = close * (1 + local_rng.normal(0, 0.006, len(all_days)))
            high = np.maximum(open_price, close) * (1 + noise)
            low = np.minimum(open_price, close) * (1 - noise)
            volume = local_rng.lognormal(mean=14.0 + (symbol_number % 4) * .12, sigma=.36, size=len(all_days))
            frame = pd.DataFrame(
                {
                    "symbol": symbol,
                    "trade_date": all_days.date,
                    "open": np.round(open_price, 3).astype("float32"),
                    "high": np.round(high, 3).astype("float32"),
                    "low": np.round(low, 3).astype("float32"),
                    "close": np.round(close, 3).astype("float32"),
                    "adj_close": np.round(close, 3).astype("float32"),
                    "volume": np.round(volume, 0).astype("float32"),
                    "data_mode": "demo_fallback",
                }
            )
            rows.append(frame[(frame.trade_date >= start) & (frame.trade_date <= end)])
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def synthetic_fundamentals(self, symbols: List[str], as_of: date) -> pd.DataFrame:
        rows = []
        years = list(range(2017, as_of.year + 1))
        for symbol in symbols:
            symbol_number = _stable_symbol_seed(symbol)
            local_rng = np.random.default_rng(self.seed + symbol_number * 19)
            quality = 0.56 + (symbol_number % 30) / 100
            value = 0.50 + (symbol_number % 28) / 100
            growth = 0.42 + (symbol_number % 36) / 100
            for year in years:
                maturity = (year - 2017) / max(as_of.year - 2017, 1)
                noise = lambda scale: float(local_rng.normal(0, scale))
                rows.append(
                    {
                        "symbol": symbol,
                        "report_date": date(year, 12, 31),
                        "roe": max(0.02, quality * 0.15 + noise(.012)),
                        "roa": max(0.01, quality * 0.07 + noise(.008)),
                        "revenue_growth": growth * 0.20 + maturity * 0.01 + noise(.035),
                        "profit_growth": growth * 0.24 + maturity * 0.015 + noise(.05),
                        "operating_cashflow": max(0.02, quality * 0.16 + noise(.025)),
                        "pe": max(5, 40 - value * 22 + noise(2.5) + growth * 4),
                        "pb": max(.5, 5.0 - value * 2.8 + noise(.3)),
                        "ps": max(.4, 9.8 - value * 5.0 + noise(.45)),
                        "dividend_yield": max(.002, value * .045 + noise(.008)),
                        "roe_stability": max(.1, quality * .80 + noise(.05)),
                        "gross_margin": max(.08, .20 + quality * .32 + noise(.025)),
                        "net_margin": max(.02, .05 + quality * .17 + noise(.018)),
                        "cashflow_profit_ratio": max(.25, .72 + quality * .28 + noise(.1)),
                        "data_mode": "demo_fallback",
                    }
                )
        return pd.DataFrame(rows)

    def fetch_benchmark(self, start: date, end: date) -> pd.DataFrame:
        df = self.generate_benchmark()
        return df[(df.trade_date >= start) & (df.trade_date <= end)].copy()
