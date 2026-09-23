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

# Demo数据只保留这份手工维护的真实公司列表(代码和名字都真实，只是
# 价格/基本面走本地模拟)——之前用前缀+序号规则批量生成的900多支虚构
# 代码和PINK_SPECS虚构OTC代码已经删除，不再作为通用目录的兜底填充；
# 通用目录的真实来源是tushare(见 app/data/tushare_provider.py)。
STOCK_SPECS: List[StockSpec] = BASE_STOCK_SPECS


def trading_days(start: date, end: date) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="B")


def _stable_symbol_seed(symbol: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(str(symbol)))


class DemoDataProvider:
    mode = "demo"

    def __init__(self, seed: int = 20260909, as_of: Optional[date] = None):
        self.seed = seed
        self.as_of = as_of or date.today()
        self.specs = BASE_STOCK_SPECS
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
