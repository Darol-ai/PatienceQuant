"""真实数据适配器：用tushare替代baostock（ADR-0046，取代ADR-0045时代的
数据源）。目录(代码/名称/行业)用一次stock_basic()批量调用
拿到全市场真实数据（比baostock时代省一次调用——industry字段就在同一张
表里，不用像query_all_stock+query_stock_industry那样分两次查再merge）。
价格/基本面仍然按需懒加载，走现有SQLite缓存架构，不做一次性全量预抓取。

连接层（限速/超时/失败重试）统一在 app.data.tushare_client 里实现，见
docs/adr/0046——本模块只管"问tushare要哪些字段、股票代码怎么转换成
xxxxxx.SH/.SZ后缀格式、前复权价怎么算"这些业务逻辑。
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional

import pandas as pd

from app.data.akshare_provider import _normalise_query, _normalise_symbol
from app.data.demo import DemoDataProvider
from app.data.tushare_client import TushareQueryFailed, run as run_tushare


def _to_ts_code(symbol: str) -> str:
    bare = _normalise_symbol(symbol)
    if len(bare) != 6:
        return symbol
    return f"{bare}.{'SH' if bare[0] == '6' else 'SZ'}"


def _from_ts_code(code: str) -> str:
    return code.split(".", 1)[0] if "." in code else code


_GROUP_KEYWORDS: List[tuple[str, List[str]]] = [
    ("金融", ["银行", "保险", "证券", "金融", "期货", "信托"]),
    ("新能源", ["汽车", "电力设备", "新能源", "电池", "光伏", "风电", "储能", "电网"]),
    ("科技", ["计算机", "软件", "通信", "电子", "半导体", "互联网", "传媒", "信息技术"]),
    ("消费", ["食品", "饮料", "家用电器", "纺织", "医药", "零售", "旅游", "农林牧渔", "商业", "美容"]),
]


def _infer_group(industry_text: str) -> tuple[str, str]:
    """把真实行业分类文本粗分到通用目录沿用的5个研究组之一——沿用
    baostock时代同一套关键词归类逻辑（tushare的行业分类文本格式跟
    baostock不完全一样，但都是中文行业名词，关键词匹配基本兼容）。
    找不到关键词的归到"红利/央国企"兜底分类。
    """
    text = industry_text or ""
    for group, keywords in _GROUP_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return group, group
    return "红利/央国企", "红利"


class TushareDataProvider:
    """真实数据适配器，通过tushare获取全市场目录+按需价格/基准指数。"""

    mode = "real"

    def __init__(self, fallback: DemoDataProvider):
        self.fallback = fallback
        self._catalog_cache: Optional[pd.DataFrame] = None
        # 最近一次fetch_prices()里，重试3次后仍拿不到真实数据的股票代码——
        # ADR-0045/0046规定这种情况不能悄悄退回demo数据顶替，调用方(最终
        # 是app/data/service.py::MarketDataService)需要能读到这个列表，
        # 把"部分行情获取失败"如实透出给用户，而不是装作一切正常。
        self.last_failed_symbols: List[str] = []

    @property
    def catalog_cached(self) -> bool:
        return self._catalog_cache is not None and not self._catalog_cache.empty

    def _code_name_table(self) -> pd.DataFrame:
        if self._catalog_cache is not None:
            return self._catalog_cache

        def _fetch(pro):
            frame = pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,industry,list_date")
            if frame is None or frame.empty:
                raise RuntimeError("tushare stock_basic返回空表")
            return frame

        try:
            all_stock = run_tushare(_fetch)
        except TushareQueryFailed:
            # 目录是"能退化成更小的真实公司集合"的场景(不是伪造数据)——
            # 全市场目录拿不到时，退到手工维护的50支真实公司(真实代码+
            # 真实名字，只是数量少)，好过整个通用目录直接空掉。这和价格/
            # 基准指数的"不退demo"不是同一类：那两个是会直接进策略计算
            # 的数值，退demo等于用假数字冒充真实结果；目录退化只是少几千
            # 个可搜索的条目，不产生虚假数值。
            self._catalog_cache = pd.DataFrame(columns=["symbol", "name", "industry"])
            return self._catalog_cache

        # stock_basic()本身就只返回股票(不像baostock的query_all_stock混杂
        # 指数/基金)，但为了跟通用目录原有口径(沪市主板60/科创板68、深市
        # 主板00/创业板30)保持一致、不引入之前没有的北交所代码，仍然按
        # 代码段过滤一次。
        is_stock = all_stock["ts_code"].str.match(r"^(60|68)\d{4}\.SH$|^(00|30)\d{4}\.SZ$")
        stocks = all_stock[is_stock].copy()
        stocks["symbol"] = stocks["ts_code"].map(_from_ts_code)
        stocks["industry"] = stocks["industry"].fillna("").replace("", "综合")
        result = stocks[["symbol", "name", "industry"]].drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
        self._catalog_cache = result
        return result

    def _attach_metadata(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "name", "exchange", "industry", "group", "sector", "tags", "source"])
        rows = []
        for row in frame.to_dict(orient="records"):
            group, sector = _infer_group(row.get("industry", ""))
            rows.append({
                **row,
                "group": group,
                "sector": sector,
                "tags": [group, "tushare"],
                "exchange": "A股",
                "source": "tushare",
            })
        return pd.DataFrame(rows)

    def lookup_symbols(self, symbols: List[str]) -> pd.DataFrame:
        requested = {_normalise_symbol(symbol) for symbol in symbols}
        table = self._code_name_table()
        if table.empty:
            return pd.DataFrame(columns=["symbol", "name", "exchange", "industry", "group", "sector", "tags", "source"])
        result = table[table.symbol.isin(requested)].copy()
        return self._attach_metadata(result)

    def search_stocks(
        self,
        query: str = "",
        limit: Optional[int] = None,
        group: str = "",
        industry: str = "",
        exchange: str = "",
    ) -> pd.DataFrame:
        table = self._code_name_table()
        if table.empty:
            return pd.DataFrame(columns=["symbol", "name", "exchange", "industry", "group", "sector", "tags", "source"])
        needle = _normalise_query(query)
        result = table
        if needle:
            result = result[
                result.symbol.str.lower().str.contains(needle, regex=False)
                | result.name.map(_normalise_query).str.contains(needle, regex=False)
                | result.industry.map(_normalise_query).str.contains(needle, regex=False)
            ]
        enriched = self._attach_metadata(result.copy())
        if group:
            enriched = enriched[enriched.group == group]
        if industry:
            enriched = enriched[enriched.industry == industry]
        if exchange:
            enriched = enriched[enriched.exchange == exchange]
        if limit is None:
            return enriched
        return enriched.head(max(1, min(int(limit), 200)))

    def stock_catalog(self) -> pd.DataFrame:
        """协议同DemoDataProvider：给种子/缓存用的小目录。"""
        return self.full_stock_catalog()

    def full_stock_catalog(self) -> pd.DataFrame:
        """tushare全市场真实代码+名称+真实行业分类，不再叠加任何虚构占位数据。"""
        table = self._code_name_table()
        if table.empty:
            # tushare登录/查询失败时，退到手工维护的50支真实公司列表——
            # 这个兜底本身也是真实公司，不会退回到虚构代码。
            return self.fallback.stock_catalog()
        return self._attach_metadata(table)

    def bootstrap(self):
        catalog = self.full_stock_catalog()
        if catalog.empty:
            return self.fallback.bootstrap()
        _, prices, fundamentals, benchmark = self.fallback.bootstrap()
        return catalog, prices, fundamentals, benchmark

    def _fetch_one_qfq(self, pro, ts_code: str, start: date, end: date) -> pd.DataFrame:
        """单支股票的前复权日线。tushare原始daily()是不复权价格，前复权
        要另外拿adj_factor()自己算：前复权价 = 原始价 × 当日复权因子 /
        区间内最新一天的复权因子（锚定在这次查询自己的最后一个交易日，
        不是锚定"今天"——同一次查询内部相对收益/动量计算不受锚点选择
        影响，只影响绝对价格水平）。"""
        daily = pro.daily(ts_code=ts_code, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        if daily is None or daily.empty:
            return pd.DataFrame()
        factor = pro.adj_factor(ts_code=ts_code, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        if factor is None or factor.empty:
            raise RuntimeError(f"{ts_code} adj_factor返回空表，无法计算前复权价")
        merged = daily.merge(factor[["trade_date", "adj_factor"]], on="trade_date", how="inner")
        if merged.empty:
            raise RuntimeError(f"{ts_code} daily/adj_factor日期对不上，无法计算前复权价")
        merged = merged.sort_values("trade_date")
        latest_factor = merged["adj_factor"].iloc[-1]
        for column in ("open", "high", "low", "close"):
            merged[column] = merged[column] * merged["adj_factor"] / latest_factor
        return merged[["trade_date", "open", "high", "low", "close", "vol"]].rename(columns={"vol": "volume"})

    def fetch_prices(self, symbols: List[str], start: date, end: date) -> pd.DataFrame:
        """按symbol逐个走独立的重试序列(而不是整批共用一次重试)——这样
        一支股票卡住重试，不会连累同一批里已经成功的其它股票要跟着
        重跑。3次重试后仍失败的symbol记在self.last_failed_symbols里，
        不在这里也不在调用方悄悄补demo数据(ADR-0045/0046)。
        """
        normalised_symbols = [_normalise_symbol(symbol) for symbol in symbols]
        frames = []
        failed: List[str] = []
        for symbol in normalised_symbols:
            ts_code = _to_ts_code(symbol)

            def _fetch(pro, ts_code=ts_code):
                return self._fetch_one_qfq(pro, ts_code, start, end)

            try:
                raw = run_tushare(_fetch)
            except TushareQueryFailed:
                failed.append(symbol)
                continue
            if raw.empty:
                # 查询本身成功，只是这个区间没有数据(比如还没上市)——不算
                # "拿不到真实数据"的失败，是真实的"没有"。
                continue
            for column in ("open", "high", "low", "close", "volume"):
                raw[column] = pd.to_numeric(raw[column], errors="coerce")
            raw = raw.dropna()
            if raw.empty:
                continue
            raw["symbol"] = symbol
            raw["adj_close"] = raw["close"]
            raw["data_mode"] = "real"
            frames.append(raw[["symbol", "trade_date", "open", "high", "low", "close", "adj_close", "volume", "data_mode"]])

        self.last_failed_symbols = failed
        if not frames:
            return pd.DataFrame(columns=["symbol", "trade_date", "open", "high", "low", "close", "adj_close", "volume", "data_mode"])
        result = pd.concat(frames, ignore_index=True)
        result["trade_date"] = pd.to_datetime(result["trade_date"], format="%Y%m%d").dt.date
        return result.sort_values(["trade_date", "symbol"])

    def fetch_benchmark(self, start: date, end: date) -> pd.DataFrame:
        def _fetch(pro):
            frame = pro.index_daily(ts_code="000300.SH", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
            if frame is None or frame.empty:
                raise RuntimeError("tushare index_daily(沪深300)返回空表")
            return frame

        try:
            frame = run_tushare(_fetch)
        except TushareQueryFailed:
            # ADR-0045/0046：基准指数直接参与超额收益计算，不退回demo曲线
            # 冒充真实指数——返回空表，调用方(app/quant_v3/real_benchmark.py
            # 风格的"没有真实对照就诚实返回None"同一个原则)自己决定怎么
            # 处理"这次没有真实基准"。
            return pd.DataFrame(columns=["trade_date", "close", "adj_close", "data_mode"])

        frame = frame[["trade_date", "close"]].copy()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d").dt.date
        frame["adj_close"] = frame["close"]
        frame["data_mode"] = "real"
        return frame.sort_values("trade_date").reset_index(drop=True)
