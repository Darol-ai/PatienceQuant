"""真实数据适配器：改用baostock而不是akshare——这个沙盒环境访问akshare
的HTTP接口会被网络拦截(ProxyError)，但baostock走的是独立的socket协议，
实测能正常工作(见 app/quant_v3/csi300_universe.py 的沪深300成分股/真实
指数管道，本模块沿用同样验证过的接口调用方式)。

目录(代码/名称/行业)用 query_all_stock + query_stock_industry 两次批量
调用一次性拿到全市场真实数据，不逐支股票查——过去尝试对Demo目录里1000+
个"半真半假"代码逐个查baostock基本面的脚本(scripts/fetch_real_universe_
*.py)已经废弃，因为那些代码里有900多支是前缀+序号规则生成的虚构代码，
根本不保证对应真实上市公司。价格/基本面仍然按需懒加载，走现有SQLite
缓存架构，不做一次性全量预抓取。
"""
from __future__ import annotations

import os
import threading
from datetime import date, timedelta
from typing import Any, List, Optional

import pandas as pd

# baostock是登录会话制协议，官方没有文档说明支持同一账号并发多线程查询——
# 实测两个线程同时各自login/query时出现过"Error -3 while decompressing
# data: invalid distance too far back"这种解压错误(两路socket数据串流)，
# 和本session早前修CSI300模型缓存竞争时遇到的是同一类问题：外部资源不是
# 线程安全的，就把访问它的临界区用锁串行化。这里用一把进程级锁包住每次
# "登录→查询→登出"的完整过程，后台目录刷新线程和用户请求线程不会再
# 同时持有两个baostock会话。
_BAOSTOCK_LOCK = threading.Lock()

from app.data.akshare_provider import _normalise_query, _normalise_symbol
from app.data.demo import DemoDataProvider

# baostock不需要代理，但如果外层环境设置了HTTP(S)_PROXY，其底层socket
# 连接可能被错误路由——登录前清掉，和scripts/fetch_real_universe_*.py
# 里验证过有效的做法一致。
_PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def _to_bao_code(symbol: str) -> str:
    bare = _normalise_symbol(symbol)
    if len(bare) != 6:
        return symbol
    return f"{'sh' if bare[0] == '6' else 'sz'}.{bare}"


def _from_bao_code(code: str) -> str:
    return code.split(".", 1)[-1] if "." in code else code


_GROUP_KEYWORDS: List[tuple[str, List[str]]] = [
    ("金融", ["银行", "保险", "证券", "金融", "期货", "信托"]),
    ("新能源", ["汽车", "电力设备", "新能源", "电池", "光伏", "风电", "储能", "电网"]),
    ("科技", ["计算机", "软件", "通信", "电子", "半导体", "互联网", "传媒", "信息技术"]),
    ("消费", ["食品", "饮料", "家用电器", "纺织", "医药", "零售", "旅游", "农林牧渔", "商业", "美容"]),
]


def _infer_group(industry_text: str) -> tuple[str, str]:
    """把baostock真实行业分类文本，粗分到通用目录沿用的5个研究组之一。

    这是从真实行业分类文本做关键词归类，不是像AKShareDataProvider旧版
    那样从公司名/代码猜行业——归类粒度依然是启发式的，但输入源本身是
    真实数据。找不到关键词的归到"红利/央国企"(公用事业/材料/工业/地产/
    交运等传统行业的兜底分类，和原有分组语义一致)。
    """
    text = industry_text or ""
    for group, keywords in _GROUP_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return group, group
    return "红利/央国企", "红利"


class BaostockDataProvider:
    """真实数据适配器，通过baostock获取全市场目录+按需价格/基本面。"""

    mode = "real"

    def __init__(self, fallback: DemoDataProvider):
        self.fallback = fallback
        self._catalog_cache: Optional[pd.DataFrame] = None

    @property
    def catalog_cached(self) -> bool:
        return self._catalog_cache is not None and not self._catalog_cache.empty

    def _login(self):
        for var in _PROXY_ENV_VARS:
            os.environ.pop(var, None)
        import baostock as bs

        bs.login()
        return bs

    def _code_name_table(self) -> pd.DataFrame:
        if self._catalog_cache is not None:
            return self._catalog_cache
        try:
            with _BAOSTOCK_LOCK:
                bs = self._login()
                try:
                    # query_all_stock(day=今天)几乎总是返回空——当天的每日
                    # 快照要收盘后才发布，"今天"这份数据在盘中/刚收盘时
                    # 还没有。从今天往前最多扫6天，拿到第一个有数据的交易
                    # 日为止(实测连续调用不同日期不会有问题，单次调用
                    # 通常1秒内返回)。
                    all_stock = pd.DataFrame()
                    for offset in range(7):
                        probe_day = (date.today() - timedelta(days=offset)).isoformat()
                        stock_rs = bs.query_all_stock(day=probe_day)
                        stock_rows = []
                        while stock_rs.next():
                            stock_rows.append(stock_rs.get_row_data())
                        if stock_rows:
                            all_stock = pd.DataFrame(stock_rows, columns=stock_rs.fields)
                            break
                    if all_stock.empty:
                        raise RuntimeError("baostock query_all_stock最近7天都没有数据")

                    industry_rs = bs.query_stock_industry()
                    industry_rows = []
                    while industry_rs.next():
                        industry_rows.append(industry_rs.get_row_data())
                    industry = pd.DataFrame(industry_rows, columns=industry_rs.fields)
                finally:
                    bs.logout()

            # query_all_stock混杂了指数/基金代码——必须连交易所前缀一起匹配，
            # 不能只看去掉前缀后的6位数字：沪市指数用sh.000xxx这个号段，
            # 和深市真实个股sz.000xxx(比如sz.000001平安银行)去掉前缀后是
            # 完全相同的"000001"，光按6位数字过滤会把上证综合指数当成个股、
            # 还会在后面drop_duplicates时顶掉真正的平安银行。只保留：
            # 沪市60xxxx/688xxx/689xxx(主板+科创板)，深市00xxxx/30xxxx(主板+
            # 创业板)。
            all_stock = all_stock.rename(columns={"code_name": "name"})
            is_stock = all_stock["code"].str.match(r"^(sh\.(60|68)\d{4}|sz\.(00|30)\d{4})$")
            stocks = all_stock[is_stock].copy()
            stocks["symbol"] = stocks["code"].map(_from_bao_code)

            if not industry.empty:
                stocks = stocks.merge(industry[["code", "industry"]], on="code", how="left")
            else:
                stocks["industry"] = ""
            stocks["industry"] = stocks["industry"].fillna("").replace("", "综合")
            result = stocks[["symbol", "name", "industry"]].drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
            self._catalog_cache = result
            return result
        except Exception:
            self._catalog_cache = pd.DataFrame(columns=["symbol", "name", "industry"])
            return self._catalog_cache

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
                "tags": [group, "baostock"],
                "exchange": "A股",
                "source": "baostock",
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
        """baostock全市场真实代码+名称+真实行业分类，不再叠加任何虚构占位数据。"""
        table = self._code_name_table()
        if table.empty:
            # baostock登录/查询失败时，退到手工维护的70支真实公司列表——
            # 这个兜底本身也是真实公司，不会退回到虚构代码。
            return self.fallback.stock_catalog()
        return self._attach_metadata(table)

    def bootstrap(self):
        catalog = self.full_stock_catalog()
        if catalog.empty:
            return self.fallback.bootstrap()
        _, prices, fundamentals, benchmark = self.fallback.bootstrap()
        return catalog, prices, fundamentals, benchmark

    def fetch_prices(self, symbols: List[str], start: date, end: date) -> pd.DataFrame:
        normalised_symbols = [_normalise_symbol(symbol) for symbol in symbols]
        try:
            frames = []
            with _BAOSTOCK_LOCK:
                bs = self._login()
                try:
                    for symbol in normalised_symbols:
                        rs = bs.query_history_k_data_plus(
                            _to_bao_code(symbol), "date,open,high,low,close,volume",
                            start_date=start.isoformat(), end_date=end.isoformat(),
                            frequency="d", adjustflag="2",
                        )
                        if rs.error_code != "0":
                            continue
                        rows = []
                        while rs.next():
                            rows.append(rs.get_row_data())
                        if not rows:
                            continue
                        raw = pd.DataFrame(rows, columns=["trade_date", "open", "high", "low", "close", "volume"])
                        for column in ("open", "high", "low", "close", "volume"):
                            raw[column] = pd.to_numeric(raw[column], errors="coerce")
                        raw = raw.dropna()
                        if raw.empty:
                            continue
                        raw["symbol"] = symbol
                        raw["adj_close"] = raw["close"]
                        raw["data_mode"] = "real"
                        frames.append(raw[["symbol", "trade_date", "open", "high", "low", "close", "adj_close", "volume", "data_mode"]])
                finally:
                    bs.logout()
            if frames:
                result = pd.concat(frames, ignore_index=True)
                result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.date
                resolved = set(result.symbol.unique())
                missing = [symbol for symbol in normalised_symbols if symbol not in resolved]
                if missing:
                    fallback = self.fallback.fetch_prices(missing, start, end)
                    if not fallback.empty:
                        result = pd.concat([result, fallback], ignore_index=True)
                return result.sort_values(["trade_date", "symbol"])
        except Exception:
            pass
        return self.fallback.fetch_prices(normalised_symbols, start, end)

    def fetch_benchmark(self, start: date, end: date) -> pd.DataFrame:
        try:
            with _BAOSTOCK_LOCK:
                bs = self._login()
                try:
                    rs = bs.query_history_k_data_plus(
                        "sh.000300", "date,close",
                        start_date=start.isoformat(), end_date=end.isoformat(), frequency="d",
                    )
                    if rs.error_code != "0":
                        raise RuntimeError("baostock query_history_k_data_plus(沪深300)失败")
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                finally:
                    bs.logout()
            if not rows:
                raise RuntimeError("baostock返回空的沪深300序列")
            frame = pd.DataFrame(rows, columns=["trade_date", "close"])
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame = frame.dropna()
            frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
            frame["adj_close"] = frame["close"]
            frame["data_mode"] = "real"
            return frame
        except Exception:
            return self.fallback.fetch_benchmark(start, end)
