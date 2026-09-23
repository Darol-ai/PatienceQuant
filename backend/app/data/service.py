from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.data.akshare_provider import _normalise_query
from app.data.tushare_provider import TushareDataProvider, _to_ts_code
from app.data.demo import DemoDataProvider
from app.data.market_refresh import BENCHMARK_INDEX, get_market_store
from app.db.models import Fundamental, Industry, ResearchGroup, Stock, Watchlist


@lru_cache
def get_demo_provider() -> DemoDataProvider:
    settings = get_settings()
    return DemoDataProvider(settings.demo_seed, settings.demo_as_of)


@lru_cache
def get_real_provider() -> TushareDataProvider:
    return TushareDataProvider(get_demo_provider())


class MarketDataService:
    def __init__(self, db: Session, real_market_data: bool = False):
        """real_market_data=True：不管 DATA_MODE 怎么设，行情和指数都读本地行情库。
        模型选股策略用它——模型是在真实行情上训练的，拿 demo 模拟价格去
        打分和成交没有意义。"""
        self.db = db
        self.real_market_data = real_market_data
        self.demo = get_demo_provider()
        self.real = get_real_provider()
        # real模式下、重试3次后仍拿不到真实价格的symbol(ADR-0045：不能
        # 悄悄退回demo数据顶替)。调用方(路由层)可以读这个属性，把"部分
        # 行情获取失败"如实透出给用户，而不是装作一切正常。每次prices()
        # 调用都会重置。
        self.last_price_fetch_failures: List[str] = []

    @property
    def provider(self):
        return self.real if get_settings().data_mode.lower() == "real" else self.demo

    def _real_market(self) -> bool:
        return self.real_market_data or get_settings().data_mode.lower() == "real"

    @property
    def mode(self) -> str:
        if self.real_market_data:
            return "real"
        if self.provider is self.demo:
            return "demo"
        return "real" if self.real.catalog_cached else "real_or_demo_fallback"

    def seed_if_empty(self) -> None:
        """启动时必须快——tushare全市场目录一次stock_basic()调用比baostock
        时代(query_all_stock+query_stock_industry两次批量调用)简单，但
        仍然是一次跨几千行的真实网络请求，real模式下不能让它挡在FastAPI
        lifespan里同步跑，否则每次重启服务都要等它跑完才开始接受请求。
        这里始终先用demo的50支真实公司做启动兜底(本地生成，毫秒级)，
        真正的全市场tushare目录交给 app.main 里和CSI300预热同一套模式的
        后台任务异步刷新——服务立刻能起来，目录数据随后自动补齐。
        """
        if self.db.scalar(select(Stock.id).limit(1)):
            if get_settings().data_mode.lower() != "real":
                self.sync_catalog()
            return
        self._insert_catalog(self.demo.stock_catalog(), hide_stale=False)
        self.db.commit()

    def sync_catalog(self) -> None:
        """刷新目录——real模式下这一步本身就慢(见seed_if_empty的说明)，
        调用方需要自己决定是同步等待还是丢进后台线程。"""
        catalog = self.real.full_stock_catalog() if get_settings().data_mode.lower() == "real" else self.demo.stock_catalog()
        self._insert_catalog(catalog, hide_stale=get_settings().data_mode.lower() != "real")
        self.db.commit()

    def sync_akshare_catalog(self) -> Dict[str, object]:
        """Refresh the searchable A-share directory from tushare when available."""
        before = len(self.stocks())
        catalog = self.real.full_stock_catalog()
        source = "tushare" if not catalog.empty and not self.real._code_name_table().empty else "demo_fallback"
        self._insert_catalog(catalog, hide_stale=False)
        self.db.commit()
        after = len(self.stocks())
        return {
            "source": source,
            "data_mode": self.mode,
            "before_count": before,
            "after_count": after,
            "added_count": max(0, after - before),
            "message": "真实股票目录已刷新" if source == "tushare" else "tushare 不可用，继续使用 Demo 股票目录",
        }

    def search_catalog(
        self,
        query: str = "",
        limit: Optional[int] = None,
        source: str = "auto",
        group: str = "",
        industry: str = "",
        exchange: str = "",
    ) -> tuple[pd.DataFrame, str]:
        """Search the lightweight catalog without calculating factor scores."""
        requested_source = source if source in {"auto", "local", "tushare"} else "auto"
        # ``auto`` is intentionally local-first for offline startup, then
        # falls through to the real tushare directory when a user searches for a
        # code/name that is not in the Demo catalog.  This keeps the UI fast
        # and deterministic without forcing users to understand provider
        # switches before they can add a real A-share symbol.
        should_use_real = requested_source == "tushare"
        if requested_source == "auto" and query.strip():
            local_probe = self.stocks()
            local_needle = _normalise_query(query)
            if local_needle:
                local_names = local_probe.name.map(_normalise_query)
                local_industries = local_probe.industry.map(_normalise_query)
                local_sectors = local_probe.sector.map(_normalise_query)
                local_matches = local_probe[
                    local_probe.symbol.str.lower().str.contains(local_needle, regex=False)
                    | local_names.str.contains(local_needle, regex=False)
                    | local_industries.str.contains(local_needle, regex=False)
                    | local_sectors.str.contains(local_needle, regex=False)
                ]
                if group:
                    local_matches = local_matches[local_matches.group == group]
                if industry:
                    local_matches = local_matches[local_matches.industry == industry]
                if exchange:
                    local_matches = local_matches[local_matches.exchange == exchange]
                if not local_matches.empty:
                    should_use_real = False
                else:
                    should_use_real = True
        if should_use_real:
            real_result = self.real.search_stocks(query, limit, group=group, industry=industry, exchange=exchange)
            if not real_result.empty:
                # Make tushare-only symbols available to custom backtests and watchlists.
                self._insert_catalog(real_result.drop(columns=["source"], errors="ignore"), hide_stale=False)
                self.db.commit()
                return real_result, "tushare"

        frame = self.stocks()
        needle = _normalise_query(query)
        if needle:
            search_name = frame.name.map(_normalise_query)
            search_industry = frame.industry.map(_normalise_query)
            search_sector = frame.sector.map(_normalise_query)
            frame = frame[
                frame.symbol.str.lower().str.contains(needle, regex=False)
                | search_name.str.contains(needle, regex=False)
                | search_industry.str.contains(needle, regex=False)
                | search_sector.str.contains(needle, regex=False)
            ]
        if group:
            frame = frame[frame.group == group]
        if industry:
            frame = frame[frame.industry == industry]
        if exchange:
            frame = frame[frame.exchange == exchange]
        if limit is not None:
            frame = frame.head(max(1, min(int(limit), 200)))
        frame = frame.copy()
        # Preserve the provenance of tushare-discovered rows after they have
        # been cached in the common ``stocks`` table. Demo rows remain plainly
        # labelled, so the picker never presents a cached real directory as
        # if it were generated Demo data.
        frame["source"] = frame.tags.map(
            lambda tags: "tushare" if isinstance(tags, list) and "tushare" in tags else (
                "local_cache" if self.mode != "demo" else "demo"
            )
        )
        cached_real = frame["source"].eq("tushare").any() if "source" in frame.columns else False
        return frame, "local_cache" if cached_real else ("demo" if self.mode == "demo" else "local_cache")

    def _insert_catalog(self, catalog: pd.DataFrame, hide_stale: bool = True) -> None:
        group_by_name: Dict[str, ResearchGroup] = {}
        for row in [{"name": "消费", "description": "品牌消费、食品饮料与家电龙头", "color": "#f5b94c"},
                    {"name": "科技", "description": "半导体、软件、通信和智能制造", "color": "#6c8cff"},
                    {"name": "新能源", "description": "新能源车、光伏、储能与电力设备", "color": "#31d0aa"},
                    {"name": "金融", "description": "银行、保险、券商与金融科技", "color": "#c084fc"},
                    {"name": "红利/央国企", "description": "高股息、能源、公用事业和央国企", "color": "#fb7185"}]:
            obj = self.db.scalar(select(ResearchGroup).where(ResearchGroup.name == row["name"]))
            if obj is None:
                obj = ResearchGroup(**row)
                self.db.add(obj)
                self.db.flush()
            group_by_name[row["name"]] = obj
        industry_names = sorted(catalog["industry"].unique().tolist())
        industry_by_name: Dict[str, Industry] = {}
        for index, name in enumerate(industry_names):
            obj = self.db.scalar(select(Industry).where(Industry.name == name))
            if obj is None:
                obj = Industry(name=name, color=["#31d0aa", "#6c8cff", "#f5b94c", "#c084fc", "#fb7185"][index % 5])
                self.db.add(obj)
                self.db.flush()
            industry_by_name[name] = obj
        existing_rows = self.db.scalars(select(Stock)).all()
        existing_by_symbol = {stock.symbol: stock for stock in existing_rows}
        for row in catalog.to_dict(orient="records"):
            stock = existing_by_symbol.get(row["symbol"])
            if stock is None:
                self.db.add(Stock(symbol=row["symbol"], name=row["name"], industry_id=industry_by_name[row["industry"]].id,
                                  research_group_id=group_by_name[row["group"]].id, sector=row["sector"], tags=row["tags"],
                                  exchange=row.get("exchange", "A股")))
            else:
                # Refresh generated metadata in-place so upgrades from the
                # previous 300/1,250-symbol demo catalog are deterministic.
                stock.name = row["name"]
                stock.industry_id = industry_by_name[row["industry"]].id
                stock.research_group_id = group_by_name[row["group"]].id
                stock.sector = row["sector"]
                stock.tags = row["tags"]
                stock.exchange = row.get("exchange", "A股")
                stock.active = True
        # Keep old demo databases forward-compatible when the generated
        # universe changes size.  We never delete rows (historical orders stay
        # queryable); stale catalog rows are simply hidden from active screens.
        if hide_stale and get_settings().data_mode.lower() == "demo":
            active_symbols = set(catalog["symbol"].tolist())
            stale = self.db.scalars(select(Stock).where(Stock.active.is_(True))).all()
            for stock in stale:
                if stock.symbol not in active_symbols:
                    stock.active = False
        self.db.commit()

    def stocks(self) -> pd.DataFrame:
        rows = self.db.execute(
            select(Stock, Industry, ResearchGroup)
            .join(Industry, Stock.industry_id == Industry.id)
            .join(ResearchGroup, Stock.research_group_id == ResearchGroup.id)
            .where(Stock.active.is_(True))
        ).all()
        return pd.DataFrame([
            {"symbol": stock.symbol, "name": stock.name, "exchange": stock.exchange,
             "industry": industry.name, "group": group.name, "sector": stock.sector, "tags": stock.tags}
            for stock, industry, group in rows
        ])

    def ensure_symbols(self, symbols: List[str]) -> Dict[str, object]:
        """Ensure selected symbols exist in the local catalog before a backtest."""
        requested = list(dict.fromkeys(str(symbol).strip() for symbol in symbols if str(symbol).strip()))
        if not requested:
            return {"requested": 0, "resolved": 0, "source": "none"}
        catalog = self.stocks()
        existing = set(catalog.symbol.tolist())
        missing = [symbol for symbol in requested if symbol not in existing]
        if missing:
            resolved = self.real.lookup_symbols(missing)
            if not resolved.empty:
                self._insert_catalog(resolved.drop(columns=["source"], errors="ignore"), hide_stale=False)
                self.db.commit()
                existing.update(resolved.symbol.tolist())
        return {
            "requested": len(requested),
            "resolved": sum(symbol in existing for symbol in requested),
            "missing": [symbol for symbol in requested if symbol not in existing],
            "source": "tushare" if missing and existing.intersection(set(requested)) else self.mode,
        }

    def universe_symbols(self, universe: str = "a_share") -> List[str]:
        """Resolve a named product universe without leaking DB details to callers."""
        catalog = self.stocks()
        if catalog.empty:
            return []
        if universe.startswith("custom:"):
            try:
                watchlist = self.db.get(Watchlist, int(universe.split(":", 1)[1]))
            except (TypeError, ValueError):
                watchlist = None
            valid_symbols = set(catalog.symbol.tolist())
            return [symbol for symbol in (watchlist.symbols if watchlist else []) if symbol in valid_symbols]
        if universe == "large_cap":
            # The deterministic demo catalog starts with its research-grade
            # large-cap basket; keep the rule explicit and reproducible.
            return catalog.loc[catalog.exchange == "A股", "symbol"].head(300).tolist()
        if universe == "hs300":
            return catalog.loc[catalog.exchange == "A股", "symbol"].head(300).tolist()
        if universe == "csi_a500":
            return catalog.loc[catalog.exchange == "A股", "symbol"].head(500).tolist()
        if universe == "all_assets":
            return catalog.symbol.tolist()
        return catalog.loc[catalog.exchange == "A股", "symbol"].tolist()

    # Dashboard/股票池这类展示型因子评分不需要对全市场每一支都打分，一个
    # 有代表性的子集就够——上限不能直接照抄large_cap/hs300的"前300只"口径:
    # tushare没有批量接口，每支股票的完整历史要单独发一次请求，实测
    # 2018-2025这个区间平均每支4秒左右。300支意味着冷缓存下单次请求
    # 最坏要20分钟，对一个同步HTTP接口来说太慢；缩到30支，最坏情况~2
    # 分钟，命中缓存后同一区间的后续请求是毫秒级(prices()里写透缓存)。
    ANALYSIS_SYMBOL_CAP = 30

    def analysis_symbols(self, symbols: List[str]) -> List[str]:
        """Return a bounded set of symbols safe for cross-sectional analytics.

        real模式下tushare目录是全市场真实代码(几千支)，`prices()`对每个
        缺价格缓存的symbol是顺序发一次tushare请求，不是批量——如果不设
        上限，全市场几千支顺序请求会挂起几十分钟甚至更久。这里不再区分
        demo/real：优先选已有价格缓存的symbol，否则直接截断到
        ANALYSIS_SYMBOL_CAP，两种模式下都有确定的响应时间上限。需要对
        任意单支股票做完整分析的场景（自定义回测选股、股票详情页、手动
        同步）走的是各自独立的按需请求路径，不受这个上限影响。
        """
        requested = list(dict.fromkeys(str(symbol) for symbol in symbols))
        demo_symbols = set(self.demo.stock_catalog().symbol.tolist())
        filtered = [symbol for symbol in requested if symbol in demo_symbols]
        candidates = filtered or requested
        if len(candidates) <= self.ANALYSIS_SYMBOL_CAP:
            return candidates or requested[:1]
        # 直接按catalog顺序切前N支会系统性地漏掉排在后面的行业分组——比如
        # 50支目录按行业顺序排列时，前30支可能只覆盖3个组，后面两个组
        # 一个都进不来（/api/research/coverage这类按行业分组展示的页面
        # 会因此"丢组"，不是随机抽样问题，是纯顺序偏差）。这里按symbol所属
        # 的group轮询取，保证每个组都有代表，直到凑满上限。
        catalog = self.stocks()
        group_by_symbol = dict(zip(catalog.symbol, catalog.group))
        buckets: Dict[str, List[str]] = {}
        for symbol in candidates:
            buckets.setdefault(group_by_symbol.get(symbol, ""), []).append(symbol)
        result: List[str] = []
        while len(result) < self.ANALYSIS_SYMBOL_CAP and any(buckets.values()):
            for group in list(buckets):
                if len(result) >= self.ANALYSIS_SYMBOL_CAP:
                    break
                if buckets[group]:
                    result.append(buckets[group].pop(0))
        return result

    @staticmethod
    def frame_data_mode(frame: pd.DataFrame) -> str:
        """Summarize the actual price rows used by a calculation.

        ``data_mode`` on the service describes the configured provider, while
        a single custom backtest can legitimately mix cached Real rows,
        deterministic Demo rows, and a Demo fallback for a tushare symbol.
        Keeping this distinction explicit prevents a mixed run from being
        presented as wholly real data.
        """
        if frame.empty or "data_mode" not in frame.columns:
            return "unknown"
        modes = {str(value) for value in frame["data_mode"].dropna().unique()}
        if modes == {"real"}:
            return "real"
        if "real" in modes:
            return "mixed"
        if modes == {"demo_fallback"}:
            return "demo_fallback"
        if "demo_fallback" in modes:
            return "demo_mixed"
        return "demo"

    def prices(
        self,
        symbols: List[str],
        start: date,
        end: date,
        allow_network: bool = False,
    ) -> pd.DataFrame:
        """日线（前复权，锚定在区间最后一个交易日）。

        real 模式只读本地行情库（ADR-0047），不再按股票临时去数据源拉：库
        里没有的股票如实记进 last_price_fetch_failures，不拿 demo 数据顶替。
        allow_network 参数保留只是为了兼容旧调用方，已不起作用——补数据
        统一走 app.data.market_refresh。demo 模式用本地确定性模拟数据。
        """
        self.last_price_fetch_failures = []
        if not symbols:
            return pd.DataFrame()
        if not self._real_market():
            return self.demo.fetch_prices(symbols, start, end)
        code_by_symbol = {symbol: _to_ts_code(symbol) for symbol in symbols}
        symbol_by_code = {code: symbol for symbol, code in code_by_symbol.items()}
        frame = get_market_store().load_daily(start, end, code_by_symbol.values(), adjust="qfq")
        found = set(frame["ts_code"].unique()) if not frame.empty else set()
        self.last_price_fetch_failures = [symbol for symbol, code in code_by_symbol.items() if code not in found]
        if frame.empty:
            return pd.DataFrame()
        result = pd.DataFrame({
            "symbol": frame["ts_code"].map(symbol_by_code),
            "trade_date": frame["trade_date"].dt.date,
            "open": frame["open"], "high": frame["high"], "low": frame["low"], "close": frame["close"],
            "adj_close": frame["close"], "volume": frame["vol"], "data_mode": "real",
        })
        return result.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    def fundamentals(self, symbols: List[str], as_of: date) -> pd.DataFrame:
        cached = self.db.scalars(
            select(Fundamental).where(Fundamental.symbol.in_(symbols), Fundamental.report_date <= as_of)
        ).all()
        frames = []
        if cached:
            cached_df = pd.DataFrame([
                {column.name: getattr(row, column.name) for column in Fundamental.__table__.columns if column.name != "id"}
                for row in cached
            ])
            frames.append(cached_df)
            covered = set(cached_df.symbol.unique())
        else:
            covered = set()
        missing = [symbol for symbol in symbols if symbol not in covered]
        if missing:
            demo = self.demo.generate_fundamentals()
            demo_frame = demo[(demo.symbol.isin(missing)) & (demo.report_date <= as_of)].copy()
            if not demo_frame.empty:
                frames.append(demo_frame)
                covered.update(demo_frame.symbol.unique())
        still_missing = [symbol for symbol in symbols if symbol not in covered]
        if still_missing:
            synthetic = self.demo.synthetic_fundamentals(still_missing, as_of)
            if not synthetic.empty:
                frames.append(synthetic[synthetic.report_date <= as_of])
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def benchmark(self, start: date, end: date) -> pd.DataFrame:
        """沪深300指数。real 模式只读本地行情库，没有就返回空表，不拿 demo 曲线顶替。"""
        if not self._real_market():
            return self.provider.fetch_benchmark(start, end)
        frame = get_market_store().load_index(BENCHMARK_INDEX, start, end)
        if frame.empty:
            return pd.DataFrame(columns=["trade_date", "close", "adj_close", "data_mode"])
        return pd.DataFrame({
            "trade_date": frame["trade_date"].dt.date,
            "close": frame["close"], "adj_close": frame["close"], "data_mode": "real",
        })
