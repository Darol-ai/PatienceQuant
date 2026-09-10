from __future__ import annotations

from datetime import date
import re
import unicodedata
from typing import Any, List, Optional

import pandas as pd

from app.data.demo import DemoDataProvider


def _normalise_symbol(value: Any) -> str:
    text = str(value or "").strip().upper()
    # Accept the common AKShare/vendor spellings:
    # ``600036``, ``600036.SH``, ``SH600036`` and ``SH.600036``.
    if "." in text:
        parts = [part for part in text.split(".") if part]
        text = next((part for part in parts if any(character.isdigit() for character in part)), parts[0] if parts else text)
    if text.startswith(("SH", "SZ", "BJ")):
        text = text[2:]
    digits = "".join(character for character in text if character.isdigit())
    return digits[-6:] if len(digits) >= 6 else text


def _normalise_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[AB]$", "", text, flags=re.IGNORECASE)


def _normalise_query(value: Any) -> str:
    """Normalise user-entered code/name text for tolerant directory search.

    Chinese A-share names are commonly copied from vendor tables with
    full-width punctuation, spaces, or a trailing ``A``/``B`` suffix
    (``万  科Ａ`` / ``万科A``).  AKShare may expose the same security as
    ``万科``.  NFKC handles full-width forms; removing a trailing class
    suffix keeps those common inputs searchable without changing six-digit
    code queries.
    """
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).lower()
    if re.search(r"[\u4e00-\u9fff]", text):
        text = re.sub(r"[ab]$", "", text)
    return text


class AKShareDataProvider:
    """Best-effort real-data adapter. Demo remains the safe fallback."""

    mode = "real"

    def __init__(self, fallback: DemoDataProvider):
        self.fallback = fallback
        self._code_name_cache: Optional[pd.DataFrame] = None

    @property
    def catalog_cached(self) -> bool:
        return self._code_name_cache is not None and not self._code_name_cache.empty

    @staticmethod
    def infer_metadata(symbol: str, name: str) -> dict[str, Any]:
        """Infer a conservative product taxonomy when AKShare only provides code/name."""
        text = f"{symbol}{name}"
        rules = [
            ("金融", "金融", "银行", ("银行", "金融"), ["银行", "农商", "城商"]),
            ("金融", "金融", "保险", ("保险", "金融"), ["保险", "人寿", "太保"]),
            ("金融", "金融", "证券", ("券商", "资本市场"), ["证券", "券商", "期货"]),
            ("消费", "消费", "白酒", ("品牌", "消费"), ["茅台", "五粮", "泸州", "汾酒", "白酒", "酒"]),
            ("消费", "消费", "食品饮料", ("消费", "食品"), ["食品", "饮料", "乳业", "味业", "啤酒", "调味"]),
            ("消费", "消费", "家电", ("家电", "消费"), ["家电", "美的", "格力", "海尔"]),
            ("消费", "医药", "医药", ("医药", "医疗"), ["医药", "药业", "制药", "生物", "医疗", "器械"]),
            ("新能源", "新能源", "新能源车", ("新能源车", "低碳"), ["新能源", "宁德", "比亚迪", "锂", "电池", "汽车"]),
            ("新能源", "新能源", "光伏设备", ("光伏", "低碳"), ["光伏", "太阳能", "隆基", "通威"]),
            ("新能源", "新能源", "电力设备", ("电力设备", "储能"), ["电气", "电源", "储能", "风电", "核电"]),
            ("科技", "科技", "半导体", ("芯片", "科技"), ["半导体", "芯片", "中芯", "华创", "微电"]),
            ("科技", "科技", "软件", ("软件", "科技"), ["软件", "信息", "数据", "云", "AI", "智能"]),
            ("科技", "科技", "电子", ("电子", "科技"), ["电子", "通信", "光电", "计算机", "科技"]),
            ("红利/央国企", "红利", "石油石化", ("能源", "央国企"), ["石油", "石化", "煤", "能源", "矿"]),
            ("红利/央国企", "红利", "电力", ("电力", "高股息"), ["电力", "水电", "华能", "国电", "长江"]),
            ("红利/央国企", "红利", "建筑装饰", ("基建", "央国企"), ["中国", "中铁", "中交", "建筑", "铁建", "中车"]),
        ]
        for group, sector, industry, tags, keywords in rules:
            if any(keyword in text for keyword in keywords):
                return {
                    "industry": industry,
                    "group": group,
                    "sector": sector,
                    "tags": list(tags) + ["AKShare"],
                    "exchange": "A股",
                }
        if symbol.startswith(("688", "689", "300", "301")):
            return {"industry": "科技成长", "group": "科技", "sector": "科技", "tags": ["成长", "AKShare"], "exchange": "A股"}
        if symbol.startswith(("600", "601")):
            return {"industry": "大盘蓝筹", "group": "红利/央国企", "sector": "红利", "tags": ["大盘股", "AKShare"], "exchange": "A股"}
        return {"industry": "A股综合", "group": "消费", "sector": "消费", "tags": ["A股", "AKShare"], "exchange": "A股"}

    def _code_name_table(self) -> pd.DataFrame:
        if self._code_name_cache is not None:
            return self._code_name_cache
        try:
            import akshare as ak  # type: ignore

            raw = ak.stock_info_a_code_name()
            if raw is None or raw.empty:
                raise RuntimeError("AKShare returned no stock list")
            columns = {str(column).lower(): column for column in raw.columns}
            code_column = columns.get("code") or columns.get("代码") or columns.get("symbol") or columns.get("证券代码")
            name_column = columns.get("name") or columns.get("名称") or columns.get("股票简称") or columns.get("简称")
            if code_column is None or name_column is None:
                raise RuntimeError("AKShare stock list columns changed")
            result = raw[[code_column, name_column]].rename(columns={code_column: "symbol", name_column: "name"}).copy()
            result["symbol"] = result["symbol"].map(_normalise_symbol)
            result["name"] = result["name"].map(_normalise_name)
            result = result[result.symbol.str.len() == 6]
            result = result.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
            self._code_name_cache = result
            return result
        except Exception:
            self._code_name_cache = pd.DataFrame(columns=["symbol", "name"])
            return self._code_name_cache

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

    def _attach_metadata(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "name", "exchange", "industry", "group", "sector", "tags", "source"])
        rows = []
        for row in frame.to_dict(orient="records"):
            metadata = self.infer_metadata(row["symbol"], row["name"])
            rows.append({**row, **metadata, "source": "akshare"})
        return pd.DataFrame(rows)

    def stock_catalog(self) -> pd.DataFrame:
        """Return a catalog through the same protocol as DemoDataProvider."""
        catalog = self.fallback.stock_catalog().copy()
        info = self._code_name_table()
        if not info.empty:
            catalog = catalog.drop(columns=["name"]).merge(info[["symbol", "name"]], on="symbol", how="left")
            catalog["name"] = catalog["name"].fillna(catalog["symbol"])
        return catalog

    def full_stock_catalog(self) -> pd.DataFrame:
        """Return fallback catalog plus every A-share code/name AKShare exposes."""
        catalog = self.stock_catalog()
        info = self._code_name_table()
        if info.empty:
            return catalog
        known = set(catalog["symbol"].tolist())
        extras = self._attach_metadata(info[~info.symbol.isin(known)].copy())
        if extras.empty:
            return catalog
        return pd.concat(
            [catalog, extras[["symbol", "name", "industry", "group", "sector", "tags", "exchange"]]],
            ignore_index=True,
        )

    def bootstrap(self):
        catalog, prices, fundamentals, benchmark = self.fallback.bootstrap()
        info = self._code_name_table()
        if info.empty:
            return catalog, prices, fundamentals, benchmark
        catalog = catalog.copy()
        catalog = catalog.drop(columns=["name"]).merge(info[["symbol", "name"]], on="symbol", how="left")
        catalog["name"] = catalog["name"].fillna(catalog["symbol"])
        return catalog, prices, fundamentals, benchmark

    def fetch_prices(self, symbols: List[str], start: date, end: date) -> pd.DataFrame:
        normalised_symbols = [_normalise_symbol(symbol) for symbol in symbols]
        try:
            import akshare as ak  # type: ignore

            frames = []
            for symbol in normalised_symbols:
                try:
                    raw = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start.strftime("%Y%m%d"),
                        end_date=end.strftime("%Y%m%d"),
                        adjust="qfq",
                    )
                except Exception:
                    continue
                if raw is None or raw.empty:
                    continue
                raw = raw.rename(columns={"日期": "trade_date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"})
                required = {"trade_date", "open", "high", "low", "close", "volume"}
                if not required.issubset(raw.columns):
                    continue
                raw["symbol"] = symbol
                raw["adj_close"] = raw["close"]
                raw["data_mode"] = "real"
                frames.append(raw[["symbol", "trade_date", "open", "high", "low", "close", "adj_close", "volume", "data_mode"]])
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
        return self.fallback.fetch_benchmark(start, end)
