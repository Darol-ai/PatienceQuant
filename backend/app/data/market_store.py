"""A 股本地行情库（ADR-0047）：系统中行情数据的唯一来源。

存储布局（backend/data/market/a_share/）：
- calendar.parquet        上交所交易日历
- daily/<年>.parquet      全市场日线原始价 + 复权因子，按年份分文件
- index/<代码>.parquet    指数日线（如 000300.SH）

按交易日批量拉取：每个交易日调 daily、adj_factor 各一次，一次拿到当天全
市场。行情库本身就是进度记录——哪些交易日已经在库里，就不再拉，所以
中途被额度卡住，下次接着补即可。按年份分文件是为了每次补数据只改当年
那一个文件（整个库推到 Git LFS 时，未变动的年份不产生新版本）。

存原始价和复权因子，不存复权后的价格：前复权价取决于"锚定到哪一天"，
存死了以后每出现一次除权都要重写全部历史；读取时现算即可。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import pandas as pd

from app.data.tushare_client import TushareQueryFailed, run as run_tushare

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "market"
# 2010 年起：2016 年起的数据只够给 2019 年的模型约两年训练样本（ADR-0052 第二步结果）
STORE_START = date(2010, 1, 1)
INDEX_START = date(2005, 1, 1)
DAILY_COLUMNS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "adj_factor"]
# 每日指标：换手率(%)、市盈率/市净率/市销率(TTM)、股息率(TTM,%)、总市值/流通市值(万元)
BASIC_COLUMNS = ["ts_code", "trade_date", "turnover_rate", "turnover_rate_f", "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv"]
_FLUSH_EVERY_DAYS = 20

Runner = Callable[[Callable[[object], pd.DataFrame]], pd.DataFrame]


@dataclass
class BackfillReport:
    requested_days: int
    filled_days: List[date] = field(default_factory=list)
    # 开市日但数据源返回空：通常是当天数据还没发布，下次补齐时会再试。
    empty_days: List[date] = field(default_factory=list)
    # 碰到额度/权限/网络错误而提前停下时的原因；None 表示全部补完。
    stopped_reason: Optional[str] = None


def _compact(day: date) -> str:
    return day.strftime("%Y%m%d")


class AShareMarketStore:
    def __init__(self, root: Path = DEFAULT_ROOT, runner: Runner = run_tushare, start: date = STORE_START):
        self.root = Path(root) / "a_share"
        self.runner = runner
        self.start = start

    # ---- 交易日历 ----

    def _calendar_path(self) -> Path:
        return self.root / "calendar.parquet"

    def open_days(self, until: date) -> List[date]:
        """[start, until] 内的开市日。本地日历覆盖不到 until 时，拉到当年年底并保存。"""
        path = self._calendar_path()
        cal = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["cal_date", "is_open"])
        cal_dates = pd.to_datetime(cal["cal_date"]).dt.date if not cal.empty else None
        if cal.empty or cal_dates.max() < until or cal_dates.min() > self.start:
            year_end = date(until.year, 12, 31)
            fetched = self.runner(lambda pro: pro.trade_cal(exchange="SSE", start_date=_compact(self.start), end_date=_compact(year_end)))
            cal = fetched[["cal_date", "is_open"]].copy()
            cal["cal_date"] = pd.to_datetime(cal["cal_date"])
            cal["is_open"] = cal["is_open"].astype(int)
            path.parent.mkdir(parents=True, exist_ok=True)
            cal.sort_values("cal_date").to_parquet(path, index=False)
        cal_dates = pd.to_datetime(cal["cal_date"]).dt.date
        mask = (cal["is_open"].astype(int) == 1) & (cal_dates >= self.start) & (cal_dates <= until)
        return sorted(cal_dates[mask].tolist())

    def latest_trading_day(self, today: Optional[date] = None) -> date:
        days = self.open_days(today or date.today())
        if not days:
            raise ValueError("交易日历里没有开市日")
        return days[-1]

    # ---- 日线 ----

    def _daily_path(self, year: int) -> Path:
        return self.root / "daily" / f"{year}.parquet"

    def stored_days(self) -> List[date]:
        folder = self.root / "daily"
        if not folder.exists():
            return []
        days = set()
        for path in folder.glob("*.parquet"):
            days.update(pd.read_parquet(path, columns=["trade_date"])["trade_date"].dt.date.unique())
        return sorted(days)

    def missing_days(self, until: date, since: Optional[date] = None) -> List[date]:
        stored = set(self.stored_days())
        lower = max(since or self.start, self.start)
        return [day for day in self.open_days(until) if day >= lower and day not in stored]

    def _fetch_day(self, day: date) -> pd.DataFrame:
        ds = _compact(day)
        daily = self.runner(lambda pro: pro.daily(trade_date=ds))
        if daily is None or daily.empty:
            return pd.DataFrame(columns=DAILY_COLUMNS)
        factor = self.runner(lambda pro: pro.adj_factor(trade_date=ds))
        if factor is None or factor.empty:
            # 日线有了复权因子还没有：半天的数据不入库，下次一起补。
            return pd.DataFrame(columns=DAILY_COLUMNS)
        merged = daily.merge(factor[["ts_code", "adj_factor"]], on="ts_code", how="left")
        merged["trade_date"] = pd.to_datetime(merged["trade_date"], format="%Y%m%d")
        return merged[DAILY_COLUMNS]

    def _append(self, frames: List[pd.DataFrame]) -> None:
        if not frames:
            return
        batch = pd.concat(frames, ignore_index=True)
        for year, rows in batch.groupby(batch["trade_date"].dt.year):
            path = self._daily_path(int(year))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                rows = pd.concat([pd.read_parquet(path), rows], ignore_index=True)
            rows = rows.drop_duplicates(["ts_code", "trade_date"], keep="last").sort_values(["trade_date", "ts_code"])
            rows.to_parquet(path, index=False, compression="zstd")

    def backfill(
        self,
        until: Optional[date] = None,
        since: Optional[date] = None,
        max_days: Optional[int] = None,
        on_progress: Optional[Callable[[date, int, int], None]] = None,
    ) -> BackfillReport:
        """把 [since, until] 里库中缺少的开市日补上。可以随时中断、下次续补。"""
        until = until or self.latest_trading_day()
        todo = self.missing_days(until, since)
        if max_days is not None:
            todo = todo[:max_days]
        report = BackfillReport(requested_days=len(todo))
        buffer: List[pd.DataFrame] = []
        try:
            for index, day in enumerate(todo, start=1):
                rows = self._fetch_day(day)
                if rows.empty:
                    report.empty_days.append(day)
                else:
                    buffer.append(rows)
                    report.filled_days.append(day)
                if len(buffer) >= _FLUSH_EVERY_DAYS:
                    self._append(buffer)
                    buffer = []
                if on_progress:
                    on_progress(day, index, len(todo))
        except TushareQueryFailed as exc:
            report.stopped_reason = str(exc)
        finally:
            # 停下前已经拿到的日期照样落盘，下次只补剩下的。
            self._append(buffer)
        return report

    def load_daily(
        self,
        start: date,
        end: date,
        ts_codes: Optional[Iterable[str]] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """读 [start, end] 的日线。adjust="qfq" 时按区间内每支股票最后一天的
        复权因子前复权（锚定在这次读取自己的最后一个交易日）；"none" 返回原始价。
        没有复权因子的行无法如实复权，qfq 模式下丢弃。"""
        frames = []
        for year in range(start.year, end.year + 1):
            path = self._daily_path(year)
            if path.exists():
                frames.append(pd.read_parquet(path))
        if not frames:
            return pd.DataFrame(columns=DAILY_COLUMNS)
        data = pd.concat(frames, ignore_index=True)
        data = data[(data["trade_date"] >= pd.Timestamp(start)) & (data["trade_date"] <= pd.Timestamp(end))]
        if ts_codes is not None:
            data = data[data["ts_code"].isin(set(ts_codes))]
        data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        if adjust == "qfq":
            data = data.dropna(subset=["adj_factor"])
            latest = data.groupby("ts_code")["adj_factor"].transform("last")
            ratio = data["adj_factor"] / latest
            for column in ("open", "high", "low", "close"):
                data[column] = data[column] * ratio
        return data

    # ---- 每日指标（tushare daily_basic，ADR-0053）----

    def _basic_path(self, year: int) -> Path:
        return self.root / "daily_basic" / f"{year}.parquet"

    def stored_basic_days(self) -> List[date]:
        folder = self.root / "daily_basic"
        if not folder.exists():
            return []
        days = set()
        for path in folder.glob("*.parquet"):
            days.update(pd.read_parquet(path, columns=["trade_date"])["trade_date"].dt.date.unique())
        return sorted(days)

    def missing_basic_days(self, until: date) -> List[date]:
        """库里已有日线、但还没有每日指标的交易日（每日指标跟着日线补，不单独超前）。"""
        stored = set(self.stored_basic_days())
        return [day for day in self.stored_days() if day <= until and day not in stored]

    def backfill_basic(self, until: Optional[date] = None, max_days: Optional[int] = None,
                       on_progress: Optional[Callable[[date, int, int], None]] = None) -> BackfillReport:
        """每个交易日一次 daily_basic 调用；可随时中断、下次续补。"""
        until = until or self.latest_trading_day()
        todo = self.missing_basic_days(until)
        if max_days is not None:
            todo = todo[:max_days]
        report = BackfillReport(requested_days=len(todo))
        buffer: List[pd.DataFrame] = []
        try:
            for index, day in enumerate(todo, start=1):
                ds = _compact(day)
                rows = self.runner(lambda pro: pro.daily_basic(trade_date=ds, fields=",".join(BASIC_COLUMNS)))
                if rows is None or rows.empty:
                    report.empty_days.append(day)
                else:
                    rows = rows.reindex(columns=BASIC_COLUMNS)
                    rows["trade_date"] = pd.to_datetime(rows["trade_date"], format="%Y%m%d")
                    buffer.append(rows)
                    report.filled_days.append(day)
                if len(buffer) >= _FLUSH_EVERY_DAYS:
                    self._append_basic(buffer)
                    buffer = []
                if on_progress:
                    on_progress(day, index, len(todo))
        except TushareQueryFailed as exc:
            report.stopped_reason = str(exc)
        finally:
            self._append_basic(buffer)
        return report

    def _append_basic(self, frames: List[pd.DataFrame]) -> None:
        if not frames:
            return
        batch = pd.concat(frames, ignore_index=True)
        for year, rows in batch.groupby(batch["trade_date"].dt.year):
            path = self._basic_path(int(year))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                rows = pd.concat([pd.read_parquet(path), rows], ignore_index=True)
            rows = rows.drop_duplicates(["ts_code", "trade_date"], keep="last").sort_values(["trade_date", "ts_code"])
            rows.to_parquet(path, index=False, compression="zstd")

    def load_basic(self, start: date, end: date, ts_codes: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """读 [start, end] 的每日指标（市值单位万元，换手率单位 %，与 tushare 相同）。"""
        frames = [pd.read_parquet(self._basic_path(y)) for y in range(start.year, end.year + 1) if self._basic_path(y).exists()]
        if not frames:
            return pd.DataFrame(columns=BASIC_COLUMNS)
        data = pd.concat(frames, ignore_index=True)
        data = data[(data["trade_date"] >= pd.Timestamp(start)) & (data["trade_date"] <= pd.Timestamp(end))]
        if ts_codes is not None:
            data = data[data["ts_code"].isin(set(ts_codes))]
        return data.reset_index(drop=True)

    # ---- 指数 ----

    def _index_path(self, ts_code: str) -> Path:
        return self.root / "index" / f"{ts_code}.parquet"

    def refresh_index(self, ts_code: str, until: Optional[date] = None) -> int:
        """指数日线按区间一次拉取（一只指数十年约 2500 行，一次调用即可），只补缺的尾部。"""
        until = until or self.latest_trading_day()
        path = self._index_path(ts_code)
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["trade_date", "close"])
        ranges = []
        if existing.empty:
            ranges.append((INDEX_START, until))
        else:
            first, last = existing["trade_date"].min().date(), existing["trade_date"].max().date()
            # 指数比个股往前多存到 2005 年：择时信号（如 RSRS 要 600 多天预热）需要更长的历史，一只指数只多一次调用
            if first > INDEX_START + timedelta(days=15):
                ranges.append((INDEX_START, first - timedelta(days=1)))
            if last + timedelta(days=1) <= until:
                ranges.append((last + timedelta(days=1), until))
        frames = []
        for begin, end in ranges:
            part = self.runner(lambda pro: pro.index_daily(ts_code=ts_code, start_date=_compact(begin), end_date=_compact(end)))
            if part is not None and not part.empty:
                frames.append(part)
        if not frames:
            return 0
        fetched = pd.concat(frames, ignore_index=True)
        fetched = fetched[["trade_date", "open", "high", "low", "close", "vol", "amount"]].copy()
        fetched["trade_date"] = pd.to_datetime(fetched["trade_date"], format="%Y%m%d")
        merged = fetched if existing.empty else pd.concat([existing, fetched], ignore_index=True)
        merged = merged.drop_duplicates("trade_date", keep="last").sort_values("trade_date")
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False, compression="zstd")
        return len(fetched)

    def load_index(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        path = self._index_path(ts_code)
        if not path.exists():
            return pd.DataFrame(columns=["trade_date", "close"])
        data = pd.read_parquet(path)
        return data[(data["trade_date"] >= pd.Timestamp(start)) & (data["trade_date"] <= pd.Timestamp(end))].reset_index(drop=True)

    # ---- 指数历史成分股（ADR-0052）----

    def _members_path(self, index_code: str) -> Path:
        return self.root / "index" / f"{index_code}_members.parquet"

    def refresh_index_members(self, index_code: str, until: Optional[date] = None) -> int:
        """历史成分股快照（tushare index_weight，每月一到两份）。按半年分批拉、只补缺的部分：
        一次调用最多返回 7000 行，按整年拉时一年 24 份快照会超出，最早那份被截断（实测）。"""
        until = until or self.latest_trading_day()
        path = self._members_path(index_code)
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["con_code", "trade_date", "weight"])
        # 要补的区间：库里最早一份之前（起点提前时）和最新一份之后
        ranges = [(self.start, until)] if existing.empty else [
            (self.start, existing["trade_date"].min().date() - timedelta(days=1)),
            (existing["trade_date"].max().date() + timedelta(days=1), until)]
        frames = []
        for begin, stop in ranges:
            while begin <= stop:
                end = min(date(begin.year, 6, 30) if begin.month <= 6 else date(begin.year, 12, 31), stop)
                part = self.runner(lambda pro, b=begin, e=end: pro.index_weight(index_code=index_code, start_date=_compact(b), end_date=_compact(e)))
                if part is not None and not part.empty:
                    frames.append(part[["con_code", "trade_date", "weight"]])
                begin = end + timedelta(days=1)
        if not frames:
            return 0
        fetched = pd.concat(frames, ignore_index=True)
        fetched["trade_date"] = pd.to_datetime(fetched["trade_date"], format="%Y%m%d")
        merged = fetched if existing.empty else pd.concat([existing, fetched], ignore_index=True)
        merged = merged.drop_duplicates(["con_code", "trade_date"], keep="last").sort_values(["trade_date", "con_code"])
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False, compression="zstd")
        return len(fetched)

    def load_index_members(self, index_code: str, full_size: int = 300) -> pd.DataFrame:
        """只返回完整的快照（成员数达到指数规模），不完整的快照丢掉，不拿残缺名单当成分股。"""
        path = self._members_path(index_code)
        if not path.exists():
            return pd.DataFrame(columns=["con_code", "trade_date", "weight"])
        data = pd.read_parquet(path)
        counts = data.groupby("trade_date")["con_code"].transform("nunique")
        return data[counts >= full_size].reset_index(drop=True)

    # ---- 状态 ----

    def status(self, today: Optional[date] = None) -> dict:
        latest_trading = self.latest_trading_day(today)
        stored = self.stored_days()
        missing = self.missing_days(latest_trading)
        return {
            "latest_stored": stored[-1].isoformat() if stored else None,
            "latest_trading_day": latest_trading.isoformat(),
            "stored_days": len(stored),
            "missing_days": len(missing),
        }
