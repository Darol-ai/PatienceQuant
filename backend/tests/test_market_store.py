from datetime import date

import pandas as pd
import pytest

from app.data.market_store import AShareMarketStore
from app.data.tushare_client import TushareQuotaExceeded

OPEN_DAYS = ["20240102", "20240103", "20240104", "20240105", "20240108"]


class FakePro:
    def __init__(self, quota_after_calls=None, empty_days=()):
        self.calls = []
        self.quota_after_calls = quota_after_calls
        self.empty_days = set(empty_days)

    def _count(self, name):
        self.calls.append(name)
        if self.quota_after_calls is not None and len(self.calls) > self.quota_after_calls:
            raise TushareQuotaExceeded("抱歉，您访问接口频率超限")

    def trade_cal(self, exchange, start_date, end_date):
        self._count("trade_cal")
        days = pd.date_range("2024-01-01", "2024-12-31").strftime("%Y%m%d")
        return pd.DataFrame({"cal_date": days, "is_open": [1 if d in OPEN_DAYS else 0 for d in days]})

    def daily(self, trade_date):
        self._count("daily")
        if trade_date in self.empty_days:
            return pd.DataFrame()
        close = 10.0 + OPEN_DAYS.index(trade_date)
        return pd.DataFrame({
            "ts_code": ["000001.SZ", "600519.SH"], "trade_date": [trade_date] * 2,
            "open": [close, 100.0], "high": [close, 100.0], "low": [close, 100.0], "close": [close, 100.0],
            "vol": [1.0, 2.0], "amount": [3.0, 4.0],
        })

    def adj_factor(self, trade_date):
        self._count("adj_factor")
        # 000001 在 0105 除权：因子从 1 变成 2
        factor = 2.0 if trade_date >= "20240105" else 1.0
        return pd.DataFrame({"ts_code": ["000001.SZ", "600519.SH"], "adj_factor": [factor, 1.0]})


def make_store(tmp_path, pro):
    return AShareMarketStore(root=tmp_path, runner=lambda fn: fn(pro), start=date(2024, 1, 1))


def test_backfill_fills_every_open_day_and_resumes_without_refetching(tmp_path):
    pro = FakePro()
    store = make_store(tmp_path, pro)

    report = store.backfill(until=date(2024, 1, 8))

    assert report.filled_days == [pd.Timestamp(d).date() for d in OPEN_DAYS]
    assert report.stopped_reason is None
    assert store.stored_days() == report.filled_days
    calls_after_first = len(pro.calls)
    again = store.backfill(until=date(2024, 1, 8))
    assert again.requested_days == 0
    assert len(pro.calls) == calls_after_first


def test_quota_error_stops_backfill_but_keeps_days_already_fetched(tmp_path):
    # trade_cal 1 次 + 第一天 daily/adj_factor 2 次 + 第二天 2 次，之后超限
    pro = FakePro(quota_after_calls=5)
    store = make_store(tmp_path, pro)

    report = store.backfill(until=date(2024, 1, 8))

    assert "超限" in report.stopped_reason
    assert report.filled_days == [date(2024, 1, 2), date(2024, 1, 3)]
    assert store.stored_days() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert store.missing_days(date(2024, 1, 8)) == [date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8)]


def test_empty_day_is_not_stored_and_is_retried_next_time(tmp_path):
    pro = FakePro(empty_days={"20240108"})
    store = make_store(tmp_path, pro)

    report = store.backfill(until=date(2024, 1, 8))

    assert report.empty_days == [date(2024, 1, 8)]
    assert date(2024, 1, 8) in store.missing_days(date(2024, 1, 8))


def test_load_daily_forward_adjusts_to_last_day_of_the_range(tmp_path):
    store = make_store(tmp_path, FakePro())
    store.backfill(until=date(2024, 1, 8))

    raw = store.load_daily(date(2024, 1, 2), date(2024, 1, 8), ["000001.SZ"], adjust="none")
    qfq = store.load_daily(date(2024, 1, 2), date(2024, 1, 8), ["000001.SZ"])

    assert raw["close"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
    # 除权前的价格按 1/2 缩放，除权后的不变
    assert qfq["close"].tolist() == pytest.approx([5.0, 5.5, 6.0, 13.0, 14.0])


def test_since_limits_backfill_to_recent_days(tmp_path):
    store = make_store(tmp_path, FakePro())

    report = store.backfill(until=date(2024, 1, 8), since=date(2024, 1, 5))

    assert report.filled_days == [date(2024, 1, 5), date(2024, 1, 8)]


class FakeProWithIndex(FakePro):
    def index_daily(self, ts_code, start_date, end_date):
        self._count("index_daily")
        days = [d for d in OPEN_DAYS if start_date <= d <= end_date]
        return pd.DataFrame({"trade_date": days, "open": 1.0, "high": 1.0, "low": 1.0,
                             "close": [3000.0 + i for i in range(len(days))], "vol": 1.0, "amount": 1.0})


def test_real_mode_service_reads_prices_and_benchmark_only_from_the_store(tmp_path, monkeypatch):
    from app.config import get_settings
    from app.data import service as service_module

    store = make_store(tmp_path, FakeProWithIndex())
    store.backfill(until=date(2024, 1, 8))
    store.refresh_index("000300.SH", until=date(2024, 1, 8))
    monkeypatch.setattr(get_settings(), "data_mode", "real")
    monkeypatch.setattr(service_module, "get_market_store", lambda: store)
    data = service_module.MarketDataService(db=None)

    prices = data.prices(["000001", "600519", "300999"], date(2024, 1, 2), date(2024, 1, 8))

    assert set(prices["symbol"]) == {"000001", "600519"}
    assert data.last_price_fetch_failures == ["300999"]  # 库里没有的股票如实报缺，不拿demo顶替
    assert prices[prices.symbol == "000001"]["close"].tolist() == pytest.approx([5.0, 5.5, 6.0, 13.0, 14.0])
    assert set(prices["data_mode"]) == {"real"}
    bench = data.benchmark(date(2024, 1, 2), date(2024, 1, 8))
    assert bench["close"].tolist() == [3000.0, 3001.0, 3002.0, 3003.0, 3004.0]
