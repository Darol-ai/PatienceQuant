"""把 BaoStock `query_history_k_data_plus` 的原始行转换成本项目的规范字段。

请求字段固定为：date,code,open,high,low,close,volume,amount,turn,pctChg
（pctChg 冗余、可从 OHLC 派生，不纳入规范字段）。BaoStock 所有数值字段都是字符串。
"""

from collections.abc import Sequence

_EXCHANGE_SUFFIX = {"sh": ".SH", "sz": ".SZ"}


def _to_vt_symbol(bao_code: str) -> str:
    exchange, code = bao_code.split(".")
    return f"{code}{_EXCHANGE_SUFFIX[exchange]}"


def _zero_if_suspended(value: str, is_suspended: bool, parser, field_name: str, code: str, date: str):
    """停牌日（tradestatus=0）BaoStock 对 volume/amount/turn 的留空方式并不
    统一——早年数据填 '0'，近年数据整行留空，两种都代表"停牌日成交额记零"
    （V3 方案 2.2 节）。tradestatus=1 却缺这些字段是真正的未知缺失，直接
    报错，不能悄悄记零。
    """
    if value == "":
        if not is_suspended:
            raise ValueError(
                f"{code} {date}：正常交易日却缺 {field_name}，这是未知缺失，不能记零"
            )
        return 0
    return parser(value)


def normalize_daily_bars(rows: Sequence[Sequence[str]]) -> list[dict]:
    """rows 需按 date,code,open,high,low,close,volume,amount,turn,pctChg,
    tradestatus 的顺序。
    """
    bars = []
    for date, code, open_, high, low, close, volume, amount, turn, _pct_chg, tradestatus in rows:
        is_suspended = tradestatus == "0"
        bars.append(
            {
                "symbol": _to_vt_symbol(code),
                "date": date,
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": _zero_if_suspended(volume, is_suspended, int, "volume", code, date),
                "amount": _zero_if_suspended(amount, is_suspended, float, "amount", code, date),
                "turnover_rate": _zero_if_suspended(turn, is_suspended, float, "turn", code, date),
                "is_suspended": is_suspended,
            }
        )
    return bars
