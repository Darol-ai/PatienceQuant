"""V3 方案 2.2 节"沿用的资格条件"——只实现其中可以直接从行情数据验证的
流动性子集（决策日正常交易 + 60日日均成交额≥1亿 + 60日内至少50日有
成交），不实现需要额外抓取财务报表数据的基本面子集（净利润/净资产/
审计意见/分红/ROE，按红利/成长/周期分组各有不同门槛）——那部分保持
`engineering_only`，如实标注未实现，不是简化成摆设（同 V3 方案第9节
"qualification_mode=engineering_only 只做账户工程验收，不报告完整策略
的正式历史绩效"的原则）。

不知道就不能记为合格：历史不足60天时无法验证，按不合格处理。
"""
from collections.abc import Sequence

_WINDOW = 60
_MIN_AVG_AMOUNT = 1.0e8
_MIN_TRADING_DAYS = 50


def is_liquidity_qualified(bars: Sequence[dict], t_index: int) -> bool:
    if t_index - (_WINDOW - 1) < 0:
        return False
    if bars[t_index]["is_suspended"]:
        return False

    window = bars[t_index - (_WINDOW - 1) : t_index + 1]
    avg_amount = sum(bar["amount"] for bar in window) / _WINDOW
    if avg_amount < _MIN_AVG_AMOUNT:
        return False

    trading_days = sum(1 for bar in window if not bar["is_suspended"])
    if trading_days < _MIN_TRADING_DAYS:
        return False

    return True
