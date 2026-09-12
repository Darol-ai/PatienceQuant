"""V3 方案第 4 节：持仓状态机——两日确认计数器、B/H 基准价、冷静期。"""


class ConsecutiveDayCounter:
    """两个模型退出条件共用的"连续两日确认"计数器。

    条件失败、或调用方显式 reset()（对应"模型或校准器版本切换时归零"），
    都会清零重新计数；两天不要求来自同一次调用之外的任何额外状态。
    """

    def __init__(self) -> None:
        self.count = 0

    def update(self, condition_met: bool) -> bool:
        """记录今天是否满足条件，返回是否已连续两天确认。"""
        self.count = self.count + 1 if condition_met else 0
        return self.count >= 2

    def reset(self) -> None:
        self.count = 0


class PositionState:
    """一轮持仓的状态：B（建仓基准价）、H（区间最高有效收盘价）、持有天数、
    两个模型退出条件各自的两日确认计数器。部分加减仓不重置 B、H。
    """

    def __init__(self, b: float) -> None:
        self.b = b
        self.h = b
        self.holding_days = 0
        self.risk_exit_counter = ConsecutiveDayCounter()
        self.take_profit_counter = ConsecutiveDayCounter()

    def record_close(self, close: float) -> None:
        """每个交易日调用一次：推高 H（不回落），累计持有天数。"""
        self.h = max(self.h, close)
        self.holding_days += 1


class CooldownTracker:
    """主动完全清仓后的冷静期：20 个完整交易日内禁止对该股票新增买入。"""

    COOLDOWN_DAYS = 20

    def __init__(self) -> None:
        self.days_remaining = 0

    def start(self) -> None:
        """清仓下一交易日起开始计时。"""
        self.days_remaining = self.COOLDOWN_DAYS

    def advance_day(self) -> None:
        if self.days_remaining > 0:
            self.days_remaining -= 1

    def is_locked_out(self) -> bool:
        return self.days_remaining > 0
