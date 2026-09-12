from datetime import date

import pandas as pd
import pytest

from app.strategies.v3_strategy import V3Strategy, neutral_signal


def test_neutral_signal_never_approves_anything():
    """还没有真正的模型时，占位信号源不能伪造出会买入的结果——组内所有
    股票的打分完全一样（没有区分度），排不出高低，不硬凑排名。"""
    strategy = V3Strategy()

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert result.weights == {}


def test_negative_but_distinguishable_scores_still_get_ranked():
    """V3 原来的"score>0"门槛在 UP 事件天生比 DOWN 事件少见（标签边界不
    对称）的情况下会有隐性偏见——公开的 Top-K 策略（Qlib/BigQuant）不看
    绝对分数正负，只要打分能分出相对高低就照样满仓排序、持续持仓。这里
    验证：全组打分都是负的，但彼此不同，也应该照样选出最不差的那批。"""
    scores = {
        "601288.SH": -0.05, "601398.SH": -0.10, "600900.SH": -0.20, "600377.SH": -0.30,
    }
    strategy = V3Strategy(signal_source=lambda symbol, as_of: {
        "601288.SH": (0.10, 0.15), "601398.SH": (0.10, 0.20),
        "600900.SH": (0.10, 0.30), "600377.SH": (0.10, 0.40),
    }.get(symbol, (0.0, 0.0)))

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    dividend_group_weights = {s: w for s, w in result.weights.items() if s in scores}
    assert set(dividend_group_weights) == {"601288.SH", "601398.SH"}  # 红利组 4 选 2，选分数最高（最不负）的两个


_DISTINCT_P_UP = {
    "601288.SH": 0.9, "601398.SH": 0.8, "600900.SH": 0.7, "600377.SH": 0.6,  # 红利
    "300308.SZ": 0.9, "300124.SZ": 0.8, "600406.SH": 0.7,                    # 成长
    "601899.SH": 0.9, "601088.SH": 0.8, "600309.SH": 0.7,                    # 周期
}


def _distinct_signal(symbol: str, as_of: date) -> tuple:
    return _DISTINCT_P_UP[symbol], 0.1


def test_top_k_ratio_selects_half_of_each_group_by_score():
    """每支股票打分不同（有真实区分度）时，每组按 top_k_ratio=0.5 选出打分
    最高的一半。等权预算的分母仍是组内全部股票数，所以入选的每只权重还是
    10%（4/3/3 只对应 40%/30%/30%）。"""
    strategy = V3Strategy(signal_source=_distinct_signal)

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert set(result.weights) == {
        "601288.SH", "601398.SH",  # 红利组 4 选 2（打分最高的两个）
        "300308.SZ", "300124.SZ",  # 成长组 3 选 2
        "601899.SH", "601088.SH",  # 周期组 3 选 2
    }
    for weight in result.weights.values():
        assert weight == pytest.approx(0.10)


def test_top_k_ratio_controls_how_many_stocks_are_selected_per_group():
    narrow = V3Strategy(signal_source=_distinct_signal, top_k_ratio=0.25)
    wide = V3Strategy(signal_source=_distinct_signal, top_k_ratio=1.0)

    narrow_result = narrow.generate_weights(date(2024, 1, 31), symbols=[])
    wide_result = wide.generate_weights(date(2024, 1, 31), symbols=[])

    assert len(narrow_result.weights) < len(wide_result.weights)
    assert len(wide_result.weights) == 10


def test_hard_stop_loss_forces_an_exit_on_daily_close():
    strategy = V3Strategy()
    strategy.notify_fill("601288.SH", "BUY", price=100.0, quantity=100, trade_date=date(2024, 1, 2))

    risk = strategy.on_daily_close(date(2024, 1, 3), pd.Series({"601288.SH": 84.0}))

    assert risk.exits == {"601288.SH": "HARD_STOP_LOSS"}


def test_group_budgets_is_overridable_for_non_official_universes():
    """分散化候选池实验(见相关 ADR)只有一个"综合"组，预算100%，不是官方
    40/30/30——group_budgets 必须真的传下去影响权重，不是只加了个没用上
    的构造参数。"""
    universe = [{"symbol": "X1", "group": "综合"}, {"symbol": "X2", "group": "综合"}]
    strategy = V3Strategy(
        signal_source=lambda symbol, as_of: {"X1": (0.9, 0.1), "X2": (0.1, 0.9)}[symbol],
        universe=universe, group_budgets={"综合": 1.0}, top_k_ratio=0.5,
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert result.weights == {"X1": pytest.approx(0.50)}


def test_hard_stop_loss_threshold_is_overridable_for_sensitivity_experiments():
    """跌 18% 用 V3 原文 -15% 门槛会触发止损，用更宽松的 -20% 门槛（研究
    "止损参数是否对趋势行情过敏"）就不会——验证阈值真的从 V3Strategy 传
    到了 risk_exits，不是只加了个没用上的构造参数。"""
    strategy = V3Strategy(hard_stop_loss_threshold=-0.20)
    strategy.notify_fill("601288.SH", "BUY", price=100.0, quantity=100, trade_date=date(2024, 1, 2))

    risk = strategy.on_daily_close(date(2024, 1, 3), pd.Series({"601288.SH": 82.0}))

    assert risk.exits == {}


def test_holding_is_preserved_when_a_stock_no_longer_passes_entry_this_month():
    """V3 方案第 4 节："持有：没有触发任何退出；月中持有；月末未达买入条件
    也不因此自动清仓，不补仓"——入场排序只管新买入，已经持有的仓位只能被
    真正的退出条件（on_daily_close）平掉，普通调仓不能把它清零。"""
    strategy = V3Strategy(signal_source=lambda symbol, as_of: (0.0, 0.0))  # 这个月零信号，谁都不批准
    strategy.notify_fill("601398.SH", "BUY", price=100.0, quantity=100, trade_date=date(2024, 1, 2))

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert result.weights["601398.SH"] == pytest.approx(0.10)


def test_selling_starts_a_cooldown_that_blocks_the_next_entry():
    strategy = V3Strategy(signal_source=lambda symbol, as_of: (0.90, 0.10))
    strategy.notify_fill("601288.SH", "BUY", price=100.0, quantity=100, trade_date=date(2024, 1, 2))
    strategy.notify_fill("601288.SH", "SELL", price=84.0, quantity=100, trade_date=date(2024, 1, 3))

    risk = strategy.on_daily_close(date(2024, 1, 4), pd.Series({"601288.SH": 84.0}))
    assert "601288.SH" in risk.blocked_symbols

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])
    row = result.ranking[result.ranking.symbol == "601288.SH"].iloc[0]
    assert row["action"] == "COOLDOWN"
    assert "601288.SH" not in result.weights
