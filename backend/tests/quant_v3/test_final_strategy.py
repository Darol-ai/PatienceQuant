from datetime import date

import pytest

from app.quant_v3.final_strategy import (
    build_final_strategy,
    final_strategy_history,
    latest_trading_day_on_or_before,
    validate_backtest_date_range,
)


def test_date_range_within_supported_years_is_accepted():
    validate_backtest_date_range(date(2020, 1, 1), date(2023, 12, 31))  # 不应抛异常


def test_date_range_before_supported_years_is_rejected():
    with pytest.raises(ValueError):
        validate_backtest_date_range(date(2016, 1, 1), date(2020, 12, 31))


def test_date_range_after_supported_years_is_rejected():
    with pytest.raises(ValueError):
        validate_backtest_date_range(date(2024, 1, 1), date(2027, 12, 31))


def test_2026_fold_added_for_paper_trading_is_within_supported_range():
    """模拟盘默认按"今天"驱动，2026年补训了一折模型（和2019-2025用同一套
    滚动训练规则），这个年份不该再被当成"超出范围"拒绝。"""
    validate_backtest_date_range(date(2026, 1, 1), date(2026, 12, 31))  # 不应抛异常


def test_build_final_strategy_returns_a_usable_strategy_instance():
    strategy = build_final_strategy()

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert result.weights  # 2024年有对应模型，应该能算出非空权重


def test_build_final_strategy_returns_a_fresh_instance_each_call():
    """不能把有状态的策略对象缓存成进程级单例——两个不相关的请求(尤其是
    并发请求)会共享并互相污染像粘性持仓计数器这样的可变状态。允许共享
    的只是底下无状态的信号源(模型/历史索引)，策略实例本身每次都要是新的。
    """
    first = build_final_strategy()
    second = build_final_strategy()

    assert first is not second
    assert first._sticky_hold_remaining is not second._sticky_hold_remaining


def test_latest_trading_day_on_or_before_returns_the_same_date_when_it_has_data():
    history = final_strategy_history()
    a_real_trading_day = date(2024, 1, 31)
    assert a_real_trading_day.isoformat() in set(history["date"])

    assert latest_trading_day_on_or_before(a_real_trading_day) == a_real_trading_day


def test_latest_trading_day_on_or_before_snaps_a_weekend_date_back_to_friday():
    """模拟盘默认按字面`date.today()`调仓——周末/节假日不是交易日，信号源
    按精确日期索引查不到那天的行情，如果不在这里先做兜底，会对每支股票
    都悄悄返回None（不报错，只是生成一个全零权重的"调仓"），看起来像是
    "今天没有交易机会"，实际是日期没对齐的bug。"""
    a_saturday = date(2024, 2, 3)  # 2024-02-02 是周五
    assert latest_trading_day_on_or_before(a_saturday) == date(2024, 2, 2)


def test_latest_trading_day_on_or_before_rejects_a_date_before_all_history():
    with pytest.raises(ValueError):
        latest_trading_day_on_or_before(date(2005, 1, 1))


def test_build_final_strategy_produces_real_weights_on_a_weekend_as_of():
    """回归测试：这个具体场景之前会悄悄退化成空权重，而不是报错或者用
    最近一个交易日的行情——`PaperTradingService`在这一层之前必须先把
    `as_of`落到最近的真实交易日上，策略本身不做这个兜底。"""
    strategy = build_final_strategy()
    snapped = latest_trading_day_on_or_before(date(2024, 2, 3))

    result = strategy.generate_weights(snapped, symbols=[])

    assert result.weights
