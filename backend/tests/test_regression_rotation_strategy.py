from datetime import date

import pytest

from app.strategies.regression_rotation_strategy import RegressionRotationStrategy

_UNIVERSE = [{"symbol": f"S{i}", "group": "综合"} for i in range(1, 5)]
_BUDGETS = {"综合": 1.0}


def test_top_k_by_predicted_score_gets_full_exposure_when_trend_is_up():
    scores = {"S1": 0.10, "S2": 0.05, "S3": -0.02, "S4": -0.10}
    strategy = RegressionRotationStrategy(
        signal_source=lambda symbol, as_of: scores[symbol],
        universe=_UNIVERSE, group_budgets=_BUDGETS, top_k_ratio=0.5,
        trend_signal=lambda as_of: 1.0,
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert set(result.weights) == {"S1", "S2"}
    assert result.weights["S1"] == pytest.approx(0.50)  # 选中2只，预算全投进这2只，不留现金


def test_trend_signal_scales_down_all_weights_in_risk_off():
    scores = {"S1": 0.10, "S2": 0.05, "S3": -0.02, "S4": -0.10}
    strategy = RegressionRotationStrategy(
        signal_source=lambda symbol, as_of: scores[symbol],
        universe=_UNIVERSE, group_budgets=_BUDGETS, top_k_ratio=0.5,
        trend_signal=lambda as_of: 0.3,
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert result.weights["S1"] == pytest.approx(0.50 * 0.3)


def test_score_weighted_gives_higher_score_more_weight():
    scores = {"S1": 0.30, "S2": 0.10, "S3": 0.05, "S4": -0.10}
    strategy = RegressionRotationStrategy(
        signal_source=lambda symbol, as_of: scores[symbol],
        universe=_UNIVERSE, group_budgets=_BUDGETS, top_k_ratio=0.75,
        trend_signal=lambda as_of: 1.0, score_weighted=True,
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert set(result.weights) == {"S1", "S2", "S3"}
    assert result.weights["S1"] > result.weights["S2"] > result.weights["S3"]
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_missing_score_is_excluded_from_ranking():
    strategy = RegressionRotationStrategy(
        signal_source=lambda symbol, as_of: None,
        universe=_UNIVERSE, group_budgets=_BUDGETS, top_k_ratio=0.5,
        trend_signal=lambda as_of: 1.0,
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert result.weights == {}


def test_momentum_override_force_includes_strongest_momentum_stocks():
    """自由探索阶段（docs/adr/0022）：诊断出模型在近乎普涨行情里会漏判
    真正的动量黑马（比如2019年模型给当年最大涨幅股打了很低的分）——加
    一层动量兜底，不管模型打分如何，每次调仓强制把动量最强的几支也
    纳入候选，作为模型选股之外的安全网。这里 S4 模型打分最低本该被
    排除，但它的动量分数(momentum_scores)最高，应该被强制纳入。"""
    scores = {"S1": 0.10, "S2": 0.05, "S3": 0.02, "S4": -0.20}  # S4模型认为最差
    momentum_scores = {"S1": 0.01, "S2": 0.02, "S3": 0.03, "S4": 0.50}  # 但S4动量最强
    strategy = RegressionRotationStrategy(
        signal_source=lambda symbol, as_of: scores[symbol],
        universe=_UNIVERSE, group_budgets=_BUDGETS, top_k_ratio=0.5,
        trend_signal=lambda as_of: 1.0,
        momentum_override_source=lambda symbol, as_of: momentum_scores[symbol],
        momentum_override_count=1,
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert "S4" in result.weights  # 模型打分最差，但动量最强，被强制纳入
    assert set(result.weights) == {"S1", "S2", "S4"}  # top_k=0.5选S1,S2 + 动量兜底加S4


def test_momentum_override_sticky_holding_survives_a_rank_drop():
    """自由探索阶段（docs/adr/0024）：诊断出2019年动量兜底"时进时出"——
    某支股票某月靠动量兜底进场，下个月动量排名掉出前列就被踢出，没能
    完整拿到持续上涨的涨幅。加粘性：一旦靠动量兜底进场，至少强制持有
    momentum_override_min_hold_months 个月，不因排名下降就提前踢出。"""
    momentum_scores_by_month = {
        1: {"S1": 0.01, "S2": 0.02, "S3": 0.03, "S4": 0.50},  # S4动量最强，进场
        2: {"S1": 0.01, "S2": 0.02, "S3": 0.03, "S4": -0.90},  # S4排名垫底，但粘性期内
        3: {"S1": 0.01, "S2": 0.02, "S3": 0.03, "S4": -0.90},  # 粘性到期后应该被踢出
    }
    model_scores = {"S1": 0.10, "S2": 0.05, "S3": 0.02, "S4": -0.20}  # 模型一直认为S4最差
    strategy = RegressionRotationStrategy(
        signal_source=lambda symbol, as_of: model_scores[symbol],
        universe=_UNIVERSE, group_budgets=_BUDGETS, top_k_ratio=0.5,
        trend_signal=lambda as_of: 1.0,
        momentum_override_source=lambda symbol, as_of: momentum_scores_by_month[as_of.month][symbol],
        momentum_override_count=1,
        momentum_override_min_hold_months=2,
    )

    month1 = strategy.generate_weights(date(2024, 1, 31), symbols=[])
    month2 = strategy.generate_weights(date(2024, 2, 29), symbols=[])
    month3 = strategy.generate_weights(date(2024, 3, 31), symbols=[])

    assert "S4" in month1.weights  # 第1个月靠动量兜底进场
    assert "S4" in month2.weights  # 第2个月动量排名垫底，但粘性期(2个月)内继续持有
    assert "S4" not in month3.weights  # 第3个月粘性到期，模型/动量都不支持，被踢出


def test_momentum_override_boost_gives_extra_weight_to_forced_stock():
    """自由探索阶段（docs/adr/0025）：动量兜底选出的黑马和模型选出的股票
    等权重摊薄在一起，稀释了黑马本该有的贡献——momentum_override_boost
    >1时，被动量兜底强制纳入的股票单独按倍数加权，不是简单等权。"""
    scores = {"S1": 0.10, "S2": 0.05, "S3": 0.02, "S4": -0.20}
    momentum_scores = {"S1": 0.01, "S2": 0.02, "S3": 0.03, "S4": 0.50}
    strategy = RegressionRotationStrategy(
        signal_source=lambda symbol, as_of: scores[symbol],
        universe=_UNIVERSE, group_budgets=_BUDGETS, top_k_ratio=0.5,
        trend_signal=lambda as_of: 1.0,
        momentum_override_source=lambda symbol, as_of: momentum_scores[symbol],
        momentum_override_count=1, momentum_override_boost=2.0,
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    # S1,S2(模型选中,各1单位) + S4(动量兜底,2单位) = 4单位，S4权重=0.5，S1/S2各0.25
    assert result.weights["S4"] == pytest.approx(0.50)
    assert result.weights["S1"] == pytest.approx(0.25)


def test_realized_rally_trigger_forces_full_coverage_regardless_of_model_score():
    """自由探索阶段（docs/adr/0031）：ADR-0018/0020已经证明"用模型自己的
    打分算广度"是bug产物，不是真实机制。这里换成基于大盘指数已实现动量
    的独立信号(不依赖模型判断)——当月触发时直接全覆盖，不管top_k_ratio
    平时设多严格。"""
    scores = {"S1": 0.10, "S2": 0.05, "S3": -0.02, "S4": -0.10}
    strategy = RegressionRotationStrategy(
        signal_source=lambda symbol, as_of: scores[symbol],
        universe=_UNIVERSE, group_budgets=_BUDGETS, top_k_ratio=0.25,
        trend_signal=lambda as_of: 1.0,
        full_coverage_trigger=lambda as_of: True,
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert set(result.weights) == {"S1", "S2", "S3", "S4"}  # 触发时全覆盖，不管top_k_ratio多严格


def test_realized_rally_trigger_off_keeps_normal_top_k_ratio():
    scores = {"S1": 0.10, "S2": 0.05, "S3": -0.02, "S4": -0.10}
    strategy = RegressionRotationStrategy(
        signal_source=lambda symbol, as_of: scores[symbol],
        universe=_UNIVERSE, group_budgets=_BUDGETS, top_k_ratio=0.25,
        trend_signal=lambda as_of: 1.0,
        full_coverage_trigger=lambda as_of: False,
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert set(result.weights) == {"S1"}  # 未触发，维持0.25的严格选择性(4只选1只)


def test_disqualified_stock_is_excluded_even_with_a_strong_score():
    """自由探索阶段（docs/adr/0036）：V3方案资格退出优先级最高——不管模型
    打分或动量兜底怎么判断，不合格的股票都不能入选，也不能被动量兜底
    强制纳入。"""
    scores = {"S1": 0.30, "S2": 0.20, "S3": 0.10, "S4": 0.05}
    qualified = {"S1": False, "S2": True, "S3": True, "S4": True}  # S1打分最高但不合格
    strategy = RegressionRotationStrategy(
        signal_source=lambda symbol, as_of: scores[symbol],
        universe=_UNIVERSE, group_budgets=_BUDGETS, top_k_ratio=0.5,
        trend_signal=lambda as_of: 1.0,
        qualification_source=lambda symbol, as_of: qualified[symbol],
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert "S1" not in result.weights
    assert set(result.weights) == {"S2", "S3"}  # 剩余3只合格候选选前一半


def test_disqualified_stock_is_not_force_included_by_momentum_override():
    scores = {"S1": 0.10, "S2": 0.05, "S3": 0.02, "S4": -0.20}
    momentum_scores = {"S1": 0.01, "S2": 0.02, "S3": 0.03, "S4": 0.50}  # S4动量最强
    qualified = {"S1": True, "S2": True, "S3": True, "S4": False}  # 但S4不合格
    strategy = RegressionRotationStrategy(
        signal_source=lambda symbol, as_of: scores[symbol],
        universe=_UNIVERSE, group_budgets=_BUDGETS, top_k_ratio=0.5,
        trend_signal=lambda as_of: 1.0,
        momentum_override_source=lambda symbol, as_of: momentum_scores[symbol],
        momentum_override_count=1,
        qualification_source=lambda symbol, as_of: qualified[symbol],
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert "S4" not in result.weights


def test_broad_rally_widens_top_k_ratio_to_full_coverage():
    """自由探索阶段（docs/adr/0018）：ADR-0016 诊断出2019/2020这类近乎
    普涨的行情里，固定 top_k_ratio 会把大部分涨幅拒之门外——用模型自己
    的打分算"广度"（正分数股票占比），广度超过阈值时当月临时把
    top_k_ratio 拉到1.0（全覆盖），不是永久放弃选股，只在模型自己判断
    "这个月几乎全员看涨"时才这样做。"""
    scores = {"S1": 0.30, "S2": 0.20, "S3": 0.10, "S4": 0.05}  # 全部为正，广度100%
    strategy = RegressionRotationStrategy(
        signal_source=lambda symbol, as_of: scores[symbol],
        universe=_UNIVERSE, group_budgets=_BUDGETS, top_k_ratio=0.5,
        trend_signal=lambda as_of: 1.0, breadth_threshold=0.8,
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert set(result.weights) == {"S1", "S2", "S3", "S4"}  # 广度100%>=0.8门槛，全覆盖不是只选一半


def test_narrow_breadth_keeps_selective_top_k_ratio():
    scores = {"S1": 0.30, "S2": -0.10, "S3": -0.20, "S4": -0.30}  # 只有1/4为正，广度25%
    strategy = RegressionRotationStrategy(
        signal_source=lambda symbol, as_of: scores[symbol],
        universe=_UNIVERSE, group_budgets=_BUDGETS, top_k_ratio=0.5,
        trend_signal=lambda as_of: 1.0, breadth_threshold=0.8,
    )

    result = strategy.generate_weights(date(2024, 1, 31), symbols=[])

    assert set(result.weights) == {"S1", "S2"}  # 广度25%<0.8门槛，维持原本的top_k_ratio=0.5选择性
