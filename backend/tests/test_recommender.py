import time

import pytest
from fastapi.testclient import TestClient

from app.ai.recommender import numbers_are_grounded, rank, template_explanation
from app.main import app


def card(sid, name, annual, mdd, sharpe, freq="monthly", market="a_share", status="ready"):
    return {"strategy_id": sid, "name": name, "description": "", "market": market, "status": status,
            "universe": "csi300", "rebalance_frequency": freq, "average_exposure": 1.0,
            "standard": {"start_date": "2019-01-01", "end_date": "2025-12-31"},
            "metrics": {"annual_return": annual, "total_return": annual * 5, "max_drawdown": mdd, "sharpe": sharpe,
                        "volatility": .2, "benchmark_return": .559, "excess_return": annual * 5 - .559},
            "years_beating_benchmark": 5, "years": 7, "yearly": [{"year": 2022, "strategy": -.13, "benchmark": -.2, "beat": True}],
            "worst_year": {"year": 2022, "strategy": -.13}, "best_year": {"year": 2020, "strategy": .46}}


CARDS = [
    card(1, "进取", .295, -.28, 1.31),
    card(2, "稳健", .057, -.118, .63, freq="weekly"),
    card(3, "中庸", .178, -.256, .91),
    card(4, "没算", .9, -.01, 9.0, status="missing"),
]


def test_rank_filters_by_drawdown_then_prefers_holding_period_then_sharpe():
    assert rank(CARDS, "a_share", None, "any")["chosen"]["name"] == "进取"
    assert rank(CARDS, "a_share", .26, "any")["chosen"]["name"] == "中庸"
    assert rank(CARDS, "a_share", .26, "short")["chosen"]["name"] == "稳健"
    result = rank(CARDS, "a_share", .05, "any")
    assert result["chosen"]["name"] == "稳健" and "超出了你能接受的范围" in result["notes"][0]
    assert rank(CARDS, "a_share", None, "long")["notes"]  # 没有季度调仓的，如实说明忽略了这一条
    assert rank(CARDS, "hk", None, "any")["chosen"] is None


def test_number_guard_rejects_invented_numbers():
    facts = [{"年化收益": "29.5%", "最大回撤": "-28.1%"}]
    assert numbers_are_grounded("年化 29.5%，回撤 28.1%", facts)
    assert numbers_are_grounded("年化约 29.4%", facts)  # 四舍五入差 0.1 以内
    assert not numbers_are_grounded("预计明年收益 35%", facts)
    text = template_explanation(CARDS[0], "测试")
    assert "29.5%" in text and "-28.0%" in text


def test_recommend_endpoint_uses_scorecards_and_never_trades(monkeypatch):
    from app.ai import recommender
    from app.ai.credentials import AICredentials

    monkeypatch.setattr(recommender, "get_ai_credentials",
                        lambda db: AICredentials(api_key=None, base_url=None, model="unused", source="env"))
    with TestClient(app) as client:
        rows = client.get("/api/strategies").json()
        ids = [r["id"] for r in rows if r["origin"] == "builtin" and r["kind"] in ("multifactor", "pb_ubl")]
        assert client.post("/api/scorecards/refresh", json={"strategy_ids": ids}).json()["started"]
        for _ in range(300):
            state = client.get("/api/scorecards").json()
            if not state["refresh"]["running"]:
                break
            time.sleep(1)
        assert not state["refresh"]["errors"], state["refresh"]["errors"]
        ready = [c for c in state["cards"] if c["status"] == "ready"]
        assert {c["strategy_id"] for c in ready} >= set(ids)
        # 成绩卡本身就是一次普通回测，能在回测历史里看到
        history = client.get("/api/backtests", params={"strategy_id": ids[0]}).json()
        assert any(h["id"] == next(c["run_id"] for c in ready if c["strategy_id"] == ids[0]) for h in history)

        orders_before = len(client.get("/api/paper/orders").json())
        response = client.post("/api/ai/recommend", json={"max_drawdown": 0.9, "holding": "medium", "question": "想稳一点"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["recommendation"]["status"] == "ready"
        assert body["explanation"]["grounded"] is True
        assert body["next_step"]["practice_url"].startswith("/backtest?strategy_id=")
        assert len(client.get("/api/paper/orders").json()) == orders_before  # 智能体不下单


def test_llm_answer_with_invented_numbers_is_replaced_by_template(monkeypatch):
    from types import SimpleNamespace

    from app.ai import recommender
    from app.ai.credentials import AICredentials

    monkeypatch.setattr(recommender, "get_ai_credentials",
                        lambda db: AICredentials(api_key="k", base_url=None, model="m", source="env"))

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer))])))

    import openai
    result = rank(CARDS, "a_share", None, "any")
    for answer, grounded in (("它年化 29.5%，回撤 28.0%。", True), ("回撤在你能接受的 25% 以内。", True), ("它明年能赚 40%。", False)):
        FakeClient.answer = answer
        monkeypatch.setattr(openai, "OpenAI", FakeClient)
        out = recommender.explain(None, result, "能接受的最大回撤 −25%", "")
        assert out["grounded"] is grounded
        assert (out["content"] == answer) is grounded
