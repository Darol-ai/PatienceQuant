from fastapi.testclient import TestClient

from app.ai.strategy_assistant import _sanitize
from app.main import app


def test_explain_trade_persists_and_returns_an_audit_id():
    with TestClient(app) as client:
        response = client.post(
            "/api/ai/explain/trade",
            json={"symbol": "600036", "action": "BUY", "score": 82, "rank": 3, "industry": "银行", "use_llm": False},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "rules"
        assert body["model_version"] == "rules"
        assert isinstance(body["audit_id"], int)

        audit = client.get("/api/ai/audit").json()
        assert any(row["id"] == body["audit_id"] for row in audit)


def test_ai_decision_records_adoption_and_rollback():
    with TestClient(app) as client:
        created = client.post(
            "/api/ai/explain/trade",
            json={"symbol": "600036", "action": "HOLD", "use_llm": False},
        ).json()
        audit_id = created["audit_id"]

        adopted = client.post(f"/api/ai/audit/{audit_id}/decision", json={"adopted": True})
        assert adopted.status_code == 200
        assert adopted.json()["adopted"] is True

        rolled_back = client.post(f"/api/ai/audit/{audit_id}/decision", json={"rolled_back": True})
        assert rolled_back.status_code == 200
        assert rolled_back.json()["rolled_back"] is True

        audit = client.get("/api/ai/audit").json()
        row = next(r for r in audit if r["id"] == audit_id)
        assert row["adopted"] is True
        assert row["rolled_back"] is True
        assert row["decided_at"] is not None


def test_ai_decision_on_missing_audit_id_is_404():
    with TestClient(app) as client:
        response = client.post("/api/ai/audit/999999/decision", json={"adopted": True})
        assert response.status_code == 404


def test_strategy_assistant_falls_back_to_rules_without_an_api_key(monkeypatch):
    """pydantic-settings 从 .env 文件读密钥，光 delenv 进程变量盖不掉它——
    直接把 strategy_assistant 模块看到的 get_settings 换成一个没有 key 的假
    settings，这样测试才是真的在测"没有 key 时的回退行为"。"""
    import app.ai.strategy_assistant as module

    class _NoKeySettings:
        openai_api_key = None
        openai_base_url = None
        openai_model = "unused"

    monkeypatch.setattr(module, "get_settings", lambda: _NoKeySettings())
    with TestClient(app) as client:
        response = client.post("/api/ai/strategy-assistant", json={"description": "低波动、大盘股为主的月度调仓策略"})
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "rules"
        assert isinstance(body["audit_id"], int)
        assert body["suggested_params"]["holdings_count"] == 10


def test_sanitize_drops_unknown_and_out_of_range_fields():
    """策略助手不能让模型编造出参数表里没有的字段，也不能让越界的数字混进去
    ——比如 max_weight 给到 0.9（远超允许的 0.05~0.5 上限）应该被丢弃，
    而不是被悄悄接受。"""
    raw = {
        "holdings_count": 12,
        "max_weight": 0.9,
        "rebalance_frequency": "daily",  # 不在允许列表里
        "made_up_field": "should be dropped",
    }

    clean = _sanitize(raw)

    assert clean == {"holdings_count": 12}
