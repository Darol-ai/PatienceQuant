from fastapi.testclient import TestClient

from app.main import app


def _create_audit(client) -> int:
    """用研报总结造一条审计记录（没有配置大模型时走规则回退）。"""
    response = client.post("/api/ai/report/analyze", files={"file": ("r.txt", "一份关于动量因子的研报。".encode(), "text/plain")})
    assert response.status_code == 200
    return response.json()["audit_id"]


def test_ai_settings_can_be_configured_from_the_api_without_leaking_the_key():
    """OpenAI key要能通过API配置(不只靠.env+重启)，但读取时不能把已保存的
    key明文吐回来——GET只暴露has_api_key这个布尔值。"""
    with TestClient(app) as client:
        before = client.get("/api/settings/ai").json()
        assert before["source"] == "env"  # .env里配了真实key，这里应为env来源

        updated = client.post("/api/settings/ai", json={"api_key": "sk-test-12345", "model": "gpt-test"}).json()
        assert updated["has_api_key"] is True
        assert updated["model"] == "gpt-test"
        assert updated["source"] == "database"
        assert "api_key" not in updated
        assert "sk-test-12345" not in str(updated)

        # 只传model，api_key不应被清空。
        partial = client.post("/api/settings/ai", json={"model": "gpt-test-2"}).json()
        assert partial["has_api_key"] is True
        assert partial["model"] == "gpt-test-2"

        reset = client.delete("/api/settings/ai").json()
        assert reset["source"] == "env"


def test_ai_decision_records_adoption_and_rollback():
    with TestClient(app) as client:
        audit_id = _create_audit(client)

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
