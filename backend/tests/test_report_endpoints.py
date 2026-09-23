from fastapi.testclient import TestClient

from app.main import app

SAMPLE_REPORT = (
    "这是一份关于大盘蓝筹股动量因子的研究报告。历史数据显示，过去20日涨幅"
    "较高的股票在未来一段时间内往往维持相对强势，同时高波动率个股的风险"
    "需要重点关注。"
).encode("utf-8")


def test_analyze_report_persists_audit_and_returns_summary():
    with TestClient(app) as client:
        response = client.post(
            "/api/ai/report/analyze",
            files={"file": ("report.txt", SAMPLE_REPORT, "text/plain")},
        )
        assert response.status_code == 200
        body = response.json()
        assert "summary" in body
        assert "strategy_reference" in body
        assert isinstance(body["audit_id"], int)

        audit = client.get("/api/ai/audit").json()
        assert any(row["id"] == body["audit_id"] and row["explanation_type"] == "report_analysis" for row in audit)


def test_analyze_report_rejects_unsupported_file_type():
    with TestClient(app) as client:
        response = client.post(
            "/api/ai/report/analyze",
            files={"file": ("report.docx", b"whatever", "application/octet-stream")},
        )
        assert response.status_code == 422
