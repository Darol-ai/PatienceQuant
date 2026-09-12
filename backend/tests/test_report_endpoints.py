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


def test_generate_factor_without_llm_reports_no_reference_value(monkeypatch):
    """没有配置语言模型时，功能②要诚实回退——不能假装研报有参考价值、
    也不能在没有真实因子的情况下还去跑回测。"""
    import app.ai.report_factor as module

    class _NoKeySettings:
        openai_api_key = None
        openai_base_url = None
        openai_model = "unused"

    monkeypatch.setattr(module, "get_settings", lambda: _NoKeySettings())
    with TestClient(app) as client:
        response = client.post(
            "/api/ai/report/generate-factor",
            files={"file": ("report.txt", SAMPLE_REPORT, "text/plain")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["has_reference_value"] is False
        assert body["factor_weights"] == {}
        assert body["backtest"] is None

        audit = client.get("/api/ai/audit").json()
        assert any(row["id"] == body["audit_id"] and row["explanation_type"] == "report_factor" for row in audit)


def test_generate_factor_runs_a_real_backtest_when_weights_are_produced(monkeypatch):
    """模拟"研报确实有参考价值"这条路径，验证因子真的被接进了回测系统——
    不检查收益好不好，只检查这条链路真的跑完了、产出了绩效指标。"""
    import app.ai.report_factor as module

    monkeypatch.setattr(
        module.ReportFactorService, "generate",
        lambda self, report_text: {
            "has_reference_value": True,
            "rationale": "测试用固定因子，不依赖真实模型调用。",
            "factor_weights": {"return_20d": 0.5, "volatility_60d": -0.3},
            "provider": "test", "model_version": "test", "confidence": None,
        },
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/ai/report/generate-factor",
            files={"file": ("report.txt", SAMPLE_REPORT, "text/plain")},
            params={"start_date": "2024-01-01", "end_date": "2024-06-30"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["has_reference_value"] is True
        assert body["backtest"] is not None
        assert "metrics" in body["backtest"]
        assert "final_assets" in body["backtest"]["metrics"]
