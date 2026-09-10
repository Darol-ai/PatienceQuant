from fastapi.testclient import TestClient

from app.main import app
from app.data.akshare_provider import _normalise_query


def test_stock_search_normalises_common_chinese_vendor_input():
    assert _normalise_query("万  科Ａ") == "万科"
    assert _normalise_query(" 600036 ") == "600036"
    assert _normalise_query("宁德时代") == "宁德时代"


def test_health_and_seeded_stocks():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        stocks = client.get("/api/stocks")
        assert stocks.status_code == 200
        assert stocks.json()["total"] >= 1000
        search = client.get("/api/stocks/search", params={"q": "招商银行", "source": "local"})
        assert search.status_code == 200
        assert any(row["symbol"] == "600036" and row["name"] == "招商银行" for row in search.json()["items"])
        code_search = client.get("/api/stocks/search", params={"q": "600036", "source": "local"})
        assert code_search.status_code == 200
        assert code_search.json()["items"][0]["name"] == "招商银行"
        full_search = client.get("/api/stocks/search", params={"source": "local"})
        assert full_search.status_code == 200
        assert full_search.json()["total"] >= 1000
        assert len(full_search.json()["items"]) == full_search.json()["total"]
        universes = client.get("/api/universes")
        assert universes.status_code == 200
        assert {item["id"] for item in universes.json()} >= {"a_share", "hs300", "csi_a500"}
        universe_counts = {item["id"]: item["count"] for item in universes.json()}
        assert universe_counts["a_share"] >= 1000
        assert universe_counts["csi_a500"] >= 500
        assert universe_counts["large_cap"] >= 300
        assert universe_counts["pink_sheets"] >= 10

        strategies = client.get("/api/strategies")
        assert strategies.status_code == 200
        summary = strategies.json()[0]
        assert summary["universe"] == "large_cap"
        assert summary["research_start_date"] == "2018-01-01"
        assert summary["study_period"]["start"] <= summary["study_period"]["end"]
        assert summary["study_period"]["end"] == "2025-12-31"
        assert "total_return" in summary["backtest_metrics"]
        assert summary["target_volatility"] > 0
        assert summary["max_drawdown_budget"] > 0
        assert summary["drawdown_brake_exposure"] > 0

        pink = client.get("/api/stocks", params={"exchange": "OTC/Pink Sheets"})
        assert pink.status_code == 200
        assert pink.json()["total"] >= 10
        assert all(row["exchange"] == "OTC/Pink Sheets" for row in pink.json()["items"])

        coverage = client.get("/api/research/coverage")
        assert coverage.status_code == 200
        assert coverage.json()["total"] >= 10
        assert coverage.json()["large_cap_count"] >= 10
        assert coverage.json()["pink_count"] >= 4
        group_counts = {}
        for item in coverage.json()["items"]:
            group_counts[item["group"]] = group_counts.get(item["group"], 0) + 1
        assert set(group_counts) == {"消费", "科技", "新能源", "金融", "红利/央国企"}
        assert min(group_counts.values()) >= 10

        too_small = client.post(
            "/api/watchlists",
            json={"name": "测试小池不应保存", "symbols": ["600519", "000333"]},
        )
        assert too_small.status_code == 422

        history = client.get("/api/stocks/601006", params={"range": "all", "interval": "monthly"})
        assert history.status_code == 200
        assert history.json()["chart"]["start_date"].startswith("2018-")
        assert history.json()["chart"]["points"] >= 90
        assert len(history.json()["prices"]) == history.json()["chart"]["points"]


def test_ai_explainer_without_external_key():
    with TestClient(app) as client:
        response = client.post("/api/ai/explain/trade", json={"symbol": "600036", "action": "BUY", "score": 80, "rank": 2})
        assert response.status_code == 200
        assert response.json()["provider"] == "rules"


def test_paper_automation_and_equity_markers():
    with TestClient(app) as client:
        status = client.get("/api/paper/automation")
        assert status.status_code == 200
        enabled = client.post("/api/paper/automation", json={"strategy_id": 1, "enabled": True, "frequency": "monthly"})
        assert enabled.status_code == 200
        assert enabled.json()["enabled"] is True
        run = client.post("/api/paper/automation/run")
        assert run.status_code == 200
        client.post("/api/paper/automation", json={"strategy_id": 1, "enabled": False, "frequency": "monthly"})
        equity = client.get("/api/paper/equity")
        assert equity.status_code == 200
        assert "trade_markers" in equity.json()
        client.post("/api/paper/reset", json={"initial_capital": 1_000_000})


def test_backtest_returns_top_n_and_stock_trade_chart():
    with TestClient(app) as client:
        response = client.post(
            "/api/backtests",
            json={
                "strategy_id": 1,
                "universe": "large_cap",
                "start_date": "2018-01-01",
                "end_date": "2025-12-31",
                "initial_capital": 1_000_000,
                "rebalance_frequency": "monthly",
                "holdings_count": 10,
                "max_weight": 0.15,
                "commission": 0.001,
                "slippage": 0.0005,
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "completed"
        assert len(result["selected_stocks"]) == 10
        assert result["metrics"]["trade_count"] > 0
        assert result["trade_markers"]

        symbol = result["selected_stocks"][0]["symbol"]
        chart = client.get(f"/api/backtests/{result['id']}/stocks/{symbol}/chart")
        assert chart.status_code == 200
        chart_data = chart.json()
        assert chart_data["symbol"] == symbol
        assert chart_data["prices"]
        assert chart_data["trade_markers"]
        assert {marker["side"] for marker in chart_data["trade_markers"]} & {"BUY", "SELL"}


def test_backtest_exposes_risk_controls_and_all_stock_chart_bundle():
    with TestClient(app) as client:
        strategy = client.get("/api/strategies").json()[0]
        assert strategy["holdings_count"] >= 10
        assert strategy["cash_buffer"] >= 0
        assert strategy["trend_filter"] is True
        assert strategy["risk_off_exposure"] < 1
        response = client.post(
            "/api/backtests",
            json={
                "strategy_id": strategy["id"],
                "universe": "large_cap",
                "start_date": "2020-01-01",
                "end_date": "2022-12-31",
                "holdings_count": 10,
                "max_weight": 0.15,
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["risk_summary"]["average_target_exposure"] <= 1
        assert result["strategy_config"]["turnover_band"] >= 0
        assert 0 < result["strategy_config"]["target_volatility"] <= 1
        assert 0 < result["strategy_config"]["max_drawdown_budget"] <= 1
        assert 0 < result["strategy_config"]["drawdown_brake_exposure"] <= 1
        assert "drawdown_brake_rebalances" in result["risk_summary"]
        assert "volatility_scaled_rebalances" in result["risk_summary"]
        bundle = client.get(f"/api/backtests/{result['id']}/stocks/charts")
        assert bundle.status_code == 200
        items = bundle.json()["items"]
        assert len(items) == 10
        assert all(item["prices"] for item in items)
        assert all("trade_markers" in item for item in items)


def test_custom_backtest_requires_ten_stocks_and_exports_complete_csv():
    with TestClient(app) as client:
        stock_response = client.get("/api/stocks", params={"universe": "a_share", "sort": "symbol", "order": "asc"})
        assert stock_response.status_code == 200
        symbols = [row["symbol"] for row in stock_response.json()["items"][:10]]
        assert len(symbols) == 10

        too_small = client.post(
            "/api/backtests",
            json={
                "strategy_id": 1,
                "universe": "custom",
                "custom_symbols": symbols[:9],
                "start_date": "2024-01-01",
                "end_date": "2025-12-31",
                "holdings_count": 10,
                "max_weight": 0.15,
            },
        )
        assert too_small.status_code == 422

        response = client.post(
            "/api/backtests",
            json={
                "strategy_id": 1,
                "universe": "custom",
                "custom_symbols": symbols,
                "start_date": "2024-01-01",
                "end_date": "2025-12-31",
                "initial_capital": 1_000_000,
                "rebalance_frequency": "monthly",
                "holdings_count": 10,
                "max_weight": 0.15,
                "commission": 0.001,
                "slippage": 0.0005,
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["universe"] == "custom"
        assert result["universe_size"] == 10
        assert len(result["selected_stocks"]) == 10
        assert result["trades"]

        csv_response = client.get(f"/api/backtests/{result['id']}/trades.csv")
        assert csv_response.status_code == 200
        assert "text/csv" in csv_response.headers["content-type"]
        assert f'backtest_{result["id"]}_trades.csv' in csv_response.headers["content-disposition"]
        lines = csv_response.content.decode("utf-8-sig").splitlines()
        assert lines[0].startswith("backtest_id,trade_date,symbol")
        assert len(lines) == len(result["trades"]) + 1


def test_completed_custom_backtest_can_apply_same_scope_to_paper():
    with TestClient(app) as client:
        client.post("/api/paper/reset", json={"initial_capital": 1_000_000})
        strategy = next(item for item in client.get("/api/strategies").json() if item["is_default"])
        symbols = ["600036", "600519", "000333", "300750", "002594", "601318", "601398", "601857", "688981", "002415"]
        response = client.post(
            "/api/backtests",
            json={
                "strategy_id": strategy["id"],
                "universe": "custom",
                "custom_symbols": symbols,
                "start_date": "2024-01-01",
                "end_date": "2025-12-31",
                "holdings_count": 10,
                "max_weight": 0.15,
            },
        )
        assert response.status_code == 200
        run = response.json()
        applied = client.post(
            f"/api/backtests/{run['id']}/apply-paper",
            json={"reset_account": True, "execute": True, "enable_automation": False},
        )
        assert applied.status_code == 200
        body = applied.json()
        assert body["applied"] is True
        assert body["watchlist"]["count"] == 10
        assert body["rebalance"]["execution_scope"]["source_backtest_run_id"] == run["id"]
        assert set(body["rebalance"]["execution_scope"]["symbols"]) == set(symbols)
        assert body["rebalance"]["orders"]
        account = client.get("/api/paper/account")
        assert account.status_code == 200
        assert account.json()["source_backtest_run_id"] == run["id"]
        repeat = client.post("/api/paper/rebalance", json={})
        assert repeat.status_code == 200
        repeat_body = repeat.json()
        assert set(repeat_body["execution_scope"]["symbols"]) == set(symbols)
        assert repeat_body["execution_scope"]["source_backtest_run_id"] == run["id"]
        assert not repeat_body["orders"], "重复执行同一回测策略不应产生重复成交"
        explicit_repeat = client.post("/api/paper/rebalance", json={"strategy_id": body["strategy"]["id"]})
        assert explicit_repeat.status_code == 200
        assert set(explicit_repeat.json()["execution_scope"]["symbols"]) == set(symbols)
        csv = client.get("/api/paper/orders.csv")
        assert csv.status_code == 200
        assert csv.content.decode("utf-8-sig").splitlines()[0].startswith("order_id,time,symbol")
        client.post("/api/paper/reset", json={"initial_capital": 1_000_000})
