import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data.akshare_provider import _normalise_query
from app.main import app


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
        # /api/stocks跑因子引擎打分，analysis_symbols()把分析规模封顶在
        # ANALYSIS_SYMBOL_CAP(30)——real模式下tushare没有批量接口，每支
        # 股票单独发一次网络请求，不封顶的话全市场几千支会顺序请求到
        # 挂起。这个上限不区分demo/real统一生效，所以这里即使是demo的
        # 50支小目录，也会看到30这个数字。
        assert stocks.json()["total"] == 30
        search = client.get("/api/stocks/search", params={"q": "招商银行", "source": "local"})
        assert search.status_code == 200
        assert any(row["symbol"] == "600036" and row["name"] == "招商银行" for row in search.json()["items"])
        code_search = client.get("/api/stocks/search", params={"q": "600036", "source": "local"})
        assert code_search.status_code == 200
        assert code_search.json()["items"][0]["name"] == "招商银行"
        full_search = client.get("/api/stocks/search", params={"source": "local"})
        assert full_search.status_code == 200
        assert full_search.json()["total"] == 50
        assert len(full_search.json()["items"]) == full_search.json()["total"]
        universes = client.get("/api/universes")
        assert universes.status_code == 200
        assert {item["id"] for item in universes.json()} >= {"a_share", "hs300", "csi_a500"}
        universe_counts = {item["id"]: item["count"] for item in universes.json()}
        assert universe_counts["a_share"] == 50
        assert universe_counts["csi_a500"] == 50
        assert universe_counts["large_cap"] == 50

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

        coverage = client.get("/api/research/coverage")
        assert coverage.status_code == 200
        assert coverage.json()["total"] >= 10
        assert coverage.json()["large_cap_count"] >= 10
        group_counts = {}
        for item in coverage.json()["items"]:
            group_counts[item["group"]] = group_counts.get(item["group"], 0) + 1
        assert set(group_counts) == {"消费", "科技", "新能源", "金融", "红利/央国企"}
        # analysis_symbols()按ANALYSIS_SYMBOL_CAP(30)在5个组之间轮询取，
        # 30/5=6——这里断言的是"每个组都不被漏掉"这个更重要的不变量，不是
        # 具体数量；具体数量随ANALYSIS_SYMBOL_CAP调整会变，但每组至少有
        # 代表这一点不应该变。
        assert min(group_counts.values()) >= 5

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
        assert result["final_assets"] == result["equity"][-1]["total_assets"]
        assert result["metrics"]["overall_return"] == result["overall_return"]
        assert result["total_profit"] == result["final_assets"] - result["initial_capital"]
        assert result["overall_return"] == result["final_assets"] / result["initial_capital"] - 1
        assert result["equity"][0]["total_assets"] == result["initial_capital"]
        assert result["equity"][-1]["cumulative_return"] == result["overall_return"]

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
        # 回测按策略规格执行（ADR-0048）：择时插槽是指数趋势，引擎层的
        # 止损/目标波动率/回撤刹车叠加层已经去掉，不再出现在配置里
        spec = result["strategy_config"]["spec"]
        assert spec["timing"]["type"] == "index_trend"
        assert spec["selection"] == {"type": "top_n", "n": 10, "pct": None}
        assert "target_volatility" not in result["strategy_config"]
        assert result["risk_summary"]["drawdown_brake_rebalances"] == 0
        assert result["risk_summary"]["volatility_scaled_rebalances"] == 0
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


def test_quant_v3_strategy_runs_through_the_shared_backtest_endpoint():
    """自由探索阶段最终确定的LightGBM策略（docs/adr/0013~0037）走独立的
    parquet数据管道，不经过SQLite股票目录——用kind字段在共享的
    /api/backtests端点里分流，结果仍然写入通用的BacktestRun/Trade/
    BacktestEquity表，前端复用现有回测详情页面。"""
    with TestClient(app) as client:
        strategies = client.get("/api/strategies")
        assert strategies.status_code == 200
        quant_v3 = next(row for row in strategies.json() if row["kind"] == "quant_v3_regression")

        response = client.post(
            "/api/backtests",
            json={
                "strategy_id": quant_v3["id"],
                "start_date": "2024-01-01",
                "end_date": "2025-12-31",
                "initial_capital": 1_000_000,
                "commission": 0.0003,
                "slippage": 0.0005,
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "completed"
        assert result["metrics"]["trade_count"] > 0
        assert result["universe_size"] == 30
        assert result["selected_stocks"]

        # 超出2019-2025训练范围的年份应该被明确拒绝，不能悄悄跑出没有
        # 意义的占位信号结果。
        out_of_range = client.post(
            "/api/backtests",
            json={"strategy_id": quant_v3["id"], "start_date": "2016-01-01", "end_date": "2017-12-31"},
        )
        assert out_of_range.status_code == 422


def test_quant_v3_backtest_stock_charts_use_the_parquet_data_pipeline():
    """真实用浏览器点开回测中心页面时发现的bug：个股买卖曲线和"全部入选
    股票买卖点审计"网格都对着通用的`Stock`表按symbol查名字/行情，我们的
    股票是"600519.SH"这种带交易所后缀的格式，通用目录里都是不带后缀的
    裸代码，查不到——静默404/静默跳过，图表区域看起来像是空的，没有任何
    报错提示这是个bug。"""
    with TestClient(app) as client:
        strategies = client.get("/api/strategies")
        quant_v3 = next(row for row in strategies.json() if row["kind"] == "quant_v3_regression")

        run_response = client.post(
            "/api/backtests",
            json={
                "strategy_id": quant_v3["id"],
                "start_date": "2024-01-01",
                "end_date": "2025-12-31",
                "initial_capital": 1_000_000,
                "commission": 0.0003,
                "slippage": 0.0005,
            },
        )
        assert run_response.status_code == 200
        run_id = run_response.json()["id"]
        a_selected_symbol = run_response.json()["selected_stocks"][0]["symbol"]
        # 行情统一读本地行情库，全平台都用不带后缀的代码（ADR-0048）
        assert a_selected_symbol.isdigit() and len(a_selected_symbol) == 6

        chart = client.get(f"/api/backtests/{run_id}/stocks/{a_selected_symbol}/chart")
        assert chart.status_code == 200
        chart_body = chart.json()
        assert chart_body["prices"], "个股行情必须真的查到数据，不能因为symbol格式不兼容悄悄返回空列表"
        assert chart_body["name"] and chart_body["name"] != a_selected_symbol

        bundle = client.get(f"/api/backtests/{run_id}/stocks/charts")
        assert bundle.status_code == 200
        items = bundle.json()["items"]
        assert items, "全部入选股票的审计网格不能因为同样的symbol格式问题被悄悄清空"
        assert all(item["prices"] for item in items)


def test_strategies_list_flags_only_validated_effective_strategies():
    """前端回测中心的策略下拉框只展示`validated=true`的策略——这个字段
    必须直接来自ACTIVE_CSI300_STRATEGIES这个唯一注册表，不能前后端各
    维护一份判断标准，否则两边容易不同步。"""
    with TestClient(app) as client:
        strategies = client.get("/api/strategies").json()
        by_kind = {row["kind"]: row["validated"] for row in strategies}
        assert by_kind["csi300_lightgbm"] is True
        assert by_kind["csi300_xgboost"] is True
        assert by_kind["csi300_ensemble"] is True
        # 默认的通用多因子策略和废弃的动量候选都没有经过"vs真实指数"的
        # 逐年验证，不能被标成有效。
        assert by_kind.get("multifactor") is not True


def test_quant_v3_strategy_params_are_protected_from_generic_edit_endpoint():
    """LightGBM策略(kind=quant_v3_regression)的stop_loss/target_volatility/
    max_drawdown_budget特意存0(ADR-0037：关闭引擎级风控叠加层)——通用的
    PUT /strategies/{id}是给MultiFactorStrategy设计的无条件全量覆盖，
    不能允许它把这些字段悄悄填回schema默认值。"""
    with TestClient(app) as client:
        strategies = client.get("/api/strategies").json()
        quant_v3 = next(row for row in strategies if row["kind"] == "quant_v3_regression")

        response = client.put(f"/api/strategies/{quant_v3['id']}", json={"name": "改个名字试试"})

        assert response.status_code == 422


def test_quant_v3_strategy_drives_paper_trading_rebalance():
    """任务书要求"要能智能化交易，做个模拟盘"——模拟盘必须真的能跑我们
    自己的LightGBM策略(kind=quant_v3_regression)，不能一直只驱动脚手架
    自带的通用多因子策略(见 docs/adr/0040)。"""
    with TestClient(app) as client:
        strategies = client.get("/api/strategies").json()
        quant_v3 = next(row for row in strategies if row["kind"] == "quant_v3_regression")

        client.post("/api/paper/reset", json={"initial_capital": 1_000_000})
        # 用一个训练范围内、明确早于"今天"的历史交易日调仓——这样默认
        # （不传as_of）的account快照会按"今天"对应的最新收盘价估值，和
        # 建仓当天不是同一天，下面"current_price != avg_cost"这条断言
        # 才是真的在验证查到了独立的当前价，而不是因为两者巧合取自同
        # 一天的收盘价而必然相等。
        response = client.post(
            "/api/paper/rebalance",
            json={"strategy_id": quant_v3["id"], "as_of": "2025-06-02"},
        )
        assert response.status_code == 200
        result = response.json()
        # 之前这里曾经因为一个真实bug悄悄跑出空结果：模拟盘默认按字面
        # `date.today()`调仓，遇到非交易日（周末/节假日）时，信号源按
        # 精确日期索引查不到那天的行情，会对每支股票都返回None——不报
        # 错，只是安静地生成一个全零权重的"调仓"。必须显式断言真的下了
        # 单，不能让这种情况又被"如果有持仓才检查"的写法悄悄放过去。
        assert result["orders"], "调仓当天必须真的下单，不能悄悄跑出空结果"
        assert result["execution_scope"]["symbols"]
        # 确认真的用了策略自己的 30 支候选池，不是误落回默认策略的 large_cap 通用池
        assert result["execution_scope"]["universe"] == "broad30"
        assert len(result["execution_scope"]["symbols"]) == 30

        snapshot = client.get("/api/paper/account")
        assert snapshot.status_code == 200
        account = snapshot.json()
        assert account["strategy_id"] == quant_v3["id"]
        assert account["positions"], "调仓下单成功后持仓不能是空的"
        # 持仓要能用我们自己的数据源查到真实当前价，不能因为symbol格式
        # 对不上SQLite目录就悄悄退化成用avg_cost充当current_price。
        assert any(p["current_price"] != p["avg_cost"] for p in account["positions"])
        client.post("/api/paper/reset", json={"initial_capital": 1_000_000})


def test_quant_v3_paper_rebalance_rejects_out_of_range_date():
    with TestClient(app) as client:
        strategies = client.get("/api/strategies").json()
        quant_v3 = next(row for row in strategies if row["kind"] == "quant_v3_regression")

        response = client.post(
            "/api/paper/rebalance",
            json={"strategy_id": quant_v3["id"], "as_of": "2017-01-01"},
        )
        assert response.status_code == 422


def test_quant_v3_research_universe_endpoint_shows_our_own_labelled_stocks():
    """课题要求"把自己研究的股票和板块都标注出来，最起码十个，要标注
    清楚"——这个端点必须直接返回本组自己的30支候选池标注，不是脚手架
    自带的demo研究看板文案。"""
    with TestClient(app) as client:
        response = client.get("/api/research/quant-v3-universe")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 10
        assert len(data["items"]) == data["total"]
        for item in data["items"]:
            assert item["thesis"]
            assert item["risk"]
            assert item["symbol"].endswith((".SH", ".SZ"))
        assert len(data["groups"]) >= 1
        assert len(data["industries"]) >= 10


@pytest.mark.parametrize("kind", ["csi300_lightgbm", "csi300_xgboost", "csi300_ensemble"])
def test_csi300_strategies_run_through_the_shared_backtest_endpoint(kind):
    """对齐主流做法：universe换成沪深300全部300支真实成分股，逐年回测
    验证过真实有效的三个策略(LightGBM/XGBoost/两者集成)都通过同一个
    /api/backtests端点跑通，用kind字段分流（不是自选30支候选池）。"""
    with TestClient(app) as client:
        strategies = client.get("/api/strategies")
        assert strategies.status_code == 200
        strategy = next(row for row in strategies.json() if row["kind"] == kind)

        response = client.post(
            "/api/backtests",
            json={
                "strategy_id": strategy["id"],
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "initial_capital": 1_000_000,
                "commission": 0.0003,
                "slippage": 0.0005,
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "completed"
        assert result["metrics"]["trade_count"] > 0
        assert result["universe_size"] == 300
        assert result["selected_stocks"]
        # 真实沪深300指数对照必须存在(不是候选池自己的等权对照)。
        assert "csi300_index_return" in result["metrics"]

        out_of_range = client.post(
            "/api/backtests",
            json={"strategy_id": strategy["id"], "start_date": "2016-01-01", "end_date": "2017-12-31"},
        )
        assert out_of_range.status_code == 422


def test_csi300_research_universe_endpoint_shows_real_data_not_demo():
    """股票池对齐沪深300：这个端点返回全部300支真实成分股的真实名字/
    真实收盘价/LightGBM真实打分排名，不是脚手架自带的Demo通用目录。"""
    with TestClient(app) as client:
        response = client.get("/api/research/csi300-universe")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 300
        assert len(data["items"]) == 300
        ranks = [item["lightgbm_rank"] for item in data["items"] if item["lightgbm_rank"] is not None]
        # 停牌到最新交易日的股票当天没有行情、不打分（也买不了），其余名次连续
        assert len(ranks) >= 290
        assert sorted(ranks) == list(range(1, len(ranks) + 1))
        for item in data["items"]:
            assert item["name"]
            if item["lightgbm_rank"] is not None:
                assert item["latest_price"] and item["latest_price"] > 0


def test_csi300_strategy_drives_paper_trading_rebalance():
    with TestClient(app) as client:
        strategies = client.get("/api/strategies").json()
        csi300_strategy = next(row for row in strategies if row["kind"] == "csi300_lightgbm")

        client.post("/api/paper/reset", json={"initial_capital": 1_000_000})
        response = client.post(
            "/api/paper/rebalance",
            json={"strategy_id": csi300_strategy["id"], "as_of": "2025-06-02"},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["orders"], "调仓当天必须真的下单，不能悄悄跑出空结果"
        assert result["execution_scope"]["universe"] == "csi300"
        assert len(result["execution_scope"]["symbols"]) == 300

        snapshot = client.get("/api/paper/account")
        assert snapshot.status_code == 200
        account = snapshot.json()
        assert account["positions"], "调仓下单成功后持仓不能是空的"

        # Dashboard的"最新交易信号"必须跟着账户当前真正绑定的策略走，不能
        # 一直写死用默认的通用多因子策略打分——那样会出现"持仓是真实策略
        # 选出来的，但信号列表却是另一个策略且是Demo数据"这种自相矛盾的
        # 展示。
        dashboard = client.get("/api/dashboard")
        assert dashboard.status_code == 200
        dashboard_data = dashboard.json()
        assert dashboard_data["signals"], "绑定了真实策略后信号列表不能是空的"
        for signal in dashboard_data["signals"]:
            assert signal["symbol"] in result["execution_scope"]["symbols"]
            assert signal["name"] != signal["symbol"], "必须查到真实股票名字，不能退化成显示代码本身"

        client.post("/api/paper/reset", json={"initial_capital": 1_000_000})


def test_builtin_library_strategies_are_listed_and_run():
    """策略库第一批（ADR-0047 第 9 条）：迁移的 5 个 + QuantsPlaybook 的 5 个都登记为内置策略，
    并且都能通过同一个回测接口跑通。"""
    with TestClient(app) as client:
        rows = client.get("/api/strategies").json()
        builtin = {row["kind"] for row in rows if row["origin"] == "builtin"}
        assert {"multifactor", "quant_v3_regression", "csi300_lightgbm", "csi300_xgboost", "csi300_ensemble",
                "pb_ens_rsrs", "pb_ens_alligator", "pb_ens_icu_ma", "pb_ubl", "pb_ideal_amplitude"} <= builtin
        by_kind = {row["kind"]: row for row in rows if row["origin"] == "builtin"}
        assert by_kind["pb_ens_rsrs"]["spec"]["timing"]["type"] == "rsrs"
        assert by_kind["pb_ubl"]["default_universe"] == "csi300"

        for kind in ("pb_ens_rsrs", "pb_ubl"):
            response = client.post("/api/backtests", json={
                "strategy_id": by_kind[kind]["id"], "start_date": "2025-01-01", "end_date": "2025-06-30"})
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["universe"] == "csi300" and result["universe_size"] == 300
            assert result["metrics"]["trade_count"] >= 0
            assert result["strategy_config"]["spec"] == by_kind[kind]["spec"]


def test_define_strategy_then_practice_and_reload_history():
    """策略制定 → 策略实践 → 回测历史（ADR-0047 第 3 条）：按规格保存新策略，
    回测后能在历史里找到，并且调出来的结果和刚跑完返回的一致。"""
    with TestClient(app) as client:
        options = client.get("/api/pipeline/options").json()
        assert {"ubl", "ideal_amplitude", "momentum"} <= {f["key"] for f in options["factors"]}
        assert {"rsrs", "icu_ma", "alligator"} <= {t["type"] for t in options["timings"]}
        assert "legacy/csi300_lightgbm" in {m["id"] for m in options["models"]}

        bad = client.post("/api/strategies/spec", json={"name": "坏策略", "spec": {"scorer": {"type": "factor_weights", "weights": {"nope": 1}},
                                                                                  "selection": {"type": "top_n", "n": 10}}})
        assert bad.status_code == 422

        spec = {"scorer": {"type": "factor_weights", "weights": {"momentum": 0.5, "ubl": 0.5}},
                "timing": {"type": "index_trend", "risk_off_exposure": 0.5},
                "selection": {"type": "top_n", "n": 10}, "weighting": {"type": "equal", "max_weight": 0.2},
                "rebalance": {"frequency": "monthly", "turnover_band": 0.02}}
        created = client.post("/api/strategies/spec", json={"name": "测试·动量加影线", "description": "测试用",
                                                            "default_universe": "large_cap", "spec": spec})
        assert created.status_code == 200, created.text
        strategy_id = created.json()["id"]
        again = client.post("/api/strategies/spec", json={"name": "测试·动量加影线", "spec": spec}).json()
        assert again["version"] == created.json()["version"] + 1  # 同名另存为新版本

        row = next(r for r in client.get("/api/strategies").json() if r["id"] == strategy_id)
        assert row["origin"] == "user" and row["default_universe"] == "large_cap"

        run = client.post("/api/backtests", json={"strategy_id": strategy_id, "start_date": "2023-01-01", "end_date": "2024-12-31"})
        assert run.status_code == 200, run.text
        result = run.json()
        assert result["strategy_config"]["spec"]["timing"]["type"] == "index_trend"

        history = client.get("/api/backtests", params={"strategy_id": strategy_id}).json()
        assert [h["id"] for h in history] == [result["id"]]
        assert history[0]["metrics"]["annual_return"] == pytest.approx(result["metrics"]["annual_return"])

        reloaded = client.get(f"/api/backtests/{result['id']}/result").json()
        for key in ("metrics", "equity", "trades", "annual_returns", "selected_stocks", "notes", "strategy_config"):
            assert reloaded[key] == result[key], key


def test_universe_analytics_without_fundamentals_returns_empty_valuation_chart():
    """真实模式下财务因子被去掉后，"估值×成长"图不能报 500，也不能拿程序生成的数字凑。"""
    from app.api.routes import _analytics_from_ranking

    ranking = pd.DataFrame({"symbol": ["1", "2"], "name": ["a", "b"], "industry": ["x", "y"], "score": [60.0, 40.0],
                            "return_12m": [0.1, -0.1], "target_weight": [0.5, 0.0]})
    result = _analytics_from_ranking(ranking, "real")
    assert result["valuation_growth"] == [] and "财务数据" in result["valuation_growth_note"]
    assert len(result["portfolio_weights"]) == 1
