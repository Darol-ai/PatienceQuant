"""演示轨迹：在隔离的演示环境（前端 5174 → 后端 8001 → 副本数据库）里模拟一个用户正常使用系统。
每一步截图，并把系统返回的关键结果写进 results_*.json。需要 playwright（pip install playwright && playwright install chromium）。

演示环境的启动方式：复制数据库和 backend/data/models 到临时目录，后端用
DATABASE_URL=sqlite:///<副本> PATIENCEQUANT_MODELS_DIR=<副本> PATIENCEQUANT_WARM_CSI300=0 PATIENCEQUANT_AUTO_TRAIN=0 起在 8001，
前端用 VITE_API_TARGET=http://localhost:8001 npx vite --config vite.config.ts --port 5174。"""
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:5174"
OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(parents=True, exist_ok=True)
results = {}
errors = []
NEW_NAME = "演示·XGBoost沪深300（加入估值与换手率因子）"


def shot(page, name, full=False):
    page.wait_for_timeout(800)
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
    print("shot", name, flush=True)


def api(page, path, method="GET", body=None):
    url = BASE + "/api" + path
    resp = page.request.post(url, data=body, timeout=900_000) if method == "POST" else page.request.get(url, timeout=900_000)
    return resp.json()


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.on("pageerror", lambda e: errors.append(str(e)[:300]))
    t0 = time.time()

    # 开跑前确认 5174 真的连到演示后端 8001：只在 8001 建标记，5174 看不到就停
    marker = page.request.post("http://127.0.0.1:8001/api/pools", data={"name": "隔离检查标记", "symbols": [
        "600519", "000333", "600036", "300750", "002594", "601318", "688981", "601088", "600900", "601006"]}).json()
    seen = any(p_["name"] == "隔离检查标记" for p_ in page.request.get(BASE + "/api/pools").json())
    real = any(p_["name"] == "隔离检查标记" for p_ in page.request.get("http://127.0.0.1:8000/api/pools").json())
    page.request.delete(f"http://127.0.0.1:8001/api/pools/{marker['id']}")
    if not seen or real:
        raise SystemExit(f"隔离检查失败：5174 看到标记={seen}，真实环境看到标记={real}")
    print("isolation ok", flush=True)

    # ---- 第 1 步：首页问智能体（SKIP_AGENT=1 时沿用已有截图，不再调用大模型）----
    import os
    if os.environ.get("SKIP_AGENT") == "1":
        pass
    else:
      page.goto(BASE + "/", wait_until="networkidle")
      shot(page, "01a-首页")
      page.get_by_role("button", name="最多亏 25%").click()
      page.get_by_role("button", name="每月调整").click()
      page.wait_for_selector(".agent-recommend", timeout=180_000)
      page.wait_for_timeout(1500)
      shot(page, "01b-智能体推荐")
      rec = page.locator(".agent-recommend").last
      results["1_recommend"] = rec.inner_text()[:1200]
      page.locator(".agent-input input, .agent-input textarea").first.fill("为什么不推荐 LSTM沪深300策略（系统内训练）？")
      page.locator(".agent-input .primary-button").click()
      page.wait_for_timeout(1000)
      page.wait_for_function("() => !document.body.innerText.includes('正在查成绩卡…')", timeout=180_000)
      page.wait_for_timeout(1500)
      shot(page, "01c-追问为什么不是某策略")
      results["1_why_not"] = page.locator(".agent-msg.agent").last.inner_text()[:1200]

    # ---- 第 2 步：股票池、研究标注、股票池分析 ----
    page.goto(BASE + "/pools", wait_until="networkidle")
    page.get_by_text("我的低频研究清单").first.click()
    page.wait_for_load_state("networkidle")
    page.locator("tr", has_text="600519").locator("button[title='编辑标注']").click()
    page.locator("input.note-input").fill("白酒消费")
    page.locator("textarea.note-input").fill("演示用示例文字：高端白酒龙头，观察现金流与估值中枢（小组的真实研究理由由组员填写）")
    shot(page, "02a-编辑研究标注")
    page.locator("tr", has_text="600519").get_by_role("button", name="保存").click()
    page.wait_for_timeout(1500)
    page.wait_for_selector(".pool-analysis-grid", timeout=120_000)
    shot(page, "02b-标注已保存")
    page.locator(".pool-analysis-grid").scroll_into_view_if_needed()
    shot(page, "02c-股票池分析", full=True)
    members = api(page, "/pools/custom:1/members")
    results["2_annotation"] = next(m for m in members["members"] if m["symbol"] == "600519")
    results["2_analysis"] = api(page, "/pools/custom:1/analysis")["summary"]

    # ---- 第 3 步：策略库 ----
    page.goto(BASE + "/library", wait_until="networkidle")
    shot(page, "03a-策略库")
    selects = page.locator(".filter-bar select, select")
    page.locator("select").nth(1).select_option("model")
    shot(page, "03b-筛选模型策略")

    # ---- 第 4 步：策略详情 ----
    page.goto(BASE + "/library/32", wait_until="networkidle")
    page.wait_for_selector(".data-note", timeout=60_000)
    shot(page, "04-策略详情-数据说明与成绩卡", full=True)
    results["4_data_note"] = page.locator(".data-note").inner_text()

    # ---- 第 5 步：制定一个模型策略并训练 ----
    page.goto(BASE + "/strategy/new", wait_until="networkidle")
    page.locator("label", has_text="策略名称").locator("input").fill(NEW_NAME)
    page.locator("label", has_text="策略说明").locator("textarea").fill("演示：沪深300历史成分股，XGBoost，旧模型的 14 个因子加上市净率和 20 日平均换手率，预测 90 日收益；选前 10% 等权，每月调仓，不择时。")
    page.locator("select.inline-select").select_option("model")
    page.locator("label", has_text="算法").locator("select").select_option("xgboost")
    page.locator(".factor-chip", has_text="市净率").click()
    page.locator(".factor-chip", has_text="20 日平均换手率").click()
    page.locator("label", has_text="选股规则").locator("select").select_option("top_pct")
    page.locator("label", has_text="x（%）").locator("input").fill("10")
    page.locator("label", has_text="单股上限").locator("input").fill("100")
    shot(page, "05a-填写训练设置", full=True)
    page.get_by_role("button", name="保存并训练").click()
    page.wait_for_url("**/library/*", timeout=120_000)
    new_id = int(page.url.rstrip("/").split("/")[-1])
    results["5_new_strategy_id"] = new_id
    page.wait_for_timeout(2500)
    shot(page, "05b-已保存-模型训练中")

    # ---- 第 6 步：数据与模型页看训练进度，等训练完成 ----
    page.goto(BASE + "/data", wait_until="networkidle")
    page.wait_for_timeout(3000)
    shot(page, "06a-训练进度", full=True)
    strategy = next(s for s in api(page, "/strategies") if s["id"] == new_id)
    model_id = strategy["spec"]["scorer"]["models"][0]
    t_train = time.time()
    while True:
        model = next(m for m in api(page, "/models") if m["id"] == model_id)
        if model["status"] not in ("queued", "training"):
            break
        time.sleep(10)
    ics = [v["ic"] for v in (model.get("metrics") or {}).values() if v.get("ic") is not None]
    results["6_model"] = {"id": model_id, "status": model["status"], "error": model.get("error"),
                          "wait_seconds": round(time.time() - t_train), "years": [model["years"][0], model["years"][-1]] if model["years"] else None,
                          "mean_ic": round(sum(ics) / len(ics), 4) if ics else None}
    page.reload(wait_until="networkidle")
    page.locator("tr.clickable-row", has_text="市净率").first.click() if page.locator("tr.clickable-row", has_text="市净率").count() else page.locator("tr.clickable-row").last.click()
    shot(page, "06b-模型逐年样本外成绩", full=True)

    # ---- 第 7 步：成绩卡自动算出 → 完整回测报告 → 解释一笔交易 ----
    while True:
        card = next((c for c in api(page, "/scorecards")["cards"] if c["strategy_id"] == new_id), None)
        if card and card["status"] == "ready":
            break
        time.sleep(10)
    m = card["metrics"]
    results["7_scorecard"] = {k: m[k] for k in ("total_return", "annual_return", "max_drawdown", "sharpe", "benchmark_return")}
    page.goto(BASE + f"/library/{new_id}", wait_until="networkidle")
    page.wait_for_selector(".data-note")
    shot(page, "07a-新策略成绩卡", full=True)
    page.get_by_text("查看成绩卡这次回测的完整报告").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(3000)
    shot(page, "07b-回测报告")
    explain = page.get_by_role("button", name="解释").first
    explain.scroll_into_view_if_needed()
    explain.click()
    page.wait_for_selector("tr.explain-row", timeout=120_000)
    page.locator("tr.explain-row").first.scroll_into_view_if_needed()
    shot(page, "07c-交易解释")
    results["7_trade_explain"] = page.locator("tr.explain-row").first.inner_text()[:800]

    # ---- 第 8 步：策略实践：换股票池和区间再回测 ----
    page.goto(BASE + f"/backtest?strategy_id={new_id}", wait_until="networkidle")
    page.locator("label", has_text="股票池").locator("select").select_option("broad30")
    page.get_by_role("button", name="最近两年").click()
    shot(page, "08a-策略实践-换条件")
    page.locator(".page-actions .primary-button").click()
    page.wait_for_url("**/backtests/*", timeout=900_000)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(3000)
    shot(page, "08b-新回测报告")
    run_id = int(page.url.rstrip("/").split("/")[-1])
    run = api(page, f"/backtests/{run_id}")
    results["8_backtest"] = {"run_id": run_id, "start": run["start_date"], "end": run["end_date"],
                             **{k: run["metrics"][k] for k in ("total_return", "annual_return", "max_drawdown", "sharpe", "benchmark_return")}}

    # ---- 第 9 步：放进模拟盘、开启自动调仓、看持仓和交易账本 ----
    page.goto(BASE + f"/library/{new_id}", wait_until="networkidle")
    page.once("dialog", lambda d: d.accept())
    page.get_by_role("button", name="放进模拟盘").click()
    page.wait_for_url("**/paper", timeout=900_000)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(2500)
    auto_btn = page.get_by_role("button", name="开启自动调仓")
    if auto_btn.count():
        auto_btn.click()
        page.wait_for_timeout(2000)
    shot(page, "09a-模拟盘", full=True)
    order_explain = page.get_by_role("button", name="解释").first
    if order_explain.count():
        order_explain.scroll_into_view_if_needed()
        order_explain.click()
        page.wait_for_timeout(5000)
        shot(page, "09b-订单解释")
    account = api(page, "/paper/account")
    orders = api(page, "/paper/orders")
    results["9_paper"] = {"strategy": account["strategy_name"], "cash": account["cash"], "total_assets": account["total_assets"],
                          "positions": len(account["positions"]), "orders": len(orders) if isinstance(orders, list) else orders,
                          "automation": api(page, "/paper/automation")}

    # ---- 第 10 步：数据与模型总览 ----
    page.goto(BASE + "/data", wait_until="networkidle")
    shot(page, "10-数据与模型", full=True)
    results["10_market"] = {k: v for k, v in api(page, "/data/market/status").items() if k in ("latest_stored", "stored_days", "missing_days")}

    results["total_seconds"] = round(time.time() - t0)
    results["page_errors"] = errors
    browser.close()

Path(OUT.parent / "results_steps2-10.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2)[:6000])
