# 回测中心个股图表对我们自己的策略静默404

## 背景

装好Playwright+headless Chromium（不需要sudo，这台机器系统库齐全，
`playwright install chromium`之后直接能跑）之后，第一次用真实浏览器
（不是TestClient/curl，是真的加载前端页面、真的点按钮）走一遍回测中心
页面：选中LightGBM策略、点击"运行回测"、等待结果渲染。

## 发现的问题

页面本体渲染正常（16张指标卡片、Top14入选股票列表都对），但浏览器
控制台报了2个404：

```
404 /api/backtests/{run_id}/stocks/601899.SH/chart
```

对应到UI上，"个股买卖曲线"和"全部入选股票买卖点审计"两块区域都是空的
（"请选择一只入选股"/"暂无入选股票曲线"），没有任何报错提示——看起来
像是"这个功能本来就这样"，实际是接口层面查不到数据。

根因和ADR-0036/0040反复出现过的问题是同一类：`get_backtest_stock_chart`
/`get_backtest_stock_charts`（`app/api/routes.py`）一直用通用的
`Stock`表按symbol查股票名字和行情（`db.scalar(select(Stock).where
(Stock.symbol == symbol, ...))`），这张表是SQLite通用目录，只认
"600519"这种不带交易所后缀的裸代码；我们自己30支候选池的symbol是
"600519.SH"这种带后缀的格式，两边天生对不上，查询必然返回`None`，
触发`HTTPException(404, "股票不存在")`——不是这支股票真的不存在，是
两套数据源的symbol格式不兼容。

这两个接口在`/api/backtests`（ADR-0036/0038）和`PaperTradingService`
（ADR-0040）都已经按`strategy.kind`分流过一次，但当时漏掉了这两个
"个股图表"相关接口，是同一类问题在第三个不同接口上的重复出现。

## 修复

新增`_quant_v3_stock_prices(symbol, start, end)`辅助函数，直接从
`final_strategy_history()`读取原始行情（保留`volume`列，
`APhaseDataService.prices()`为了给回测引擎用而砍掉了这一列，图表还需要
它）。`get_backtest_stock_chart`/`get_backtest_stock_charts`都先查
`BacktestRun.strategy_id`对应的`Strategy.kind`，是`quant_v3_regression`
就走`BROAD_STOCKS`查名字+`_quant_v3_stock_prices`查行情，否则维持原来
走通用`Stock`表+`MarketDataService`的路径。

## 验证

真实浏览器复测（同样的LightGBM策略、同样点"运行回测"）：控制台0个
404，个股买卖曲线和全部14支入选股票的审计网格都渲染出真实的价格曲线
和买卖点标记（K线走势+三角形买入点+菱形卖出点）。新增集成测试
`test_quant_v3_backtest_stock_charts_use_the_parquet_data_pipeline`
覆盖两个接口，断言价格数据非空、股票名字不是"退化成symbol本身"。全量
测试204/204通过。

## 副产品发现：测试套件和手动开发服务器共用同一个SQLite文件

排查这次改动时，用curl直接调`/api/paper/account`观察模拟盘净值曲线，
发现曲线异常（只有一个孤立数据点，数值和账户总资产对不上）。追查后
发现原因不是模拟盘逻辑的bug：仓库里没有任何`conftest.py`或对
`get_db`的依赖覆盖（`app/db/session.py`），意味着全部197~204个
`TestClient(app)`测试和手动`uvicorn`开发服务器用的是**同一个**
`data/patience_quant.db`文件——每次跑`pytest`都会往这个文件里真实写入
回测记录、策略克隆、模拟盘调仓/重置，从不清理。这解释了`/api/strategies`
里那70多条"回测#X模拟盘"重复策略、`trades`表里几百次历史回测积累
下来的重复行，以及我手工调完模拟盘之后跑测试套件、状态又被测试的
`reset`悄悄冲掉的现象。

用户确认后一并修复：新增`tests/conftest.py`。这里不能只做
`app.dependency_overrides[get_db]`——`app/main.py`的`lifespan`（建表/
`migrate_lightweight_schema()`/`seed_database()`）和自动调仓循环直接用
`app.db.session`里的模块级`engine`/`SessionLocal`全局对象，完全不经过
`Depends(get_db)`，只覆盖依赖注入这一条路径挡不住它们。真正生效的
隔离点是：在任何`app.*`模块被导入之前，把`DATABASE_URL`环境变量指向
一个进程私有的临时sqlite文件——`app/config.py`的`Settings`是
pydantic-settings，`get_settings()`在`app.db.session`导入时立刻求值
并绑定`engine`，conftest.py的模块级代码保证在这一切发生之前执行。
测试session结束后自动删除这个临时文件（`autouse=True`的
session-scope fixture）。

验证：记录改动前`data/patience_quant.db`的mtime/大小，跑完整套件后
mtime分毫未变，证明测试确实完全不再碰开发数据库；套件本身
205/205通过（204+这次新增的图表回归测试）。
