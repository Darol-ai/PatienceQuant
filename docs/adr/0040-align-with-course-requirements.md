# 对齐任务书：模拟盘接入我们的策略 + 研究标注换成本组真实内容

## 背景

对照课题作业要求逐条检查发现两处缺口：①"要能智能化交易，做个模拟盘"
——模拟盘（`/api/paper/*`）此前完全没接入我们的LightGBM策略，一直只能
驱动脚手架自带的通用多因子策略；②"把自己研究的股票和板块都标注出来，
最起码十个，要标注清楚"——现有的"研究看板"展示的是脚手架自带的demo
文案，不是本组对30支候选池的真实研究依据。

## 1. 模拟盘补训2026年模型 + 接入我们的策略

模拟盘默认按"今天"驱动（`demo_as_of`默认`date.today()`，这台环境里
已经是2026年），但我们的LightGBM模型只训练到2025年
（`SUPPORTED_YEAR_RANGE=(2019,2025)`）。用和2019-2025完全相同的滚动
训练规则（训练截止Y-1年7月、Y-1下半年校准）补了一折2026年模型
（`scripts/train_broad_regression_h90_ensemble_2026.py`，训练样本已有
超10万条，不需要重新抓数据），`SUPPORTED_YEAR_RANGE`扩到`(2019,2026)`。

`PaperTradingService`（`app/portfolio/service.py`）改动：
- 新增`_resolve_paper_data(strategy)`：检测到策略`kind=="quant_v3_regression"`
  时返回`APhaseDataService(final_strategy_history())`，否则返回构造时
  注入的`self.data`（SQLite/MarketDataService）——两边symbol格式不一样
  （我们是"600519.SH"带交易所后缀，SQLite通用目录是不带后缀的
  "600519"），查当前价/公司名必须按持仓所属策略切换数据源，不能一直
  用同一个。
- `snapshot()`/`rebalance()`都改成先解析`strategy`、resolve对应的
  `data`，`rebalance()`额外在检测到我们的策略时：跳过`universe`/自定义
  股票池解析、固定用30支候选池、用`build_final_strategy()`代替
  `MultiFactorStrategy(FactorEngine(...))`计算目标权重、把
  `as_of`按`validate_backtest_date_range`校验（拒绝训练范围外的日期，
  转成一个清楚的报错而不是悄悄返回空结果）。
- 修了两处因为直接复用`RegressionRotationStrategy`的`ranking`格式而
  暴露出的兼容性小问题：`ranking`没有MultiFactorStrategy特有的
  "action"列、`score`可能是`None`（信号源缺数据时的诚实占位）——下游
  按列/值是否存在做了防御性兜底，不是新bug，是两套策略ranking格式
  本来就不完全一样，需要兼容两者。
- `catalog`(股票名称/行业)在我们自己的parquet数据源里没有"name"/
  "industry"这两列（只有价量数据），`snapshot()`按列是否存在兜底，
  不再假设这两列一定在。

新增集成测试：真的调用`/api/paper/rebalance`+`/api/paper/account`，
验证下单symbol是带交易所后缀的格式（确认真的用了我们自己的股票池）、
持仓当前价能用我们自己的数据源查到真实报价（不是退化成用成本价充当
现价）；以及超出训练范围的日期被拒绝（422）。

## 2. 研究标注换成本组真实内容

`/api/research/coverage`这个既有端点架构上和SQLite目录+多因子排序强
耦合（`ranking.symbol.isin(annotation_by_symbol)`，两边symbol格式还
对不上），直接复用改造成本高。新增独立端点
`GET /api/research/quant-v3-universe`（`app/quant_v3/research_notes.py`），
直接返回本组30支候选池的真实标注——每支股票的行业、分组、实际研究
依据（为什么纳入候选池）、风险点（包括几处真实的诚实记录，比如002475
立讯精密/002714牧原股份被明确标注为"2019年诊断动量误判问题的关键
案例"，不是泛泛而谈）。前端`Stocks.tsx`页面顶部加了"本组研究候选池"
卡片直接展示这张表，位置在页面最上方，不需要额外翻找。

测试新增：`app/quant_v3/research_notes.py`的纯函数测试(2个，确认30支
全部有非空研究依据/风险点)+API集成测试(1个)。

## 验证

后端197/197测试全绿，前端`npm run build`通过。
