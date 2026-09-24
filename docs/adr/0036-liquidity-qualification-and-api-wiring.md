# 股票资格判断（流动性子集）+ 接入主API

## 资格判断：实现流动性子集，基本面子集如实标注未实现

V3 方案 2.2 节"沿用的资格条件"包含两类：①流动性/交易类（决策日正常
交易、60日日均成交额≥1亿、60日内至少50日有成交）——可以直接从我们
已有的行情数据验证；②基本面类（净利润/净资产/审计意见连续三年达标、
分红/ROE门槛，按红利/成长/周期分组各有不同标准）——需要额外抓取财务
报表数据，且我们最终策略已经不用官方三组结构（自由探索阶段扁平化成
单一"综合"组），套用分组化的基本面门槛意义不大。

按 A 阶段一贯的 `engineering_only` 原则，只实现①，②如实标注未实现，
不是简化成摆设。新增：

- `app/quant_v3/qualification.py::is_liquidity_qualified(bars, t_index)`：
  纯函数，决策日停牌/日均成交额不足/60日有效交易天数不足任一项不满足
  即不合格；历史不足60天也判不合格（不知道就不能记为合格）。
- `app/quant_v3/qualification_signal.py::QualificationSignalSource`：包装
  成 `(symbol, as_of) -> bool` 信号源。
- `RegressionRotationStrategy` 新增 `qualification_source` 参数（默认
  None，不启用，所有股票视为合格）：不合格的股票既不能被 Top-K 选中，
  也不能被动量兜底强制纳入、也不能靠持仓粘性保留——资格退出的优先级
  高于模型/动量判断，和 V3 方案 4.1 节的优先级顺序一致。

测试新增：`test_qualification.py`(5个)+`test_qualification_signal.py`
(3个)+策略级2个（合格性过滤Top-K/动量兜底），共184/184全绿。

## 诚实的代价：加了资格判断后总收益下降，但这是真实约束不是bug

| | 不加资格判断(ADR-0028/0030最终基线) | 加流动性资格判断 |
|---|---|---|
| 跑赢年数 | 6/7 | 6/7（持平） |
| 7年复合收益 | +428.0% | **+380.7%（下降）** |
| 2019超额 | -11.4% | -13.8%（略差） |

下降是真实的、预期内的代价——候选池里几支上市较晚或阶段性流动性偏低
的股票（比如新能源车/新上市次新股）在特定月份被过滤掉，损失了一部分
它们贡献的收益。**这个下降不应该被隐藏或者调整回避**——加了真实的
合规约束就要接受它的真实成本，这和整个项目一直坚持的诚实原则一致。
默认启用（生产配置里 `qualification_source` 传入真实的
`QualificationSignalSource`），不为了好看的回测数字而关掉。

## 接入主API

`/api/backtests` 原来内部固定构造 `MultiFactorStrategy` + `MarketDataService`
（SQLite后端，服务整个应用的通用股票目录）。`BacktestEngine` 构造函数
虽然类型标注是 `MarketDataService`，但是鸭子类型，只要求
`stocks()/prices()/fundamentals()/benchmark()/frame_data_mode()` 这几个
方法——`APhaseDataService`（读parquet）已经满足，此前"研报生成因子"
功能已经验证过这条路径可行。

**不替换SQLite**——SQLite 在这个应用里管的是策略配置/回测记录/交易
流水/AI审计这些"应用状态"，和我们策略用什么行情数据源是两回事。
`BacktestRun`/`Trade`/`BacktestEquity` 这几张表本来就是通用的，不管
底层策略对象是什么，跑完之后写入的方式完全一样。

实现方式：`Strategy` 表新增 `kind` 字段（默认 `"multifactor"`，走原有
路径不变；新值 `"quant_v3_regression"` 标记我们的最终策略），
`seed.py` 插入一条对应的 `Strategy` 行。`/api/backtests` 路由检测到
`strategy.kind == "quant_v3_regression"` 时跳过 SQLite 股票目录解析，
直接用 `APhaseDataService(parquet)` + 组装好的 `RegressionRotationStrategy`
（固定30支候选池、集成模型、动量兜底、资格判断全部按最终配置写死，
不接受 `payload.universe`/`custom_symbols` 等对该策略无意义的参数），
跑完后结果按原有方式写入 `BacktestRun`/`BacktestMetric`/`BacktestEquity`/
`Trade`，前端复用现有回测详情页面无需改造。
