# 改造起点从 AlphaStock_A + vnpy 换成 PatienceQuant

## 背景

[[ADR-0001]]/[[ADR-0002]] 定的是 fork `AlphaStock_A`(FastAPI+Vue3 骨架)+ `vnpy.alpha`
(选股/回测引擎)。用户又找到一个已经跑通的全栈参考项目 `PatienceQuant`：8 个前端页面
(Dashboard/股票池/策略中心/回测中心/模拟盘/自动交易/AI 解释/个股详情)、30+ 接口、
SQLAlchemy 数据模型（Stock/StockPrice/Fundamental/Strategy/BacktestRun/Portfolio/Position/
Order/Trade/Signal/DailyAccount）、Docker 化、自带测试。经过对照分析：它的骨架完整度
远超我们从零搭的 AlphaStock_A+vnpy 组合，但策略内核是它自己发明的通用多因子+LightGBM
简化方案，不是我们组的 V3。

（命名提醒：`PatienceQuant` 自己的 `StrategyConfig.name` 默认值叫"沪深300增强趋势价值
成长策略 V3"——这是它自己的版本号，和我们组的《选股与LightGBM模拟交易方案_V3.md》
是两个不相关的"V3"，写文档/代码时注意区分，不要混用。）

## 决定

改造起点换成 `PatienceQuant`。放弃 fork `AlphaStock_A` 和搭建 `vnpy`/`vnpy.alpha` 环境的
既定计划（[[ADR-0001]]、[[ADR-0002]] 中和"用哪个仓库当骨架"相关的部分作废，其余关于
"愿景文档只做纵切""不做 Rust/ClickHouse/Spark"等范围决定继续有效）。

已经用 TDD 写好的 V3 策略模块（`src/quant/labels.py`、`risk_exits.py`、`position_state.py`、
`budget.py`、`entry_signal.py`，58 条测试）原样保留其内部逻辑，迁移进
`PatienceQuant/backend/app/`，替换掉现有的 `app/strategies/multifactor.py` +
`lightgbm_signal.py`；新写一个实现 `BaseStrategy` 接口的 `V3Strategy`，产出符合
`StrategyResult` 契约的结果，这样 `backtest/engine.py`、`portfolio/service.py`、
前端页面都不用改。

数据层：`PatienceQuant` 目前只有 AKShare（走 eastmoney 接口，这台机器上验证过会被限流/
连接重置，[[docs/setup.md]] 记过）。补一个 BaoStock provider 作为更稳定的选项，复用
`quant.baostock_adapter` 里已经测过的规范化逻辑。存储沿用它现成的 SQLAlchemy/SQLite，
不再单独搭 DuckDB——没有这层 ORM 抽象时 DuckDB 是合理选择，现在有更省事的路径就不重复
造轮子。

## 理由

`PatienceQuant` 缺的正是我们已经做的（V3 策略的精确规则），我们缺的正是它已经做好的
（完整前后端骨架、Docker、8 个页面、30+ 接口、自动调仓调度、CSV 导出）。补它缺的这块，
比在 AlphaStock_A 上重新搭它已有的这些骨架工作量小得多。
