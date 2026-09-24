# 回测报告指标补齐：对齐PRD §6.3报告指标要求

## 背景

对照《金融量化交易平台产品需求文档》做并行核查（课题作业要求.md之外的
第二份需求文档）。PRD绝大部分内容（OMS/EMS撮合分离、高频/低频分层、
AI模型治理的冠军挑战者/影子模式、多租户RBAC、策略市场发布审核、三层
事件级风控、灾备SLA、6-24个月分阶段实施）明显是企业级平台的愿景范围，
ADR-0001已将其定性为"愿景文档"，本课程项目只承诺对齐其§19.1功能验收
标准，这个边界继续维持，不因为看了全文就扩大范围。

逐条核对§19.1后确认：可复现性、订单/成交/持仓/资金可追踪、AI决策留痕
（`AIExplanation`表，ADR-0006）都已满足；策略发布审核、多租户权限隔离
两项明确超出范围，不做。

唯一发现的、成本可控且确实是真缺口的点：PRD §6.3列出的回测报告指标
比我们实际输出的更全——我们只有Sharpe、胜率、换手率，PRD还要求
Sortino、Calmar、信息比率、盈亏比、平均持有期。VaR/CVaR/蒙特卡洛/
归因分析成本明显更高，且不是纯计算就能加的，本次不做。

## 改动

`app/backtest/engine.py`的`BacktestEngine._metrics()`新增5个字段，全部
基于已有的每日净值曲线（`equity`/`benchmark`）和成交流水（`trades`）
计算，不需要新的数据源或架构改动：

- `sortino`：只用下行日收益的标准差（`ddof=0`）做分母，其余同Sharpe
  的年化换算；没有下行日或下行标准差为0时记0.0。
- `calmar`：`annual_return / abs(max_drawdown)`；最大回撤为0时记0.0。
- `information_ratio`：策略与基准的每日收益之差（超额收益序列）的
  年化均值/标准差；两者按DataFrame索引内连接对齐，避免了长度不一致
  的问题。
- `profit_loss_ratio`（盈亏比）、`avg_holding_days`（平均持有天数）：
  这两个指标必须先把"逐笔成交流水"（每次BUY/SELL一行，没有配对）
  重建成"完整的回合交易"（一次开仓到平仓，带盈亏和持有天数）才能算，
  引擎里原本没有这个重建逻辑（`entry_prices`只是运行时的持仓均价，
  用完即弃）。新增`BacktestEngine._round_trip_trades(trades)`：按
  symbol维护FIFO买入队列，每笔卖出按时间顺序匹配最早的未平仓买入批次
  （支持部分平仓、跨批次拆分），返回每笔回合交易的`(symbol, entry_date,
  exit_date, quantity, pnl, holding_days)`。`profit_loss_ratio`=平均
  盈利/平均亏损绝对值（没有亏损交易时记0.0而不是`inf`，因为这个dict
  直接序列化进JSON响应）；`avg_holding_days`=所有回合交易持有天数的
  均值。

这些新字段是`_metrics()`返回的普通dict的新key，下游序列化路径
（`BacktestMetric`是通用name/value行、API响应里的`"metrics"`字段是
未加约束的dict）不需要任何schema或数据库迁移改动就能自动透传。

前端`BacktestCenter.tsx`在原有的11个指标卡片后面追加了5个同样样式的
`<Metric>`卡片（Sortino/Calmar/信息比率/盈亏比/平均持仓天数）。

## 验证

- 新增单测：`test_backtest_metrics_include_sortino_calmar_and_information_ratio`
  （验证Sortino在有下行波动时不小于Sharpe、Calmar公式、IR在策略与
  基准走势不同时非零）、
  `test_backtest_metrics_profit_loss_ratio_and_holding_days_from_round_trips`
  （手算两笔完整回合交易的盈亏比和平均持有天数，核对准确）、
  `test_round_trip_trades_matches_fifo_lots_across_partial_sells`
  （验证FIFO批次匹配在跨批次部分平仓时按买入顺序正确拆分数量/持有
  天数/盈亏）。
- 后端全量测试200/200通过（新增3个，原197个不受影响）。
- 前端`npm run build`通过。
