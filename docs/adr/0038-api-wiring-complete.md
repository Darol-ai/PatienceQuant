# 接入主API完成

## 实现

- `Strategy` 表新增 `kind` 字段（`app/db/models.py` + `app/db/session.py`
  轻量迁移，默认 `"multifactor"`，原有策略/路径行为不变）。
- `app/db/seed.py::_ensure_quant_v3_strategy`：插入一条
  `kind="quant_v3_regression"` 的策略行（"LightGBM动量增强策略(30支
  候选池)"），`stop_loss`/`turnover_band`/`target_volatility`/
  `max_drawdown_budget` 全部置0——ADR-0037证明的引擎级风控叠加层净
  拖累，这样`_strategy_config()`读出来的`StrategyConfig`天然就是"纯
  策略"配置，不需要在路由里特殊处理。
- `app/quant_v3/final_strategy.py`：把最终策略的完整装配逻辑（30支候选
  池+扩展历史+horizon=90 5模型集成+top_k_ratio=0.4+动量兜底+流动性
  资格判断）集中一处，`lru_cache`避免每次请求重新从磁盘加载35个
  LightGBM模型文件；`validate_backtest_date_range`拒绝2019-2025训练
  范围之外的回测请求（超出范围会退化成占位信号，不能悄悄跑出没有
  意义的结果）。
- `/api/backtests` 路由：检测到 `strategy.kind == "quant_v3_regression"`
  时跳过SQLite股票目录解析（`MarketDataService`/`_universe_symbols`），
  改用 `APhaseDataService(parquet)` + `build_final_strategy()`，跑完后
  走**和原有路径完全相同**的持久化代码（`BacktestRun`/`BacktestMetric`/
  `BacktestEquity`/`Trade`），前端复用现有回测详情页面，不需要新页面。
- `/api/strategies` 响应加了 `kind` 字段，前端 `BacktestCenter.tsx` 据此
  判断：选中这个策略时隐藏股票池/调仓周期/Top N/单股最大权重这几个对
  它无意义的控件，显示一条说明其固定配置和支持区间的提示。

## 验证

新增集成测试 `test_quant_v3_strategy_runs_through_the_shared_backtest_endpoint`：
真的通过 `TestClient` 打 `/api/backtests`，验证200状态码、真实交易记录、
30支候选池、以及超出2019-2025范围时返回422（不是悄悄跑出空结果）。
后端189/189全绿，前端 `npm run build` 通过。**没有改动SQLite的既有
表结构以外的任何东西**（只加了`kind`一列），原有 `MultiFactorStrategy`
路径的所有测试保持不变、全部通过，确认没有破坏既有功能。
