# 接入API之后的代码审查：4个问题，全部修复

用户要求接入主API后回头做一次审查（"看看还有什么bug或者可以优化的"）。
发现4个问题，全部修复：

## 1. 并发bug：有状态策略被缓存成进程级单例

`app/quant_v3/final_strategy.py::build_final_strategy()` 原来用
`lru_cache(maxsize=1)` 把整个 `RegressionRotationStrategy` 实例缓存成
单例。`/api/backtests` 路由是同步 `def`，FastAPI 会丢进线程池执行，
两个并发请求会在不同线程里真正同时调用同一个策略对象的
`generate_weights()`，读写其内部可变字典 `_sticky_hold_remaining`。

**修复**：拆成两层——`_cached_signal_sources()`（模型/历史索引这些
无状态、加载昂贵的部分，继续用 `lru_cache` 跨请求共享）+
`build_final_strategy()`（不再缓存，每次调用构造一个全新的策略实例，
构造本身很便宜，只是包了几个共享引用）。新增测试
`test_build_final_strategy_returns_a_fresh_instance_each_call` 锁定
"每次调用返回不同实例"这个行为。

## 2. Schema约束和我们的意图冲突

`StrategyPayload` 的 `stop_loss`/`target_volatility`/`max_drawdown_budget`
要求 `gt=0`，但我们给LightGBM策略特意存的是 `0`（ADR-0037：关闭引擎级
风控叠加层，那层是净拖累）。`PUT /api/strategies/{id}` 对payload每个
字段做无条件覆盖式写入，如果这个端点被调用（前端目前不调用，但
FastAPI默认开着的`/docs`能直接调），会把这些字段静默填回schema默认值，
重新引入已经修复的问题，且没有任何报错提示。

**修复**：①schema约束从 `gt=0` 放宽到 `ge=0`（0本身是"关闭这条风控"
的合法取值，不是非法输入）；②`update_strategy` 加了守卫——检测到
`strategy.kind != "multifactor"` 时直接拒绝（422），这类策略的参数
由代码固定管理，通用编辑接口不应该碰。新增测试
`test_quant_v3_strategy_params_are_protected_from_generic_edit_endpoint`。

## 3. 体验缺口：切换策略不同步日期范围

`BacktestCenter.tsx` 切换策略下拉框只更新了 `strategy_id`，日期字段
停留在上一个策略的区间——LightGBM策略只支持2019-2025，切过去很容易
带着上一个策略遗留的区间直接触发后端422报错，用户不清楚原因。

**修复**：切换策略时把 `research_start_date`/`research_end_date` 同步
到表单的开始/结束日期。`npm run build` 验证通过。

## 4. 性能优化：批量预测代替逐股票单独predict

`EnsembleRegressionSignalSource.__call__` 原来每支股票、每个交易日都
单独构造一次单行DataFrame、调用5个模型分别predict——30支股票一次调仓
就是150次predict调用。

**修复**：加了按天缓存（`_predict_all_for_day`，和 `model_signal.py`
的 `_cross_sectional_features_for_day` 是同一个思路）：第一次查某天时
把当天有数据的全部股票一次性批量predict（每个模型1次，共5次），之后
同一天的其余股票直接查缓存。**验证是纯性能优化、不改变任何输出**：
重跑 `scripts/run_broad_regression_per_year_backtest.py`，7年复合收益
和逐年数字与优化前逐位相同（425.4% vs 基准303.9%，6/7年跑赢）；API
集成测试耗时从约15秒降到约8秒（约2倍提速）。

## 验证

后端191/191测试全绿（新增3个：单例修复1个+schema守卫1个+已有的最终
策略测试补充），前端 `npm run build` 通过。
