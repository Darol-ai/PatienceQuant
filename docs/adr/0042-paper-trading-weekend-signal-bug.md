# 模拟盘在非交易日调仓会悄悄跑出空结果（周末/节假日bug）

## 背景

按用户要求，对ADR-0040刚接入的"模拟盘跑我们自己的LightGBM策略"功能做
一次真实场景的手工验证（不只是单测/TestClient，是真的起后端、真的调
HTTP接口），因为此前只在API层面确认过接口结构对，从没有在这台环境的
真实"今天"（沙盒环境当前日期2026-09-12，是周六）下实际跑过一次调仓。

## 发现的问题

调用`POST /api/paper/rebalance`（`strategy_id`=我们的LightGBM策略，
不传`as_of`，默认取`date.today()`）返回`orders: []`、`signals`里全部
30支股票`score: null`、`qualified: false`、`target_weight: 0`——静默
产出一个"零调仓"，没有任何报错。

根因：`EnsembleRegressionWalkForwardSignalSource`/`MomentumSignalSource`/
`QualificationSignalSource`（`app/quant_v3/`下三个信号源）内部都是把
每支股票的行情按精确日期建索引（`{date: row_index}`），查询时用
`t_index = index_by_symbol[symbol].get(as_of)`——如果`as_of`当天没有
对应的行情记录（周末/节假日不开市，或者当天数据还没来得及入库），
`get()`返回`None`，后续`compute_features`直接跳过这支股票，最终这支
股票的分数是`None`。三个信号源全都会撞到这个问题，所以是全部30支
股票同时失效，不是个别股票的数据缺口。

`PaperTradingService.snapshot()`/`rebalance()`把`as_of`默认设成字面
`date.today()`（沙盒环境里就是真实系统时钟的"今天"），从不检查这天
是不是交易日——这是ADR-0040引入的，之前的默认策略路径用的是
`self.data.demo.as_of`（脚手架自带的demo数据服务自己维护的"当前
模拟日"，本来就保证落在交易日上），只有我们新接入的quant_v3分支绕开
了这层保护，直接用真实墙钟时间。

**为什么之前的集成测试没测出来**：`test_quant_v3_strategy_drives_paper_
trading_rebalance`（ADR-0040新增）里验证"持仓当前价是真的查到的、不是
退化成avg_cost"这条断言写成了`if account["positions"]: assert ...`——
测试跑的时候`account["positions"]`本来就是空的（因为测试同样没传
`as_of`，同样撞上了这个bug），条件为假，断言被短路掉，测试全绿但从没
真的执行过这条检查。这是一个"测试看起来测了，实际从没触发过关键路径"
的典型例子。

## 修复

`app/quant_v3/final_strategy.py`新增`latest_trading_day_on_or_before
(as_of)`：在历史行情里找`<= as_of`的最大日期，找不到就明确抛
`ValueError`（而不是让调用方在后面拿到一堆`None`分数）。

`app/portfolio/service.py`的`snapshot()`和`rebalance()`，在quant_v3
分支里，先用这个函数把`as_of`落到"这天或更早的最近一个真实交易日"，
再传给`build_final_strategy().generate_weights(...)`和`data.prices(...)`
——不管`as_of`是调用方显式传入的还是默认取的`date.today()`，都统一走
这层兜底，语义上就是"给我这天（或更早）最近一次收盘的状态"，这也是
金融软件里对非交易日请求的标准解释方式。

同时修正了那条被短路的测试：改成显式传一个明确早于"今天"的历史交易
日（2025-06-02）做调仓，这样默认`as_of`（今天，对应最近交易日）的
account快照拿到的`current_price`必然来自不同的一天，`current_price !=
avg_cost`这条断言才是真的在验证，而不是巧合相等或者巧合被跳过。同时把
`if account["positions"]:`这种条件断言去掉，改成硬性`assert
account["positions"]`——因为空持仓本身现在就是需要被测试直接拒绝的
情况，不该被当成"可以跳过后续检查"的分支。

新增4个针对`latest_trading_day_on_or_before`本身的单测（精确匹配、
周末回退到周五、超出历史范围报错、`build_final_strategy`在被兜底过的
周末日期上真的能算出非空权重）。

## 验证

真实起后端进程（不是TestClient），在这台环境的真实"今天"
（2026-09-12，周六）用curl实测：修复前`orders: []`、
`snapshot.positions` 长度0；修复后13笔真实BUY成交、持仓13支、价格
来自2026-09-11（周五）收盘。后端全量测试通过（新增8个：4个
`latest_trading_day_on_or_before`单测+已有集成测试的断言收紧）。
