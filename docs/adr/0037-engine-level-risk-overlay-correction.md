# 重大纠正：引擎级风控层这几个月一直在真实生效，之前"没有止损"的说法不准确

## 发现过程

给接入主API做准备、通读 `BacktestEngine.run()` 完整代码时发现：不管
传给引擎的 `strategy` 对象是什么，引擎在逐日回放循环里都会用调用方
传入的第二个参数 `strategy_config`（一个 `StrategyConfig` 实例）**额外
叠加一层通用风控**：

- `stop_loss=0.18`：任何持仓浮亏达到18%，强制清仓（`desired[symbol]=0`）。
- `turnover_band=0.03`：目标权重变动小于3%时按兵不动，压低换手。
- `target_volatility=0.22`：滚动63日年化波动率超过22%时按比例缩小暴露。
- `max_drawdown_budget=0.15`/`drawdown_brake_exposure=0.50`：组合回撤
  超过15%时把暴露砍到50%。

这套逻辑独立于策略对象自己的 `generate_weights()`/`on_daily_close()`，
只要调用 `BacktestEngine.run(config, strategy_config, ...)` 时
`strategy_config` 不是全部关掉这些字段，它就会生效。**我们从 ADR-0013
到 ADR-0036 的所有 `scripts/run_broad_regression_per_year_backtest.py`
及其变体，全部用的是 `StrategyConfig()`（默认值，没有关闭这些字段）**，
也就是说这套引擎级风控这几个月一直在真实生效。

## 和此前多篇ADR说法的矛盾

`app/strategies/regression_rotation_strategy.py` 顶部注释、以及
ADR-0010/0013/0022 等多处写过"我们的策略没有止损/跟踪止损，ADR-0010
已经验证止损层是冗余的"——**这个说法只在"策略对象自己没实现止损逻辑"
这个字面意义上成立，但不代表整个回测系统的真实行为**。实测确认(2019-
2025全量交易记录)：`保护性止损18%`触发了20次，`回撤刹车`触发46次，
`波动率缩放`触发上百次、把仓位压到65%~98%不等——这套没被写进任何一篇
ADR的机制，这几个月一直在真实影响我们汇报的每一个数字。

## 关掉后重测：表现反而更好

用 `StrategyConfig(stop_loss=0, turnover_band=0, target_volatility=0,
max_drawdown_budget=0)` 显式关闭这层引擎级风控，让实际运行的东西和
一直以来的文字描述保持一致：

| | 含引擎级风控(此前一直如此) | 关闭引擎级风控("纯"策略) |
|---|---|---|
| 跑赢年数 | 6/7 | 6/7（持平） |
| 7年复合收益 | +428.0% | **+450.96%（更高）** |
| 2020超额 | +13.8% | **+23.3%（明显更好）** |
| 2019超额 | -11.4% | -11.1%（基本持平） |

**这层意外生效的风控是净拖累，不是中性的**——它在2020这种普涨年份里
的止损/波动率缩放机制，卖飞了后续继续上涨的仓位，这和 ADR-0009/0010
诊断过的"止损在普涨行情里变成拖累"是同一个道理，只是这次是在引擎层面
以我完全没意识到的方式重演了一遍。

## 结论：更新最终配置和结论

`scripts/run_broad_regression_per_year_backtest.py` 已更新为显式传入
`PURE_STRATEGY_CONFIG = StrategyConfig(stop_loss=0, turnover_band=0,
target_volatility=0, max_drawdown_budget=0)`，不再用默认的
`StrategyConfig()`。加上 ADR-0036 的流动性资格判断后，**最终生产配置
的真实成绩：6/7年跑赢基准，7年复合收益+425.4%（基准+303.9%）**。

这次修正和 ADR-0019（基准计算bug）、ADR-0028（LightGBM随机种子bug）
是同一类教训——**"看起来理所当然、没有专门去验证的假设"最容易藏
bug**。这次是"以为自定义策略对象的行为就是回测的全部行为"，实际上
共享引擎还有一层没读到的默认逻辑在起作用。已经把这条经验也补进
`feedback_honest_backtest_reporting` 记忆：接入任何共享引擎/框架代码
时，必须通读它的完整执行路径，不能只看自己传入的参数够不够，要确认
框架本身有没有隐藏的默认行为。
