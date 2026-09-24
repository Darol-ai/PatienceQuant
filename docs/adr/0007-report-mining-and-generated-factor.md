# 研报挖掘两项新能力：①总结+策略参考 ②生成因子并接入回测

## 决定

在已有的 AI 助手体系（策略助手/风险助手，见 [[ADR-0006]]）基础上加两个能力，
对应 PRD 9.1 节"市场摘要"和"因子发现"：

1. **`app/ai/report_analysis.py`**：上传研报（.pdf/.txt/.md，`app/ai/report_extraction.py`
   负责解析、截断到 12000 字）→ LLM 给出摘要 + 关键点 + 策略参考。只解读，
   不生成代码、不自动创建策略。
2. **`app/ai/report_factor.py`** + **`app/quant_v3/factor_backtest.py`**：判断研报是否
   对因子/建模有参考价值，如果有，把逻辑翻译成一个**线性因子**——LLM 只能
   从 `quant_v3.model_training.FEATURE_COLUMNS`（V3 已用的 14 项白名单特征）
   里选、给 -1~1 的权重，越界或编造的字段直接丢弃（`sanitize_feature_weights`）。
   生成的因子权重接进 `GeneratedFactorStrategy`（简单的 top-N 打分等权策略，
   不做止损/冷静期这类状态机），通过 `run_generated_factor_backtest()` 真的
   在 A 阶段 10 支股票历史数据上跑一次 `BacktestEngine`，把绩效指标随审计
   记录一起返回。

两个接口都复用 [[ADR-0006]] 建的审计表 `AIExplanation`（`explanation_type`
分别是 `report_analysis`/`report_factor`），落一条记录、返回 `audit_id`。

## 理由：不执行 LLM 生成的任意代码

"生成算法直接接回测"最直接的做法是让 LLM 写 Python 代码再 `exec`，但这套
系统没有 PRD 提到的策略执行沙箱（隔离 CPU/内存/网络/系统调用），执行任意
生成代码是不可控的攻击面。改成"LLM 只能从白名单特征里选权重拼线性因子"，
既满足"确实被接进回测系统跑"的要求，又把风险面收窄到和 `strategy_assistant.py`
一样的"白名单字段 + 数值裁剪"模式，不新增攻击面。

## 已知结果，如实记录

真机验证：功能①摘要/关键点/策略参考都正常，且模型自己加了"不构成投资建议"
的免责声明。功能②用一份动量+波动率主题的测试研报，LLM 正确映射出
`return_20d`(+0.6)/`volatility_60d`(-0.5)/`max_drawdown_60d`(-0.4)/
`volume_ratio_20d`(+0.3)，回测跑出 142 笔交易、总收益 +62.9%。**这个数字
不代表策略有效**——生成的因子没有经过 V3 那套训练/校准/样本外验证，是
"研报文字 → 权重 → 无风控月度调仓"直接跑出来的结果，大概率是这个小样本
（10 支精选大盘股 + 这两年行情）本身的贝塔，不是被验证过的 alpha。产品
需求本身也不要求这个因子真的赚钱，只要求链路是真的在跑——链路是真的。
