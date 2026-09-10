# PatienceQuant · 低频智能量化投资系统

一个面向 A 股长期低频投资的全栈 MVP：

```text
数据 → 股票池 → 多因子策略 → 回测 → 模拟盘 → 自动调仓 → 交易记录 → Dashboard → AI 解释
```

## 快速启动

Docker 后端固定使用 Python 3.11。`make setup` 会优先探测本机 Python 3.11；当前机器若只有 Python 3.9，也会自动兼容回退。前端需要 Node 20+：

```bash
make setup
make backend   # http://localhost:8000
make frontend  # http://localhost:5173
```

或者使用 Docker：

```bash
docker compose up --build
# 浏览器打开 http://localhost:8080
```

默认使用确定性离线 Demo 数据，页面会明确标记 `DEMO MODE`。如果需要尝试 AKShare：

```bash
uv pip install --python .venv/bin/python -e './backend[real-data]'
DATA_MODE=real make backend
```

AKShare 访问失败会自动回退 Demo Provider，因此不影响离线演示。

在“回测中心 → 手动选择股票”中，可以切换 `AKShare` 目录，直接用中文名称或股票代码搜索并添加股票。点击“刷新 AKShare 目录”可将代码/名称目录缓存到 SQLite；点击“同步所选行情”可按当前回测区间尝试下载选中股票的前复权日线。若网络、限流或字段变化导致 AKShare 不可用，页面会回退到本地目录和确定性 Demo fallback，并明确保留数据源状态。

如果希望 Docker 默认优先使用 AKShare：

```bash
DATA_MODE=real docker compose up --build
```

## 模块边界

- `backend/app/data`：统一市场数据 Provider 和 SQLite 缓存
- `backend/app/factors`：因子计算、横截面标准化、缺失值填充
- `backend/app/strategies`：策略配置、选股和目标权重
- `backend/app/backtest`：无未来数据泄漏的月度/周度/季度回测
- `backend/app/portfolio`：Paper Trading 账户、订单和成交账本
- `backend/app/ai`：规则解释器和可选 OpenAI-compatible 适配器
- `frontend/src`：企业级暗色量化 Dashboard

参考项目的复用边界：借鉴 Qlib 的数据/因子/回测分层、FinRL-X 的权重中心契约和 FinRL 的交易/账本思路；不把大型框架整体嵌入产品。

## 演示路径

1. 打开 Dashboard，查看总资产、净值和信号。
2. 在股票池查看 5 个研究组、1,000 只跨行业 Demo 股票、171 个细分行业和行业分析图。
3. 在策略中心修改因子权重。
4. 在回测中心选择大盘股等预置股票池，或切换到“手动选择股票”，搜索并勾选至少 10 只股票。
5. 框定 2018-2025 等研究年份并运行回测，查看最终 Top N、累计/年化收益、回撤、Sharpe、年度收益和完整成交记录。
6. 点击任一入选股票，在独立价格曲线上查看该股票的 BUY/SELL 成交点；组合净值曲线也会标注买卖点。
7. 点击“导出 CSV”下载该回测的完整交易账本，包含日期、代码、名称、市场、研究组、行业、方向、数量、价格、金额、手续费、策略和触发原因。
8. 在自动交易执行一次调仓。
9. 在自动交易页开启按周/月/季度执行的自动调仓；后台轮询到期后会通过 Paper Broker 生成订单。
10. 在模拟盘查看策略净值、BUY/SELL 成交点、持仓、现金、浮动盈亏和订单账本。
11. 在 AI 投研解释一笔 BUY/SELL/HOLD 信号。

策略风险预算

默认 V3 不是“保证收益”模型，而是把收益目标和风险约束同时纳入可验证规则：

- 沪深300相对强度 + 动量增强，配合估值/质量的低吸与过热减仓；
- 目标年化波动率、最大回撤预算、回撤刹车暴露可在策略中心修改；
- 当历史组合回撤或滚动波动率达到阈值时，回测和模拟盘都会按同一规则缩小股票暴露；
- 回测结果会记录风险闸门、回撤刹车和波动率缩放的触发次数，实际收益和最大回撤仍以所选区间、交易成本和数据模式为准，未来表现不保证。
