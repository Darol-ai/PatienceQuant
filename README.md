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
3. 在策略中心使用“快速制定策略”设置策略名称、股票池、调仓频率和持仓数量；LightGBM 信号与风险控制默认开启，确有需要时再展开高级设置调整因子和阈值。
4. 在回测中心选择大盘股等预置股票池，或切换到“手动选择股票”，搜索并勾选至少 10 只股票。
5. 框定 2018-2025 等研究年份并运行回测，查看最终 Top N、累计/年化收益、回撤、Sharpe、年度收益和完整成交记录。
6. 点击任一入选股票，在独立价格曲线上查看该股票的 BUY/SELL 成交点；组合净值曲线也会标注买卖点。
7. 点击“导出 CSV”下载该回测的完整交易账本，包含日期、代码、名称、市场、研究组、行业、方向、数量、价格、金额、手续费、策略和触发原因。
8. 在自动交易执行一次调仓。
9. 在自动交易页开启按周/月/季度执行的自动调仓；后台轮询到期后会通过 Paper Broker 生成订单。
10. 在模拟盘查看策略净值、BUY/SELL 成交点、持仓、现金、浮动盈亏和订单账本。
11. 在 AI 投研解释一笔 BUY/SELL/HOLD 信号。

## LightGBM 策略信号

策略中心和回测中心共用同一套模型配置。模型输出三分类概率：`p_up`、`p_down` 和 `p_neutral`；默认使用 `p_up >= 0.60` 且 `p_down <= 0.25` 作为入场参考。回测的最终 Top N 会返回模型概率和 `model_signal`，方便审计每次选股依据。

LightGBM 是可选的生产依赖：

```bash
uv pip install --python .venv/bin/python -e './backend'
```

未安装 LightGBM 时，离线 Demo 会使用确定性的兼容信号模型，页面和回测备注会明确标记模型后端，不会伪造真实模型训练结果。

## 用户体验与结果审计

- 策略中心优先展示少量核心配置，高级参数按需调整。
- 回测中心支持搜索并勾选至少 10 只股票，研究区间默认覆盖 2018—2025。
- 结果统一展示最终 Top N、累计/年化收益、最大回撤、Sharpe、年度收益、风险闸门和完整成交记录。
- 成交账本可导出 CSV；数据源状态、Demo/Real 模式和模型后端会随结果保留。

## 前端界面预览

以下截图来自本地运行的 Demo Mode，统一使用 `1440×900` 短视口（未使用整页长截图），每张图只保留一个功能的首屏或结果视图：

<table>
  <tr>
    <td valign="top" width="50%">
      <strong>Dashboard · 收益与风险总览</strong><br>
      总资产、年化收益、最大回撤、Sharpe、策略/沪深300净值和买卖点。
      <br><br>
      <img src="docs/screenshots/dashboard.png" alt="Dashboard：收益与风险总览" width="520">
    </td>
    <td valign="top" width="50%">
      <strong>股票池 · 搜索、评分与行业分布</strong><br>
      1,000+ 示例股票、研究组、行业、综合评分和 BUY/HOLD/SELL 信号。
      <br><br>
      <img src="docs/screenshots/stock-pool.png" alt="股票池：搜索、评分与行业分布" width="520">
    </td>
  </tr>
  <tr>
    <td valign="top">
      <strong>策略中心 · 快速制定策略与 LightGBM</strong><br>
      先设置策略名称、股票池、调仓频率和持仓数量；LightGBM 信号层与风险控制默认开启，高级参数按需展开。
      <br><br>
      <img src="docs/screenshots/strategy-center.png" alt="策略中心：快速制定策略与 LightGBM 默认开启" width="520">
    </td>
    <td valign="top">
      <strong>回测中心 · 搜索并勾选股票</strong><br>
      切换到手动股票池，搜索或筛选股票并勾选至少 10 只；研究区间可直接选择 `2018-01-01` 至 `2025-12-31`。
      <br><br>
      <img src="docs/screenshots/backtest-center.png" alt="回测中心：手动搜索并勾选至少 10 只股票" width="520">
    </td>
  </tr>
  <tr>
    <td valign="top">
      <strong>回测结果 · Top N、收益与风险指标</strong><br>
      展示候选池与最终 Top N、累计/年化收益、Sharpe、最大回撤、沪深300对比、年度收益和完整成交记录。
      <br><br>
      <img src="docs/screenshots/backtest-result.png" alt="回测结果：最终 Top N、收益指标与风险闸门" width="520">
    </td>
    <td valign="top">
      <strong>自动交易 · 调仓订单与触发原因</strong><br>
      从策略评分到目标权重、风险控制、BUY/SELL/HOLD 和 Paper Broker 执行。
      <br><br>
      <img src="docs/screenshots/auto-trading.png" alt="自动交易：调仓订单与触发原因" width="520">
    </td>
  </tr>
  <tr>
    <td valign="top">
      <strong>模拟盘 · 资产、现金与持仓收益</strong><br>
      展示策略净值、资金变化、持仓市值、浮动盈亏和成交点。
      <br><br>
      <img src="docs/screenshots/paper-trading.png" alt="模拟盘：资产、现金与持仓收益" width="520">
    </td>
    <td valign="top">
      <strong>股票详情 · 多年份价格曲线</strong><br>
      支持 1/3/5 年及全部历史、日/周/月粒度切换，并保留关键价格区间。
      <br><br>
      <img src="docs/screenshots/stock-detail.png" alt="股票详情：多年份价格曲线" width="520">
    </td>
  </tr>
  <tr>
    <td valign="top">
      <strong>AI 投研 · 可审计的交易解释</strong><br>
      基于评分、因子、行业权重和风险指标生成解释；AI 只解释，不直接下单。
      <br><br>
      <img src="docs/screenshots/ai-research.png" alt="AI 投研：可审计的交易解释" width="520">
    </td>
  </tr>
</table>

策略风险预算

默认 V3 不是“保证收益”模型，而是把收益目标和风险约束同时纳入可验证规则：

- 沪深300相对强度 + 动量增强，配合估值/质量的低吸与过热减仓；
- 目标年化波动率、最大回撤预算、回撤刹车暴露可在策略中心修改；
- 当历史组合回撤或滚动波动率达到阈值时，回测和模拟盘都会按同一规则缩小股票暴露；
- 回测结果会记录风险闸门、回撤刹车和波动率缩放的触发次数，实际收益和最大回撤仍以所选区间、交易成本和数据模式为准，未来表现不保证。
