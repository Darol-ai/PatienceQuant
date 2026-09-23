import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, Download, ExternalLink } from 'lucide-react'
import { Fragment, useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, dataModeLabel, formatMoney, formatPercent } from '../api'
import { BarChart, DrawdownChart, EquityChart, StockTradeChart } from '../components/Charts'
import { Card, ErrorState, LoadingState, PageHeader, PanelHeader } from '../components/UI'
import { DataNote } from '../components/DataNote'
import { describeSpec, timingText, universeLabel } from '../strategy'

// 回测报告（ADR-0051）：每次回测一个固定地址。数据全部来自 GET /backtests/{id}/result，
// 刚跑完和从历史里调出来的是同一份。
const TRADE_PAGE_SIZE = 25

export function BacktestReport() {
  const { id } = useParams()
  const runId = Number(id)
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { data: result, isLoading, isError, error } = useQuery({
    queryKey: ['backtest-result', runId],
    queryFn: async () => (await api.get(`/backtests/${runId}/result`)).data,
  })
  const { data: options } = useQuery({ queryKey: ['pipeline-options'], queryFn: async () => (await api.get('/pipeline/options')).data })
  const { data: pools } = useQuery({ queryKey: ['pools'], queryFn: async () => (await api.get('/pools')).data })
  const [selectedSymbol, setSelectedSymbol] = useState('')
  const [tradePage, setTradePage] = useState(1)
  const [exporting, setExporting] = useState(false)
  const [exportMessage, setExportMessage] = useState('')
  const [showAllCharts, setShowAllCharts] = useState(false)
  const [paperMessage, setPaperMessage] = useState('')
  const [explanations, setExplanations] = useState<Record<number, string>>({})

  useEffect(() => {
    if (!result || selectedSymbol) return
    const firstTraded = result.selected_stocks?.find((stock: any) => result.trades?.some((trade: any) => trade.symbol === stock.symbol))
    setSelectedSymbol(firstTraded?.symbol || result.selected_stocks?.[0]?.symbol || '')
  }, [result, selectedSymbol])

  const applyPaperMutation = useMutation({
    mutationFn: async () => (await api.post(`/backtests/${runId}/apply-paper`, { execute: true, reset_account: false, enable_automation: false }, { timeout: 600_000 })).data,
    onSuccess: data => {
      setPaperMessage(`已应用到模拟盘：${data.strategy?.name || '回测策略'} · 本次生成 ${data.rebalance?.orders?.length || 0} 笔订单`)
      queryClient.invalidateQueries({ queryKey: ['paper-account'] })
    },
    onError: (error: any) => setPaperMessage(error?.response?.data?.detail || '应用到模拟盘失败'),
  })
  const explain = useMutation({
    mutationFn: async (tradeId: number) => ({ tradeId, data: (await api.post(`/backtests/${runId}/trades/explain`, { trade_id: tradeId }, { timeout: 120_000 })).data }),
    onSuccess: ({ tradeId, data }) => setExplanations(current => ({ ...current, [tradeId]: data.text })),
    onError: (error: any) => setExportMessage(error?.response?.data?.detail || '解释失败'),
  })
  const { data: stockChart, isFetching: stockChartLoading } = useQuery({
    queryKey: ['backtest-stock-chart', runId, selectedSymbol],
    queryFn: async () => (await api.get(`/backtests/${runId}/stocks/${selectedSymbol}/chart`)).data,
    enabled: !!result && !!selectedSymbol,
  })
  const { data: allStockCharts, isFetching: allStockChartsLoading } = useQuery({
    queryKey: ['backtest-stock-charts', runId],
    queryFn: async () => (await api.get(`/backtests/${runId}/stocks/charts`)).data,
    enabled: !!result && showAllCharts,
  })

  if (isLoading) return <LoadingState text="正在加载回测报告…" />
  if (isError || !result) return <ErrorState text={(error as any)?.response?.data?.detail || '回测报告加载失败'} />

  const trades = result.trades || []
  const totalTradePages = Math.max(1, Math.ceil(trades.length / TRADE_PAGE_SIZE))
  const visibleTrades = trades.slice((tradePage - 1) * TRADE_PAGE_SIZE, tradePage * TRADE_PAGE_SIZE)
  const initialCapital = Number(result.initial_capital)
  const finalAssets = Number(result.final_assets)
  const totalProfit = Number(result.total_profit)
  const overallReturn = Number(result.overall_return)
  const downloadTrades = async () => {
    setExporting(true)
    setExportMessage('')
    try {
      const response = await api.get(`/backtests/${runId}/trades.csv`, { responseType: 'blob' })
      const blobUrl = URL.createObjectURL(response.data)
      const anchor = document.createElement('a')
      anchor.href = blobUrl
      anchor.download = `backtest_${runId}_trades.csv`
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(blobUrl)
      setExportMessage(`已导出 ${trades.length} 条完整成交记录`)
    } catch {
      setExportMessage('导出失败，请稍后重试')
    } finally {
      setExporting(false)
    }
  }

  return (
    <>
      <PageHeader
        eyebrow={`回测报告 #${runId}`}
        title={<span className="title-with-back"><Link to={`/library/${result.strategy_id}`} className="icon-button"><ArrowLeft size={16}/></Link>{result.strategy_name}</span>}
        description={`${result.start_date} ~ ${result.end_date} · ${universeLabel(result.universe, pools)} · 初始 ${formatMoney(initialCapital)} · 手续费 ${formatPercent(result.commission ?? 0, 2)}、滑点 ${formatPercent(result.slippage ?? 0, 2)} · ${describeSpec(result.strategy_config?.spec, options)}`}
        actions={<Link className="secondary-button" to={`/backtest?strategy_id=${result.strategy_id}`}>换个条件再回测</Link>}
      />
      <DataNote strategyId={result.strategy_id} start={result.start_date} end={result.end_date} universe={result.universe}
        totalReturn={result.metrics?.total_return} benchmarkReturn={result.metrics?.benchmark_return}/>
      {result.notes?.length > 0 && <div className="strategy-note" style={{ marginBottom: 16 }}><div><b>这次回测的说明</b>{result.notes.slice(0, 6).map((note: string) => <p key={note}>{note}</p>)}</div></div>}

          <div className="backtest-result-callout">
            <b>
              {result.start_date} — {result.end_date}，总体收益率 {formatPercent(overallReturn)}
            </b>
            <span>
              初始资金 {formatMoney(initialCapital)} → 期末总资产 {formatMoney(finalAssets)}，总收益 {formatMoney(totalProfit)}；候选池{' '}
              {result.universe_size || result.custom_symbols?.length || '—'} 只，买卖费用已计入复利资产曲线。
            </span>
          </div>
          <div className="metric-grid compact">
            <Metric label="期末总资产" value={formatMoney(finalAssets)} />
            <Metric label="总体收益率" value={formatPercent(overallReturn)} />
            <Metric label="总收益额" value={formatMoney(totalProfit)} />
            <Metric label="年化收益" value={formatPercent(result.metrics.annual_return)} />
            <Metric label="最大回撤" value={formatPercent(result.metrics.max_drawdown)} />
            <Metric label="Sharpe" value={Number(result.metrics.sharpe || 0).toFixed(2)} />
            <Metric label="同期沪深300" value={formatPercent(result.metrics.benchmark_return)} />
            <Metric label="超额收益" value={formatPercent(result.metrics.excess_return)} />
            <Metric label="胜率" value={formatPercent(result.metrics.win_rate)} />
            <Metric label="成交次数" value={Number(result.metrics.trade_count || 0).toFixed(0)} />
            <Metric label="换手率" value={formatPercent(result.metrics.turnover)} />
            <Metric label="Sortino" value={Number(result.metrics.sortino || 0).toFixed(2)} />
            <Metric label="Calmar" value={Number(result.metrics.calmar || 0).toFixed(2)} />
            <Metric label="信息比率" value={Number(result.metrics.information_ratio || 0).toFixed(2)} />
            <Metric label="盈亏比" value={Number(result.metrics.profit_loss_ratio || 0).toFixed(2)} />
            <Metric label="平均持仓天数" value={Number(result.metrics.avg_holding_days || 0).toFixed(1)} />
          </div>

          <Card className="risk-summary-card">
            <PanelHeader
              title="策略执行摘要"
              subtitle="这次回测实际执行的规格；可以把同样的规格和股票池应用到模拟盘。"
              action={
                <div className="page-actions">
                  <button
                    className="primary-button"
                    onClick={() => applyPaperMutation.mutate()}
                    disabled={applyPaperMutation.isPending}
                  >
                    <ExternalLink size={14} />
                    {applyPaperMutation.isPending ? '应用中…' : '应用到模拟盘并执行'}
                  </button>
                  <button className="secondary-button" onClick={() => navigate('/paper')}>
                    查看模拟盘
                  </button>
                </div>
              }
            />
            {paperMessage && <div className="export-message">{paperMessage}</div>}
            <div className="risk-summary-grid">
              <div><span>打分方式</span><b>{result.strategy_config?.scorer === 'model' ? `模型 · ${(result.strategy_config?.models || []).length} 个` : '因子权重'}</b></div>
              <div><span>选股规则</span><b>{result.strategy_config?.top_pct ? `前 ${formatPercent(result.strategy_config.top_pct)}` : `前 ${result.strategy_config?.holdings_count ?? '—'} 名`}</b></div>
              <div><span>权重方案</span><b>{result.strategy_config?.weighting === 'score' ? '按分数加权' : '等权'} · 单股上限 {formatPercent(result.strategy_config?.max_weight ?? 1)}</b></div>
              <div><span>择时信号</span><b>{result.strategy_config?.spec ? timingText(result.strategy_config.spec, options) : '—'}</b></div>
              <div><span>换手阈值</span><b>{formatPercent(result.strategy_config?.turnover_band || 0)}</b></div>
              <div><span>择时降仓调仓</span><b>{Number(result.risk_summary?.risk_gate_rebalances || 0)} 次</b></div>
              <div><span>平均股票仓位</span><b>{formatPercent(result.risk_summary?.average_target_exposure || 0)}</b></div>
              <div><span>数据模式</span><b>{dataModeLabel(result.data_mode)}</b></div>
            </div>
          </Card>

          <Card>
            <PanelHeader
              title={`最终 Top ${result.selected_stocks?.length || 0} 入选股票`}
              subtitle={
                result.universe === 'custom'
                  ? `用户候选池 ${result.universe_size || result.custom_symbols?.length || 0} 只 → 综合评分后最终持仓 Top N；点击任一股票查看它在本次回测年限内的价格与 BUY/SELL 点。`
                  : '点击任一股票，查看它在本次回测年限内的价格与 BUY/SELL 点。'
              }
            />
            <div className="selected-stock-pills">
              {result.selected_stocks?.map((stock: any) => (
                <button
                  key={stock.symbol}
                  className={selectedSymbol === stock.symbol ? 'active' : ''}
                  onClick={() => setSelectedSymbol(stock.symbol)}
                >
                  <b>{stock.name}</b>
                  <span>
                    {stock.symbol}{stock.industry ? ` · ${stock.industry}` : ''}
                  </span>
                  <small>
                    第{stock.rank}名 · 目标权重 {formatPercent(stock.target_weight)} · {stock.action || 'HOLD'}
                  </small>
                </button>
              ))}
            </div>
          </Card>

          <Card>
            <PanelHeader
              title={stockChart ? `${stockChart.name} ${stockChart.symbol} · 个股买卖曲线` : '个股买卖曲线'}
              subtitle={
                stockChart
                  ? `${stockChart.start_date} — ${stockChart.end_date} · ${stockChart.trade_count} 个真实回测成交点 · 个股区间涨跌 ${formatPercent(stockChart.return)}`
                  : '请选择入选股票'
              }
            />
            {stockChartLoading ? (
              <LoadingState text="正在加载个股成交曲线…" />
            ) : stockChart?.prices?.length ? (
              <StockTradeChart data={stockChart.prices} trades={stockChart.trade_markers} />
            ) : (
              <div className="empty-chart">该股票在回测区间暂无行情。</div>
            )}
          </Card>

          <Card>
            <PanelHeader
              title="全部入选股票买卖点审计"
              subtitle="每张曲线都使用本次回测区间的复权价格，并叠加实际成交 BUY / SELL 点；可逐只核对策略执行。"
              action={
                <button className="secondary-button" onClick={() => setShowAllCharts(value => !value)}>
                  {showAllCharts ? '收起全部曲线' : `展开全部 ${result.selected_stocks?.length || 0} 只`}
                </button>
              }
            />
            {showAllCharts && (
              <>
                {allStockChartsLoading && <LoadingState text="正在加载全部股票买卖曲线…" />}
                {!allStockChartsLoading && allStockCharts?.items?.length ? (
                  <div className="stock-chart-grid">
                    {allStockCharts.items.map((chart: any) => (
                      <div className="stock-chart-tile" key={chart.symbol}>
                        <div className="stock-chart-tile-head">
                          <div>
                            <b>{chart.name}</b>
                            <span>{chart.symbol} · {chart.trade_count} 个成交点</span>
                          </div>
                          <strong className={chart.return >= 0 ? 'text-mint' : 'text-rose'}>
                            {formatPercent(chart.return)}
                          </strong>
                        </div>
                        {chart.prices?.length ? (
                          <StockTradeChart data={chart.prices} trades={chart.trade_markers} height={270} />
                        ) : (
                          <div className="empty-chart compact">该股票暂无可用行情</div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  !allStockChartsLoading && <div className="muted-empty">暂无入选股票曲线。</div>
                )}
              </>
            )}
          </Card>

          <div className="dashboard-grid">
            <Card className="span-8">
              <PanelHeader
                title="组合策略净值 vs 沪深300"
                subtitle={`${dataModeLabel(result.data_mode)} · 组合买卖点来自实际成交`}
              />
              <EquityChart data={result.equity} trades={result.trade_markers || result.trades} />
            </Card>
            <Card className="span-4">
              <PanelHeader title="回撤曲线" subtitle="回测期间风险变化" />
              <DrawdownChart data={result.equity} />
            </Card>
          </div>

          <div className="dashboard-grid">
            <Card className="span-4">
              <PanelHeader title="年度收益" />
              <BarChart data={result.annual_returns} />
            </Card>
            <Card className="span-8">
              <PanelHeader
                title="完整交易记录"
                subtitle={`${trades.length} 条成交记录 · 点击股票代码切换个股曲线`}
                action={
                  <button className="secondary-button" onClick={downloadTrades} disabled={exporting}>
                    <Download size={14} />
                    {exporting ? '正在导出…' : '导出 CSV'}
                  </button>
                }
              />
              {exportMessage && <div className="export-message">{exportMessage}</div>}
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>日期</th>
                      <th>股票</th>
                      <th>方向</th>
                      <th>数量</th>
                      <th>价格</th>
                      <th>金额</th>
                      <th>手续费</th>
                      <th>原因</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleTrades.map((trade: any, index: number) => (
                      <Fragment key={`${trade.trade_date}-${trade.symbol}-${index}`}>
                      <tr>
                        <td>{trade.trade_date}</td>
                        <td>
                          <button className="stock-link-button" onClick={() => setSelectedSymbol(trade.symbol)}>
                            {trade.symbol}
                          </button>
                        </td>
                        <td>
                          <span className={`order-side ${trade.side.toLowerCase()}`}>{trade.side}</span>
                        </td>
                        <td>{Number(trade.quantity).toLocaleString()}</td>
                        <td>{Number(trade.price).toFixed(2)}</td>
                        <td>{formatMoney(trade.amount)}</td>
                        <td>{formatMoney(trade.fee)}</td>
                        <td>{trade.reason}</td>
                        <td>{trade.id && <button className="stock-link-button" disabled={explain.isPending} onClick={() => explain.mutate(trade.id)}>解释</button>}</td>
                      </tr>
                      {explanations[trade.id] && <tr className="explain-row"><td colSpan={9}>{explanations[trade.id]}</td></tr>}
                      </Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
              {trades.length > TRADE_PAGE_SIZE && (
                <div className="pagination">
                  <span>
                    第 {tradePage} / {totalTradePages} 页 · 共 {trades.length} 条
                  </span>
                  <div>
                    <button
                      className="secondary-button"
                      disabled={tradePage <= 1}
                      onClick={() => setTradePage(page => Math.max(1, page - 1))}
                    >
                      上一页
                    </button>
                    <button
                      className="secondary-button"
                      disabled={tradePage >= totalTradePages}
                      onClick={() => setTradePage(page => Math.min(totalTradePages, page + 1))}
                    >
                      下一页
                    </button>
                  </div>
                </div>
              )}
            </Card>
          </div>
    </>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card small">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}
