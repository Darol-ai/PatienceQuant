import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { CalendarRange, Check, Download, ExternalLink, Play, RefreshCw, RotateCcw, Search, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, dataModeLabel, formatMoney, formatPercent } from '../api'
import { BarChart, DrawdownChart, EquityChart, StockTradeChart } from '../components/Charts'
import { Card, ErrorState, LoadingState, PageHeader, PanelHeader } from '../components/UI'

type BacktestForm = {
  strategy_id: number
  universe: string
  custom_symbols: string[]
  start_date: string
  end_date: string
  initial_capital: number
  rebalance_frequency: string
  holdings_count: number
  max_weight: number
  commission: number
  slippage: number
}

type StockOption = {
  symbol: string
  name: string
  exchange: string
  group: string
  industry: string
  sector: string
  source?: string
  score?: number
}

const TRADE_PAGE_SIZE = 25

export function BacktestCenter() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { data: strategies } = useQuery({
    queryKey: ['strategies'],
    queryFn: async () => (await api.get('/strategies')).data,
  })
  const { data: universes } = useQuery({
    queryKey: ['universes'],
    queryFn: async () => (await api.get('/universes')).data,
  })
  const [form, setForm] = useState<BacktestForm>({
    strategy_id: 0,
    universe: 'large_cap',
    custom_symbols: [],
    start_date: '2018-01-01',
    end_date: '2025-12-31',
    initial_capital: 1000000,
    rebalance_frequency: 'monthly',
    holdings_count: 10,
    max_weight: 0.15,
    commission: 0.001,
    slippage: 0.0005,
  })
  const [result, setResult] = useState<any>(null)
  const [selectedSymbol, setSelectedSymbol] = useState('')
  const [stockSearch, setStockSearch] = useState('')
  const [stockGroup, setStockGroup] = useState('')
  const [stockExchange, setStockExchange] = useState('')
  const [tradePage, setTradePage] = useState(1)
  const [exporting, setExporting] = useState(false)
  const [exportMessage, setExportMessage] = useState('')
  const [syncMessage, setSyncMessage] = useState('')
  const [showAllCharts, setShowAllCharts] = useState(false)
  const [selectedStocks, setSelectedStocks] = useState<Record<string, StockOption>>({})
  const [directoryMode, setDirectoryMode] = useState<'auto' | 'local' | 'akshare'>('auto')
  const [debouncedStockSearch, setDebouncedStockSearch] = useState('')
  const [paperMessage, setPaperMessage] = useState('')

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedStockSearch(stockSearch.trim()), 280)
    return () => window.clearTimeout(timer)
  }, [stockSearch])

  useEffect(() => {
    const defaultStrategy = strategies?.find((item: any) => item.is_default) || strategies?.[0]
    if (!defaultStrategy) return
    setForm(current =>
      current.strategy_id === defaultStrategy.id
        ? current
        : {
            ...current,
            strategy_id: defaultStrategy.id,
            universe: defaultStrategy.universe || current.universe,
            rebalance_frequency: defaultStrategy.rebalance_frequency || current.rebalance_frequency,
            holdings_count: Math.max(10, defaultStrategy.holdings_count || current.holdings_count),
            max_weight: defaultStrategy.max_weight || current.max_weight,
          },
    )
  }, [strategies])

  const selectedStrategy = strategies?.find((item: any) => item.id === form.strategy_id)
  const isQuantV3Strategy = selectedStrategy?.kind === 'quant_v3_regression'
  const selectedSymbols = form.custom_symbols
  const manualUniverse = !isQuantV3Strategy && form.universe === 'custom'
  const { data: stockSearchData, isFetching: stocksLoading } = useQuery({
    queryKey: ['backtest-stock-search', debouncedStockSearch, stockGroup, stockExchange, directoryMode],
    queryFn: async () =>
      (
        await api.get('/stocks/search', {
          params: {
            q: debouncedStockSearch,
            group: stockGroup,
            exchange: stockExchange,
            source: directoryMode,
          },
        })
      ).data,
    enabled: manualUniverse,
    staleTime: 60 * 1000,
  })
  const stocks: StockOption[] = stockSearchData?.items || []
  const filteredStocks = stocks
  const directorySource =
    stockSearchData?.source === 'akshare'
      ? 'AKShare 股票目录（可缓存到本地）'
      : stockSearchData?.source === 'local_cache'
        ? '本地缓存目录'
        : '本地 Demo 目录；搜不到时自动回退 AKShare'
  const canRun = Boolean(form.strategy_id) && (isQuantV3Strategy || !manualUniverse || selectedSymbols.length >= 10)

  const toggleSymbol = (stock: StockOption) => {
    const symbol = stock.symbol
    setSelectedStocks(current => {
      const next = { ...current }
      if (next[symbol]) delete next[symbol]
      else next[symbol] = stock
      return next
    })
    setForm(current => ({
      ...current,
      universe: 'custom',
      custom_symbols: current.custom_symbols.includes(symbol)
        ? current.custom_symbols.filter(item => item !== symbol)
        : [...current.custom_symbols, symbol],
    }))
  }

  const removeSymbol = (symbol: string) => {
    setSelectedStocks(current => {
      const next = { ...current }
      delete next[symbol]
      return next
    })
    setForm(current => ({
      ...current,
      custom_symbols: current.custom_symbols.filter(item => item !== symbol),
    }))
  }

  const selectVisibleStocks = () => {
    const visible = filteredStocks.slice(0, Math.max(10, form.holdings_count))
    setSelectedStocks(current => ({
      ...current,
      ...Object.fromEntries(visible.map(stock => [stock.symbol, stock])),
    }))
    setForm(current => ({
      ...current,
      universe: 'custom',
      custom_symbols: Array.from(new Set([...current.custom_symbols, ...visible.map(stock => stock.symbol)])),
    }))
  }

  const clearSelectedStocks = () => {
    setSelectedStocks({})
    setForm(current => ({ ...current, custom_symbols: [] }))
  }

  const catalogSyncMutation = useMutation({
    mutationFn: async () => (await api.post('/data/catalog/sync')).data,
    onSuccess: data =>
      setSyncMessage(
        `${data.message || 'AKShare 股票目录已刷新'} · 当前目录 ${data.after_count || 0} 只，新增 ${data.added_count || 0} 只`,
      ),
    onError: () => setSyncMessage('AKShare 目录刷新失败，仍可使用本地 Demo 目录'),
  })

  const pricesSyncMutation = useMutation({
    mutationFn: async () =>
      (
        await api.post('/data/sync', {
          symbols: selectedSymbols,
          start_date: form.start_date,
          end_date: form.end_date,
        })
      ).data,
    onSuccess: data =>
      setSyncMessage(
        data.data_mode === 'real'
          ? `已从 AKShare 同步 ${data.real_rows || 0} 条行情记录`
          : 'AKShare 行情暂不可用，系统将继续使用 Demo/缓存数据运行',
      ),
    onError: () => setSyncMessage('行情同步失败，回测仍可使用 Demo/fallback 数据运行'),
  })

  const mutation = useMutation({
    mutationFn: async () => (await api.post('/backtests', form)).data,
    onSuccess: data => {
      setResult(data)
      setTradePage(1)
      setExportMessage('')
      setShowAllCharts(true)
      const firstTraded = data.selected_stocks?.find((stock: any) =>
        data.trades?.some((trade: any) => trade.symbol === stock.symbol),
      )
      setSelectedSymbol(firstTraded?.symbol || data.selected_stocks?.[0]?.symbol || '')
    },
  })

  const applyPaperMutation = useMutation({
    mutationFn: async () =>
      (await api.post(`/backtests/${result?.id}/apply-paper`, {
        execute: true,
        reset_account: false,
        enable_automation: false,
      })).data,
    onSuccess: data => {
      const orderCount = data.rebalance?.orders?.length || 0
      setPaperMessage(`已应用到模拟盘：${data.strategy?.name || '回测策略'} · 本次生成 ${orderCount} 笔订单`)
      queryClient.invalidateQueries({ queryKey: ['paper-account'] })
      queryClient.invalidateQueries({ queryKey: ['paper-equity'] })
      queryClient.invalidateQueries({ queryKey: ['paper-automation'] })
      queryClient.invalidateQueries({ queryKey: ['orders'] })
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
    },
  })

  const { data: stockChart, isFetching: stockChartLoading } = useQuery({
    queryKey: ['backtest-stock-chart', result?.id, selectedSymbol],
    queryFn: async () => (await api.get(`/backtests/${result.id}/stocks/${selectedSymbol}/chart`)).data,
    enabled: !!result?.id && !!selectedSymbol,
  })
  const { data: allStockCharts, isFetching: allStockChartsLoading } = useQuery({
    queryKey: ['backtest-stock-charts', result?.id],
    queryFn: async () => (await api.get(`/backtests/${result.id}/stocks/charts`)).data,
    enabled: !!result?.id && showAllCharts,
  })

  const trades = result?.trades || []
  const totalTradePages = Math.max(1, Math.ceil(trades.length / TRADE_PAGE_SIZE))
  const visibleTrades = trades.slice((tradePage - 1) * TRADE_PAGE_SIZE, tradePage * TRADE_PAGE_SIZE)

  const downloadTrades = async () => {
    if (!result?.id) return
    setExporting(true)
    setExportMessage('')
    try {
      const response = await api.get(`/backtests/${result.id}/trades.csv`, { responseType: 'blob' })
      const blobUrl = URL.createObjectURL(response.data)
      const anchor = document.createElement('a')
      anchor.href = blobUrl
      anchor.download = `backtest_${result.id}_trades.csv`
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

  const initialCapital = Number(result?.initial_capital ?? form.initial_capital)
  const finalAssets = Number(
    result?.final_assets ??
      result?.metrics?.final_assets ??
      result?.equity?.[result.equity.length - 1]?.total_assets ??
      result?.equity?.[result.equity.length - 1]?.equity ??
      initialCapital * (1 + Number(result?.metrics?.total_return || 0)),
  )
  const totalProfit = Number(result?.total_profit ?? result?.metrics?.total_profit ?? finalAssets - initialCapital)
  const overallReturn = Number(
    result?.overall_return ?? result?.metrics?.overall_return ?? result?.metrics?.total_return ?? finalAssets / initialCapital - 1,
  )

  return (
    <>
      <PageHeader
        eyebrow="EVALUATION / 04"
        title="回测中心"
        description="选择股票池、手动选出至少 10 只股票并框定年份，验证低频买卖策略的真实收益。"
        actions={
          <button className="primary-button" onClick={() => mutation.mutate()} disabled={mutation.isPending || !canRun}>
            <Play size={16} />
            {mutation.isPending ? '正在回测…' : '运行回测'}
          </button>
        }
      />

      <div className="backtest-layout">
        <div>
          <Card>
            <PanelHeader title="回测参数" subtitle="收盘后计算信号，下一交易日执行，避免未来数据泄漏" />
            <div className="form-grid">
              <label>
                策略
                <select
                  value={form.strategy_id}
                  onChange={event => {
                    const nextId = Number(event.target.value)
                    const next = strategies?.find((item: any) => item.id === nextId)
                    setForm(current => ({
                      ...current,
                      strategy_id: nextId,
                      // 每个策略支持的回测区间不一样(比如LightGBM策略只有
                      // 2019-2025年的滚动训练模型)，切换策略时把日期范围
                      // 同步过去，避免用着上一个策略的区间跑出422报错。
                      start_date: next?.research_start_date || current.start_date,
                      end_date: next?.research_end_date || current.end_date,
                    }))
                  }}
                >
                  {strategies?.map((strategy: any) => (
                    <option key={strategy.id} value={strategy.id}>
                      {strategy.name} · V{strategy.version}
                    </option>
                  ))}
                </select>
              </label>
              {isQuantV3Strategy && (
                <div className="strategy-note full">
                  <div>
                    <b>固定30支跨行业候选池 · 月度调仓</b>
                    <p>由策略自动管理股票池和调仓周期，无需手动选择。只支持 2019-01-01 至 2025-12-31 之间的回测区间（逐年滚动训练的模型覆盖范围）。</p>
                  </div>
                </div>
              )}
              {!isQuantV3Strategy && (
                <label>
                  股票池
                  <select value={form.universe} onChange={event => setForm({ ...form, universe: event.target.value })}>
                    {universes?.map((item: any) => (
                      <option key={item.id} value={item.id}>
                        {item.name} · {item.count}只
                      </option>
                    ))}
                    <option value="custom">手动选择股票 · {selectedSymbols.length}只</option>
                  </select>
                </label>
              )}
              {!isQuantV3Strategy && (
                <label>
                  调仓周期
                  <select
                    value={form.rebalance_frequency}
                    onChange={event => setForm({ ...form, rebalance_frequency: event.target.value })}
                  >
                    <option value="weekly">每周</option>
                    <option value="monthly">每月</option>
                    <option value="quarterly">每季度</option>
                  </select>
                </label>
              )}
              <label>
                <span>
                  <CalendarRange size={14} />开始日期
                </span>
                <input
                  type="date"
                  value={form.start_date}
                  onChange={event => setForm({ ...form, start_date: event.target.value })}
                />
              </label>
              <label>
                <span>
                  <CalendarRange size={14} />结束日期
                </span>
                <input
                  type="date"
                  value={form.end_date}
                  onChange={event => setForm({ ...form, end_date: event.target.value })}
                />
              </label>
              <label>
                初始资金
                <input
                  type="number"
                  min="1"
                  value={form.initial_capital}
                  onChange={event => setForm({ ...form, initial_capital: Number(event.target.value) })}
                />
              </label>
              {!isQuantV3Strategy && (
                <label>
                  最终选股 Top N（至少10只）
                  <input
                    type="number"
                    min="10"
                    max="100"
                    value={form.holdings_count}
                    onChange={event =>
                      setForm({ ...form, holdings_count: Math.max(10, Number(event.target.value)) })
                    }
                  />
                </label>
              )}
              {!isQuantV3Strategy && (
                <label>
                  单股最大权重
                  <div className="input-with-suffix">
                    <input
                      type="number"
                      min="1"
                      max="100"
                      value={form.max_weight * 100}
                      onChange={event => setForm({ ...form, max_weight: Number(event.target.value) / 100 })}
                    />
                    <span>%</span>
                  </div>
                </label>
              )}
              <label>
                手续费
                <div className="input-with-suffix">
                  <input
                    type="number"
                    step=".01"
                    min="0"
                    value={form.commission * 100}
                    onChange={event => setForm({ ...form, commission: Number(event.target.value) / 100 })}
                  />
                  <span>%</span>
                </div>
              </label>
              <label>
                滑点
                <div className="input-with-suffix">
                  <input
                    type="number"
                    step=".01"
                    min="0"
                    value={form.slippage * 100}
                    onChange={event => setForm({ ...form, slippage: Number(event.target.value) / 100 })}
                  />
                  <span>%</span>
                </div>
              </label>
            </div>
            <div className="date-presets">
              <button onClick={() => setForm({ ...form, start_date: '2018-01-01', end_date: '2024-12-31' })}>
                研究区间 2018-2024
              </button>
              <button onClick={() => setForm({ ...form, start_date: '2018-01-01', end_date: '2025-12-31' })}>
                研究 + 验证 2018-2025
              </button>
              <button
                onClick={() =>
                  setForm({ ...form, start_date: '2026-01-01', end_date: new Date().toISOString().slice(0, 10) })
                }
              >
                模拟盘区间
              </button>
            </div>
          </Card>

          {manualUniverse && (
            <Card className="stock-picker-card">
              <PanelHeader
                title="手动选择候选股票"
                subtitle={`已选择 ${selectedSymbols.length} 只；至少 10 只才能运行，策略会在候选池中排名并执行低频买卖。当前目录：${directorySource}`}
                action={
                  <div className="stock-picker-actions">
                    <button
                      className="secondary-button"
                      onClick={() => catalogSyncMutation.mutate()}
                      disabled={catalogSyncMutation.isPending}
                    >
                      <RefreshCw size={13} className={catalogSyncMutation.isPending ? 'spin' : ''} />
                      {catalogSyncMutation.isPending ? '刷新中…' : '刷新 AKShare 目录'}
                    </button>
                    <button
                      className="secondary-button"
                      onClick={() => pricesSyncMutation.mutate()}
                      disabled={pricesSyncMutation.isPending || selectedSymbols.length === 0}
                    >
                      <Download size={13} />
                      {pricesSyncMutation.isPending ? '同步中…' : '同步所选行情'}
                    </button>
                    <button className="secondary-button" onClick={selectVisibleStocks} disabled={stocksLoading}>
                      选当前前 {Math.max(10, form.holdings_count)} 只
                    </button>
                    <button className="secondary-button danger-button" onClick={clearSelectedStocks}>
                      清空
                    </button>
                  </div>
                }
              />
              {syncMessage && <div className="export-message">{syncMessage}</div>}
              <div className="stock-picker-toolbar">
                <div className="directory-mode-toggle" aria-label="股票目录来源">
                  <button
                    className={directoryMode === 'auto' ? 'active' : ''}
                    onClick={() => setDirectoryMode('auto')}
                  >
                    智能搜索
                  </button>
                  <button
                    className={directoryMode === 'local' ? 'active' : ''}
                    onClick={() => setDirectoryMode('local')}
                  >
                    本地 Demo
                  </button>
                  <button
                    className={directoryMode === 'akshare' ? 'active' : ''}
                    onClick={() => setDirectoryMode('akshare')}
                  >
                    AKShare
                  </button>
                </div>
                <div className="search-box">
                  <Search size={15} />
                  <input
                    value={stockSearch}
                    onChange={event => setStockSearch(event.target.value)}
                    placeholder="搜索代码、名称或行业"
                  />
                  {stockSearch && (
                    <button className="picker-clear" onClick={() => setStockSearch('')} aria-label="清除搜索">
                      <X size={13} />
                    </button>
                  )}
                </div>
                <select value={stockGroup} onChange={event => setStockGroup(event.target.value)}>
                  <option value="">全部研究组</option>
                  {Array.from(new Set(stocks.map(stock => stock.group)))
                    .sort()
                    .map(group => (
                      <option key={group} value={group}>
                        {group}
                      </option>
                    ))}
                </select>
                <select value={stockExchange} onChange={event => setStockExchange(event.target.value)}>
                  <option value="">全部市场</option>
                  {Array.from(new Set(stocks.map(stock => stock.exchange)))
                    .sort()
                    .map(exchange => (
                      <option key={exchange} value={exchange}>
                        {exchange}
                      </option>
                    ))}
                </select>
              </div>
              <div className="selected-symbols">
                {selectedSymbols.length ? (
                  selectedSymbols.map(symbol => {
                    const stock = selectedStocks[symbol]
                    return (
                      <button key={symbol} className="selected-symbol" onClick={() => removeSymbol(symbol)}>
                        {stock?.name || symbol} <span>{symbol}</span> <X size={12} />
                      </button>
                    )
                  })
                ) : (
                  <span className="selected-symbols-empty">还没有选择股票。建议至少选择 10 只流动性较好的大盘股。</span>
                )}
              </div>
              <div className="stock-picker-meta">
                <span>
                  当前筛选 {filteredStocks.length} 只 · 显示全部
                </span>
                {selectedSymbols.length < 10 && <b>还需选择 {10 - selectedSymbols.length} 只</b>}
              </div>
              {stocksLoading ? (
                <LoadingState text="正在加载股票目录…" />
              ) : (
                <div className="stock-picker-grid">
                  {filteredStocks.map(stock => {
                    const selected = selectedSymbols.includes(stock.symbol)
                    return (
                      <button
                        key={stock.symbol}
                        className={`stock-picker-item ${selected ? 'selected' : ''}`}
                        onClick={() => toggleSymbol(stock)}
                        aria-pressed={selected}
                      >
                        <span className="stock-picker-check">{selected ? <Check size={13} /> : null}</span>
                        <span className="stock-picker-name">
                          <b>{stock.name}</b>
                          <small>
                            {stock.symbol} · {stock.exchange}
                          </small>
                        </span>
                        <span className="stock-picker-tags">
                          <small>{stock.group}</small>
                          <small>{stock.industry}</small>
                          {stock.source && <small>{stock.source === 'akshare' ? 'AKShare' : 'Demo'}</small>}
                        </span>
                      </button>
                    )
                  })}
                </div>
              )}
              {!stocksLoading && !filteredStocks.length && <div className="muted-empty">没有匹配的股票。</div>}
            </Card>
          )}
        </div>

        <Card className="period-card">
          <PanelHeader title="策略执行口径" subtitle="低频、可审计、无未来数据" />
          <div className="timeline">
            <div>
              <span className="timeline-dot mint" />
              <b>{manualUniverse ? '手动框定候选股票池' : '框定候选股票池'}</b>
              <span>
                {manualUniverse
                  ? `本次已选择 ${selectedSymbols.length} 只股票`
                  : '大盘股、A股、Pink Sheets 或已保存自定义池'}
              </span>
            </div>
            <div>
              <span className="timeline-dot blue" />
              <b>综合评分选 Top N</b>
              <span>五大因子横截面排名，至少 10 只，可提高到 100 只</span>
            </div>
            <div>
              <span className="timeline-dot gold" />
              <b>低频自动买卖</b>
              <span>BUY / SELL / HOLD，下一交易日成交，费用进入净值</span>
            </div>
          </div>
          <div className="method-note">
            <RotateCcw size={16} />
            <span>所有收益、曲线和买卖点均由本次回测实际成交计算，不硬编码收益率。</span>
          </div>
          {manualUniverse && selectedSymbols.length < 10 && (
            <div className="selection-warning">请先选择至少 10 只股票，运行按钮会在满足条件后启用。</div>
          )}
        </Card>
      </div>

      {mutation.isError && <ErrorState text="回测失败，请检查日期、Top N 和股票池。" />}

      {result && (
        <>
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
            <Metric label="沪深300" value={formatPercent(result.metrics.benchmark_return)} />
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
              title="策略执行摘要与风险闸门"
              subtitle="这组规则同时用于本次回测和模拟盘，不承诺收益；实际表现以所选区间和数据模式为准。"
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
                  <button className="secondary-button" onClick={() => navigate('/trading')}>
                    查看自动交易
                  </button>
                </div>
              }
            />
            {paperMessage && <div className="export-message">{paperMessage}</div>}
            <div className="risk-summary-grid">
              <div><span>现金缓冲</span><b>{formatPercent(result.strategy_config?.cash_buffer || 0)}</b></div>
              <div><span>风险关闭暴露</span><b>{formatPercent(result.strategy_config?.risk_off_exposure || 0)}</b></div>
              <div><span>换手忽略带</span><b>{formatPercent(result.strategy_config?.turnover_band || 0)}</b></div>
              <div><span>保护性止损</span><b>{formatPercent(result.strategy_config?.stop_loss || 0)}</b></div>
              <div><span>目标年化波动率</span><b>{formatPercent(result.strategy_config?.target_volatility || 0)}</b></div>
              <div><span>最大回撤预算</span><b>{formatPercent(result.strategy_config?.max_drawdown_budget || 0)}</b></div>
              <div><span>回撤刹车暴露</span><b>{formatPercent(result.strategy_config?.drawdown_brake_exposure || 0)}</b></div>
              <div><span>趋势闸门</span><b>{result.strategy_config?.trend_filter ? '沪深300 50/200日均线' : '关闭'}</b></div>
              <div><span>风险闸门调仓</span><b>{Number(result.risk_summary?.risk_gate_rebalances || 0)} 次</b></div>
              <div><span>回撤刹车调仓</span><b>{Number(result.risk_summary?.drawdown_brake_rebalances || 0)} 次</b></div>
              <div><span>波动率缩放调仓</span><b>{Number(result.risk_summary?.volatility_scaled_rebalances || 0)} 次</b></div>
              <div><span>平均股票暴露</span><b>{formatPercent(result.risk_summary?.average_target_exposure || 0)}</b></div>
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
                    {stock.symbol} · {stock.group} · {stock.sector}/{stock.industry}
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
                    </tr>
                  </thead>
                  <tbody>
                    {visibleTrades.map((trade: any, index: number) => (
                      <tr key={`${trade.trade_date}-${trade.symbol}-${index}`}>
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
                      </tr>
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
      )}
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
