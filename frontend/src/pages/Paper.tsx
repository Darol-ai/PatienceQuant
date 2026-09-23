import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Download, Play, RefreshCcw, RotateCcw, WalletCards } from 'lucide-react'
import { Fragment, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, formatMoney, formatPercent } from '../api'
import { EquityChart } from '../components/Charts'
import { Card, ErrorState, LoadingState, PageHeader, PanelHeader, SignalBadge, Toast } from '../components/UI'
import { describeSpec, frequencyLabel, universeLabel } from '../strategy'

// 模拟盘（ADR-0051：合并原"模拟盘"和"自动交易"）：绑定的策略、净值与持仓、手动调仓、
// 自动调仓开关、交易账本。虚拟资金，本地撮合，不连接券商。
export function Paper() {
  const queryClient = useQueryClient()
  const refreshAll = () => ['paper-account', 'paper-equity', 'paper-orders', 'paper-automation'].forEach(key => queryClient.invalidateQueries({ queryKey: [key] }))
  const { data: account, isLoading, isError } = useQuery({ queryKey: ['paper-account'], queryFn: async () => (await api.get('/paper/account')).data, refetchInterval: 30_000 })
  const { data: equity } = useQuery({ queryKey: ['paper-equity'], queryFn: async () => (await api.get('/paper/equity')).data, refetchInterval: 30_000 })
  const { data: orders } = useQuery({ queryKey: ['paper-orders'], queryFn: async () => (await api.get('/paper/orders')).data })
  const { data: automation } = useQuery({ queryKey: ['paper-automation'], queryFn: async () => (await api.get('/paper/automation')).data })
  const { data: strategies } = useQuery({ queryKey: ['strategies'], queryFn: async () => (await api.get('/strategies')).data })
  const { data: options } = useQuery({ queryKey: ['pipeline-options'], queryFn: async () => (await api.get('/pipeline/options')).data })
  const { data: pools } = useQuery({ queryKey: ['pools'], queryFn: async () => (await api.get('/pools')).data })
  const [strategyId, setStrategyId] = useState(0)
  const [message, setMessage] = useState('')
  const [explanations, setExplanations] = useState<Record<number, string>>({})
  const library = (strategies || []).filter((s: any) => s.origin !== 'paper_snapshot' || s.id === account?.strategy_id)

  useEffect(() => {
    if (!strategyId && account && library.length) setStrategyId(account.strategy_id || library[0].id)
  }, [account, library, strategyId])

  const rebalance = useMutation({
    mutationFn: async () => (await api.post('/paper/rebalance', { strategy_id: strategyId }, { timeout: 600_000 })).data,
    onSuccess: data => { setMessage(`调仓完成：${data.orders?.length || 0} 笔订单`); refreshAll() },
    onError: (e: any) => setMessage(e?.response?.data?.detail || '调仓失败'),
  })
  const reset = useMutation({
    mutationFn: async () => (await api.post('/paper/reset', { initial_capital: 1_000_000 })).data,
    onSuccess: () => { setMessage('模拟账户已重置'); refreshAll() },
  })
  const bound = (strategies || []).find((s: any) => s.id === account?.strategy_id)
  const automate = useMutation({
    mutationFn: async (enabled: boolean) => (await api.post('/paper/automation', { strategy_id: account?.strategy_id, enabled, frequency: bound?.spec?.rebalance?.frequency || 'monthly' })).data,
    onSuccess: () => refreshAll(),
    onError: (e: any) => setMessage(e?.response?.data?.detail || '设置自动调仓失败'),
  })
  const explain = useMutation({
    mutationFn: async (orderId: number) => ({ orderId, data: (await api.post(`/paper/orders/${orderId}/explain`, {}, { timeout: 120_000 })).data }),
    onSuccess: ({ orderId, data }) => setExplanations(current => ({ ...current, [orderId]: data.text })),
    onError: (e: any) => setMessage(e?.response?.data?.detail || '解释失败'),
  })
  const downloadOrders = async () => {
    const response = await api.get('/paper/orders.csv', { responseType: 'blob' })
    const url = URL.createObjectURL(response.data)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'paper_orders.csv'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  if (isLoading) return <LoadingState />
  if (isError || !account) return <ErrorState />
  const switching = strategyId && strategyId !== account.strategy_id

  return <>
    <PageHeader eyebrow="PAPER" title="模拟盘" description="虚拟资金、本地撮合，不连接券商。按绑定策略的规格调仓，和回测用的是同一套执行逻辑。"
      actions={<>
        <button className="secondary-button" onClick={refreshAll}><RefreshCcw size={15}/>刷新</button>
        <button className="secondary-button danger-button" onClick={() => window.confirm('清空持仓和交易账本，资金恢复为 100 万？') && reset.mutate()}><RotateCcw size={15}/>重置账户</button>
      </>} />
    <div className="metric-grid">
      <div className="metric-card highlight"><div className="metric-top"><span>总资产</span><div className="metric-icon"><WalletCards size={18}/></div></div>
        <strong>{formatMoney(account.total_assets)}</strong>
        <div className={account.cumulative_return >= 0 ? 'metric-change positive' : 'metric-change negative'}>{formatPercent(account.cumulative_return)} 累计收益</div></div>
      <div className="metric-card"><span>今日收益</span><strong className={account.daily_return >= 0 ? 'text-mint' : 'text-rose'}>{formatPercent(account.daily_return)}</strong><span className="muted-label">相对上一次记录</span></div>
      <div className="metric-card"><span>持仓市值</span><strong>{formatMoney(account.market_value)}</strong><span className="muted-label">仓位 {formatPercent(account.market_value / Math.max(account.total_assets, 1), 0)}</span></div>
      <div className="metric-card"><span>现金</span><strong>{formatMoney(account.cash)}</strong><span className="muted-label">初始 {formatMoney(account.initial_capital)}</span></div>
    </div>

    <div className="detail-grid-2">
      <Card>
        <PanelHeader title="绑定的策略" subtitle={account.last_rebalance_at ? `最后调仓 ${account.last_rebalance_at.slice(0, 16).replace('T', ' ')}` : '还没有调仓'} />
        {bound ? <div className="strategy-note"><div>
          <b>{bound.origin === 'paper_snapshot' ? <>{bound.name}{account.source_backtest_run_id && <> · <Link to={`/backtests/${account.source_backtest_run_id}`}>来源回测</Link></>}</> : <Link to={`/library/${bound.id}`}>{bound.name}</Link>}</b>
          <p>{describeSpec(bound.spec, options)}</p>
          <p>执行股票池：{account.execution_symbols?.length ? `${account.execution_symbols.length} 只（来自回测 #${account.source_backtest_run_id ?? '—'}）` : universeLabel(account.execution_universe, pools)}</p>
        </div></div> : <p className="muted-note">还没有绑定策略。</p>}
        <div className="form-grid" style={{ marginTop: 14 }}>
          <label className="full">换一个策略，或按当前策略再调一次仓<select value={strategyId} onChange={e => setStrategyId(Number(e.target.value))}>
            {library.map((s: any) => <option key={s.id} value={s.id}>{s.name}{s.id === account.strategy_id ? '（当前）' : ''}</option>)}
          </select></label>
        </div>
        <button className="primary-button full-button" disabled={!strategyId || rebalance.isPending} onClick={() => {
          if (!switching || window.confirm('换策略会按新策略的默认股票池重新调仓，确定吗？')) rebalance.mutate()
        }}><Play size={15}/>{rebalance.isPending ? '调仓中…' : switching ? '换成这个策略并调仓' : '按当前策略调仓一次'}</button>
      </Card>
      <Card>
        <PanelHeader title="自动调仓" subtitle="后台按策略的调仓频率自动执行" />
        <div className="account-info">
          <div><span>状态</span><b className={automation?.enabled ? 'text-mint' : ''}>{automation?.enabled ? '已开启' : '未开启'}</b></div>
          <div><span>频率</span><b>{frequencyLabel[bound?.spec?.rebalance?.frequency] || '—'}（跟随策略）</b></div>
          <div><span>下次执行</span><b>{automation?.next_rebalance_date || '—'}</b></div>
          <div><span>上次自动执行</span><b>{automation?.last_auto_run_at ? automation.last_auto_run_at.slice(0, 16).replace('T', ' ') : '—'}</b></div>
        </div>
        <button className="secondary-button full-button" disabled={!account.strategy_id || automate.isPending} onClick={() => automate.mutate(!automation?.enabled)}>
          {automation?.enabled ? '关闭自动调仓' : '开启自动调仓'}
        </button>
        <p className="muted-note">模拟盘不会连接真实券商，所有订单只写入本地交易账本。</p>
      </Card>
    </div>

    <Card>
      <PanelHeader title="账户净值与买卖点" subtitle="每次调仓后记录一次资产" />
      {equity?.equity?.length ? <EquityChart data={equity.equity} trades={equity.trade_markers || []}/> : <div className="empty-chart">调仓后这里会显示净值曲线和买卖点。</div>}
    </Card>

    <Card>
      <PanelHeader title="当前持仓" subtitle={`${account.positions.length} 只 · 按最新可用价格计算浮动盈亏`} />
      <div className="table-wrap"><table>
        <thead><tr><th>股票</th><th>数量</th><th>成本价</th><th>最新价</th><th>市值</th><th>浮动盈亏</th><th>收益率</th></tr></thead>
        <tbody>{account.positions.map((p: any) => <tr key={p.symbol}>
          <td><Link className="stock-link-button" to={`/stocks/${p.symbol}`}>{p.name && p.name !== p.symbol ? `${p.name} ` : ''}{p.symbol}</Link></td>
          <td>{p.quantity}</td><td>{p.avg_cost.toFixed(2)}</td><td>{p.current_price.toFixed(2)}</td><td>{formatMoney(p.market_value)}</td>
          <td className={p.pnl >= 0 ? 'text-mint' : 'text-rose'}>{formatMoney(p.pnl)}</td>
          <td className={p.pnl_pct >= 0 ? 'text-mint' : 'text-rose'}>{formatPercent(p.pnl_pct)}</td>
        </tr>)}</tbody>
      </table></div>
    </Card>

    <Card>
      <PanelHeader title="交易账本" subtitle="最近 100 条模拟订单；点「解释」按调仓当天的打分说明为什么买卖" action={<button className="secondary-button" onClick={downloadOrders}><Download size={14}/>导出 CSV</button>} />
      <div className="table-wrap"><table>
        <thead><tr><th>时间</th><th>股票</th><th>方向</th><th>数量</th><th>价格</th><th>金额</th><th>策略</th><th>原因</th><th>状态</th><th></th></tr></thead>
        <tbody>{(orders || []).map((o: any) => <Fragment key={o.id}>
          <tr>
            <td>{o.time?.slice(0, 16).replace('T', ' ')}</td>
            <td>{o.name && o.name !== o.symbol ? `${o.name} ` : ''}{o.symbol}</td>
            <td><span className={`order-side ${o.side.toLowerCase()}`}>{o.side}</span></td>
            <td>{o.quantity}</td><td>{Number(o.price).toFixed(2)}</td><td>{formatMoney(o.amount)}</td>
            <td>{o.strategy}</td><td>{o.reason}</td><td><SignalBadge signal={o.status.toUpperCase()}/></td>
            <td><button className="stock-link-button" disabled={explain.isPending} onClick={() => explain.mutate(o.id)}>解释</button></td>
          </tr>
          {explanations[o.id] && <tr className="explain-row"><td colSpan={10}>{explanations[o.id]}</td></tr>}
        </Fragment>)}</tbody>
      </table></div>
    </Card>
    {message && <Toast message={message} type={rebalance.isError || automate.isError ? 'error' : 'success'} />}
  </>
}
