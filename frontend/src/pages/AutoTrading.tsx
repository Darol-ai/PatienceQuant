import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Bot, Download, Play, ShieldCheck, Zap } from 'lucide-react'
import { ReactNode } from 'react'
import { useEffect, useState } from 'react'
import { api, formatMoney, formatPercent } from '../api'
import { EquityChart } from '../components/Charts'
import { Card, ErrorState, LoadingState, PageHeader, PanelHeader, SignalBadge, Toast } from '../components/UI'

export function AutoTrading() {
  const queryClient = useQueryClient()
  const [result, setResult] = useState<any>(null)
  const { data: allStrategies } = useQuery({ queryKey: ['strategies'], queryFn: async () => (await api.get('/strategies')).data })
  // 和回测中心一样，列出策略库里的策略（不含模拟盘冻结副本）——用?.保留
  // "还没加载完"时的undefined，不能默认成[]，否则下面`if (!strategies)
  // return <LoadingState/>`这个判断会被空数组绕过，页面在数据还没到
  // 的时候就先渲染出一个空列表。
  const strategies = allStrategies?.filter((item: any) => item.origin !== 'paper_snapshot')
  const { data: paperAccount } = useQuery({
    queryKey: ['paper-account'],
    queryFn: async () => (await api.get('/paper/account')).data,
    refetchInterval: 20_000,
  })
  const { data: orders } = useQuery({ queryKey: ['orders'], queryFn: async () => (await api.get('/paper/orders')).data })
  const { data: paperEquity } = useQuery({ queryKey: ['paper-equity'], queryFn: async () => (await api.get('/paper/equity')).data, refetchInterval: 20_000 })
  const { data: automation } = useQuery({ queryKey: ['paper-automation'], queryFn: async () => (await api.get('/paper/automation')).data, refetchInterval: 20_000 })
  const [strategyId, setStrategyId] = useState(0)
  const [exporting, setExporting] = useState(false)
  useEffect(() => {
    if (!strategies || paperAccount === undefined || strategyId !== 0) return
    const defaultStrategy = strategies.find((strategy: any) => strategy.is_default) || strategies[0]
    const preferredStrategy = paperAccount?.strategy_id
      ? strategies.find((strategy: any) => strategy.id === paperAccount.strategy_id)
      : null
    if (preferredStrategy || defaultStrategy) setStrategyId((preferredStrategy || defaultStrategy).id)
  }, [strategies, paperAccount, strategyId])
  const [rebalanceError, setRebalanceError] = useState('')
  const mutation = useMutation({
    // 沪深300策略冷启动要读70个模型文件+建两份90万行索引，服务启动时
    // 已经在后台预热过，但保底还是把超时放宽，避免万一没预热完撞上
    // 默认的120秒超时又是一次"悄悄失败"。
    mutationFn: async () => (await api.post('/paper/rebalance', { strategy_id: strategyId || undefined }, { timeout: 600_000 })).data,
    onMutate: () => setRebalanceError(''),
    onSuccess: data => { setResult(data); queryClient.invalidateQueries({ queryKey: ['orders'] }); queryClient.invalidateQueries({ queryKey: ['paper-account'] }); queryClient.invalidateQueries({ queryKey: ['paper-equity'] }); queryClient.invalidateQueries({ queryKey: ['dashboard'] }) },
    onError: (error: any) => {
      setRebalanceError(
        error?.code === 'ECONNABORTED'
          ? '调仓运行超时（超过10分钟），请稍后重试。'
          : error?.response?.data?.detail || '调仓执行失败，请检查数据和账户状态。',
      )
    },
  })
  const automationMutation = useMutation({ mutationFn: async (enabled: boolean) => (await api.post('/paper/automation', { strategy_id: strategyId, enabled, frequency: 'monthly' })).data, onSuccess: () => queryClient.invalidateQueries({ queryKey: ['paper-automation'] }) })
  const downloadOrders = async () => {
    setExporting(true)
    try {
      const response = await api.get('/paper/orders.csv', { responseType: 'blob' })
      const url = URL.createObjectURL(response.data)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = 'paper_orders.csv'
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(url)
    } finally {
      setExporting(false)
    }
  }
  if (!strategies) return <LoadingState />
  const executionSymbols = Array.isArray(paperAccount?.execution_symbols) ? paperAccount.execution_symbols : []
  const executionScope = executionSymbols.length
    ? `当前沿用 ${executionSymbols.length} 只手选股票 · ${executionSymbols.slice(0, 5).join('、')}${executionSymbols.length > 5 ? '…' : ''}`
    : `当前股票池：${paperAccount?.execution_universe || 'large_cap'}`
  return <><PageHeader eyebrow="EXECUTION / 06" title="自动交易" description="策略评分、组合构建、风险控制和模拟执行，在一次调仓里完成。" actions={<div className="page-actions"><button className="secondary-button" onClick={() => automationMutation.mutate(!automation?.enabled)} disabled={automationMutation.isPending || !strategyId || paperAccount === undefined}>{automation?.enabled ? '关闭自动调仓' : '开启自动调仓'}</button><button className="primary-button" onClick={() => mutation.mutate()} disabled={mutation.isPending || !strategyId || paperAccount === undefined}><Play size={16}/>{mutation.isPending ? '执行中…' : '执行一次调仓'}</button></div>} /><div className="execution-flow"><Flow icon={<Bot/>} title="打分" text="因子权重 / 模型预测"/><span>→</span><Flow icon={<Zap/>} title="选股与权重" text="前 N 名 / 前 x% + 单股上限"/><span>→</span><Flow icon={<ShieldCheck/>} title="择时仓位" text="指数趋势 0～100%"/><span>→</span><Flow icon={<Play/>} title="模拟执行" text="订单 + 成交账本"/></div><div className="dashboard-grid"><Card className="span-4"><PanelHeader title="执行配置" subtitle={executionScope}/><label className="stack-label">执行策略<select value={strategyId} onChange={e => setStrategyId(Number(e.target.value))}>{strategies.map((strategy: any) => <option key={strategy.id} value={strategy.id}>{strategy.name} · V{strategy.version}</option>)}</select></label><div className="execution-checks"><div><ShieldCheck size={15}/><span>单股上限</span><b>{paperAccount?.strategy_spec ? formatPercent(paperAccount.strategy_spec.weighting.max_weight) : '—'}</b></div><div><ShieldCheck size={15}/><span>择时信号</span><b>{paperAccount?.strategy_spec ? (paperAccount.strategy_spec.timing.type === 'index_trend' ? '沪深300趋势' : '不择时') : '—'}</b></div><div><ShieldCheck size={15}/><span>换手阈值</span><b>{paperAccount?.strategy_spec ? formatPercent(paperAccount.strategy_spec.rebalance.turnover_band) : '—'}</b></div><div><ShieldCheck size={15}/><span>模拟模式，不连接券商</span><b>PASS</b></div><div><ShieldCheck size={15}/><span>自动调仓</span><b className={automation?.enabled ? 'text-mint' : 'text-rose'}>{automation?.enabled ? `ON · 下次 ${automation.next_rebalance_date || '待定'}` : 'OFF'}</b></div></div></Card><Card className="span-8"><PanelHeader title="本次调仓结果" subtitle={result ? `完成 ${result.orders.length} 笔订单` : '点击执行后查看交易信号'} />{result ? <div className="table-wrap"><table><thead><tr><th>股票</th><th>方向</th><th>数量</th><th>价格</th><th>金额</th><th>触发原因</th></tr></thead><tbody>{result.orders.map((order: any, i: number) => <tr key={i}><td><b>{order.symbol}</b></td><td><span className={`order-side ${order.side.toLowerCase()}`}>{order.side}</span></td><td>{order.quantity}</td><td>{order.price.toFixed(2)}</td><td>{formatMoney(order.amount)}</td><td>{order.reason}</td></tr>)}</tbody></table></div> : <div className="execution-empty"><Bot size={28}/><b>等待 Quant Engine 信号</b><span>当前策略会基于最新数据生成 BUY / SELL / HOLD。</span></div>}</Card></div><Card><PanelHeader title="策略收益与自动交易买卖点" subtitle="曲线点位与下方交易账本共享同一 Paper Broker 数据"/>{paperEquity?.equity?.length ? <EquityChart data={paperEquity.equity} trades={paperEquity.trade_markers || []}/> : <div className="empty-chart">执行一次调仓后，模拟盘净值曲线和 BUY/SELL 点会显示在这里。</div>}</Card><Card><PanelHeader title="交易账本" subtitle="最近 100 条模拟订单" action={<button className="secondary-button" onClick={downloadOrders} disabled={exporting}><Download size={14}/>{exporting ? '导出中…' : '导出 CSV'}</button>}/><div className="table-wrap"><table><thead><tr><th>时间</th><th>股票</th><th>方向</th><th>数量</th><th>金额</th><th>策略</th><th>状态</th></tr></thead><tbody>{orders?.map((order: any) => <tr key={order.id}><td>{order.time?.slice(0, 16).replace('T', ' ')}</td><td><b>{order.symbol}</b></td><td><span className={`order-side ${order.side.toLowerCase()}`}>{order.side}</span></td><td>{order.quantity}</td><td>{formatMoney(order.amount)}</td><td>{order.strategy}</td><td><SignalBadge signal={order.status.toUpperCase()}/></td></tr>)}</tbody></table></div></Card>{rebalanceError && <Toast message={rebalanceError} type="error"/>}</>
}

function Flow({ icon, title, text }: { icon: ReactNode; title: string; text: string }) { return <div className="flow-node"><div>{icon}</div><b>{title}</b><span>{text}</span></div> }
