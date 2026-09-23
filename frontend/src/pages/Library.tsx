import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Plus, RefreshCw } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, formatPercent } from '../api'
import { Card, ErrorState, LoadingState, PageHeader } from '../components/UI'
import { frequencyLabel, originLabel, timingLabel, universeLabel } from '../strategy'

// 策略库（ADR-0051）：所有策略的成绩卡对比，是"策略"的唯一入口。
type SortKey = 'annual_return' | 'max_drawdown' | 'sharpe'

export function Library() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [origin, setOrigin] = useState('all')
  const [scorer, setScorer] = useState('all')
  const [timing, setTiming] = useState('all')
  const [sort, setSort] = useState<SortKey>('sharpe')
  const { data, isLoading, isError } = useQuery({
    queryKey: ['scorecards'],
    queryFn: async () => (await api.get('/scorecards')).data,
    refetchInterval: query => (query.state.data?.refresh?.running ? 4000 : false),
  })
  const refresh = useMutation({
    mutationFn: async () => (await api.post('/scorecards/refresh', {})).data,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['scorecards'] }),
  })
  const rows = useMemo(() => {
    const list = (data?.cards || []).filter((c: any) =>
      (origin === 'all' || c.origin === origin) &&
      (scorer === 'all' || c.scorer === scorer) &&
      (timing === 'all' || (timing === 'none' ? c.timing === 'none' : c.timing !== 'none')))
    const value = (c: any) => (c.status === 'ready' ? c.metrics[sort] : -Infinity)
    return [...list].sort((a, b) => value(b) - value(a))
  }, [data, origin, scorer, timing, sort])

  if (isLoading) return <LoadingState />
  if (isError) return <ErrorState text="策略库加载失败" />
  const missing = (data.cards || []).filter((c: any) => c.status !== 'ready').length
  const running = data.refresh?.running

  return <>
    <PageHeader eyebrow="LIBRARY" title="策略库" description={`每个策略在同一套标准条件下的成绩卡（${data.standard.start_date} ~ ${data.standard.end_date}，初始 100 万，各自的默认股票池，含手续费和滑点），可以直接比较。`}
      actions={<>
        {running ? <span className="strategy-version">正在计算成绩卡 {data.refresh.done}/{data.refresh.total}：{data.refresh.current || '…'}</span>
          : missing > 0 && <button className="secondary-button" onClick={() => refresh.mutate()} disabled={refresh.isPending}><RefreshCw size={14}/>计算缺少的 {missing} 张成绩卡</button>}
        <Link className="primary-button" to="/strategy/new"><Plus size={15}/>新建策略</Link>
      </>} />
    {data.refresh?.errors?.length > 0 && <p className="muted-note warn">有 {data.refresh.errors.length} 个策略没算出来：{data.refresh.errors.map((e: any) => `${e.name}（${e.error}）`).join('；')}</p>}
    <Card>
      <div className="filter-bar">
        <select value={origin} onChange={e => setOrigin(e.target.value)}><option value="all">全部来源</option><option value="builtin">内置</option><option value="user">我的</option></select>
        <select value={scorer} onChange={e => setScorer(e.target.value)}><option value="all">全部打分方式</option><option value="factor_weights">因子权重</option><option value="model">模型</option></select>
        <select value={timing} onChange={e => setTiming(e.target.value)}><option value="all">择时不限</option><option value="none">不择时</option><option value="with">有择时</option></select>
        <select value={sort} onChange={e => setSort(e.target.value as SortKey)}><option value="sharpe">按夏普排序</option><option value="annual_return">按年化排序</option><option value="max_drawdown">按回撤排序（小的在前）</option></select>
      </div>
      <div className="table-wrap"><table className="clickable-rows">
        <thead><tr><th>策略</th><th>来源</th><th>打分</th><th>择时</th><th>调仓</th><th>默认股票池</th><th>年化</th><th>最大回撤</th><th>夏普</th><th>跑赢沪深300</th><th>平均仓位</th></tr></thead>
        <tbody>{rows.map((c: any) => <tr key={c.strategy_id} onClick={() => navigate(`/library/${c.strategy_id}`)}>
          <td><b>{c.name}</b>{c.version > 1 ? ` · V${c.version}` : ''}</td>
          <td><span className="group-tag">{originLabel[c.origin] || c.origin}</span></td>
          <td>{c.scorer === 'model' ? '模型' : '因子'}</td>
          <td>{timingLabel[c.timing] || c.timing}</td>
          <td>{frequencyLabel[c.rebalance_frequency] || c.rebalance_frequency}</td>
          <td>{universeLabel(c.universe)}</td>
          {c.status === 'ready' ? <>
            <td>{formatPercent(c.metrics.annual_return)}</td>
            <td className="text-rose">{formatPercent(c.metrics.max_drawdown)}</td>
            <td>{Number(c.metrics.sharpe).toFixed(2)}</td>
            <td>{c.years_beating_benchmark}/{c.years} 年</td>
            <td>{formatPercent(c.average_exposure ?? 1, 0)}</td>
          </> : <td colSpan={5} className="muted-label">还没有成绩卡</td>}
        </tr>)}</tbody>
      </table></div>
    </Card>
  </>
}
