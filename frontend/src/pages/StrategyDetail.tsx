import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, CopyPlus, Play, RefreshCw, WalletCards } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, formatMoney, formatPercent } from '../api'
import { BarChart } from '../components/Charts'
import { frameworkText, isTraining, meanOf, ModelStatus, ModelYears } from '../components/ModelBits'
import { CardMetrics } from '../components/ScorecardBits'
import { DataNote } from '../components/DataNote'
import { Card, ErrorState, LoadingState, PageHeader, PanelHeader, Toast } from '../components/UI'
import { frequencyLabel, originLabel, scorerText, selectionText, timingText, universeLabel, weightingText } from '../strategy'

// 策略详情（ADR-0051）：规格、成绩卡、逐年收益、全部回测历史；回测它 / 放进模拟盘 / 基于它新建。
export function StrategyDetail() {
  const { id } = useParams()
  const strategyId = Number(id)
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [message, setMessage] = useState('')
  const { data: strategies, isLoading } = useQuery({ queryKey: ['strategies'], queryFn: async () => (await api.get('/strategies')).data })
  const { data: options } = useQuery({ queryKey: ['pipeline-options'], queryFn: async () => (await api.get('/pipeline/options')).data })
  const { data: pools } = useQuery({ queryKey: ['pools'], queryFn: async () => (await api.get('/pools')).data })
  const { data: cards } = useQuery({
    queryKey: ['scorecards'],
    queryFn: async () => (await api.get('/scorecards')).data,
    refetchInterval: query => (query.state.data?.refresh?.running ? 4000 : false),
  })
  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: async () => (await api.get('/models')).data,
    refetchInterval: query => (isTraining(query.state.data) ? 3000 : false),
  })
  const { data: history } = useQuery({
    queryKey: ['backtest-history', strategyId],
    queryFn: async () => (await api.get('/backtests', { params: { strategy_id: strategyId } })).data,
  })
  const refresh = useMutation({
    mutationFn: async () => (await api.post('/scorecards/refresh', { strategy_ids: [strategyId] })).data,
    onSuccess: data => {
      setMessage(data.started ? '已开始计算成绩卡，后台进行' : '已有成绩卡正在计算，稍后再试')
      queryClient.invalidateQueries({ queryKey: ['scorecards'] })
    },
  })
  const toPaper = useMutation({
    mutationFn: async () => (await api.post('/paper/rebalance', { strategy_id: strategyId }, { timeout: 600_000 })).data,
    onSuccess: data => {
      // 调仓接口返回了新的账户快照：直接放进缓存，模拟盘页不会先显示调仓前的旧数据
      if (data.snapshot) queryClient.setQueryData(['paper-account'], data.snapshot)
      ;['paper-account', 'paper-equity', 'paper-orders', 'paper-automation'].forEach(key => queryClient.invalidateQueries({ queryKey: [key] }))
      setMessage(`已绑定到模拟盘并调仓：${data.orders?.length || 0} 笔订单`)
      navigate('/paper')
    },
    onError: (error: any) => setMessage(error?.response?.data?.detail || '放进模拟盘失败'),
  })

  if (isLoading) return <LoadingState />
  const strategy = (strategies || []).find((s: any) => s.id === strategyId)
  if (!strategy) return <ErrorState text="策略不存在" />
  const card = (cards?.cards || []).find((c: any) => c.strategy_id === strategyId)
  const spec = strategy.spec
  const running = cards?.refresh?.running
  const factorLabel: Record<string, string> = Object.fromEntries((options?.training_factors || []).map((f: any) => [f.key, f.label]))
  const usedModels = spec.scorer.type === 'model' ? spec.scorer.models.map((mid: string) => (models || []).find((m: any) => m.id === mid) || { id: mid, name: mid, status: 'failed', error: '模型库里找不到这个模型', factors: [], years: [] }) : []
  // 因子权重策略用到的因子：每个因子组按等权合成组内因子
  const groupFactors = spec.scorer.type === 'factor_weights'
    ? Object.entries(spec.scorer.weights).filter(([, w]) => Number(w) > 0).map(([g, w]) => ({ group: g, weight: Number(w), keys: options?.factor_groups?.[g] || [g] }))
    : []

  return <>
    <PageHeader eyebrow={`策略库 · ${originLabel[strategy.origin] || strategy.origin}`} title={<span className="title-with-back"><Link to="/library" className="icon-button"><ArrowLeft size={16}/></Link>{strategy.name}{strategy.version > 1 ? ` · V${strategy.version}` : ''}</span>}
      description={strategy.description || describeFallback(strategy)}
      actions={<>
        <Link className="secondary-button" to={`/strategy/new?from=${strategyId}`}><CopyPlus size={15}/>基于它新建</Link>
        <button className="secondary-button" disabled={toPaper.isPending} onClick={() => {
          if (window.confirm(`把模拟盘绑定到「${strategy.name}」并立即按它调仓？（虚拟资金，不连接券商）`)) toPaper.mutate()
        }}><WalletCards size={15}/>{toPaper.isPending ? '调仓中…' : '放进模拟盘'}</button>
        <Link className="primary-button" to={`/backtest?strategy_id=${strategyId}`}><Play size={15}/>回测它</Link>
      </>} />

    <DataNote strategyId={strategyId} start={cards?.standard?.start_date} end={cards?.standard?.end_date}
      totalReturn={card?.status === 'ready' ? card.metrics.total_return : undefined}
      benchmarkReturn={card?.status === 'ready' ? card.metrics.benchmark_return : undefined}/>

    <div className="detail-grid-2">
      <Card>
        <PanelHeader title="策略规格" subtitle="保存后不再修改；要改请「基于它新建」" />
        <div className="spec-list">
          <div><span>① 打分</span><b>{scorerText(spec, models ? { ...options, models } : options)}</b></div>
          <div><span>② 选股规则</span><b>{selectionText(spec)}</b></div>
          <div><span>③ 权重方案</span><b>{weightingText(spec)}</b></div>
          <div><span>④ 择时信号</span><b>{timingText(spec, options)}</b></div>
          <div><span>⑤ 调仓</span><b>{frequencyLabel[spec.rebalance.frequency]}，换手阈值 {formatPercent(spec.rebalance.turnover_band, 0)}</b></div>
          <div><span>默认股票池</span><b>{universeLabel(strategy.default_universe, pools)}</b></div>
        </div>
      </Card>
      <Card>
        <PanelHeader title="成绩卡" subtitle={cards ? `${cards.standard.start_date} ~ ${cards.standard.end_date} · 100 万 · 默认股票池` : ''}
          action={card?.status !== 'ready' && !running ? <button className="secondary-button" onClick={() => refresh.mutate()}><RefreshCw size={13}/>计算成绩卡</button> : undefined} />
        {card?.status === 'ready' ? <>
          <CardMetrics card={card}/>
          <div className="spec-list compact">
            <div><span>累计收益</span><b>{formatPercent(card.metrics.total_return)}</b></div>
            <div><span>同期沪深300</span><b>{formatPercent(card.metrics.benchmark_return)}</b></div>
            <div><span>年化波动</span><b>{formatPercent(card.metrics.volatility)}</b></div>
            <div><span>最差年份</span><b>{card.worst_year?.year} 年 {formatPercent(card.worst_year?.strategy ?? 0)}</b></div>
          </div>
          <Link className="text-link" to={`/backtests/${card.run_id}`}>查看成绩卡这次回测的完整报告 →</Link>
        </> : <p className="muted-note">{running ? `正在计算成绩卡：${cards.refresh.current || '…'}` : '这个策略还没有成绩卡。'}</p>}
      </Card>
    </div>

    {spec.scorer.type === 'model' && usedModels.map((m: any) => <Card key={m.id}>
      <PanelHeader title={`打分模型：${m.name}`} subtitle={m.origin === 'legacy' ? '旧模型，离线训练后原样导入' : `${frameworkText[m.framework] || m.framework} · ${m.trained_on || ''} · 预测未来 ${m.horizon_days} 个交易日收益`} action={<ModelStatus model={m}/>} />
      {m.status === 'failed' && m.error && <p className="muted-note warn">{m.error}</p>}
      {(m.status === 'queued' || m.status === 'training') && <p className="muted-note">模型在后台训练，训练好后会自动计算这个策略的成绩卡；进度也可以在「数据与模型」里看。</p>}
      <div className="spec-list compact">
        <div><span>用到的因子（{m.factors.length} 个）</span><b>{m.factors.map((k: string) => factorLabel[k] || k).join('、') || '—'}</b></div>
        <div><span>可打分年份</span><b>{m.years.length ? `${m.years[0]}–${m.years[m.years.length - 1]}` : '—'}</b></div>
        {meanOf(m, 'ic') != null && <div><span>平均样本外 IC</span><b>{meanOf(m, 'ic')!.toFixed(4)}</b></div>}
        {m.config && <div><span>训练设置</span><b>{m.framework === 'lstm' ? `过去 20 个交易日的因子序列（每天换成百分位）· 隐藏层 32 · 最多 15 轮早停 · ${m.config.seeds} 个种子`
          : `${m.config.n_estimators} 棵树 · 学习率 ${m.config.learning_rate} · ${m.framework === 'lightgbm' ? `叶子数 ${m.config.num_leaves}` : `树深度 ${m.config.max_depth}`} · ${m.config.seeds} 个种子${m.config.cs_rank ? ' · 因子取池内百分位' : ''}`}{m.config.membership === 'latest' ? ' · 按今天的成分股名单' : ''}</b></div>}
      </div>
      {m.status === 'ready' && <ModelYears model={m}/>}
    </Card>)}

    {groupFactors.length > 0 && <Card>
      <PanelHeader title="用到的因子" subtitle="每个因子先换成当天池内百分位；因子组取组内百分位的平均；再按权重加总" />
      <div className="spec-list compact">{groupFactors.map(g => {
        const option = (options?.factors || []).find((f: any) => f.key === g.group)
        return <div key={g.group}><span>{option?.label || g.group} · {formatPercent(g.weight, 0)}</span>
          <b>{g.keys.map((k: string) => factorLabel[k] || k).join('、') || '—'}{option && !option.available ? `（${option.unavailable_reason}）` : ''}</b></div>
      })}</div>
    </Card>}

    {card?.status === 'ready' && <Card>
      <PanelHeader title="逐年收益" subtitle="策略 vs 沪深300" />
      <BarChart data={card.yearly} compare={{ key: 'benchmark', label: '沪深300' }} />
    </Card>}

    <Card>
      <PanelHeader title="回测历史" subtitle={`这个策略保存下来的全部回测 · ${history?.length ?? 0} 次`} />
      {history?.length ? <div className="table-wrap"><table className="clickable-rows">
        <thead><tr><th>编号</th><th>区间</th><th>股票池</th><th>初始资金</th><th>年化</th><th>最大回撤</th><th>夏普</th><th>备注</th></tr></thead>
        <tbody>{history.map((run: any) => <tr key={run.id} onClick={() => run.status === 'completed' && navigate(`/backtests/${run.id}`)}>
          <td>#{run.id}</td>
          <td>{run.start_date} ~ {run.end_date}</td>
          <td>{universeLabel(run.universe, pools)}</td>
          <td>{formatMoney(run.initial_capital)}</td>
          {run.status === 'completed' ? <>
            <td>{formatPercent(run.metrics?.annual_return ?? 0)}</td>
            <td className="text-rose">{formatPercent(run.metrics?.max_drawdown ?? 0)}</td>
            <td>{Number(run.metrics?.sharpe ?? 0).toFixed(2)}</td>
          </> : <td colSpan={3} className="muted-label">{run.status === 'failed' ? `失败：${run.error_message || ''}` : run.status}</td>}
          <td>{card?.run_id === run.id ? '成绩卡' : run.seeded ? '系统预置' : ''}</td>
        </tr>)}</tbody>
      </table></div> : <p className="muted-note">还没有回测过。</p>}
    </Card>
    {message && <Toast message={message} type={toPaper.isError ? 'error' : 'success'} />}
  </>
}

function describeFallback(strategy: any): string {
  return `创建于 ${String(strategy.created_at || '').slice(0, 10)}`
}
