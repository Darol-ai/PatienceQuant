import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { CalendarRange, Play } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import { Card, LoadingState, PageHeader, PanelHeader } from '../components/UI'
import { describeSpec, universeLabel } from '../strategy'

// 策略实践（ADR-0047 第 3 条、ADR-0051）：只填这次回测的条件——区间、资金、股票池、费用；
// 策略本身的参数都来自策略规格。跑完跳到回测报告页。
export function Practice() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [params] = useSearchParams()
  const { data: strategies } = useQuery({ queryKey: ['strategies'], queryFn: async () => (await api.get('/strategies')).data })
  const { data: pools } = useQuery({ queryKey: ['pools'], queryFn: async () => (await api.get('/pools')).data })
  const { data: options } = useQuery({ queryKey: ['pipeline-options'], queryFn: async () => (await api.get('/pipeline/options')).data })
  const library = (strategies || []).filter((s: any) => s.origin !== 'paper_snapshot')
  const [form, setForm] = useState({ strategy_id: 0, universe: '', start_date: '2019-01-01', end_date: '2025-12-31',
                                     initial_capital: 1_000_000, commission: 0.001, slippage: 0.0005 })
  const [error, setError] = useState('')
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    if (form.strategy_id || !library.length) return
    const wanted = Number(params.get('strategy_id'))
    const initial = library.find((s: any) => s.id === wanted) || library.find((s: any) => s.is_default) || library[0]
    setForm(f => ({ ...f, strategy_id: initial.id, universe: initial.default_universe }))
  }, [library, params, form.strategy_id])

  const run = useMutation({
    mutationFn: async () => (await api.post('/backtests', form, { timeout: 600_000 })).data,
    onMutate: () => setError(''),
    onSuccess: data => {
      queryClient.invalidateQueries({ queryKey: ['backtest-history', form.strategy_id] })
      navigate(`/backtests/${data.id}`)
    },
    onError: (e: any) => setError(e?.code === 'ECONNABORTED' ? '回测超过 10 分钟没有返回，请稍后在策略详情的回测历史里查看。' : e?.response?.data?.detail || '回测失败，请检查参数'),
  })
  useEffect(() => {
    if (!run.isPending) return setElapsed(0)
    const started = Date.now()
    const timer = window.setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 1000)
    return () => window.clearInterval(timer)
  }, [run.isPending])

  if (!strategies || !pools) return <LoadingState />
  const strategy = library.find((s: any) => s.id === form.strategy_id)
  const models = strategy?.spec?.scorer?.type === 'model'
    ? (options?.models || []).filter((m: any) => strategy.spec.scorer.models.includes(m.id)) : []
  const years = models.flatMap((m: any) => m.years)

  return <>
    <PageHeader eyebrow="PRACTICE" title="策略实践" description="选一个策略，定好这次回测的区间、资金、股票池和费用；策略本身的参数不在这里改。跑完会打开回测报告，并存进这个策略的回测历史。"
      actions={<button className="primary-button" onClick={() => run.mutate()} disabled={!form.strategy_id || run.isPending}>
        <Play size={16}/>{run.isPending ? `正在回测…（${elapsed}s）` : '运行回测'}
      </button>} />
    {error && <div className="form-error" style={{ marginBottom: 16 }}>{error}</div>}
    <div className="strategy-layout">
      <Card>
        <PanelHeader title="回测条件" subtitle="收盘后出信号，下一交易日成交" />
        <div className="form-grid">
          <label className="full">策略<select value={form.strategy_id} onChange={e => {
            const next = library.find((s: any) => s.id === Number(e.target.value))
            setForm({ ...form, strategy_id: next.id, universe: next.default_universe })
          }}>{library.map((s: any) => <option key={s.id} value={s.id}>{s.name}{s.version > 1 ? ` · V${s.version}` : ''}</option>)}</select></label>
          <label className="full">股票池<select value={form.universe} onChange={e => setForm({ ...form, universe: e.target.value })}>
            {pools.map((p: any) => <option key={p.id} value={p.id}>{p.name} · {p.count} 只{p.id === strategy?.default_universe ? '（策略默认）' : ''}</option>)}
          </select></label>
          <label><span><CalendarRange size={14}/>开始日期</span><input type="date" value={form.start_date} onChange={e => setForm({ ...form, start_date: e.target.value })}/></label>
          <label><span><CalendarRange size={14}/>结束日期</span><input type="date" value={form.end_date} onChange={e => setForm({ ...form, end_date: e.target.value })}/></label>
          <label>初始资金（元）<input type="number" min={10000} value={form.initial_capital} onChange={e => setForm({ ...form, initial_capital: Number(e.target.value) })}/></label>
          <label>手续费（%）<input type="number" step="0.01" min={0} value={form.commission * 100} onChange={e => setForm({ ...form, commission: Number(e.target.value) / 100 })}/></label>
          <label>滑点（%）<input type="number" step="0.01" min={0} value={form.slippage * 100} onChange={e => setForm({ ...form, slippage: Number(e.target.value) / 100 })}/></label>
        </div>
        <div className="date-presets">
          <button onClick={() => setForm({ ...form, start_date: '2019-01-01', end_date: '2025-12-31' })}>成绩卡区间 2019–2025</button>
          <button onClick={() => setForm({ ...form, start_date: '2022-01-01', end_date: '2022-12-31' })}>熊市 2022</button>
          <button onClick={() => setForm({ ...form, start_date: '2024-01-01', end_date: '2025-12-31' })}>最近两年</button>
          <button onClick={() => setForm({ ...form, start_date: '2026-01-01', end_date: new Date().toISOString().slice(0, 10) })}>今年以来</button>
        </div>
        <p className="muted-note">自定义股票池在「股票池」页里管理。</p>
      </Card>
      <Card>
        <PanelHeader title="这个策略会怎么做" subtitle="规格在策略制定时定好" />
        {strategy && <div className="strategy-note"><div><b>{strategy.name}</b><p>{describeSpec(strategy.spec, options)}</p>
          <p>默认股票池：{universeLabel(strategy.default_universe, pools)}</p>
          {years.length > 0 && <p>用到的模型只能给 {Math.min(...years)}–{Math.max(...years)} 年打分，回测区间要落在这些年份里。</p>}
        </div></div>}
        {strategy && <Link className="text-link" to={`/library/${strategy.id}`}>查看这个策略的成绩卡和回测历史 →</Link>}
      </Card>
    </div>
  </>
}
