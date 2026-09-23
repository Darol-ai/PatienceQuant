import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowRight, Layers, Save, SlidersHorizontal, Sparkles } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, formatPercent } from '../api'
import { Card, ErrorState, LoadingState, PageHeader, PanelHeader, Toast } from '../components/UI'

// 策略制定（ADR-0047 第 3 条）：策略 = 打分 → ① 选股 → ② 权重 → ③ × 择时 → ④ 调仓。
// 策略保存后不再修改，改任何一项都另存为新策略；回测放在"策略实践"页。

type Spec = {
  market: string
  scorer: { type: 'factor_weights'; weights: Record<string, number> } | { type: 'model'; models: string[] }
  timing: { type: string; [key: string]: any }
  selection: { type: 'top_n' | 'top_pct'; n?: number | null; pct?: number | null }
  weighting: { type: 'equal' | 'score'; max_weight: number }
  rebalance: { frequency: string; turnover_band: number }
}

type Draft = { name: string; description: string; default_universe: string; spec: Spec }

const originLabel: Record<string, string> = { builtin: '内置', user: '我的' }
const frequencyLabel: Record<string, string> = { weekly: '每周', monthly: '每月', quarterly: '每季度' }

function blankDraft(): Draft {
  return {
    name: '我的新策略',
    description: '',
    default_universe: 'csi300',
    spec: {
      market: 'a_share',
      scorer: { type: 'factor_weights', weights: { momentum: 0.6, risk: 0.4 } },
      timing: { type: 'none' },
      selection: { type: 'top_n', n: 30, pct: null },
      weighting: { type: 'equal', max_weight: 0.1 },
      rebalance: { frequency: 'monthly', turnover_band: 0.02 },
    },
  }
}

export function describeSpec(spec: Spec | undefined, options?: any): string {
  if (!spec) return ''
  const modelName = (id: string) => options?.models?.find((m: any) => m.id === id)?.name || id
  const factorName = (key: string) => options?.factors?.find((f: any) => f.key === key)?.label || key
  const scorer = spec.scorer.type === 'model'
    ? `模型打分（${spec.scorer.models.map(modelName).join(' + ')}）`
    : `因子打分（${Object.entries(spec.scorer.weights).filter(([, w]) => Number(w) > 0).map(([k, w]) => `${factorName(k)} ${Math.round(Number(w) * 100)}%`).join('、')}）`
  const selection = spec.selection.type === 'top_n' ? `前 ${spec.selection.n} 名` : `前 ${formatPercent(spec.selection.pct || 0)}`
  const weighting = `${spec.weighting.type === 'equal' ? '等权' : '按分数加权'}，单股 ≤ ${formatPercent(spec.weighting.max_weight)}`
  const timing = options?.timings?.find((t: any) => t.type === spec.timing.type)?.label || spec.timing.type
  return `${scorer} → ${selection} → ${weighting} → × ${timing} → ${frequencyLabel[spec.rebalance.frequency] || spec.rebalance.frequency}调仓`
}

export function StrategyCenter() {
  const queryClient = useQueryClient()
  const { data: strategies, isLoading, isError } = useQuery({ queryKey: ['strategies'], queryFn: async () => (await api.get('/strategies')).data })
  const { data: options } = useQuery({ queryKey: ['pipeline-options'], queryFn: async () => (await api.get('/pipeline/options')).data })
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [draft, setDraft] = useState<Draft>(blankDraft())
  const [message, setMessage] = useState('')
  const [savedId, setSavedId] = useState<number | null>(null)
  const library = useMemo(() => (strategies || []).filter((s: any) => s.origin !== 'paper_snapshot'), [strategies])
  const selected = library.find((s: any) => s.id === selectedId)

  useEffect(() => {
    if (selectedId === null && library.length) loadStrategy(library.find((s: any) => s.is_default) || library[0])
  }, [library])

  function loadStrategy(row: any) {
    setSelectedId(row.id)
    setSavedId(null)
    setDraft({
      name: row.origin === 'builtin' ? `${row.name}（我的版本）` : row.name,
      description: row.description || '',
      default_universe: row.default_universe || 'csi300',
      spec: JSON.parse(JSON.stringify(row.spec)),
    })
  }

  const setSpec = (patch: Partial<Spec>) => setDraft(d => ({ ...d, spec: { ...d.spec, ...patch } }))
  const save = useMutation({
    mutationFn: async () => (await api.post('/strategies/spec', draft)).data,
    onSuccess: data => {
      setMessage(`已另存为「${data.name}」V${data.version}`)
      setSavedId(data.id)
      setSelectedId(data.id)
      queryClient.invalidateQueries({ queryKey: ['strategies'] })
    },
    onError: (error: any) => setMessage(error?.response?.data?.detail || '保存失败，请检查参数'),
  })

  if (isLoading || !options) return <LoadingState />
  if (isError) return <ErrorState text="策略库加载失败" />

  const spec = draft.spec
  const weights = spec.scorer.type === 'factor_weights' ? spec.scorer.weights : {}
  const weightTotal = Object.values(weights).reduce((sum, w) => sum + Number(w || 0), 0)
  const timingOption = options.timings.find((t: any) => t.type === spec.timing.type)
  const unavailableUsed = spec.scorer.type === 'factor_weights'
    ? options.factors.filter((f: any) => !f.available && Number(weights[f.key] || 0) > 0)
    : []
  const canSave = draft.name.trim() && (spec.scorer.type === 'model' ? spec.scorer.models.length > 0 : weightTotal > 0)

  return <>
    <PageHeader eyebrow="RESEARCH / 03" title="策略制定" description="从策略库挑一个作为起点，按 打分 → 选股 → 权重 → 择时 → 调仓 五步配置，另存为自己的策略；回测在「策略实践」里做。"
      actions={<button className="primary-button" onClick={() => save.mutate()} disabled={!canSave || save.isPending}><Save size={16}/>{save.isPending ? '保存中…' : '另存为新策略'}</button>} />
    <div className="strategy-layout">
      <div>
        <Card>
          <PanelHeader title="基本信息" subtitle="策略保存后不再修改，改任何一项都会另存为新策略" />
          <div className="form-grid">
            <label>策略名称<input value={draft.name} onChange={e => setDraft({ ...draft, name: e.target.value })}/></label>
            <label>默认股票池（回测时可换）<select value={draft.default_universe} onChange={e => setDraft({ ...draft, default_universe: e.target.value })}>
              {options.universes.map((u: any) => <option key={u.id} value={u.id}>{u.name}</option>)}
            </select></label>
            <label className="full">策略说明<textarea value={draft.description} onChange={e => setDraft({ ...draft, description: e.target.value })}/></label>
          </div>
        </Card>

        <Card>
          <PanelHeader title="① 打分" subtitle="给股票池里每支股票一个分数；后面几步只认分数" action={
            <select className="inline-select" value={spec.scorer.type} onChange={e => setSpec({ scorer: e.target.value === 'model'
              ? { type: 'model', models: [options.models[0]?.id].filter(Boolean) }
              : { type: 'factor_weights', weights: { momentum: 0.6, risk: 0.4 } } })}>
              <option value="factor_weights">因子权重</option><option value="model">模型预测</option>
            </select>} />
          {spec.scorer.type === 'factor_weights' ? <div className="weight-editor">
            {options.factors.map((f: any) => <div className="weight-row" key={f.key}>
              <div className="weight-label"><span className={`weight-icon ${f.key}`}><SlidersHorizontal size={14}/></span><b>{f.label}</b><small>{f.available ? f.description : `${f.description} · ${f.unavailable_reason}`}</small></div>
              <input type="range" min="0" max="100" step="5" disabled={!f.available && !Number(weights[f.key] || 0)} value={Math.round(Number(weights[f.key] || 0) * 100)}
                onChange={e => setSpec({ scorer: { type: 'factor_weights', weights: { ...weights, [f.key]: Number(e.target.value) / 100 } } })}/>
              <strong>{Math.round(Number(weights[f.key] || 0) * 100)}%</strong>
            </div>)}
            <p className="muted-note">权重会按合计自动归一（当前合计 {Math.round(weightTotal * 100)}%）。每个因子先换成股票池内的百分位，再按权重加总。</p>
            {unavailableUsed.length > 0 && <p className="muted-note warn">当前是真实行情模式，{unavailableUsed.map((f: any) => f.label).join('、')} 没有真实数据，回测时按 0 权重计算。</p>}
          </div> : <div className="model-pick">
            {options.models.map((m: any) => {
              const checked = spec.scorer.type === 'model' && spec.scorer.models.includes(m.id)
              return <label key={m.id} className={`model-option ${checked ? 'active' : ''}`}>
                <input type="checkbox" checked={checked} onChange={() => {
                  const current = spec.scorer.type === 'model' ? spec.scorer.models : []
                  setSpec({ scorer: { type: 'model', models: checked ? current.filter(x => x !== m.id) : [...current, m.id] } })
                }}/>
                <div><b>{m.name}</b><span>{m.description}</span><small>训练股票池：{m.trained_on} · 可打分年份 {m.years[0]}–{m.years[m.years.length - 1]}</small></div>
              </label>
            })}
            <p className="muted-note">选多个模型时取预测分数的平均。模型只能给有训练结果的年份打分，回测区间会被限制在这些年份内。</p>
          </div>}
        </Card>

        <Card>
          <PanelHeader title="② 选股规则 与 ③ 权重方案" subtitle="按分数从高到低选；不提供'分数超过某值才买'，因为不同打分方式的分数量纲不同" />
          <div className="form-grid">
            <label>选股规则<select value={spec.selection.type} onChange={e => setSpec({ selection: e.target.value === 'top_n' ? { type: 'top_n', n: 30, pct: null } : { type: 'top_pct', pct: 0.1, n: null } })}>
              <option value="top_n">前 N 名</option><option value="top_pct">前 x%（按股票池大小）</option>
            </select></label>
            {spec.selection.type === 'top_n'
              ? <label>N<input type="number" min={1} max={500} value={spec.selection.n || 1} onChange={e => setSpec({ selection: { ...spec.selection, n: Math.max(1, Number(e.target.value)) } })}/></label>
              : <label>x（%）<input type="number" min={1} max={100} value={Math.round((spec.selection.pct || 0) * 100)} onChange={e => setSpec({ selection: { ...spec.selection, pct: Math.min(100, Math.max(1, Number(e.target.value))) / 100 } })}/></label>}
            <label>权重方案<select value={spec.weighting.type} onChange={e => setSpec({ weighting: { ...spec.weighting, type: e.target.value as any } })}>
              <option value="equal">等权</option><option value="score">按分数加权</option>
            </select></label>
            <label>单股上限（%）<input type="number" min={1} max={100} value={Math.round(spec.weighting.max_weight * 100)} onChange={e => setSpec({ weighting: { ...spec.weighting, max_weight: Math.min(100, Math.max(1, Number(e.target.value))) / 100 } })}/></label>
          </div>
          <p className="muted-note">入选股票都到了单股上限还不够满仓时，剩下的留作现金，不强行加仓。</p>
        </Card>

        <Card>
          <PanelHeader title="④ 择时信号 与 ⑤ 调仓" subtitle="择时决定整体仓位（0～100%），不决定买哪些；只在调仓日判断" />
          <div className="form-grid">
            <label>择时信号<select value={spec.timing.type} onChange={e => {
              const option = options.timings.find((t: any) => t.type === e.target.value)
              setSpec({ timing: { type: e.target.value, ...Object.fromEntries((option?.params || []).map((p: any) => [p.key, p.default])) } })
            }}>{options.timings.map((t: any) => <option key={t.type} value={t.type}>{t.label}</option>)}</select></label>
            {(timingOption?.params || []).map((p: any) => <label key={p.key}>{p.label}{p.percent ? '（%）' : ''}<input type="number" step={p.percent ? 5 : p.step} min={p.percent ? p.min * 100 : p.min} max={p.percent ? p.max * 100 : p.max}
              value={p.percent ? Math.round(Number(spec.timing[p.key] ?? p.default) * 100) : Number(spec.timing[p.key] ?? p.default)}
              onChange={e => setSpec({ timing: { ...spec.timing, [p.key]: p.percent ? Number(e.target.value) / 100 : Number(e.target.value) } })}/></label>)}
            <label>调仓频率<select value={spec.rebalance.frequency} onChange={e => setSpec({ rebalance: { ...spec.rebalance, frequency: e.target.value } })}>
              <option value="weekly">每周</option><option value="monthly">每月</option><option value="quarterly">每季度</option>
            </select></label>
            <label>换手阈值（%）<input type="number" min={0} max={50} value={Math.round(spec.rebalance.turnover_band * 100)} onChange={e => setSpec({ rebalance: { ...spec.rebalance, turnover_band: Math.max(0, Number(e.target.value)) / 100 } })}/></label>
          </div>
          {timingOption?.description && <p className="muted-note">{timingOption.description}</p>}
          <p className="muted-note">调仓时只交易目标持仓与当前持仓的差额；比例变化小于换手阈值的不动；按 100 股一手取整。</p>
        </Card>
      </div>

      <div>
        <Card className="sticky-card">
          <PanelHeader title="这个策略会怎么做" subtitle="保存前确认一遍" />
          <div className="strategy-note"><Sparkles size={17}/><div><b>{draft.name || '未命名策略'}</b><p>{describeSpec(spec, options)}</p></div></div>
          {savedId
            ? <Link className="primary-button full-button" to={`/backtest?strategy_id=${savedId}`}><ArrowRight size={15}/>去策略实践回测这个策略</Link>
            : <p className="muted-note">另存后可以直接去「策略实践」回测。</p>}
        </Card>
        <Card>
          <PanelHeader title="策略库" subtitle="点一个作为起点；内置策略不能改，改完会另存为你的版本" action={<Layers size={16}/>} />
          <div className="library-list">
            {['builtin', 'user'].map(origin => {
              const rows = library.filter((s: any) => (s.origin || 'user') === origin)
              if (!rows.length) return null
              return <div key={origin}>
                <div className="library-group">{originLabel[origin]} · {rows.length}</div>
                {rows.map((row: any) => <button key={row.id} className={`library-item ${row.id === selectedId ? 'active' : ''}`} onClick={() => loadStrategy(row)}>
                  <b>{row.name}{row.version > 1 ? ` · V${row.version}` : ''}{row.is_default ? ' · 默认' : ''}</b>
                  <span>{describeSpec(row.spec, options)}</span>
                  <small>{row.backtest_metrics?.annual_return != null
                    ? `最近一次回测 ${row.study_period?.start} ~ ${row.study_period?.end}：年化 ${formatPercent(row.backtest_metrics.annual_return)}，最大回撤 ${formatPercent(row.backtest_metrics.max_drawdown)}`
                    : '还没有回测'}</small>
                </button>)}
              </div>
            })}
          </div>
          {selected && <p className="muted-note">当前起点：{selected.name}（{originLabel[selected.origin] || selected.origin}）</p>}
        </Card>
      </div>
    </div>
    {message && <Toast message={message} type={save.isError ? 'error' : 'success'} />}
  </>
}
