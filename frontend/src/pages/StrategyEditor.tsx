import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, Save, SlidersHorizontal, Sparkles } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import { Card, ErrorState, LoadingState, PageHeader, PanelHeader, Toast } from '../components/UI'
import { describeSpec, originLabel, Spec } from '../strategy'
import { TrainingConfig, TrainingForm } from '../components/TrainingForm'

// 策略制定（ADR-0047 第 3 条、ADR-0051）：从策略库的「基于它新建」或「新建策略」进入的编辑器。
// 策略 = 打分 → ① 选股 → ② 权重 → ③ × 择时 → ④ 调仓；保存一律另存为新策略，存完进入它的详情页。

type Draft = { name: string; description: string; default_universe: string; spec: Spec }

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

export function StrategyEditor() {
  const queryClient = useQueryClient()
  const { data: strategies, isLoading, isError } = useQuery({ queryKey: ['strategies'], queryFn: async () => (await api.get('/strategies')).data })
  const { data: options } = useQuery({ queryKey: ['pipeline-options'], queryFn: async () => (await api.get('/pipeline/options')).data })
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const fromId = Number(params.get('from')) || null
  const [draft, setDraft] = useState<Draft>(blankDraft())
  const [loadedFrom, setLoadedFrom] = useState<number | null>(null)
  const [message, setMessage] = useState('')
  // 模型选股策略：reuseModels 非空表示沿用起点策略的模型（训练设置没改）；否则按 training 训练新模型
  const [training, setTraining] = useState<TrainingConfig | null>(null)
  const [reuseModels, setReuseModels] = useState<string[] | null>(null)
  const { data: pools } = useQuery({ queryKey: ['pools'], queryFn: async () => (await api.get('/pools')).data })
  const source = fromId ? (strategies || []).find((s: any) => s.id === fromId) : null

  useEffect(() => {
    if (!source || loadedFrom === source.id) return
    setLoadedFrom(source.id)
    setDraft({
      name: `${source.name}（我的版本）`,
      description: source.description || '',
      default_universe: source.default_universe || 'csi300',
      spec: JSON.parse(JSON.stringify(source.spec)),
    })
    setReuseModels(source.spec.scorer.type === 'model' ? source.spec.scorer.models : null)
  }, [source, loadedFrom])

  function trainingFrom(modelIds: string[] | undefined): TrainingConfig {
    const model = (options?.models || []).find((m: any) => m.id === modelIds?.[0])
    if (model?.config) return { ...options.training_defaults, ...model.config }
    // 旧模型没有训练记录：14 个因子、90 日，股票池按它训练时的池；改成用历史成分股训练
    return { ...options.training_defaults, ...(model ? { framework: model.framework, pool: model.id.includes('broad30') ? 'broad30' : 'csi300' } : {}) }
  }

  const setSpec = (patch: Partial<Spec>) => setDraft(d => ({ ...d, spec: { ...d.spec, ...patch } }))
  const save = useMutation({
    mutationFn: async () => {
      const { scorer, ...rest } = draft.spec
      if (scorer.type === 'model' && !reuseModels) {
        return (await api.post('/strategies/model', { name: draft.name, description: draft.description,
          default_universe: draft.default_universe, training, spec: rest })).data
      }
      return (await api.post('/strategies/spec', draft)).data
    },
    onSuccess: data => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] })
      queryClient.invalidateQueries({ queryKey: ['scorecards'] })
      navigate(`/library/${data.id}`)
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
  const canSave = draft.name.trim() && (spec.scorer.type === 'model'
    ? (reuseModels ? reuseModels.length > 0 : !!training && training.factors.length > 0)
    : weightTotal > 0)

  return <>
    <PageHeader eyebrow="策略库 · 新建" title={<span className="title-with-back"><Link to={fromId ? `/library/${fromId}` : '/library'} className="icon-button"><ArrowLeft size={16}/></Link>策略制定</span>}
      description={source ? `以「${source.name}」（${originLabel[source.origin] || source.origin}）为起点。按 打分 → 选股 → 权重 → 择时 → 调仓 五步配置，保存后是一个新策略，原策略不变。` : '从空白开始。按 打分 → 选股 → 权重 → 择时 → 调仓 五步配置，保存后进入策略库。'}
      actions={<button className="primary-button" onClick={() => save.mutate()} disabled={!canSave || save.isPending}><Save size={16}/>{save.isPending ? '保存中…' : spec.scorer.type === 'model' && !reuseModels ? '保存并训练' : '另存为新策略'}</button>} />
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
            <select className="inline-select" value={spec.scorer.type} onChange={e => {
              if (e.target.value === 'model') {
                setReuseModels(null)
                setTraining(options.training_defaults)
                setSpec({ scorer: { type: 'model', models: [] } })
              } else setSpec({ scorer: { type: 'factor_weights', weights: { momentum: 0.6, risk: 0.4 } } })
            }}>
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
          </div> : reuseModels ? <div className="model-pick">
            <div className="strategy-note"><div>
              <b>沿用起点策略的模型</b>
              {reuseModels.map(id => {
                const m = options.models.find((x: any) => x.id === id)
                return <p key={id}>{m?.name || id}{m ? ` · ${m.factors.length} 个输入因子 · 预测 ${m.horizon_days} 日 · 可打分 ${m.years[0]}–${m.years[m.years.length - 1]}` : ''}</p>
              })}
              <p>只改选股、权重、择时、调仓时沿用同一个模型，不重新训练。</p>
            </div></div>
            <button className="secondary-button" onClick={() => { setTraining(trainingFrom(reuseModels)); setReuseModels(null) }}>改训练设置（会训练一个新模型）</button>
          </div> : training && <TrainingForm value={training} onChange={setTraining} options={options} strategies={strategies || []} pools={pools || []}/>}
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
          <PanelHeader title="④ 择时信号 与 ⑤ 调仓" subtitle="择时决定整体仓位（0～100%），不决定买哪些；每个交易日按收盘信号调整仓位" />
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
          <p className="muted-note">保存后会进入这个策略的详情页，在那里计算成绩卡或回测。</p>
        </Card>
      </div>
    </div>
    {message && <Toast message={message} type={save.isError ? 'error' : 'success'} />}
  </>
}
