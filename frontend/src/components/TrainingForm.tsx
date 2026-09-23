import { useMemo, useState } from 'react'

// 模型选股策略的训练设置（ADR-0052）：算法、输入因子（来自因子库）、预测周期、训练股票池、少量超参数。
// 滚动训练规则固定：给第 Y 年打分的模型只用 Y−1 年 6 月底前已揭晓的样本训练、Y−1 年下半年验证。

export type TrainingConfig = {
  framework: 'lightgbm' | 'xgboost'
  factors: string[]
  horizon: number
  pool: string
  membership?: string
  n_estimators: number
  learning_rate: number
  num_leaves: number
  max_depth: number
  seeds: number
  cs_rank: boolean
}

export function TrainingForm({ value, onChange, options, strategies, pools }: {
  value: TrainingConfig
  onChange: (next: TrainingConfig) => void
  options: any
  strategies: any[]
  pools: any[]
}) {
  const [importFrom, setImportFrom] = useState('')
  const set = (patch: Partial<TrainingConfig>) => onChange({ ...value, ...patch })
  const groups = useMemo(() => {
    const byGroup: Record<string, any[]> = {}
    for (const f of options.training_factors) (byGroup[f.group] ||= []).push(f)
    return byGroup
  }, [options])
  const trainable = new Set(options.training_factors.map((f: any) => f.key))
  const factorStrategies = strategies.filter((s: any) => s.spec?.scorer?.type === 'factor_weights' && s.origin !== 'paper_snapshot')

  function toggle(key: string) {
    set({ factors: value.factors.includes(key) ? value.factors.filter(k => k !== key) : [...value.factors, key] })
  }

  function importFactors() {
    const source = factorStrategies.find((s: any) => String(s.id) === importFrom)
    if (!source) return
    const keys = Object.entries(source.spec.scorer.weights).filter(([, w]) => Number(w) > 0)
      .flatMap(([k]) => options.factor_groups[k] || [k]).filter(k => trainable.has(k))
    set({ factors: Array.from(new Set(keys)) })
  }

  return <div className="training-form">
    <div className="form-grid">
      <label>算法<select value={value.framework} onChange={e => set({ framework: e.target.value as any })}>
        <option value="lightgbm">LightGBM</option><option value="xgboost">XGBoost（用 GPU）</option>
      </select></label>
      <label>预测周期<select value={value.horizon} onChange={e => set({ horizon: Number(e.target.value) })}>
        <option value={20}>未来 20 个交易日收益</option><option value={60}>未来 60 个交易日收益</option><option value={90}>未来 90 个交易日收益</option>
      </select></label>
      <label className="full">训练股票池<select value={value.pool} onChange={e => set({ pool: e.target.value })}>
        {pools.filter((p: any) => p.id !== 'a_share').map((p: any) => <option key={p.id} value={p.id}>{p.name} · {p.count} 只{p.id === 'csi300' ? '（按历史成分股）' : ''}</option>)}
      </select></label>
    </div>

    <div className="training-factors">
      <div className="training-factors-head">
        <b>输入因子 · 已选 {value.factors.length} 个</b>
        <span>
          <select value={importFrom} onChange={e => setImportFrom(e.target.value)}>
            <option value="">从某个因子权重策略导入…</option>
            {factorStrategies.map((s: any) => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
          <button className="secondary-button" disabled={!importFrom} onClick={importFactors}>导入</button>
          <button className="secondary-button" onClick={() => set({ factors: options.training_defaults.factors })}>恢复旧模型的 14 个</button>
        </span>
      </div>
      {Object.entries(groups).map(([group, factors]) => <div key={group} className="factor-group">
        <span className="library-group">{group}</span>
        <div className="factor-chips">{factors.map((f: any) => <label key={f.key} className={`factor-chip ${value.factors.includes(f.key) ? 'active' : ''}`} title={f.description}>
          <input type="checkbox" checked={value.factors.includes(f.key)} onChange={() => toggle(f.key)}/>{f.label}
        </label>)}</div>
      </div>)}
      <p className="muted-note">基本面、估值、盈利质量类因子需要财务数据，还没接入，不能用来训练。</p>
    </div>

    <div className="form-grid">
      <label>树的数量<input type="number" min={50} max={2000} step={50} value={value.n_estimators} onChange={e => set({ n_estimators: Number(e.target.value) })}/></label>
      <label>学习率<input type="number" min={0.001} max={0.5} step={0.01} value={value.learning_rate} onChange={e => set({ learning_rate: Number(e.target.value) })}/></label>
      {value.framework === 'lightgbm'
        ? <label>叶子数<input type="number" min={4} max={255} value={value.num_leaves} onChange={e => set({ num_leaves: Number(e.target.value) })}/></label>
        : <label>树深度<input type="number" min={2} max={12} value={value.max_depth} onChange={e => set({ max_depth: Number(e.target.value) })}/></label>}
      <label>随机种子个数（预测取平均）<input type="number" min={1} max={5} value={value.seeds} onChange={e => set({ seeds: Math.min(5, Math.max(1, Number(e.target.value))) })}/></label>
      <label className="checkbox-row full"><span>因子先换成当天在股票池里的百分位（默认用原始值，和旧模型一致）</span>
        <input type="checkbox" checked={value.cs_rank} onChange={e => set({ cs_rank: e.target.checked })}/></label>
    </div>
    <p className="muted-note">滚动训练规则固定：给第 Y 年打分的模型，只用 Y−1 年 6 月底前已揭晓的样本训练、Y−1 年下半年的样本做验证，在第 Y 年上预测；数据从 2016 年开始，最早能给 2019 年打分。保存后在后台训练，训练好会自动算成绩卡。</p>
  </div>
}
