import { formatPercent } from '../api'

// 模型状态与样本外成绩的小部件（ADR-0052），数据与模型页和策略详情页共用。
export const statusText: Record<string, string> = { queued: '排队中', training: '训练中', ready: '可用', failed: '训练失败', cancelled: '已取消' }
export const frameworkText: Record<string, string> = { lightgbm: 'LightGBM', xgboost: 'XGBoost', lstm: 'LSTM', transformer: 'Transformer' }

export function ModelStatus({ model }: { model: any }) {
  return <span className={`model-status ${model.status}`}>{statusText[model.status] || model.status}{model.progress ? ` · ${model.progress}` : ''}</span>
}

export const isTraining = (models: any[] | undefined) => (models || []).some(m => m.status === 'queued' || m.status === 'training')

// 逐年样本外成绩：年份 → {ic, icir, top10_excess, samples} 或 {skipped}
export function yearRows(model: any): [string, any][] {
  return Object.entries(model.metrics || {}).filter(([, v]: any) => v && v.ic !== undefined)
}

export function meanOf(model: any, key: 'ic' | 'top10_excess'): number | null {
  const values = yearRows(model).map(([, v]) => v[key]).filter((v: any) => typeof v === 'number')
  return values.length ? values.reduce((a: number, b: number) => a + b, 0) / values.length : null
}

export function ModelYears({ model }: { model: any }) {
  const rows = yearRows(model)
  if (!rows.length) return <p className="muted-note">{model.origin === 'legacy' ? '旧模型是离线训练后原样导入的，没有记录逐年样本外成绩。' : '还没有样本外成绩。'}</p>
  return <div className="table-wrap"><table>
    <thead><tr><th>年份</th><th>训练样本</th><th>样本外 IC</th><th>ICIR</th><th>前 10% 超额（相对池均值）</th></tr></thead>
    <tbody>{rows.map(([year, v]) => <tr key={year}>
      <td>{year}</td><td>{v.train_samples?.toLocaleString() ?? '—'}</td>
      <td className={v.ic > 0 ? 'text-mint' : 'text-rose'}>{v.ic?.toFixed(4) ?? '—'}</td>
      <td>{v.icir?.toFixed(2) ?? '—'}</td>
      <td className={v.top10_excess > 0 ? 'text-mint' : 'text-rose'}>{v.top10_excess == null ? '—' : formatPercent(v.top10_excess)}</td>
    </tr>)}</tbody>
  </table></div>
}
