import { formatPercent } from './api'

// 策略规格（后端 app/pipeline/spec.py 的 StrategySpec）和几个页面共用的文字说明。

export type Spec = {
  market: string
  scorer: { type: 'factor_weights'; weights: Record<string, number> } | { type: 'model'; models: string[] }
  timing: { type: string; [key: string]: any }
  selection: { type: 'top_n' | 'top_pct'; n?: number | null; pct?: number | null }
  weighting: { type: 'equal' | 'score'; max_weight: number }
  rebalance: { frequency: string; turnover_band: number }
}

export const frequencyLabel: Record<string, string> = { weekly: '每周', monthly: '每月', quarterly: '每季度' }
export const originLabel: Record<string, string> = { builtin: '内置', user: '我的', paper_snapshot: '模拟盘副本' }
export const timingLabel: Record<string, string> = {
  none: '不择时', index_trend: '均线趋势', rsrs: 'RSRS', icu_ma: 'ICU 均线', alligator: '鳄鱼线',
  llt: 'LLT 趋势线', ma_channel: '均线交叉通道突破', one_way_vol: '单向波动差', rps_vol: 'RPS 单向波动差',
  high_moment: '高阶矩', volume_resonance: '价量共振', qrs: 'QRS',
}

export function scorerText(spec: Spec | undefined, options?: any): string {
  if (!spec) return ''
  if (spec.scorer.type === 'model') {
    const name = (id: string) => options?.models?.find((m: any) => m.id === id)?.name || id
    return `模型打分（${spec.scorer.models.map(name).join(' + ')}）`
  }
  const label = (key: string) => options?.factors?.find((f: any) => f.key === key)?.label || key
  return `因子打分（${Object.entries(spec.scorer.weights).filter(([, w]) => Number(w) > 0).map(([k, w]) => `${label(k)} ${Math.round(Number(w) * 100)}%`).join('、')}）`
}

export function selectionText(spec: Spec): string {
  return spec.selection.type === 'top_n' ? `前 ${spec.selection.n} 名` : `前 ${formatPercent(spec.selection.pct || 0, 0)}`
}

export function weightingText(spec: Spec): string {
  return `${spec.weighting.type === 'equal' ? '等权' : '按分数加权'}，单股 ≤ ${formatPercent(spec.weighting.max_weight, 0)}`
}

export function timingText(spec: Spec, options?: any): string {
  return options?.timings?.find((t: any) => t.type === spec.timing.type)?.label || timingLabel[spec.timing.type] || spec.timing.type
}

export function describeSpec(spec: Spec | undefined, options?: any): string {
  if (!spec) return ''
  return `${scorerText(spec, options)} → ${selectionText(spec)} → ${weightingText(spec)} → × ${timingText(spec, options)} → ${frequencyLabel[spec.rebalance.frequency] || spec.rebalance.frequency}调仓`
}

const UNIVERSE_NAMES: Record<string, string> = { csi300: '沪深300成分股', broad30: '30支候选池', a_share: 'A股全市场', custom: '迁移前的记录' }

export function universeLabel(id: string | undefined, pools?: any[]): string {
  if (!id) return '—'
  return pools?.find((p: any) => p.id === id)?.name || UNIVERSE_NAMES[id] || id
}
