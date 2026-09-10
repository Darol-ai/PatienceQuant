import axios from 'axios'

export const api = axios.create({ baseURL: '/api', timeout: 120000 })

export const formatMoney = (value = 0) =>
  new Intl.NumberFormat('zh-CN', { style: 'currency', currency: 'CNY', maximumFractionDigits: 0 }).format(value)

export const formatPercent = (value = 0, digits = 2) => `${(value * 100).toFixed(digits)}%`
export const formatNumber = (value = 0, digits = 2) => Number(value).toFixed(digits)
const DATA_MODE_LABELS: Record<string, string> = {
  real: 'Real 数据',
  mixed: 'Real + Demo 混合数据',
  demo_fallback: 'Demo fallback 数据',
  demo_mixed: 'Demo + fallback 数据',
  demo: 'Demo 数据',
}

export const dataModeLabel = (mode = 'demo') => DATA_MODE_LABELS[mode] || `${mode} 数据`

export type Stock = {
  symbol: string; name: string; exchange: string; industry: string; group: string; tags: string[]; score: number; rank: number;
  action: 'BUY' | 'HOLD' | 'SELL' | 'WATCH'; current_price: number; pe: number; revenue_growth: number;
  return_12m: number; volatility: number; max_drawdown: number; target_weight: number;
  score_fundamental: number; score_valuation: number; score_quality: number; score_momentum: number; score_risk: number;
}

export type Strategy = {
  id: number; name: string; version: number; description: string; weights: Record<string, number>;
  holdings_count: number; max_weight: number; rebalance_frequency: string; is_default: boolean;
  target_volatility?: number; max_drawdown_budget?: number; drawdown_brake_exposure?: number;
}
