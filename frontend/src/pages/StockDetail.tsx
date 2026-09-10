import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, BrainCircuit, CalendarDays, TrendingDown, TrendingUp } from 'lucide-react'
import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import ReactECharts from 'echarts-for-react'
import { api, formatPercent } from '../api'
import { Card, ErrorState, LoadingState, PageHeader, PanelHeader, SignalBadge } from '../components/UI'

const ranges = [{ value: '1y', label: '1年' }, { value: '3y', label: '3年' }, { value: '5y', label: '5年' }, { value: 'all', label: '全历史' }]
const intervals = [{ value: 'daily', label: '日线' }, { value: 'weekly', label: '周线' }, { value: 'monthly', label: '月线' }]

export function StockDetail() {
  const { symbol } = useParams()
  const [range, setRange] = useState('all')
  const [interval, setInterval] = useState('monthly')
  const { data, isLoading, isError } = useQuery({ queryKey: ['stock', symbol, range, interval], queryFn: async () => (await api.get(`/stocks/${symbol}`, { params: { range, interval } })).data, enabled: !!symbol })
  if (isLoading) return <LoadingState text="正在加载历史行情…" />
  if (isError || !data) return <ErrorState text="股票详情加载失败" />
  const stock = data.stock
  const chart = data.chart || {}
  const option = {
    animationDuration: 350,
    tooltip: { trigger: 'axis', backgroundColor: '#101d2d', borderColor: '#29405b', textStyle: { color: '#e8f1fb' }, valueFormatter: (value: number) => `¥${Number(value).toFixed(2)}` },
    grid: { left: 8, right: 12, top: 16, bottom: 38, containLabel: true },
    xAxis: { type: 'category', data: data.prices.map((p: any) => p.trade_date), axisLabel: { color: '#7890a8', formatter: (v: string) => interval === 'daily' ? v.slice(0, 7) : v.slice(0, 10) }, axisLine: { lineStyle: { color: '#29405b' } } },
    yAxis: { type: 'value', scale: true, axisLabel: { color: '#7890a8', formatter: (v: number) => `¥${v}` }, splitLine: { lineStyle: { color: 'rgba(120,144,168,.1)' } } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 3, borderColor: '#29405b', textStyle: { color: '#7890a8' } }],
    series: [{ name: '复权收盘价', type: 'line', showSymbol: interval !== 'daily', symbolSize: 5, smooth: interval !== 'daily', data: data.prices.map((p: any) => p.adj_close), lineStyle: { color: '#31d0aa', width: 2.5 }, areaStyle: { color: 'rgba(49,208,170,.15)' } }],
  }
  return <><PageHeader eyebrow={`EQUITY / ${stock.symbol}`} title={<span className="title-with-back"><Link to="/stocks"><ArrowLeft size={20}/></Link>{stock.name}<small>{stock.symbol} · {stock.exchange || 'A股'} · {stock.industry}</small></span>} description={`${stock.group}研究组 · ${stock.sector}板块 · ${stock.tags?.join(' / ')}`} actions={<><SignalBadge signal={stock.action}/><Link className="secondary-button" to="/ai"><BrainCircuit size={16}/>AI 研究</Link></>} /><div className="detail-grid"><Card className="span-8"><PanelHeader title="历史价格走势" subtitle="复权收盘价 · 支持日线、周线、月线与全历史" action={<div className="chart-controls"><div className="chart-control-group">{ranges.map(item => <button key={item.value} className={range === item.value ? 'active' : ''} onClick={() => setRange(item.value)}>{item.label}</button>)}</div><div className="chart-control-group">{intervals.map(item => <button key={item.value} className={interval === item.value ? 'active' : ''} onClick={() => setInterval(item.value)}>{item.label}</button>)}</div></div>}/><ReactECharts option={option} style={{height: 410}}/><div className="chart-summary"><span><CalendarDays size={14}/> {chart.start_date || '—'} 至 {chart.end_date || '—'}</span><span>{chart.points || 0} 个{interval === 'daily' ? '交易日' : interval === 'weekly' ? '周度' : '月度'}点</span><b className={chart.return >= 0 ? 'text-mint' : 'text-rose'}>{chart.return == null ? '—' : `${formatPercent(chart.return)} 区间涨跌`}</b><span>最高 ¥{Number(chart.high || 0).toFixed(2)} · 最低 ¥{Number(chart.low || 0).toFixed(2)}</span></div></Card><Card className="span-4"><PanelHeader title="量化画像" subtitle="当前截面因子状态"/><div className="factor-score-list">{[['综合评分', stock.score, 'mint'], ['基本面', stock.score_fundamental, 'blue'], ['估值', stock.score_valuation, 'gold'], ['盈利质量', stock.score_quality, 'purple'], ['动量', stock.score_momentum, 'mint'], ['风险', stock.score_risk, 'rose']].map(([label, value, tone]) => <div className="factor-row" key={label as string}><div><span>{label}</span><b>{Number(value).toFixed(1)}</b></div><div className="factor-bar"><i className={tone as string} style={{width: `${Number(value)}%`}}/></div></div>)}</div></Card></div><div className="detail-grid"><Card><PanelHeader title="关键数据" subtitle="研究快照"/><div className="stat-grid"><div><span>PE</span><strong>{stock.pe.toFixed(1)}x</strong></div><div><span>营收增长</span><strong className="text-mint">{formatPercent(stock.revenue_growth)}</strong></div><div><span>12M 动量</span><strong className={stock.return_12m >= 0 ? 'text-mint' : 'text-rose'}>{formatPercent(stock.return_12m)}</strong></div><div><span>波动率</span><strong>{formatPercent(stock.volatility)}</strong></div><div><span>最大回撤</span><strong className="text-rose">-{formatPercent(stock.max_drawdown)}</strong></div><div><span>目标权重</span><strong>{formatPercent(stock.target_weight)}</strong></div></div></Card><Card><PanelHeader title="交易解读" subtitle="规则解释器可追溯"/><div className="explain-callout"><div className="callout-icon">{stock.action === 'BUY' ? <TrendingUp/> : <TrendingDown/>}</div><div><b>{stock.action === 'BUY' ? '进入组合候选' : '保持跟踪'}</b><p>{stock.name} 当前综合评分 {stock.score.toFixed(1)}，排名第 {stock.rank}。最终交易仍由 Quant Engine 统一生成。</p></div></div></Card></div></>
}
