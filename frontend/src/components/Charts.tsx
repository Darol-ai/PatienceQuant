import ReactECharts from 'echarts-for-react'

const axis = { axisLine: { lineStyle: { color: '#29405b' } }, axisLabel: { color: '#7890a8', fontSize: 11 }, splitLine: { lineStyle: { color: 'rgba(120,144,168,.1)' } } }

export function EquityChart({ data, trades = [] }: { data: any[]; trades?: any[] }) {
  const markerSeries = (side: string, color: string, symbol: string) => ({
    name: side === 'BUY' ? '买入点' : '卖出点',
    type: 'scatter',
    symbol,
    symbolSize: 11,
    data: trades.filter((trade: any) => trade.side === side).map((trade: any) => ({
      value: [trade.trade_date, trade.equity],
      trade_symbol: trade.symbol,
      quantity: trade.quantity,
      amount: trade.amount,
      reason: trade.reason,
    })),
    itemStyle: { color, shadowBlur: 8, shadowColor: color },
    tooltip: { formatter: (params: any) => `<b>${side === 'BUY' ? '买入' : '卖出'} ${params.data.trade_symbol || ''}</b><br/>${params.data.value[0]} · ${params.data.quantity} 股<br/>金额 ¥${Number(params.data.amount || 0).toLocaleString()}<br/>${params.data.reason || ''}` },
    z: 5,
  })
  const option = { tooltip: { trigger: 'axis', backgroundColor: '#101d2d', borderColor: '#29405b', textStyle: { color: '#e8f1fb' } }, legend: { data: ['策略净值', '沪深300', '买入点', '卖出点'], top: 4, right: 10, textStyle: { color: '#8ba0b5' } }, grid: { left: 16, right: 16, top: 45, bottom: 26, containLabel: true }, xAxis: { type: 'category', boundaryGap: false, data: data.map(i => i.trade_date), ...axis }, yAxis: { type: 'value', scale: true, ...axis }, dataZoom: [{ type: 'inside' }], series: [{ name: '策略净值', type: 'line', showSymbol: false, smooth: .15, data: data.map(i => i.equity), lineStyle: { color: '#31d0aa', width: 2.5 }, areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: 'rgba(49,208,170,.28)' }, { offset: 1, color: 'rgba(49,208,170,0)' }] } } }, { name: '沪深300', type: 'line', showSymbol: false, data: data.map(i => i.benchmark), lineStyle: { color: '#6c8cff', width: 1.8 } }, markerSeries('BUY', '#31d0aa', 'triangle'), markerSeries('SELL', '#fb7185', 'diamond')] }
  return <ReactECharts option={option} style={{ height: 330 }} />
}

export function StockTradeChart({ data, trades = [], height = 360 }: { data: any[]; trades?: any[]; height?: number }) {
  const markerSeries = (side: string, color: string, markerSymbol: string) => ({
    name: side === 'BUY' ? '买入' : '卖出', type: 'scatter', symbol: markerSymbol, symbolSize: 13, z: 6,
    data: trades.filter((trade: any) => trade.side === side).map((trade: any) => ({
      value: [trade.trade_date, trade.price],
      trade_symbol: trade.symbol,
      quantity: trade.quantity,
      amount: trade.amount,
      reason: trade.reason,
    })),
    itemStyle: { color, shadowBlur: 9, shadowColor: color },
    tooltip: { formatter: (params: any) => `<b>${side === 'BUY' ? '买入' : '卖出'} ${params.data.trade_symbol || ''}</b><br/>${params.data.value[0]}<br/>成交价 ¥${Number(params.data.value[1]).toFixed(2)}<br/>${params.data.quantity} 股 · ¥${Number(params.data.amount).toLocaleString()}<br/>${params.data.reason || ''}` },
  })
  const option = { tooltip: { trigger: 'axis', backgroundColor: '#101d2d', borderColor: '#29405b', textStyle: { color: '#e8f1fb' } }, legend: { data: ['复权收盘价', '买入', '卖出'], right: 10, textStyle: { color: '#8ba0b5' } }, grid: { left: 12, right: 15, top: 42, bottom: 38, containLabel: true }, xAxis: { type: 'category', data: data.map(i => i.trade_date), ...axis }, yAxis: { type: 'value', scale: true, axisLabel: { color: '#7890a8', formatter: (v: number) => `¥${v}` }, splitLine: axis.splitLine }, dataZoom: [{ type: 'inside' }, { type: 'slider', height: 17, bottom: 3, borderColor: '#29405b', textStyle: { color: '#7890a8' } }], series: [{ name: '复权收盘价', type: 'line', showSymbol: false, data: data.map(i => i.adj_close), lineStyle: { color: '#6c8cff', width: 2 }, areaStyle: { color: 'rgba(108,140,255,.12)' } }, markerSeries('BUY', '#31d0aa', 'triangle'), markerSeries('SELL', '#fb7185', 'diamond')] }
  return <ReactECharts option={option} style={{ height }} />
}

export function DrawdownChart({ data }: { data: any[] }) {
  const option = { tooltip: { trigger: 'axis' }, grid: { left: 12, right: 16, top: 20, bottom: 25, containLabel: true }, xAxis: { type: 'category', data: data.map(i => i.trade_date), ...axis }, yAxis: { type: 'value', axisLabel: { formatter: (v: number) => `${(v*100).toFixed(0)}%`, color: '#7890a8' }, splitLine: axis.splitLine }, series: [{ type: 'line', showSymbol: false, data: data.map(i => i.drawdown), lineStyle: { color: '#fb7185' }, areaStyle: { color: 'rgba(251,113,133,.22)' } }] }
  return <ReactECharts option={option} style={{ height: 240 }} />
}

export function PieChart({ data, nameKey = 'industry', valueKey = 'count' }: { data: any[]; nameKey?: string; valueKey?: string }) {
  const option = { tooltip: { trigger: 'item' }, legend: { orient: 'vertical', right: 0, top: 'middle', textStyle: { color: '#8ba0b5' } }, series: [{ type: 'pie', radius: ['48%', '72%'], center: ['36%', '50%'], padAngle: 2, itemStyle: { borderRadius: 5 }, label: { show: false }, data: data.map(i => ({ name: i[nameKey], value: i[valueKey] })) }] }
  return <ReactECharts option={option} style={{ height: 280 }} />
}

export function ScatterChart({ data }: { data: any[] }) {
  const option = { tooltip: { formatter: (p: any) => `${p.data[3]} ${p.data[4]}<br/>PE ${p.data[0].toFixed(1)} · 营收增长 ${(p.data[1]*100).toFixed(1)}%<br/>评分 ${p.data[2].toFixed(1)}` }, grid: { left: 15, right: 15, top: 20, bottom: 25, containLabel: true }, xAxis: { name: 'PE', ...axis }, yAxis: { name: '营收增长', axisLabel: { formatter: (v: number) => `${(v*100).toFixed(0)}%`, color: '#7890a8' }, splitLine: axis.splitLine }, visualMap: { min: 20, max: 90, dimension: 2, orient: 'horizontal', left: 'center', bottom: 0, inRange: { color: ['#fb7185', '#f5b94c', '#31d0aa'] }, textStyle: { color: '#7890a8' } }, series: [{ type: 'scatter', symbolSize: (v: number[]) => Math.max(8, v[2] / 5), data: data.map(i => [i.pe, i.revenue_growth, i.score, i.name, i.symbol]) }] }
  return <ReactECharts option={option} style={{ height: 310 }} />
}

export function BarChart({ data, category = 'year', value = 'strategy', percent = true, compare }: { data: any[]; category?: string; value?: string; percent?: boolean; compare?: { key: string; label: string } }) {
  const option = { tooltip: { trigger: 'axis' }, grid: { left: 15, right: 12, top: 20, bottom: 22, containLabel: true }, xAxis: { type: 'category', data: data.map(i => i[category]), ...axis }, yAxis: { type: 'value', axisLabel: { formatter: (v: number) => percent ? `${(v*100).toFixed(0)}%` : v, color: '#7890a8' }, splitLine: axis.splitLine }, series: [{ type: 'bar', barMaxWidth: 30, name: '策略', color: '#31d0aa', data: data.map(i => ({ value: i[value], itemStyle: { color: i[value] >= 0 ? '#31d0aa' : '#fb7185', borderRadius: i[value] >= 0 ? [4,4,0,0] : [0,0,4,4] } })) }, ...(compare ? [{ type: 'bar', barMaxWidth: 30, name: compare.label, data: data.map(i => i[compare.key]), itemStyle: { color: '#94a3b8' } }] : [])], ...(compare ? { legend: { top: 0, textStyle: { color: '#7890a8' } } } : {}) }
  return <ReactECharts option={option} style={{ height: 260 }} />
}

// 股票池分析（ADR-0053）：成员近一年收益-波动散点，按板块/行业着色
export function PoolScatterChart({ points }: { points: any[] }) {
  const groups = Array.from(new Set(points.map(p => p.group)))
  const option = {
    tooltip: { formatter: (p: any) => `<b>${p.data[3]} ${p.data[2]}</b><br/>${p.seriesName}<br/>近一年收益 ${(p.data[1] * 100).toFixed(1)}%<br/>年化波动 ${(p.data[0] * 100).toFixed(1)}%` },
    legend: { type: 'scroll', bottom: 0, textStyle: { color: '#8ba0b5' } },
    grid: { left: 15, right: 20, top: 30, bottom: 60, containLabel: true },
    xAxis: { name: '年化波动', nameLocation: 'middle', nameGap: 26, ...axis, axisLabel: { color: '#7890a8', formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
    yAxis: { name: '近一年收益', ...axis, axisLabel: { color: '#7890a8', formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
    series: groups.map(g => ({ name: g, type: 'scatter', symbolSize: 10, data: points.filter(p => p.group === g).map(p => [p.volatility, p.return, p.name, p.symbol]) })),
  }
  return <ReactECharts option={option} style={{ height: 340 }} />
}

export function PoolCurveChart({ data }: { data: any[] }) {
  const option = {
    tooltip: { trigger: 'axis', valueFormatter: (v: number) => `${((v - 1) * 100).toFixed(1)}%` },
    legend: { data: ['股票池等权', '沪深300'], top: 0, textStyle: { color: '#8ba0b5' } },
    grid: { left: 15, right: 16, top: 34, bottom: 25, containLabel: true },
    xAxis: { type: 'category', boundaryGap: false, data: data.map(i => i.date), ...axis },
    yAxis: { type: 'value', scale: true, ...axis, axisLabel: { color: '#7890a8', formatter: (v: number) => `${((v - 1) * 100).toFixed(0)}%` } },
    series: [{ name: '股票池等权', type: 'line', showSymbol: false, color: '#31d0aa', data: data.map(i => i.pool), lineStyle: { width: 2.2 } },
             { name: '沪深300', type: 'line', showSymbol: false, color: '#6c8cff', data: data.map(i => i.benchmark), lineStyle: { width: 1.6 } }],
  }
  return <ReactECharts option={option} style={{ height: 280 }} />
}
