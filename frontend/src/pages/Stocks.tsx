import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Filter, ListFilter, Search, SlidersHorizontal, X } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useEffect, useMemo, useState } from 'react'
import { api, formatPercent } from '../api'
import { BarChart, PieChart, ScatterChart } from '../components/Charts'
import { Card, DataBadge, ErrorState, LoadingState, PageHeader, PanelHeader, SignalBadge, Toast } from '../components/UI'

export function Stocks() {
  const queryClient = useQueryClient()
  const [search, setSearch] = useState('')
  const [group, setGroup] = useState('')
  const [exchange, setExchange] = useState('')
  const [signal, setSignal] = useState('')
  const [universe, setUniverse] = useState('all_assets')
  const [page, setPage] = useState(1)
  const [showWatchlist, setShowWatchlist] = useState(false)
  const [watchlistName, setWatchlistName] = useState('我的长期研究池')
  const [watchlistSymbols, setWatchlistSymbols] = useState('600519, 000858, 600036, 300750, 002594, 601318, 688981, 000333, 601088, 600900')
  const [message, setMessage] = useState('')
  const { data, isLoading, isError } = useQuery({ queryKey: ['stocks', search, group, exchange, universe, signal], queryFn: async () => (await api.get('/stocks', { params: { search, group, exchange, universe, signal, sort: 'score' } })).data })
  const { data: ownResearch } = useQuery({ queryKey: ['quant-v3-research'], queryFn: async () => (await api.get('/research/quant-v3-universe')).data })
  const { data: analytics } = useQuery({ queryKey: ['universe-analytics'], queryFn: async () => (await api.get('/analytics/universe')).data })
  const groups = useMemo<string[]>(() => [...new Set<string>((data?.items || []).map((item: any) => String(item.group)))], [data])
  const pageSize = 50
  const pageCount = Math.max(1, Math.ceil((data?.items?.length || 0) / pageSize))
  const visibleItems = useMemo(() => (data?.items || []).slice((page - 1) * pageSize, page * pageSize), [data, page])
  useEffect(() => setPage(1), [search, group, exchange, signal])
  const watchlistMutation = useMutation({
    mutationFn: async () => (await api.post('/watchlists', { name: watchlistName, symbols: watchlistSymbols.split(/[\s,，]+/).filter(Boolean) })).data,
    onSuccess: result => { setShowWatchlist(false); setMessage(`已创建 ${result.name} · ${result.count} 只股票`); queryClient.invalidateQueries({ queryKey: ['universes'] }) },
  })
  if (isLoading) return <LoadingState />
  if (isError) return <ErrorState />
  return <>
    <PageHeader eyebrow="UNIVERSE / 02" title="股票池" description="从 1,000+ 只覆盖多行业的 A 股与 OTC 扩展资产中，发现值得长期研究的机会。" actions={<><DataBadge mode={data?.data_mode} /><button className="secondary-button" onClick={() => setShowWatchlist(true)}><ListFilter size={16} />自定义股票池</button></>} />
    {ownResearch && (
      <Card>
        <PanelHeader
          title={`本组研究候选池 · ${ownResearch.total} 支跨行业股票`}
          subtitle={`LightGBM量化策略实际使用的候选池 · 研究区间 ${ownResearch.research_window.start} ~ ${ownResearch.research_window.end} · 覆盖 ${ownResearch.industries.length} 个行业`}
        />
        <p className="factor-copy" style={{ margin: '0 0 14px' }}>{ownResearch.strategy_summary}</p>
        <div className="table-wrap">
          <table className="wide-table">
            <thead><tr><th>代码</th><th>名称</th><th>行业</th><th>研究依据</th><th>风险点</th></tr></thead>
            <tbody>
              {ownResearch.items.map((row: any) => (
                <tr key={row.symbol}>
                  <td>{row.symbol}</td>
                  <td><b>{row.name}</b></td>
                  <td><span className="group-tag">{row.industry}</span></td>
                  <td>{row.thesis}</td>
                  <td>{row.risk}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    )}
    <div className="filter-bar"><div className="search-box"><Search size={17} /><input placeholder="搜索股票名称或代码" value={search} onChange={e => setSearch(e.target.value)} /></div><select value={universe} onChange={e => setUniverse(e.target.value)}><option value="all_assets">全部研究资产</option><option value="large_cap">大盘股核心池</option><option value="a_share">A股全市场</option><option value="pink_sheets">Pink Sheets</option></select><select value={exchange} onChange={e => setExchange(e.target.value)}><option value="">全部市场</option><option value="A股">A股</option><option value="OTC/Pink Sheets">OTC / Pink Sheets（Demo）</option></select><select value={group} onChange={e => setGroup(e.target.value)}><option value="">全部研究组</option>{groups.map(item => <option key={item}>{item}</option>)}</select><select value={signal} onChange={e => setSignal(e.target.value)}><option value="">全部信号</option><option>BUY</option><option>HOLD</option><option>SELL</option><option>WATCH</option></select><button className="icon-button"><SlidersHorizontal size={17} /></button></div>
    <div className="group-pills">{['全部', ...groups].map(item => <button className={group === (item === '全部' ? '' : item) ? 'group-pill active' : 'group-pill'} onClick={() => setGroup(item === '全部' ? '' : item)} key={item}>{item}<span>{item === '全部' ? data?.total : data?.items.filter((x: any) => x.group === item).length}</span></button>)}</div>
    <Card><PanelHeader title="研究池全景" subtitle={`${data?.total} 个标的 · 按综合评分排序 · 每页 ${pageSize} 条`} action={<span className="muted-label"><Filter size={14} />横截面评分</span>} /><div className="table-wrap"><table className="wide-table"><thead><tr><th>排名</th><th>股票</th><th>市场</th><th>研究组</th><th>板块 / 行业</th><th>综合评分</th><th>PE</th><th>营收增长</th><th>12M 动量</th><th>目标权重</th><th>信号</th></tr></thead><tbody>{visibleItems.map((row: any) => <tr key={row.symbol}><td className="rank">{String(row.rank).padStart(4, '0')}</td><td><Link className="stock-cell" to={`/stocks/${row.symbol}`}><div className="stock-avatar">{row.symbol.slice(-2)}</div><div><b>{row.name}</b><span>{row.symbol}</span></div></Link></td><td><span className={`group-tag ${row.exchange === 'OTC/Pink Sheets' ? 'pink-tag' : ''}`}>{row.exchange}</span></td><td><span className="group-tag">{row.group}</span></td><td>{row.sector} / {row.industry}</td><td><div className="score-cell"><b>{row.score.toFixed(1)}</b><div className="score-bar"><i style={{ width: `${row.score}%` }} /></div></div></td><td>{row.pe.toFixed(1)}</td><td className={row.revenue_growth >= 0 ? 'text-mint' : 'text-rose'}>{formatPercent(row.revenue_growth)}</td><td className={row.return_12m >= 0 ? 'text-mint' : 'text-rose'}>{formatPercent(row.return_12m)}</td><td>{formatPercent(row.target_weight)}</td><td><SignalBadge signal={row.action} /></td></tr>)}</tbody></table></div><div className="pagination"><span>第 {page} / {pageCount} 页 · 共 {data?.total} 只</span><div><button className="secondary-button" onClick={() => setPage(current => Math.max(1, current - 1))} disabled={page === 1}>上一页</button><button className="secondary-button" onClick={() => setPage(current => Math.min(pageCount, current + 1))} disabled={page === pageCount}>下一页</button></div></div></Card>
    {analytics && <div className="dashboard-grid analytics-grid"><Card className="span-4"><PanelHeader title="行业分布" subtitle="研究池标的数量" /><PieChart data={analytics.industry_distribution} /></Card><Card className="span-4"><PanelHeader title="评分分布" subtitle="横截面综合评分" /><BarChart data={analytics.score_distribution} category="range" value="count" percent={false} /></Card><Card className="span-4"><PanelHeader title="估值 × 成长" subtitle="气泡大小代表综合评分" /><ScatterChart data={analytics.valuation_growth} /></Card></div>}
    {showWatchlist && <div className="modal-backdrop" onClick={() => setShowWatchlist(false)}><div className="modal-card" onClick={event => event.stopPropagation()}><div className="modal-title"><div><span className="eyebrow">WATCHLIST</span><h2>创建自定义股票池</h2></div><button className="icon-button" onClick={() => setShowWatchlist(false)}><X size={16} /></button></div><label className="stack-label">股票池名称<input value={watchlistName} onChange={e => setWatchlistName(e.target.value)} /></label><label className="stack-label">股票代码<span className="field-hint">支持逗号、空格或换行分隔；至少需要 10 只有效股票，才能运行 Top N 低频策略</span><textarea value={watchlistSymbols} onChange={e => setWatchlistSymbols(e.target.value)} /></label><div className="modal-actions"><button className="secondary-button" onClick={() => setShowWatchlist(false)}>取消</button><button className="primary-button" onClick={() => watchlistMutation.mutate()} disabled={watchlistMutation.isPending}>{watchlistMutation.isPending ? '保存中…' : '保存股票池'}</button></div>{watchlistMutation.isError && <div className="form-error">股票代码无效、少于 10 只或名称已存在，请检查后重试。</div>}</div></div>}
    {message && <Toast message={message} />}
  </>
}
