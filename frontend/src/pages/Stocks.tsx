import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ListFilter, X } from 'lucide-react'
import { useState } from 'react'
import { api } from '../api'
import { BarChart, PieChart, ScatterChart } from '../components/Charts'
import { Card, PageHeader, PanelHeader, Toast } from '../components/UI'

export function Stocks() {
  const queryClient = useQueryClient()
  const [showWatchlist, setShowWatchlist] = useState(false)
  const [watchlistName, setWatchlistName] = useState('我的长期研究池')
  const [watchlistSymbols, setWatchlistSymbols] = useState('600519, 000858, 600036, 300750, 002594, 601318, 688981, 000333, 601088, 600900')
  const [message, setMessage] = useState('')
  const { data: ownResearch } = useQuery({ queryKey: ['quant-v3-research'], queryFn: async () => (await api.get('/research/quant-v3-universe')).data })
  const { data: csi300Research } = useQuery({ queryKey: ['csi300-research'], queryFn: async () => (await api.get('/research/csi300-universe')).data })
  const { data: analytics } = useQuery({ queryKey: ['universe-analytics'], queryFn: async () => (await api.get('/analytics/universe')).data })
  const watchlistMutation = useMutation({
    mutationFn: async () => (await api.post('/watchlists', { name: watchlistName, symbols: watchlistSymbols.split(/[\s,，]+/).filter(Boolean) })).data,
    onSuccess: result => { setShowWatchlist(false); setMessage(`已创建 ${result.name} · ${result.count} 只股票`); queryClient.invalidateQueries({ queryKey: ['universes'] }) },
  })
  return <>
    <PageHeader eyebrow="UNIVERSE / 02" title="股票池" description="策略实际使用的可交易范围（universe）——不是随便浏览的股票列表。" actions={<button className="secondary-button" onClick={() => setShowWatchlist(true)}><ListFilter size={16} />自定义股票池</button>} />
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
    {csi300Research && (
      <Card>
        <PanelHeader
          title={`沪深300候选池 · 全部 ${csi300Research.total} 支真实成分股`}
          subtitle={`真实价格与LightGBM策略最新真实打分排名 · 截至 ${csi300Research.as_of} · 下表展示当前排名前30(策略实际持仓)`}
        />
        <p className="factor-copy" style={{ margin: '0 0 14px' }}>{csi300Research.strategy_summary}</p>
        <div className="table-wrap">
          <table className="wide-table">
            <thead><tr><th>排名</th><th>代码</th><th>名称</th><th>最新收盘价</th><th>LightGBM分数</th></tr></thead>
            <tbody>
              {csi300Research.items.slice(0, 30).map((row: any) => (
                <tr key={row.symbol}>
                  <td className="rank">{String(row.lightgbm_rank).padStart(3, '0')}</td>
                  <td>{row.symbol}</td>
                  <td><b>{row.name}</b></td>
                  <td>{row.latest_price != null ? `¥${row.latest_price.toFixed(2)}` : '—'}</td>
                  <td>{row.lightgbm_score != null ? row.lightgbm_score.toFixed(4) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    )}
    {analytics && <div className="dashboard-grid analytics-grid"><Card className="span-4"><PanelHeader title="行业分布" subtitle="研究池标的数量" /><PieChart data={analytics.industry_distribution} /></Card><Card className="span-4"><PanelHeader title="评分分布" subtitle="横截面综合评分" /><BarChart data={analytics.score_distribution} category="range" value="count" percent={false} /></Card><Card className="span-4"><PanelHeader title="估值 × 成长" subtitle="气泡大小代表综合评分" />{analytics.valuation_growth?.length ? <ScatterChart data={analytics.valuation_growth} /> : <div className="empty-chart">{analytics.valuation_growth_note || '暂无数据'}</div>}</Card></div>}
    {showWatchlist && <div className="modal-backdrop" onClick={() => setShowWatchlist(false)}><div className="modal-card" onClick={event => event.stopPropagation()}><div className="modal-title"><div><span className="eyebrow">WATCHLIST</span><h2>创建自定义股票池</h2></div><button className="icon-button" onClick={() => setShowWatchlist(false)}><X size={16} /></button></div><label className="stack-label">股票池名称<input value={watchlistName} onChange={e => setWatchlistName(e.target.value)} /></label><label className="stack-label">股票代码<span className="field-hint">支持逗号、空格或换行分隔；至少需要 10 只有效股票，才能运行 Top N 低频策略</span><textarea value={watchlistSymbols} onChange={e => setWatchlistSymbols(e.target.value)} /></label><div className="modal-actions"><button className="secondary-button" onClick={() => setShowWatchlist(false)}>取消</button><button className="primary-button" onClick={() => watchlistMutation.mutate()} disabled={watchlistMutation.isPending}>{watchlistMutation.isPending ? '保存中…' : '保存股票池'}</button></div>{watchlistMutation.isError && <div className="form-error">股票代码无效、少于 10 只或名称已存在，请检查后重试。</div>}</div></div>}
    {message && <Toast message={message} />}
  </>
}
