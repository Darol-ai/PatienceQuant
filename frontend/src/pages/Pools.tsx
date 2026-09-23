import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Pencil, Plus, Save, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, formatPercent } from '../api'
import { PieChart, PoolCurveChart, PoolScatterChart } from '../components/Charts'
import { Card, LoadingState, PageHeader, PanelHeader, Toast } from '../components/UI'

// 股票池管理（ADR-0051）：固定股票池只读，自定义股票池可建、可改、可删（被策略或模拟盘用着时不能删）。
// 回测时在策略实践页下拉选择。
export function Pools() {
  const queryClient = useQueryClient()
  const { data: pools, isLoading } = useQuery({ queryKey: ['pools'], queryFn: async () => (await api.get('/pools')).data })
  const [selected, setSelected] = useState('csi300')
  const [editing, setEditing] = useState<{ id?: string; name: string; codes: string } | null>(null)
  const [message, setMessage] = useState('')
  const { data: members, isFetching } = useQuery({
    queryKey: ['pool-members', selected],
    queryFn: async () => (await api.get(`/pools/${selected}/members`)).data,
    enabled: !!selected && !editing,
  })
  const { data: notes } = useQuery({
    queryKey: ['broad30-notes'],
    queryFn: async () => (await api.get('/research/quant-v3-universe')).data,
    enabled: selected === 'broad30',
  })
  const { data: analysis, isFetching: analyzing } = useQuery({
    queryKey: ['pool-analysis', selected],
    queryFn: async () => (await api.get(`/pools/${selected}/analysis`)).data,
    enabled: !!selected && selected !== 'a_share' && !editing,
  })
  const [noteEdit, setNoteEdit] = useState<{ symbol: string; sector: string; thesis: string } | null>(null)
  const saveNote = useMutation({
    mutationFn: async () => (await api.put(`/pools/${selected}/notes/${noteEdit!.symbol}`, { sector: noteEdit!.sector, thesis: noteEdit!.thesis })).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pool-members', selected] })
      queryClient.invalidateQueries({ queryKey: ['pool-analysis', selected] })
      setNoteEdit(null)
    },
    onError: (error: any) => setMessage(error?.response?.data?.detail || '保存标注失败'),
  })
  const pool = (pools || []).find((p: any) => p.id === selected)
  const annotated = (members?.members || []).filter((m: any) => m.sector || m.thesis).length

  const save = useMutation({
    mutationFn: async () => {
      const symbols = editing!.codes.split(/[\s,，、;；]+/).map(s => s.trim().replace(/\.(SH|SZ)$/i, '')).filter(Boolean)
      const body = { name: editing!.name, symbols }
      return editing!.id ? (await api.put(`/pools/${editing!.id}`, body)).data : (await api.post('/pools', body)).data
    },
    onSuccess: data => {
      queryClient.invalidateQueries({ queryKey: ['pools'] })
      queryClient.invalidateQueries({ queryKey: ['pool-members', data.id] })
      setEditing(null)
      setSelected(data.id)
      setMessage(`已保存「${data.name}」，${data.count} 只股票`)
    },
    onError: (e: any) => setMessage(e?.response?.data?.detail || '保存失败'),
  })
  const remove = useMutation({
    mutationFn: async (id: string) => (await api.delete(`/pools/${id}`)).data,
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['pools'] }); setSelected('csi300'); setMessage('已删除') },
    onError: (e: any) => setMessage(e?.response?.data?.detail || '删除失败'),
  })
  useEffect(() => setMessage(''), [selected])

  if (isLoading) return <LoadingState />
  const notesBySymbol = Object.fromEntries((notes?.items || []).map((n: any) => [n.symbol.split('.')[0], n]))

  return <>
    <PageHeader eyebrow="POOLS" title="股票池" description="股票池是回测时选股的范围，不属于策略本身。固定股票池由系统提供；自定义股票池在这里管理，回测时在「策略实践」里选。"
      actions={<button className="primary-button" onClick={() => setEditing({ name: '', codes: '' })}><Plus size={15}/>新建自定义股票池</button>} />
    <div className="pools-layout">
      <Card>
        <PanelHeader title="全部股票池" />
        <div className="library-list">
          {['fixed', 'custom'].map(kind => {
            const rows = (pools || []).filter((p: any) => p.kind === kind)
            if (!rows.length) return null
            return <div key={kind}>
              <div className="library-group">{kind === 'fixed' ? '系统提供' : '自定义'} · {rows.length}</div>
              {rows.map((p: any) => <button key={p.id} className={`library-item ${p.id === selected && !editing ? 'active' : ''}`} onClick={() => { setEditing(null); setSelected(p.id) }}>
                <b>{p.name}</b><span>{p.count} 只 · {p.description}</span>
              </button>)}
            </div>
          })}
        </div>
      </Card>

      {editing ? <Card>
        <PanelHeader title={editing.id ? '修改自定义股票池' : '新建自定义股票池'} subtitle="至少 10 只；代码之间用空格、逗号或换行分隔，带不带 .SH/.SZ 都可以" />
        <div className="form-grid">
          <label className="full">名称<input value={editing.name} onChange={e => setEditing({ ...editing, name: e.target.value })}/></label>
          <label className="full">股票代码<textarea rows={10} value={editing.codes} onChange={e => setEditing({ ...editing, codes: e.target.value })} placeholder="600519 000333 600036 …"/></label>
        </div>
        <div className="page-actions" style={{ marginTop: 12 }}>
          <button className="secondary-button" onClick={() => setEditing(null)}>取消</button>
          <button className="primary-button" disabled={!editing.name.trim() || save.isPending} onClick={() => save.mutate()}><Save size={15}/>{save.isPending ? '保存中…' : '保存'}</button>
        </div>
      </Card> : <Card>
        <PanelHeader title={pool ? `${pool.name} · ${members?.count ?? pool.count} 只` : '股票池'} subtitle={pool?.description}
          action={pool?.kind === 'custom' ? <div className="page-actions">
            <button className="secondary-button" onClick={() => setEditing({ id: pool.id, name: pool.name, codes: (members?.members || []).map((m: any) => m.symbol).join(' ') })}>修改</button>
            <button className="secondary-button danger-button" disabled={remove.isPending} onClick={() => window.confirm(`删除「${pool.name}」？`) && remove.mutate(pool.id)}><Trash2 size={14}/>删除</button>
          </div> : undefined} />
        {selected === 'csi300' && <p className="muted-note">这里列的是今天的成分股；回测和模型训练在每个调仓日只用当时真实在指数里的股票（历史成分股，2010 年起）。</p>}
        {members?.annotatable && <p className="muted-note">研究标注：已标注 {annotated}/{members.count} 只。作业要求把研究的股票和板块标注清楚，点每行的铅笔填写"所属板块/主题"和"研究理由"。</p>}
        {isFetching ? <LoadingState text="正在加载成员…" /> : <div className="table-wrap"><table>
          <thead><tr><th>代码</th><th>名称</th><th>行业</th>{selected === 'broad30' && <><th>研究依据</th><th>风险</th></>}{members?.annotatable && <><th>所属板块/主题</th><th>研究理由</th><th></th></>}</tr></thead>
          <tbody>{(members?.members || []).map((m: any) => <tr key={m.symbol}>
            <td><Link className="stock-link-button" to={`/stocks/${m.symbol}`}>{m.symbol}</Link></td>
            <td>{m.name}</td>
            <td>{m.industry || notesBySymbol[m.symbol]?.industry || '—'}</td>
            {selected === 'broad30' && <><td className="wrap-cell">{notesBySymbol[m.symbol]?.thesis}</td><td className="wrap-cell">{notesBySymbol[m.symbol]?.risk}</td></>}
            {members?.annotatable && (noteEdit?.symbol === m.symbol ? <>
              <td><input className="note-input" maxLength={40} placeholder="如：白酒消费" value={noteEdit!.sector} onChange={e => setNoteEdit({ ...noteEdit!, sector: e.target.value })}/></td>
              <td><textarea className="note-input" maxLength={400} placeholder="为什么研究它" value={noteEdit!.thesis} onChange={e => setNoteEdit({ ...noteEdit!, thesis: e.target.value })}/></td>
              <td className="note-actions"><button className="primary-button" disabled={saveNote.isPending} onClick={() => saveNote.mutate()}><Save size={13}/>保存</button>
                <button className="secondary-button" onClick={() => setNoteEdit(null)}>取消</button></td>
            </> : <>
              <td>{m.sector || <span className="muted-note">未标注</span>}</td>
              <td className="wrap-cell">{m.thesis}</td>
              <td><button className="icon-button" title="编辑标注" onClick={() => setNoteEdit({ symbol: m.symbol, sector: m.sector || '', thesis: m.thesis || '' })}><Pencil size={13}/></button></td>
            </>)}
          </tr>)}</tbody>
        </table></div>}
      </Card>}
    </div>
    {!editing && selected !== 'a_share' && <Card>
      <PanelHeader title="股票池分析" subtitle={analysis ? `最近一年（${analysis.start} ~ ${analysis.end}），只看这些股票本身，与策略无关${selected === 'csi300' ? '；按今天的成分股名单' : ''}` : '最近一年'} />
      {analyzing || !analysis ? <LoadingState text="正在分析…" /> : <>
        <div className="account-info">
          <div><span>股票池等权组合</span><b className={analysis.summary.pool_return >= 0 ? 'text-mint' : 'text-rose'}>{formatPercent(analysis.summary.pool_return)}</b></div>
          <div><span>同期沪深300</span><b>{formatPercent(analysis.summary.benchmark_return)}</b></div>
          <div><span>成员收益中位数</span><b>{formatPercent(analysis.summary.median_return)}</b></div>
          <div><span>上涨的成员占比</span><b>{formatPercent(analysis.summary.up_ratio, 0)}</b></div>
        </div>
        <div className="pool-analysis-grid">
          <div><h4>按{analysis.grouped_by}分布</h4><PieChart data={analysis.groups} nameKey="name" valueKey="count"/></div>
          <div><h4>成员近一年收益与波动（{analysis.summary.analyzed} 只）</h4><PoolScatterChart points={analysis.points}/></div>
        </div>
        <h4>股票池等权组合 vs 沪深300</h4>
        <PoolCurveChart data={analysis.curve}/>
        <p className="muted-note">等权组合：每天对当天有行情的成员等权，不含交易成本，只用来看这批股票整体的走势。近一年交易不足 60 天的股票不画进散点图。</p>
      </>}
    </Card>}
    {message && <Toast message={message} type={save.isError || remove.isError ? 'error' : 'success'} />}
  </>
}
