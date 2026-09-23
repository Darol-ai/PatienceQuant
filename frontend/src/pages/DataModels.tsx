import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Boxes, Database, RefreshCw } from 'lucide-react'
import { Fragment, useState } from 'react'
import { api, formatPercent } from '../api'
import { frameworkText, isTraining, meanOf, ModelStatus, ModelYears } from '../components/ModelBits'
import { Card, LoadingState, PageHeader, PanelHeader } from '../components/UI'

const originText: Record<string, string> = { legacy: '旧模型（离线训练，原样导入）', trained: '系统内训练' }

// 数据与模型（ADR-0051）：本地行情库的状态与补齐；模型库。
export function DataModels() {
  const queryClient = useQueryClient()
  const { data: market, isLoading } = useQuery({
    queryKey: ['market-status'],
    queryFn: async () => (await api.get('/data/market/status')).data,
    refetchInterval: query => (query.state.data?.refresh?.running ? 3000 : 60000),
  })
  const [openModel, setOpenModel] = useState('')
  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: async () => (await api.get('/models')).data,
    refetchInterval: query => (isTraining(query.state.data) ? 3000 : false),
  })
  const { data: options } = useQuery({ queryKey: ['pipeline-options'], queryFn: async () => (await api.get('/pipeline/options')).data })
  const cancel = useMutation({
    mutationFn: async (id: string) => (await api.post(`/models/${id}/cancel`)).data,
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['models'] }),
  })
  const factorLabel = Object.fromEntries((options?.training_factors || []).map((f: any) => [f.key, f.label]))
  const refresh = useMutation({
    mutationFn: async () => (await api.post('/data/market/refresh')).data,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['market-status'] }),
  })
  if (isLoading) return <LoadingState />
  const running = market?.refresh?.running

  return <>
    <PageHeader eyebrow="DATA & MODELS" title="数据与模型" description="本地行情库是系统里行情数据的唯一来源：回测、模拟盘、打分、模型都只读它；数据源只在补齐时调用。" />
    <Card>
      <PanelHeader title="本地行情库" subtitle="A 股日线（2016 年起，含复权因子）与沪深300 指数（2005 年起）" action={<Database size={16}/>} />
      <div className="account-info">
        <div><span>行情截至</span><b>{market?.latest_stored || '—'}</b></div>
        <div><span>最近交易日</span><b>{market?.latest_trading_day || '—'}</b></div>
        <div><span>已存交易日</span><b>{market?.stored_days ?? '—'} 天</b></div>
        <div><span>缺少</span><b className={market?.missing_days ? 'text-rose' : 'text-mint'}>{market?.missing_days ? `${market.missing_days} 个交易日` : '无'}</b></div>
      </div>
      {running && <p className="muted-note">正在补齐 {market.refresh.done}/{market.refresh.total}：{market.refresh.current_day || '…'}</p>}
      {market?.refresh?.stopped_reason && <p className="muted-note warn">上次补齐中途停止：{market.refresh.stopped_reason}</p>}
      <button className="secondary-button" disabled={running || refresh.isPending || !market?.missing_days} onClick={() => refresh.mutate()}>
        <RefreshCw size={14}/>{running ? '补齐中…' : market?.missing_days ? '补齐到最近交易日' : '已是最新'}
      </button>
      <p className="muted-note">服务启动时会自动补齐；当天的行情通常收盘后才发布，发布前会显示缺 1 个交易日。</p>
    </Card>
    <Card>
      <PanelHeader title="模型库" subtitle="模型属于制定它的策略；训练设置完全相同时系统自动复用，不需要在这里挑选。点一行看逐年样本外成绩" action={<Boxes size={16}/>} />
      <div className="table-wrap"><table>
        <thead><tr><th>模型</th><th>状态</th><th>平均样本外 IC</th><th>平均前 10% 超额</th><th>训练股票池</th><th>预测</th><th>因子</th><th>可打分年份</th><th></th></tr></thead>
        <tbody>{(models || []).map((m: any) => {
          const ic = meanOf(m, 'ic'), excess = meanOf(m, 'top10_excess')
          return <Fragment key={m.id}>
            <tr className="clickable-row" onClick={() => setOpenModel(openModel === m.id ? '' : m.id)}>
              <td><b>{m.name}</b><div className="muted-note">{frameworkText[m.framework] || m.framework} · {originText[m.origin] || m.origin} · {m.data}</div></td>
              <td><ModelStatus model={m}/>{m.status === 'failed' && m.error && <div className="muted-note warn">{m.error}</div>}</td>
              <td className={ic == null ? '' : ic > 0 ? 'text-mint' : 'text-rose'}>{ic == null ? '—' : ic.toFixed(4)}</td>
              <td className={excess == null ? '' : excess > 0 ? 'text-mint' : 'text-rose'}>{excess == null ? '—' : formatPercent(excess)}</td>
              <td>{m.trained_on}</td>
              <td>{m.horizon_days} 日</td>
              <td title={m.factors.map((k: string) => factorLabel[k] || k).join('、')}>{m.factors.length} 个{m.config?.cs_rank ? '（池内百分位）' : ''}</td>
              <td>{m.years.length ? `${m.years[0]}–${m.years[m.years.length - 1]}` : '—'}</td>
              <td>{(m.status === 'queued' || m.status === 'training') && <button className="secondary-button" disabled={cancel.isPending}
                onClick={e => { e.stopPropagation(); if (window.confirm(`取消训练「${m.name}」？`)) cancel.mutate(m.id) }}>取消</button>}</td>
            </tr>
            {openModel === m.id && <tr><td colSpan={9}>
              <p className="muted-note">输入因子：{m.factors.map((k: string) => factorLabel[k] || k).join('、')}</p>
              {m.description && <p className="muted-note">{m.description}</p>}
              <ModelYears model={m}/>
            </td></tr>}
          </Fragment>
        })}</tbody>
      </table></div>
      <p className="muted-note">样本外 IC：每天模型分数与之后实际收益的秩相关，按天平均；前 10% 超额：每天分数最高的 10% 股票相对全池平均的未来收益。两者都只用该年模型没见过的数据。新模型在「策略制定」里选「模型预测」后填训练设置生成。</p>
    </Card>
  </>
}
