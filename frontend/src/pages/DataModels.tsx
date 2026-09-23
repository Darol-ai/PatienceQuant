import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Boxes, Database, RefreshCw } from 'lucide-react'
import { api } from '../api'
import { Card, LoadingState, PageHeader, PanelHeader } from '../components/UI'

const originText: Record<string, string> = { legacy: '旧模型（离线训练，原样导入）', trained: '系统内训练' }
const frameworkText: Record<string, string> = { lightgbm: 'LightGBM', xgboost: 'XGBoost', lstm: 'LSTM', transformer: 'Transformer' }

// 数据与模型（ADR-0051）：本地行情库的状态与补齐；模型库。
export function DataModels() {
  const queryClient = useQueryClient()
  const { data: market, isLoading } = useQuery({
    queryKey: ['market-status'],
    queryFn: async () => (await api.get('/data/market/status')).data,
    refetchInterval: query => (query.state.data?.refresh?.running ? 3000 : 60000),
  })
  const { data: options } = useQuery({ queryKey: ['pipeline-options'], queryFn: async () => (await api.get('/pipeline/options')).data })
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
      <PanelHeader title="模型库" subtitle="模型选股策略只能引用这里的模型；每个模型逐年滚动训练，只能给有训练结果的年份打分" action={<Boxes size={16}/>} />
      <div className="table-wrap"><table>
        <thead><tr><th>模型</th><th>类型</th><th>来源</th><th>训练股票池</th><th>预测周期</th><th>可打分年份</th><th>说明</th></tr></thead>
        <tbody>{(options?.models || []).map((m: any) => <tr key={m.id}>
          <td><b>{m.name}</b></td>
          <td>{frameworkText[m.framework] || m.framework}</td>
          <td>{originText[m.origin] || m.origin}</td>
          <td>{m.trained_on}</td>
          <td>{m.horizon_days} 个交易日</td>
          <td>{m.years.length ? `${m.years[0]}–${m.years[m.years.length - 1]}` : '—'}</td>
          <td className="wrap-cell">{m.description}</td>
        </tr>)}</tbody>
      </table></div>
      <p className="muted-note">系统内训练（LightGBM / XGBoost / LSTM / Transformer）暂未开放。</p>
    </Card>
  </>
}
