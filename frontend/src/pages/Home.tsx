import { useQuery } from '@tanstack/react-query'
import { Database, Trophy, WalletCards } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api, formatMoney, formatPercent } from '../api'
import { AgentChat } from '../components/AgentChat'
import { Card, PageHeader, PanelHeader } from '../components/UI'

// 首页（ADR-0051）：主体是智能体对话；右侧窄栏说明"这是一个有策略库、有账户、有数据的系统"。
export function Home() {
  const { data: cards } = useQuery({ queryKey: ['scorecards'], queryFn: async () => (await api.get('/scorecards')).data })
  const { data: account } = useQuery({ queryKey: ['paper-account'], queryFn: async () => (await api.get('/paper/account')).data })
  const { data: market } = useQuery({ queryKey: ['market-status'], queryFn: async () => (await api.get('/data/market/status')).data })
  const ready = (cards?.cards || []).filter((c: any) => c.status === 'ready')
  const top = [...ready].sort((a, b) => b.metrics.sharpe - a.metrics.sharpe).slice(0, 3)

  return <>
    <PageHeader eyebrow="PATIENCEQUANT" title="挑一个适合你的策略" description="回答几个问题，助手会从策略库里按成绩卡推荐一个策略；你可以追问、比较，再去回测验证。" />
    <div className="home-grid">
      <Card className="home-chat"><AgentChat /></Card>
      <div className="home-side">
        <Card>
          <PanelHeader title="成绩卡前三" subtitle={`按夏普排序 · 共 ${ready.length} 个策略有成绩卡`} action={<Trophy size={16}/>} />
          <div className="side-list">
            {top.map((c: any, i: number) => <Link key={c.strategy_id} to={`/library/${c.strategy_id}`} className="side-item">
              <b>{i + 1}. {c.name}</b>
              <span>年化 {formatPercent(c.metrics.annual_return)} · 回撤 {formatPercent(c.metrics.max_drawdown)} · 夏普 {Number(c.metrics.sharpe).toFixed(2)}</span>
            </Link>)}
            {!top.length && <p className="muted-note">还没有成绩卡，去策略库计算。</p>}
          </div>
          <Link className="text-link" to="/library">查看全部策略 →</Link>
        </Card>
        <Card>
          <PanelHeader title="模拟盘" subtitle="当前绑定的策略" action={<WalletCards size={16}/>} />
          {account ? <div className="side-list">
            <div className="side-item static"><b>{account.strategy_name || '还没有绑定策略'}</b>
              <span>总资产 {formatMoney(account.total_assets)} · 累计 <em className={account.cumulative_return >= 0 ? 'text-mint' : 'text-rose'}>{formatPercent(account.cumulative_return)}</em></span></div>
          </div> : <p className="muted-note">加载中…</p>}
          <Link className="text-link" to="/paper">进入模拟盘 →</Link>
        </Card>
        <Card>
          <PanelHeader title="本地行情库" subtitle="所有行情的唯一来源" action={<Database size={16}/>} />
          {market && <div className="side-list"><div className="side-item static">
            <b>A 股行情截至 {market.latest_stored || '—'}</b>
            <span>{market.missing_days ? `落后最近交易日 ${market.missing_days} 天` : '已是最近交易日'} · 共 {market.stored_days} 个交易日</span>
          </div></div>}
          <Link className="text-link" to="/data">数据与模型 →</Link>
        </Card>
      </div>
    </div>
  </>
}
