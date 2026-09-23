import { ReactNode, useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { NavLink, useLocation } from 'react-router-dom'
import { Activity, BarChart3, Bot, BrainCircuit, BriefcaseBusiness, Database, Layers, ListChecks, Menu, Moon, Sun, X } from 'lucide-react'
import { api } from '../api'

// 导航结构见 docs/adr/0051
const navigation = [
  { to: '/', label: '首页', icon: Bot },
  { to: '/library', label: '策略库', icon: Layers },
  { to: '/backtest', label: '策略实践', icon: BarChart3 },
  { to: '/paper', label: '模拟盘', icon: BriefcaseBusiness },
  { to: '/pools', label: '股票池', icon: ListChecks },
  { to: '/data', label: '数据与模型', icon: Database },
  { to: '/ai', label: 'AI 工具', icon: BrainCircuit },
]

export function Layout({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)
  // 默认浅色（之前默认深色）；记住用户切换过的选择，不然每次刷新都要
  // 重新点一次太阳图标。
  const [light, setLight] = useState(() => {
    try {
      const saved = localStorage.getItem('theme')
      return saved ? saved === 'light' : true
    } catch {
      return true
    }
  })
  const location = useLocation()
  useEffect(() => setOpen(false), [location.pathname])
  useEffect(() => {
    document.documentElement.dataset.theme = light ? 'light' : 'dark'
    try {
      localStorage.setItem('theme', light ? 'light' : 'dark')
    } catch {
      // localStorage被禁用——主题这次会话内还是正常切换，只是刷新后
      // 回到默认浅色，不是值得中断操作的错误。
    }
  }, [light])
  const { data: account } = useQuery({ queryKey: ['paper-account'], queryFn: async () => (await api.get('/paper/account')).data })
  // 本地行情库状态（ADR-0047）：补齐进行中每3秒刷新一次进度，平时每分钟查一次。
  const queryClient = useQueryClient()
  const { data: market } = useQuery({
    queryKey: ['market-status'],
    queryFn: async () => (await api.get('/data/market/status')).data,
    refetchInterval: query => (query.state.data?.refresh?.running ? 3000 : 60000),
  })
  const marketRefresh = useMutation({
    mutationFn: async () => (await api.post('/data/market/refresh')).data,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['market-status'] }),
  })
  const refreshing = market?.refresh?.running || marketRefresh.isPending
  const marketLabel = market?.error
    ? '行情库状态不可用'
    : refreshing
      ? `行情补齐中 ${market?.refresh?.done ?? 0}/${market?.refresh?.total ?? '…'}`
      : market?.latest_stored
        ? `A 股行情截至 ${market.latest_stored}${market.missing_days ? ` · 缺 ${market.missing_days} 个交易日` : ''}`
        : 'A 股行情库为空'
  const marketTitle = market?.refresh?.stopped_reason
    ? `上次补齐中途停止：${market.refresh.stopped_reason}`
    : '当天行情通常在收盘后才发布，发布前会显示缺 1 个交易日'
  return (
    <div className="app-shell">
      <aside className={`sidebar ${open ? 'open' : ''}`}>
        <div className="brand"><div className="brand-mark"><Activity size={21} /></div><div><b>PatienceQuant</b><span>策略库 · 回测 · 模拟盘</span></div></div>
        <nav>{navigation.map(({ to, label, icon: Icon }) => <NavLink key={to} to={to} end={to === '/'} className={({isActive}) => isActive || (to === '/library' && location.pathname.startsWith('/strategy/')) || (to === '/backtest' && location.pathname.startsWith('/backtests/')) ? 'nav-item active' : 'nav-item'}><Icon size={19}/><span>{label}</span></NavLink>)}</nav>
        <div className="sidebar-spacer" />
        <div className="system-state"><div><Database size={15}/><span>{market?.latest_stored ? `行情截至 ${market.latest_stored}` : '行情库状态未知'}</span></div><div><BriefcaseBusiness size={15}/><span>{account?.strategy_name ? `模拟盘：${account.strategy_name}` : '模拟盘未绑定策略'}</span></div></div>
        <div className="sidebar-foot">课程演示 · 不连接券商、不用真钱</div>
      </aside>
      {open && <button className="sidebar-mask" onClick={() => setOpen(false)} aria-label="关闭菜单"/>}
      <main className="main-area">
        <header className="topbar">
          <button className="icon-button menu-button" onClick={() => setOpen(!open)}>{open ? <X/> : <Menu/>}</button>
          <div className="topbar-title"><span>A 股低频量化</span><b>按成绩卡选策略，先回测再上模拟盘</b></div>
          <div className="topbar-actions"><div className="market-status" title={marketTitle}><i/>{marketLabel}{!refreshing && market?.missing_days ? <button className="text-link" style={{ background: 'none', border: 0, padding: 0, marginLeft: 6 }} onClick={() => marketRefresh.mutate()}>补齐</button> : null}</div><button className="icon-button" onClick={() => setLight(!light)}>{light ? <Moon size={18}/> : <Sun size={18}/>}</button></div>
        </header>
        <div className="page-content">{children}</div>
      </main>
    </div>
  )
}
