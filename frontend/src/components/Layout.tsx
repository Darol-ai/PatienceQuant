import { ReactNode, useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { NavLink, useLocation } from 'react-router-dom'
import { Activity, BarChart3, Bot, BrainCircuit, BriefcaseBusiness, ChevronRight, Database, FlaskConical, LayoutDashboard, Menu, Moon, Search, Settings2, ShieldCheck, Sun, X } from 'lucide-react'
import { api } from '../api'

const navigation = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/stocks', label: '股票池', icon: Search },
  { to: '/strategy', label: '策略中心', icon: FlaskConical },
  { to: '/backtest', label: '回测中心', icon: BarChart3 },
  { to: '/paper', label: '模拟盘', icon: BriefcaseBusiness },
  { to: '/trading', label: '自动交易', icon: Bot },
  { to: '/ai', label: 'AI 投研', icon: BrainCircuit },
]

export function Layout({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const [light, setLight] = useState(false)
  const location = useLocation()
  useEffect(() => setOpen(false), [location.pathname])
  useEffect(() => { document.documentElement.dataset.theme = light ? 'light' : 'dark' }, [light])
  // 之前这里是写死的"Demo 数据就绪"文字，不管模拟盘实际绑定的是什么
  // 策略都不会变——现在跟着账户真正绑定的策略走，绑定我们验证过的
  // 真实策略(csi300_*/quant_v3_regression)时才说"真实策略数据"。
  const { data: account } = useQuery({ queryKey: ['paper-account'], queryFn: async () => (await api.get('/paper/account')).data })
  const isRealStrategy = account?.strategy_kind?.startsWith('csi300_') || account?.strategy_kind === 'quant_v3_regression'
  return (
    <div className="app-shell">
      <aside className={`sidebar ${open ? 'open' : ''}`}>
        <div className="brand"><div className="brand-mark"><Activity size={21} /></div><div><b>PatienceQuant</b><span>低频智能量化</span></div></div>
        <div className="workspace-card"><div className="dot-live"/><div><span>当前工作区</span><strong>A股低频多因子</strong></div><ChevronRight size={16}/></div>
        <nav>{navigation.map(({ to, label, icon: Icon }) => <NavLink key={to} to={to} end={to === '/'} className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}><Icon size={19}/><span>{label}</span></NavLink>)}</nav>
        <div className="sidebar-spacer" />
        <div className="system-state"><div><Database size={15}/><span>{isRealStrategy ? '真实策略数据' : 'Demo 数据就绪'}</span></div><div><ShieldCheck size={15}/><span>Paper Trading</span></div></div>
        <div className="user-card"><div className="avatar">PQ</div><div><strong>Quant Research</strong><span>MVP Workspace</span></div><Settings2 size={17}/></div>
      </aside>
      {open && <button className="sidebar-mask" onClick={() => setOpen(false)} aria-label="关闭菜单"/>}
      <main className="main-area">
        <header className="topbar">
          <button className="icon-button menu-button" onClick={() => setOpen(!open)}>{open ? <X/> : <Menu/>}</button>
          <div className="topbar-title"><span>Investment Intelligence</span><b>长期主义，从数据开始</b></div>
          <div className="topbar-actions"><div className="market-status"><i/>A 股数据已更新</div><button className="icon-button" onClick={() => setLight(!light)}>{light ? <Moon size={18}/> : <Sun size={18}/>}</button></div>
        </header>
        <div className="page-content">{children}</div>
      </main>
    </div>
  )
}
