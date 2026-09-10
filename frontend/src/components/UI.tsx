import { ReactNode } from 'react'
import { AlertCircle, ArrowDownRight, ArrowUpRight, DatabaseZap, LoaderCircle } from 'lucide-react'

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow: string; title: ReactNode; description: string; actions?: ReactNode }) {
  return <div className="page-header"><div><span className="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{description}</p></div>{actions && <div className="page-actions">{actions}</div>}</div>
}

export function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}>{children}</section>
}

export function PanelHeader({ title, subtitle, action }: { title: string; subtitle?: string; action?: ReactNode }) {
  return <div className="panel-header"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>
}

export function MetricCard({ label, value, change, icon, tone = 'neutral' }: { label: string; value: string; change?: string; icon?: ReactNode; tone?: 'positive'|'negative'|'neutral' }) {
  const positive = tone === 'positive'
  return <div className="metric-card"><div className="metric-top"><span>{label}</span><div className="metric-icon">{icon}</div></div><strong>{value}</strong>{change && <div className={`metric-change ${tone}`}>{positive ? <ArrowUpRight size={14}/> : tone === 'negative' ? <ArrowDownRight size={14}/> : null}{change}</div>}</div>
}

export function DataBadge({ mode = 'demo' }: { mode?: string }) {
  const labels: Record<string, string> = {
    real: 'REAL DATA',
    real_or_demo_fallback: 'REAL → DEMO FALLBACK',
    mixed: 'REAL + DEMO',
    demo_fallback: 'DEMO FALLBACK',
    demo_mixed: 'DEMO + FALLBACK',
    demo: 'DEMO MODE',
  }
  const real = mode === 'real'
  return <span className={`data-badge ${real ? 'real' : mode !== 'demo' ? 'mixed' : ''}`}><DatabaseZap size={13}/>{labels[mode] || mode.toUpperCase()}</span>
}

export function SignalBadge({ signal }: { signal: string }) {
  return <span className={`signal-badge ${signal.toLowerCase()}`}>{signal}</span>
}

export function LoadingState({ text = '正在加载量化数据…' }: { text?: string }) {
  return <div className="state-box"><LoaderCircle className="spin"/><span>{text}</span></div>
}

export function ErrorState({ text = '数据加载失败，请稍后重试。' }: { text?: string }) {
  return <div className="state-box error"><AlertCircle/><span>{text}</span></div>
}

export function Toast({ message, type = 'success' }: { message: string; type?: 'success'|'error' }) {
  return <div className={`toast ${type}`}>{message}</div>
}
