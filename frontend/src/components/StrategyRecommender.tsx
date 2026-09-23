import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Bot, RefreshCw, Sparkles } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api, formatPercent } from '../api'
import { Card, PanelHeader } from './UI'

// 智能体推荐（ADR-0047 第 8 条）：先问偏好，再按成绩卡上的事实筛选排序，只推荐一个策略；
// 大模型只负责解释，不能下单。

const drawdownChoices = [
  { value: 0.15, label: '最多亏 15%' },
  { value: 0.25, label: '最多亏 25%' },
  { value: 0.35, label: '最多亏 35%' },
  { value: null, label: '不在乎回撤' },
]
const holdingChoices = [
  { value: 'short', label: '短：每周调整' },
  { value: 'medium', label: '中：每月调整' },
  { value: 'long', label: '长：每季度调整' },
  { value: 'any', label: '都可以' },
]

function Choice({ active, onClick, children }: { active: boolean; onClick: () => void; children: string }) {
  return <button className={`group-pill ${active ? 'active' : ''}`} onClick={onClick}>{children}</button>
}

export function StrategyRecommender() {
  const queryClient = useQueryClient()
  const [drawdown, setDrawdown] = useState<number | null | undefined>(undefined)
  const [holding, setHolding] = useState<string | undefined>(undefined)
  const [question, setQuestion] = useState('')
  const { data: cards } = useQuery({
    queryKey: ['scorecards'],
    queryFn: async () => (await api.get('/scorecards')).data,
    refetchInterval: query => (query.state.data?.refresh?.running ? 4000 : false),
  })
  const refresh = useMutation({
    mutationFn: async () => (await api.post('/scorecards/refresh', {})).data,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['scorecards'] }),
  })
  const recommend = useMutation({
    mutationFn: async () => (await api.post('/ai/recommend', { market: 'a_share', max_drawdown: drawdown, holding, question }, { timeout: 120_000 })).data,
  })
  const ready = (cards?.cards || []).filter((c: any) => c.status === 'ready').length
  const total = cards?.cards?.length || 0
  const running = cards?.refresh?.running
  const result = recommend.data
  const chosen = result?.recommendation

  return <Card>
    <PanelHeader title="策略推荐" subtitle="先回答三个问题；按成绩卡上的事实筛选排序，只推荐一个策略，不会替你下单" action={<Bot size={16}/>} />
    <div className="recommend-status">
      <span>成绩卡 {ready}/{total}（标准条件：{cards?.standard?.start_date} ~ {cards?.standard?.end_date}，100 万，各策略默认股票池）</span>
      {running
        ? <b>正在计算 {cards.refresh.done}/{cards.refresh.total}：{cards.refresh.current || '…'}</b>
        : ready < total && <button className="secondary-button" onClick={() => refresh.mutate()} disabled={refresh.isPending}><RefreshCw size={13}/>计算缺少的成绩卡</button>}
    </div>
    {cards?.refresh?.errors?.length > 0 && <p className="muted-note warn">有 {cards.refresh.errors.length} 个策略没算出来：{cards.refresh.errors.map((e: any) => `${e.name}（${e.error}）`).join('；')}</p>}

    <div className="recommend-questions">
      <div><b>1. 市场</b><div className="group-pills"><Choice active onClick={() => undefined}>A股</Choice><span className="muted-note">港股等数据源就绪后再开放</span></div></div>
      <div><b>2. 能接受的最大回撤</b><div className="group-pills">{drawdownChoices.map(c => <Choice key={String(c.value)} active={drawdown === c.value} onClick={() => setDrawdown(c.value)}>{c.label}</Choice>)}</div></div>
      <div><b>3. 打算多久调整一次持仓</b><div className="group-pills">{holdingChoices.map(c => <Choice key={c.value} active={holding === c.value} onClick={() => setHolding(c.value)}>{c.label}</Choice>)}</div></div>
      <label className="stack-label">还想补充什么（可选）<textarea className="strategy-assist-input" rows={2} value={question} onChange={e => setQuestion(e.target.value)} placeholder="比如：更在意别亏太多"/></label>
    </div>
    <button className="primary-button full-button" disabled={drawdown === undefined || !holding || recommend.isPending || ready === 0} onClick={() => recommend.mutate()}>
      <Sparkles size={16}/>{recommend.isPending ? '正在挑选…' : ready === 0 ? '先计算成绩卡' : '帮我挑一个策略'}
    </button>

    {recommend.isError && <p className="muted-note warn">推荐失败，请稍后重试。</p>}
    {result && !chosen && <div className="ai-result"><p>{result.notes?.join('；')}</p></div>}
    {chosen && <div className="ai-result">
      <div className="recommend-pick">
        <div><span>推荐</span><b>{chosen.name}</b><small>{chosen.description}</small></div>
        <div className="recommend-metrics">
          <div><span>年化</span><b>{formatPercent(chosen.metrics.annual_return)}</b></div>
          <div><span>最大回撤</span><b className="text-rose">{formatPercent(chosen.metrics.max_drawdown)}</b></div>
          <div><span>夏普</span><b>{Number(chosen.metrics.sharpe).toFixed(2)}</b></div>
          <div><span>跑赢沪深300</span><b>{chosen.years_beating_benchmark}/{chosen.years} 年</b></div>
          <div><span>平均仓位</span><b>{formatPercent(chosen.average_exposure ?? 1)}</b></div>
        </div>
      </div>
      <p className="recommend-text">{String(result.explanation.content).replace(/\*\*/g, '')}</p>
      {result.explanation.llm_error && <p className="muted-note warn">{result.explanation.llm_error}</p>}
      <div className="risk-box">
        <b>怎么选出来的</b>
        {result.rules.map((r: string) => <span key={r}>• {r}</span>)}
        {result.notes.map((n: string) => <span key={n}>• {n}</span>)}
      </div>
      {result.alternatives?.length > 0 && <div className="risk-box">
        <b>备选</b>
        {result.alternatives.map((c: any) => <span key={c.strategy_id}>• {c.name}：年化 {formatPercent(c.metrics.annual_return)}，最大回撤 {formatPercent(c.metrics.max_drawdown)}，夏普 {Number(c.metrics.sharpe).toFixed(2)}</span>)}
      </div>}
      <div className="provider-line"><span>解释来源</span><b>{result.explanation.provider} · {result.explanation.model_version}</b></div>
      <Link className="secondary-button full-button" to={result.next_step.practice_url}>去策略实践回测这个策略</Link>
      <p className="muted-note">成绩卡是历史回测结果，不代表未来收益。智能体不会下单，是否用到模拟盘由你决定。</p>
    </div>}
  </Card>
}
