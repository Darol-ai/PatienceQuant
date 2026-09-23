import { useMutation, useQuery } from '@tanstack/react-query'
import { Bot, RotateCcw, Send } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import { CardCompareTable, CardMetrics } from './ScorecardBits'

// 智能体（ADR-0051）：引导式对话。先依次问偏好（都有快捷按钮），问齐后按成绩卡推荐单个策略；
// 之后可以追问：改偏好重新推荐 / 为什么不是某个策略 / 比较两个策略 / 解释名词。
// 选哪个策略由后端规则决定，大模型只负责翻译打字的问题和写解释。对话只保存在当前浏览器。

type Prefs = { market: string; max_drawdown?: number | null; holding?: string }
type Message = { role: 'agent' | 'user'; text: string; reply?: any; chips?: Chip[] }
type Chip = { label: string; kind: 'drawdown' | 'holding' | 'action'; value: any }

const STORAGE_KEY = 'agentChat.v1'
const drawdownChips: Chip[] = [
  { label: '最多亏 15%', kind: 'drawdown', value: 0.15 },
  { label: '最多亏 25%', kind: 'drawdown', value: 0.25 },
  { label: '最多亏 35%', kind: 'drawdown', value: 0.35 },
  { label: '不在乎回撤', kind: 'drawdown', value: null },
]
const holdingChips: Chip[] = [
  { label: '每周调整', kind: 'holding', value: 'short' },
  { label: '每月调整', kind: 'holding', value: 'medium' },
  { label: '每季度调整', kind: 'holding', value: 'long' },
  { label: '都可以', kind: 'holding', value: 'any' },
]
const greeting: Message = {
  role: 'agent',
  text: '你好，我会按你的偏好，从策略库里挑一个最合适的策略，并用它的成绩卡说明理由。我不会替你下单。\n目前只支持 A 股。先说说：你能接受的最大亏损是多少？',
  chips: drawdownChips,
}

function load(): { messages: Message[]; prefs: Prefs } {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) return JSON.parse(saved)
  } catch {
    // 读不到就从头开始
  }
  return { messages: [greeting], prefs: { market: 'a_share' } }
}

function followUpChips(reply: any): Chip[] {
  const chips: Chip[] = []
  for (const alt of reply?.alternatives || []) {
    chips.push({ label: `为什么不是「${alt.name}」？`, kind: 'action', value: { type: 'why_not', strategy_id: alt.strategy_id } })
  }
  if (reply?.recommendation && reply?.alternatives?.[0]) {
    chips.push({ label: '和第一个备选比较', kind: 'action', value: { type: 'compare', strategy_ids: [reply.recommendation.strategy_id, reply.alternatives[0].strategy_id] } })
  }
  chips.push({ label: '换个回撤要求', kind: 'action', value: { type: 'ask_drawdown' } })
  chips.push({ label: '换个调仓频率', kind: 'action', value: { type: 'ask_holding' } })
  chips.push({ label: '夏普比率是什么？', kind: 'action', value: { type: 'explain_term', term: '夏普比率' } })
  chips.push({ label: '最大回撤是什么？', kind: 'action', value: { type: 'explain_term', term: '最大回撤' } })
  return chips
}

export function AgentChat() {
  const initial = load()
  const [messages, setMessages] = useState<Message[]>(initial.messages)
  const [prefs, setPrefs] = useState<Prefs>(initial.prefs)
  const [text, setText] = useState('')
  const [whyNotId, setWhyNotId] = useState('')
  const bottom = useRef<HTMLDivElement>(null)
  const { data: cards } = useQuery({ queryKey: ['scorecards'], queryFn: async () => (await api.get('/scorecards')).data })

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ messages, prefs }))
    } catch {
      // 存不下也不影响当前对话
    }
    bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [messages, prefs])

  const turn = useMutation({
    mutationFn: async (body: any) => (await api.post('/ai/agent', body, { timeout: 150_000 })).data,
    onSuccess: data => {
      setPrefs(data.prefs)
      const reply = data.reply
      setMessages(m => {
        // 追问的回答下面继续给出最近一次推荐的追问按钮，不然问过一次按钮就没了
        const lastRecommend = [...m].reverse().find(msg => msg.reply?.type === 'recommend' && msg.reply.recommendation)?.reply
        const source = reply.type === 'recommend' && reply.recommendation ? reply : lastRecommend
        return [...m, { role: 'agent', text: reply.text, reply, chips: source ? followUpChips(source) : undefined }]
      })
    },
    onError: (error: any) => setMessages(m => [...m, { role: 'agent', text: error?.response?.data?.detail || '出错了，请稍后再试。' }]),
  })

  const complete = prefs.max_drawdown !== undefined && !!prefs.holding
  const say = (userText: string) => setMessages(m => [...m, { role: 'user', text: userText }])

  function onChip(chip: Chip) {
    say(chip.label)
    if (chip.kind === 'drawdown') {
      const next = { ...prefs, max_drawdown: chip.value }
      setPrefs(next)
      if (next.holding) turn.mutate({ prefs: next, action: { type: 'recommend' } })
      else setMessages(m => [...m, { role: 'agent', text: '好的。你打算多久调整一次持仓？', chips: holdingChips }])
    } else if (chip.kind === 'holding') {
      const next = { ...prefs, holding: chip.value }
      setPrefs(next)
      turn.mutate({ prefs: next, action: { type: 'recommend' } })
    } else if (chip.value.type === 'ask_drawdown') {
      setMessages(m => [...m, { role: 'agent', text: '能接受的最大亏损改成多少？', chips: drawdownChips }])
    } else if (chip.value.type === 'ask_holding') {
      setMessages(m => [...m, { role: 'agent', text: '调仓频率改成？', chips: holdingChips }])
    } else {
      turn.mutate({ prefs, action: chip.value })
    }
  }

  function send() {
    const value = text.trim()
    if (!value) return
    say(value)
    setText('')
    turn.mutate({ prefs: { market: 'a_share', max_drawdown: prefs.max_drawdown ?? null, holding: prefs.holding || 'any' }, text: value })
  }

  function reset() {
    setMessages([greeting])
    setPrefs({ market: 'a_share' })
  }

  const lastIndex = messages.length - 1
  const library = (cards?.cards || []).filter((c: any) => c.origin !== 'paper_snapshot')
  return <div className="agent-chat">
    <div className="agent-head">
      <div><Bot size={18}/><b>策略推荐助手</b><span>按成绩卡上的事实推荐，不下单</span></div>
      <button className="secondary-button" onClick={reset}><RotateCcw size={13}/>重新开始</button>
    </div>
    <div className="agent-messages">
      {messages.map((message, index) => <div key={index} className={`agent-msg ${message.role}`}>
        <div className="agent-bubble">
          {message.reply?.type === 'recommend' && message.reply.recommendation ? <RecommendReply reply={message.reply}/> :
            <p>{String(message.text || '').replace(/\*\*/g, '')}</p>}
          {message.reply?.cards?.length > 0 && <CardCompareTable cards={message.reply.cards}/>}
        </div>
        {message.chips && index === lastIndex && !turn.isPending && <div className="agent-chips">
          {message.chips.map(chip => <button key={chip.label} className="group-pill" onClick={() => onChip(chip)}>{chip.label}</button>)}
        </div>}
      </div>)}
      {turn.isPending && <div className="agent-msg agent"><div className="agent-bubble"><p className="muted-label">正在查成绩卡…</p></div></div>}
      <div ref={bottom}/>
    </div>
    {complete && <div className="agent-whynot">
      <span>问问某个策略为什么没被推荐：</span>
      <select value={whyNotId} onChange={e => setWhyNotId(e.target.value)}>
        <option value="">选择策略…</option>
        {library.map((c: any) => <option key={c.strategy_id} value={c.strategy_id}>{c.name}</option>)}
      </select>
      <button className="secondary-button" disabled={!whyNotId || turn.isPending} onClick={() => {
        const card = library.find((c: any) => String(c.strategy_id) === whyNotId)
        say(`为什么不推荐「${card?.name}」？`)
        turn.mutate({ prefs, action: { type: 'why_not', strategy_id: Number(whyNotId) } })
      }}>问</button>
    </div>}
    <div className="agent-input">
      <input value={text} onChange={e => setText(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') send() }}
        placeholder={complete ? '也可以直接打字，比如"回撤放宽到 35% 呢"、"比较集成模型和上下影线"' : '先点上面的按钮回答，也可以直接打字'} disabled={turn.isPending}/>
      <button className="primary-button" onClick={send} disabled={!text.trim() || turn.isPending}><Send size={15}/></button>
    </div>
  </div>
}

function RecommendReply({ reply }: { reply: any }) {
  const chosen = reply.recommendation
  return <div className="agent-recommend">
    <span className="eyebrow">推荐</span>
    <h3>{chosen.name}</h3>
    <small>{chosen.description}</small>
    <CardMetrics card={chosen}/>
    <p className="recommend-text">{String(reply.explanation?.content || '').replace(/\*\*/g, '')}</p>
    {reply.explanation?.llm_error && <p className="muted-note warn">{reply.explanation.llm_error}</p>}
    <div className="risk-box">
      <b>怎么选出来的</b>
      {reply.rules.map((r: string) => <span key={r}>• {r}</span>)}
      {reply.notes.map((n: string) => <span key={n}>• {n}</span>)}
    </div>
    <div className="agent-links">
      <Link className="secondary-button" to={reply.next_step.strategy_url}>查看这个策略</Link>
      <Link className="primary-button" to={reply.next_step.practice_url}>去回测它</Link>
    </div>
    <p className="muted-note">成绩卡是 {chosen.standard.start_date} ~ {chosen.standard.end_date} 的历史回测，不代表未来收益。解释来源：{reply.explanation?.provider} · {reply.explanation?.model_version}</p>
  </div>
}
