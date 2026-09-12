import { useMutation } from '@tanstack/react-query'
import { BrainCircuit, FileText, Sparkles, UploadCloud } from 'lucide-react'
import { useRef, useState } from 'react'
import { api, formatPercent } from '../api'
import { Card, PageHeader, PanelHeader, Toast } from '../components/UI'

type ExplainResult = {
  headline: string
  content: string
  risks: string[]
  provider: string
  model_version: string
  confidence: number | null
  audit_id: number
}

type StrategyAssistResult = {
  content: string
  provider: string
  model_version: string
  confidence: number | null
  audit_id: number
  suggested_params: Record<string, unknown>
}

type ReportAnalysisResult = {
  summary: string
  key_points: string[]
  strategy_reference: string
  provider: string
  model_version: string
  audit_id: number
  filename: string
}

type ReportFactorResult = {
  has_reference_value: boolean
  rationale: string
  factor_weights: Record<string, number>
  provider: string
  model_version: string
  audit_id: number
  filename: string
  backtest: { metrics: Record<string, number>; trade_count: number } | null
  backtest_error: string | null
}

function AuditDecision({ auditId }: { auditId: number }) {
  const [decision, setDecision] = useState<'adopted' | 'rolled_back' | null>(null)
  const mutation = useMutation({
    mutationFn: async (payload: { adopted?: boolean; rolled_back?: boolean }) =>
      (await api.post(`/ai/audit/${auditId}/decision`, payload)).data,
  })
  return (
    <div className="audit-decision">
      <span>审计 #{auditId} · 是否采纳这次 AI 输出？</span>
      <button
        className="secondary-button"
        disabled={mutation.isPending}
        onClick={() => { mutation.mutate({ adopted: true }); setDecision('adopted') }}
      >采纳</button>
      <button
        className="secondary-button"
        disabled={mutation.isPending}
        onClick={() => { mutation.mutate({ adopted: false, rolled_back: true }); setDecision('rolled_back') }}
      >不采纳 / 回滚</button>
      {decision && <span className="audit-decided">已记录：{decision === 'adopted' ? '已采纳' : '已回滚'}</span>}
    </div>
  )
}

function ReportFilePicker({ file, onChange }: { file: File | null; onChange: (file: File | null) => void }) {
  const inputRef = useRef<HTMLInputElement>(null)
  return (
    <div className="report-picker" onClick={() => inputRef.current?.click()}>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt,.md"
        hidden
        onChange={e => onChange(e.target.files?.[0] ?? null)}
      />
      <UploadCloud size={18}/>
      <div>
        <b>{file ? file.name : '点击选择研报文件'}</b>
        <span>支持 PDF / TXT / Markdown</span>
      </div>
    </div>
  )
}

export function AIResearch() {
  const [form, setForm] = useState({ symbol: '600036', action: 'BUY', score: 82, rank: 3, industry: '银行', volatility: .23, max_drawdown: .18, target_weight: .15, use_llm: true })
  const [result, setResult] = useState<ExplainResult | null>(null)
  const mutation = useMutation({ mutationFn: async () => (await api.post('/ai/explain/trade', form)).data, onSuccess: setResult })

  const [strategyDescription, setStrategyDescription] = useState('低波动、以沪深300大盘股为主、每月调仓一次的稳健策略，控制最大回撤')
  const [strategyResult, setStrategyResult] = useState<StrategyAssistResult | null>(null)
  const strategyMutation = useMutation({
    mutationFn: async () => (await api.post('/ai/strategy-assistant', { description: strategyDescription })).data,
    onSuccess: setStrategyResult,
  })

  const [reportFile, setReportFile] = useState<File | null>(null)
  const [reportResult, setReportResult] = useState<ReportAnalysisResult | null>(null)
  const reportMutation = useMutation({
    mutationFn: async () => {
      const body = new FormData()
      body.append('file', reportFile as File)
      return (await api.post('/ai/report/analyze', body)).data
    },
    onSuccess: setReportResult,
  })

  const [factorFile, setFactorFile] = useState<File | null>(null)
  const [factorResult, setFactorResult] = useState<ReportFactorResult | null>(null)
  const factorMutation = useMutation({
    mutationFn: async () => {
      const body = new FormData()
      body.append('file', factorFile as File)
      return (await api.post('/ai/report/generate-factor', body, { params: { top_n: 5, start_date: '2024-01-01', end_date: '2025-12-31' } })).data
    },
    onSuccess: setFactorResult,
  })

  return <>
    <PageHeader
      eyebrow="INTELLIGENCE / 07"
      title="AI 投研助手"
      description="解释量化引擎为什么产生信号、把自然语言和研报翻译成策略参数与因子建议，但不替代 Quant Engine 做交易决定。"
      actions={<span className="ai-mode"><Sparkles size={15}/>{form.use_llm ? 'LLM 研究摘要（OpenAI 兼容）' : '规则解释器 · 无需 API Key'}</span>}
    />
    <div className="ai-layout">
      <Card>
        <PanelHeader title="交易上下文" subtitle="输入一笔信号，生成可追溯解释"/>
        <div className="form-grid">
          <label>股票代码<input value={form.symbol} onChange={e => setForm({...form, symbol: e.target.value})}/></label>
          <label>交易方向<select value={form.action} onChange={e => setForm({...form, action: e.target.value})}><option>BUY</option><option>HOLD</option><option>SELL</option></select></label>
          <label>综合评分<input type="number" value={form.score} onChange={e => setForm({...form, score: Number(e.target.value)})}/></label>
          <label>排名<input type="number" value={form.rank} onChange={e => setForm({...form, rank: Number(e.target.value)})}/></label>
          <label>所属行业<input value={form.industry} onChange={e => setForm({...form, industry: e.target.value})}/></label>
          <label>目标权重<div className="input-with-suffix"><input type="number" value={form.target_weight * 100} onChange={e => setForm({...form, target_weight: Number(e.target.value) / 100})}/><span>%</span></div></label>
        </div>
        <label className="llm-toggle"><input type="checkbox" checked={form.use_llm} onChange={e => setForm({...form, use_llm: e.target.checked})}/> 用语言模型生成研究摘要（未配置 Key 时自动回退到规则解释器）</label>
        <button className="primary-button full-button" onClick={() => mutation.mutate()} disabled={mutation.isPending}><BrainCircuit size={16}/>{mutation.isPending ? '正在分析…' : '生成交易解释'}</button>
      </Card>
      <Card className="ai-result-card">
        <PanelHeader title="研究结论" subtitle="规则驱动 · 可审计"/>
        {result ? (
          <div className="ai-result">
            <div className="ai-headline"><div className="ai-orb"><BrainCircuit/></div><div><span>QUANT EXPLANATION</span><h3>{result.headline}</h3></div></div>
            <p>{result.content}</p>
            <div className="risk-box"><b>风险提示</b>{result.risks.map((risk: string) => <span key={risk}>• {risk}</span>)}</div>
            <div className="provider-line"><span>解释来源</span><b>{result.provider} · {result.model_version}</b></div>
            {result.confidence != null && <div className="provider-line"><span>置信度</span><b>{(result.confidence * 100).toFixed(0)}%</b></div>}
            <AuditDecision auditId={result.audit_id}/>
          </div>
        ) : (
          <div className="ai-placeholder"><div className="ai-orb large"><BrainCircuit/></div><b>让数据告诉你，为什么交易。</b><span>Quant Engine 提供评分、风险和目标权重，助手将它们翻译成自然语言。</span></div>
        )}
      </Card>
    </div>

    <Card>
      <PanelHeader title="策略助手" subtitle="自然语言 → 策略参数建议（只是建议，不会自动创建或修改策略）"/>
      <textarea
        className="strategy-assist-input"
        rows={3}
        value={strategyDescription}
        onChange={e => setStrategyDescription(e.target.value)}
        placeholder="用一段话描述你想要的策略风格、持仓数量、调仓频率、风险偏好……"
      />
      <button className="primary-button full-button" onClick={() => strategyMutation.mutate()} disabled={strategyMutation.isPending}>
        <Sparkles size={16}/>{strategyMutation.isPending ? '正在生成建议…' : '生成策略参数建议'}
      </button>
      {strategyResult && (
        <div className="ai-result">
          <p>{strategyResult.content}</p>
          {Object.keys(strategyResult.suggested_params).length > 0 && (
            <div className="risk-box">
              <b>建议参数（复制到策略中心手动创建）</b>
              {Object.entries(strategyResult.suggested_params).map(([key, value]) => (
                <span key={key}>• {key}: {String(value)}</span>
              ))}
            </div>
          )}
          <div className="provider-line"><span>建议来源</span><b>{strategyResult.provider} · {strategyResult.model_version}</b></div>
          <AuditDecision auditId={strategyResult.audit_id}/>
        </div>
      )}
    </Card>

    <div className="ai-layout">
      <Card>
        <PanelHeader title="研报总结助手" subtitle="上传研报 → 摘要、关键点与策略参考（只解读，不生成代码）"/>
        <ReportFilePicker file={reportFile} onChange={setReportFile}/>
        <button className="primary-button full-button" onClick={() => reportMutation.mutate()} disabled={!reportFile || reportMutation.isPending}>
          <FileText size={16}/>{reportMutation.isPending ? '正在解读研报…' : '生成研报总结'}
        </button>
        {reportResult && (
          <div className="ai-result">
            <p>{reportResult.summary}</p>
            {reportResult.key_points.length > 0 && (
              <div className="risk-box">
                <b>关键点</b>
                {reportResult.key_points.map(point => <span key={point}>• {point}</span>)}
              </div>
            )}
            <div className="risk-box"><b>策略参考（仅参考，不构成建议）</b><span>{reportResult.strategy_reference}</span></div>
            <div className="provider-line"><span>解读来源</span><b>{reportResult.provider} · {reportResult.model_version}</b></div>
            <AuditDecision auditId={reportResult.audit_id}/>
          </div>
        )}
      </Card>

      <Card>
        <PanelHeader title="研报生成因子" subtitle="判断研报是否有建模参考价值，生成因子并直接接入回测系统验证"/>
        <ReportFilePicker file={factorFile} onChange={setFactorFile}/>
        <button className="primary-button full-button" onClick={() => factorMutation.mutate()} disabled={!factorFile || factorMutation.isPending}>
          <Sparkles size={16}/>{factorMutation.isPending ? '正在生成因子并回测…' : '生成因子并回测（2024-2025）'}
        </button>
        {factorResult && (
          <div className="ai-result">
            <div className={`factor-reference-badge ${factorResult.has_reference_value ? 'positive' : 'neutral'}`}>
              {factorResult.has_reference_value ? '判定：对因子建模有参考价值' : '判定：没有可对应的因子建模价值'}
            </div>
            <p>{factorResult.rationale}</p>
            {Object.keys(factorResult.factor_weights).length > 0 && (
              <div className="risk-box">
                <b>生成的因子权重（白名单特征，超出范围的权重已自动丢弃）</b>
                {Object.entries(factorResult.factor_weights).map(([name, weight]) => (
                  <span key={name}>• {name}: {weight > 0 ? '+' : ''}{weight.toFixed(2)}</span>
                ))}
              </div>
            )}
            {factorResult.backtest && (
              <div className="risk-box">
                <b>回测结果（A 阶段 10 支股票，2024-2025；仅验证链路能跑通，不代表策略有效）</b>
                <span>累计收益：{formatPercent(factorResult.backtest.metrics.total_return)}</span>
                <span>最大回撤：{formatPercent(factorResult.backtest.metrics.max_drawdown)}</span>
                <span>交易笔数：{factorResult.backtest.trade_count}</span>
              </div>
            )}
            {factorResult.backtest_error && <div className="risk-box"><b>回测未执行</b><span>{factorResult.backtest_error}</span></div>}
            <div className="provider-line"><span>生成来源</span><b>{factorResult.provider} · {factorResult.model_version}</b></div>
            <AuditDecision auditId={factorResult.audit_id}/>
          </div>
        )}
      </Card>
    </div>

    <Card>
      <PanelHeader title="AI 使用边界" subtitle="系统安全原则"/>
      <div className="guardrail-grid">
        <div><b>只解释/建议，不下单</b><span>LLM 不产生目标权重，也不会绕过风险控制；策略助手和研报因子只给建议/白名单参数，不自动创建或覆盖策略。</span></div>
        <div><b>不执行生成代码</b><span>研报生成的因子只能是白名单特征的加权组合，不会执行 LLM 写出的任意代码。</span></div>
        <div><b>全程留痕</b><span>每次调用都记录模型版本、输入、输出，并支持人工标记采纳/回滚（AI 使用审计）。</span></div>
      </div>
    </Card>
    {(mutation.isError || strategyMutation.isError || reportMutation.isError || factorMutation.isError) && <Toast message="AI 调用失败" type="error"/>}
  </>
}
