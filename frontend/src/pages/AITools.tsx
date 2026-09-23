import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FileText, Settings, UploadCloud, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { Card, PageHeader, PanelHeader, Toast } from '../components/UI'

type AISettings = {
  has_api_key: boolean
  base_url: string | null
  model: string
  source: 'database' | 'env'
}

function AISettingsModal({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient()
  const { data: settings } = useQuery<AISettings>({ queryKey: ['ai-settings'], queryFn: async () => (await api.get('/settings/ai')).data })
  const [apiKey, setApiKey] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [model, setModel] = useState('')
  const [message, setMessage] = useState('')

  useEffect(() => {
    if (settings) {
      setBaseUrl(settings.base_url || '')
      setModel(settings.model || '')
    }
  }, [settings])

  const saveMutation = useMutation({
    mutationFn: async () =>
      (await api.post('/settings/ai', {
        api_key: apiKey.trim() || undefined,
        base_url: baseUrl.trim() || undefined,
        model: model.trim() || undefined,
      })).data,
    onSuccess: (data: AISettings) => {
      queryClient.setQueryData(['ai-settings'], data)
      setApiKey('')
      setMessage('已保存，立即生效（无需重启服务）')
    },
  })

  const resetMutation = useMutation({
    mutationFn: async () => (await api.delete('/settings/ai')).data,
    onSuccess: (data: AISettings) => {
      queryClient.setQueryData(['ai-settings'], data)
      setApiKey('')
      setMessage('已恢复为部署环境（.env）里的默认配置')
    },
  })

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={event => event.stopPropagation()}>
        <div className="modal-title">
          <div><span className="eyebrow">SETTINGS</span><h2>OpenAI API 配置</h2></div>
          <button className="icon-button" onClick={onClose}><X size={16}/></button>
        </div>
        <p className="factor-copy" style={{ margin: '0 0 14px' }}>
          在这里保存的配置存在数据库里，优先于部署环境的 .env；改完立即对下一次 AI 调用生效，不需要重启服务。
          当前来源：<b>{settings?.source === 'database' ? '数据库（前端已配置）' : '.env（部署默认值）'}</b>
          {' · '}已配置 Key：<b>{settings?.has_api_key ? '是' : '否'}</b>
        </p>
        <label className="stack-label">
          API Key
          <input
            type="password"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            placeholder={settings?.has_api_key ? '已配置，留空则不修改' : '未配置（sk-...）'}
          />
        </label>
        <label className="stack-label">
          Base URL（可选，OpenAI 兼容网关）
          <input value={baseUrl} onChange={e => setBaseUrl(e.target.value)} placeholder="https://api.openai.com/v1" />
        </label>
        <label className="stack-label">
          模型名称
          <input value={model} onChange={e => setModel(e.target.value)} placeholder="gpt-4.1-mini" />
        </label>
        <div className="modal-actions">
          <button
            className="secondary-button"
            onClick={() => resetMutation.mutate()}
            disabled={resetMutation.isPending || settings?.source !== 'database'}
          >
            恢复默认（.env）
          </button>
          <button className="primary-button" onClick={() => saveMutation.mutate()} disabled={saveMutation.isPending}>
            {saveMutation.isPending ? '保存中…' : '保存'}
          </button>
        </div>
        {saveMutation.isError && <div className="form-error">保存失败，请检查输入后重试。</div>}
        {message && <div className="factor-copy" style={{ marginTop: 10 }}>{message}</div>}
      </div>
    </div>
  )
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

export function AITools() {
  const [showSettings, setShowSettings] = useState(false)
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

  return <>
    <PageHeader
      eyebrow="AI TOOLS"
      title="AI 工具"
      description="研报总结：上传研报，得到摘要和要点，只做解读，不影响任何策略。策略推荐在首页；交易解释在回测报告和模拟盘的每笔交易旁边。"
      actions={<button className="secondary-button" onClick={() => setShowSettings(true)}><Settings size={15}/>大模型设置</button>}
    />
    {showSettings && <AISettingsModal onClose={() => setShowSettings(false)} />}
    <div className="ai-single">
      <Card>
        <PanelHeader title="研报总结" subtitle="上传研报 → 摘要、关键点与策略参考（只解读，不生成代码）"/>
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

    </div>

    <Card>
      <PanelHeader title="AI 使用边界" subtitle="系统安全原则"/>
      <div className="guardrail-grid">
        <div><b>只解读，不下单</b><span>大模型不产生目标权重、不创建或修改策略、不下单。策略推荐里选哪个策略由规则决定，大模型只写解释。</span></div>
        <div><b>数字要有出处</b><span>推荐解释里的百分比必须能在成绩卡里找到，否则丢弃大模型的回答，改用规则生成的解释。</span></div>
        <div><b>全程留痕</b><span>每次调用都记录模型版本、输入、输出，可以人工标记采纳或回滚。</span></div>
      </div>
    </Card>
    {reportMutation.isError && <Toast message="AI 调用失败" type="error"/>}
  </>
}
