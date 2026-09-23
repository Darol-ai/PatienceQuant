import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { CalendarRange, Check, Copy, Database, Play, SlidersHorizontal, Sparkles, TrendingUp } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, formatPercent } from '../api'
import { Card, ErrorState, LoadingState, PageHeader, PanelHeader, Toast } from '../components/UI'

const labels: Record<string, string> = { fundamental: '基本面', valuation: '估值', quality: '盈利质量', momentum: '动量', risk: '风险' }
const factorHint: Record<string, string> = { fundamental: 'ROE · 成长 · 现金流（暂无真实数据）', valuation: 'PE · PB · PS · 股息率（暂无真实数据）', quality: '稳定性 · 毛利 · 净利（暂无真实数据）', momentum: '3M · 6M · 12M 收益', risk: '波动 · 回撤 · Beta' }

export function StrategyCenter() {
  const queryClient = useQueryClient()
  const { data, isLoading, isError } = useQuery({ queryKey: ['strategies'], queryFn: async () => (await api.get('/strategies')).data })
  const { data: coverage } = useQuery({ queryKey: ['research-coverage'], queryFn: async () => (await api.get('/research/coverage')).data })
  const [selected, setSelected] = useState<any>(null)
  const [message, setMessage] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  // 这个页面的权重滑块/风控参数表单是专门给multifactor(因子权重组合)
  // 这一种策略类型设计的——LightGBM/XGBoost这类训练好的模型策略没有
  // "权重"这个概念，硬套上去表单会显示一堆和模型实际逻辑无关的滑块。
  // 平台默认策略(is_default)不一定是multifactor(比如生产环境实际在用
  // 的可能是训练好的沪深300集成模型)，所以这里只从multifactor里选一个
  // 来加载编辑，不能直接拿第一个/is_default的就用。
  const multifactorStrategies = data?.filter((item: any) => (item.kind || 'multifactor') === 'multifactor')
  const productionDefault = data?.find((item: any) => item.is_default)
  useEffect(() => { if (multifactorStrategies?.[0] && !selected) setSelected(multifactorStrategies[0]) }, [multifactorStrategies, selected])
  const mutation = useMutation({
    mutationFn: async () => {
      const saved = (await api.post('/strategies', selected)).data
      // large_cap universe最多300支，real模式下逐支顺序查tushare(限速
      // 0.3~0.5秒/次，见ADR-0046)，冷缓存下这一步实测要几分钟——跟
      // BacktestCenter.tsx里自定义回测同样的原因，不能用api实例默认的
      // 120秒超时(会在后端还在正常跑的时候被前端提前掐断)，这里同样
      // 放宽到10分钟。
      const backtest = (await api.post('/backtests', {
        strategy_id: saved.id,
        universe: selected.universe || 'large_cap',
        start_date: selected.research_start_date || '2018-01-01',
        end_date: selected.research_end_date || '2025-12-31',
        initial_capital: 1000000,
        rebalance_frequency: selected.rebalance_frequency,
        holdings_count: selected.holdings_count,
        max_weight: selected.max_weight,
        commission: .001,
        slippage: .0005,
      }, { timeout: 600_000 })).data
      return { saved, backtest }
    },
    onSuccess: ({ saved, backtest }) => {
      setSelected({ ...selected, id: saved.id, version: saved.version, is_default: true, backtest_run_id: backtest.id, backtest_metrics: backtest.metrics, study_period: { start: backtest.start_date, end: backtest.end_date } })
      setMessage(`策略已保存并完成回测：累计收益 ${formatPercent(backtest.metrics.total_return)}`)
      queryClient.invalidateQueries({ queryKey: ['strategies'] })
    },
    // 之前这里没有onError——超时或者后端报错时按钮悄悄弹回"保存并回测"、
    // 没有任何提示，看起来像是"点了没反应/卡住了"。300支股票的real模式
    // 回测冷缓存下可能真的要跑几分钟，这里明确区分"还在等"和"确实失败
    // 了"，不让用户猜。
    onError: (error: any) => {
      setMessage(
        error?.code === 'ECONNABORTED'
          ? '保存并回测超时（超过10分钟）——300支股票的真实数据回测冷缓存下本来就慢，请稍后重试；如果反复超时，可能是后端刚重启、价格缓存还没预热完。'
          : error?.response?.data?.detail || '保存并回测失败，请检查策略参数后重试。',
      )
    },
  })
  if (isLoading) return <LoadingState />
  if (isError || !selected) return <ErrorState text="策略配置加载失败" />
  const total = Object.values(selected.weights).reduce((sum: number, value: any) => sum + Number(value), 0)
  const metrics = selected.backtest_metrics || {}
  const study = selected.study_period || { start: selected.research_start_date || '2018-01-01', end: selected.research_end_date || '2025-12-31' }
  return <>
    <PageHeader eyebrow="RESEARCH / 03" title="策略中心" description="面向当前与未来模拟盘，设计低频低买高卖规则，并用历史区间验证能否跑赢或贴近沪深300。" actions={<><span className="strategy-version">V{selected.version} · {selected.is_default ? '当前默认' : '因子权重实验'}</span><button className="primary-button" onClick={() => mutation.mutate()} disabled={mutation.isPending}><Play size={16}/>{mutation.isPending ? '保存并回测中…' : '保存并回测'}</button></>} />
    {productionDefault && productionDefault.kind !== 'multifactor' && (
      <div className="strategy-note" style={{ marginBottom: 16 }}>
        <Sparkles size={17}/>
        <div>
          <b>这个页面编辑的是"因子权重"这一类策略，跟平台当前实际默认策略不是同一个</b>
          <p>
            平台/模拟盘当前实际在用的默认策略是<b>「{productionDefault.name}」</b>
            （{productionDefault.kind === 'csi300_ensemble' ? 'LightGBM+XGBoost集成模型' : productionDefault.kind === 'csi300_lightgbm' ? 'LightGBM模型' : productionDefault.kind === 'csi300_xgboost' ? 'XGBoost模型' : '训练好的模型'}，
            年化 {productionDefault.backtest_metrics?.annual_return != null ? formatPercent(productionDefault.backtest_metrics.annual_return) : '—'}）——
            这是训练好的机器学习模型，没有"权重滑块"这种可调参数，不能在这个页面编辑。
            下面这套因子权重滑块是一个独立的、可以自由实验的策略类型，跑出来的收益不代表平台默认策略的实际表现。
          </p>
        </div>
      </div>
    )}
    <Card className="strategy-performance"><div className="strategy-performance-head"><div><span className="eyebrow">STRATEGY PERFORMANCE / CALCULATED</span><h2>{selected.name} · 历史可验证收益</h2><p>历史区间只用于验证规则；未来收益不保证，模拟盘会按最新可用数据重新生成信号。</p></div><div className="study-period-badge"><CalendarRange size={15}/><span>综合数据区间</span><b>{study.start} — {study.end}</b><small>{selected.backtest_run_id ? `回测 #${selected.backtest_run_id}` : '尚未运行回测'}</small></div></div><div className="performance-metrics"><div><span>这几年一共赚了</span><strong className={metrics.total_return >= 0 ? 'text-mint' : 'text-rose'}>{metrics.total_return == null ? '—' : formatPercent(metrics.total_return)}</strong></div><div><span>年化收益</span><strong>{metrics.annual_return == null ? '—' : formatPercent(metrics.annual_return)}</strong></div><div><span>最大回撤</span><strong className="text-rose">{metrics.max_drawdown == null ? '—' : formatPercent(metrics.max_drawdown)}</strong></div><div><span>Sharpe</span><strong>{metrics.sharpe == null ? '—' : Number(metrics.sharpe).toFixed(2)}</strong></div><div><span>沪深300</span><strong>{metrics.benchmark_return == null ? '—' : formatPercent(metrics.benchmark_return)}</strong></div><div><span>超额收益</span><strong className={metrics.excess_return >= 0 ? 'text-mint' : 'text-rose'}>{metrics.excess_return == null ? '—' : formatPercent(metrics.excess_return)}</strong></div><div><span>交易次数</span><strong>{metrics.trade_count == null ? '—' : Number(metrics.trade_count).toFixed(0)}</strong></div><div><span>持仓数量</span><strong>{selected.holdings_count} 只</strong></div></div><Link className="secondary-button strategy-backtest-link" to={`/backtest?strategy_id=${selected.id}`}><TrendingUp size={15}/>用当前配置运行回测</Link></Card>
    <Card className="quick-start-card"><PanelHeader title="快速制定策略" subtitle="先设置核心选项，其他参数使用稳健默认值"/><div className="quick-start-grid"><label>策略名称<input value={selected.name} onChange={e => setSelected({...selected, name: e.target.value})}/></label><label>研究股票池<select value={selected.universe || 'large_cap'} onChange={e => setSelected({...selected, universe: e.target.value})}><option value="large_cap">大盘股核心池</option><option value="a_share">A股全市场</option><option value="all_assets">A股 + OTC扩展</option></select></label><label>调仓频率<select value={selected.rebalance_frequency} onChange={e => setSelected({...selected, rebalance_frequency: e.target.value})}><option value="monthly">每月</option><option value="quarterly">每季度</option><option value="weekly">每周</option></select></label><label>持仓数量<div className="number-input"><button onClick={() => setSelected({...selected, holdings_count: Math.max(10, selected.holdings_count - 1)})}>−</button><b>{selected.holdings_count} 只</b><button onClick={() => setSelected({...selected, holdings_count: Math.min(100, selected.holdings_count + 1)})}>+</button></div></label></div><div className="quick-start-actions"><span><Sparkles size={14}/> 打分 → 选前 N 名 → 按分数加权 → × 沪深300趋势择时 → 调仓</span><button className="secondary-button" onClick={() => setShowAdvanced(!showAdvanced)}>{showAdvanced ? '收起高级设置' : '查看高级设置'}</button></div></Card>
    <div className="strategy-layout"><div><Card><PanelHeader title="策略身份与数据区间" subtitle="股票池、调仓频率和研究年限会随策略版本保存"/><div className="form-grid"><label>策略名称<input value={selected.name} onChange={e => setSelected({...selected, name: e.target.value})}/></label><label>调仓周期<select value={selected.rebalance_frequency} onChange={e => setSelected({...selected, rebalance_frequency: e.target.value})}><option value="weekly">每周</option><option value="monthly">每月</option><option value="quarterly">每季度</option></select></label><label>股票池<select value={selected.universe || 'large_cap'} onChange={e => setSelected({...selected, universe: e.target.value})}><option value="large_cap">大盘股核心池</option><option value="a_share">A股全市场</option><option value="all_assets">A股全部</option></select></label><label>研究开始<input type="date" value={selected.research_start_date || '2018-01-01'} onChange={e => setSelected({...selected, research_start_date: e.target.value})}/></label><label>研究结束<input type="date" value={selected.research_end_date || '2025-12-31'} onChange={e => setSelected({...selected, research_end_date: e.target.value})}/></label><label className="full">策略描述<textarea value={selected.description} onChange={e => setSelected({...selected, description: e.target.value})}/></label></div></Card><Card><PanelHeader title="因子权重" subtitle="五组因子横截面标准化后合成综合评分" action={<span className={`weight-total ${Math.abs(total - 1) < .01 ? 'valid' : 'invalid'}`}>{(total * 100).toFixed(0)}% <Check size={14}/></span>}/><div className="weight-editor">{Object.entries(selected.weights).map(([key, value]: [string, any]) => <div className="weight-row" key={key}><div className="weight-label"><span className={`weight-icon ${key}`}><SlidersHorizontal size={14}/></span><b>{labels[key]}</b><small>{factorHint[key]}</small></div><input type="range" min="0" max="50" step="1" value={Number(value) * 100} onChange={e => setSelected({...selected, weights: {...selected.weights, [key]: Number(e.target.value) / 100}})}/><strong>{(Number(value) * 100).toFixed(0)}%</strong></div>)}</div></Card></div><div><Card className="sticky-card"><PanelHeader title="组合约束" subtitle="风险控制与资产分配"/><div className="constraint-list"><label>持仓数量<div className="number-input"><button onClick={() => setSelected({...selected, holdings_count: Math.max(10, selected.holdings_count - 1)})}>−</button><b>{selected.holdings_count}</b><button onClick={() => setSelected({...selected, holdings_count: Math.min(100, selected.holdings_count + 1)})}>+</button></div></label><label>单股最大权重<div className="input-with-suffix"><input type="number" min="5" max="100" value={selected.max_weight * 100} onChange={e => setSelected({...selected, max_weight: Number(e.target.value) / 100})}/><span>%</span></div></label><label>风险关闭时股票暴露<div className="input-with-suffix"><input type="number" min="10" max="100" value={Number(selected.risk_off_exposure ?? .75) * 100} onChange={e => setSelected({...selected, risk_off_exposure: Number(e.target.value) / 100})}/><span>%</span></div></label><label>换手忽略带<div className="input-with-suffix"><input type="number" min="0" max="25" value={Number(selected.turnover_band ?? .03) * 100} onChange={e => setSelected({...selected, turnover_band: Number(e.target.value) / 100})}/><span>%</span></div></label><label className="checkbox-row"><span>沪深300趋势闸门</span><input type="checkbox" checked={selected.trend_filter ?? true} onChange={e => setSelected({...selected, trend_filter: e.target.checked})}/></label></div><div className="strategy-note"><Sparkles size={17}/><div><b>当前/未来执行提示</b><p>以沪深300为比较基准；每期按因子综合分选出前 {selected.holdings_count} 名、按分数加权，指数趋势转弱时降低整体仓位。未来表现不会自动复制历史收益。</p></div></div><button className="secondary-button full-button" onClick={() => { navigator.clipboard?.writeText(JSON.stringify(selected, null, 2)); setMessage('策略 JSON 已复制') }}><Copy size={15}/>复制配置</button></Card><Card><PanelHeader title="买卖规则与因子说明" subtitle="Quant Engine 的可审计决策口径"/><div className="factor-copy"><p><b>① 打分</b>：五组因子各自在股票池里算百分位排名（0～100），按权重加总成综合分。<b>真实行情下目前只有动量、风险两组有真实数据</b>；基本面、估值、盈利质量需要财务数据，本地行情库还没有，这三组按 0 权重计算（不会用程序生成的数字凑数）。</p><p><b>② 选股</b>：综合分前 {selected.holdings_count} 名。<b>③ 权重</b>：按综合分比例分配，单股不超过 {formatPercent(selected.max_weight)}，超出部分留作现金。</p><p><b>④ 择时</b>：沪深300低于50日均线时整体仓位降到 {formatPercent((1 + Number(selected.risk_off_exposure ?? .75)) / 2)}，低于200日均线时降到 {formatPercent(selected.risk_off_exposure ?? .75)}。</p><p><b>⑤ 调仓</b>：{selected.rebalance_frequency === 'monthly' ? '月度' : selected.rebalance_frequency === 'weekly' ? '周度' : '季度'}信号、下一交易日执行，只交易目标与当前持仓的差额；比例变化小于 {formatPercent(selected.turnover_band ?? .03)} 的不动；按 100 股一手取整，含手续费与滑点。</p><p><b>动量 {formatPercent(selected.weights.momentum)}</b>：3/6/12 个月收益率；<b>风险 {formatPercent(selected.weights.risk)}</b>：波动率、最大回撤和 Beta 低者优先。</p><p>迁移说明：原来的"LightGBM 信号层"是用当天因子分自己训练自己、不含未来信息，已经去掉；保护性止损、目标波动率、回撤刹车、低吸高抛这些叠加层也不再属于策略（见 ADR-0048）。</p></div></Card></div></div>
    {coverage && <Card className="research-coverage-card"><PanelHeader title="我的研究标注" subtitle={`已标注 ${coverage.total} 个标的 · 大盘核心 ${coverage.large_cap_count} 个`} action={<span className="data-badge"><Database size={13}/>股票与板块</span>} /><div className="research-sector-row">{coverage.sectors.map((sector: any) => <span className="research-sector-pill" key={sector.name}>{sector.name}<b>{sector.count}</b></span>)}</div><div className="research-annotation-grid">{coverage.items.map((item: any, index: number) => <div className="research-annotation" key={item.symbol}><div className="research-annotation-top"><span className="research-index">{String(index + 1).padStart(2, '0')}</span><div><b>{item.name}</b><span>{item.symbol} · {item.sector} / {item.industry}</span></div><span className="group-tag">{item.bucket}</span></div><p>{item.thesis}</p><small>风险观察：{item.risk}</small></div>)}</div></Card>}
    {message && <Toast message={message} />}
  </>
}
