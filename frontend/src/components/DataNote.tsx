import { useQuery } from '@tanstack/react-query'
import { Database } from 'lucide-react'
import { api, formatPercent } from '../api'

// 数据说明（ADR-0053）：策略用了哪几年的数据，以及这段回测一共赚了多少个点。
export function DataNote({ strategyId, start, end, universe, totalReturn, benchmarkReturn }: {
  strategyId: number
  start?: string
  end?: string
  universe?: string
  totalReturn?: number
  benchmarkReturn?: number
}) {
  const { data } = useQuery({
    queryKey: ['data-note', strategyId, start, end, universe],
    queryFn: async () => (await api.get(`/strategies/${strategyId}/data-note`, { params: { start_date: start, end_date: end, universe } })).data,
  })
  if (!data) return null
  return <div className="strategy-note data-note"><Database size={17}/><div>
    <b>数据说明{totalReturn != null ? `：${start} ~ ${end} 累计 ${formatPercent(totalReturn)}（${Math.round(totalReturn * 100)} 个点）` : ''}
      {benchmarkReturn != null ? `，同期沪深300 ${formatPercent(benchmarkReturn)}` : ''}</b>
    {data.lines.map((line: string) => <p key={line}>{line}</p>)}
  </div></div>
}
