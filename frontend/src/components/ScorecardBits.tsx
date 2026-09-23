import { Link } from 'react-router-dom'
import { formatPercent } from '../api'
import { frequencyLabel } from '../strategy'

// 成绩卡的几种展示：首页小列表、对话里的对照表、详情页指标格。

export function CardMetrics({ card }: { card: any }) {
  const m = card.metrics || {}
  return <div className="recommend-metrics">
    <div><span>年化</span><b>{formatPercent(m.annual_return ?? 0)}</b></div>
    <div><span>最大回撤</span><b className="text-rose">{formatPercent(m.max_drawdown ?? 0)}</b></div>
    <div><span>夏普</span><b>{Number(m.sharpe ?? 0).toFixed(2)}</b></div>
    <div><span>跑赢沪深300</span><b>{card.years_beating_benchmark}/{card.years} 年</b></div>
    <div><span>平均仓位</span><b>{formatPercent(card.average_exposure ?? 1, 0)}</b></div>
  </div>
}

export function CardCompareTable({ cards }: { cards: any[] }) {
  if (!cards?.length) return null
  return <div className="table-wrap"><table>
    <thead><tr><th>策略</th><th>年化</th><th>最大回撤</th><th>夏普</th><th>跑赢年份</th><th>平均仓位</th><th>调仓</th></tr></thead>
    <tbody>{cards.map(c => <tr key={c.strategy_id}>
      <td><Link className="stock-link-button" to={`/library/${c.strategy_id}`}>{c.name}</Link></td>
      {c.status === 'ready' ? <>
        <td>{formatPercent(c.metrics.annual_return)}</td>
        <td className="text-rose">{formatPercent(c.metrics.max_drawdown)}</td>
        <td>{Number(c.metrics.sharpe).toFixed(2)}</td>
        <td>{c.years_beating_benchmark}/{c.years}</td>
        <td>{formatPercent(c.average_exposure ?? 1, 0)}</td>
      </> : <td colSpan={5} className="muted-label">还没有成绩卡</td>}
      <td>{frequencyLabel[c.rebalance_frequency] || c.rebalance_frequency}</td>
    </tr>)}</tbody>
  </table></div>
}
