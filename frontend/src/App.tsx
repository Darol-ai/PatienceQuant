import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { LoadingState } from './components/UI'

const page = <T extends string>(loader: () => Promise<Record<T, any>>, name: T) => lazy(() => loader().then(module => ({ default: module[name] })))
const Home = page(() => import('./pages/Home'), 'Home')
const Library = page(() => import('./pages/Library'), 'Library')
const StrategyDetail = page(() => import('./pages/StrategyDetail'), 'StrategyDetail')
const StrategyEditor = page(() => import('./pages/StrategyEditor'), 'StrategyEditor')
const Practice = page(() => import('./pages/Practice'), 'Practice')
const BacktestReport = page(() => import('./pages/BacktestReport'), 'BacktestReport')
const Paper = page(() => import('./pages/Paper'), 'Paper')
const Pools = page(() => import('./pages/Pools'), 'Pools')
const StockDetail = page(() => import('./pages/StockDetail'), 'StockDetail')
const DataModels = page(() => import('./pages/DataModels'), 'DataModels')
const AITools = page(() => import('./pages/AITools'), 'AITools')

// 导航结构见 docs/adr/0051：首页（智能体）、策略库、策略实践、模拟盘、股票池、数据与模型、AI 工具。
export default function App() {
  return (
    <Layout>
      <Suspense fallback={<LoadingState text="正在加载…" />}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/library" element={<Library />} />
          <Route path="/library/:id" element={<StrategyDetail />} />
          <Route path="/strategy/new" element={<StrategyEditor />} />
          <Route path="/backtest" element={<Practice />} />
          <Route path="/backtests/:id" element={<BacktestReport />} />
          <Route path="/paper" element={<Paper />} />
          <Route path="/pools" element={<Pools />} />
          <Route path="/stocks/:symbol" element={<StockDetail />} />
          <Route path="/data" element={<DataModels />} />
          <Route path="/ai" element={<AITools />} />
          {/* 旧地址 */}
          <Route path="/strategy" element={<Navigate to="/library" replace />} />
          <Route path="/trading" element={<Navigate to="/paper" replace />} />
          <Route path="/stocks" element={<Navigate to="/pools" replace />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </Suspense>
    </Layout>
  )
}
