import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { LoadingState } from './components/UI'

const Dashboard = lazy(() => import('./pages/Dashboard').then(module => ({ default: module.Dashboard })))
const Stocks = lazy(() => import('./pages/Stocks').then(module => ({ default: module.Stocks })))
const StockDetail = lazy(() => import('./pages/StockDetail').then(module => ({ default: module.StockDetail })))
const StrategyCenter = lazy(() => import('./pages/StrategyCenter').then(module => ({ default: module.StrategyCenter })))
const BacktestCenter = lazy(() => import('./pages/BacktestCenter').then(module => ({ default: module.BacktestCenter })))
const PaperTrading = lazy(() => import('./pages/PaperTrading').then(module => ({ default: module.PaperTrading })))
const AutoTrading = lazy(() => import('./pages/AutoTrading').then(module => ({ default: module.AutoTrading })))
const AIResearch = lazy(() => import('./pages/AIResearch').then(module => ({ default: module.AIResearch })))

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<LoadingState text="正在加载工作台…" />}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/stocks" element={<Stocks />} />
          <Route path="/stocks/:symbol" element={<StockDetail />} />
          <Route path="/strategy" element={<StrategyCenter />} />
          <Route path="/backtest" element={<BacktestCenter />} />
          <Route path="/paper" element={<PaperTrading />} />
          <Route path="/trading" element={<AutoTrading />} />
          <Route path="/ai" element={<AIResearch />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </Suspense>
    </Layout>
  )
}
