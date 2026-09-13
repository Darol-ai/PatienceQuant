import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './styles.css'

// staleTime拉长到5分钟、关掉切回浏览器标签页时的自动重新请求——这些
// 页面的数据都是回测/模拟盘调仓触发的，不是每秒变化的实时行情，之前
// 默认的30秒+切标签页自动刷新，会让"离开页面几分钟再回来"或者"切一下
// 浏览器标签页再切回来"都要重新等一次真实计算（比如LightGBM在300支
// 股票上重新打分），体感就是"每次都要重新加载"。真正需要live更新的
// 模拟盘/自动交易页面自己单独配了 refetchInterval，不受这里影响。
const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 5 * 60_000, retry: 1, refetchOnWindowFocus: false } },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter><App /></BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
)

