import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 只用到 process.env，不为此引入 @types/node
declare const process: { env: Record<string, string | undefined> }

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // 演示环境可以指向另一个后端：VITE_API_TARGET=http://localhost:8001 npx vite --port 5174
    proxy: { '/api': process.env.VITE_API_TARGET || 'http://localhost:8000' },
  },
})

