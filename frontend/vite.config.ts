import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

// 只用到 process.env，不为此引入 @types/node
declare const process: { env: Record<string, string | undefined> }

// 外网访问（Cloudflare 临时隧道）用：设置 PUBLIC_BASIC_AUTH="用户名:密码" 时，页面和 /api 都要先通过
// HTTP Basic 认证，并放行 *.trycloudflare.com 域名。不设置时（内网 5173）行为不变。
const publicAuth = process.env.PUBLIC_BASIC_AUTH

function basicAuth(credentials: string): Plugin {
  const expected = 'Basic ' + btoa(unescape(encodeURIComponent(credentials)))
  return {
    name: 'public-basic-auth',
    configureServer(server) {
      server.middlewares.use((req: any, res: any, next: () => void) => {
        if (req.headers.authorization === expected) return next()
        res.statusCode = 401
        res.setHeader('WWW-Authenticate', 'Basic realm="PatienceQuant", charset="UTF-8"')
        res.setHeader('Content-Type', 'text/plain; charset=utf-8')
        res.end('需要登录')
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), ...(publicAuth ? [basicAuth(publicAuth)] : [])],
  server: {
    port: 5173,
    allowedHosts: publicAuth ? ['.trycloudflare.com'] : undefined,
    // 演示环境可以指向另一个后端：VITE_API_TARGET=http://localhost:8001 npx vite --port 5174
    proxy: { '/api': process.env.VITE_API_TARGET || 'http://localhost:8000' },
  },
})
