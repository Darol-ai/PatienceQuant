PYTHON ?= python3

.PHONY: setup data reproduce dev backend frontend test demo docker-up

# 后端需要 Python 3.10 以上（开发与成绩卡快照用 3.13）；默认安装训练依赖（Linux 上是 GPU 版 torch）。
setup:
	@PY=$$(for v in python3.13 python3.12 python3.11 python3.10; do command -v $$v >/dev/null 2>&1 && { echo $$v; break; }; done); \
	if [ -z "$$PY" ]; then echo "需要 Python 3.10 以上（推荐 3.13）"; exit 1; fi; \
	if command -v uv >/dev/null 2>&1; then \
		[ -x .venv/bin/python ] || uv venv --python $$PY .venv; \
		uv pip install --python .venv/bin/python -r backend/requirements-lock.txt -e './backend[dev,real-data,training]'; \
	else \
		[ -x .venv/bin/python ] || $$PY -m venv .venv; \
		.venv/bin/python -m pip install -r backend/requirements-lock.txt -e './backend[dev,real-data,training]'; \
	fi
	cd frontend && npm install

# 下载数据包（本地行情库、每日指标、旧模型文件，约 1.2GB，GitHub Release 附件）
data:
	$(PYTHON) scripts/fetch_data.py

# 一键复现：训练内置模型、计算全部成绩卡，并与 docs/成绩卡快照.json 逐项对比（GPU 约 1 小时）
reproduce:
	cd backend && ../.venv/bin/python scripts/reproduce.py

backend:
	cd backend && ../.venv/bin/python -m uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

dev:
	@echo "在两个终端里分别运行 make backend 和 make frontend，然后打开 http://localhost:5173"

test:
	cd backend && ../.venv/bin/python -m pytest -q
	cd frontend && npx tsc -b && npm run build

demo:
	@echo "前端：http://localhost:5173  接口文档：http://localhost:8000/docs"

docker-up:
	docker compose up --build
