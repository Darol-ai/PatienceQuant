PYTHON ?= python3

.PHONY: setup dev backend frontend test demo docker-up

setup:
	@if command -v uv >/dev/null 2>&1; then \
		if [ ! -x .venv/bin/python ]; then \
			if command -v python3.11 >/dev/null 2>&1; then \
				uv venv --python python3.11 .venv; \
			else \
				uv venv --python 3.9 .venv || uv venv .venv; \
			fi; \
		fi; \
		uv pip install --python .venv/bin/python -e './backend[dev]'; \
	else \
		if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi; \
		.venv/bin/python -m pip install -e './backend[dev]'; \
	fi
	cd frontend && npm install

backend:
	cd backend && ../.venv/bin/python -m uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

dev:
	@echo "Run 'make backend' and 'make frontend' in two terminals, or use 'make docker-up'."

test:
	cd backend && ../.venv/bin/python -m pytest -q
	cd frontend && npm run build

demo:
	@echo "Dashboard: http://localhost:5173  API docs: http://localhost:8000/docs"

docker-up:
	docker compose up --build
