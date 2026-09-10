# PatienceQuant Backend

FastAPI + SQLite implementation of the low-frequency multi-factor investing MVP.

```bash
cd backend
uv sync --extra dev
uv run uvicorn app.main:app --reload --port 8000
```

The default provider is deterministic offline Demo data. Set `DATA_MODE=real` and install
`uv sync --extra real-data` to enable best-effort AKShare refresh with automatic fallback.

