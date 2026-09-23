"""不能让整个 pytest 套件继续和手动 `uvicorn` 开发服务器共用同一个
`data/patience_quant.db`（docs/adr/0043 的副产品发现）——之前每次跑测试
都会真实往那个文件里写回测记录、策略克隆、模拟盘调仓/重置，从不清理，
是仓库里 `/api/strategies` 出现几十条重复"回测#X 模拟盘"策略、
`trades` 表越滚越大的真正原因。

`app.db.session` 在导入时就用 `get_settings().database_url` 建好了
模块级的 `engine`/`SessionLocal`（`app.main` 的 lifespan 也是直接用这两
个全局对象做建表/迁移/播种，不经过 `Depends(get_db)`），所以唯一能在
所有代码路径生效的隔离点是：在任何 `app.*` 模块被导入之前，把
`DATABASE_URL` 环境变量指向一个进程私有的临时 sqlite 文件。conftest.py
的模块级代码保证在同目录下任何测试文件之前执行，这里做的事必须在文件
顶层、在此文件里第一次 `import app` 之前完成。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TEST_DB_PATH = Path(tempfile.gettempdir()) / f"patience_quant_test_{os.getpid()}.db"
_TEST_DB_PATH.unlink(missing_ok=True)
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"

# app.main的lifespan会在后台线程里预热沪深300策略的模型缓存(真实冷启动
# 要5分多钟)，这是给手动起的开发服务器用的，不是给测试套件用的——每个
# `with TestClient(app) as client:`都会重新进一次lifespan，测试本身该
# 用到这些缓存的地方(CSI300相关测试)自然会通过真实调用把它们建起来，
# 不需要额外再触发一次5分钟的后台预热跟测试的HTTP调用抢CPU。
# 通用目录换成tushare后DATA_MODE默认值也改成了real——测试套件不能因为
# 这个默认值变化就在每次seed时去真实发网络请求(慢、依赖外部服务可用性、
# CI环境不一定有网络)，所以显式钉死用demo目录(50支真实公司的本地模拟
# 数据，见app/data/demo.py)。CSI300/quant_v3那几个真实策略走独立的parquet
# 管道，不受这个开关影响，仍然是真实数据。
os.environ["DATA_MODE"] = "demo"

os.environ["PATIENCEQUANT_WARM_CSI300"] = "0"

# backend/.env 里配了真实的大模型 key——测试不能真的去调它（花钱、结果不确定）。
# 需要测大模型分支的测试自己 monkeypatch get_ai_credentials。
os.environ["OPENAI_API_KEY"] = ""

import pytest


@pytest.fixture(scope="session", autouse=True)
def _cleanup_test_database():
    yield
    _TEST_DB_PATH.unlink(missing_ok=True)
