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

import pytest


@pytest.fixture(scope="session", autouse=True)
def _cleanup_test_database():
    yield
    _TEST_DB_PATH.unlink(missing_ok=True)
