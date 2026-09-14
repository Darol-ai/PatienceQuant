from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.models import Strategy
from app.db.session import Base


def test_missing_csi300_data_files_do_not_crash_startup(monkeypatch):
    """沪深300策略集依赖的parquet文件不在git里(见docs/部署-真实数据准备.md)，
    首次部署如果还没跑准备脚本，这些文件就是缺失的——_ensure_csi300_strategies
    必须能优雅跳过，不能让整个应用起不来。

    用独立的内存sqlite而不是TestClient/共享测试db：整个pytest进程共用
    一个临时db文件(见conftest.py)，跑到这个测试时前面的测试早就把这几个
    策略seed好了——_ensure_csi300_strategies对每个kind有"已存在就跳过"
    的检查，不会重新调用csi300_stocks()，monkeypatch根本测不到想测的
    异常路径。必须给一个保证是空的、全新的db，才能真正验证"文件缺失时
    从零开始注册"这条路径。
    """
    import app.db.seed as seed_module

    def _raise(*args, **kwargs):
        raise FileNotFoundError("csi300_constituents.parquet 不存在（模拟未准备数据）")

    monkeypatch.setattr(seed_module, "csi300_stocks", _raise)

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    with Session() as db:
        seed_module._ensure_csi300_strategies(db)
        kinds = {row.kind for row in db.scalars(select(Strategy)).all()}
        assert not kinds & {"csi300_lightgbm", "csi300_xgboost", "csi300_ensemble"}
