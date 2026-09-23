import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.training.trainer import FIRST_TEST_YEAR, TrainingConfig


def test_training_config_validation_and_fingerprint():
    base = dict(framework="lightgbm", factors=["return_20d", "volatility_20d"])
    a = TrainingConfig(**base).validate()
    b = TrainingConfig(framework="lightgbm", factors=["volatility_20d", "return_20d"]).validate()
    assert a.fingerprint() == b.fingerprint()  # 因子顺序不影响是不是同一个模型
    assert a.fingerprint() != TrainingConfig(**base, horizon=20).validate().fingerprint()
    for bad in (dict(framework="svm"), dict(factors=["roe"]), dict(factors=[]), dict(horizon=45), dict(seeds=9)):
        with pytest.raises(ValueError):
            TrainingConfig(**{**base, **bad}).validate()


def _wait_ready(client, model_id, timeout=600):
    for _ in range(timeout):
        model = client.get(f"/api/models/{model_id}/detail").json()
        if model["status"] in ("ready", "failed", "cancelled"):
            return model
        time.sleep(1)
    raise AssertionError("训练超时")


def test_define_model_strategy_trains_then_backtests_and_reuses_model():
    """策略制定里训练模型（ADR-0052）：保存策略 → 后台训练 → 可以回测；同样的训练设置沿用同一个模型。"""
    training = {"framework": "lightgbm", "factors": ["return_20d", "return_60d", "volatility_20d", "ma_deviation_20d"],
                "horizon": 20, "pool": "broad30", "n_estimators": 60, "seeds": 1}
    spec = {"selection": {"type": "top_pct", "pct": 0.3}, "rebalance": {"frequency": "monthly"}}
    with TestClient(app) as client:
        created = client.post("/api/strategies/model", json={"name": "测试·训练模型", "default_universe": "broad30",
                                                             "training": training, "spec": spec})
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["reused_model"] is False and body["model_id"].startswith("trained/")
        model = _wait_ready(client, body["model_id"])
        assert model["status"] == "ready", model.get("error")
        assert model["years"] and model["years"][0] >= FIRST_TEST_YEAR
        assert model["years"] == list(range(model["years"][0], model["years"][-1] + 1))
        assert any(v.get("ic") is not None for v in model["metrics"].values())
        assert model["factors"] == training["factors"]

        run = client.post("/api/backtests", json={"strategy_id": body["id"], "start_date": "2024-01-01", "end_date": "2024-12-31"})
        assert run.status_code == 200, run.text
        assert run.json()["metrics"]["trade_count"] > 0

        again = client.post("/api/strategies/model", json={"name": "测试·训练模型（加择时）", "default_universe": "broad30",
                                                           "training": dict(training, factors=list(reversed(training["factors"]))),
                                                           "spec": {**spec, "timing": {"type": "rsrs"}}}).json()
        assert again["reused_model"] is True and again["model_id"] == body["model_id"]

        factors = {f["key"]: f for f in client.get("/api/factors").json()}
        assert any("测试·训练模型" in user for user in factors["return_20d"]["used_by"])
        assert factors["roe"]["trainable"] is False


def test_factor_weight_strategy_can_use_any_daily_library_factor():
    """因子权重策略可以直接用因子库里的单个日线因子（ADR-0052 第三步）。"""
    with TestClient(app) as client:
        options = client.get("/api/pipeline/options").json()
        library = {f["key"] for f in options["factors"] if f.get("kind") == "library"}
        assert {"salience_str", "terrified_score", "apb_20d", "return_20d"} <= library
        assert {"llt", "ma_channel", "one_way_vol", "rps_vol", "high_moment", "volume_resonance", "qrs"} <= {t["type"] for t in options["timings"]}

        created = client.post("/api/strategies/spec", json={
            "name": "测试·APB+STR", "default_universe": "broad30",
            "spec": {"scorer": {"type": "factor_weights", "weights": {"apb_20d": 0.5, "salience_str": 0.5}},
                     "timing": {"type": "llt"}, "selection": {"type": "top_n", "n": 10}}})
        assert created.status_code == 200, created.text
        run = client.post("/api/backtests", json={"strategy_id": created.json()["id"], "start_date": "2024-01-01", "end_date": "2024-06-30"})
        assert run.status_code == 200, run.text
        assert run.json()["metrics"]["trade_count"] > 0
