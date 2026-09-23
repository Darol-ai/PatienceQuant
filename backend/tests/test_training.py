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


def test_trained_builtins_rebuild_missing_models(monkeypatch, tmp_path):
    """新环境里没有内置模型策略的模型：登记策略时模型编号由训练设置决定，启动时缺了就排队训练（ADR-0053）。"""
    from app.pipeline import library
    from app.training import jobs, trainer

    monkeypatch.setattr(trainer, "MODELS_ROOT", tmp_path)
    queued = []
    monkeypatch.setattr(jobs, "enqueue", queued.append)
    monkeypatch.setenv("PATIENCEQUANT_AUTO_TRAIN", "1")

    ids = library.ensure_builtin_models()
    expected = [f"trained/{library.trained_builtin_config(item).fingerprint()}" for item in library.TRAINED_BUILTINS]
    assert ids == expected == queued
    assert all(trainer.read_record(i)["status"] == "queued" for i in ids)
    # 已经登记过（排队中或可用）的不会重复排队
    assert library.ensure_builtin_models() == [] and len(queued) == len(library.TRAINED_BUILTINS)

    monkeypatch.setenv("PATIENCEQUANT_AUTO_TRAIN", "0")
    assert library.ensure_builtin_models() == []


def test_lstm_trains_and_scores_consistently(monkeypatch, tmp_path):
    """LSTM：训练时对样本外的预测，和回测打分时按同一天序列算出的预测一致（口径相同）。"""
    import numpy as np
    from datetime import date

    from app.data.market_refresh import get_market_store
    from app.pipeline.library import broad30_symbols
    from app.pipeline.scorers import ModelScorer
    from app.pipeline.spec import ModelScorerSpec
    from app.training import lstm, trainer

    monkeypatch.setattr(trainer, "MODELS_ROOT", tmp_path)
    store = get_market_store()
    until = date(2021, 12, 31)
    config = trainer.TrainingConfig(framework="lstm", factors=["return_20d", "volatility_20d", "ma_deviation_20d"],
                                    pool="broad30", seeds=1).validate()
    metrics = trainer.train(config, "trained/lstm_test", until, progress=lambda _: None)
    assert metrics["2021"]["ic"] is not None
    trainer.write_record({"id": "trained/lstm_test", "status": "ready", "config": config.__dict__, "name": "lstm test"})

    symbols = broad30_symbols()
    scorer = ModelScorer(ModelScorerSpec(models=["trained/lstm_test"]), store)
    scorer.prepare(symbols, date(2021, 3, 1), date(2021, 3, 31))
    scores = scorer.score(date(2021, 3, 31), symbols).scores.dropna()
    assert len(scores) >= 20

    # 用训练时的整段面板重算同一天的序列，逐只比对
    values, panels, _ = trainer._pool_panels(config, until, lambda _: None)
    close = panels["close"]
    ranked = lstm.ranked_array(values, config.factors, panels["suspended"])
    day = close.index.get_loc(np.datetime64("2021-03-31"))
    columns = [list(close.columns).index(s) for s in scores.index]
    fold = __import__("app.pipeline.model_library", fromlist=["load_folds"]).load_folds("trained/lstm_test")[2021]
    expected = np.mean([p(lstm.sequences_on(ranked, day, columns)) for p in fold.predictors], axis=0)
    np.testing.assert_allclose(scores.to_numpy(), expected, rtol=1e-4, atol=1e-5)
