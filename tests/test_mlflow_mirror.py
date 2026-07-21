"""Reproduce-then-resolve coverage for the MLflow training-session mirror.

Task: add MLflow experiment tracking that mirrors the training metadata
memory_client.py already logs to trained_data/.memory/training_sessions.json,
without changing that existing local logging, and fail-open (an MLflow
error must never break training or the live loop).

Coverage:
  A. mirror_training_session() logs a real, queryable MLflow run (params,
     metrics, tags) to a real local sqlite-backed tracking store — no mocks.
  B. mirror_training_session() fails open: when the tracking directory is
     unwritable (a real disk condition, not a simulated one), it returns
     False and does not raise.
  C. mirror_training_session() is a no-op (returns False, does not raise)
     when MLFLOW_AVAILABLE is False, mirroring the guard already used in
     src/training/enterprise_training.py.
  D. memory_client.MLEngineMemory.log_training() still writes the local
     JSON training_sessions.json exactly as before AND produces a matching
     MLflow run — mirror, not replace.
  E. memory_client.MLEngineMemory.log_training() does not raise even when
     the MLflow mirror is pointed at a broken tracking store — the caller
     (training loop) must survive.
"""
import json

import mlflow
import pytest

from src.training import mlflow_mirror
from memory_client import MLEngineMemory


SAMPLE_SESSION = {
    "pair": "EUR_USD",
    "granularity": "M15",
    "models_trained": ["transformer", "xgboost", "ridge"],
    "metrics": {"val_accuracy": 0.523, "gap": 0.09, "not_a_number": "n/a"},
    "hyperparams": {"epochs": 50, "batch_size": 128, "nested": {"lr": 0.001}},
    "duration_seconds": 123.4,
    "warm_start": True,
    "timestamp": "2026-07-07T12:00:00",
}


def test_mirror_logs_a_real_queryable_run(tmp_path):
    ok = mlflow_mirror.mirror_training_session(
        SAMPLE_SESSION,
        tracking_dir=tmp_path,
        experiment_name="test_mirror_exp",
    )
    assert ok is True

    db_path = tmp_path / "mlflow.db"
    assert db_path.exists()

    mlflow.set_tracking_uri(f"sqlite:///{db_path.absolute()}")
    experiment = mlflow.get_experiment_by_name("test_mirror_exp")
    assert experiment is not None

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    row = runs.iloc[0]

    assert row["tags.pair"] == "EUR_USD"
    assert row["tags.granularity"] == "M15"
    assert row["tags.source"] == "memory_client"
    assert row["params.warm_start"] == "True"
    assert row["params.models_trained"] == "transformer,xgboost,ridge"
    assert row["params.epochs"] == "50"
    assert row["params.nested.lr"] == "0.001"
    assert float(row["metrics.val_accuracy"]) == pytest.approx(0.523)
    assert float(row["metrics.gap"]) == pytest.approx(0.09)
    # Non-numeric metric values are skipped, not crashed on.
    assert "metrics.not_a_number" not in runs.columns


def test_mirror_fails_open_on_unwritable_tracking_dir(tmp_path):
    # Real disk failure: put a FILE where the tracking dir needs to be a
    # directory, so Path.mkdir(parents=True, exist_ok=True) raises for real.
    blocked_path = tmp_path / "blocked"
    blocked_path.write_text("not a directory")

    ok = mlflow_mirror.mirror_training_session(
        SAMPLE_SESSION,
        tracking_dir=blocked_path,
        experiment_name="test_mirror_exp_fail",
    )

    assert ok is False  # must not raise


def test_mirror_noop_when_mlflow_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(mlflow_mirror, "MLFLOW_AVAILABLE", False)

    ok = mlflow_mirror.mirror_training_session(
        SAMPLE_SESSION,
        tracking_dir=tmp_path,
        experiment_name="test_mirror_exp_unavailable",
    )

    assert ok is False
    # Confirm it really did nothing — no db file created.
    assert not (tmp_path / "mlflow.db").exists()


def test_log_training_mirrors_without_changing_local_json(tmp_path, monkeypatch):
    mlflow_dir = tmp_path / "mlruns"
    monkeypatch.setattr(mlflow_mirror, "DEFAULT_TRACKING_DIR", mlflow_dir)

    memory = MLEngineMemory(storage_path=tmp_path / "memory")
    memory.log_training(
        "GBP_JPY",
        {"accuracy": 0.61},
        granularity="H1",
        models_trained=["transformer"],
        hyperparams={"epochs": 10},
        duration_seconds=5.0,
        warm_start=False,
    )

    # Existing local JSON logging is unchanged.
    sessions = json.loads((tmp_path / "memory" / "training_sessions.json").read_text())
    assert len(sessions) == 1
    assert sessions[0]["pair"] == "GBP_JPY"
    assert sessions[0]["metrics"] == {"accuracy": 0.61}

    # And it now also shows up in MLflow.
    mlflow.set_tracking_uri(f"sqlite:///{(mlflow_dir / 'mlflow.db').absolute()}")
    experiment = mlflow.get_experiment_by_name(mlflow_mirror.DEFAULT_EXPERIMENT)
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["tags.pair"] == "GBP_JPY"


def test_log_training_survives_broken_mlflow_mirror(tmp_path, monkeypatch):
    # Point the mirror at a location that will fail (file where a dir is
    # expected) and confirm log_training() still completes and still writes
    # the local JSON record — the training loop must never break.
    blocked_dir = tmp_path / "blocked_mlruns"
    blocked_dir.write_text("not a directory")
    monkeypatch.setattr(mlflow_mirror, "DEFAULT_TRACKING_DIR", blocked_dir)

    memory = MLEngineMemory(storage_path=tmp_path / "memory2")
    memory.log_training("USD_JPY", {"accuracy": 0.5})  # must not raise

    sessions = json.loads((tmp_path / "memory2" / "training_sessions.json").read_text())
    assert len(sessions) == 1
    assert sessions[0]["pair"] == "USD_JPY"
