"""Coverage for scripts/mlflow_backfill_training_sessions.py.

Reproduce-then-resolve for a bug an independent verifier found in the first
draft: MLflow's `filter_string` query does not reliably match run names
containing an apostrophe (verified against the pinned mlflow==3.8.1), which
would silently defeat the duplicate-run check and re-mirror such sessions
on every backfill run. The fix compares run names client-side in Python
instead of relying on MLflow's filter-string escaping.
"""
import json
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.training import mlflow_mirror  # noqa: E402
from memory_client import MLEngineMemory  # noqa: E402
import mlflow_backfill_training_sessions as backfill_mod  # noqa: E402


def _isolate_mlflow(monkeypatch, mlflow_dir):
    monkeypatch.setattr(mlflow_mirror, "DEFAULT_TRACKING_DIR", mlflow_dir)
    monkeypatch.setattr(backfill_mod, "DEFAULT_TRACKING_DIR", mlflow_dir)


def test_backfill_mirrors_sessions_not_yet_in_mlflow(tmp_path, monkeypatch):
    mlflow_dir = tmp_path / "mlruns"
    _isolate_mlflow(monkeypatch, mlflow_dir)

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    sessions = [
        {
            "pair": "EUR_USD", "granularity": "M15", "models_trained": ["transformer"],
            "metrics": {"val_accuracy": 0.5}, "hyperparams": {}, "duration_seconds": 1.0,
            "warm_start": False, "timestamp": "2026-01-01T00:00:00",
        },
        {
            "pair": "GBP_USD", "granularity": "H1", "models_trained": ["transformer"],
            "metrics": {"val_accuracy": 0.52}, "hyperparams": {}, "duration_seconds": 1.0,
            "warm_start": False, "timestamp": "2026-01-02T00:00:00",
        },
    ]
    (memory_dir / "training_sessions.json").write_text(json.dumps(sessions))

    monkeypatch.setattr(backfill_mod, "MLEngineMemory", lambda: MLEngineMemory(storage_path=memory_dir))

    summary = backfill_mod.backfill(dry_run=False)
    assert summary == {"mirrored": 2, "skipped": 0, "failed": 0, "total": 2}

    mlflow.set_tracking_uri(f"sqlite:///{(mlflow_dir / 'mlflow.db').absolute()}")
    experiment = mlflow.get_experiment_by_name(mlflow_mirror.DEFAULT_EXPERIMENT)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 2


def test_backfill_is_idempotent_on_rerun(tmp_path, monkeypatch):
    mlflow_dir = tmp_path / "mlruns"
    _isolate_mlflow(monkeypatch, mlflow_dir)

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    sessions = [{
        "pair": "USD_JPY", "granularity": "M15", "models_trained": ["transformer"],
        "metrics": {"val_accuracy": 0.51}, "hyperparams": {}, "duration_seconds": 1.0,
        "warm_start": False, "timestamp": "2026-01-03T00:00:00",
    }]
    (memory_dir / "training_sessions.json").write_text(json.dumps(sessions))
    monkeypatch.setattr(backfill_mod, "MLEngineMemory", lambda: MLEngineMemory(storage_path=memory_dir))

    first = backfill_mod.backfill(dry_run=False)
    second = backfill_mod.backfill(dry_run=False)

    assert first == {"mirrored": 1, "skipped": 0, "failed": 0, "total": 1}
    assert second == {"mirrored": 0, "skipped": 1, "failed": 0, "total": 1}

    mlflow.set_tracking_uri(f"sqlite:///{(mlflow_dir / 'mlflow.db').absolute()}")
    experiment = mlflow.get_experiment_by_name(mlflow_mirror.DEFAULT_EXPERIMENT)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1  # no duplicate created on rerun


def test_backfill_dedup_survives_apostrophe_in_run_name(tmp_path, monkeypatch):
    # Regression test: the pair name here deliberately contains a single
    # quote to exercise the exact MLflow filter_string escaping failure an
    # independent verifier found in the first draft's _already_mirrored().
    mlflow_dir = tmp_path / "mlruns"
    _isolate_mlflow(monkeypatch, mlflow_dir)

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    sessions = [{
        "pair": "O'BRIEN_FX", "granularity": "M15", "models_trained": ["transformer"],
        "metrics": {"val_accuracy": 0.5}, "hyperparams": {}, "duration_seconds": 1.0,
        "warm_start": False, "timestamp": "2026-01-04T00:00:00",
    }]
    (memory_dir / "training_sessions.json").write_text(json.dumps(sessions))
    monkeypatch.setattr(backfill_mod, "MLEngineMemory", lambda: MLEngineMemory(storage_path=memory_dir))

    first = backfill_mod.backfill(dry_run=False)
    second = backfill_mod.backfill(dry_run=False)

    assert first["mirrored"] == 1
    assert second["skipped"] == 1  # must be detected as already-mirrored, not re-created

    mlflow.set_tracking_uri(f"sqlite:///{(mlflow_dir / 'mlflow.db').absolute()}")
    experiment = mlflow.get_experiment_by_name(mlflow_mirror.DEFAULT_EXPERIMENT)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1  # exactly one run, no apostrophe-escaping duplicate
