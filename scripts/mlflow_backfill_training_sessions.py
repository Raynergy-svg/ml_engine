#!/usr/bin/env python3
"""Backfill historical memory_client training sessions into MLflow.

Optional, idempotent, offline-only. Reads every locally logged training
session from ``trained_data/.memory/training_sessions.json``
(memory_client.py's ``log_training()``) and mirrors any that are not
already present in the local MLflow tracking store
(``trained_data/mlruns/``) so training history that predates this feature
is visible via ``mlflow ui --backend-store-uri sqlite:///trained_data/mlruns/mlflow.db``.

Safe to re-run: existing runs are detected by run name and skipped.
Does not touch trading, execution, halt, or config state.

Usage:
    python scripts/mlflow_backfill_training_sessions.py [--dry-run]
"""
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from memory_client import MLEngineMemory  # noqa: E402
from src.training.mlflow_mirror import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    DEFAULT_TRACKING_DIR,
    MLFLOW_AVAILABLE,
    mirror_training_session,
)


def _run_name_for(session: dict) -> str:
    pair = str(session.get("pair", "unknown"))
    timestamp = str(session.get("timestamp", ""))
    return f"{pair}_{timestamp}" if timestamp else pair


def _existing_run_names(experiment_name: str) -> set:
    """All run names already present in the target experiment.

    Compared client-side in Python rather than via MLflow's
    ``filter_string`` query: that query's quote-escaping does not reliably
    match run names containing an apostrophe (verified against the pinned
    mlflow==3.8.1), which would silently defeat the duplicate check for
    such names and re-mirror them on every backfill run.
    """
    import mlflow

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return set()
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=50000)
    if runs.empty or "tags.mlflow.runName" not in runs.columns:
        return set()
    return set(runs["tags.mlflow.runName"].dropna())


def backfill(*, dry_run: bool = False) -> dict:
    """Mirror every not-yet-mirrored local training session into MLflow.

    Returns a summary dict: {mirrored, skipped, failed, total}.
    """
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(DEFAULT_TRACKING_DIR / 'mlflow.db').absolute()}")

    memory = MLEngineMemory()
    sessions = memory.get_all_training_sessions()
    logger.info(f"Found {len(sessions)} locally logged training sessions")

    existing_names = _existing_run_names(DEFAULT_EXPERIMENT)

    mirrored, skipped, failed = 0, 0, 0
    for session in sessions:
        run_name = _run_name_for(session)

        if run_name in existing_names:
            skipped += 1
            continue

        if dry_run:
            logger.info(f"[dry-run] would mirror: {run_name}")
            mirrored += 1
            continue

        if mirror_training_session(session):
            mirrored += 1
            existing_names.add(run_name)  # guard against duplicate pair+timestamp within this same batch
        else:
            failed += 1

    summary = {"mirrored": mirrored, "skipped": skipped, "failed": failed, "total": len(sessions)}
    logger.info(f"Backfill complete: {summary}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be mirrored without writing to MLflow",
    )
    args = parser.parse_args()

    if not MLFLOW_AVAILABLE:
        logger.error("mlflow is not installed; nothing to backfill.")
        return 1

    backfill(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
