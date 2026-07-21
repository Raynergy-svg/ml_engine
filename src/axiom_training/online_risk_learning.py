"""Automatic risk-policy retraining after each newly processed trade outcome."""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.axiom_training.risk_gym import (
    RISK_POLICY_MODEL_PATH,
    RISK_POLICY_REPORT_PATH,
)
from src.axiom_training.risk_runtime import promote_candidate


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_REL = Path("trained_data/axiom/online_risk_learning.json")
HISTORY_REL = Path("trained_data/axiom/online_risk_learning.jsonl")
LOCK_REL = Path("trained_data/axiom/online_risk_learning.lock")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")


class OnlineRiskLearner:
    """Retrain on new feedback and atomically replace only passing champions."""

    def __init__(
        self,
        *,
        root: Path = REPO_ROOT,
        trainer: Callable[[], tuple[dict[str, Any], dict[str, Any] | None]] | None = None,
        promoter: Callable[..., dict[str, Any]] = promote_candidate,
    ) -> None:
        self.root = Path(root)
        self._trainer = trainer
        self._promoter = promoter

    def _train(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if self._trainer is not None:
            return self._trainer()
        from scripts.train_axiom_risk_policy import train_and_evaluate

        return train_and_evaluate()

    def run(self, *, trade_id: str, environment: str = "practice") -> dict[str, Any]:
        """Run one serialized online update. Never raises to the trade-close path."""
        started_at = _utc_now()
        event: dict[str, Any] = {
            "started_at": started_at,
            "trade_id": str(trade_id),
            "environment": environment,
            "status": "started",
            "auto_retrain": True,
            "auto_promote_passing_candidate": True,
        }
        lock_path = self.root / LOCK_REL
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("a+")
        try:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                event.update({"status": "skipped", "reason": "training_already_running"})
                return self._record(event)
            if environment != "practice":
                event.update({"status": "refused", "reason": "practice_environment_required"})
                return self._record(event)

            report, model = self._train()
            report_path = self.root / RISK_POLICY_REPORT_PATH.relative_to(REPO_ROOT)
            model_path = self.root / RISK_POLICY_MODEL_PATH.relative_to(REPO_ROOT)
            if model is not None:
                _atomic_json(model_path, model)
            # Best-effort W&B lineage per online cycle (offline unless WANDB_MODE
            # says otherwise). Telemetry only — a W&B failure never touches the
            # training result, and the status lands inside the persisted report.
            try:
                from scripts.train_axiom_risk_policy import log_report_to_wandb

                wandb_status = log_report_to_wandb(
                    report, model_path, run_name=f"online-trade-{trade_id}",
                )
            except Exception as exc:  # noqa: BLE001 - lineage must never block learning
                wandb_status = {"available": False, "error": repr(exc)}
                report["wandb"] = wandb_status
            event["wandb"] = wandb_status
            _atomic_json(report_path, report)

            event.update({
                "dataset_rows": report.get("dataset_rows"),
                "train_rows": report.get("train_rows"),
                "validation_rows": report.get("validation_rows"),
                "holdout_rows": report.get("holdout_rows"),
                "candidates_tested": report.get("candidates_tested"),
                "candidate_status": report.get("status"),
                "candidate_passed": report.get("passing_model") is True,
                "failure_reasons": report.get("reasons") or [],
            })
            if model is None or report.get("passing_model") is not True:
                event.update({
                    "status": "candidate_rejected",
                    "champion_replaced": False,
                    "reason": "verification_failed",
                })
                return self._record(event)

            promotion = self._promoter(
                root=self.root,
                environment="practice",
                actor="axiom_online_learning",
            )
            event.update({
                "status": "trained_and_promoted",
                "champion_replaced": True,
                "champion_model_sha256": promotion.get("model_sha256"),
                "promoted_at": promotion.get("promoted_at"),
            })
            return self._record(event)
        except Exception as exc:  # noqa: BLE001 - never break close processing
            event.update({
                "status": "error",
                "champion_replaced": False,
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return self._record(event)
        finally:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            lock_handle.close()

    def _record(self, event: dict[str, Any]) -> dict[str, Any]:
        event["completed_at"] = _utc_now()
        _atomic_json(self.root / STATE_REL, event)
        _append_jsonl(self.root / HISTORY_REL, event)
        return event


def read_online_learning_status(*, root: Path = REPO_ROOT) -> dict[str, Any]:
    path = Path(root) / STATE_REL
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {
            "status": "waiting_for_next_closed_trade",
            "auto_retrain": True,
            "auto_promote_passing_candidate": True,
        }
    return value if isinstance(value, dict) else {"status": "unavailable"}
