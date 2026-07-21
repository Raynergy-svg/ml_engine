"""Immutable promotion boundary for AXIOM's shadow action-policy model."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.axiom_training.action_policy import (
    ACTION_POLICY_MODEL_PATH,
    ACTION_POLICY_REPORT_PATH,
)
from src.axiom_training.policy_advisor import POLICY_ADVICE_PATH
from src.axiom_training.promotion_criteria import (
    PromotionVerdict,
    evaluate_promotion,
    promotion_verdict_to_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_MODEL_REL = ACTION_POLICY_MODEL_PATH.relative_to(REPO_ROOT)
CANDIDATE_REPORT_REL = ACTION_POLICY_REPORT_PATH.relative_to(REPO_ROOT)
CHAMPION_MODEL_REL = Path("trained_data/axiom/action_policy_champion.pkl")
CHAMPION_REPORT_REL = Path("trained_data/axiom/action_policy_champion_report.json")
PROMOTION_REL = Path("trained_data/axiom/action_policy_promotion.json")
AUDIT_REL = Path("trained_data/axiom/action_policy_promotion_audit.jsonl")


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n").encode(),
    )


def _append_audit(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")


def _last_audit(path: Path) -> dict[str, Any]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def validate_action_candidate(*, root: Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(root)
    model_path = root / CANDIDATE_MODEL_REL
    report_path = root / CANDIDATE_REPORT_REL
    report = _read_json(report_path)
    reasons: list[str] = []
    bundle: dict[str, Any] = {}
    try:
        with model_path.open("rb") as handle:
            value = pickle.load(handle)  # noqa: S301 - trusted local candidate artifact
        if isinstance(value, dict):
            bundle = value
        else:
            reasons.append("candidate_bundle_invalid")
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
        reasons.append("candidate_model_missing_or_invalid")
    if report.get("status") != "trained" or report.get("replay_pass") is not True:
        reasons.append("training_replay_not_passed")
    stress = report.get("adversarial_stress")
    if report.get("stress_pass") is not True or not isinstance(stress, dict) or stress.get("passed") is not True:
        reasons.append("adversarial_stress_not_passed")
    if bundle and bundle.get("trained_at") != report.get("trained_at"):
        reasons.append("model_report_training_timestamp_mismatch")
    if bundle and bundle.get("stress_pass") is not True:
        reasons.append("model_stress_attestation_missing")
    if bundle and bundle.get("runtime_consumes_model") is not False:
        reasons.append("execution_authority_boundary_invalid")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "model_sha256": _sha256(model_path),
        "report_sha256": _sha256(report_path),
        "report": report,
    }


def promote_action_candidate(
    *,
    root: Path = REPO_ROOT,
    actor: str,
    promotion_verdict: PromotionVerdict | None = None,
) -> dict[str, Any]:
    """Promote only to shadow proposal ordering; never grant execution authority."""
    root = Path(root)
    ts = datetime.now(timezone.utc).isoformat()
    validation = validate_action_candidate(root=root)
    verdict = promotion_verdict or evaluate_promotion(
        advice_path=root / POLICY_ADVICE_PATH.relative_to(REPO_ROOT),
        report_path=root / CANDIDATE_REPORT_REL,
    )
    reasons = [*validation["reasons"], *verdict.reasons]
    if reasons:
        event = {
            "ts": ts,
            "event": "promotion_refused",
            "actor": actor,
            "reasons": list(dict.fromkeys(reasons)),
            "candidate_model_sha256": validation["model_sha256"],
        }
        _append_audit(root / AUDIT_REL, event)
        raise ValueError("action candidate promotion refused: " + "; ".join(event["reasons"]))

    model_bytes = (root / CANDIDATE_MODEL_REL).read_bytes()
    report_bytes = (root / CANDIDATE_REPORT_REL).read_bytes()
    _atomic_bytes(root / CHAMPION_MODEL_REL, model_bytes)
    _atomic_bytes(root / CHAMPION_REPORT_REL, report_bytes)
    state = {
        "schema_version": "axiom-action-promotion-v1",
        "enabled": True,
        "mode": "shadow_ordering",
        "actor": actor,
        "promoted_at": ts,
        "model_path": str(CHAMPION_MODEL_REL),
        "report_path": str(CHAMPION_REPORT_REL),
        "model_sha256": _sha256(root / CHAMPION_MODEL_REL),
        "report_sha256": _sha256(root / CHAMPION_REPORT_REL),
        "candidate_model_sha256": validation["model_sha256"],
        "candidate_report_sha256": validation["report_sha256"],
        "runtime_execution_authority": False,
        "proposal_auto_approval": False,
        "broker_mutation": False,
        "stress_battery_version": (validation["report"].get("adversarial_stress") or {}).get("battery_version"),
        "promotion_verdict": promotion_verdict_to_dict(verdict),
    }
    _atomic_json(root / PROMOTION_REL, state)
    _append_audit(root / AUDIT_REL, {"ts": ts, "event": "promoted", **state})
    return state


def read_action_promotion_status(*, root: Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(root)
    state = _read_json(root / PROMOTION_REL)
    latest_event = _last_audit(root / AUDIT_REL)
    if not state:
        return {
            "enabled": False,
            "valid": False,
            "mode": "candidate_shadow_evaluation",
            "reasons": latest_event.get("reasons") or ["promotion_not_enabled"],
            "latest_event": latest_event,
        }
    reasons: list[str] = []
    if state.get("enabled") is not True:
        reasons.append("promotion_not_enabled")
    if state.get("mode") != "shadow_ordering":
        reasons.append("promotion_mode_invalid")
    if state.get("runtime_execution_authority") is not False:
        reasons.append("execution_authority_boundary_invalid")
    model_path = root / CHAMPION_MODEL_REL
    report_path = root / CHAMPION_REPORT_REL
    if state.get("model_sha256") != _sha256(model_path):
        reasons.append("champion_model_hash_mismatch")
    if state.get("report_sha256") != _sha256(report_path):
        reasons.append("champion_report_hash_mismatch")
    return {
        **state,
        "valid": not reasons,
        "reasons": reasons,
        "latest_event": latest_event,
    }
