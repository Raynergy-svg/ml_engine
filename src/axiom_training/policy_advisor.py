"""Shadow advisor for AXIOM's trained action-policy model.

The advisor compares a persisted action-policy model against the LLM's current
resident-loop proposal. It never executes an action and never changes the
proposal list; it only writes telemetry for promotion hygiene.
"""
from __future__ import annotations

import json
import hashlib
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from src.agent_runtime.policy import ActionTier
from src.axiom_operator.session import append_jsonl, utc_now
from src.axiom_training.action_policy import (
    ACTION_POLICY_MODEL_PATH,
    ACTION_POLICY_REPORT_PATH,
    ACTION_REGISTRY,
    build_inference_rows,
    features_from_training_row,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AXIOM_DIR = REPO_ROOT / "trained_data" / "axiom"
POLICY_ADVICE_PATH = AXIOM_DIR / "policy_advice.jsonl"
ADVISOR_VERSION = "2026-07-09-v2"


@dataclass(frozen=True)
class PolicySuggestion:
    domain: str
    predicted_action: str
    confidence: Optional[float]
    valid_for_domain: bool
    safety_tier: Optional[str]
    allowed_autonomy: Optional[str]


@dataclass(frozen=True)
class PolicyAdvice:
    advice_id: str
    cycle_id: str
    ts: str
    status: str
    policy_mode: str
    model_path: str
    model_version: str
    report_status: Optional[str]
    replay_pass: bool
    stress_pass: bool
    runtime_consumes_model: bool
    llm_actions: List[str]
    suggestions: List[PolicySuggestion] = field(default_factory=list)
    selected_actions: List[str] = field(default_factory=list)
    agreement_actions: List[str] = field(default_factory=list)
    disagreement_actions: List[str] = field(default_factory=list)
    invalid_predictions: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)
    advisor_version: str = ADVISOR_VERSION
    schema_version: str = "2026-07-09-v1"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_bundle(path: Path) -> Dict[str, Any]:
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("action-policy bundle must be a dict with a 'model' key")
    return payload


def _confidence_for(model: Any, features: List[Dict[str, Any]], predicted: List[str]) -> List[Optional[float]]:
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        return [None for _ in predicted]
    try:
        probas = model.predict_proba(features)
    except Exception:  # noqa: BLE001 - advisor must degrade to prediction-only telemetry
        return [None for _ in predicted]
    classes = [str(cls) for cls in getattr(model, "classes_", [])]
    out: List[Optional[float]] = []
    for pred, row in zip(predicted, probas):
        try:
            idx = classes.index(str(pred))
            out.append(round(float(row[idx]), 6))
        except (ValueError, TypeError, IndexError):
            out.append(None)
    return out


def _tier_for(action: str) -> Optional[str]:
    spec = ACTION_REGISTRY.get(action)
    if spec is not None:
        return spec.safety_tier
    try:
        return ActionTier(action).value
    except ValueError:
        return None


def _suggestions_from_model(bundle: Mapping[str, Any], rows: List[Dict[str, Any]]) -> List[PolicySuggestion]:
    model = bundle.get("model")
    features = [features_from_training_row(row) for row in rows]
    predicted = [str(action) for action in model.predict(features)]
    confidences = _confidence_for(model, features, predicted)
    suggestions: List[PolicySuggestion] = []
    for row, action, confidence in zip(rows, predicted, confidences):
        valid_actions = row.get("valid_actions") if isinstance(row.get("valid_actions"), list) else []
        spec = ACTION_REGISTRY.get(action)
        suggestions.append(PolicySuggestion(
            domain=str(row.get("domain") or "unknown"),
            predicted_action=action,
            confidence=confidence,
            valid_for_domain=action in valid_actions,
            safety_tier=_tier_for(action),
            allowed_autonomy=spec.allowed_autonomy if spec is not None else None,
        ))
    return suggestions


def build_policy_advice(
    cycle: Mapping[str, Any],
    *,
    model_path: Path = ACTION_POLICY_MODEL_PATH,
    report_path: Path = ACTION_POLICY_REPORT_PATH,
) -> PolicyAdvice:
    cycle_id = str(cycle.get("cycle_id") or "")
    report = _read_json(report_path)
    model_version = str(report.get("trained_at") or "untrained")
    version_digest = hashlib.sha256(model_version.encode("utf-8")).hexdigest()[:12]
    llm_actions = [
        str(action.get("action"))
        for action in (cycle.get("proposed_actions") or [])
        if isinstance(action, Mapping) and action.get("action")
    ]

    base = {
        "advice_id": f"axiom-policy-advice-{ADVISOR_VERSION}-{version_digest}-{cycle_id or 'missing-cycle'}",
        "cycle_id": cycle_id,
        "ts": str(cycle.get("ts") or utc_now()),
        "policy_mode": "shadow",
        "model_path": str(model_path),
        "model_version": model_version,
        "report_status": report.get("status"),
        "replay_pass": bool(report.get("replay_pass")),
        "stress_pass": bool(report.get("stress_pass")),
        "runtime_consumes_model": False,
        "llm_actions": llm_actions,
    }
    if not Path(model_path).exists():
        return PolicyAdvice(status="model_missing", detail={"reason": "no trained model artifact"}, **base)
    if report and (
        report.get("status") != "trained"
        or not report.get("replay_pass")
        or not report.get("stress_pass")
    ):
        return PolicyAdvice(status="report_not_ready", detail={"report": report}, **base)

    observations = cycle.get("observations") if isinstance(cycle.get("observations"), Mapping) else {}
    rows = build_inference_rows(
        observations,
        autonomy_enabled=bool(cycle.get("autonomy_enabled")),
        cli_available=bool(cycle.get("cli_available")),
    )
    try:
        bundle = _load_bundle(model_path)
        suggestions = _suggestions_from_model(bundle, rows)
    except Exception as exc:  # noqa: BLE001 - failed model read cannot affect the resident loop
        return PolicyAdvice(status="advisor_error", detail={"error": repr(exc)}, **base)

    valid_predicted_actions = [
        suggestion.predicted_action
        for suggestion in suggestions
        if suggestion.valid_for_domain
    ]
    agreement = sorted(set(valid_predicted_actions) & set(llm_actions))
    disagreement = sorted(set(valid_predicted_actions) - set(llm_actions))
    invalid = sorted({s.predicted_action for s in suggestions if not s.valid_for_domain})
    if invalid and not valid_predicted_actions:
        status = "invalid_prediction"
    elif agreement and not disagreement:
        status = "agreement"
    elif agreement:
        status = "partial_agreement"
    else:
        status = "disagreement"
    return PolicyAdvice(
        status=status,
        suggestions=suggestions,
        selected_actions=sorted(set(valid_predicted_actions)),
        agreement_actions=agreement,
        disagreement_actions=disagreement,
        invalid_predictions=invalid,
        detail={
            "training_source": bundle.get("training_source"),
            "label_counts": bundle.get("label_counts"),
            "domain_counts": bundle.get("domain_counts"),
            "invalid_prediction_count": len(invalid),
            "note": "shadow advisor only; no runtime action selected from this prediction",
        },
        **base,
    )


def _existing_advice_ids(path: Path) -> set[str]:
    try:
        lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return set()
    ids: set[str] = set()
    for line in lines:
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("advice_id"):
            ids.add(str(payload["advice_id"]))
    return ids


def append_policy_advice(advice: PolicyAdvice, *, path: Path = POLICY_ADVICE_PATH) -> bool:
    path = Path(path)
    if advice.advice_id in _existing_advice_ids(path):
        return False
    append_jsonl(path, asdict(advice))
    return True


def advise_and_append(
    cycle: Mapping[str, Any],
    *,
    model_path: Path = ACTION_POLICY_MODEL_PATH,
    report_path: Path = ACTION_POLICY_REPORT_PATH,
    advice_path: Path = POLICY_ADVICE_PATH,
) -> Dict[str, Any]:
    advice = build_policy_advice(cycle, model_path=model_path, report_path=report_path)
    written = append_policy_advice(advice, path=advice_path)
    payload = asdict(advice)
    payload["written"] = written
    return payload


def backfill_policy_advice(
    cycles: List[Mapping[str, Any]],
    *,
    model_path: Path = ACTION_POLICY_MODEL_PATH,
    report_path: Path = ACTION_POLICY_REPORT_PATH,
    advice_path: Path = POLICY_ADVICE_PATH,
) -> Dict[str, Any]:
    statuses: Dict[str, int] = {}
    written = 0
    skipped_existing = 0
    for cycle in cycles:
        advice = advise_and_append(
            cycle,
            model_path=model_path,
            report_path=report_path,
            advice_path=advice_path,
        )
        status = str(advice.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        if advice.get("written"):
            written += 1
        else:
            skipped_existing += 1
    return {
        "cycles_seen": len(cycles),
        "written": written,
        "skipped_existing": skipped_existing,
        "by_status": statuses,
    }


__all__ = [
    "POLICY_ADVICE_PATH",
    "PolicyAdvice",
    "PolicySuggestion",
    "advise_and_append",
    "append_policy_advice",
    "backfill_policy_advice",
    "build_policy_advice",
]
