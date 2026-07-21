"""Promotion criteria for AXIOM's shadow action-policy advisor.

This gate only evaluates whether the trained model is ready to influence
proposal ordering in shadow/safe mode. It does not authorize execution, live
trading, escalation auto-approval, or bypass any PolicyEngine guard.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.axiom_training.action_policy import ACTION_POLICY_REPORT_PATH
from src.axiom_training.policy_advisor import POLICY_ADVICE_PATH
from src.axiom_training.runtime_health import (
    RuntimeHealthReport,
    evaluate_runtime_health,
    runtime_health_to_dict,
)


@dataclass(frozen=True)
class PromotionThresholds:
    min_shadow_cycles: int = 200
    min_agreement_rate: float = 0.65
    max_invalid_rate: float = 0.0
    max_runtime_health_status: str = "HEALTHY"


@dataclass(frozen=True)
class PromotionVerdict:
    verdict: str
    ready: bool
    reasons: List[str]
    metrics: Dict[str, Any]
    thresholds: PromotionThresholds
    runtime_health: RuntimeHealthReport
    scope: str = (
        "shadow proposal-ordering only; no execution authority, no escalation "
        "auto-approval, no broker mutation"
    )
    next_allowed_step: str = "continue_shadow"
    schema_version: str = "2026-07-10-v1"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    try:
        lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return []
    rows: List[Dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _latest_by_cycle(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cycle_id = str(row.get("cycle_id") or "")
        if cycle_id:
            latest[cycle_id] = row
    return latest


def _status_rank(status: str) -> int:
    order = {"HEALTHY": 0, "DEGRADED": 1, "CRITICAL": 2}
    return order.get(str(status).upper(), 2)


def evaluate_promotion(
    *,
    advice_path: Path = POLICY_ADVICE_PATH,
    report_path: Path = ACTION_POLICY_REPORT_PATH,
    thresholds: Optional[PromotionThresholds] = None,
    runtime_health: Optional[RuntimeHealthReport] = None,
) -> PromotionVerdict:
    thresholds = thresholds or PromotionThresholds()
    training_report = _read_json(report_path)
    advice_rows = _read_jsonl(advice_path)
    latest = _latest_by_cycle(advice_rows)
    latest_rows = list(latest.values())
    total = len(latest_rows)
    agreements = sum(1 for row in latest_rows if row.get("status") == "agreement")
    partials = sum(1 for row in latest_rows if row.get("status") == "partial_agreement")
    invalid = sum(1 for row in latest_rows if row.get("status") == "invalid_prediction")
    agreement_rate = agreements / total if total else 0.0
    assisted_agreement_rate = (agreements + partials) / total if total else 0.0
    invalid_rate = invalid / total if total else 0.0
    runtime_health = runtime_health or evaluate_runtime_health()

    reasons: List[str] = []
    if training_report.get("status") != "trained" or not training_report.get("replay_pass"):
        reasons.append("training report is not trained/replay_pass")
    if not training_report.get("stress_pass"):
        reasons.append("action-policy adversarial stress battery did not pass")
    if total < thresholds.min_shadow_cycles:
        reasons.append(f"need {thresholds.min_shadow_cycles} shadow advice cycles, have {total}")
    if agreement_rate < thresholds.min_agreement_rate:
        reasons.append(
            f"agreement_rate {agreement_rate:.1%} < {thresholds.min_agreement_rate:.1%}"
        )
    if invalid_rate > thresholds.max_invalid_rate:
        reasons.append(f"invalid_rate {invalid_rate:.1%} > {thresholds.max_invalid_rate:.1%}")
    if _status_rank(runtime_health.status) > _status_rank(thresholds.max_runtime_health_status):
        reasons.append(f"runtime health is {runtime_health.status}, requires {thresholds.max_runtime_health_status}")

    ready = not reasons
    return PromotionVerdict(
        verdict="READY_FOR_SHADOW_ORDERING" if ready else "BLOCKED",
        ready=ready,
        reasons=reasons,
        metrics={
            "shadow_advice_cycles": total,
            "agreement": agreements,
            "partial_agreement": partials,
            "invalid_prediction": invalid,
            "agreement_rate": round(agreement_rate, 4),
            "assisted_agreement_rate": round(assisted_agreement_rate, 4),
            "invalid_rate": round(invalid_rate, 4),
            "training_status": training_report.get("status"),
            "training_replay_pass": bool(training_report.get("replay_pass")),
            "runtime_consumes_model": bool(training_report.get("runtime_consumes_model")),
        },
        thresholds=thresholds,
        runtime_health=runtime_health,
        next_allowed_step="enable_shadow_ordering_candidate" if ready else "continue_shadow",
    )


def promotion_verdict_to_dict(verdict: PromotionVerdict) -> Dict[str, Any]:
    payload = asdict(verdict)
    payload["runtime_health"] = runtime_health_to_dict(verdict.runtime_health)
    return payload


__all__ = [
    "PromotionThresholds",
    "PromotionVerdict",
    "evaluate_promotion",
    "promotion_verdict_to_dict",
]
