"""AXIOM action-policy dataset and registry.

The trained model this module supports is an operator-policy model: given
dashboard/tool observations, rank or classify the next safe Axiom action. It is
not a broker execution model. Runtime safety remains owned by
``src.agent_runtime.policy`` and the dashboard control layer.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
AXIOM_DIR = REPO_ROOT / "trained_data" / "axiom"
ACTION_POLICY_EXAMPLES_PATH = AXIOM_DIR / "action_policy_examples.jsonl"
ACTION_POLICY_MODEL_PATH = AXIOM_DIR / "action_policy_model.pkl"
ACTION_POLICY_REPORT_PATH = AXIOM_DIR / "action_policy_report.json"


@dataclass(frozen=True)
class ActionDefinition:
    action: str
    domain: str
    safety_tier: str
    description: str
    expected_effect: str
    allowed_autonomy: str


ACTION_REGISTRY: Dict[str, ActionDefinition] = {
    "observe": ActionDefinition(
        "observe", "operations", "operational",
        "Read current state and wait for more evidence.",
        "No mutation; refreshes situational awareness.", "auto",
    ),
    "recheck": ActionDefinition(
        "recheck", "operations", "operational",
        "Re-read state after an incident or stale condition.",
        "Confirms whether an incident remains live.", "auto",
    ),
    "write_learning": ActionDefinition(
        "write_learning", "operations", "operational",
        "Persist a compact lesson or working-memory update.",
        "Improves future recall without changing trading state.", "bounded",
    ),
    "acknowledge_resolved_alert": ActionDefinition(
        "acknowledge_resolved_alert", "operations", "operational",
        "Acknowledge an alert only after evidence shows it is resolved.",
        "Removes stale operator noise.", "bounded",
    ),
    "halt_lane": ActionDefinition(
        "halt_lane", "operations", "deescalation",
        "Halt a named lane or global trading switch.",
        "Risk-decreasing stop of future actions.", "bounded",
    ),
    "stop_loop": ActionDefinition(
        "stop_loop", "operations", "deescalation",
        "Stop a whitelisted stale or runaway loop.",
        "Contains process-level runaway behavior.", "bounded",
    ),
    "request_bucket_trim": ActionDefinition(
        "request_bucket_trim", "hedge_layer", "escalation",
        "Ask the operator to approve a precise exposure-reducing trim.",
        "No broker mutation; creates a human-required trim proposal.", "proposal_only",
    ),
    "propose_hedge": ActionDefinition(
        "propose_hedge", "hedge_layer", "escalation",
        "Surface a hedge candidate with cost/risk evidence.",
        "No broker mutation; routes hedge decision to operator review.", "proposal_only",
    ),
    "reject_hedge": ActionDefinition(
        "reject_hedge", "hedge_layer", "operational",
        "Reject a hedge candidate with unresolved cost or exposure data.",
        "Keeps raw execution blocked or analysis-only.", "auto",
    ),
    "keep_shadow": ActionDefinition(
        "keep_shadow", "crypto", "operational",
        "Keep a research lane in shadow mode until evidence improves.",
        "Prevents premature promotion.", "auto",
    ),
    "retire_signal": ActionDefinition(
        "retire_signal", "crypto", "escalation",
        "Propose retiring a weak research signal.",
        "Routes a strategy-lifecycle decision to the operator.", "proposal_only",
    ),
    "request_wandb_retrain": ActionDefinition(
        "request_wandb_retrain", "wandb_training", "escalation",
        "Request a W&B-backed retrain for a stale or degraded model family.",
        "Creates a training job/proposal; does not promote artifacts.", "proposal_only",
    ),
    "reject_model_candidate": ActionDefinition(
        "reject_model_candidate", "wandb_training", "operational",
        "Reject a candidate model that fails replay or holdout gates.",
        "Keeps champion unchanged.", "bounded",
    ),
    "promote_model_candidate": ActionDefinition(
        "promote_model_candidate", "wandb_training", "escalation",
        "Promote a model candidate after ship-gate and replay pass.",
        "Changes live model selection; operator approval required.", "proposal_only",
    ),
    "self_improve_add_lessons_recall_row": ActionDefinition(
        "self_improve_add_lessons_recall_row", "self_improve", "self_improve",
        "Add a bounded recall row to the local lessons memory.",
        "Improves operator-loop recall through a gated edit.", "bounded",
    ),
    "self_improve_purge_agent_weight_keys": ActionDefinition(
        "self_improve_purge_agent_weight_keys", "self_improve", "self_improve",
        "Purge invalid agent-weight metadata keys through the self-improve gate.",
        "Removes telemetry pollution after tests and safety gates pass.", "bounded",
    ),
    "escalate_to_operator": ActionDefinition(
        "escalate_to_operator", "operations", "escalation",
        "Ask the operator for a decision when evidence or authority is insufficient.",
        "No mutation; preserves safety boundary.", "proposal_only",
    ),
}


@dataclass(frozen=True)
class ActionPolicyExample:
    ts: str
    cycle_id: str
    actor: str
    domain: str
    observation: Dict[str, Any]
    valid_actions: List[str]
    chosen_action: str
    safety_tier: Optional[str]
    executed: bool
    shadow: bool
    proposal: bool
    denied: bool
    outcome_label: str
    reward: float
    rationale: str
    source: str = "resident_loop"


class ConstrainedActionPolicy:
    """Classifier wrapper that makes the domain allowlist part of inference.

    The base estimator still supplies relative probabilities. Predictions are
    selected only from labels explicitly present in ``valid_actions`` features;
    if the training label set has no allowed class, the universally safe
    ``observe`` action is returned.
    """

    def __init__(self, base_model: Any):
        self.base_model = base_model

    @property
    def classes_(self):
        return self.base_model.classes_

    def predict_proba(self, features):
        return self.base_model.predict_proba(features)

    def predict(self, features):
        probabilities = self.base_model.predict_proba(features)
        classes = [str(value) for value in self.classes_]
        predictions = []
        for row, row_probabilities in zip(features, probabilities):
            allowed = {
                action for action in ACTION_REGISTRY
                if bool(row.get(f"valid_actions.has:{action}"))
            }
            candidates = [
                (float(probability), action)
                for action, probability in zip(classes, row_probabilities)
                if action in allowed
            ]
            predictions.append(max(candidates)[1] if candidates else "observe")
        return predictions


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return len(rows)


def infer_domains(observations: Mapping[str, Any]) -> List[str]:
    domains = {"operations"}
    keys = set(observations)
    if {"oanda_account_state", "agent_weights", "trade_feedback"} & keys:
        domains.add("fx_multi_trader")
    if "hedge_layer" in keys or "risk_trim" in keys:
        domains.add("hedge_layer")
    if any(k.startswith("shadow_ledger:crypto") for k in keys):
        domains.add("crypto")
    if any(k.startswith("shadow_ledger:track_b") for k in keys):
        domains.add("filing_text")
    if "wandb_training" in keys or "learning_loop" in keys:
        domains.add("wandb_training")
    if "learnings" in keys or "agent_weights" in keys:
        domains.add("self_improve")
    return sorted(domains)


def valid_actions_for_domain(domain: str) -> List[str]:
    actions = [
        action for action, spec in ACTION_REGISTRY.items()
        if spec.domain == domain or spec.domain == "operations"
    ]
    return sorted(actions)


def _safe_get(mapping: Mapping[str, Any], *path: str) -> Any:
    value: Any = mapping
    for part in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def summarize_observations(
    observations: Mapping[str, Any], *, autonomy_enabled: bool, cli_available: bool,
) -> Dict[str, Any]:
    account = _safe_get(observations, "oanda_account_state", "data") or {}
    health = _safe_get(observations, "health_check", "data") or {}
    gate = _safe_get(observations, "gate_health", "data") or {}
    lanes = _safe_get(observations, "lane_halt_status", "data", "lanes") or {}
    trade_feedback = _safe_get(observations, "trade_feedback", "data") or {}
    shadow_keys = sorted(k for k in observations if k.startswith("shadow_ledger:"))
    return {
        "observation_keys": sorted(observations.keys()),
        "domains": infer_domains(observations),
        "autonomy_enabled": bool(autonomy_enabled),
        "cli_available": bool(cli_available),
        "gate": gate.get("gate"),
        "gate_hard_no_ok": gate.get("hard_no_ok"),
        "lanes_running": health.get("running"),
        "oanda_trend_proc": health.get("oanda_trend_proc"),
        "tier7_pid_alive": health.get("tier7_pid_alive"),
        "global_halted": _safe_get(observations, "lane_halt_status", "data", "global_halted"),
        "halted_lanes": sorted([lane for lane, halted in lanes.items() if halted]),
        "account_connected": account.get("connected"),
        "account_stale": account.get("stale"),
        "open_trade_count": account.get("open_trade_count"),
        "drawdown_pct": account.get("drawdown_pct"),
        "trade_feedback_count": trade_feedback.get("count"),
        "shadow_lanes_seen": [k.split(":", 1)[1] for k in shadow_keys],
    }


def _flatten_features(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _flatten_features(f"{prefix}.{key}" if prefix else str(key), child, out)
        return
    if isinstance(value, list):
        out[f"{prefix}.count"] = len(value)
        for item in value[:12]:
            out[f"{prefix}.has:{item}"] = True
        return
    if isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix] = value


def features_from_training_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert an action-policy training/inference row into model features."""
    out: Dict[str, Any] = {
        "domain": row.get("domain"),
        "safety_tier": row.get("safety_tier"),
        "executed": bool(row.get("executed")),
        "shadow": bool(row.get("shadow")),
        "proposal": bool(row.get("proposal")),
        "denied": bool(row.get("denied")),
    }
    observation = row.get("observation") if isinstance(row.get("observation"), Mapping) else {}
    _flatten_features("observation", observation, out)
    valid = row.get("valid_actions") if isinstance(row.get("valid_actions"), list) else []
    out["valid_actions.count"] = len(valid)
    for action in valid:
        out[f"valid_actions.has:{action}"] = True
    return out


def build_inference_rows(
    observations: Mapping[str, Any], *, autonomy_enabled: bool, cli_available: bool,
) -> List[Dict[str, Any]]:
    """Build shadow-inference rows from live observations.

    These rows deliberately contain no chosen action. They mirror the feature
    shape used during training so a persisted model can score them in shadow.
    """
    base_observation = summarize_observations(
        observations,
        autonomy_enabled=autonomy_enabled,
        cli_available=cli_available,
    )
    rows: List[Dict[str, Any]] = []
    for domain in infer_domains(observations):
        rows.append({
            "domain": domain,
            "observation": base_observation,
            "valid_actions": valid_actions_for_domain(domain),
            "safety_tier": None,
            "executed": False,
            "shadow": False,
            "proposal": False,
            "denied": False,
        })
    return rows or [{
        "domain": "operations",
        "observation": base_observation,
        "valid_actions": valid_actions_for_domain("operations"),
        "safety_tier": None,
        "executed": False,
        "shadow": False,
        "proposal": False,
        "denied": False,
    }]


def _action_domain(action: str, observations: Mapping[str, Any]) -> str:
    spec = ACTION_REGISTRY.get(action)
    if spec is not None:
        return spec.domain
    if action.startswith("self_improve_"):
        return "self_improve"
    domains = infer_domains(observations)
    if "hedge_layer" in domains and ("trim" in action or "hedge" in action):
        return "hedge_layer"
    if "wandb_training" in domains and ("train" in action or "model" in action):
        return "wandb_training"
    return domains[0] if domains else "operations"


def label_outcome(outcome: Mapping[str, Any], *, verified: bool) -> tuple[str, float]:
    if outcome.get("denied"):
        return "invalid_action", -1.0
    if outcome.get("proposal"):
        return ("good_escalation", 0.5) if verified else ("unsafe_proposal", -1.0)
    if outcome.get("shadow"):
        return "safe_but_incomplete", 0.2
    if outcome.get("executed") and verified:
        return "safe_correct", 1.0
    if outcome.get("executed") and not verified:
        return "unsafe_proposal", -1.0
    return "missed_required_action", -0.5


def build_examples_from_cycle(cycle: Mapping[str, Any]) -> List[ActionPolicyExample]:
    observations = cycle.get("observations") if isinstance(cycle.get("observations"), Mapping) else {}
    proposed = cycle.get("proposed_actions") if isinstance(cycle.get("proposed_actions"), list) else []
    outcomes = cycle.get("outcomes") if isinstance(cycle.get("outcomes"), list) else []
    proposed_by_action = {
        str(item.get("action")): item
        for item in proposed
        if isinstance(item, Mapping) and item.get("action")
    }
    verified = bool(cycle.get("verified"))
    base_observation = summarize_observations(
        observations,
        autonomy_enabled=bool(cycle.get("autonomy_enabled")),
        cli_available=bool(cycle.get("cli_available")),
    )
    rows: List[ActionPolicyExample] = []
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            continue
        action = str(outcome.get("action") or "")
        if not action:
            continue
        domain = _action_domain(action, observations)
        label, reward = label_outcome(outcome, verified=verified)
        proposed_action = proposed_by_action.get(action, {})
        rows.append(ActionPolicyExample(
            ts=str(cycle.get("ts") or utc_now()),
            cycle_id=str(cycle.get("cycle_id") or ""),
            actor=str(cycle.get("actor") or ""),
            domain=domain,
            observation=base_observation,
            valid_actions=valid_actions_for_domain(domain),
            chosen_action=action,
            safety_tier=outcome.get("tier"),
            executed=bool(outcome.get("executed")),
            shadow=bool(outcome.get("shadow")),
            proposal=bool(outcome.get("proposal")),
            denied=bool(outcome.get("denied")),
            outcome_label=label,
            reward=reward,
            rationale=str(proposed_action.get("rationale") or ""),
        ))
    if not rows:
        domain = "operations"
        rows.append(ActionPolicyExample(
            ts=str(cycle.get("ts") or utc_now()),
            cycle_id=str(cycle.get("cycle_id") or ""),
            actor=str(cycle.get("actor") or ""),
            domain=domain,
            observation=base_observation,
            valid_actions=valid_actions_for_domain(domain),
            chosen_action="observe",
            safety_tier="operational",
            executed=False,
            shadow=False,
            proposal=False,
            denied=False,
            outcome_label="safe_correct" if verified else "safe_but_incomplete",
            reward=0.5 if verified else 0.2,
            rationale="No action proposed; observation cycle only.",
        ))
    return rows


def append_cycle_examples(cycle: Mapping[str, Any], *, path: Path = ACTION_POLICY_EXAMPLES_PATH) -> int:
    examples = build_examples_from_cycle(cycle)
    return _append_jsonl(path, [asdict(example) for example in examples])


def read_examples(path: Path = ACTION_POLICY_EXAMPLES_PATH, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return []
    if limit is not None:
        lines = lines[-int(limit):]
    rows: List[Dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_training_report(report: Mapping[str, Any], *, path: Path = ACTION_POLICY_REPORT_PATH) -> None:
    _atomic_write_json(path, dict(report))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_policy_status(
    *,
    examples_path: Path = ACTION_POLICY_EXAMPLES_PATH,
    report_path: Path = ACTION_POLICY_REPORT_PATH,
    model_path: Path = ACTION_POLICY_MODEL_PATH,
) -> Dict[str, Any]:
    examples = read_examples(examples_path)
    report: Dict[str, Any] = {}
    try:
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            report = loaded
    except (OSError, ValueError):
        report = {}
    by_domain: Dict[str, int] = {}
    by_label: Dict[str, int] = {}
    recent = examples[-20:]
    for row in examples:
        domain = str(row.get("domain") or "unknown")
        label = str(row.get("outcome_label") or "unknown")
        by_domain[domain] = by_domain.get(domain, 0) + 1
        by_label[label] = by_label.get(label, 0) + 1
    model_exists = model_path.exists()
    return {
        "has_dataset": bool(examples),
        "dataset_rows": len(examples),
        "rows_by_domain": by_domain,
        "rows_by_label": by_label,
        "recent_examples": recent,
        "model_exists": model_exists,
        "model_path": _display_path(model_path) if model_exists else None,
        "last_report": report or None,
        "policy_mode": "shadow",
        "runtime_allowed": False,
        "source": {
            "examples": _display_path(examples_path),
            "model": _display_path(model_path),
            "report": _display_path(report_path),
        },
    }
