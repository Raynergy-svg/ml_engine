"""Promotion and practice-runtime adapter for AXIOM's risk controller.

The trained candidate stays immutable. Promotion writes a separate state file
containing hashes of the exact model/report pair that passed. Runtime accepts
that state only for the OANDA practice environment and only as a multiplier in
``[0, 1]`` over the existing ATR/risk-gated target units.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.axiom_training.risk_gym import REGIME_ORDINAL, RiskPolicy, RiskPolicyParams
from src.axiom_training.risk_release_safety import (
    SAFETY_REFERENCE_PARAMS,
    compare_risk_params,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REL = Path("trained_data/axiom/risk_policy_candidate.json")
REPORT_REL = Path("trained_data/axiom/risk_policy_report.json")
CHAMPION_MODEL_REL = Path("trained_data/axiom/risk_policy_champion.json")
CHAMPION_REPORT_REL = Path("trained_data/axiom/risk_policy_champion_report.json")
PROMOTION_REL = Path("trained_data/axiom/risk_policy_promotion.json")
RUNTIME_REL = Path("trained_data/axiom/risk_policy_runtime.json")
AUDIT_REL = Path("trained_data/axiom/risk_policy_promotion_audit.jsonl")
FEEDBACK_REL = Path("logs/trade_feedback.jsonl")
PEAK_NAV_REL = Path("trained_data/oanda/peak_nav.json")
EXPOSURE_HISTORY_REL = Path("trained_data/hedge/exposure_history.jsonl")
MAX_ARTIFACT_AGE_HOURS = 48.0


@dataclass(frozen=True)
class PromotionValidation:
    valid: bool
    reasons: tuple[str, ...]
    model_sha256: str | None
    report_sha256: str | None
    model: dict[str, Any] | None
    report: dict[str, Any] | None


@dataclass(frozen=True)
class RuntimeRiskDecision:
    promotion_enabled: bool
    valid: bool
    scale: float
    reason: str
    trailing_volatility: float | None = None
    target_volatility: float | None = None
    drawdown: float | None = None
    loss_streak: int | None = None
    portfolio_concentration: float | None = None
    exposure_resolved_fraction: float | None = None
    exposure_source: str | None = None
    strategy_risk_multiplier: float | None = None
    correlation_cluster_stress: float | None = None
    regime_transition_risk: float | None = None
    liquidity_stress: float | None = None
    confidence_shortfall: float | None = None
    model_disagreement: float | None = None
    model_sha256: str | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


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


def _append_audit(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def validate_candidate(
    *,
    root: Path = REPO_ROOT,
    now: datetime | None = None,
    model_rel: Path = MODEL_REL,
    report_rel: Path = REPORT_REL,
    require_fresh: bool = True,
) -> PromotionValidation:
    root = Path(root)
    model_path = root / model_rel
    report_path = root / report_rel
    model = _read_json(model_path)
    report = _read_json(report_path)
    reasons: list[str] = []
    model_hash = _sha256(model_path)
    report_hash = _sha256(report_path)
    if model is None or model_hash is None:
        reasons.append("model_missing_or_invalid")
    if report is None or report_hash is None:
        reasons.append("report_missing_or_invalid")
    if model is not None and report is not None:
        if model.get("model_type") != "axiom_risk_controller_v1":
            reasons.append("unexpected_model_type")
        if report.get("status") != "passed" or report.get("passing_model") is not True:
            reasons.append("training_report_not_passed")
        if report.get("reasons"):
            reasons.append("training_report_has_failure_reasons")
        if report.get("direction_training") is not False:
            reasons.append("direction_training_not_disabled")
        if model.get("trained_at") != report.get("trained_at"):
            reasons.append("model_report_training_timestamp_mismatch")
        try:
            state_schema = int(model.get("state_schema_version", 1) or 1)
        except (TypeError, ValueError):
            state_schema = 0
            reasons.append("state_schema_version_invalid")
        forbidden = set(model.get("forbidden_features") or [])
        required_forbidden = (
            {"direction", "future_return", "pair"}
            if state_schema >= 2 else {"direction", "future_return", "strategy_identity"}
        )
        if not required_forbidden.issubset(forbidden):
            reasons.append("forbidden_feature_contract_incomplete")
        features = set(model.get("features") or [])
        if not {"portfolio_concentration", "exposure_resolved_fraction"}.issubset(features):
            reasons.append("portfolio_exposure_features_missing")
        if state_schema >= 2:
            rich_features = {
                "strategy_risk_prior", "correlation_cluster_stress",
                "regime_transition_risk", "liquidity_stress",
                "confidence_shortfall", "model_disagreement",
            }
            if not rich_features.issubset(features):
                reasons.append("rich_risk_state_features_missing")
            priors = model.get("strategy_risk_multipliers")
            if not isinstance(priors, dict) or not priors:
                reasons.append("strategy_risk_multipliers_missing")
        stress = report.get("stress_results")
        if not isinstance(stress, dict) or not stress:
            reasons.append("stress_results_missing")
        elif any(not isinstance(row, dict) or row.get("passed") is not True for row in stress.values()):
            reasons.append("stress_gate_failed")
        exposure_verification = report.get("portfolio_exposure_verification")
        if not isinstance(exposure_verification, dict) or exposure_verification.get("passed") is not True:
            reasons.append("portfolio_exposure_verification_failed")
        adversarial_stress = report.get("adversarial_stress")
        if not isinstance(adversarial_stress, dict) or adversarial_stress.get("passed") is not True:
            reasons.append("adversarial_stress_failed")
        behavioral_invariants = report.get("behavioral_invariants")
        if not isinstance(behavioral_invariants, dict) or behavioral_invariants.get("passed") is not True:
            reasons.append("behavioral_invariants_failed")
        # Champion-challenger gate: enforced only when the section is present.
        # The 2026-07-11 01:37Z champion report predates this gate; requiring
        # the key unconditionally would fail-close the LIVE runtime to scale 0
        # on a report that passed every gate that existed when it shipped.
        # Every report produced after this commit includes the section, so this
        # becomes effectively unconditional one promotion from now.
        champion_challenger = report.get("champion_challenger")
        if champion_challenger is not None and (
            not isinstance(champion_challenger, dict)
            or champion_challenger.get("passed") is not True
        ):
            reasons.append("champion_challenger_failed")
        trained_at = _parse_time(model.get("trained_at"))
        if trained_at is None:
            reasons.append("trained_at_invalid")
        else:
            age_h = ((now or _utc_now()) - trained_at).total_seconds() / 3600.0
            if require_fresh and (age_h < -0.05 or age_h > MAX_ARTIFACT_AGE_HOURS):
                reasons.append("candidate_stale")
        params = model.get("params")
        try:
            if not isinstance(params, dict):
                raise ValueError("params must be a mapping")
            RiskPolicyParams(**params).validate()
            if float(params.get("min_scale", 1.0)) > 0.05:
                reasons.append("minimum_scale_exceeds_safety_cap")
            reference_comparison = compare_risk_params(
                params, asdict(SAFETY_REFERENCE_PARAMS),
            )
            if not reference_comparison["passed"]:
                reasons.append("frozen_safety_reference_regression")
        except (TypeError, ValueError):
            reasons.append("model_params_invalid")
        if model.get("runtime_allowed") is not False or model.get("shadow_only") is not True:
            reasons.append("candidate_boundary_flags_invalid")
    return PromotionValidation(
        valid=not reasons,
        reasons=tuple(reasons),
        model_sha256=model_hash,
        report_sha256=report_hash,
        model=model,
        report=report,
    )


def _challenger_safety_comparison(
    challenger_model: Mapping[str, Any], champion_model: Mapping[str, Any],
) -> dict[str, Any]:
    """Forbid a challenger from increasing risk in frozen critical states."""
    try:
        challenger_params = challenger_model["params"]
        champion_params = champion_model["params"]
    except (KeyError, TypeError):
        return {"passed": False, "reasons": ["challenger_or_champion_params_invalid"]}
    comparison = compare_risk_params(challenger_params, champion_params)
    for state in comparison.get("states", {}).values():
        state["champion_scale"] = state.pop("reference_scale")
    return comparison


def promote_candidate(
    *,
    root: Path = REPO_ROOT,
    environment: str,
    actor: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Promote the exact passing artifact for practice sizing consumption."""
    root = Path(root)
    ts = (now or _utc_now()).isoformat()
    if environment != "practice":
        event = {
            "ts": ts, "event": "promotion_refused", "actor": actor,
            "environment": environment, "reasons": ["practice_environment_required"],
        }
        _append_audit(root / AUDIT_REL, event)
        raise ValueError("AXIOM risk policy promotion is practice-only")
    validation = validate_candidate(root=root, now=now)
    if not validation.valid:
        event = {
            "ts": ts, "event": "promotion_refused", "actor": actor,
            "environment": environment, "reasons": list(validation.reasons),
        }
        _append_audit(root / AUDIT_REL, event)
        raise ValueError(f"candidate promotion refused: {', '.join(validation.reasons)}")
    existing_champion = _read_json(root / CHAMPION_MODEL_REL)
    challenger_comparison = None
    if existing_champion:
        challenger_comparison = _challenger_safety_comparison(
            validation.model or {}, existing_champion,
        )
        if not challenger_comparison["passed"]:
            event = {
                "ts": ts, "event": "promotion_refused", "actor": actor,
                "environment": environment,
                "reasons": [
                    f"critical_state_regression:{name}"
                    for name in challenger_comparison["reasons"]
                ],
                "challenger_comparison": challenger_comparison,
            }
            _append_audit(root / AUDIT_REL, event)
            raise ValueError(
                "candidate promotion refused: " + ", ".join(event["reasons"])
            )
    safety_reference_comparison = compare_risk_params(
        (validation.model or {}).get("params") or {},
        asdict(SAFETY_REFERENCE_PARAMS),
    )
    if not safety_reference_comparison["passed"]:
        raise ValueError("candidate promotion refused: frozen_safety_reference_regression")
    # Candidate files remain replaceable training outputs. Runtime receives
    # immutable champion copies so a future retrain cannot mutate the active
    # policy without another explicit promotion.
    _atomic_json(root / CHAMPION_MODEL_REL, validation.model or {})
    _atomic_json(root / CHAMPION_REPORT_REL, validation.report or {})
    champion_validation = validate_candidate(
        root=root,
        now=now,
        model_rel=CHAMPION_MODEL_REL,
        report_rel=CHAMPION_REPORT_REL,
        require_fresh=False,
    )
    if not champion_validation.valid:
        raise ValueError(
            f"champion copy validation failed: {', '.join(champion_validation.reasons)}"
        )
    state = {
        "schema_version": "axiom-risk-promotion-v1",
        "enabled": True,
        "runtime_allowed": True,
        "shadow_only": False,
        "mode": "live_practice",
        "environment": "practice",
        "actor": actor,
        "promoted_at": ts,
        "model_path": str(CHAMPION_MODEL_REL),
        "report_path": str(CHAMPION_REPORT_REL),
        "candidate_model_sha256": validation.model_sha256,
        "candidate_report_sha256": validation.report_sha256,
        "model_sha256": champion_validation.model_sha256,
        "report_sha256": champion_validation.report_sha256,
        "model_type": "axiom_risk_controller_v1",
        "de_risk_only": True,
        "max_runtime_scale": 1.0,
        "challenger_comparison": challenger_comparison,
        "safety_reference_comparison": safety_reference_comparison,
    }
    _atomic_json(root / PROMOTION_REL, state)
    _append_audit(root / AUDIT_REL, {"ts": ts, "event": "promoted", **state})
    return state


def _validated_promotion(
    *, root: Path, environment: str, now: datetime | None = None,
) -> tuple[dict[str, Any] | None, PromotionValidation, list[str]]:
    state = _read_json(root / PROMOTION_REL)
    reasons: list[str] = []
    if state is None or state.get("enabled") is not True:
        validation = validate_candidate(root=root, now=now)
        reasons.append("promotion_not_enabled")
        return state, validation, reasons
    try:
        model_rel = Path(str(state.get("model_path")))
        report_rel = Path(str(state.get("report_path")))
    except (TypeError, ValueError):
        model_rel = CHAMPION_MODEL_REL
        report_rel = CHAMPION_REPORT_REL
    if model_rel != CHAMPION_MODEL_REL or report_rel != CHAMPION_REPORT_REL:
        reasons.append("promotion_paths_not_canonical_champion")
        model_rel = CHAMPION_MODEL_REL
        report_rel = CHAMPION_REPORT_REL
    validation = validate_candidate(
        root=root,
        now=now,
        model_rel=model_rel,
        report_rel=report_rel,
        require_fresh=False,
    )
    if environment != "practice" or state.get("environment") != "practice":
        reasons.append("practice_environment_required")
    if state.get("mode") != "live_practice" or state.get("de_risk_only") is not True:
        reasons.append("promotion_mode_invalid")
    if not validation.valid:
        reasons.extend(validation.reasons)
    if state.get("model_sha256") != validation.model_sha256:
        reasons.append("promoted_model_hash_mismatch")
    if state.get("report_sha256") != validation.report_sha256:
        reasons.append("promoted_report_hash_mismatch")
    return state, validation, reasons


def _fx_returns(path: Path) -> list[float]:
    values: list[float] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        try:
            payload = json.loads(line)
            shaped = float(payload.get("shaped_reward"))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        if math.isfinite(shaped):
            values.append(float(np.clip(shaped / 20.0, -3.0, 3.0) * 0.01))
    return values


def _nav_drawdown(nav: float, peak_path: Path) -> float:
    peak = float(nav)
    payload = _read_json(peak_path)
    try:
        peak = max(peak, float((payload or {}).get("peak_nav", nav)))
    except (TypeError, ValueError):
        peak = float(nav)
    return max(0.0, (peak - float(nav)) / peak) if peak > 0 else 0.0


def _latest_exposure_state(path: Path) -> tuple[float, float, float, str]:
    """Convert the latest hedge snapshot into direction-free portfolio-risk state."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0.0, 0.0, 0.0, "exposure_history_missing"
    payload: dict[str, Any] | None = None
    for line in reversed(lines):
        try:
            candidate = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        return 0.0, 0.0, 0.0, "exposure_history_unreadable"
    buckets = payload.get("net_correlation_buckets_notional_home")
    values: list[float] = []
    if isinstance(buckets, Mapping):
        for raw in buckets.values():
            try:
                value = abs(float(raw))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    gross = sum(values)
    concentration = max(values, default=0.0) / gross if gross > 0.0 else 0.0
    shares = [value / gross for value in values] if gross > 0.0 else []
    hhi = sum(share * share for share in shares)
    cluster_stress = (
        (hhi - 1.0 / len(shares)) / (1.0 - 1.0 / len(shares))
        if len(shares) > 1 else (1.0 if shares else 0.0)
    )
    try:
        total = max(0, int(payload.get("open_trade_count", 0)))
        resolved = max(0, total - len(payload.get("skipped_instruments") or []))
    except (TypeError, ValueError):
        total = resolved = 0
    resolved_fraction = resolved / total if total else 0.0
    return (
        float(np.clip(concentration, 0.0, 1.0)),
        float(np.clip(resolved_fraction, 0.0, 1.0)),
        float(np.clip(cluster_stress, 0.0, 1.0)),
        str(path.relative_to(path.parents[2])) if len(path.parents) > 2 else str(path),
    )


def _feedback_risk_state(path: Path) -> dict[str, float]:
    """Read lagged, direction-free execution/uncertainty state from outcomes."""
    payloads: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            payload = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)

    regimes = [
        str(row.get("regime") or "").upper()
        for row in payloads if str(row.get("regime") or "").upper() in REGIME_ORDINAL
    ]
    transition = 0.0
    if len(regimes) >= 2:
        transition = abs(REGIME_ORDINAL[regimes[-1]] - REGIME_ORDINAL[regimes[-2]]) / 3.0

    execution_costs: list[float] = []
    for row in payloads[-20:]:
        try:
            slippage = abs(float(row.get("slippage_pips", 0.0) or 0.0))
            spread = abs(float(row.get("spread_pips", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(slippage) and math.isfinite(spread):
            execution_costs.append(max(slippage, spread))
    recent_cost = float(np.mean(execution_costs[-5:])) if execution_costs else 0.0

    def latest_unit(key: str) -> float | None:
        for row in reversed(payloads):
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return float(np.clip(value, 0.0, 1.0))
        return None

    confidence = latest_unit("confidence")
    disagreement = latest_unit("model_disagreement")
    return {
        "regime_transition_risk": float(np.clip(transition, 0.0, 1.0)),
        "liquidity_stress": float(np.clip(recent_cost / 4.0, 0.0, 1.0)),
        "confidence_shortfall": 1.0 - confidence if confidence is not None else 0.0,
        "model_disagreement": disagreement or 0.0,
    }


def evaluate_runtime_risk(
    *,
    root: Path = REPO_ROOT,
    environment: str,
    nav: float,
    open_trade_count: int | None = None,
    strategy: str = "oanda_fx_trend",
    regime_transition_risk: float | None = None,
    liquidity_stress: float | None = None,
    confidence_shortfall: float | None = None,
    model_disagreement: float | None = None,
    now: datetime | None = None,
) -> RuntimeRiskDecision:
    """Return the promoted de-risk multiplier; invalid active state fails to zero."""
    root = Path(root)
    state, validation, reasons = _validated_promotion(
        root=root, environment=environment, now=now,
    )
    if state is None or state.get("enabled") is not True:
        return RuntimeRiskDecision(False, True, 1.0, "promotion_not_enabled")
    if reasons:
        decision = RuntimeRiskDecision(
            True, False, 0.0, ";".join(dict.fromkeys(reasons)),
            model_sha256=validation.model_sha256,
        )
        write_runtime_status(root=root, decision=decision, environment=environment, now=now)
        return decision

    model = validation.model or {}
    params = RiskPolicyParams(**model["params"])
    values = _fx_returns(root / FEEDBACK_REL)
    target_vol = float((model.get("target_vol_by_strategy") or {}).get(strategy, 0.01))
    trailing = values[-20:]
    trailing_vol = float(np.std(trailing, ddof=1)) if len(trailing) > 1 else target_vol
    loss_streak = 0
    for value in reversed(values):
        if value >= 0:
            break
        loss_streak += 1
    drawdown = _nav_drawdown(nav, root / PEAK_NAV_REL)
    if open_trade_count == 0:
        concentration, resolved_fraction, cluster_stress, exposure_source = (
            0.0, 1.0, 0.0, "current_account_flat"
        )
    else:
        concentration, resolved_fraction, cluster_stress, exposure_source = _latest_exposure_state(
            root / EXPOSURE_HISTORY_REL
        )
    feedback_state = _feedback_risk_state(root / FEEDBACK_REL)
    transition = float(np.clip(
        feedback_state["regime_transition_risk"]
        if regime_transition_risk is None else regime_transition_risk, 0.0, 1.0,
    ))
    liquidity = float(np.clip(
        feedback_state["liquidity_stress"]
        if liquidity_stress is None else liquidity_stress, 0.0, 1.0,
    ))
    confidence_gap = float(np.clip(
        feedback_state["confidence_shortfall"]
        if confidence_shortfall is None else confidence_shortfall, 0.0, 1.0,
    ))
    disagreement = float(np.clip(
        feedback_state["model_disagreement"]
        if model_disagreement is None else model_disagreement, 0.0, 1.0,
    ))
    strategy_multiplier = float(np.clip(
        (model.get("strategy_risk_multipliers") or {}).get(strategy, 1.0), 0.0, 1.0,
    ))
    scale = RiskPolicy(params).scale(
        trailing_vol=trailing_vol,
        target_vol=target_vol,
        drawdown=drawdown,
        loss_streak=loss_streak,
        portfolio_concentration=concentration,
        exposure_resolved_fraction=resolved_fraction,
        strategy_risk_multiplier=strategy_multiplier,
        correlation_cluster_stress=cluster_stress,
        regime_transition_risk=transition,
        liquidity_stress=liquidity,
        confidence_shortfall=confidence_gap,
        model_disagreement=disagreement,
    )
    scale = float(np.clip(scale, 0.0, 1.0))
    decision = RuntimeRiskDecision(
        True, True, scale, "promoted_model_applied",
        trailing_volatility=trailing_vol,
        target_volatility=target_vol,
        drawdown=drawdown,
        loss_streak=loss_streak,
        portfolio_concentration=concentration,
        exposure_resolved_fraction=resolved_fraction,
        exposure_source=exposure_source,
        strategy_risk_multiplier=strategy_multiplier,
        correlation_cluster_stress=cluster_stress,
        regime_transition_risk=transition,
        liquidity_stress=liquidity,
        confidence_shortfall=confidence_gap,
        model_disagreement=disagreement,
        model_sha256=validation.model_sha256,
    )
    write_runtime_status(root=root, decision=decision, environment=environment, now=now)
    return decision


def scale_target_units(want_units: Mapping[str, int], scale: float) -> dict[str, int]:
    """Apply a de-risk-only multiplier. Invalid/out-of-range values fail to zero."""
    try:
        value = float(scale)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        value = 0.0
    return {instrument: int(int(units) * value) for instrument, units in want_units.items()}


def write_runtime_status(
    *,
    root: Path,
    decision: RuntimeRiskDecision,
    environment: str,
    now: datetime | None = None,
) -> None:
    _atomic_json(Path(root) / RUNTIME_REL, {
        "schema_version": "axiom-risk-runtime-v1",
        "updated_at": (now or _utc_now()).isoformat(),
        "environment": environment,
        "mode": "live_practice" if decision.promotion_enabled and decision.valid else "fail_closed",
        **asdict(decision),
    })


def read_promotion_status(
    *, root: Path = REPO_ROOT, environment: str = "practice",
) -> dict[str, Any]:
    root = Path(root)
    state, validation, reasons = _validated_promotion(root=root, environment=environment)
    runtime = _read_json(root / RUNTIME_REL)
    receipt_at = _parse_time((runtime or {}).get("updated_at"))
    promoted_at = _parse_time((state or {}).get("promoted_at"))
    receipt_after_promotion = bool(
        receipt_at and promoted_at and receipt_at >= promoted_at
    )
    return {
        "enabled": bool(state and state.get("enabled")),
        "valid": bool(state and state.get("enabled") and not reasons),
        "mode": state.get("mode") if state else "shadow",
        "environment": state.get("environment") if state else None,
        "promoted_at": state.get("promoted_at") if state else None,
        "model_sha256": state.get("model_sha256") if state else None,
        "candidate_valid": validation.valid,
        "reasons": list(dict.fromkeys(reasons)),
        "runtime": runtime,
        "consumed_by_live_sizing": bool(
            runtime
            and receipt_after_promotion
            and runtime.get("promotion_enabled") is True
            and runtime.get("valid") is True
            and runtime.get("model_sha256") == (state or {}).get("model_sha256")
        ),
    }
