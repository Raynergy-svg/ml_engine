"""Frozen challenger-vs-champion safety states for AXIOM risk releases."""
from __future__ import annotations

from typing import Any, Mapping

from src.axiom_training.risk_gym import RiskPolicy, RiskPolicyParams


# Conservative champion active when the stress-release protocol was introduced.
# This anchor prevents risk posture from ratcheting upward through a sequence of
# individually compared intermediate champions.
SAFETY_REFERENCE_PARAMS = RiskPolicyParams(
    target_vol_multiplier=1.25,
    drawdown_soft=0.06,
    drawdown_hard=0.14,
    loss_streak_decay=0.55,
    min_scale=0.05,
    max_scale=0.85,
    concentration_soft=0.35,
    concentration_hard=0.65,
    unresolved_exposure_scale=0.50,
)

CRITICAL_STATES = {
    "elevated_volatility": dict(
        trailing_vol=0.03, target_vol=0.01, drawdown=0.0, loss_streak=0,
        portfolio_concentration=0.0, exposure_resolved_fraction=1.0,
    ),
    "drawdown": dict(
        trailing_vol=0.01, target_vol=0.01, drawdown=0.10, loss_streak=0,
        portfolio_concentration=0.0, exposure_resolved_fraction=1.0,
    ),
    "loss_streak": dict(
        trailing_vol=0.01, target_vol=0.01, drawdown=0.0, loss_streak=4,
        portfolio_concentration=0.0, exposure_resolved_fraction=1.0,
    ),
    "concentration": dict(
        trailing_vol=0.01, target_vol=0.01, drawdown=0.0, loss_streak=0,
        portfolio_concentration=0.60, exposure_resolved_fraction=1.0,
    ),
    "unresolved_exposure": dict(
        trailing_vol=0.01, target_vol=0.01, drawdown=0.0, loss_streak=0,
        portfolio_concentration=0.0, exposure_resolved_fraction=0.0,
    ),
    "combined_crisis": dict(
        trailing_vol=0.02, target_vol=0.01, drawdown=0.08, loss_streak=3,
        portfolio_concentration=0.50, exposure_resolved_fraction=0.60,
    ),
    "correlation_cluster_stress": dict(
        trailing_vol=0.01, target_vol=0.01, drawdown=0.0, loss_streak=0,
        correlation_cluster_stress=1.0,
    ),
    "regime_transition": dict(
        trailing_vol=0.01, target_vol=0.01, drawdown=0.0, loss_streak=0,
        regime_transition_risk=1.0,
    ),
    "liquidity_deterioration": dict(
        trailing_vol=0.01, target_vol=0.01, drawdown=0.0, loss_streak=0,
        liquidity_stress=1.0,
    ),
    "confidence_disagreement": dict(
        trailing_vol=0.01, target_vol=0.01, drawdown=0.0, loss_streak=0,
        confidence_shortfall=1.0, model_disagreement=1.0,
    ),
}


def compare_risk_params(
    challenger_params: Mapping[str, Any], reference_params: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        challenger = RiskPolicy(RiskPolicyParams(**dict(challenger_params)))
        reference = RiskPolicy(RiskPolicyParams(**dict(reference_params)))
    except (TypeError, ValueError):
        return {"passed": False, "reasons": ["challenger_or_reference_params_invalid"]}
    comparisons = {}
    reasons = []
    for name, state in CRITICAL_STATES.items():
        challenger_scale = challenger.scale(**state)
        reference_scale = reference.scale(**state)
        passed = challenger_scale <= reference_scale + 1e-12
        comparisons[name] = {
            "passed": passed,
            "challenger_scale": challenger_scale,
            "reference_scale": reference_scale,
        }
        if not passed:
            reasons.append(name)
    return {"passed": not reasons, "reasons": reasons, "states": comparisons}
