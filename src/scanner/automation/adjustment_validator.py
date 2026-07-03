"""Validates proposed config adjustments against ScannerConfig schema.

US-508: Intercepts every self-tuner write and rejects orphan keys before
they can silently create dead-letter attributes in ScannerConfig.

Root cause of 2026-04-16 incident: config_adjustments.json contained
'min_confidence_threshold' and 'atr_sl_multiplier_low_regime' — keys that
don't exist in ScannerConfig. setattr() created orphan attributes while the
real fields remained unchanged. This validator prevents that at proposal time.
"""

from __future__ import annotations

import dataclasses
import logging
import typing
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Known orphan keys from phase 90 incident (2026-04-16)
# Maps wrong proposed key → correct ScannerConfig field name
KNOWN_ORPHAN_CORRECTIONS: dict[str, str] = {
    # Phase 90 incident — 5 confirmed dead proposals
    "min_confidence_threshold": "min_confidence",
    "atr_sl_multiplier_low_regime": "atr_sl_multiplier",
    # Common near-misses
    "confidence_threshold": "min_confidence",
    "sl_multiplier": "atr_sl_multiplier",
    "tp_multiplier": "atr_tp_multiplier",
    "uncertainty_score": "max_uncertainty_score",
    "model_disagreement": "max_model_disagreement",
    "vote_threshold": "weighted_vote_threshold",
    "risk_pct": "risk_per_trade_pct",
    "max_risk": "max_open_risk_pct",
}

# Numeric bounds: {field_name: (min_value, max_value)}
# Prevents runaway tuning from sending config into unsafe territory.
FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "atr_sl_multiplier": (0.3, 3.0),
    "atr_tp_multiplier": (0.5, 5.0),
    "min_confidence": (10.0, 100.0),
    "min_momentum": (0.0, 1.0),
    "max_uncertainty_score": (0.10, 0.99),
    "max_model_disagreement": (0.05, 0.99),
    "weighted_vote_threshold": (0.30, 0.99),
    "risk_per_trade_pct": (0.001, 0.20),
    "max_open_risk_pct": (0.01, 0.50),
    "final_score_threshold": (0.20, 0.95),
    "min_risk_reward_ratio": (0.8, 10.0),
    "meta_labeler_threshold": (0.40, 0.70),
    "max_uncertainty_std": (0.05, 0.50),
    "staleness_uncertainty_threshold": (0.10, 0.90),
    "staleness_age_threshold_days": (1.0, 90.0),
    "agent_weight_learning_rate": (0.001, 0.5),
    "trailing_atr_multiplier": (0.3, 5.0),
    "high_prob_threshold": (0.50, 0.99),
    "min_risk_reward_ratio": (0.8, 10.0),
}


@dataclass
class ValidationResult:
    valid: bool
    field_exists: bool
    type_matches: bool
    within_bounds: bool
    error_message: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ValidationResult":
        return cls(
            valid=d.get("valid", False),
            field_exists=d.get("field_exists", False),
            type_matches=d.get("type_matches", False),
            within_bounds=d.get("within_bounds", False),
            error_message=d.get("error_message", ""),
        )


def _get_scanner_config_fields() -> dict[str, dataclasses.Field]:
    """Return {field_name: Field} via dataclasses.fields() — no hardcoded list."""
    from src.scanner.config import ScannerConfig
    return {f.name: f for f in dataclasses.fields(ScannerConfig)}


def _get_scanner_config_hints() -> dict[str, Any]:
    """Return resolved type hints for ScannerConfig."""
    try:
        from src.scanner.config import ScannerConfig
        return typing.get_type_hints(ScannerConfig)
    except Exception:
        return {}


def _check_type(field_name: str, value: Any, hints: dict[str, Any]) -> tuple[bool, str]:
    """Check if value is compatible with the field's type annotation.

    Only validates simple scalar types (float, int, bool, str).
    Complex types (List, Dict, Optional, Path) are skipped — too costly
    to validate deeply and rarely the source of orphan-key bugs.
    """
    expected = hints.get(field_name)
    if expected is None:
        return True, ""

    origin = getattr(expected, "__origin__", None)
    if origin is not None:
        # Complex type (List[str], Dict[str, float], Optional[X]) — skip
        return True, ""

    if expected is float:
        try:
            float(value)
            return True, ""
        except (TypeError, ValueError):
            return False, f"Cannot cast {value!r} to float for field '{field_name}'"

    if expected is int:
        if isinstance(value, bool):
            return False, f"Field '{field_name}' expects int, got bool"
        try:
            int(value)
            return True, ""
        except (TypeError, ValueError):
            return False, f"Cannot cast {value!r} to int for field '{field_name}'"

    if expected is bool:
        if not isinstance(value, bool):
            return False, (
                f"Field '{field_name}' expects bool, got {type(value).__name__}. "
                "Pass True/False, not 1/0."
            )
        return True, ""

    if expected is str:
        if not isinstance(value, str):
            return False, f"Field '{field_name}' expects str, got {type(value).__name__}"
        return True, ""

    return True, ""


def _check_bounds(field_name: str, value: Any) -> tuple[bool, str]:
    """Check numeric value is within safe operating bounds."""
    bounds = FIELD_BOUNDS.get(field_name)
    if bounds is None:
        return True, ""
    lo, hi = bounds
    try:
        v = float(value)
        if not (lo <= v <= hi):
            return (
                False,
                f"Value {value} for '{field_name}' out of safe bounds [{lo}, {hi}]",
            )
    except (TypeError, ValueError):
        return True, ""  # type check will catch the real error
    return True, ""


# Hard NOs (L-003/L-004): fields the adjustment pipeline may NEVER carry,
# regardless of approval. oanda_environment is the practice pin — no proposal,
# approval, overlay, or self-heal action may scaffold a path that flips it
# (2026-07-03: added when the durable config overlay made "an approved
# adjustment reaches every process" a real propagation channel).
PROTECTED_FIELDS = frozenset({"oanda_environment"})


def validate_adjustment(key: str, value: Any, reason: str) -> ValidationResult:
    """Validate a proposed config adjustment against ScannerConfig schema.

    Enumerates ScannerConfig fields via dataclasses.fields() — no hardcoded list.
    A result with valid=False must NOT reach config_adjustments.json.

    Args:
        key: Proposed ScannerConfig field name.
        value: Proposed new value.
        reason: Human-readable justification (logged but not validated).

    Returns:
        ValidationResult with per-dimension diagnostics.
    """
    if key in PROTECTED_FIELDS:
        return ValidationResult(
            valid=False,
            field_exists=True,
            type_matches=False,
            within_bounds=False,
            error_message=(
                f"'{key}' is a PROTECTED field (Hard NO, L-003) — the adjustment "
                "pipeline may never carry it. Refused unconditionally."
            ),
        )

    fields = _get_scanner_config_fields()

    if key not in fields:
        correction = KNOWN_ORPHAN_CORRECTIONS.get(key, "")
        hint = f" Did you mean '{correction}'?" if correction else ""
        return ValidationResult(
            valid=False,
            field_exists=False,
            type_matches=False,
            within_bounds=False,
            error_message=f"Unknown field '{key}' — not in ScannerConfig.{hint}",
        )

    hints = _get_scanner_config_hints()

    type_ok, type_err = _check_type(key, value, hints)
    if not type_ok:
        return ValidationResult(
            valid=False,
            field_exists=True,
            type_matches=False,
            within_bounds=False,
            error_message=type_err,
        )

    bounds_ok, bounds_err = _check_bounds(key, value)
    if not bounds_ok:
        return ValidationResult(
            valid=False,
            field_exists=True,
            type_matches=True,
            within_bounds=False,
            error_message=bounds_err,
        )

    return ValidationResult(
        valid=True,
        field_exists=True,
        type_matches=True,
        within_bounds=True,
        error_message="",
    )
