"""Promotion-policy profitability gates (1b/1c) — real classes, no mocks.

Covers the 2026-06-10 upgrade: balanced-accuracy floor, class-collapse
rejection, cost-aware expectancy floor, and the fail-closed flip for
unreadable production metrics.
"""
from __future__ import annotations

from src.training.promotion_policy import (
    MIN_ABSOLUTE_ACCURACY,
    MIN_EXPECTANCY_PIPS,
    should_promote,
)


def _candidate(**over) -> dict:
    """A candidate that passes every gate unless overridden."""
    base = {
        "accuracy": 0.58,
        "balanced_accuracy": 0.56,
        "expectancy_pips": 1.2,
        "profit_factor": 1.4,
        "class_collapse": False,
        "long_share": 0.55,
        "short_share": 0.45,
        "holdout_samples": 150,
        "timeframes_passed": 2,
        "max_component_age_days": 3.0,
        "ensemble_complete": True,
    }
    base.update(over)
    return base


def test_all_short_predictor_rejected_despite_high_raw_accuracy():
    """The May-2026 incident shape: 70% raw on a SHORT-heavy slice."""
    decision = should_promote(
        candidate=_candidate(
            accuracy=0.70,
            balanced_accuracy=0.50,
            class_collapse=True,
            long_share=0.02,
            short_share=0.98,
        ),
        production=None,
    )
    assert decision.promote is False
    assert "balanced" in decision.reason or "collapse" in decision.reason


def test_class_collapse_alone_rejects():
    decision = should_promote(
        candidate=_candidate(class_collapse=True, long_share=0.95, short_share=0.05),
        production=None,
    )
    assert decision.promote is False
    assert "collapse" in decision.reason


def test_negative_expectancy_rejects_accurate_model():
    """Accurate but unprofitable after spread = noise, not edge."""
    decision = should_promote(
        candidate=_candidate(expectancy_pips=-0.8),
        production=None,
    )
    assert decision.promote is False
    assert "expectancy" in decision.reason
    assert MIN_EXPECTANCY_PIPS == 0.0


def test_profitable_balanced_candidate_promotes_first_run():
    decision = should_promote(candidate=_candidate(), production=None)
    assert decision.promote is True


def test_legacy_candidate_without_new_metrics_still_evaluated():
    """Backward compat: gates 1b/1c inert when metrics absent."""
    legacy = _candidate()
    for k in ("balanced_accuracy", "expectancy_pips", "class_collapse",
              "long_share", "short_share", "profit_factor"):
        legacy.pop(k, None)
    decision = should_promote(candidate=legacy, production=None)
    assert decision.promote is True


def test_unreadable_production_metrics_fails_closed():
    """Was fail-open ('promote by default'); now holds for manual review."""
    decision = should_promote(
        candidate=_candidate(),
        production={"accuracy": None},
    )
    assert decision.promote is False
    assert "MANUAL" in decision.reason or "unreadable" in decision.reason


def test_balanced_accuracy_floor_uses_same_constant_as_raw():
    decision = should_promote(
        candidate=_candidate(balanced_accuracy=MIN_ABSOLUTE_ACCURACY - 0.01),
        production=None,
    )
    assert decision.promote is False
