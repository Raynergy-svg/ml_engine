"""Model promotion policy — the gate between :staging and :production.

This file defines WHEN a freshly-trained model (currently at `:staging`)
is allowed to replace the current `:production` model.

Why it lives here (not in the trainer or the registry code):
- The trainer's job is to produce a model; it doesn't decide fitness for prod.
- The registry's job is storage + alias management; it doesn't decide value.
- THIS file encodes risk appetite — it is the single throat to cut when the
  autonomy loop misbehaves. Change this, change the bot's promotion gate.

Inputs:
    candidate  — metrics dict from the just-trained model (holdout accuracy,
                 validation auc, brier score, sample size, etc.)
    production — metrics dict from the currently-deployed model, or None if
                 there is no prior production (first-run case).

Output:
    PromotionDecision(promote: bool, reason: str, diagnostics: dict)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PromotionDecision:
    """The verdict from the policy. Auditable — reason is always populated."""
    promote: bool
    reason: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
# PROMOTION GATE CONFIGURATION — EDIT THESE TO TUNE RISK APPETITE
# ─────────────────────────────────────────────────────────────────
# These are the FIVE numbers that encode "when is a new model good enough?"
# They are INTENTIONALLY defined as module constants (not a yaml file, not a
# class) so they show up in git diff and code review. Every change to these
# values IS a policy change.

# 1. Absolute floor — a candidate must clear this even if production is worse.
#    Rationale: never ship a coin-flip model just because prod is broken.
#    Source: buddy_meta.pkl currently ships primary_accuracy=0.527. Setting the
#    floor at 0.52 means "no worse than what we already deploy" — the minimum
#    viable bar.
MIN_ABSOLUTE_ACCURACY: float = 0.52

# 2. Relative delta — candidate must beat production by at least this much.
#    Source: rules/trading.md lines 28-30 — on 2026-04-02 Buddy promoted 3 rules
#    after observing 0.02 threshold shifts produce measurable outcome changes.
#    0.02 is the empirical "signal above training noise" delta for this bot.
MIN_RELATIVE_ACCURACY_GAIN: float = 0.02

# 3. Minimum holdout sample size — don't trust small samples.
#    Source: learnings.md — Kelly fraction estimator uses a 50-trade rolling
#    window. Matching that window size keeps promotion decisions and position-
#    sizing on the same statistical footing. Larger is safer but makes retrains
#    on fresh pairs too slow to react.
MIN_HOLDOUT_SAMPLES: int = 50

# 4. Regression tolerance — how much worse is acceptable in a secondary metric?
#    Example: candidate has +1% accuracy but +40% Brier (badly calibrated). Fail.
#    Conservative default; no live-fire data yet to tune it from.
MAX_BRIER_REGRESSION: float = 0.05

# 5. Auto-promote vs human-in-loop.
#    CRITICAL: set to False because learnings.md (2026-04-15) documents that the
#    config-adjustment consumer pipeline is currently dead code — 4 CRITICAL
#    adjustments were proposed, 0 applied. Until we observe TRAINING_COMPLETED →
#    MODEL_PROMOTED event round-trips working 3+ times, promotion stays manual.
#    Flip to True only after the feedback loop is demonstrably closed.
AUTO_PROMOTE: bool = False

# 6. Multi-timeframe coverage — candidate must prove itself on the trading TF.
#    Lowered from 2→1 on 2026-04-16: the bot trades H1 exclusively. Requiring
#    it to predict M15/H4 (timeframes it wasn't trained for) blocked every real
#    retrain while a synthetic activation test (fake 2-TF pass) sat at :production.
#    M15/H4 results stay logged in per_timeframe metadata for observability.
MIN_TIMEFRAMES_PASSED: int = 1

# 7. Model freshness — hard block if candidate ensemble has any component older
#    than this many days. Source: learnings.md (2026-04-15) — 10-loss streak
#    total PL -$2,637 happened with a 28-day-old modular_ensemble and 22-day-old
#    joint_gates. Markets shift regime faster than a month. Even an accuracy-
#    matching candidate is untrustworthy if it was trained on stale data.
#    Consumed via `candidate["max_component_age_days"]`.
MAX_MODEL_AGE_DAYS: int = 14

# 8. Ensemble completeness — require EVERY model group's meta.json mtime to be
#    >= the retrain start timestamp. Source: learnings.md (2026-04-15) —
#    "PATTERN/model_staleness_persists_through_partial_retrain": autonomous
#    trainer updated transformer_direction but per-pair models stayed at March
#    19-24. System reported "RETRAINING COMPLETED" while half the ensemble was
#    still stale. Consumed via `candidate["ensemble_complete"]` bool.
REQUIRE_ENSEMBLE_COMPLETE: bool = True


# ─────────────────────────────────────────────────────────────────
# POLICY IMPLEMENTATION — the logic below consumes the 5 constants above.
# ─────────────────────────────────────────────────────────────────

def should_promote(
    candidate: Dict[str, Any],
    production: Optional[Dict[str, Any]],
) -> PromotionDecision:
    """Decide whether `candidate` should replace `production`.

    First-run case: when `production` is None (no prior production exists),
    the candidate only needs to clear the absolute floor + sample size.
    """
    diag: Dict[str, Any] = {
        "candidate": candidate,
        "production": production,
        "policy_version": "v1",
    }

    # ── Gate 1: absolute floor ──────────────────────────────────
    cand_acc = _safe_float(candidate.get("accuracy"))
    if cand_acc is None:
        return PromotionDecision(False, "candidate missing 'accuracy' metric", diag)
    if cand_acc < MIN_ABSOLUTE_ACCURACY:
        return PromotionDecision(
            False,
            f"candidate accuracy {cand_acc:.4f} < absolute floor {MIN_ABSOLUTE_ACCURACY:.4f}",
            diag,
        )

    # ── Gate 2: sample size ─────────────────────────────────────
    cand_n = _safe_int(candidate.get("holdout_samples") or candidate.get("n_samples"))
    if cand_n is None or cand_n < MIN_HOLDOUT_SAMPLES:
        return PromotionDecision(
            False,
            f"holdout samples {cand_n} < minimum {MIN_HOLDOUT_SAMPLES}",
            diag,
        )

    # ── Gate 5: multi-timeframe coverage (applies to first-run too) ─
    # Evaluated early so H1-overfit candidates fail even when there is no
    # production to compare against.
    tfs_passed = _safe_int(candidate.get("timeframes_passed"))
    if tfs_passed is not None:
        diag["timeframes_passed"] = tfs_passed
        if tfs_passed < MIN_TIMEFRAMES_PASSED:
            return PromotionDecision(
                False,
                f"multi-timeframe coverage {tfs_passed} < required {MIN_TIMEFRAMES_PASSED} "
                f"(model may be H1-overfit)",
                diag,
            )

    # ── Gate 7: model freshness (stale-regime protection) ───────
    # Derived from 2026-04-15 10-loss streak: 28d-old models predicting April
    # regime with March-trained directions. Applies to first-run too.
    max_age = _safe_float(candidate.get("max_component_age_days"))
    if max_age is not None:
        diag["max_component_age_days"] = max_age
        if max_age > MAX_MODEL_AGE_DAYS:
            return PromotionDecision(
                False,
                f"candidate ensemble has a component {max_age:.1f}d old > "
                f"MAX_MODEL_AGE_DAYS={MAX_MODEL_AGE_DAYS}. Retrain all groups before promoting.",
                diag,
            )

    # ── Gate 8: ensemble completeness (partial-retrain protection) ─
    # Catches the 2026-04-15 partial-retrain bug where transformer_direction
    # refreshed but per-pair models stayed at March mtimes. Gate is inert when
    # the verifier did not supply the flag (backward compat).
    if REQUIRE_ENSEMBLE_COMPLETE:
        complete = candidate.get("ensemble_complete")
        if complete is False:  # explicit False only; None = not checked
            missing = candidate.get("ensemble_stale_groups") or []
            diag["ensemble_stale_groups"] = missing
            return PromotionDecision(
                False,
                f"ensemble incomplete — stale groups: {missing}. Partial retrain will not be promoted.",
                diag,
            )

    # ── First-run case ──────────────────────────────────────────
    if production is None:
        return PromotionDecision(
            True,
            f"first production model: accuracy={cand_acc:.4f}, n={cand_n}",
            diag,
        )

    prod_acc = _safe_float(production.get("accuracy"))
    if prod_acc is None:
        # Production metadata is corrupt — fail safe: promote candidate because
        # the prod metrics are untrustworthy anyway.
        return PromotionDecision(
            True,
            "production metrics unreadable; promoting candidate by default",
            diag,
        )

    # ── Gate 3: relative improvement ────────────────────────────
    delta = cand_acc - prod_acc
    diag["accuracy_delta"] = delta
    if delta < MIN_RELATIVE_ACCURACY_GAIN:
        return PromotionDecision(
            False,
            f"accuracy delta {delta:+.4f} < required gain {MIN_RELATIVE_ACCURACY_GAIN:+.4f}",
            diag,
        )

    # ── Gate 4: secondary metric regression ─────────────────────
    cand_brier = _safe_float(candidate.get("brier"))
    prod_brier = _safe_float(production.get("brier"))
    if cand_brier is not None and prod_brier is not None:
        brier_delta = cand_brier - prod_brier
        diag["brier_delta"] = brier_delta
        if brier_delta > MAX_BRIER_REGRESSION:
            return PromotionDecision(
                False,
                f"Brier regression +{brier_delta:.4f} > allowed +{MAX_BRIER_REGRESSION:.4f}",
                diag,
            )

    # ── Gate 6: auto-promote vs pending ─────────────────────────
    if not AUTO_PROMOTE:
        return PromotionDecision(
            False,
            f"policy pass (Δacc={delta:+.4f}) but AUTO_PROMOTE=False — awaiting manual approval",
            {**diag, "pending_approval": True},
        )

    return PromotionDecision(
        True,
        f"promoted: Δacc={delta:+.4f}, n={cand_n}, brier_delta={diag.get('brier_delta', 'n/a')}",
        diag,
    )


# ── helpers ──────────────────────────────────────────────────────

def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        result = float(v)
        if result != result:  # NaN
            return None
        return result
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None
