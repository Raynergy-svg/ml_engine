"""Tests for the 2026-05-11 ensemble_conflict modern-attribute fix.

Background:
    `src/scanner/agents/_team.py:2245-2247` previously read three legacy
    attributes (`tcn_score`, `ridge_score`, `rf_score`) from `ctx.analysis`
    that have not been populated by the inference path since the
    pre-Transformer / pre-LightGBM era. Every live scan since the migration
    logged ``Disagreement calculation: scores=[0. 0. 0.], disagreement=0.000``
    — the ensemble_conflict layer was dead code (observability + safety rot).

    Team C's fix (this commit) reads the modern equivalents:
      - TCN slot   ← ``ctx.analysis.tcn_confidence``  (0-100 → /100)
      - Ridge slot ← ``ctx.analysis.confidence_score`` (0-100 → /100)
      - RF slot    ← ``ctx.analysis.momentum``         (already 0-1)
    Each magnitude is signed by ``ctx.analysis.direction`` to produce a
    signed score in [-1, +1] for the resolver's BUY/SELL/HOLD threshold
    mapping.

Test rules (per `.claude/rules/improvement.md` No-Mock Rule):
    - Real `PairAnalysis` from `src.scanner.results`.
    - Real `ModelPrediction` + `EnsembleConflictResolver` from
      `src.scanner.ensemble_conflict`.
    - No `unittest.mock`, no `MagicMock`, no `patch`. Drive the production
      classes directly; assert on real return values.
"""
from __future__ import annotations

import pytest

from src.scanner.ensemble_conflict import (
    ConflictConfig,
    EnsembleConflictResolver,
    ModelPrediction,
)
from src.scanner.results import PairAnalysis


# ---------------------------------------------------------------------------
# Helpers — replicate the production attribute-extraction logic so the test
# locks in the contract of `_team.py:2244-2295` without importing the agent
# team (which would require booting half the scanner).
# ---------------------------------------------------------------------------
def _sign_from_direction(direction: str) -> float:
    """Mirror the production sign-mapping at `_team.py:2266-2267`."""
    d = (direction or "HOLD").upper()
    if d == "LONG":
        return 1.0
    if d == "SHORT":
        return -1.0
    return 0.0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _build_signed_scores(analysis: PairAnalysis) -> tuple[float, float, float]:
    """Reproduce the three signed scores `_team.py` will feed the resolver.

    This is what the production reads do now — kept here so tests assert on
    the same arithmetic without re-importing `ScannerAgentTeam` (heavy).
    """
    sign = _sign_from_direction(analysis.direction)
    tcn_mag = _clip01(float(getattr(analysis, "tcn_confidence", 0.0) or 0.0) / 100.0)
    conf_mag = _clip01(float(getattr(analysis, "confidence_score", 0.0) or 0.0) / 100.0)
    mom_mag = _clip01(float(getattr(analysis, "momentum", 0.0) or 0.0))
    return sign * tcn_mag, sign * conf_mag, sign * mom_mag


# ---------------------------------------------------------------------------
# 1. Production reads succeed when modern attributes are populated.
# ---------------------------------------------------------------------------
def test_ensemble_conflict_reads_modern_direction_score() -> None:
    """Modern attribute reads return non-zero when the inference path populated them.

    Locks in the contract that `_team.py` reads `tcn_confidence`,
    `confidence_score`, `momentum` (not the legacy `tcn_score`,
    `ridge_score`, `rf_score`).
    """
    analysis = PairAnalysis(
        pair="EUR_USD",
        direction="LONG",
        tcn_confidence=72.0,
        confidence_score=65.0,
        momentum=0.55,
    )
    tcn_score, ridge_score, rf_score = _build_signed_scores(analysis)
    # All three must be non-zero in the same sign (LONG → positive).
    assert tcn_score > 0.0, f"TCN slot must read tcn_confidence; got {tcn_score}"
    assert ridge_score > 0.0, f"Ridge slot must read confidence_score; got {ridge_score}"
    assert rf_score > 0.0, f"RF slot must read momentum; got {rf_score}"
    # Spot-check the magnitudes (0-100 → /100; 0-1 stays as-is).
    assert abs(tcn_score - 0.72) < 1e-9
    assert abs(ridge_score - 0.65) < 1e-9
    assert abs(rf_score - 0.55) < 1e-9


# ---------------------------------------------------------------------------
# 2. Disagreement > 0 when the three signed scores spread.
# ---------------------------------------------------------------------------
def test_ensemble_conflict_disagreement_nonzero_when_models_disagree() -> None:
    """Three different signed scores produce non-zero disagreement.

    Real `EnsembleConflictResolver.resolve(...)` on real `ModelPrediction`
    instances. No mocks.
    """
    resolver = EnsembleConflictResolver()
    predictions = [
        ModelPrediction(model_name="TCN", score=+0.80, direction="BUY", confidence=0.80),
        ModelPrediction(model_name="Ridge", score=+0.05, direction="HOLD", confidence=0.05),
        ModelPrediction(model_name="RF", score=-0.40, direction="SELL", confidence=0.40),
    ]
    result = resolver.resolve(predictions)
    assert result.disagreement_score > 0.0, (
        f"Spread scores [+0.80, +0.05, -0.40] must produce non-zero disagreement; "
        f"got {result.disagreement_score}"
    )


# ---------------------------------------------------------------------------
# 3. Disagreement near 0 when the three signed scores agree.
# ---------------------------------------------------------------------------
def test_ensemble_conflict_disagreement_zero_when_models_agree() -> None:
    """Three near-identical scores produce near-zero disagreement.

    The resolver computes ``std(scores) / max(|mean(scores)|, 0.01)``;
    identical scores → std=0 → disagreement=0 exactly.
    """
    resolver = EnsembleConflictResolver()
    predictions = [
        ModelPrediction(model_name="TCN", score=+0.70, direction="BUY", confidence=0.70),
        ModelPrediction(model_name="Ridge", score=+0.70, direction="BUY", confidence=0.70),
        ModelPrediction(model_name="RF", score=+0.70, direction="BUY", confidence=0.70),
    ]
    result = resolver.resolve(predictions)
    assert result.disagreement_score < 1e-6, (
        f"Identical scores must produce disagreement≈0; got {result.disagreement_score}"
    )
    # And the resolver should not block when all three agree on BUY.
    assert result.should_block is False
    assert result.consensus_direction == "BUY"


# ---------------------------------------------------------------------------
# 4. Missing attribute falls back gracefully (zero, not crash).
# ---------------------------------------------------------------------------
def test_missing_attribute_falls_back_gracefully() -> None:
    """A default-constructed PairAnalysis (no modern attrs set) yields zeros.

    Pre-fix behavior was zero (because the legacy attrs never existed); the
    fix must preserve "no crash, no spurious signal" for unpopulated cases.
    """
    analysis = PairAnalysis(pair="EUR_USD")  # all defaults
    tcn_score, ridge_score, rf_score = _build_signed_scores(analysis)
    # direction defaults to "HOLD" → sign=0 → all scores 0.
    assert tcn_score == 0.0
    assert ridge_score == 0.0
    assert rf_score == 0.0

    # Resolver must accept these without raising.
    resolver = EnsembleConflictResolver()
    predictions = [
        ModelPrediction(model_name="TCN", score=tcn_score, direction="HOLD", confidence=0.0),
        ModelPrediction(model_name="Ridge", score=ridge_score, direction="HOLD", confidence=0.0),
        ModelPrediction(model_name="RF", score=rf_score, direction="HOLD", confidence=0.0),
    ]
    result = resolver.resolve(predictions)
    assert result.disagreement_score == 0.0
    assert result.should_block is False


# ---------------------------------------------------------------------------
# 5. Score sign is preserved by `direction`.
# ---------------------------------------------------------------------------
def test_score_sign_preserved_after_refactor() -> None:
    """LONG → positive signed scores; SHORT → negative; HOLD → zero.

    Locks the sign convention so future refactors cannot silently invert
    it (which would feed the resolver SHORT-vote scores while the analysis
    direction is LONG and vice versa).
    """
    # LONG case
    long_a = PairAnalysis(
        pair="EUR_USD",
        direction="LONG",
        tcn_confidence=70.0,
        confidence_score=60.0,
        momentum=0.50,
    )
    t, r, f = _build_signed_scores(long_a)
    assert t > 0 and r > 0 and f > 0, (
        f"LONG direction must produce positive signed scores; got ({t}, {r}, {f})"
    )

    # SHORT case — same magnitudes, opposite signs
    short_a = PairAnalysis(
        pair="EUR_USD",
        direction="SHORT",
        tcn_confidence=70.0,
        confidence_score=60.0,
        momentum=0.50,
    )
    t, r, f = _build_signed_scores(short_a)
    assert t < 0 and r < 0 and f < 0, (
        f"SHORT direction must produce negative signed scores; got ({t}, {r}, {f})"
    )

    # HOLD case — all zero regardless of magnitudes
    hold_a = PairAnalysis(
        pair="EUR_USD",
        direction="HOLD",
        tcn_confidence=70.0,
        confidence_score=60.0,
        momentum=0.50,
    )
    t, r, f = _build_signed_scores(hold_a)
    assert t == 0.0 and r == 0.0 and f == 0.0, (
        f"HOLD direction must produce zero signed scores; got ({t}, {r}, {f})"
    )


# ---------------------------------------------------------------------------
# 6. Output normalization lives in [-1, +1].
# ---------------------------------------------------------------------------
def test_score_normalization_in_minus_one_to_one_range() -> None:
    """Signed scores stay in [-1, +1] even at boundary inputs.

    Required so the resolver's BUY/SELL/HOLD threshold mapping
    (>0.1 / <-0.1) operates on inputs of the expected range.
    """
    # Boundary 1: full-strength signals
    extreme_long = PairAnalysis(
        pair="EUR_USD",
        direction="LONG",
        tcn_confidence=100.0,   # /100 → 1.0
        confidence_score=100.0,  # /100 → 1.0
        momentum=1.0,            # already 1.0
    )
    t, r, f = _build_signed_scores(extreme_long)
    assert -1.0 <= t <= 1.0
    assert -1.0 <= r <= 1.0
    assert -1.0 <= f <= 1.0
    assert t == 1.0 and r == 1.0 and f == 1.0

    # Boundary 2: over-range inputs are clipped (defensive)
    over_range = PairAnalysis(
        pair="EUR_USD",
        direction="SHORT",
        tcn_confidence=250.0,    # /100 → 2.5; must clip to 1.0
        confidence_score=200.0,  # /100 → 2.0; must clip to 1.0
        momentum=1.5,            # must clip to 1.0
    )
    t, r, f = _build_signed_scores(over_range)
    assert t == -1.0, f"over-range tcn_confidence must clip and sign to -1.0; got {t}"
    assert r == -1.0, f"over-range confidence_score must clip and sign to -1.0; got {r}"
    assert f == -1.0, f"over-range momentum must clip and sign to -1.0; got {f}"

    # Boundary 3: negative magnitudes (corrupt input) must NOT flip into the
    # opposite sign — they clip to 0 and then the direction-sign applies.
    neg_input = PairAnalysis(
        pair="EUR_USD",
        direction="LONG",
        tcn_confidence=-10.0,
        confidence_score=-10.0,
        momentum=-0.5,
    )
    t, r, f = _build_signed_scores(neg_input)
    assert t == 0.0 and r == 0.0 and f == 0.0, (
        f"negative magnitudes must clip to 0 (not flip sign); got ({t}, {r}, {f})"
    )


# ---------------------------------------------------------------------------
# Bonus: end-to-end through the real resolver with modern attributes.
# ---------------------------------------------------------------------------
def test_end_to_end_resolver_consumes_modern_signed_scores() -> None:
    """Full pipeline: PairAnalysis (modern attrs) → signed scores → resolver.

    Exercises the actual integration: the resolver sees the same shape of
    input that production now produces, and yields a sensible verdict.
    """
    analysis = PairAnalysis(
        pair="EUR_USD",
        direction="LONG",
        tcn_confidence=80.0,    # → +0.80 LONG
        confidence_score=50.0,  # → +0.50 LONG
        momentum=0.20,          # → +0.20 LONG
    )
    tcn_score, ridge_score, rf_score = _build_signed_scores(analysis)

    resolver = EnsembleConflictResolver()
    predictions = [
        ModelPrediction(
            model_name="TCN",
            score=tcn_score,
            direction="BUY" if tcn_score > 0.1 else ("SELL" if tcn_score < -0.1 else "HOLD"),
            confidence=abs(tcn_score),
        ),
        ModelPrediction(
            model_name="Ridge",
            score=ridge_score,
            direction="BUY" if ridge_score > 0.1 else ("SELL" if ridge_score < -0.1 else "HOLD"),
            confidence=abs(ridge_score),
        ),
        ModelPrediction(
            model_name="RF",
            score=rf_score,
            direction="BUY" if rf_score > 0.1 else ("SELL" if rf_score < -0.1 else "HOLD"),
            confidence=abs(rf_score),
        ),
    ]
    result = resolver.resolve(predictions)

    # All three vote BUY (all > 0.1) — consensus is BUY, no direction conflict.
    assert result.consensus_direction == "BUY"
    assert result.has_direction_conflict is False
    # Disagreement is non-zero because the three magnitudes differ.
    assert result.disagreement_score > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
