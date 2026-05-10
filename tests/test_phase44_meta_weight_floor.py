"""Phase 44 — Option 4: meta_weight floor raised from 0.5 to 0.7.

Old formula: meta_weight = 0.5 + 0.5 * meta_confidence  →  range [0.5, 1.0]
New formula: meta_weight = 0.7 + 0.3 * meta_confidence  →  range [0.7, 1.0]

Rationale: when the trade journal is sparse/empty (meta_confidence ≈ 0.5),
the old formula compressed final confidence by 24% (× 0.76) on top of the
agreement and disagreement factors. With the journal archived earlier,
meta=0.52 was the live observed value — every signal was suffering this
extra compression. Raising the floor reduces the journal-emptiness
penalty without removing the meta-confidence signal entirely.

NO MOCKS — per CLAUDE.md No-Mock Rule (promoted 2026-05-01). All tests
construct real `ConfidenceCalibrationSystem` against an isolated tmp_path
calibration file.
"""

from __future__ import annotations

import math

import pytest

from src.scanner.confidence_calibration import (
    CalibrationConfig,
    ConfidenceCalibrationSystem,
)


def _make_system(tmp_path) -> ConfidenceCalibrationSystem:
    """Real system with isolated calibration file (no mocks)."""
    cfg = CalibrationConfig(
        calibration_file=str(tmp_path / "calibration.json"),
    )
    return ConfidenceCalibrationSystem(config=cfg)


def test_meta_zero_applies_floor_07(tmp_path):
    """meta_confidence=0.0 → final = platt × agreement × disagreement × 0.7 (was 0.5)."""
    system = _make_system(tmp_path)
    platt = 0.80
    agreement = 0.90  # > 0.5 floor, used as-is
    disagreement_level = "LOW"  # factor = 1.0
    meta = 0.0

    final = system._combine_confidence_components(
        platt_calibrated=platt,
        agent_agreement=agreement,
        disagreement_level=disagreement_level,
        meta_confidence=meta,
    )

    expected = platt * agreement * 1.0 * 0.7
    assert math.isclose(final, expected, rel_tol=1e-9), (
        f"floor regression: meta=0 should yield weight 0.7, got final={final:.6f} "
        f"vs expected={expected:.6f}"
    )
    # Sanity: old formula would have produced platt*agreement*1.0*0.5
    old = platt * agreement * 1.0 * 0.5
    assert final > old, f"new floor must exceed old floor; new={final}, old={old}"


def test_meta_one_unchanged_at_10(tmp_path):
    """meta_confidence=1.0 → meta_weight=1.0 (top of range, identical to old formula)."""
    system = _make_system(tmp_path)
    platt = 0.80
    agreement = 0.90
    disagreement_level = "LOW"
    meta = 1.0

    final = system._combine_confidence_components(
        platt_calibrated=platt,
        agent_agreement=agreement,
        disagreement_level=disagreement_level,
        meta_confidence=meta,
    )

    expected = platt * agreement * 1.0 * 1.0
    assert math.isclose(final, expected, rel_tol=1e-9), (
        f"top-of-range regression: meta=1 should yield weight 1.0, got final={final:.6f}"
    )


def test_meta_half_yields_weight_085(tmp_path):
    """meta_confidence=0.5 → meta_weight = 0.7 + 0.3*0.5 = 0.85 (was 0.75)."""
    system = _make_system(tmp_path)
    platt = 0.80
    agreement = 0.90
    disagreement_level = "LOW"  # factor = 1.0
    meta = 0.5

    final = system._combine_confidence_components(
        platt_calibrated=platt,
        agent_agreement=agreement,
        disagreement_level=disagreement_level,
        meta_confidence=meta,
    )

    expected_weight = 0.85
    expected_final = platt * agreement * 1.0 * expected_weight
    assert math.isclose(final, expected_final, rel_tol=1e-9), (
        f"linear interpolation broke: meta=0.5 should yield weight 0.85, "
        f"got final={final:.6f} vs expected={expected_final:.6f}"
    )


def test_smoke_live_observed_values(tmp_path):
    """Smoke test against the live observed values that motivated the change.

    Live observation (journal archived, meta=0.52):
      platt=0.582, agreement=0.5, disagreement=HIGH (0.92), meta=0.52
      old final = 0.582 × 0.5 × 0.92 × (0.5 + 0.5*0.52) = 0.582 × 0.5 × 0.92 × 0.76
                ≈ 0.20355  (over the full chain, with agreement floor)
      new final = 0.582 × 0.5 × 0.92 × (0.7 + 0.3*0.52) = 0.582 × 0.5 × 0.92 × 0.856
                ≈ 0.22929

    Lift ratio = 0.856 / 0.76 ≈ 1.126× (12.6% raise on the same input).
    """
    system = _make_system(tmp_path)
    platt = 0.582
    agreement = 0.5  # at the agreement_factor floor
    disagreement_level = "HIGH"  # factor = 0.92
    meta = 0.52

    final = system._combine_confidence_components(
        platt_calibrated=platt,
        agent_agreement=agreement,
        disagreement_level=disagreement_level,
        meta_confidence=meta,
    )

    # Component-by-component truth: agreement_factor = max(0.5, 0.5) = 0.5
    new_meta_weight = 0.7 + 0.3 * meta  # = 0.856
    expected_final = platt * 0.5 * 0.92 * new_meta_weight  # ≈ 0.22929
    assert math.isclose(final, expected_final, rel_tol=1e-9), (
        f"smoke regression: live-values final={final:.6f} vs expected={expected_final:.6f}"
    )

    # Lift vs old formula
    old_meta_weight = 0.5 + 0.5 * meta  # = 0.76
    old_final = platt * 0.5 * 0.92 * old_meta_weight
    lift_ratio = final / old_final
    assert math.isclose(lift_ratio, new_meta_weight / old_meta_weight, rel_tol=1e-9)
    assert lift_ratio > 1.12 and lift_ratio < 1.14, (
        f"expected ~1.126× lift, got {lift_ratio:.4f}× "
        f"(new={final:.6f}, old={old_final:.6f})"
    )


def test_range_endpoints_match_docstring(tmp_path):
    """The new formula's range is [0.7, 1.0]. Verify both endpoints exactly."""
    system = _make_system(tmp_path)
    # Use platt=1.0, agreement=1.0, disagreement=LOW(1.0) so final == meta_weight.
    base_kwargs = dict(
        platt_calibrated=1.0,
        agent_agreement=1.0,
        disagreement_level="LOW",
    )
    weight_at_zero = system._combine_confidence_components(meta_confidence=0.0, **base_kwargs)
    weight_at_one = system._combine_confidence_components(meta_confidence=1.0, **base_kwargs)
    assert math.isclose(weight_at_zero, 0.7, rel_tol=1e-9), (
        f"floor not 0.7: got {weight_at_zero:.6f}"
    )
    assert math.isclose(weight_at_one, 1.0, rel_tol=1e-9), (
        f"ceiling not 1.0: got {weight_at_one:.6f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
