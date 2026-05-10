"""Tests for Phase 44 disagreement_factor loosening (option 2).

The Phase 44 calibrator was over-compressing trading signals. The
`disagreement_factors` mapping in `_combine_confidence_components` was
crushing HIGH (0.8) and CRITICAL (0.6) cases far below what the empirical
loss data justified.

These tests verify the loosened mapping:
    LOW=1.0, MODERATE=0.95, HIGH=0.92, CRITICAL=0.85
applies as expected through the real `_combine_confidence_components`
code path, using a real `ConfidenceCalibrationSystem` against a real
`CalibrationConfig` pointing at real disk under `tmp_path`.

NO MOCKS — per `.claude/rules/improvement.md` "No-Mock Rule" (2026-05-01).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.scanner.confidence_calibration import (
    CalibrationConfig,
    ConfidenceCalibrationSystem,
)


# Reference values per Phase 44 option-2 loosening (2026-05-09).
EXPECTED_FACTORS = {
    "LOW": 1.0,
    "MODERATE": 0.95,
    "HIGH": 0.92,
    "CRITICAL": 0.85,
}


@pytest.fixture
def system(tmp_path: Path) -> ConfidenceCalibrationSystem:
    """Real ConfidenceCalibrationSystem backed by real disk under tmp_path."""
    cfg = CalibrationConfig(
        calibration_file=str(tmp_path / "calibration.json"),
    )
    return ConfidenceCalibrationSystem(cfg)


def _final(
    system: ConfidenceCalibrationSystem,
    *,
    platt_calibrated: float,
    disagreement_level: str,
    agent_agreement: float = 1.0,
    meta_confidence: float = 1.0,
) -> float:
    """Drive the real `_combine_confidence_components` path.

    With agent_agreement=1.0 and meta_confidence=1.0:
        agreement_factor = max(0.5, 1.0) = 1.0
        meta_weight = 0.5 + 0.5 * 1.0 = 1.0
    so the returned final == platt_calibrated * disagreement_factor,
    which lets us recover disagreement_factor exactly.
    """
    return system._combine_confidence_components(
        platt_calibrated=platt_calibrated,
        agent_agreement=agent_agreement,
        disagreement_level=disagreement_level,
        meta_confidence=meta_confidence,
    )


@pytest.mark.parametrize("level,expected", list(EXPECTED_FACTORS.items()))
def test_disagreement_factor_per_level(
    system: ConfidenceCalibrationSystem, level: str, expected: float
) -> None:
    """Each level produces the exact loosened factor through the real path."""
    platt = 0.70
    final = _final(system, platt_calibrated=platt, disagreement_level=level)
    # final == platt * expected_factor (other factors == 1.0 by construction)
    assert final == pytest.approx(platt * expected, abs=1e-9), (
        f"{level}: expected disagreement_factor={expected}, "
        f"got effective factor={final / platt:.4f}"
    )


def test_unknown_level_falls_back_to_high_default(
    system: ConfidenceCalibrationSystem,
) -> None:
    """An unrecognized level uses the new 0.92 fallback (was 0.8)."""
    platt = 0.70
    final = _final(system, platt_calibrated=platt, disagreement_level="GIBBERISH")
    assert final == pytest.approx(platt * 0.92, abs=1e-9)


def test_high_disagreement_floor_above_old_value(
    system: ConfidenceCalibrationSystem,
) -> None:
    """HIGH-disagreement final stays at >= 0.92 * platt (vs the old 0.8 floor).

    Smoke test the operator-described scenario: with HIGH disagreement and
    agreement/meta neutralized, the final must clear the previous HIGH-tier
    crush ceiling.
    """
    platt = 0.70
    final = _final(system, platt_calibrated=platt, disagreement_level="HIGH")
    assert final >= 0.92 * platt - 1e-9
    # And — critically — strictly above the old 0.8 * platt cap.
    assert final > 0.8 * platt + 1e-6


def test_loosening_versus_old_mapping_at_raw_0p582(
    system: ConfidenceCalibrationSystem,
) -> None:
    """At platt=0.582 with HIGH disagreement, new mapping >> old mapping.

    Reproduces the operator's worked example from the Phase 44 brief. With
    agent_agreement=1.0 and meta_confidence=1.0, disagreement_factor is the
    only attenuator. New floor (0.92) must exceed old floor (0.8) by ~12%.
    """
    platt = 0.582
    new_final = _final(system, platt_calibrated=platt, disagreement_level="HIGH")
    expected_new = platt * 0.92
    expected_old = platt * 0.8
    assert new_final == pytest.approx(expected_new, abs=1e-9)
    assert new_final > expected_old + 1e-6


def test_ordering_invariant_preserved(
    system: ConfidenceCalibrationSystem,
) -> None:
    """LOW > MODERATE > HIGH > CRITICAL ordering survives the loosening."""
    platt = 0.70
    finals = {
        level: _final(system, platt_calibrated=platt, disagreement_level=level)
        for level in EXPECTED_FACTORS
    }
    assert finals["LOW"] > finals["MODERATE"] > finals["HIGH"] > finals["CRITICAL"]


def test_low_is_neutral(system: ConfidenceCalibrationSystem) -> None:
    """LOW disagreement still passes platt through unchanged (factor 1.0)."""
    platt = 0.65
    final = _final(system, platt_calibrated=platt, disagreement_level="LOW")
    assert final == pytest.approx(platt, abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
