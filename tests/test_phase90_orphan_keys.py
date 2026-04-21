"""Regression test: replay the 5 orphan keys from the 2026-04-16 phase 90 incident.

Root cause: self_heal_reflection wrote 5 keys to config_adjustments.json.
2 of those keys (min_confidence_threshold, atr_sl_multiplier_low_regime)
were NOT fields in ScannerConfig — setattr() created orphan attributes
that no code read, so the intended thresholds were never updated.

This test ensures every one of those 5 keys is correctly categorised
(valid or invalid) by the validator.
"""

import dataclasses
import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch


@dataclass
class _MockScannerConfig:
    """Minimal stand-in matching the fields relevant to phase 90 proposals."""
    atr_sl_multiplier: float = 1.0
    max_model_disagreement: float = 0.65
    min_confidence: float = 50.0
    weighted_vote_threshold: float = 0.45
    risk_per_trade_pct: float = 0.02


def _mock_get_fields():
    return {f.name: f for f in dataclasses.fields(_MockScannerConfig)}


def _mock_get_hints():
    import typing
    return typing.get_type_hints(_MockScannerConfig)


# ---------------------------------------------------------------------------
# Phase 90 incident proposals (from config_adjustments.json 2026-04-16)
# ---------------------------------------------------------------------------

# The 5 proposals written by self_heal_reflection on 2026-04-16.
# Format: (proposed_key, proposed_value, expected_valid, expected_field_exists)
PHASE_90_PROPOSALS = [
    # 2 confirmed DEAD orphan keys
    ("min_confidence_threshold", 68.0, False, False),   # wrong key — should be min_confidence
    ("atr_sl_multiplier_low_regime", 1.2, False, False),# wrong key — should be atr_sl_multiplier
    # 3 VALID keys that actually existed
    ("atr_sl_multiplier", 1.2, True, True),
    ("max_model_disagreement", 0.25, True, True),
    ("min_confidence", 68.0, True, True),
    # Additional proposals not in the original 5 but from same incident
    ("weighted_vote_threshold", 0.85, True, True),
    ("risk_per_trade_pct", 0.025, True, True),
]


class TestPhase90OrphanKeys(unittest.TestCase):
    """Replay every proposal from the 2026-04-16 incident and verify validator output."""

    def _validate(self, key, value):
        from src.scanner.automation.adjustment_validator import validate_adjustment
        with (
            patch(
                "src.scanner.automation.adjustment_validator._get_scanner_config_fields",
                return_value=_mock_get_fields(),
            ),
            patch(
                "src.scanner.automation.adjustment_validator._get_scanner_config_hints",
                return_value=_mock_get_hints(),
            ),
        ):
            return validate_adjustment(key, value, "phase90 incident replay")

    def test_min_confidence_threshold_is_orphan(self):
        """The original wrong key must be rejected."""
        result = self._validate("min_confidence_threshold", 68.0)
        self.assertFalse(result.valid, "min_confidence_threshold should fail — orphan key")
        self.assertFalse(result.field_exists)
        # Hint should point to the correct field
        self.assertIn("min_confidence", result.error_message)

    def test_atr_sl_multiplier_low_regime_is_orphan(self):
        """The second wrong key must be rejected."""
        result = self._validate("atr_sl_multiplier_low_regime", 1.2)
        self.assertFalse(result.valid, "atr_sl_multiplier_low_regime should fail — orphan key")
        self.assertFalse(result.field_exists)
        self.assertIn("atr_sl_multiplier", result.error_message)

    def test_valid_proposals_pass(self):
        """The 3 correct keys from the same incident should pass validation."""
        valid_proposals = [
            ("atr_sl_multiplier", 1.2),
            ("max_model_disagreement", 0.25),
            ("min_confidence", 68.0),
        ]
        for key, value in valid_proposals:
            with self.subTest(key=key):
                result = self._validate(key, value)
                self.assertTrue(result.valid, f"{key} should be valid")
                self.assertTrue(result.field_exists)

    def test_all_phase90_proposals_correctly_classified(self):
        """Exhaustive replay of all phase 90 proposals."""
        for key, value, expected_valid, expected_exists in PHASE_90_PROPOSALS:
            with self.subTest(key=key, value=value):
                result = self._validate(key, value)
                self.assertEqual(
                    result.valid, expected_valid,
                    f"key={key!r}: expected valid={expected_valid}, got {result.valid}. "
                    f"error_message={result.error_message!r}",
                )
                self.assertEqual(
                    result.field_exists, expected_exists,
                    f"key={key!r}: expected field_exists={expected_exists}",
                )

    def test_no_orphan_reaches_approved_store(self):
        """Simulate the full proposal cycle: orphan keys must be marked 'invalid'."""
        for key, value, expected_valid, _ in PHASE_90_PROPOSALS:
            with self.subTest(key=key):
                result = self._validate(key, value)
                # Derive expected status
                expected_status = "pending" if expected_valid else "invalid"
                actual_status = "pending" if result.valid else "invalid"
                self.assertEqual(actual_status, expected_status)

    def test_orphan_correction_hints_present(self):
        """Both orphan keys should provide helpful correction hints."""
        orphan_corrections = [
            ("min_confidence_threshold", "min_confidence"),
            ("atr_sl_multiplier_low_regime", "atr_sl_multiplier"),
        ]
        for orphan_key, correct_key in orphan_corrections:
            with self.subTest(orphan=orphan_key):
                result = self._validate(orphan_key, 1.0)
                self.assertIn(
                    correct_key,
                    result.error_message,
                    f"Expected correction hint '{correct_key}' in error for '{orphan_key}'",
                )


if __name__ == "__main__":
    unittest.main()
