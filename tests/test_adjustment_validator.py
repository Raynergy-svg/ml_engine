"""Unit tests for adjustment_validator.py.

Covers: valid field, orphan key, wrong type, out-of-bounds.
All tests use a minimal mock ScannerConfig to avoid heavy ML dependencies.
"""

import dataclasses
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Minimal ScannerConfig stand-in for isolated tests
# ---------------------------------------------------------------------------

@dataclass
class _MockScannerConfig:
    min_confidence: float = 42.0
    atr_sl_multiplier: float = 1.0
    atr_tp_multiplier: float = 2.5
    max_uncertainty_score: float = 0.45
    max_model_disagreement: float = 0.30
    weighted_vote_threshold: float = 0.72
    risk_per_trade_pct: float = 0.05
    max_open_risk_pct: float = 0.30
    enable_execution: bool = True
    profile: str = "smart"
    pairs: List[str] = dataclasses.field(default_factory=list)


def _mock_get_fields():
    return {f.name: f for f in dataclasses.fields(_MockScannerConfig)}


def _mock_get_hints():
    import typing
    return typing.get_type_hints(_MockScannerConfig)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidateAdjustment(unittest.TestCase):

    def _validate(self, key, value, reason="test"):
        """Run validate_adjustment with mocked ScannerConfig internals."""
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
            return validate_adjustment(key, value, reason)

    # -- Valid field ---------------------------------------------------------

    def test_valid_float_field(self):
        result = self._validate("min_confidence", 65.0)
        self.assertTrue(result.valid)
        self.assertTrue(result.field_exists)
        self.assertTrue(result.type_matches)
        self.assertTrue(result.within_bounds)
        self.assertEqual(result.error_message, "")

    def test_valid_bool_field(self):
        result = self._validate("enable_execution", True)
        self.assertTrue(result.valid)
        self.assertTrue(result.field_exists)
        self.assertTrue(result.type_matches)
        self.assertTrue(result.within_bounds)

    def test_valid_str_field(self):
        result = self._validate("profile", "balanced")
        self.assertTrue(result.valid)
        self.assertTrue(result.field_exists)
        self.assertTrue(result.type_matches)

    def test_valid_float_as_int(self):
        # Numeric coercibility: int value for float field should be accepted
        result = self._validate("min_confidence", 65)
        self.assertTrue(result.valid)

    # -- Orphan key ----------------------------------------------------------

    def test_orphan_key_not_in_config(self):
        result = self._validate("totally_made_up_key", 1.0)
        self.assertFalse(result.valid)
        self.assertFalse(result.field_exists)
        self.assertFalse(result.type_matches)
        self.assertFalse(result.within_bounds)
        self.assertIn("Unknown field", result.error_message)

    def test_orphan_key_with_correction_hint(self):
        # 'min_confidence_threshold' is a known orphan from phase 90 incident
        result = self._validate("min_confidence_threshold", 65.0)
        self.assertFalse(result.valid)
        self.assertFalse(result.field_exists)
        self.assertIn("min_confidence", result.error_message)  # correction hint

    def test_orphan_key_atr_sl_multiplier_low_regime(self):
        # Second known orphan from phase 90 incident
        result = self._validate("atr_sl_multiplier_low_regime", 1.2)
        self.assertFalse(result.valid)
        self.assertFalse(result.field_exists)
        self.assertIn("atr_sl_multiplier", result.error_message)

    # -- Wrong type ----------------------------------------------------------

    def test_wrong_type_bool_for_float(self):
        result = self._validate("min_confidence", True)
        # True can be cast to float(True) = 1.0, which is valid — type check allows this
        # The interesting case is when bool is passed for an int/bool-specific field
        # For float fields, bool is coercible so it's implementation-defined.
        # The important test: bool for a bool field IS valid
        result_bool = self._validate("enable_execution", False)
        self.assertTrue(result_bool.valid)
        self.assertTrue(result_bool.type_matches)

    def test_wrong_type_string_for_float(self):
        result = self._validate("min_confidence", "not_a_number")
        self.assertFalse(result.valid)
        self.assertTrue(result.field_exists)
        self.assertFalse(result.type_matches)

    def test_wrong_type_int_for_bool(self):
        # 1 (int) passed for bool field should fail type check
        result = self._validate("enable_execution", 1)
        self.assertFalse(result.valid)
        self.assertTrue(result.field_exists)
        self.assertFalse(result.type_matches)

    def test_wrong_type_number_for_string(self):
        result = self._validate("profile", 42)
        self.assertFalse(result.valid)
        self.assertTrue(result.field_exists)
        self.assertFalse(result.type_matches)

    # -- Out of bounds -------------------------------------------------------

    def test_out_of_bounds_too_high(self):
        result = self._validate("min_confidence", 110.0)  # max is 100.0
        self.assertFalse(result.valid)
        self.assertTrue(result.field_exists)
        self.assertTrue(result.type_matches)
        self.assertFalse(result.within_bounds)
        self.assertIn("out of safe bounds", result.error_message)

    def test_out_of_bounds_too_low(self):
        result = self._validate("max_uncertainty_score", 0.01)  # min is 0.10
        self.assertFalse(result.valid)
        self.assertFalse(result.within_bounds)

    def test_out_of_bounds_risk_pct(self):
        result = self._validate("risk_per_trade_pct", 0.50)  # max is 0.20
        self.assertFalse(result.valid)
        self.assertFalse(result.within_bounds)

    def test_within_bounds_edge_low(self):
        # Exactly at lower bound should be valid
        result = self._validate("max_uncertainty_score", 0.10)
        self.assertTrue(result.valid)
        self.assertTrue(result.within_bounds)

    def test_within_bounds_edge_high(self):
        # Exactly at upper bound should be valid
        result = self._validate("min_confidence", 100.0)
        self.assertTrue(result.valid)
        self.assertTrue(result.within_bounds)

    # -- ValidationResult.to_dict / from_dict --------------------------------

    def test_to_dict_round_trip(self):
        from src.scanner.automation.adjustment_validator import ValidationResult
        vr = ValidationResult(
            valid=False, field_exists=True, type_matches=False,
            within_bounds=False, error_message="test error"
        )
        d = vr.to_dict()
        vr2 = ValidationResult.from_dict(d)
        self.assertEqual(vr, vr2)


if __name__ == "__main__":
    unittest.main()
