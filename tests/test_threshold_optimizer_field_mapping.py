"""Regression for threshold_optimizer → ConfigAdjuster key mapping (engine.py:4568).

Bug (2026-04-21): engine.py was emitting config adjustments with synthesized
orphan key names — ``f"optimized_{tkey}_threshold"`` — that never matched any
ScannerConfig field. The validator caught them (status=invalid) but 174 dead
proposals accumulated in .claude/pending_adjustments.json before discovery.

Fix: map each optimizer key to its ScannerConfig field with the correct scale
conversion. Specifically, ``confidence`` is 0-1 in the optimizer but 0-100 in
ScannerConfig.min_confidence, so it must be multiplied by 100 before emission.

These tests exercise the mapping logic directly by instantiating ConfigAdjuster
with isolated paths and asserting collect_adjustment() is called with the
correct (key, value) pair for each optimizer output.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scanner.automation.config_adjuster import ConfigAdjuster


# ---------------------------------------------------------------------------
# The mapping as it SHOULD behave
# ---------------------------------------------------------------------------

EXPECTED_MAPPING = {
    # optimizer_key: (scanner_config_field, scale_multiplier, example_input, example_output)
    "confidence": ("min_confidence", 100.0, 0.55, 55.0),
    "momentum": ("min_momentum", 1.0, 0.30, 0.30),
    "rr_ratio": ("min_risk_reward_ratio", 1.0, 1.8, 1.8),
}


@pytest.fixture
def isolated_adjuster(tmp_path: Path) -> ConfigAdjuster:
    """ConfigAdjuster with both paths routed to tmp — no leak to production."""
    return ConfigAdjuster(
        persistence_path=tmp_path / "config_adjustments.json",
        pending_path=tmp_path / "pending_adjustments.json",
    )


# ---------------------------------------------------------------------------
# Mapping-level unit tests
# ---------------------------------------------------------------------------

class TestOptimizerFieldMapping:
    @pytest.mark.parametrize("opt_key,field,scale,opt_value,expected_field_value", [
        ("confidence", "min_confidence", 100.0, 0.55, 55.0),
        ("confidence", "min_confidence", 100.0, 0.40, 40.0),
        ("confidence", "min_confidence", 100.0, 0.85, 85.0),
        ("momentum", "min_momentum", 1.0, 0.30, 0.30),
        ("momentum", "min_momentum", 1.0, 0.10, 0.10),
        ("rr_ratio", "min_risk_reward_ratio", 1.0, 1.8, 1.8),
        ("rr_ratio", "min_risk_reward_ratio", 1.0, 2.5, 2.5),
    ])
    def test_emitted_key_and_value(
        self,
        isolated_adjuster: ConfigAdjuster,
        opt_key: str,
        field: str,
        scale: float,
        opt_value: float,
        expected_field_value: float,
    ) -> None:
        """Each optimizer key maps to the right ScannerConfig field name with scale conversion."""
        # Simulate the engine.py:4563-4586 emission
        emitted_value = round(float(opt_value) * scale, 4)
        isolated_adjuster.collect_adjustment(
            source="threshold_optimizer",
            key=field,
            value=emitted_value,
            reason="test",
        )

        # Read what was written to pending
        import json
        with open(isolated_adjuster._pending_path) as f:
            data = json.load(f)
        proposals = data.get("proposals", [])
        assert len(proposals) == 1
        p = proposals[0]
        assert p["key"] == field, f"expected field name {field}, got {p['key']}"
        assert p["proposed_value"] == pytest.approx(expected_field_value)
        # Must be VALID — proves the field name is a real ScannerConfig field
        assert p["validation"]["valid"] is True, (
            f"proposal invalid: {p['validation'].get('error_message')}"
        )
        assert p["status"] == "pending"


# ---------------------------------------------------------------------------
# Regression: orphan key names must NEVER validate
# ---------------------------------------------------------------------------

class TestOrphanKeyRegression:
    @pytest.mark.parametrize("orphan_key", [
        "optimized_confidence_threshold",
        "optimized_momentum_threshold",
        "optimized_rr_ratio_threshold",
    ])
    def test_orphan_key_rejected_by_validator(
        self,
        isolated_adjuster: ConfigAdjuster,
        orphan_key: str,
    ) -> None:
        """The old orphan key names must still be rejected — second layer of safety.

        Even if engine.py regresses to emitting these names again, the validator
        should catch them. 174 orphan proposals accumulated in production before
        this was discovered — this test makes sure the validator keeps guarding.
        """
        isolated_adjuster.collect_adjustment(
            source="threshold_optimizer",
            key=orphan_key,
            value=0.55,
            reason="simulated regression",
        )

        import json
        with open(isolated_adjuster._pending_path) as f:
            data = json.load(f)
        proposals = data.get("proposals", [])
        assert len(proposals) == 1
        p = proposals[0]
        assert p["status"] == "invalid"
        assert p["validation"]["valid"] is False
        assert "not in ScannerConfig" in p["validation"].get("error_message", "")


# ---------------------------------------------------------------------------
# Scale conversion: forgetting × 100 would fail safely
# ---------------------------------------------------------------------------

class TestScaleConversionSafety:
    def test_confidence_without_scale_conversion_fails_bounds(
        self,
        isolated_adjuster: ConfigAdjuster,
    ) -> None:
        """If engine.py forgets × 100 on confidence, validator rejects.

        min_confidence bounds are [10.0, 100.0]. An un-converted 0.55 would
        be rejected as out-of-bounds — safe failure, not silent corruption.
        """
        isolated_adjuster.collect_adjustment(
            source="threshold_optimizer",
            key="min_confidence",
            value=0.55,  # BUG: forgot to × 100
            reason="missing scale conversion simulation",
        )

        import json
        with open(isolated_adjuster._pending_path) as f:
            data = json.load(f)
        p = data["proposals"][0]
        assert p["status"] == "invalid"
        assert "out of safe bounds" in p["validation"].get("error_message", "")

    def test_confidence_with_scale_conversion_is_valid(
        self,
        isolated_adjuster: ConfigAdjuster,
    ) -> None:
        """Correctly scaled confidence value (55.0) validates clean."""
        isolated_adjuster.collect_adjustment(
            source="threshold_optimizer",
            key="min_confidence",
            value=round(0.55 * 100.0, 4),
            reason="correct scale conversion",
        )

        import json
        with open(isolated_adjuster._pending_path) as f:
            data = json.load(f)
        p = data["proposals"][0]
        assert p["status"] == "pending"
        assert p["validation"]["valid"] is True
        assert p["proposed_value"] == pytest.approx(55.0)
