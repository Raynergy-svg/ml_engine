"""Unit tests for DynamicDrawdownManager (src/scanner/automation/dynamic_drawdown.py).

Closes a zero-coverage gap surfaced by the 2026-05-01 test coverage audit.

The manager maps a drawdown severity = current_dd / max_dd into one of three
modes:
  - severity > protective_threshold (default 0.6) → "protective": tighter trailing
  - severity < aggressive_threshold (default 0.3) → "aggressive": wider trailing
  - in between → "normal": defaults

Both protective and aggressive modes interpolate linearly between the normal
defaults and their respective endpoint values, so the boundary cases must be
continuous (no jump at threshold). Severity is clamped to [0, 1].
"""

from __future__ import annotations

import pytest

from src.scanner.automation.dynamic_drawdown import (
    DrawdownThresholds,
    DynamicDrawdownManager,
)


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self):
        mgr = DynamicDrawdownManager()
        status = mgr.get_status()
        assert status["normal"]["breakeven"] == 0.50
        assert status["normal"]["lock"] == 0.75
        assert status["protective"]["breakeven"] == 0.30
        assert status["protective"]["lock"] == 0.50
        assert status["aggressive"]["breakeven"] == 0.60
        assert status["aggressive"]["lock"] == 0.85
        assert status["thresholds"]["protective"] == 0.6
        assert status["thresholds"]["aggressive"] == 0.3

    def test_custom_thresholds_persist(self):
        mgr = DynamicDrawdownManager(
            normal_breakeven_pct=0.55,
            protective_lock_pct=0.40,
            protective_threshold=0.7,
            aggressive_threshold=0.2,
        )
        status = mgr.get_status()
        assert status["normal"]["breakeven"] == 0.55
        assert status["protective"]["lock"] == 0.40
        assert status["thresholds"]["protective"] == 0.7
        assert status["thresholds"]["aggressive"] == 0.2


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

class TestModeSelection:
    def test_normal_mode_in_middle_band(self):
        mgr = DynamicDrawdownManager()
        # severity = 0.075 / 0.15 = 0.5 → in [0.3, 0.6] band → normal
        out = mgr.get_dynamic_thresholds(current_dd=0.075, max_dd=0.15)
        assert out.mode == "normal"
        assert out.breakeven_pct == 0.50
        assert out.lock_pct == 0.75
        assert out.severity == 0.5

    def test_protective_mode_above_threshold(self):
        mgr = DynamicDrawdownManager()
        # severity = 0.12 / 0.15 = 0.8 → > 0.6 → protective
        out = mgr.get_dynamic_thresholds(current_dd=0.12, max_dd=0.15)
        assert out.mode == "protective"
        # Protective interpolates toward smaller values
        assert out.breakeven_pct < 0.50
        assert out.lock_pct < 0.75

    def test_aggressive_mode_below_threshold(self):
        mgr = DynamicDrawdownManager()
        # severity = 0.015 / 0.15 = 0.1 → < 0.3 → aggressive
        out = mgr.get_dynamic_thresholds(current_dd=0.015, max_dd=0.15)
        assert out.mode == "aggressive"
        # Aggressive interpolates toward larger values
        assert out.breakeven_pct > 0.50
        assert out.lock_pct > 0.75


# ---------------------------------------------------------------------------
# Boundary continuity (interpolation correctness)
# ---------------------------------------------------------------------------

class TestBoundaryContinuity:
    def test_protective_at_threshold_equals_normal(self):
        """At severity = protective_threshold, interpolation t=0 → returns normal values.

        Exact-equality test: severity > protective_threshold goes into the
        protective branch with t = (severity - threshold) / (1 - threshold).
        At severity == threshold the strict inequality is false, so we land in
        the normal branch — the *just-above* case must approach normal.
        """
        mgr = DynamicDrawdownManager()
        # Just barely past 0.6 threshold
        out = mgr.get_dynamic_thresholds(current_dd=0.6 * 0.15 + 1e-9, max_dd=0.15)
        # Should be effectively normal (t ≈ 0)
        assert out.mode == "protective"
        assert out.breakeven_pct == pytest.approx(0.50, abs=1e-3)
        assert out.lock_pct == pytest.approx(0.75, abs=1e-3)

    def test_aggressive_at_threshold_equals_normal(self):
        mgr = DynamicDrawdownManager()
        # Severity exactly equal to 0.3 lands in normal branch (strict <)
        out = mgr.get_dynamic_thresholds(current_dd=0.3 * 0.15, max_dd=0.15)
        assert out.mode == "normal"
        assert out.breakeven_pct == 0.50
        assert out.lock_pct == 0.75

    def test_protective_at_max_severity_hits_endpoint(self):
        mgr = DynamicDrawdownManager()
        # severity = 1.0 → t = (1.0 - 0.6) / 0.4 = 1.0 → fully protective
        out = mgr.get_dynamic_thresholds(current_dd=0.15, max_dd=0.15)
        assert out.mode == "protective"
        assert out.breakeven_pct == 0.30
        assert out.lock_pct == 0.50
        assert out.severity == 1.0

    def test_aggressive_at_zero_severity_hits_endpoint(self):
        mgr = DynamicDrawdownManager()
        # severity = 0.0 → t = 1.0 → fully aggressive
        out = mgr.get_dynamic_thresholds(current_dd=0.0, max_dd=0.15)
        assert out.mode == "aggressive"
        assert out.breakeven_pct == 0.60
        assert out.lock_pct == 0.85
        assert out.severity == 0.0


# ---------------------------------------------------------------------------
# Severity clamping & edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_severity_clamped_above_one(self):
        # current_dd > max_dd → severity clamped to 1.0
        mgr = DynamicDrawdownManager()
        out = mgr.get_dynamic_thresholds(current_dd=0.30, max_dd=0.15)
        assert out.severity == 1.0
        assert out.mode == "protective"
        # Hits protective endpoint
        assert out.breakeven_pct == 0.30
        assert out.lock_pct == 0.50

    def test_negative_current_dd_uses_absolute_value(self):
        # Drawdowns are sometimes recorded as negatives — abs() applied
        mgr = DynamicDrawdownManager()
        positive = mgr.get_dynamic_thresholds(current_dd=0.075, max_dd=0.15)
        negative = mgr.get_dynamic_thresholds(current_dd=-0.075, max_dd=0.15)
        assert positive.severity == negative.severity
        assert positive.mode == negative.mode

    def test_max_dd_zero_returns_safe_normal_defaults(self):
        # Guard: max_dd <= 0 must not divide by zero — returns normal mode
        mgr = DynamicDrawdownManager()
        out = mgr.get_dynamic_thresholds(current_dd=0.10, max_dd=0.0)
        assert out.mode == "normal"
        assert out.severity == 0.0
        assert out.breakeven_pct == 0.50
        assert out.lock_pct == 0.75

    def test_max_dd_negative_returns_safe_normal_defaults(self):
        mgr = DynamicDrawdownManager()
        out = mgr.get_dynamic_thresholds(current_dd=0.05, max_dd=-0.10)
        assert out.mode == "normal"

    def test_zero_current_dd_yields_full_aggressive(self):
        mgr = DynamicDrawdownManager()
        out = mgr.get_dynamic_thresholds(current_dd=0.0, max_dd=0.15)
        assert out.mode == "aggressive"
        assert out.severity == 0.0


# ---------------------------------------------------------------------------
# Monotonicity (system invariants)
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_breakeven_decreases_as_severity_rises_in_protective_band(self):
        """As drawdown worsens past the protective threshold, breakeven should
        only get tighter — never relax."""
        mgr = DynamicDrawdownManager()
        levels = [0.65, 0.75, 0.85, 0.95, 1.00]
        last = 1.0
        for s in levels:
            out = mgr.get_dynamic_thresholds(current_dd=s * 0.15, max_dd=0.15)
            assert out.breakeven_pct <= last + 1e-9
            last = out.breakeven_pct

    def test_lock_decreases_as_severity_rises_in_protective_band(self):
        mgr = DynamicDrawdownManager()
        levels = [0.65, 0.75, 0.85, 0.95, 1.00]
        last = 1.0
        for s in levels:
            out = mgr.get_dynamic_thresholds(current_dd=s * 0.15, max_dd=0.15)
            assert out.lock_pct <= last + 1e-9
            last = out.lock_pct

    def test_breakeven_decreases_as_severity_rises_in_aggressive_band(self):
        """As drawdown grows from 0 toward aggressive_threshold, breakeven should
        decrease toward the normal default (becoming tighter)."""
        mgr = DynamicDrawdownManager()
        levels = [0.05, 0.10, 0.15, 0.20, 0.25]
        last = 1.0
        for s in levels:
            out = mgr.get_dynamic_thresholds(current_dd=s * 0.15, max_dd=0.15)
            assert out.breakeven_pct <= last + 1e-9
            last = out.breakeven_pct

    def test_protective_breakeven_always_lower_than_normal(self):
        """Invariant: protective endpoint < normal default."""
        mgr = DynamicDrawdownManager()
        out_norm = mgr.get_dynamic_thresholds(current_dd=0.075, max_dd=0.15)  # severity=0.5
        out_prot = mgr.get_dynamic_thresholds(current_dd=0.15, max_dd=0.15)   # severity=1.0
        assert out_prot.breakeven_pct < out_norm.breakeven_pct
        assert out_prot.lock_pct < out_norm.lock_pct

    def test_aggressive_breakeven_always_higher_than_normal(self):
        mgr = DynamicDrawdownManager()
        out_norm = mgr.get_dynamic_thresholds(current_dd=0.075, max_dd=0.15)  # severity=0.5
        out_aggr = mgr.get_dynamic_thresholds(current_dd=0.0, max_dd=0.15)    # severity=0.0
        assert out_aggr.breakeven_pct > out_norm.breakeven_pct
        assert out_aggr.lock_pct > out_norm.lock_pct


# ---------------------------------------------------------------------------
# Output shape & rounding
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_returns_drawdown_thresholds_object(self):
        mgr = DynamicDrawdownManager()
        out = mgr.get_dynamic_thresholds(current_dd=0.10, max_dd=0.15)
        assert isinstance(out, DrawdownThresholds)
        assert hasattr(out, "breakeven_pct")
        assert hasattr(out, "lock_pct")
        assert hasattr(out, "severity")
        assert hasattr(out, "mode")

    def test_values_rounded_to_4_decimals(self):
        # Constructed manager that produces non-trivial floats
        mgr = DynamicDrawdownManager(
            normal_breakeven_pct=0.5,
            protective_breakeven_pct=0.31,
        )
        out = mgr.get_dynamic_thresholds(current_dd=0.10, max_dd=0.15)
        # Severity 0.10/0.15 = 0.6666... clamped/rounded to 4 dp
        assert round(out.severity, 4) == out.severity
        assert round(out.breakeven_pct, 4) == out.breakeven_pct
        assert round(out.lock_pct, 4) == out.lock_pct

    def test_get_status_returns_full_config(self):
        mgr = DynamicDrawdownManager()
        status = mgr.get_status()
        for key in ("normal", "protective", "aggressive", "thresholds"):
            assert key in status
        for key in ("protective", "aggressive"):
            assert key in status["thresholds"]


# ---------------------------------------------------------------------------
# Custom configuration
# ---------------------------------------------------------------------------

class TestCustomConfig:
    def test_custom_protective_threshold_alters_branch_selection(self):
        # Tighter protective band: triggers above 0.4 instead of 0.6
        mgr = DynamicDrawdownManager(protective_threshold=0.4)
        # severity = 0.5 — would be normal with default 0.6, but protective at 0.4
        out = mgr.get_dynamic_thresholds(current_dd=0.075, max_dd=0.15)
        assert out.mode == "protective"

    def test_custom_aggressive_threshold_alters_branch_selection(self):
        # Wider aggressive band: triggers below 0.5 instead of 0.3
        mgr = DynamicDrawdownManager(aggressive_threshold=0.5)
        # severity = 0.4 — would be normal with default 0.3, but aggressive at 0.5
        out = mgr.get_dynamic_thresholds(current_dd=0.06, max_dd=0.15)
        assert out.mode == "aggressive"

    def test_aggressive_threshold_zero_avoids_division_by_zero(self):
        # Edge case: aggressive_threshold = 0 with severity = 0
        mgr = DynamicDrawdownManager(aggressive_threshold=0.0)
        # severity 0 is NOT < 0, so we land in normal
        out = mgr.get_dynamic_thresholds(current_dd=0.0, max_dd=0.15)
        assert out.mode == "normal"
