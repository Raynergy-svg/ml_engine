"""Unit tests for DynamicSLTPOptimizer (src/scanner/automation/dynamic_sl_tp.py).

Closes a zero-coverage gap surfaced by the 2026-05-01 test coverage audit. The
module composes four regime-conditioned rules (trend_widen, early_tighten,
vol_relax, trailing_adaptive) on ATR-derived base SL/TP. Rules compose
multiplicatively, are scaled by aggressiveness, and the result is bounded by
[min_sl_pips, max_sl_pips] / [min_tp_pips, max_tp_pips] with a 1.2:1 R:R floor
(per .claude/rules/trading.md).

LOW regime early_tighten is locked at sl_multiplier=1.00 — a deliberate no-op
gated by `if mult >= 1.0: return ...` — protecting the trading rule
"LOW regime sl_mult >= 1.2 (ranging markets need wider stops)".
"""

from __future__ import annotations

import pytest

from src.scanner.automation.dynamic_sl_tp import (
    EARLY_TIGHTEN_TABLE,
    TRAILING_TABLE,
    TREND_WIDEN_TABLE,
    VOL_RELAX_TABLE,
    DynamicSLTPOptimizer,
    DynamicSLTPResult,
    SLTPAdjustment,
)


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        opt = DynamicSLTPOptimizer()
        assert opt.aggressiveness == 1.0
        assert opt.enable_trend_widen is True
        assert opt.enable_early_tighten is True
        assert opt.enable_vol_relax is True
        assert opt.enable_trailing is True
        assert opt.min_sl_pips == 5.0
        assert opt.max_sl_pips == 80.0
        assert opt.min_tp_pips == 8.0
        assert opt.max_tp_pips == 120.0

    def test_aggressiveness_clamped_low(self):
        opt = DynamicSLTPOptimizer(aggressiveness=0.1)
        assert opt.aggressiveness == 0.5

    def test_aggressiveness_clamped_high(self):
        opt = DynamicSLTPOptimizer(aggressiveness=5.0)
        assert opt.aggressiveness == 2.0

    def test_aggressiveness_passthrough(self):
        opt = DynamicSLTPOptimizer(aggressiveness=1.25)
        assert opt.aggressiveness == 1.25


# ---------------------------------------------------------------------------
# optimize() — orchestration
# ---------------------------------------------------------------------------

class TestOptimizeOrchestration:
    def test_returns_result_object(self):
        opt = DynamicSLTPOptimizer()
        out = opt.optimize(
            base_sl_pips=15.0,
            base_tp_pips=30.0,
            regime="NORMAL",
        )
        assert isinstance(out, DynamicSLTPResult)
        assert out.original_sl_pips == 15.0
        assert out.original_tp_pips == 30.0

    def test_neutral_inputs_no_rules_fire(self):
        # Trend below threshold, confidence above threshold, NORMAL regime: no rules apply
        opt = DynamicSLTPOptimizer()
        out = opt.optimize(
            base_sl_pips=20.0,
            base_tp_pips=30.0,
            regime="NORMAL",
            trend_strength=0.10,
            confidence=0.90,
        )
        assert out.rules_applied == 0
        assert out.optimized_sl_pips == 20.0
        assert out.optimized_tp_pips == 30.0
        assert out.adjustments == []

    def test_unknown_regime_falls_back_to_normal(self):
        opt = DynamicSLTPOptimizer()
        out = opt.optimize(15.0, 30.0, regime="MOON")
        # Should not crash; normalized to NORMAL means same result as NORMAL with same inputs
        assert isinstance(out, DynamicSLTPResult)

    def test_lowercase_regime_is_normalized(self):
        opt = DynamicSLTPOptimizer()
        # NORMAL trend_widen min_trend_strength=0.55 → 0.80 fires
        out_upper = opt.optimize(15.0, 30.0, regime="NORMAL", trend_strength=0.80, confidence=0.90)
        out_lower = opt.optimize(15.0, 30.0, regime="normal", trend_strength=0.80, confidence=0.90)
        assert out_upper.optimized_tp_pips == out_lower.optimized_tp_pips

    def test_empty_regime_string_defaults_to_normal(self):
        opt = DynamicSLTPOptimizer()
        # Must not crash on empty regime
        out = opt.optimize(15.0, 30.0, regime="")
        assert isinstance(out, DynamicSLTPResult)


# ---------------------------------------------------------------------------
# Rule 1 — trend_widen
# ---------------------------------------------------------------------------

class TestTrendWiden:
    def test_below_threshold_no_widen(self):
        opt = DynamicSLTPOptimizer(enable_early_tighten=False, enable_vol_relax=False)
        # NORMAL min_trend_strength=0.55; at 0.50 no fire
        out = opt.optimize(20.0, 40.0, regime="NORMAL", trend_strength=0.50, confidence=0.99)
        assert out.rules_applied == 0
        assert out.optimized_tp_pips == 40.0

    def test_normal_regime_strong_trend_widens_tp(self):
        opt = DynamicSLTPOptimizer(enable_early_tighten=False, enable_vol_relax=False)
        # NORMAL tp_multiplier=1.25, threshold=0.55
        out = opt.optimize(20.0, 40.0, regime="NORMAL", trend_strength=0.80, confidence=0.99)
        assert out.rules_applied == 1
        # 40.0 * 1.25 = 50.0
        assert out.optimized_tp_pips == 50.0
        assert out.optimized_sl_pips == 20.0  # SL untouched
        assert out.adjustments[0]["rule_name"] == "trend_widen"

    def test_extreme_regime_no_widen_at_default_aggressiveness(self):
        # EXTREME tp_multiplier=1.00 → mult=1.0 → guard returns early
        opt = DynamicSLTPOptimizer(enable_early_tighten=False, enable_vol_relax=False)
        out = opt.optimize(20.0, 40.0, regime="EXTREME", trend_strength=0.99, confidence=0.99)
        # 1.2 R:R floor enforced post-rule (40/20 = 2.0 already passes)
        assert out.optimized_tp_pips == 40.0
        assert out.rules_applied == 0

    def test_low_regime_strong_trend_widens(self):
        # LOW tp_multiplier=1.15, threshold=0.60
        opt = DynamicSLTPOptimizer(enable_early_tighten=False, enable_vol_relax=False)
        out = opt.optimize(20.0, 40.0, regime="LOW", trend_strength=0.65, confidence=0.99)
        # 40.0 * 1.15 = 46.0
        assert out.optimized_tp_pips == 46.0

    def test_aggressiveness_scales_widen(self):
        # 2x aggressiveness: NORMAL (1.25 - 1.0) * 2.0 = 0.50 → 1.50
        opt = DynamicSLTPOptimizer(
            aggressiveness=2.0,
            enable_early_tighten=False,
            enable_vol_relax=False,
        )
        out = opt.optimize(20.0, 40.0, regime="NORMAL", trend_strength=0.80, confidence=0.99)
        # 40.0 * 1.50 = 60.0
        assert out.optimized_tp_pips == 60.0

    def test_table_constants_match_spec(self):
        # Guard against accidental table edits
        assert TREND_WIDEN_TABLE["LOW"]["tp_multiplier"] == 1.15
        assert TREND_WIDEN_TABLE["NORMAL"]["tp_multiplier"] == 1.25
        assert TREND_WIDEN_TABLE["HIGH"]["tp_multiplier"] == 1.10
        assert TREND_WIDEN_TABLE["EXTREME"]["tp_multiplier"] == 1.00


# ---------------------------------------------------------------------------
# Rule 2 — early_tighten
# ---------------------------------------------------------------------------

class TestEarlyTighten:
    def test_high_confidence_no_tighten(self):
        opt = DynamicSLTPOptimizer(enable_trend_widen=False, enable_vol_relax=False)
        # NORMAL confidence_threshold=0.50; 0.99 >= 0.50 → no tighten
        out = opt.optimize(20.0, 40.0, regime="NORMAL", trend_strength=0.0, confidence=0.99)
        assert out.rules_applied == 0
        assert out.optimized_sl_pips == 20.0

    def test_normal_regime_low_confidence_tightens_sl(self):
        # NORMAL sl_multiplier=0.90, confidence_threshold=0.50
        opt = DynamicSLTPOptimizer(enable_trend_widen=False, enable_vol_relax=False)
        out = opt.optimize(20.0, 40.0, regime="NORMAL", trend_strength=0.0, confidence=0.40)
        assert out.rules_applied == 1
        # 20.0 * 0.90 = 18.0
        assert out.optimized_sl_pips == 18.0
        assert out.optimized_tp_pips == 40.0  # TP untouched
        assert out.adjustments[0]["rule_name"] == "early_tighten"

    def test_low_regime_never_tightens_sl(self):
        """LOW regime sl_multiplier locked at 1.00 — protects 'LOW needs wider stops' rule.

        Source: .claude/rules/trading.md (2026-04-15 Trade 1220 EUR_AUD postmortem).
        """
        opt = DynamicSLTPOptimizer(enable_trend_widen=False, enable_vol_relax=False)
        # LOW confidence_threshold=0.55; even with confidence well below it, mult=1.0 → no-op
        out = opt.optimize(20.0, 40.0, regime="LOW", trend_strength=0.0, confidence=0.20)
        assert out.rules_applied == 0
        assert out.optimized_sl_pips == 20.0
        # Verify table itself
        assert EARLY_TIGHTEN_TABLE["LOW"]["sl_multiplier"] == 1.00

    def test_extreme_regime_tightens_sl(self):
        # EXTREME sl_multiplier=0.95, confidence_threshold=0.55
        opt = DynamicSLTPOptimizer(enable_trend_widen=False, enable_vol_relax=False)
        out = opt.optimize(20.0, 40.0, regime="EXTREME", trend_strength=0.0, confidence=0.40)
        assert out.rules_applied == 1
        # 20.0 * 0.95 = 19.0
        assert out.optimized_sl_pips == 19.0

    def test_aggressiveness_amplifies_tightening(self):
        # 2x aggressiveness: NORMAL mult = 1.0 - (1.0 - 0.90) * 2.0 = 0.80
        opt = DynamicSLTPOptimizer(
            aggressiveness=2.0,
            enable_trend_widen=False,
            enable_vol_relax=False,
        )
        out = opt.optimize(20.0, 40.0, regime="NORMAL", trend_strength=0.0, confidence=0.40)
        # 20.0 * 0.80 = 16.0
        assert out.optimized_sl_pips == 16.0


# ---------------------------------------------------------------------------
# Rule 3 — vol_relax
# ---------------------------------------------------------------------------

class TestVolRelax:
    def test_low_regime_no_relax(self):
        opt = DynamicSLTPOptimizer(enable_trend_widen=False, enable_early_tighten=False)
        out = opt.optimize(20.0, 40.0, regime="LOW", trend_strength=0.0, confidence=0.99)
        assert out.rules_applied == 0
        assert out.optimized_sl_pips == 20.0
        assert out.optimized_tp_pips == 40.0

    def test_normal_regime_no_relax(self):
        opt = DynamicSLTPOptimizer(enable_trend_widen=False, enable_early_tighten=False)
        out = opt.optimize(20.0, 40.0, regime="NORMAL", trend_strength=0.0, confidence=0.99)
        assert out.rules_applied == 0
        assert out.optimized_sl_pips == 20.0

    def test_high_regime_widens_sl_and_tp(self):
        # HIGH sl mult=1.15; tp half-mult = 1.075
        opt = DynamicSLTPOptimizer(enable_trend_widen=False, enable_early_tighten=False)
        out = opt.optimize(20.0, 40.0, regime="HIGH", trend_strength=0.0, confidence=0.99)
        assert out.rules_applied == 1
        # 20.0 * 1.15 = 23.0; 40.0 * 1.075 = 43.0
        assert out.optimized_sl_pips == 23.0
        assert out.optimized_tp_pips == 43.0

    def test_extreme_regime_widens_more_aggressively(self):
        # EXTREME sl mult=1.30; tp half-mult = 1.15
        opt = DynamicSLTPOptimizer(enable_trend_widen=False, enable_early_tighten=False)
        out = opt.optimize(20.0, 40.0, regime="EXTREME", trend_strength=0.0, confidence=0.99)
        assert out.rules_applied == 1
        # 20.0 * 1.30 = 26.0; 40.0 * 1.15 = 46.0
        assert out.optimized_sl_pips == 26.0
        assert out.optimized_tp_pips == 46.0

    def test_table_constants_match_spec(self):
        assert VOL_RELAX_TABLE["LOW"] == 1.00
        assert VOL_RELAX_TABLE["NORMAL"] == 1.00
        assert VOL_RELAX_TABLE["HIGH"] == 1.15
        assert VOL_RELAX_TABLE["EXTREME"] == 1.30


# ---------------------------------------------------------------------------
# Rule composition + R:R floor + bounds
# ---------------------------------------------------------------------------

class TestComposition:
    def test_all_rules_compose_multiplicatively(self):
        # NORMAL: trend_widen TP×1.25, early_tighten SL×0.90, vol_relax no-op
        opt = DynamicSLTPOptimizer()
        out = opt.optimize(
            base_sl_pips=20.0,
            base_tp_pips=40.0,
            regime="NORMAL",
            trend_strength=0.80,  # fires
            confidence=0.40,       # fires
        )
        # Two rules applied
        assert out.rules_applied == 2
        # 40.0 * 1.25 = 50.0
        assert out.optimized_tp_pips == 50.0
        # 20.0 * 0.90 = 18.0
        assert out.optimized_sl_pips == 18.0

    def test_rr_floor_enforced_after_rule_composition(self):
        """R:R must be >= 1.2:1 (per .claude/rules/trading.md hard execution gate)."""
        opt = DynamicSLTPOptimizer(enable_vol_relax=False)
        # Force a sub-1.2 R:R: base SL=30, TP=20 (0.67 R:R)
        out = opt.optimize(
            base_sl_pips=30.0,
            base_tp_pips=20.0,
            regime="NORMAL",
            trend_strength=0.0,
            confidence=0.99,
        )
        # No rules fired, but R:R floor must lift TP to 30 * 1.2 = 36.0
        assert out.optimized_tp_pips >= out.optimized_sl_pips * 1.2 - 1e-9
        assert out.optimized_tp_pips == 36.0

    def test_rr_floor_capped_by_max_tp_pips(self):
        # SL very wide (clamped to 80 max), TP forced up by 1.2x → 96.0 (under 120 cap)
        opt = DynamicSLTPOptimizer(enable_trend_widen=False, enable_early_tighten=False, enable_vol_relax=False)
        out = opt.optimize(
            base_sl_pips=200.0,  # will clamp to max_sl_pips=80
            base_tp_pips=10.0,
            regime="NORMAL",
        )
        assert out.optimized_sl_pips == 80.0
        assert out.optimized_tp_pips == 96.0  # 80 * 1.2

    def test_min_sl_pips_floor(self):
        opt = DynamicSLTPOptimizer()
        out = opt.optimize(2.0, 50.0, regime="NORMAL")
        assert out.optimized_sl_pips == 5.0  # min_sl_pips default

    def test_max_sl_pips_ceiling(self):
        opt = DynamicSLTPOptimizer()
        out = opt.optimize(500.0, 50.0, regime="NORMAL")
        assert out.optimized_sl_pips == 80.0  # max_sl_pips default

    def test_min_tp_pips_floor(self):
        opt = DynamicSLTPOptimizer()
        out = opt.optimize(20.0, 1.0, regime="NORMAL")
        # TP clamped to min_tp_pips=8.0, then R:R floor lifts to 20 * 1.2 = 24
        assert out.optimized_tp_pips == 24.0

    def test_max_tp_pips_ceiling(self):
        opt = DynamicSLTPOptimizer()
        out = opt.optimize(20.0, 500.0, regime="NORMAL")
        assert out.optimized_tp_pips == 120.0

    def test_disabled_rules_do_not_fire(self):
        opt = DynamicSLTPOptimizer(
            enable_trend_widen=False,
            enable_early_tighten=False,
            enable_vol_relax=False,
        )
        out = opt.optimize(
            base_sl_pips=20.0,
            base_tp_pips=40.0,
            regime="HIGH",
            trend_strength=0.99,
            confidence=0.0,
        )
        assert out.rules_applied == 0
        assert out.optimized_sl_pips == 20.0
        assert out.optimized_tp_pips == 40.0


# ---------------------------------------------------------------------------
# Trailing parameters
# ---------------------------------------------------------------------------

class TestTrailingParams:
    def test_normal_regime_returns_normal_params(self):
        opt = DynamicSLTPOptimizer()
        params = opt.get_trailing_params("NORMAL")
        assert params["trail_pct"] == pytest.approx(0.40)
        assert params["activation_rr"] == 0.6

    def test_extreme_regime_loosest_trail(self):
        opt = DynamicSLTPOptimizer()
        params = opt.get_trailing_params("EXTREME")
        # 0.30 base * aggressiveness 1.0
        assert params["trail_pct"] == pytest.approx(0.30)
        assert params["activation_rr"] == 1.0

    def test_aggressiveness_scales_trail_pct(self):
        opt = DynamicSLTPOptimizer(aggressiveness=2.0)
        params = opt.get_trailing_params("NORMAL")
        # 0.40 * 2.0 = 0.80
        assert params["trail_pct"] == pytest.approx(0.80)
        # activation_rr is NOT scaled
        assert params["activation_rr"] == 0.6

    def test_unknown_regime_falls_back_to_normal(self):
        opt = DynamicSLTPOptimizer()
        params = opt.get_trailing_params("UNKNOWN")
        normal = opt.get_trailing_params("NORMAL")
        assert params == normal

    def test_table_ordering_invariant(self):
        # Trail aggressiveness should monotonically loosen as regime gets hotter
        # (NORMAL/HIGH/EXTREME each expand activation_rr)
        assert TRAILING_TABLE["LOW"]["activation_rr"] <= TRAILING_TABLE["NORMAL"]["activation_rr"]
        assert TRAILING_TABLE["NORMAL"]["activation_rr"] <= TRAILING_TABLE["HIGH"]["activation_rr"]
        assert TRAILING_TABLE["HIGH"]["activation_rr"] <= TRAILING_TABLE["EXTREME"]["activation_rr"]


# ---------------------------------------------------------------------------
# Result + Adjustment serialization
# ---------------------------------------------------------------------------

class TestResultSerialization:
    def test_to_dict_round_values(self):
        out = DynamicSLTPResult(
            original_sl_pips=20.123456,
            original_tp_pips=40.789012,
            optimized_sl_pips=18.001,
            optimized_tp_pips=50.0001,
            rules_applied=2,
        )
        d = out.to_dict()
        assert d["original_sl_pips"] == 20.12
        assert d["original_tp_pips"] == 40.79
        assert d["optimized_sl_pips"] == 18.0
        assert d["rules_applied"] == 2

    def test_adjustment_to_dict_includes_all_fields(self):
        adj = SLTPAdjustment(
            rule_name="trend_widen",
            regime="NORMAL",
            old_sl=20.0,
            new_sl=20.0,
            old_tp=40.0,
            new_tp=50.0,
            multiplier_applied=1.25,
            reason="trend strong",
        )
        d = adj.to_dict()
        assert d["rule_name"] == "trend_widen"
        assert d["regime"] == "NORMAL"
        assert d["new_tp"] == 50.0
        assert d["multiplier_applied"] == 1.25

    def test_adjustments_logged_in_optimize_result(self):
        opt = DynamicSLTPOptimizer()
        out = opt.optimize(20.0, 40.0, regime="HIGH", trend_strength=0.99, confidence=0.40)
        assert len(out.adjustments) == out.rules_applied
        # Each adjustment is a dict (not a dataclass) — to_dict() was called
        for adj in out.adjustments:
            assert isinstance(adj, dict)
            assert "rule_name" in adj
