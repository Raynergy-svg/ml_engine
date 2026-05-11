"""Agent B observability fixes — three latent bugs surfaced by the 4-agent TUI
audit on 2026-05-10. All three fixes land in one commit; this test file is the
real-disk, no-mocks verification per `.claude/rules/improvement.md` No-Mock Rule.

Fixes covered:
  1. Log format honesty — engine.py:4695/4751/4783 `logger.info("...re-qualified")`
     lines claimed an action that pre-fix never happened (the preceding
     `result.is_tradeable = True` raised AttributeError silently). With the
     setter added in `src/scanner/results.py`, the log lines now describe a
     genuine state transition.
  2. `is_tradeable` setter — added to `PairAnalysis` so engine.py re-qualification
     sites (US-127 ThresholdOptimizer, US-132 EXTREME regime, US-142 fast-track)
     can override the computed verdict without raising AttributeError.
  3. `virtual_trades.jsonl` gate disambiguation — added
     `gate_failures_detail: List[{gate, observed, threshold, reason}]` with
     canonical gate names (`CANONICAL_GATES`), preserving the legacy
     `gate_failures` string list for backwards compatibility.

No mocks. Real `PairAnalysis`, real `VirtualTradeLogger`, real disk via
`tmp_path`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.results import PairAnalysis
from src.scanner.virtual_trade_logger import VirtualTradeLogger, CANONICAL_GATES


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: is_tradeable setter (foundational — the other two fixes depend on it)
# ─────────────────────────────────────────────────────────────────────────────


def _make_tradeable_blocker() -> PairAnalysis:
    """Build a PairAnalysis that the base `is_tradeable` computation REJECTS.

    `gates_passed=False` is enough to make the base property return False; the
    rest are reasonable defaults that won't trip secondary invariants.
    """
    return PairAnalysis(
        pair="EUR_USD",
        direction="LONG",
        confidence=0.55,
        gates_passed=False,           # forces base is_tradeable=False
        agent_passed=True,
        execution_quality_passed=True,
        model_disagreement=0.10,
        volatility_regime="NORMAL",
        error=None,
    )


class TestIsTradeableSetter:
    """Verify the setter round-trips on a real PairAnalysis instance."""

    def test_pre_fix_repro_setter_must_not_raise_attributeerror(self):
        """Pre-fix this raised AttributeError (no setter); post-fix it succeeds."""
        p = _make_tradeable_blocker()
        # No try/except — must not raise.
        p.is_tradeable = True
        # And the value must round-trip.
        assert p.is_tradeable is True

    def test_setter_false_clears_override(self):
        p = _make_tradeable_blocker()
        p.is_tradeable = True
        assert p.is_tradeable is True
        p.is_tradeable = False
        # Base computation rejects (gates_passed=False) → property returns False.
        assert p.is_tradeable is False

    def test_setter_honors_direction_invariant(self):
        """HOLD must remain non-tradeable even with override — safety invariant."""
        p = _make_tradeable_blocker()
        p.direction = "HOLD"
        p.is_tradeable = True
        assert p.is_tradeable is False, (
            "Override must not bypass direction-not-in-{LONG,SHORT} invariant"
        )

    def test_setter_honors_error_invariant(self):
        """error != None must remain non-tradeable even with override — safety."""
        p = _make_tradeable_blocker()
        p.error = "model load failed"
        p.is_tradeable = True
        assert p.is_tradeable is False, (
            "Override must not bypass error-is-None invariant"
        )

    def test_setter_does_not_force_gates_passed(self):
        """Setter only flips the override flag — callers must set gates_passed.

        Documents the contract: engine.py:4694/4746/4779 already do
        `result.gates_passed = True` alongside `result.is_tradeable = True`,
        so the setter alone need not propagate to gates_passed.
        """
        p = _make_tradeable_blocker()
        p.is_tradeable = True
        assert p.gates_passed is False, (
            "Setter must NOT silently mutate gates_passed; "
            "callers set both fields explicitly."
        )

    def test_engine_repro_three_call_sites_compile_and_run(self):
        """Reproduce the exact assignment pattern engine.py uses at three sites.

        Before the fix, this loop raised AttributeError at iteration 1.
        """
        analyses = [_make_tradeable_blocker() for _ in range(3)]
        for result in analyses:
            # Same code shape as engine.py:4693/4745/4778 — no try/except.
            result.is_tradeable = True
            result.gates_passed = True
        # All three now report tradeable.
        for result in analyses:
            assert result.is_tradeable is True
            assert result.gates_passed is True


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: log format honesty — the engine.py logger.info("re-qualified") lines
# emit a string that describes a real state mutation. With the setter fixed,
# the log claim and the post-mutation state are now consistent.
# ─────────────────────────────────────────────────────────────────────────────


class TestLogFormatHonesty:
    """Verify the state the log line CLAIMS is the state we actually observe.

    Pre-fix: `logger.info("US-127 re-qualified ...")` fired AFTER
    `result.is_tradeable = True` raised AttributeError — so the log claim was
    impossible to verify; the assignment never took effect. With the setter
    in place, the same code path now produces a real, observable mutation
    that matches what the log line says.
    """

    def test_us_127_reclaim_state_matches_log_claim(self):
        """Reproduce engine.py:4693-4698 mutation. State must match the log line."""
        result = _make_tradeable_blocker()
        # Pre-condition: not tradeable (matches engine.py:4690 `not result.is_tradeable`)
        assert result.is_tradeable is False

        # Exact same assignments as engine.py:4693-4694
        result.is_tradeable = True
        result.gates_passed = True

        # The log line at 4695-4698 claims "re-qualified". Verify the claim:
        # the result IS now tradeable and gates_passed IS True.
        assert result.is_tradeable is True
        assert result.gates_passed is True

    def test_us_132_extreme_reclaim_state_matches_log_claim(self):
        """Reproduce engine.py:4744-4750. State must match the log line."""
        result = _make_tradeable_blocker()
        # Exact same assignments as engine.py:4745-4750
        result.is_tradeable = True
        result.gates_passed = True
        result.risk_pct = result.risk_pct * 0.50  # extreme_regime_size_multiplier

        # Log line claims "EXTREME policy re-qualified ... size_mult=0.50".
        # All three mutations succeed:
        assert result.is_tradeable is True
        assert result.gates_passed is True
        assert result.risk_pct == pytest.approx(0.05 * 0.50)  # default 0.05 * 0.50

    def test_us_142_fasttrack_state_matches_log_claim(self):
        """Reproduce engine.py:4778-4782. State must match the log line.

        This is the highest-risk pre-fix site — it had NO surrounding
        try/except, so the AttributeError propagated to the outer scan-level
        handler, dropping the pair entirely. The log line at 4783-4787 NEVER
        fired pre-fix. Post-fix, the mutation and log are consistent.
        """
        result = _make_tradeable_blocker()
        # Exact same assignments as engine.py:4778-4782
        result.is_tradeable = True
        result.gates_passed = True
        result.risk_pct = result.risk_pct * 0.75  # fasttrack_risk_multiplier
        # Note: `fasttrack` is not a field on PairAnalysis — engine.py:4782 sets
        # it as a dynamic attribute. We replicate the dynamic-attr assignment.
        result.fasttrack = True

        assert result.is_tradeable is True
        assert result.gates_passed is True
        assert result.risk_pct == pytest.approx(0.05 * 0.75)
        assert getattr(result, "fasttrack", False) is True


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3: virtual_trades.jsonl gate disambiguation
# ─────────────────────────────────────────────────────────────────────────────


class TestVirtualTradeGateDisambiguation:
    """Verify the structured `gate_failures_detail` uniquely identifies each gate."""

    def test_canonical_gates_taxonomy(self):
        """Five canonical gate names cover the five engine.py rejection sites."""
        assert CANONICAL_GATES == {
            "confidence", "momentum", "risk", "volatility", "agent",
        }

    def test_detail_field_persisted_to_disk(self, tmp_path):
        """Round-trip: write detail → read JSONL → verify structured payload."""
        log_path = tmp_path / "virtual_trades.jsonl"
        vtl = VirtualTradeLogger(log_path=log_path)

        wrote = vtl.log_virtual_trade(
            pair="EUR_USD",
            direction="LONG",
            confidence=0.42,
            gate_failures=["confidence=0.42 < min=0.60"],
            features={},
            agent_scores={},
            gate_failures_detail=[
                {"gate": "confidence", "observed": 0.42, "threshold": 0.60,
                 "reason": "confidence_passed=False"},
            ],
        )
        assert wrote is True
        assert log_path.exists()

        entry = json.loads(log_path.read_text().strip())
        assert "gate_failures_detail" in entry
        assert len(entry["gate_failures_detail"]) == 1
        detail = entry["gate_failures_detail"][0]
        assert detail["gate"] == "confidence"
        assert detail["observed"] == 0.42
        assert detail["threshold"] == 0.60
        assert detail["reason"] == "confidence_passed=False"

    def test_disambiguates_confidence_vs_momentum_gate_fails(self, tmp_path):
        """Pre-fix: both "confidence=..." and "momentum=..." landed in distinct
        buckets but with no canonical taxonomy. Post-fix: get_stats counters
        use canonical gate names, so consumers can answer "did the confidence
        gate veto?" by reading the `"confidence"` bucket directly.
        """
        log_path = tmp_path / "virtual_trades.jsonl"
        vtl = VirtualTradeLogger(log_path=log_path)

        # Two rejections, distinct gates
        vtl.log_virtual_trade(
            pair="EUR_USD", direction="LONG", confidence=0.42,
            gate_failures=["confidence=0.42 < min=0.60"],
            gate_failures_detail=[
                {"gate": "confidence", "observed": 0.42, "threshold": 0.60,
                 "reason": "confidence_passed=False"},
            ],
        )
        vtl.log_virtual_trade(
            pair="GBP_USD", direction="SHORT", confidence=0.55,
            gate_failures=["momentum=0.200"],
            gate_failures_detail=[
                {"gate": "momentum", "observed": 0.20, "threshold": 0.30,
                 "reason": "momentum_passed=False"},
            ],
        )

        stats = vtl.get_stats()
        # Canonical buckets, not free-form prefixes.
        top = dict(stats["top_gate_failures"])
        assert top.get("confidence") == 1
        assert top.get("momentum") == 1

    def test_disambiguates_risk_vs_volatility_vs_agent_gate_fails(self, tmp_path):
        """Three near-identical-looking string failures land in three
        distinct canonical buckets. Pre-fix `"risk_passed=False"`,
        `"volatility_gate_passed=False"`, and `"agent_passed=False"` were
        three free-form keys that no consumer could map to the three gates.
        """
        log_path = tmp_path / "virtual_trades.jsonl"
        vtl = VirtualTradeLogger(log_path=log_path)

        vtl.log_virtual_trade(
            pair="EUR_USD", direction="LONG", confidence=0.55,
            gate_failures=["risk_passed=False"],
            gate_failures_detail=[
                {"gate": "risk", "observed": 0.05, "threshold": 0.03,
                 "reason": "risk_passed=False"},
            ],
        )
        vtl.log_virtual_trade(
            pair="EUR_USD", direction="LONG", confidence=0.55,
            gate_failures=["volatility_gate_passed=False"],
            gate_failures_detail=[
                {"gate": "volatility", "observed": "UNKNOWN", "threshold": None,
                 "reason": "volatility_gate_passed=False"},
            ],
        )
        vtl.log_virtual_trade(
            pair="EUR_USD", direction="LONG", confidence=0.55,
            gate_failures=["agent_passed=False"],
            gate_failures_detail=[
                {"gate": "agent", "observed": 0.40, "threshold": 0.55,
                 "reason": "agent_passed=False"},
            ],
        )

        stats = vtl.get_stats()
        top = dict(stats["top_gate_failures"])
        assert top.get("risk") == 1
        assert top.get("volatility") == 1
        assert top.get("agent") == 1
        # Critically: the legacy buckets MUST NOT appear because detail is
        # supplied — the counter uses canonical names exclusively when detail
        # is present.
        assert "risk_passed" not in top
        assert "volatility_gate_passed" not in top
        assert "agent_passed" not in top

    def test_legacy_string_gate_failures_preserved_for_backcompat(self, tmp_path):
        """Existing tests/consumers read `gate_failures` (list of strings).
        The legacy field MUST still be present in the JSONL payload.
        """
        log_path = tmp_path / "virtual_trades.jsonl"
        vtl = VirtualTradeLogger(log_path=log_path)
        vtl.log_virtual_trade(
            pair="EUR_USD", direction="LONG", confidence=0.42,
            gate_failures=["confidence=0.42 < min=0.60", "agent_passed=False"],
            gate_failures_detail=[
                {"gate": "confidence", "observed": 0.42, "threshold": 0.60,
                 "reason": "confidence_passed=False"},
                {"gate": "agent", "observed": 0.40, "threshold": 0.55,
                 "reason": "agent_passed=False"},
            ],
        )
        entry = json.loads(log_path.read_text().strip())
        # Legacy list of strings — backwards compatibility contract.
        assert "gate_failures" in entry
        assert "confidence=0.42 < min=0.60" in entry["gate_failures"]
        assert "agent_passed=False" in entry["gate_failures"]

    def test_non_canonical_gate_dropped_silently_with_debug_log(self, tmp_path):
        """Non-CANONICAL gate names are dropped to keep the taxonomy clean."""
        log_path = tmp_path / "virtual_trades.jsonl"
        vtl = VirtualTradeLogger(log_path=log_path)
        vtl.log_virtual_trade(
            pair="EUR_USD", direction="LONG", confidence=0.42,
            gate_failures=["weird=False"],
            gate_failures_detail=[
                {"gate": "totally_made_up_gate", "observed": 1, "threshold": 0,
                 "reason": "weird=False"},
                {"gate": "confidence", "observed": 0.42, "threshold": 0.60,
                 "reason": "confidence_passed=False"},
            ],
        )
        entry = json.loads(log_path.read_text().strip())
        # Only the canonical entry survives.
        assert len(entry["gate_failures_detail"]) == 1
        assert entry["gate_failures_detail"][0]["gate"] == "confidence"

    def test_counter_rebuild_uses_canonical_when_present(self, tmp_path):
        """Fresh VTL instance reading existing file uses canonical detail."""
        path = tmp_path / "vt.jsonl"
        l1 = VirtualTradeLogger(log_path=path)
        l1.log_virtual_trade(
            pair="EUR_USD", direction="LONG", confidence=0.42,
            gate_failures=["risk_passed=False"],
            gate_failures_detail=[
                {"gate": "risk", "observed": 0.05, "threshold": 0.03,
                 "reason": "risk_passed=False"},
            ],
        )

        # New instance reads file and rebuilds counters via canonical detail
        l2 = VirtualTradeLogger(log_path=path)
        assert l2._total_logged == 1
        assert l2._gate_failure_counts["risk"] == 1
        # Legacy bucket name must NOT have been incremented this time.
        assert l2._gate_failure_counts.get("risk_passed", 0) == 0

    def test_counter_rebuild_falls_back_to_legacy_prefix_when_detail_absent(
        self, tmp_path
    ):
        """Pre-fix entries without `gate_failures_detail` still parse via prefix."""
        path = tmp_path / "vt.jsonl"
        # Hand-write a legacy entry (no `gate_failures_detail` key at all)
        legacy_entry = {
            "timestamp": "2026-05-09T12:00:00+00:00",
            "pair": "EUR_USD",
            "direction": "LONG",
            "raw_confidence": 0.42,
            "gate_failures": ["confidence=0.42 < 0.60"],
            "features": {},
            "agent_scores": {},
        }
        path.write_text(json.dumps(legacy_entry) + "\n")

        l = VirtualTradeLogger(log_path=path)
        assert l._total_logged == 1
        # Legacy prefix parse — "confidence" is the prefix-before-=.
        assert l._gate_failure_counts["confidence"] == 1
