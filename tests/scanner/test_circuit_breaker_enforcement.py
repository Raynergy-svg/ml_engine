"""Regression tests for the P0 safety fixes that landed 2026-04-20.

Protects two hard-block properties that must NEVER silently revert:

1. **Execution-boundary circuit-breaker enforcement** — `execute_trade` must
   reject any trade whose `ctx` carries `blocked_by_circuit_breaker=True`
   OR a non-empty `circuit_breakers_triggered` list, even if upstream agent
   voting (somehow) set `agent_passed=True`. The four known reason codes are:
       - ``trend_contra``
       - ``staleness_hard_block``
       - ``ensemble_direction_conflict``
       - ``disagreement_hard_block``
   Source: ``src/scanner/execution.py`` "DEFENSE-IN-DEPTH: Signal quality
   validation" block (~lines 1838-1860).

2. **Staleness hard-block in the uncertainty agent** — when
   ``get_model_freshness()`` reports `oldest_age_days > staleness_age_threshold_days`
   AND ``uncertainty_score > staleness_uncertainty_threshold`` (default 0.35),
   ``_evaluate_uncertainty`` must set ``block_trade=True`` and
   ``reason_code="staleness_hard_block"`` — bypassing the
   ``soft_uncertainty_blocking`` flag. This is the critical property we lock
   in; a soft penalty cost $2,637 over a 10-trade losing streak before the
   hard block was re-added.
   Source: ``src/scanner/agents/_team.py`` "US-503: Tighten uncertainty
   threshold when models are stale" block (~lines 2066-2096).

All tests must run offline (no live OANDA, no real model files). Shared
plumbing (mock OANDA client, minimal valid ctx, rejection assertion,
staleness fixtures) lives in ``tests/scanner/_breaker_helpers.py`` — add a
5th breaker by adding one entry to ``KNOWN_BREAKER_CODES`` below, nothing
else.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tests.scanner._breaker_helpers import (
    assert_rejects_with_breaker,
    build_minimal_valid_ctx,
    make_execution_manager,
    make_team,
    make_uncertainty_context,
)


@pytest.fixture(autouse=True)
def isolated_runtime_state(tmp_path, monkeypatch):
    """Isolate cwd so StateEngine reads a halted=False state.json.

    ``execute_trade`` re-checks ``StateEngine().get_halted()`` mid-cycle
    (execution.py, added 2026-05-12) and ``STATE_PATH`` is cwd-relative
    (``.claude/state.json``). Without this fixture, running pytest from the
    repo root leaks the LIVE runtime halt flag into these tests: when the
    live system is halted (the normal fail-closed posture), every trade is
    rejected with ``BLOCKED: state.halted=True`` BEFORE the circuit-breaker
    gate under test ever runs. Real file on real disk — no mocks.
    """
    monkeypatch.chdir(tmp_path)
    state_dir = tmp_path / ".claude"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        json.dumps({"halted": False, "mode": "dry_run", "schema_version": "2"})
    )


# ---------------------------------------------------------------------------
# Add a 5th circuit-breaker code here — that's it. The parametrized test
# below picks it up automatically and the helpers do the rest.
# ---------------------------------------------------------------------------
KNOWN_BREAKER_CODES = [
    "trend_contra",
    "staleness_hard_block",
    "ensemble_direction_conflict",
    "disagreement_hard_block",
]


# ---------------------------------------------------------------------------
# Test 1 — parametrized per circuit-breaker code
# ---------------------------------------------------------------------------
class TestExecutionCircuitBreakerCodes:
    """Each of the four promoted circuit-breaker codes must halt execution."""

    @pytest.mark.parametrize("code", KNOWN_BREAKER_CODES)
    def test_execution_rejects_each_circuit_breaker_code(self, code: str) -> None:
        manager = make_execution_manager()
        ctx = build_minimal_valid_ctx(
            blocked_by_circuit_breaker=True,
            circuit_breakers_triggered=[code],
        )
        assert_rejects_with_breaker(manager, ctx, expected_code=code)


# ---------------------------------------------------------------------------
# Test 2 — bare flag with empty list still blocks (failsafe)
# ---------------------------------------------------------------------------
class TestExecutionBareCircuitBreakerFlag:
    """If ``blocked_by_circuit_breaker=True`` but the list is empty, we must
    still reject. The execution layer falls back to ``['unspecified']`` in
    the error string — confirming it trusts the flag over the list."""

    def test_execution_rejects_bare_flag_even_with_empty_list(self) -> None:
        manager = make_execution_manager()
        ctx = build_minimal_valid_ctx(
            blocked_by_circuit_breaker=True,
            circuit_breakers_triggered=[],
        )

        # Use the shared helper to assert the gate fires; the 'unspecified'
        # literal is the failsafe path, so we also verify it directly.
        assert_rejects_with_breaker(manager, ctx, expected_code="unspecified")


# ---------------------------------------------------------------------------
# Test 3 — happy path: no breakers tripped, gate passes
# ---------------------------------------------------------------------------
class TestExecutionNoCircuitBreakers:
    """When ``blocked_by_circuit_breaker=False`` and the list is empty,
    the circuit-breaker gate itself must NOT reject. Downstream gates
    (duplicate positions, drawdown guardian, NAV fetch, etc.) may reject
    later — that's fine. We only care this specific gate lets the flow
    continue."""

    def test_execution_passes_when_no_breakers(self) -> None:
        manager = make_execution_manager()
        ctx = build_minimal_valid_ctx(
            blocked_by_circuit_breaker=False,
            circuit_breakers_triggered=[],
        )

        # Mock monitor_open_trades so the duplicate-position check downstream
        # doesn't explode on the MagicMock OANDA client. This isolates the
        # assertion to the circuit-breaker gate.
        with patch.object(manager, "monitor_open_trades", return_value=[]):
            result = manager.execute_trade(
                pair="EUR_USD",
                direction="LONG",
                confidence=0.60,
                current_price=1.1000,
                atr=0.0010,
                analysis_context=ctx,
            )

        # Whatever happens downstream, it must NOT be the circuit-breaker gate.
        if result.success is False:
            assert result.error is not None
            assert "circuit_breakers_triggered" not in result.error, (
                f"Circuit-breaker gate wrongly rejected a clean ctx. "
                f"error={result.error!r}"
            )


# ---------------------------------------------------------------------------
# Test 4 — staleness e2e at the agent level
# ---------------------------------------------------------------------------
class TestStalenessHardBlockAgent:
    """High-value regression: staleness + high uncertainty must HARD-block
    in ``_evaluate_uncertainty``, even with ``soft_uncertainty_blocking=True``.

    This property was the direct remediation for the 2026-04-15 10-trade
    / -$2,637 loss streak. If this test fails, the P0 fix has regressed
    and soft blocking will silently save trades that should die on staleness.
    """

    def test_staleness_hard_block_reaches_execution_via_uncertainty_agent(
        self,
    ) -> None:
        ctx = make_uncertainty_context()
        team = make_team(ctx.config)

        # Monkey-patch the staleness probe at its import site on _team.py.
        # The production code reads it via `if get_model_freshness is not None`.
        stale_payload = {"oldest_age_days": 30.0}
        with patch(
            "src.scanner.agents._team.get_model_freshness",
            return_value=stale_payload,
        ):
            verdict = team._evaluate_uncertainty(ctx)

        # Sanity: the uncertainty score we compute really does exceed the
        # staleness threshold — otherwise the test would pass vacuously.
        assert verdict.metadata["uncertainty_score"] > 0.35, (
            "Test fixture broken: computed uncertainty_score must exceed "
            "staleness_uncertainty_threshold (0.35) for the hard-block to "
            f"trigger. Got {verdict.metadata['uncertainty_score']!r}."
        )

        # The locking assertions:
        assert verdict.block_trade is True, (
            "Staleness + high uncertainty MUST hard-block. "
            "soft_uncertainty_blocking=True must NOT downgrade this to a "
            "confidence penalty."
        )
        assert verdict.reason_code == "staleness_hard_block", (
            f"reason_code must be 'staleness_hard_block' so the execution "
            f"boundary's defense-in-depth recognises it. Got "
            f"{verdict.reason_code!r}."
        )
        assert verdict.passed is False
        assert "stale" in verdict.reason.lower(), (
            f"Human-readable reason must mention staleness. "
            f"Got: {verdict.reason!r}"
        )
