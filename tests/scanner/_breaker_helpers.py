"""Shared test helpers for circuit-breaker and staleness regression tests.

Leading underscore keeps pytest's collector from treating this as a test
module. Everything here is plumbing — no assertions, no fixtures that live
in ``conftest.py`` scope.

Design goal: adding a 5th circuit-breaker code should require exactly ONE
new entry in ``KNOWN_BREAKER_CODES`` in the parametrized test — zero
boilerplate in this file. See ``test_circuit_breaker_enforcement.py`` for
the intended call sites.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from src.scanner.agents import AgentDecisionContext, ScannerAgentTeam
from src.scanner.execution import ExecutionConfig, ExecutionManager, ExecutionResult
from src.scanner.results import PairAnalysis


# ---------------------------------------------------------------------------
# Execution-boundary helpers
# ---------------------------------------------------------------------------
def make_execution_manager(**cfg_overrides: Any) -> ExecutionManager:
    """Build an ``ExecutionManager`` with a ``MagicMock`` OANDA client.

    Mirrors the fixture pattern in ``tests/test_execution_manager_core.py``.
    The returned manager is ready to call ``execute_trade()`` on; all OANDA
    I/O is mocked, so these tests run fully offline.

    Example:
        >>> mgr = make_execution_manager()
        >>> result = mgr.execute_trade(
        ...     pair="EUR_USD", direction="LONG", confidence=0.6,
        ...     current_price=1.1, atr=0.001,
        ...     analysis_context=build_minimal_valid_ctx(),
        ... )
    """
    cfg = ExecutionConfig(**cfg_overrides)
    mock_oanda = MagicMock()
    return ExecutionManager(config=cfg, oanda_client=mock_oanda)


def build_minimal_valid_ctx(**overrides: Any) -> dict[str, Any]:
    """Return a ctx dict that passes every defense-in-depth gate by default.

    Callers override only the keys they want to exercise — typically
    ``blocked_by_circuit_breaker`` and ``circuit_breakers_triggered`` — to
    isolate a single gate under test.

    Example:
        >>> ctx = build_minimal_valid_ctx(
        ...     blocked_by_circuit_breaker=True,
        ...     circuit_breakers_triggered=["trend_contra"],
        ... )
    """
    ctx: dict[str, Any] = {
        "agent_passed": True,
        "confidence": 0.60,
        "model_disagreement": 0.10,
        "volatility_regime": "NORMAL",
        "blocked_by_circuit_breaker": False,
        "circuit_breakers_triggered": [],
        "spread_pips": 1.0,
    }
    ctx.update(overrides)
    return ctx


def assert_rejects_with_breaker(
    manager: ExecutionManager,
    ctx: dict[str, Any],
    expected_code: str,
) -> None:
    """Call ``execute_trade`` and assert the circuit-breaker gate rejected it.

    Runs a fixed-signature ``execute_trade`` call (EUR_USD / LONG / 0.60
    confidence / 1.1000 price / 0.0010 ATR — the same signature every test
    in this file uses) and verifies three properties of the result:

        1. ``result.success is False``
        2. ``"circuit_breakers_triggered"`` appears in ``result.error``
        3. ``expected_code`` appears in ``result.error``

    One helper = one call in the test body.

    Example:
        >>> mgr = make_execution_manager()
        >>> ctx = build_minimal_valid_ctx(
        ...     blocked_by_circuit_breaker=True,
        ...     circuit_breakers_triggered=["trend_contra"],
        ... )
        >>> assert_rejects_with_breaker(mgr, ctx, expected_code="trend_contra")
    """
    result = manager.execute_trade(
        pair="EUR_USD",
        direction="LONG",
        confidence=0.60,
        current_price=1.1000,
        atr=0.0010,
        analysis_context=ctx,
    )

    assert isinstance(result, ExecutionResult)
    assert result.success is False, (
        f"Circuit breaker code '{expected_code}' MUST block execution — "
        f"defense-in-depth regression. result={result}"
    )
    assert result.error is not None
    assert "circuit_breakers_triggered" in result.error, (
        f"Error must surface the circuit-breaker key. Got: {result.error!r}"
    )
    assert expected_code in result.error, (
        f"Error must echo the specific reason code '{expected_code}'. "
        f"Got: {result.error!r}"
    )


# ---------------------------------------------------------------------------
# Agent-layer (staleness) helpers
# ---------------------------------------------------------------------------
def make_staleness_config(**overrides: Any) -> SimpleNamespace:
    """Return a ``SimpleNamespace`` config with staleness thresholds set.

    Keeps ``soft_uncertainty_blocking=True`` on purpose: the staleness hard
    block MUST fire even when soft blocking is active. That's the property
    the e2e test locks in.

    Example:
        >>> cfg = make_staleness_config(staleness_age_threshold_days=3.0)
    """
    defaults: dict[str, Any] = {
        "confidence_threshold": 0.55,
        "momentum_threshold": 0.40,
        "atr_sl_multiplier": 1.5,
        "atr_tp_multiplier": 2.5,
        "max_uncertainty_score": 0.40,
        "max_model_disagreement": 0.50,
        "disagreement_hard_floor": 0.50,
        "staleness_age_threshold_days": 7.0,
        "staleness_uncertainty_threshold": 0.35,
        # Critical: proves the hard-block bypasses soft blocking.
        "soft_uncertainty_blocking": True,
        "ensemble_conflict_enabled": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_uncertainty_context(**analysis_overrides: Any) -> AgentDecisionContext:
    """Build an ``AgentDecisionContext`` whose uncertainty score > 0.35.

    With ``confidence=0.50`` and the default ``confidence_threshold=0.60``::

        confidence_uncertainty = clip01((0.60 - 0.50) / 0.10) = 1.0
        uncertainty_score      = 1.0 * 0.60 + 0.0 * 0.40 = 0.60   > 0.35

    Empty feature dataframes zero out ``model_disagreement``, keeping the
    test focused on the staleness path.

    Example:
        >>> ctx = make_uncertainty_context()
        >>> team = make_team(ctx.config)
    """
    analysis_kwargs: dict[str, Any] = dict(
        pair="EUR_USD",
        direction="LONG",
        confidence=0.50,
        atr_pips=10.0,
        current_price=1.1000,
    )
    analysis_kwargs.update(analysis_overrides)
    analysis = PairAnalysis(**analysis_kwargs)
    return AgentDecisionContext(
        analysis=analysis,
        df_raw=pd.DataFrame(),
        df_feat=pd.DataFrame(),
        gate_details={},
        config=make_staleness_config(),
    )


def make_team(config: Any = None) -> ScannerAgentTeam:
    """Create a ``ScannerAgentTeam`` with ``_load_learned_weights`` patched to {}.

    Mirrors the helper in ``tests/test_scanner_agents.py``. Uses the passed
    config, or a fresh ``make_staleness_config()`` if none given.

    Example:
        >>> team = make_team(make_staleness_config())
    """
    with patch.object(ScannerAgentTeam, "_load_learned_weights", return_value={}):
        return ScannerAgentTeam(config or make_staleness_config())
