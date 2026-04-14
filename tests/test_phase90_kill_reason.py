"""Phase 90 US-403: Tests for kill_reason field on PairAnalysis."""
from __future__ import annotations

import pytest

from src.scanner.results import PairAnalysis


def test_kill_reason_default_is_none():
    """PairAnalysis.kill_reason defaults to None."""
    a = PairAnalysis(pair="EUR_USD")
    assert a.kill_reason is None


def test_kill_reason_set_to_confidence_gate():
    """kill_reason can be set to 'confidence_gate'."""
    a = PairAnalysis(pair="EUR_USD", kill_reason="confidence_gate")
    assert a.kill_reason == "confidence_gate"


def test_kill_reason_set_to_momentum_gate():
    """kill_reason can be set to 'momentum_gate'."""
    a = PairAnalysis(pair="GBP_USD", kill_reason="momentum_gate")
    assert a.kill_reason == "momentum_gate"


def test_kill_reason_set_to_disagreement_gate():
    """kill_reason can be set to 'disagreement_gate'."""
    a = PairAnalysis(pair="USD_JPY", kill_reason="disagreement_gate")
    assert a.kill_reason == "disagreement_gate"


def test_kill_reason_set_to_drawdown_gate():
    """kill_reason can be set to 'drawdown_gate'."""
    a = PairAnalysis(pair="AUD_USD", kill_reason="drawdown_gate")
    assert a.kill_reason == "drawdown_gate"


def test_kill_reason_set_to_accuracy_gate():
    """kill_reason can be set to 'accuracy_gate'."""
    a = PairAnalysis(pair="EUR_GBP", kill_reason="accuracy_gate")
    assert a.kill_reason == "accuracy_gate"


def test_kill_reason_set_to_other():
    """kill_reason can be set to 'other'."""
    a = PairAnalysis(pair="NZD_USD", kill_reason="other")
    assert a.kill_reason == "other"


def test_tradeable_pair_has_no_kill_reason():
    """A tradeable PairAnalysis should have kill_reason=None."""
    a = PairAnalysis(
        pair="EUR_USD",
        direction="LONG",
        confidence=0.75,
        gates_passed=True,
        kill_reason=None,
        agent_passed=True,
        volatility_regime="NORMAL",
        model_disagreement=0.1,
    )
    assert a.kill_reason is None


def test_kill_reason_appears_in_engine_py():
    """Grep check: kill_reason appears in engine.py at least 5 times."""
    import subprocess
    result = subprocess.run(
        ["grep", "-c", "kill_reason", "/Users/buddy/Documents/ml_engine/src/scanner/engine.py"],
        capture_output=True, text=True,
    )
    count = int(result.stdout.strip())
    assert count >= 5, f"Expected >=5 kill_reason occurrences in engine.py, found {count}"


def test_kill_reason_appears_in_results_py():
    """Grep check: kill_reason field defined in results.py."""
    import subprocess
    result = subprocess.run(
        ["grep", "-c", "kill_reason", "/Users/buddy/Documents/ml_engine/src/scanner/results.py"],
        capture_output=True, text=True,
    )
    count = int(result.stdout.strip())
    assert count >= 1, f"Expected >=1 kill_reason occurrences in results.py, found {count}"
