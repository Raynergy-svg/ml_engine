"""Tests for src/scanner/automation/validation_stats.py (US-604)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pytest

from src.scanner.automation.validation_stats import (
    GATES_FIRED_KEYS,
    ROW_SCHEMA_KEYS,
    ScanDistributionStats,
)


@dataclass
class _FakeAnalysis:
    """Minimal stand-in for PairAnalysis with the fields we read."""
    pair: str = "EUR_USD"
    direction: str = "LONG"
    confidence: float = 0.68
    weighted_vote_score: float = 0.72
    model_disagreement: float = 0.18
    volatility_regime: str = "NORMAL"
    is_tradeable: bool = True
    confidence_passed: bool = True
    momentum_passed: bool = True
    risk_passed: bool = True
    agent_passed: bool = True
    agent_soft_penalty_applied: bool = False
    execution_quality_passed: bool = True
    blocked_by_circuit_breaker: bool = False
    circuit_breakers_triggered: List[str] = field(default_factory=list)
    agent_reason_codes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    kill_reason: Optional[str] = None


def _read_rows(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_jsonl_schema_validation(tmp_path: Path) -> None:
    """Every row contains the documented schema keys + nested gates_fired shape."""
    log = tmp_path / "dry_run.jsonl"
    stats = ScanDistributionStats(log_path=log)

    tradeable = _FakeAnalysis(pair="EUR_USD")
    blocked = _FakeAnalysis(
        pair="USD_JPY",
        is_tradeable=False,
        confidence_passed=False,
        kill_reason="confidence_gate",
    )
    written = stats.record_cycle([tradeable, blocked])
    assert written == 2

    rows = _read_rows(log)
    assert len(rows) == 2
    for row in rows:
        assert set(row.keys()) == ROW_SCHEMA_KEYS
        assert set(row["gates_fired"].keys()) == GATES_FIRED_KEYS
        assert isinstance(row["block_reasons"], list)
        assert isinstance(row["tradeable"], bool)
        assert isinstance(row["would_submit"], bool)

    # Tradeable row: no block reasons, would_submit=True
    assert rows[0]["tradeable"] is True
    assert rows[0]["would_submit"] is True
    assert rows[0]["block_reasons"] == []

    # Blocked row: confidence_gate must be in both gates_fired and block_reasons
    assert rows[1]["tradeable"] is False
    assert rows[1]["would_submit"] is False
    assert rows[1]["gates_fired"]["confidence_gate"] is True
    assert "confidence_gate" in rows[1]["block_reasons"]


def test_gate_firing_histogram(tmp_path: Path) -> None:
    """Each gate maps to the right field and the analyzer aggregates correctly."""
    log = tmp_path / "dry_run.jsonl"
    stats = ScanDistributionStats(log_path=log)

    trend_vetoed = _FakeAnalysis(
        pair="EUR_USD",
        is_tradeable=False,
        agent_reason_codes=["trend_veto"],
    )
    mr_vetoed = _FakeAnalysis(
        pair="GBP_USD",
        is_tradeable=False,
        agent_reason_codes=["mean_reversion_veto"],
    )
    stale = _FakeAnalysis(
        pair="USD_JPY",
        is_tradeable=False,
        agent_reason_codes=["staleness_veto"],
    )
    momentum_fail = _FakeAnalysis(
        pair="AUD_USD",
        is_tradeable=False,
        momentum_passed=False,
    )
    risk_fail = _FakeAnalysis(
        pair="NZD_USD",
        is_tradeable=False,
        risk_passed=False,
    )

    stats.record_cycle(
        [trend_vetoed, mr_vetoed, stale, momentum_fail, risk_fail],
        correlation_blocked_pairs=["AUD_USD"],
    )

    rows = _read_rows(log)
    assert len(rows) == 5

    # Histogram roll-up (mirrors analyze_dry_run.analyze logic).
    from collections import Counter
    fired: Counter = Counter()
    for r in rows:
        for g, v in r["gates_fired"].items():
            if v:
                fired[g] += 1

    assert fired["trend_veto"] == 1
    assert fired["mr_veto"] == 1
    assert fired["staleness_veto"] == 1
    assert fired["momentum_gate"] == 1
    assert fired["risk_gate"] == 1
    # AUD_USD was passed in correlation_blocked_pairs
    assert fired["correlation_blocked"] == 1

    # Analyzer integration
    from scripts.analyze_dry_run import analyze
    summary = analyze(rows)
    assert summary["total_candidates"] == 5
    assert summary["would_submit_count"] == 0
    assert summary["gate_firing_rate"]["trend_veto"] == pytest.approx(1 / 5)


def test_zero_candidates_cycle_writes_nothing(tmp_path: Path) -> None:
    """An empty or HOLD/error-only cycle appends no rows."""
    log = tmp_path / "dry_run.jsonl"
    stats = ScanDistributionStats(log_path=log)

    # 1) Empty list
    assert stats.record_cycle([]) == 0
    assert not log.exists()

    # 2) All HOLD / errored — still counts as a 0-candidate cycle
    hold = _FakeAnalysis(pair="EUR_USD", direction="HOLD")
    errored = _FakeAnalysis(pair="USD_JPY", error="data_fetch_failed")
    assert stats.record_cycle([hold, errored]) == 0
    assert not log.exists()

    # 3) One real candidate → exactly one row
    good = _FakeAnalysis(pair="GBP_USD")
    assert stats.record_cycle([good]) == 1
    assert log.exists()
    assert len(_read_rows(log)) == 1
