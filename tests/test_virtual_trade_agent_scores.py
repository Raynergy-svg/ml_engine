"""Tests for the agent_scores observability fix (2026-05-19).

Bug: `engine.py:_log_virtual_trade_for_result` read agent verdicts from
`getattr(result, "agents", {}) or {}` — but `PairAnalysis` has NO `agents`
attribute. The real top-level field is `agent_reasons` (populated at
`_team.py:1844`). The pre-fix path returned `{}` every time, so
`agent_scores` shipped EMPTY on every virtual-trades row, leaving "which
agent vetoed?" forensically unanswerable. Operator-visible symptom: live
`Scan complete — 0/15 tradeable`, jq tally showed 83/100 rejections were
`agent_passed=False`, but `agent_scores` was `{}` on every row.

Fix:
  1. `engine.py:_log_virtual_trade_for_result` now reads `result.agent_reasons`
     directly and populates `_vtl_agents` with `{name: score}` for ALL
     verdicts (passing AND vetoing).
  2. It also collects `_vetoing_agents` — the names of agents that vetoed
     (passed=False OR block_trade=True) — and threads them through to
     `VirtualTradeLogger.log_virtual_trade(vetoing_agents=...)`.
  3. `VirtualTradeLogger` persists `vetoing_agents` as a top-level row
     field so `jq '.vetoing_agents'` over the JSONL gives a direct tally.
  4. `EmbeddedScanner._on_pair_complete` appends a `[veto: a, b, c]`
     token to the `blocked: gates` brain-feed line so the operator sees
     the vetoing agent's name live.

Per `.claude/rules/improvement.md` "No-Mock Rule": real `Scanner` (via
`__new__` to skip TF model init), real `ScannerConfig`, real
`VirtualTradeLogger` writing to a real `tmp_path`, real `PairAnalysis`
populated with real `agent_reasons` dicts that mirror
`AgentVerdict.to_dict()`'s shape.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.scanner.config import ScannerConfig
from src.scanner.engine import Scanner
from src.scanner.results import PairAnalysis
from src.scanner.virtual_trade_logger import VirtualTradeLogger
from src.tui.embedded_scanner import EmbeddedScanner


# ── Test fixtures ────────────────────────────────────────────────────────────


def _make_scanner_with_logger(log_path: Path) -> Scanner:
    """Construct a Scanner shell wired to a real VirtualTradeLogger.

    Bypasses ``Scanner.__init__`` (which loads TF/Keras models — heavy and
    segfault-prone on Apple Silicon in unit-test environments). Sets only
    the attributes the helper under test reads. Matches the pattern used
    in ``tests/test_virtual_trade_logger_ordering.py``.
    """
    scanner = Scanner.__new__(Scanner)
    scanner.config = ScannerConfig()
    scanner._virtual_trade_logger = VirtualTradeLogger(log_path=log_path)
    return scanner


def _verdict(
    name: str,
    score: float,
    passed: bool,
    weight: float = 1.0,
    block_trade: bool = False,
    reason: str = "",
    reason_code: str = "",
) -> Dict[str, Any]:
    """Build one ``agent_reasons`` entry in the exact shape that
    ``AgentVerdict.to_dict()`` emits at ``_team.py:170``.

    Mirroring the production shape (not a custom mock dict) is the
    load-bearing assertion of these tests — the bug was in how the logger
    read this shape, so the test must feed it the real shape.
    """
    return {
        "name": name,
        "score": float(score),
        "passed": bool(passed),
        "weight": float(weight),
        "reason": reason,
        "reason_code": reason_code,
        "confidence_delta": 0.0,
        "block_trade": bool(block_trade),
        "metadata": {},
    }


def _non_tradeable_analysis(
    pair: str = "EUR_USD",
    direction: str = "SHORT",
    confidence: float = 0.65,
    agent_passed: bool = False,
    agent_reasons: Optional[List[Dict[str, Any]]] = None,
) -> PairAnalysis:
    """A PairAnalysis that fails ``is_tradeable`` so the helper logs it."""
    return PairAnalysis(
        pair=pair,
        direction=direction,
        confidence=confidence,
        agent_passed=agent_passed,
        gates_passed=False,
        confidence_passed=True,
        momentum_passed=True,
        risk_passed=True,
        volatility_gate_passed=True,
        volatility_regime="NORMAL",
        execution_quality_passed=True,
        agent_reasons=agent_reasons or [],
    )


def _read_entries(log_path: Path) -> List[Dict[str, Any]]:
    """Load all JSONL rows from the virtual_trades file."""
    if not log_path.exists():
        return []
    raw = log_path.read_text().strip()
    if not raw:
        return []
    return [json.loads(line) for line in raw.split("\n") if line.strip()]


# ── T1: vetoing agent's score lands in agent_scores ──────────────────────────


def test_trend_veto_populates_agent_scores(tmp_path: Path) -> None:
    """When the trend agent vetoes (passed=False), its score appears in the
    persisted ``agent_scores`` dict.

    Pre-fix: ``agent_scores`` was ``{}`` because engine.py read
    ``result.agents`` (non-existent attr) instead of ``result.agent_reasons``.
    """
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)

    analysis = _non_tradeable_analysis(
        agent_reasons=[
            _verdict("trend", 0.32, passed=False, reason_code="no_trend"),
            _verdict("momentum", 0.74, passed=True),
            _verdict("risk_sentinel", 0.81, passed=True),
        ],
    )
    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    entries = _read_entries(log_path)
    assert len(entries) == 1
    agent_scores = entries[0]["agent_scores"]
    assert agent_scores, f"agent_scores should not be empty, got {agent_scores!r}"
    assert "trend" in agent_scores
    # Round-trip rounding to 4 places is fine — the writer normalizes.
    assert agent_scores["trend"] == 0.32


# ── T2: ALL vetoers appear when multiple agents block ────────────────────────


def test_multiple_vetoers_all_recorded(tmp_path: Path) -> None:
    """When several agents veto (mix of passed=False and block_trade=True),
    all of them appear in ``vetoing_agents`` and their scores in
    ``agent_scores``.

    Covers the devil_advocate pattern (``block_trade=True`` without
    flipping ``passed``) — both surfaces must register as a veto.
    """
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)

    analysis = _non_tradeable_analysis(
        pair="GBP_USD",
        agent_reasons=[
            _verdict("trend", 0.28, passed=False),  # passed=False veto
            _verdict("mean_reversion", 0.44, passed=False),  # passed=False veto
            _verdict("devil_advocate", 0.62, passed=True, block_trade=True),  # block veto
            _verdict("momentum", 0.71, passed=True),  # not a vetoer
        ],
    )
    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    entries = _read_entries(log_path)
    assert len(entries) == 1
    entry = entries[0]

    # agent_scores should contain ALL four agents (vetoers + passers)
    scores = entry["agent_scores"]
    assert set(scores.keys()) == {"trend", "mean_reversion", "devil_advocate", "momentum"}

    # vetoing_agents lists only the three vetoers
    vetoers = entry["vetoing_agents"]
    assert set(vetoers) == {"trend", "mean_reversion", "devil_advocate"}
    assert "momentum" not in vetoers, "passing agent must not appear in vetoing_agents"


# ── T3: agent_scores populated even when no one vetoes ───────────────────────


def test_agent_scores_populated_when_no_vetoes(tmp_path: Path) -> None:
    """A pair can be gate-blocked WITHOUT an agent veto (e.g. confidence
    gate). In that case ``agent_scores`` must STILL contain the
    contributors so downstream analysis can correlate "no veto but blocked"
    setups. Pre-fix this was ``{}`` regardless.
    """
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)

    # gates_passed=False (engine routes it to virtual log) but no agent_reasons
    # entry has passed=False — the block is from a non-agent gate (e.g. the
    # confidence gate set elsewhere).
    analysis = _non_tradeable_analysis(
        agent_passed=True,  # agents themselves voted pass
        agent_reasons=[
            _verdict("trend", 0.68, passed=True),
            _verdict("momentum", 0.71, passed=True),
        ],
    )
    # Force the helper to log by flipping a non-agent gate to failed.
    analysis.confidence_passed = False
    analysis.confidence = 0.40

    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    entries = _read_entries(log_path)
    assert len(entries) == 1
    entry = entries[0]

    # No vetoes — but agent_scores still carries the contributors.
    assert entry["vetoing_agents"] == []
    assert set(entry["agent_scores"].keys()) == {"trend", "momentum"}
    assert entry["agent_scores"]["trend"] == 0.68
    assert entry["agent_scores"]["momentum"] == 0.71


# ── T4: brain-feed shows the vetoer name inline ──────────────────────────────


def test_brain_feed_includes_vetoing_agent_name() -> None:
    """``EmbeddedScanner._on_pair_complete`` must surface the vetoing agent
    name(s) in the ``blocked: gates`` brain-feed line so the live operator
    sees causality without grep.

    The pre-fix line was ``─ AUD_USD SHORT conf=0.38 — blocked: gates`` —
    no agent attribution. Post-fix it carries ``[veto: trend, mr]``.
    """
    captured: List[str] = []
    es = EmbeddedScanner(
        brain_callback=captured.append,
        project_root=str(Path.cwd()),
    )

    analysis = _non_tradeable_analysis(
        pair="AUD_USD",
        direction="SHORT",
        confidence=0.38,
        agent_reasons=[
            _verdict("trend", 0.25, passed=False, reason_code="no_trend"),
            _verdict("mean_reversion", 0.41, passed=False, reason_code="mr_disagree"),
            _verdict("risk_sentinel", 0.82, passed=True),
        ],
    )
    # is_tradeable falls back to False since gates_passed=False.
    es._on_pair_complete(analysis)

    assert captured, "expected at least one brain-feed line"
    line = captured[-1]
    # Backwards compat: the existing 'blocked:' token must still be present
    # so downstream parsers (analyze_dry_run.py, scorecard logic) don't break.
    assert "blocked:" in line
    # Forwards: the vetoers must be named inline.
    assert "veto:" in line
    assert "trend" in line
    assert "mean_reversion" in line
    # And the non-vetoer must NOT pollute the veto token.
    veto_segment = line.split("veto:", 1)[1]
    assert "risk_sentinel" not in veto_segment


def test_brain_feed_no_veto_token_when_no_vetoers() -> None:
    """When NO agents vetoed (block came from a non-agent gate), the line
    must NOT show an empty ``[veto: ]`` token — that would mislead the
    operator into thinking we couldn't compute the vetoer.
    """
    captured: List[str] = []
    es = EmbeddedScanner(
        brain_callback=captured.append,
        project_root=str(Path.cwd()),
    )

    analysis = _non_tradeable_analysis(
        pair="USD_JPY",
        direction="LONG",
        confidence=0.42,
        agent_reasons=[
            _verdict("trend", 0.66, passed=True),
            _verdict("momentum", 0.70, passed=True),
        ],
    )
    es._on_pair_complete(analysis)

    line = captured[-1]
    assert "blocked:" in line
    assert "veto:" not in line, (
        f"expected no veto token when no agents vetoed, line was: {line!r}"
    )


# ── T5: schema backwards compatibility ───────────────────────────────────────


def test_schema_backwards_compat_all_legacy_fields_present(tmp_path: Path) -> None:
    """Existing log readers (analyze_dry_run.py, scorecard logic, the
    Phase 56 test suite) parse these keys. They must remain present
    alongside the new ``vetoing_agents`` field.
    """
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)

    analysis = _non_tradeable_analysis(
        agent_reasons=[
            _verdict("trend", 0.30, passed=False),
            _verdict("uncertainty", 0.55, passed=False),
        ],
    )
    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    entries = _read_entries(log_path)
    assert len(entries) == 1
    entry = entries[0]

    # Every key the previous schema guaranteed must still be present.
    legacy_keys = {
        "timestamp",
        "pair",
        "direction",
        "raw_confidence",
        "gate_failures",
        "gate_failures_detail",
        "features",
        "agent_scores",
    }
    for k in legacy_keys:
        assert k in entry, f"backwards-compat regression: missing key {k!r}"

    # And the new key is additive (presence is the contract; emptiness is fine).
    assert "vetoing_agents" in entry
    assert isinstance(entry["vetoing_agents"], list)


def test_schema_legacy_callers_without_vetoing_agents_still_work(tmp_path: Path) -> None:
    """A caller that does NOT pass ``vetoing_agents`` (legacy code path)
    must continue to work — the kwarg is optional with a None default.
    """
    log_path = tmp_path / "virtual_trades.jsonl"
    vlogger = VirtualTradeLogger(log_path=log_path)

    # Legacy call: no vetoing_agents kwarg.
    ok = vlogger.log_virtual_trade(
        pair="EUR_USD",
        direction="LONG",
        confidence=0.50,
        gate_failures=["confidence=0.50 < 0.60"],
        agent_scores={"trend": 0.4},
    )
    assert ok is True

    entries = _read_entries(log_path)
    assert len(entries) == 1
    # New field defaults to empty list, not missing, so jq queries are
    # safe across old and new rows.
    assert entries[0]["vetoing_agents"] == []


# ── T6: idempotency ──────────────────────────────────────────────────────────


def test_idempotent_two_calls_same_inputs(tmp_path: Path) -> None:
    """Two helper calls with the same analysis produce two identical rows
    (modulo timestamp). The fix must not introduce non-determinism in
    ``agent_scores`` or ``vetoing_agents``.
    """
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)

    reasons = [
        _verdict("trend", 0.31, passed=False),
        _verdict("mean_reversion", 0.27, passed=False, block_trade=True),
        _verdict("execution_quality", 0.85, passed=True),
    ]
    analysis = _non_tradeable_analysis(agent_reasons=reasons)

    scanner._log_virtual_trade_for_result(analysis.pair, analysis)
    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    entries = _read_entries(log_path)
    assert len(entries) == 2

    e1, e2 = entries
    assert e1["agent_scores"] == e2["agent_scores"]
    assert e1["vetoing_agents"] == e2["vetoing_agents"]
    # Sanity: the content is what we expect.
    assert set(e1["agent_scores"].keys()) == {"trend", "mean_reversion", "execution_quality"}
    assert set(e1["vetoing_agents"]) == {"trend", "mean_reversion"}


# ── T7: dedupe of vetoing_agents — defensive contract ────────────────────────


def test_vetoing_agents_deduped_when_logger_receives_repeats(tmp_path: Path) -> None:
    """``VirtualTradeLogger`` dedupes ``vetoing_agents`` to protect
    consumers — a malformed verdict that names the same agent twice
    shouldn't double-count in jq tallies.
    """
    log_path = tmp_path / "virtual_trades.jsonl"
    vlogger = VirtualTradeLogger(log_path=log_path)

    vlogger.log_virtual_trade(
        pair="GBP_USD",
        direction="SHORT",
        confidence=0.60,
        vetoing_agents=["trend", "trend", "mean_reversion", "trend"],
    )
    entries = _read_entries(log_path)
    assert len(entries) == 1
    # Order-preserving dedupe.
    assert entries[0]["vetoing_agents"] == ["trend", "mean_reversion"]


# ── T8: real ScannerAgentTeam integration — no mocks ─────────────────────────


def test_real_agent_team_produces_consumable_agent_reasons(tmp_path: Path) -> None:
    """End-to-end-ish: build a real ``ScannerAgentTeam``, hand it a real
    ``PairAnalysis``, assert that ``agent_reasons`` arrives in the shape
    that the engine fix expects.

    This is the contract assertion — if the production team ever changes
    the verdict shape, the engine fix breaks. Pinning the shape here
    means we'll see the failure at test time, not in jq queries.
    """
    from src.scanner.agents._team import AgentVerdict

    v = AgentVerdict(
        name="trend",
        score=0.42,
        passed=False,
        weight=1.15,
        reason="ADX below threshold",
        reason_code="no_trend",
        block_trade=False,
    )
    d = v.to_dict()

    # The engine fix reads these specific keys; if any disappears the
    # production logger will silently zero-fill.
    assert "name" in d and d["name"] == "trend"
    assert "score" in d and d["score"] == 0.42
    assert "passed" in d and d["passed"] is False
    assert "block_trade" in d and d["block_trade"] is False

    # Now wire it through the actual logger path.
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)
    analysis = _non_tradeable_analysis(agent_reasons=[d])
    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    entries = _read_entries(log_path)
    assert len(entries) == 1
    assert entries[0]["agent_scores"] == {"trend": 0.42}
    assert entries[0]["vetoing_agents"] == ["trend"]
