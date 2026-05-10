"""Test virtual_trade_logger ordering bug fix (2026-05-09).

Bug: VirtualTradeLogger fired BEFORE _run_sub_inference_agents_for_pair, so
it captured the default (pre-sub-inference) `agent_passed=False` even when
sub-inference subsequently flipped it to True. This produced misleading log
analysis (operators chasing a phantom "agent vetoer" that didn't exist).

Fix: virtual-trade logging moved out of `_scan_pair` into a new helper
`_log_virtual_trade_for_result` called from `scan()` AFTER
`_run_sub_inference_pass` completes.

These tests follow the project No-Mock Rule: real `Scanner` (constructed via
`__new__` to skip TF model init), real `VirtualTradeLogger` writing to a real
`tmp_path`, real `PairAnalysis`, real disk reads to verify recorded values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.engine import Scanner
from src.scanner.results import PairAnalysis
from src.scanner.virtual_trade_logger import VirtualTradeLogger


def _make_scanner_with_logger(log_path: Path) -> Scanner:
    """Construct a Scanner shell wired to a real VirtualTradeLogger on tmp_path.

    Bypasses `Scanner.__init__` (which loads TF/Keras models — segfault-prone
    on Apple Silicon and unrelated to this ordering bug). Sets only the
    attributes the helper under test reads.
    """
    from src.scanner.config import ScannerConfig

    scanner = Scanner.__new__(Scanner)
    scanner.config = ScannerConfig()
    scanner._virtual_trade_logger = VirtualTradeLogger(log_path=log_path)
    return scanner


def _make_non_tradeable_analysis(
    pair: str = "GBP_USD",
    direction: str = "SHORT",
    confidence: float = 0.65,
    agent_passed: bool = False,
) -> PairAnalysis:
    """Build a PairAnalysis that fails `is_tradeable` (gates_passed=False)."""
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
        agent_reasons=[{"name": "trend", "score": 0.7}],
    )


# ── Helper-level behavior ──────────────────────────────────────────────────


def test_helper_records_agent_passed_false_when_initial(tmp_path):
    """Initial pre-sub-inference state: agent_passed=False → logged as failure."""
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)

    analysis = _make_non_tradeable_analysis(agent_passed=False)
    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    assert log_path.exists(), "logger should have written an entry"
    entries = [json.loads(l) for l in log_path.read_text().strip().split("\n") if l.strip()]
    assert len(entries) == 1
    assert "agent_passed=False" in entries[0]["gate_failures"]


def test_helper_records_no_agent_failure_when_post_sub_inference_passed(tmp_path):
    """Post-sub-inference flip: agent_passed=True → NOT logged as failure.

    This is the load-bearing assertion. Pre-fix, the logger was called before
    sub-inference flipped agent_passed=True, so 'agent_passed=False' showed up
    in gate_failures even when sub-inference voted 3/3 pass. Post-fix, the
    helper sees the FINAL agent_passed value.
    """
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)

    analysis = _make_non_tradeable_analysis(agent_passed=False)
    # Simulate _run_sub_inference_agents_for_pair flipping agent_passed to True
    # (votes >= vote_required), as engine.py:3028 does.
    analysis.agent_passed = True
    analysis.agent_votes = 3
    analysis.agent_total = 3

    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    assert log_path.exists()
    entries = [json.loads(l) for l in log_path.read_text().strip().split("\n") if l.strip()]
    assert len(entries) == 1
    failures = entries[0]["gate_failures"]
    assert "agent_passed=False" not in failures, (
        "Bug regression: logger captured the pre-sub-inference default "
        f"instead of the FINAL agent_passed=True value. Failures: {failures}"
    )


def test_helper_skips_when_low_confidence(tmp_path):
    """Below MIN_CONFIDENCE_TO_LOG (0.10) → skip, no file write."""
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)

    analysis = _make_non_tradeable_analysis(confidence=0.05)
    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    # Logger min_confidence is 0.10; 0.05 falls below the helper's gate too.
    assert not log_path.exists() or log_path.read_text().strip() == ""


def test_helper_skips_when_direction_is_hold(tmp_path):
    """HOLD direction → skip (only LONG/SHORT logged)."""
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)

    analysis = _make_non_tradeable_analysis(direction="HOLD")
    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    assert not log_path.exists() or log_path.read_text().strip() == ""


def test_helper_skips_when_no_logger_configured(tmp_path):
    """Defensive: missing logger should no-op, not crash."""
    scanner = _make_scanner_with_logger(tmp_path / "x.jsonl")
    scanner._virtual_trade_logger = None

    analysis = _make_non_tradeable_analysis()
    # Should not raise
    scanner._log_virtual_trade_for_result(analysis.pair, analysis)


def test_helper_skips_when_analysis_has_error(tmp_path):
    """Analyses with error set → skip (no synthetic virtual-trade entry)."""
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)

    analysis = _make_non_tradeable_analysis()
    analysis.error = "scan failed"
    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    assert not log_path.exists() or log_path.read_text().strip() == ""


# ── Source-level ordering invariants ───────────────────────────────────────


def test_scan_pair_no_longer_calls_virtual_trade_logger():
    """`_scan_pair` must not invoke `_virtual_trade_logger.log_virtual_trade(`.

    Regression guard: pre-fix, the call was inline in `_scan_pair` (engine.py
    around line 4870). Post-fix, the call lives in `_log_virtual_trade_for_result`
    invoked from `scan()` after sub-inference.
    """
    engine_src = Path(__file__).resolve().parent.parent / "src" / "scanner" / "engine.py"
    text = engine_src.read_text()

    # Locate _scan_pair body
    start = text.index("def _scan_pair(")
    # End at the start of the next top-level method definition at the same indent
    after_start = text[start + 1:]
    end_offset = after_start.index("\n    def ")  # next method at the same indent
    scan_pair_body = text[start:start + 1 + end_offset]

    assert "_virtual_trade_logger.log_virtual_trade(" not in scan_pair_body, (
        "_scan_pair must not invoke the virtual trade logger directly — "
        "ordering bug regression. See test docstring."
    )


def test_scan_calls_helper_after_sub_inference_pass():
    """`scan()` must invoke `_log_virtual_trade_for_result` AFTER `_run_sub_inference_pass`.

    Static check: the call ordering in `scan()` is the bug fix. If a future
    refactor moves the helper call before sub-inference, this test fails loudly.
    """
    engine_src = Path(__file__).resolve().parent.parent / "src" / "scanner" / "engine.py"
    text = engine_src.read_text()

    # Find scan method (not _scan_pair, not _run_sub_inference_pass)
    scan_def_idx = text.index("\n    def scan(")
    # Restrict the search window to the scan() body (until the next top-level method)
    after = text[scan_def_idx + 1:]
    end = after.index("\n    def ")
    scan_body = text[scan_def_idx:scan_def_idx + 1 + end]

    sub_inf_idx = scan_body.find("_run_sub_inference_pass(")
    helper_idx = scan_body.find("_log_virtual_trade_for_result(")

    assert sub_inf_idx >= 0, "scan() must call _run_sub_inference_pass"
    assert helper_idx >= 0, "scan() must call _log_virtual_trade_for_result"
    assert helper_idx > sub_inf_idx, (
        "Ordering bug: _log_virtual_trade_for_result must run AFTER "
        "_run_sub_inference_pass so the recorded agent_passed reflects the "
        "FINAL post-sub-inference verdict."
    )


def test_helper_method_exists_on_scanner():
    """The new helper must exist as a Scanner method."""
    assert hasattr(Scanner, "_log_virtual_trade_for_result")
    assert callable(getattr(Scanner, "_log_virtual_trade_for_result"))


# ── End-to-end ordering through real sub-inference mutation ────────────────


def test_simulated_pipeline_records_post_sub_inference_state(tmp_path):
    """Drive the pipeline mutation order and verify the log reflects FINAL state.

    Reproduces the production sequence:
      1. _scan_pair produces a PairAnalysis with agent_passed=False (default)
      2. _run_sub_inference_agents_for_pair mutates analysis.agent_passed=True
         (votes >= vote_required) — engine.py:3028
      3. _log_virtual_trade_for_result fires post-sub-inference

    Asserts: the recorded entry reflects step (2)'s mutation, not step (1)'s default.
    Pre-fix, this would fail because the logger fired at step (1).
    """
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)

    # Step 1: _scan_pair output (pre-sub-inference)
    analysis = _make_non_tradeable_analysis(agent_passed=False, confidence=0.649)

    # Step 2: simulate sub-inference flip (the actual mutation engine.py:3025-3028 does)
    analysis.agent_votes = 3
    analysis.agent_total = 3
    analysis.agent_score = 0.78
    analysis.agent_passed = True

    # Step 3: log AFTER sub-inference (post-fix order)
    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    assert log_path.exists()
    entries = [json.loads(l) for l in log_path.read_text().strip().split("\n") if l.strip()]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["pair"] == "GBP_USD"
    assert entry["direction"] == "SHORT"
    # The load-bearing assertion: NO 'agent_passed=False' in failures.
    assert "agent_passed=False" not in entry["gate_failures"], (
        f"Logger captured pre-sub-inference state. Failures: {entry['gate_failures']}"
    )
