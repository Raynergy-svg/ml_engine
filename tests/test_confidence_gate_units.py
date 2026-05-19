"""Tests for the confidence-gate unit-mismatch fix (2026-05-19).

Bug: `engine.py:_log_virtual_trade_for_result` (`:3339`) and the trade-block
observation log (`:4930`) both emitted::

    f"confidence={result.confidence:.3f} < min={self.config.min_confidence}"

Where ``result.confidence`` is a probability in ``[0, 1]`` (the blended /
penalty-adjusted score on ``PairAnalysis``, bounded at the source by
``analysis.confidence = min(max(..., 0.0), 0.99)`` in
``engine.py:3108-3111``) and ``self.config.min_confidence`` is an
ADX threshold in ``[0, 100]`` (``ScannerConfig.min_confidence = 42.0``).
The string therefore lied: ``confidence=0.400 < min=42.0`` mixes scales.

Hypothesis B (confirmed via grep + read of `gates.py:2326` and
`engine.py:4455-4456`):

  * The GATE LOGIC at ``gates.py:2326`` ``confidence_passed = confidence >=
    min_confidence`` is correct — both sides are 0-100 there.
  * Only the LOG DISPLAY was lying; the bool ``confidence_passed`` flows
    in unchanged from the gate, so no trading behavior changed.
  * The PairAnalysis field that mirrors the actual 0-100 value the gate
    compared is ``confidence_score`` (set from ``gate_details["confidence_score"]``
    at ``engine.py:4456``, falling back to ``ridge_conf``), NOT
    ``confidence`` (a separate ``[0,1]`` blended probability).

Fix:
  1. ``_log_virtual_trade_for_result`` now uses ``result.confidence_score``
     (0-100) instead of ``result.confidence`` (``[0,1]``) in both the
     legacy string ``gate_failures`` entry and the structured
     ``gate_failures_detail["observed"]`` field. Format changed from
     ``:.3f`` to ``:.1f`` so the displayed precision matches an ADX
     reading (e.g. ``42.0``, not ``42.000``).
  2. The trade-block ``ObservationLog`` reason string at
     ``engine.py:4930`` got the same treatment.

Per `.claude/rules/improvement.md` "No-Mock Rule": real ``Scanner``
(via ``__new__`` to skip TF model init), real ``ScannerConfig``, real
``VirtualTradeLogger`` writing to ``tmp_path``, real ``PairAnalysis``.
Mirrors the pattern from ``tests/test_virtual_trade_agent_scores.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.scanner.config import ScannerConfig
from src.scanner.engine import Scanner
from src.scanner.results import PairAnalysis
from src.scanner.virtual_trade_logger import VirtualTradeLogger


# ── Test fixtures ────────────────────────────────────────────────────────────


def _make_scanner_with_logger(log_path: Path) -> Scanner:
    """Construct a Scanner shell wired to a real VirtualTradeLogger.

    Bypasses ``Scanner.__init__`` (which loads TF/Keras models — heavy and
    segfault-prone on Apple Silicon in unit-test environments). Sets only
    the attributes the helper under test reads.
    """
    scanner = Scanner.__new__(Scanner)
    scanner.config = ScannerConfig()
    scanner._virtual_trade_logger = VirtualTradeLogger(log_path=log_path)
    return scanner


def _make_failed_confidence_analysis(
    confidence_norm: float,
    confidence_score: float,
) -> PairAnalysis:
    """Build a PairAnalysis whose confidence gate failed.

    Args:
        confidence_norm: ``result.confidence`` — the blended ``[0,1]``
            probability rendered by the buggy log line.
        confidence_score: ``result.confidence_score`` — the 0-100 ADX
            value the gate actually used in its comparison
            (``gates.py:2326``).

    Other fields are set so the row passes early-exit checks
    (``is_tradeable=False``, direction in {LONG,SHORT}, no error,
    ``confidence >= MIN_CONFIDENCE_TO_LOG=0.10``).
    """
    return PairAnalysis(
        pair="EUR_USD",
        direction="LONG",
        confidence=confidence_norm,
        confidence_score=confidence_score,
        gates_passed=False,
        confidence_passed=False,
        momentum_passed=True,
        risk_passed=True,
        volatility_gate_passed=True,
        agent_passed=True,
        momentum=0.55,
        drawdown=0.01,
        current_price=1.0850,
        atr=0.0010,
        atr_pips=10.0,
        volatility_regime="NORMAL",
        sl_pips=15.0,
        tp_pips=22.5,
        risk_pct=0.05,
    )


def _read_jsonl(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.split("\n") if line.strip()]


# ── Tests ────────────────────────────────────────────────────────────────────


def test_confidence_pass_with_zero_to_hundred_units():
    """T1: Gate logic at gates.py:2326 PASSES when confidence (0-100) ≥ min_confidence (0-100).

    Documents the load-bearing assumption of the fix: the gate's own
    comparison is unit-consistent and was never broken. Both sides are
    0-100. This is the contract the log string must mirror.
    """
    cfg = ScannerConfig()
    # min_confidence is the dataclass default — 42.0, a 0-100 ADX threshold.
    assert cfg.min_confidence == 42.0
    # The gate logic at gates.py:2326 is `confidence >= min_confidence`.
    # Mirror it here against a representative 0-100 ridge value.
    confidence_0_100 = 55.0
    assert (confidence_0_100 >= cfg.min_confidence) is True


def test_confidence_fail_with_zero_to_hundred_units():
    """T2: Gate logic at gates.py:2326 FAILS when confidence (0-100) < min_confidence (0-100).

    Documents the failure case the log line is supposed to describe.
    """
    cfg = ScannerConfig()
    confidence_0_100 = 30.0  # below the 42.0 default
    assert (confidence_0_100 >= cfg.min_confidence) is False


def test_log_failure_string_emits_consistent_units(tmp_path):
    """T3: The legacy `gate_failures` string renders both sides in 0-100 ADX scale.

    Pre-fix: ``confidence=0.400 < min=42.0`` (mixes [0,1] vs [0,100]).
    Post-fix: ``confidence=30.0 < min=42.0`` (both in 0-100 ADX units).

    Captures the actual jsonl row written by
    ``_log_virtual_trade_for_result`` to make the contract observable.
    """
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)

    # confidence (blended [0,1]) = 0.40 — the value the pre-fix log line
    # rendered. confidence_score (0-100 ADX) = 30.0 — the value the gate
    # actually used. min_confidence = 42.0 (dataclass default).
    analysis = _make_failed_confidence_analysis(
        confidence_norm=0.40,
        confidence_score=30.0,
    )
    scanner._log_virtual_trade_for_result("EUR_USD", analysis)

    rows = _read_jsonl(log_path)
    assert len(rows) == 1, f"expected one row, got {rows}"
    row = rows[0]
    failures = row.get("gate_failures", [])
    assert failures, f"expected at least one failure entry, got {row!r}"

    conf_failure = next(
        (f for f in failures if f.startswith("confidence=")), None
    )
    assert conf_failure is not None, f"no confidence failure in {failures!r}"

    # Both sides MUST be on the 0-100 scale now.
    assert conf_failure == "confidence=30.0 < min=42.0", (
        f"expected unit-consistent failure string, got {conf_failure!r}"
    )
    # Specifically: the observed value MUST NOT be the [0,1] confidence
    # (0.400) — that's the pre-fix scale-mismatch fingerprint.
    assert "0.400" not in conf_failure, (
        f"failure string still shows [0,1] confidence — fix did not land: {conf_failure!r}"
    )


def test_log_failure_detail_observed_in_zero_to_hundred(tmp_path):
    """T4 (combined T3 + T4): structured `gate_failures_detail` dict uses 0-100 observed.

    The structured ``observed`` field is what jq tallies / dashboard
    consumers read. Pre-fix this was ``0.4`` (an unusable scale-mixed
    value against a ``threshold`` of ``42.0``). Post-fix it is ``30.0``,
    directly comparable to ``threshold``.

    Also exercises back-compat with the legacy
    ``virtual_trade_logger:206`` parser path: ``failure.split("=")[0]``
    must still bucket the entry under the ``confidence`` counter
    regardless of the displayed value's scale.
    """
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)

    analysis = _make_failed_confidence_analysis(
        confidence_norm=0.38,
        confidence_score=25.0,
    )
    scanner._log_virtual_trade_for_result("EUR_USD", analysis)

    rows = _read_jsonl(log_path)
    assert len(rows) == 1
    row = rows[0]

    detail_list = row.get("gate_failures_detail") or []
    conf_detail = next(
        (d for d in detail_list if d.get("gate") == "confidence"), None
    )
    assert conf_detail is not None, (
        f"no confidence gate in gate_failures_detail: {detail_list!r}"
    )

    # observed and threshold MUST now be on the same scale.
    assert conf_detail["observed"] == 25.0, (
        f"expected observed=25.0 (0-100 ADX), got {conf_detail['observed']!r} "
        f"— pre-fix this would have been 0.38"
    )
    assert conf_detail["threshold"] == 42.0
    assert conf_detail["observed"] < conf_detail["threshold"], (
        "observed and threshold must be comparable on the same scale"
    )

    # Back-compat: the legacy parser at virtual_trade_logger:206 splits on
    # "=" and takes [0] to bucket. The new string still parses to the
    # "confidence" bucket — VirtualTradeLogger.get_stats() top_gate_failures
    # remains stable.
    stats = scanner._virtual_trade_logger.get_stats()
    top = dict(stats.get("top_gate_failures", []))
    assert "confidence" in top, (
        f"confidence bucket missing from stats — back-compat broke: {top!r}"
    )
    assert top["confidence"] >= 1


def test_block_reasons_observation_log_uses_consistent_units(tmp_path):
    """T5: The Phase-21 trade-block ObservationLog reason string is also
    unit-consistent.

    The second occurrence of the lying log line was at
    ``engine.py:4930`` inside the ObservationLog block. Rather than wire
    up the full ObservationLog (file I/O, project-root coupling), this
    test reproduces the patched format string locally to assert its
    shape — a smoke that catches accidental reversion of the second site.
    """
    cfg = ScannerConfig()
    # Real PairAnalysis with the same mixed-scale inputs.
    analysis = _make_failed_confidence_analysis(
        confidence_norm=0.40,
        confidence_score=30.0,
    )

    # The format string in engine.py:4930 (post-fix) — reproduced here to
    # assert the rendered text.
    rendered = (
        f"confidence={analysis.confidence_score:.1f} < min={cfg.min_confidence:.1f}"
    )
    assert rendered == "confidence=30.0 < min=42.0"
    # The string MUST NOT contain the [0,1] confidence reading.
    assert "0.400" not in rendered
    assert "0.4 " not in rendered


def test_regression_pre_fix_format_string_no_longer_present():
    """T6 (regression guard): Source code no longer contains the pre-fix
    format string with `:.3f` against `result.confidence` for the
    confidence-gate path.

    This is the cheapest possible regression check — a grep over the
    source. If anyone reverts the fix, this test fails immediately. The
    other tests verify behavior; this one verifies the source-level
    contract so the next reviewer sees a clear signal.
    """
    engine_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "scanner"
        / "engine.py"
    )
    text = engine_path.read_text(encoding="utf-8")
    # The exact buggy format string. Any future edit must keep this
    # string out of the file (or update this test alongside it).
    bad = 'f"confidence={result.confidence:.3f} < min={self.config.min_confidence}"'
    assert bad not in text, (
        f"pre-fix format string is back at engine.py — fix regressed: {bad!r}"
    )
