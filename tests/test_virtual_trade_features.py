"""Tests for the virtual-trade `features` capture fix (2026-07-30).

Bug: `engine.py:_log_virtual_trade_for_result` passed a HARDCODED EMPTY DICT
literal — `features={}` — to `VirtualTradeLogger.log_virtual_trade(...)`. The
logger's signature has always accepted a real snapshot
(`virtual_trade_logger.py:102`, `features: Optional[Dict[str, float]]`) and the
downstream RL consumer (`src/training/rl/trajectory_loader.py`) expects one, so
the key was present but empty on all 4427 rows of
`trained_data/virtual_trades.jsonl`. That is the load-bearing blocker on the
Phase J1 RL prerequisite "rejected setups are recorded with full features"
(`docs/architecture/audit/phase_I_J1_J7_models.md` §6 prereq #4, gap G3).

Rejection features are unbackfillable, so every scan cycle logged with `{}`
permanently destroyed that row's forensic value.

Fix:
  1. `_log_virtual_trade_for_result` now builds `_vtl_features` from values
     ALREADY computed and carried on the `PairAnalysis` at rejection time —
     model head outputs, market/regime state, the uncertainty triplet,
     execution-quality inputs, model health, agent-consensus vote maths, the
     counterfactual trade's sizing, and the ensemble component scores stashed
     from `gate_details["scores"]` at `engine.py:4871-4876`.
  2. Values are coerced to float defensively; non-numeric shapes (None,
     categorical strings, nested dicts, NaN/inf) are SKIPPED, never
     zero-filled — per `.claude/rules/improvement.md` "Train<->Inference
     Contract Gates".
  3. The enclosing `except Exception` was raised from DEBUG to WARNING so a
     capture failure is visible in `logs/buddy_debug.log` (no-silent-failure
     rule) while remaining non-fatal — capture must never block a scan.

Per `.claude/rules/improvement.md` "No-Mock Rule": real `ScannerConfig`, real
`VirtualTradeLogger` writing to a real `tmp_path`, real `PairAnalysis` objects,
real JSONL read back off disk. No `unittest.mock`, no `MagicMock`, no `patch`,
no test-double classes. The `Scanner.__new__` shell (skipping the TF/Keras model
load in `__init__`) mirrors the established pattern in
`tests/test_virtual_trade_agent_scores.py:57` and
`tests/test_virtual_trade_logger_ordering.py`.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

from src.scanner.config import ScannerConfig
from src.scanner.engine import Scanner
from src.scanner.results import PairAnalysis
from src.scanner.virtual_trade_logger import VirtualTradeLogger


# ── Fixtures (real objects only) ─────────────────────────────────────────────


def _make_scanner_with_logger(log_path: Path) -> Scanner:
    """Real Scanner shell wired to a real VirtualTradeLogger on real disk."""
    scanner = Scanner.__new__(Scanner)
    scanner.config = ScannerConfig()
    scanner._virtual_trade_logger = VirtualTradeLogger(log_path=log_path)
    return scanner


def _verdict(name: str, score: float, passed: bool) -> Dict[str, Any]:
    """One `agent_reasons` entry in the shape `AgentVerdict.to_dict()` emits."""
    return {
        "name": name,
        "score": float(score),
        "passed": bool(passed),
        "weight": 1.0,
        "reason": "",
        "reason_code": "",
        "confidence_delta": 0.0,
        "block_trade": False,
        "metadata": {},
    }


def _rejected_analysis(**overrides: Any) -> PairAnalysis:
    """A realistically-populated PairAnalysis that fails `is_tradeable`.

    Field values mirror what `engine.py:4836-4869` constructs on a live scan,
    so the test exercises the same decision surface the fix reads.
    """
    kwargs: Dict[str, Any] = dict(
        pair="EUR_USD",
        direction="SHORT",
        confidence=0.61,
        # gate/model head outputs
        tcn_confidence=0.58,
        tcn_probability=0.44,
        ridge_confidence=47.5,
        momentum=0.33,
        xgb_momentum=0.33,
        momentum_acceleration=True,
        momentum_passed=False,
        confidence_score=41.2,
        confidence_passed=False,
        drawdown=0.021,
        rf_drawdown=0.021,
        risk_passed=True,
        volatility_gate_passed=True,
        entry_score=0.55,
        # market state
        current_price=1.0842,
        atr=0.00091,
        atr_pips=9.1,
        volatility_percentile=0.62,
        volatility_regime="HIGH",
        trend_strength=0.27,
        # uncertainty triplet
        uncertainty_score=0.38,
        confidence_variance=0.04,
        model_disagreement=0.19,
        # execution quality
        execution_quality_passed=True,
        execution_quality_score=0.72,
        spread_pips=1.4,
        est_slippage_pips=0.3,
        liquidity_score=0.88,
        # model health
        pair_model_accuracy=0.517,
        model_drift_score=0.12,
        # agent consensus
        agent_votes=2,
        agent_total=5,
        agent_score=0.41,
        agent_passed=False,
        weighted_vote_score=0.58,
        weighted_vote_threshold=0.65,
        agent_reasons=[
            _verdict("trend", 0.31, passed=False),
            _verdict("risk_sentinel", 0.79, passed=True),
        ],
        # counterfactual sizing
        sl_pips=12.5,
        tp_pips=25.0,
        risk_pct=0.05,
        recommended_lots=0.4,
        gates_passed=False,
    )
    kwargs.update(overrides)
    return PairAnalysis(**kwargs)


def _read_entries(log_path: Path) -> List[Dict[str, Any]]:
    """Load all JSONL rows written to the real virtual_trades file."""
    if not log_path.exists():
        return []
    raw = log_path.read_text().strip()
    if not raw:
        return []
    return [json.loads(line) for line in raw.split("\n") if line.strip()]


# ── T1: the headline regression — features is NON-EMPTY and float-valued ─────


def test_logged_virtual_trade_carries_non_empty_float_features(
    tmp_path: Path,
) -> None:
    """The pre-fix payload was `{}` on every row. It must now be populated,
    and every value must be a real float (the RL consumer's contract)."""
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)
    analysis = _rejected_analysis()

    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    entries = _read_entries(log_path)
    assert len(entries) == 1, f"expected exactly one row, got {len(entries)}"

    features = entries[0]["features"]
    assert isinstance(features, dict)
    assert features, "features must NOT be the pre-fix empty dict"
    assert len(features) >= 20, (
        f"decision surface should carry the full snapshot, got {len(features)} "
        f"keys: {sorted(features)}"
    )
    for key, value in features.items():
        assert isinstance(key, str) and key, f"bad feature key: {key!r}"
        assert isinstance(value, float), (
            f"feature {key!r} must be float, got {type(value).__name__}: {value!r}"
        )
        assert math.isfinite(value), f"feature {key!r} is non-finite: {value!r}"


# ── T2: the captured values are the REAL decision inputs, not placeholders ───


def test_features_mirror_the_analysis_decision_surface(tmp_path: Path) -> None:
    """Spot-check that each captured group carries the value the rejection was
    actually made from — not a default, not a zero-fill."""
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)
    analysis = _rejected_analysis()

    scanner._log_virtual_trade_for_result(analysis.pair, analysis)
    features = _read_entries(log_path)[0]["features"]

    # model head outputs
    assert features["confidence"] == 0.61
    assert features["confidence_score"] == 41.2
    assert features["tcn_probability"] == 0.44
    assert features["ridge_confidence"] == 47.5
    assert features["momentum"] == 0.33
    # bool is coerced, not dropped
    assert features["momentum_acceleration"] == 1.0
    # market state
    assert features["atr_pips"] == 9.1
    assert features["current_price"] == 1.0842
    assert features["volatility_percentile"] == 0.62
    # uncertainty triplet — drives the hard vetoes
    assert features["uncertainty_score"] == 0.38
    assert features["model_disagreement"] == 0.19
    # execution quality
    assert features["spread_pips"] == 1.4
    assert features["liquidity_score"] == 0.88
    # model health
    assert features["pair_model_accuracy"] == 0.517
    # agent consensus vote maths
    assert features["weighted_vote_score"] == 0.58
    assert features["weighted_vote_threshold"] == 0.65
    assert features["agent_total"] == 5.0
    # counterfactual trade sizing — needed to score the rejected action
    assert features["sl_pips"] == 12.5
    assert features["tp_pips"] == 25.0

    # agent_scores (fixed 2026-05-19) must remain populated — no regression.
    agent_scores = _read_entries(log_path)[0]["agent_scores"]
    assert agent_scores, "the 2026-05-19 agent_scores fix must not regress"
    assert agent_scores["trend"] == 0.31


# ── T3: the categorical regime is encoded, not dropped ───────────────────────


def test_volatility_regime_is_encoded_as_ordinal(tmp_path: Path) -> None:
    """`volatility_regime` is the one categorical on the decision surface. The
    row schema is float-valued, so it ships as an ordinal; UNKNOWN (itself a
    hard `is_tradeable` veto) is -1.0, which is distinct from any real regime."""
    for regime, expected in (
        ("LOW", 0.0),
        ("NORMAL", 1.0),
        ("HIGH", 2.0),
        ("EXTREME", 3.0),
        ("UNKNOWN", -1.0),
    ):
        log_path = tmp_path / f"vt_{regime}.jsonl"
        scanner = _make_scanner_with_logger(log_path)
        analysis = _rejected_analysis(volatility_regime=regime)

        scanner._log_virtual_trade_for_result(analysis.pair, analysis)

        features = _read_entries(log_path)[0]["features"]
        assert features["volatility_regime_ordinal"] == expected, (
            f"regime {regime} should encode to {expected}"
        )
        # The raw string must never leak into a float-valued schema.
        assert "volatility_regime" not in features


# ── T4: ensemble component scores from gate_details are captured ─────────────


def test_ensemble_scores_are_captured_and_nested_shapes_skipped(
    tmp_path: Path,
) -> None:
    """`_ensemble_scores` is `gate_details["scores"]` stashed onto the result at
    `engine.py:4871-4876` — the per-head numbers the final gate verdict was
    computed from. Scalars are captured; the nested `ensemble_weights` sub-dict
    is skipped rather than flattened or stringified."""
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)
    analysis = _rejected_analysis()
    # Exact shape emitted by modular_inference.py:5107.
    analysis._ensemble_scores = {
        "core_score": 0.487,
        "final_score": 0.402,
        "direction_score": 0.44,
        "confidence_score": 0.51,
        "momentum_score": 0.33,
        "risk_score": 0.62,
        "threshold": 0.55,
        "used_learned_weights": True,
        "ensemble_weights": {
            "direction": 0.35,
            "confidence": 0.30,
            "momentum": 0.20,
            "risk": 0.15,
        },
    }
    analysis._core_score = 0.487
    analysis._final_score = 0.402

    scanner._log_virtual_trade_for_result(analysis.pair, analysis)
    features = _read_entries(log_path)[0]["features"]

    assert features["ensemble_core_score"] == 0.487
    assert features["ensemble_final_score"] == 0.402
    assert features["ensemble_direction_score"] == 0.44
    assert features["ensemble_risk_score"] == 0.62
    assert features["ensemble_threshold"] == 0.55
    assert features["ensemble_used_learned_weights"] == 1.0
    # Nested dict skipped, and it must not collide with the flat scores.
    assert "ensemble_ensemble_weights" not in features
    # The ensemble's own confidence_score must not clobber the analysis field.
    assert features["ensemble_confidence_score"] == 0.51
    assert features["confidence_score"] == 41.2


def test_missing_ensemble_scores_still_logs_a_populated_snapshot(
    tmp_path: Path,
) -> None:
    """A pair that took the technical-fallback path has no `_ensemble_scores`.
    That must not empty the snapshot or raise."""
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)
    analysis = _rejected_analysis()
    assert not hasattr(analysis, "_ensemble_scores")

    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    features = _read_entries(log_path)[0]["features"]
    assert features, "snapshot must survive a missing ensemble payload"
    assert "ensemble_core_score" not in features
    assert features["confidence"] == 0.61


def test_malformed_ensemble_scores_shape_is_guarded(tmp_path: Path) -> None:
    """A non-dict `_ensemble_scores` (corrupt upstream state) must be skipped,
    not crash the capture — same guard style the agent_scores block uses."""
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)
    analysis = _rejected_analysis()
    analysis._ensemble_scores = "not-a-dict"
    analysis._core_score = 0.31

    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    features = _read_entries(log_path)[0]["features"]
    assert features
    # Falls back to the independently-stashed scalar.
    assert features["ensemble_core_score"] == 0.31


# ── T5: non-numeric / non-finite values are skipped, never zero-filled ───────


def test_non_finite_and_non_numeric_values_are_skipped_not_zero_filled(
    tmp_path: Path,
) -> None:
    """Zero-filling a missing feature is forbidden by the Train<->Inference
    Contract Gates — an absent key is honest, a 0.0 stand-in lies. NaN also
    must never reach the JSONL (it is not valid strict JSON)."""
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)
    analysis = _rejected_analysis()
    analysis._ensemble_scores = {
        "core_score": float("nan"),
        "final_score": float("inf"),
        "risk_score": None,
        "direction_score": "0.44",
        "momentum_score": 0.33,
    }

    scanner._log_virtual_trade_for_result(analysis.pair, analysis)

    raw = log_path.read_text()
    assert "NaN" not in raw and "Infinity" not in raw, (
        "non-finite values must never be serialized into the JSONL"
    )
    features = _read_entries(log_path)[0]["features"]
    assert "ensemble_core_score" not in features  # NaN skipped
    assert "ensemble_final_score" not in features  # inf skipped
    assert "ensemble_risk_score" not in features   # None skipped
    assert "ensemble_direction_score" not in features  # str not coerced
    assert features["ensemble_momentum_score"] == 0.33  # the real one survives


# ── T6: capture failure is WARNING-visible and never breaks the scan ─────────


def test_capture_failure_logs_warning_and_does_not_propagate(
    tmp_path: Path,
    caplog: Any,
) -> None:
    """A swallowed DEBUG in a data-capture path violates the no-silent-failure
    rule. The failure must surface at WARNING with the pair and the exception —
    and must NOT propagate, because capture can never be allowed to block a
    scan cycle.

    The failure is induced with a real (untyped) dataclass field value:
    `momentum` as a string makes the existing `f"momentum={...:.3f}"` format in
    the gate-failure block raise. Real object, real exception — no mock.
    """
    log_path = tmp_path / "virtual_trades.jsonl"
    scanner = _make_scanner_with_logger(log_path)
    analysis = _rejected_analysis(momentum="not-a-number", momentum_passed=False)

    with caplog.at_level(logging.WARNING, logger="src.scanner.engine"):
        # Must return normally — no exception escapes into the scan loop.
        assert scanner._log_virtual_trade_for_result(analysis.pair, analysis) is None

    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "virtual trade capture FAILED" in r.getMessage()
    ]
    assert warnings, (
        "capture failure must be logged at WARNING, not swallowed at DEBUG; "
        f"records seen: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
    message = warnings[0].getMessage()
    assert "EUR_USD" in message, f"warning must name the pair: {message}"
    assert warnings[0].exc_info is not None, "warning must carry the stack"
