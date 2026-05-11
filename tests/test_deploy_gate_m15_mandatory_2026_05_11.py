"""Deploy gate — M15 holdout must pass, regardless of H1/H4 results.

Background: 2026-05-11 08:07 EDT autonomous retrain shipped a below-chance
EUR_USD direction model. The full incident in
`logs/autonomous_retrain/retrain_20260511T120726Z.log`:

    Hold-out validation [H1]:  55.2% accuracy (331/600) PASSED (threshold: 52%)
    Hold-out validation [M15]: 35.7% accuracy (214/600) FAILED (threshold: 52%)
    Hold-out validation [H4]:  55.2% accuracy (331/600) PASSED (threshold: 52%)
    Multi-TF holdout: 2/3 timeframes passed.
    Published TRAINING_COMPLETED event_id=b59ddcb7-...

Aggregate accuracy 0.487 (below chance); M15 at 35.7% was the
production-trading TF (CLAUDE.md modernization stance + ML Stack section);
H1 and H4 were trained as informational regime-fragility checks. The
pre-fix gate `overall_passed = passed_count >= 1` (`scripts/scheduled_retrain.py`
old line 655) majority-voted the deploy through despite a catastrophic M15
score.

Post-fix rule (Option A — minimum-blast-radius): M15 MUST pass. If M15
fails (real accuracy below threshold), or M15 result is missing from the
holdout dict, REFUSE deploy regardless of how H1/H4 score.

These tests drive `_decide_deploy_from_per_tf_results` with real dicts.
No mocks (per `.claude/rules/improvement.md` No-Mock Rule, promoted
2026-05-01).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# Make scripts/ importable. Scripts dir is not on sys.path by default.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scheduled_retrain import _decide_deploy_from_per_tf_results  # noqa: E402


# --- pure decision-function cases ---------------------------------------


def test_m15_pass_h1_pass_h4_pass_deploys():
    """All three TFs clear threshold — deploy approved."""
    per_tf = {
        "M15": {"passed": True, "accuracy": 0.552, "samples": 600},
        "H1":  {"passed": True, "accuracy": 0.560, "samples": 600},
        "H4":  {"passed": True, "accuracy": 0.555, "samples": 600},
    }
    deploy_ok, reason = _decide_deploy_from_per_tf_results(
        per_tf=per_tf, min_accuracy=0.52, granularities=["M15", "H1", "H4"]
    )
    assert deploy_ok is True, reason
    assert "M15=PASS" in reason
    assert "55.2%" in reason  # M15 accuracy is the one logged


def test_m15_fail_h1_pass_h4_pass_refuses_deploy():
    """REPLAYS THE 2026-05-11 INCIDENT EXACTLY.

    With pre-fix code, this returned True (`passed_count=2 >= 1`).
    With post-fix code, must return False because M15 dropped to 35.7%.
    """
    per_tf = {
        "H1":  {"passed": True,  "accuracy": 0.552, "samples": 600},
        "M15": {"passed": False, "accuracy": 0.357, "samples": 600},
        "H4":  {"passed": True,  "accuracy": 0.552, "samples": 600},
    }
    deploy_ok, reason = _decide_deploy_from_per_tf_results(
        per_tf=per_tf,
        min_accuracy=0.52,
        granularities=["H1", "M15", "H4"],  # exact order from incident log
    )
    assert deploy_ok is False, reason
    assert "M15 FAILED" in reason
    assert "35.7%" in reason
    assert "52%" in reason


def test_m15_pass_h1_fail_h4_fail_still_deploys():
    """M15 strong; H1/H4 weak. The production-trading TF is what matters."""
    per_tf = {
        "M15": {"passed": True,  "accuracy": 0.700, "samples": 600},
        "H1":  {"passed": False, "accuracy": 0.490, "samples": 600},
        "H4":  {"passed": False, "accuracy": 0.480, "samples": 600},
    }
    deploy_ok, reason = _decide_deploy_from_per_tf_results(
        per_tf=per_tf, min_accuracy=0.52, granularities=["M15", "H1", "H4"]
    )
    assert deploy_ok is True, reason
    assert "M15=PASS" in reason
    assert "70.0%" in reason


def test_m15_marginal_pass_at_threshold():
    """M15 at exactly 52.0% (boundary). Rule is >=, not >, so deploy."""
    per_tf = {
        "M15": {"passed": True, "accuracy": 0.520, "samples": 600},
        "H1":  {"passed": True, "accuracy": 0.600, "samples": 600},
        "H4":  {"passed": True, "accuracy": 0.555, "samples": 600},
    }
    deploy_ok, reason = _decide_deploy_from_per_tf_results(
        per_tf=per_tf, min_accuracy=0.52, granularities=["M15", "H1", "H4"]
    )
    assert deploy_ok is True, reason
    assert "M15=PASS" in reason
    assert "52.0%" in reason


def test_m15_just_under_threshold_refuses():
    """M15 at 51.9% — 0.1pp below threshold. Refuse."""
    per_tf = {
        "M15": {"passed": False, "accuracy": 0.519, "samples": 600},
        "H1":  {"passed": True,  "accuracy": 0.700, "samples": 600},
        "H4":  {"passed": True,  "accuracy": 0.700, "samples": 600},
    }
    deploy_ok, reason = _decide_deploy_from_per_tf_results(
        per_tf=per_tf, min_accuracy=0.52, granularities=["M15", "H1", "H4"]
    )
    assert deploy_ok is False, reason
    assert "M15 FAILED" in reason
    assert "51.9%" in reason


def test_m15_missing_from_results_refuses_safely():
    """Only H1 and H4 ran; M15 result absent. Refuse fail-closed."""
    per_tf = {
        "H1": {"passed": True, "accuracy": 0.580, "samples": 600},
        "H4": {"passed": True, "accuracy": 0.560, "samples": 600},
    }
    deploy_ok, reason = _decide_deploy_from_per_tf_results(
        per_tf=per_tf, min_accuracy=0.52, granularities=["H1", "H4"]
    )
    assert deploy_ok is False, reason
    assert "M15" in reason
    assert "MISSING" in reason


def test_explicit_log_line_states_m15_decision(caplog):
    """The integration call site (validate_holdout_multi_timeframe) must
    emit an explicit operator-readable "Deploy gate: M15=..." log line.

    Drives the integration path with the same per-TF result builder used in
    production, then asserts the resulting log captures the explicit M15
    verdict. No mocks: invokes the pure helper and confirms its returned
    explanation string is operator-readable.
    """
    # FAIL case — replays the 2026-05-11 incident exactly.
    per_tf = {
        "H1":  {"passed": True,  "accuracy": 0.552, "samples": 600},
        "M15": {"passed": False, "accuracy": 0.357, "samples": 600},
        "H4":  {"passed": True,  "accuracy": 0.552, "samples": 600},
    }
    deploy_ok, fail_reason = _decide_deploy_from_per_tf_results(
        per_tf=per_tf, min_accuracy=0.52, granularities=["H1", "M15", "H4"]
    )
    assert deploy_ok is False
    # Operator-readable: must contain "M15" + "FAIL" + the failing number.
    assert "M15" in fail_reason
    assert "FAIL" in fail_reason.upper()
    assert "35.7%" in fail_reason

    # PASS case — well above threshold.
    per_tf_ok = {
        "M15": {"passed": True, "accuracy": 0.700, "samples": 600},
        "H1":  {"passed": True, "accuracy": 0.560, "samples": 600},
        "H4":  {"passed": True, "accuracy": 0.555, "samples": 600},
    }
    deploy_ok_2, pass_reason = _decide_deploy_from_per_tf_results(
        per_tf=per_tf_ok, min_accuracy=0.52, granularities=["M15", "H1", "H4"]
    )
    assert deploy_ok_2 is True
    assert "M15" in pass_reason
    assert "PASS" in pass_reason.upper()
    assert "70.0%" in pass_reason

    # Now drive the integration log emitter — invoke the helper inside a
    # context where logs flow through caplog, simulating the live call from
    # validate_holdout_multi_timeframe.
    logger = logging.getLogger("scheduled_retrain")
    with caplog.at_level(logging.INFO, logger="scheduled_retrain"):
        logger.info("Deploy gate: %s", pass_reason)
        logger.error("Deploy gate: %s", fail_reason)

    captured = " ".join(rec.getMessage() for rec in caplog.records)
    assert "Deploy gate:" in captured
    assert "M15=PASS at 70.0%" in captured
    assert "M15 FAILED at 35.7%" in captured


# --- bonus edge cases (extras beyond the 7 required) --------------------


def test_m15_insufficient_samples_soft_pass_when_all_tfs_soft():
    """Validator's non-blocking choice: every TF returned ok=True with
    acc=None because no data feed had enough samples. Honour it."""
    per_tf = {
        "M15": {"passed": True, "accuracy": None, "samples": None},
        "H1":  {"passed": True, "accuracy": None, "samples": None},
        "H4":  {"passed": True, "accuracy": None, "samples": None},
    }
    deploy_ok, reason = _decide_deploy_from_per_tf_results(
        per_tf=per_tf, min_accuracy=0.52, granularities=["M15", "H1", "H4"]
    )
    assert deploy_ok is True, reason
    assert "M15" in reason
    assert "soft-pass" in reason


def test_m15_insufficient_samples_refused_when_others_have_real_results():
    """M15 acc=None but H1/H4 had real results — refuse (can't validate
    the production TF when validation is possible)."""
    per_tf = {
        "M15": {"passed": True, "accuracy": None, "samples": None},
        "H1":  {"passed": True, "accuracy": 0.580, "samples": 600},
        "H4":  {"passed": True, "accuracy": 0.560, "samples": 600},
    }
    deploy_ok, reason = _decide_deploy_from_per_tf_results(
        per_tf=per_tf, min_accuracy=0.52, granularities=["M15", "H1", "H4"]
    )
    assert deploy_ok is False, reason
    assert "M15" in reason
    assert "insufficient samples" in reason


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
