"""Orphaned-action quarantine — suppress dead writes, keep the framework.

No mocks: real ``ScannerConfig``, real ``SelfHeal``, real disk via
``tmp_path``.

Context. The live lane is the OANDA practice trend lane
(``src/equity/oanda_trend.py``). Every self-heal handler writes to a target
that lane does not read -- verified by grep: zero references to
``config_adjustments.json``, ``agent_weights.json`` or any scanner gate
threshold. The files that ARE read by ``src/scanner/engine.py`` and
``src/core/modular_inference.py`` belong to the directional FX scanner stack,
which LESSONS L-016 retires and whose scanner is not running.

The quarantine suppresses the orphaned WRITE while preserving detection,
debounce, budgets, logging and the circuit breaker. Approving the live FX trend
lane does not revive the retired stack.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.scanner.config import ScannerConfig
from src.scanner.feedback.self_heal import SelfHeal

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_LANE = REPO_ROOT / "src" / "equity" / "oanda_trend.py"


def _engine(tmp_path: Path) -> SelfHeal:
    """Real SelfHeal with every shared disk path redirected to tmp_path.

    Mirrors the redirection the existing suite uses, so this test can never
    touch project state.
    """
    engine = SelfHeal(config=ScannerConfig())
    engine.JOURNAL_PATH = tmp_path / "trade_journal.json"
    engine.CONFIG_ADJUSTMENTS_PATH = tmp_path / "config_adjustments.json"
    engine.DEBOUNCE_STATE_PATH = tmp_path / "debounce.json"
    engine.DECISION_LOG_PATH = tmp_path / "decisions.jsonl"
    engine.ERROR_LOG_PATH = tmp_path / "unhandled_errors.jsonl"
    engine.DEGRADED_FLAG_PATH = tmp_path / "degraded.flag"
    engine.ACTION_BUDGET_PATH = tmp_path / "action_budget.json"
    engine.AGENT_WEIGHTS_PATH = tmp_path / "agent_weights.json"
    engine.RETRAIN_REQUESTS_DIR = tmp_path / "retrain_requests"
    return engine


def _diag(check: str, action: str) -> dict:
    return {
        "status": "DEGRADED",
        "issues": [{"check": check, "severity": "warning", "detail": "test"}],
        "recommended_actions": [action],
    }


ORPHANED = [
    ("direction_accuracy", "retrain_gates"),
    ("rl_model_staleness", "retrain_rl_position_sizer"),
    ("agent_weight_boundary", "soft_reset_agent_weight:trend"),
    ("gate_firing_rate", "reset_gate_threshold_to_default:min_confidence"),
    ("gate_firing_rate", "tighten_gate_threshold:min_confidence"),
    ("drawdown_streak", "reduce_risk_per_trade_pct"),
]


# ---------------------------------------------------------------------------
# The quarantine itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("check,action", ORPHANED)
def test_every_orphaned_action_is_recorded_but_not_executed(tmp_path, check, action):
    engine = _engine(tmp_path)

    result = engine.apply(_diag(check, action))

    taken = result["actions_taken"]
    assert len(taken) == 1
    assert taken[0]["action"] == action
    assert taken[0]["success"] is False, "an orphaned action must never report success"
    assert "quarantined_no_live_consumer" in taken[0]["detail"]


@pytest.mark.parametrize("check,action", ORPHANED)
def test_quarantined_action_writes_nothing_to_disk(tmp_path, check, action):
    """The whole point: the dead write does not happen."""
    engine = _engine(tmp_path)
    before = sorted(p.name for p in tmp_path.rglob("*") if p.is_file())

    engine.apply(_diag(check, action))

    after = sorted(p.name for p in tmp_path.rglob("*") if p.is_file())
    new = set(after) - set(before)
    assert not (new & {"config_adjustments.json", "agent_weights.json"}), (
        f"quarantined action still wrote: {new}"
    )


def test_retrain_marker_accumulation_stops(tmp_path):
    """42 markers accumulated since 2026-07-07 for an artifact nothing reads."""
    engine = _engine(tmp_path)
    markers = tmp_path / "retrain_requests"

    for _ in range(3):
        engine.apply(_diag("rl_model_staleness", "retrain_rl_position_sizer"))

    written = list(markers.glob("*.json")) if markers.exists() else []
    assert written == [], f"quarantine must stop marker accumulation, got {written}"


def test_every_dispatch_verb_is_quarantined_or_deliberately_live(tmp_path):
    """Drift guard.

    If a handler is added later it must be an explicit decision whether it is
    live-mapped or quarantined -- not an accident. Today ALL are quarantined;
    a new live handler must be added to ``_LIVE_MAPPED`` here with a real
    trend-lane consumer.
    """
    engine = _engine(tmp_path)
    _LIVE_MAPPED: set = set()  # nothing is live-mapped yet, by design

    for verb in engine._dispatch:
        assert (
            verb in engine._QUARANTINED_ACTIONS or verb in _LIVE_MAPPED
        ), f"handler {verb!r} is neither quarantined nor declared live-mapped"


# ---------------------------------------------------------------------------
# The framework must survive
# ---------------------------------------------------------------------------


def test_diagnosis_is_still_visible(tmp_path):
    """Quarantine suppresses the WRITE, not the detection."""
    engine = _engine(tmp_path)

    result = engine.apply(_diag("rl_model_staleness", "retrain_rl_position_sizer"))

    assert result["actions_taken"], "the diagnosis must remain visible to the operator"
    assert result["actions_taken"][0]["action"] == "retrain_rl_position_sizer"
    assert "status" in result


def test_unknown_actions_still_fail_closed(tmp_path):
    engine = _engine(tmp_path)

    result = engine.apply(_diag("whatever", "definitely_not_a_real_action"))

    taken = result["actions_taken"]
    assert taken and taken[0]["success"] is False


def test_quarantine_registry_documents_a_reason_for_every_entry(tmp_path):
    """A silent suppression list is how this class of rot starts."""
    engine = _engine(tmp_path)

    for verb, reason in engine._QUARANTINED_ACTIONS.items():
        assert isinstance(reason, str) and len(reason) > 20, (
            f"{verb} needs a substantive reason, got {reason!r}"
        )


# ---------------------------------------------------------------------------
# The premise: the live lane really does not read these
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not LIVE_LANE.exists(), reason="live trend lane not present")
@pytest.mark.parametrize("target", [
    "config_adjustments", "agent_weights.json", "gate_threshold",
])
def test_live_trend_lane_does_not_read_quarantined_targets(target):
    """If this ever fails, the quarantine premise is void -- re-evaluate.

    This is the assertion that keeps the quarantine honest: it is justified
    ONLY while the live lane genuinely ignores these files.
    """
    src = LIVE_LANE.read_text()
    assert target not in src, (
        f"{target!r} now appears in oanda_trend.py -- the live lane may consume it, "
        "so the corresponding quarantine entry must be re-examined"
    )


@pytest.mark.skipif(not LIVE_LANE.exists(), reason="live trend lane not present")
def test_live_lane_sizes_with_its_own_controls():
    """Positive half: name what the lane DOES use, so a swap is visible."""
    src = LIVE_LANE.read_text()
    for expected in ("risk_normalized_units", "evaluate_runtime_risk", "scale_target_units"):
        assert expected in src, f"live sizing control {expected!r} disappeared"
