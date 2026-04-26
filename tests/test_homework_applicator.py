"""Tests for TrainingSignalApplicator — closes the v1 gap by writing
training signal deltas back to agent_weights.json.

See spec §4.5 (Training Signal payload) + Phase 96.5 close-the-loop fix.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from src.scanner.automation.homework.applicator import TrainingSignalApplicator
from src.scanner.automation.homework.types import TrainingSignal


def _make_signal(
    regime: str = "NORMAL",
    deltas: dict | None = None,
    regime_prior_deltas: dict | None = None,
) -> TrainingSignal:
    return TrainingSignal(
        homework_id="hw-test",
        trade_id="t-1",
        outcome="SL",
        agent_weight_deltas=dict(deltas or {}),
        regime_prior_deltas=dict(regime_prior_deltas or {}),
        heuristic_fired=None,
        operator_action="approved",
        operator_note=None,
        regime=regime,
    )


@pytest.fixture
def weights_file(tmp_path: Path) -> Path:
    path = tmp_path / "agent_weights.json"
    path.write_text(json.dumps({
        "NORMAL": {"trend": 1.0, "mean_reversion": 0.9, "uncertainty": 1.1},
        "HIGH": {"trend": 1.15, "mean_reversion": 0.9},
        "LOW": {"trend": 1.0, "mean_reversion": 0.95},
        "_meta": {"last_updated": "2026-04-25T00:00:00Z"},
    }))
    return path


def test_apply_single_signal_updates_correct_regime(weights_file: Path) -> None:
    applicator = TrainingSignalApplicator(weights_path=weights_file)
    signal = _make_signal(regime="NORMAL", deltas={"trend": 0.05})
    result = applicator.apply(signal)

    after = json.loads(weights_file.read_text())
    assert after["NORMAL"]["trend"] == pytest.approx(1.05)
    # Other regimes untouched
    assert after["HIGH"]["trend"] == 1.15
    assert "trend" in result["agents_updated"]
    assert result["regime"] == "NORMAL"


def test_clipping_at_max(weights_file: Path) -> None:
    data = json.loads(weights_file.read_text())
    data["NORMAL"]["trend"] = 1.99
    weights_file.write_text(json.dumps(data))

    applicator = TrainingSignalApplicator(weights_path=weights_file)
    signal = _make_signal(regime="NORMAL", deltas={"trend": 0.05})
    result = applicator.apply(signal)

    after = json.loads(weights_file.read_text())
    assert after["NORMAL"]["trend"] == pytest.approx(2.0)
    assert "trend" in result["clipped"]


def test_clipping_at_min(weights_file: Path) -> None:
    data = json.loads(weights_file.read_text())
    data["NORMAL"]["trend"] = 0.12
    weights_file.write_text(json.dumps(data))

    applicator = TrainingSignalApplicator(weights_path=weights_file)
    signal = _make_signal(regime="NORMAL", deltas={"trend": -0.05})
    result = applicator.apply(signal)

    after = json.loads(weights_file.read_text())
    assert after["NORMAL"]["trend"] == pytest.approx(0.1)
    assert "trend" in result["clipped"]


def test_missing_regime_logs_and_skips(weights_file: Path, caplog) -> None:
    applicator = TrainingSignalApplicator(weights_path=weights_file)
    signal = _make_signal(regime="EXTREME", deltas={"trend": 0.05})

    before = weights_file.read_text()
    result = applicator.apply(signal)
    after = weights_file.read_text()

    # File NOT rewritten, no agents updated
    assert before == after
    assert result["agents_updated"] == []


def test_empty_deltas_is_noop(weights_file: Path) -> None:
    applicator = TrainingSignalApplicator(weights_path=weights_file)
    signal = _make_signal(regime="NORMAL", deltas={})
    mtime_before = weights_file.stat().st_mtime_ns

    result = applicator.apply(signal)

    mtime_after = weights_file.stat().st_mtime_ns
    assert mtime_before == mtime_after
    assert result["agents_updated"] == []


def test_atomic_write_no_tmp_left_behind(weights_file: Path) -> None:
    applicator = TrainingSignalApplicator(weights_path=weights_file)
    signal = _make_signal(regime="NORMAL", deltas={"trend": 0.01})
    applicator.apply(signal)

    leftover = list(weights_file.parent.glob("*.tmp"))
    assert leftover == []


def test_concurrent_apply_serialized_by_flock(weights_file: Path) -> None:
    """Two threads applying signals to same regime — both deltas must land cumulatively."""
    applicator = TrainingSignalApplicator(weights_path=weights_file)
    barrier = threading.Barrier(2)

    def worker(delta_value: float) -> None:
        barrier.wait()
        applicator.apply(_make_signal(regime="NORMAL", deltas={"trend": delta_value}))

    t1 = threading.Thread(target=worker, args=(0.05,))
    t2 = threading.Thread(target=worker, args=(0.03,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    after = json.loads(weights_file.read_text())
    # 1.0 + 0.05 + 0.03 == 1.08 (regardless of order, both deltas must apply)
    assert after["NORMAL"]["trend"] == pytest.approx(1.08)


def test_metadata_preserved_on_rewrite(weights_file: Path) -> None:
    """_meta key (or any non-regime metadata) survives the rewrite."""
    applicator = TrainingSignalApplicator(weights_path=weights_file)
    signal = _make_signal(regime="NORMAL", deltas={"trend": 0.01})
    applicator.apply(signal)

    after = json.loads(weights_file.read_text())
    assert "_meta" in after
    assert after["_meta"]["last_updated"] == "2026-04-25T00:00:00Z"


def test_regime_prior_deltas_applied(weights_file: Path) -> None:
    """If regime_prior_deltas targets an existing regime row, it adds that delta to that row's agent.

    The current agent_weights.json schema is dict[regime, dict[agent, weight]] — so regime_prior_deltas
    behaves as a way to nudge weights in regimes OTHER than the signal's primary regime.
    """
    applicator = TrainingSignalApplicator(weights_path=weights_file)
    signal = _make_signal(
        regime="NORMAL",
        deltas={"trend": 0.01},
        regime_prior_deltas={"HIGH": {"trend": 0.02}},
    )
    applicator.apply(signal)

    after = json.loads(weights_file.read_text())
    assert after["NORMAL"]["trend"] == pytest.approx(1.01)
    assert after["HIGH"]["trend"] == pytest.approx(1.17)


def test_corrupt_weights_file_is_safe(tmp_path: Path) -> None:
    """If weights file is malformed JSON, apply must not crash and must not write garbage."""
    path = tmp_path / "weights.json"
    path.write_text("{not valid json")
    applicator = TrainingSignalApplicator(weights_path=path)
    signal = _make_signal(regime="NORMAL", deltas={"trend": 0.05})

    result = applicator.apply(signal)
    assert result["agents_updated"] == []


def test_unknown_agent_in_deltas_is_added(weights_file: Path) -> None:
    """If signal references an agent not currently in the regime, it gets added at delta value (clipped)."""
    applicator = TrainingSignalApplicator(weights_path=weights_file)
    signal = _make_signal(regime="NORMAL", deltas={"new_agent": 0.5})
    applicator.apply(signal)

    after = json.loads(weights_file.read_text())
    # Started from 1.0 (default for unknown agent), +0.5 = 1.5
    assert after["NORMAL"]["new_agent"] == pytest.approx(1.5)
