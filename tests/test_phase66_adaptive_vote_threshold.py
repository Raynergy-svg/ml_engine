"""Phase 66 US-390 + integration — AdaptiveVoteThreshold unit and integration tests.

Tests cover:
- tick() triggers correctly on NEAR_THRESHOLD
- tick() respects cooldown
- tick() respects max_adaptations
- tick() respects floor guard
- non-NEAR_THRESHOLD classifications always return None
- save/load round-trip
- get_status() returns all required fields
- Wiring greps in engine.py
- Behavioral simulation of full adaptation lifecycle
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.scanner.adaptive_vote_threshold import (
    AdaptiveVoteThreshold,
    CLASS_NEAR_THRESHOLD,
)

ENGINE_SRC = Path("src/scanner/engine.py")
CONTINUOUS_SRC = Path("src/scanner/automation/continuous.py")
SYNTHESIZER_SRC = Path("src/scanner/scan_health_synthesizer.py")


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_avt(**kwargs) -> tuple[AdaptiveVoteThreshold, Path]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    path.unlink()
    avt = AdaptiveVoteThreshold(state_path=path, **kwargs)
    return avt, path


# ─�� US-390 Unit Tests ────────────────────────────────────────────────────────


class TestAdaptiveVoteThreshold_Trigger:
    """tick() triggers adaptation when all guards pass."""

    def test_triggers_on_near_threshold(self):
        avt, _ = _make_avt(step=0.02, cooldown=20, max_adaptations=5, floor=0.40)
        result = avt.tick(CLASS_NEAR_THRESHOLD, current_threshold=0.55)
        assert result == pytest.approx(0.53, abs=1e-4)

    def test_adaptation_count_increments(self):
        avt, _ = _make_avt()
        avt.tick(CLASS_NEAR_THRESHOLD, 0.55)
        assert avt.get_status()["adaptation_count"] == 1

    def test_cooldown_set_after_adaptation(self):
        avt, _ = _make_avt(cooldown=20)
        avt.tick(CLASS_NEAR_THRESHOLD, 0.55)
        assert avt.get_status()["cooldown_remaining"] == 20

    def test_total_reduction_accumulates(self):
        avt, _ = _make_avt(step=0.02, cooldown=0)
        avt.tick(CLASS_NEAR_THRESHOLD, 0.55)
        avt.tick(CLASS_NEAR_THRESHOLD, 0.53)
        assert avt.get_status()["total_reduction"] == pytest.approx(0.04, abs=1e-5)


class TestAdaptiveVoteThreshold_Guards:
    """Guard conditions block adaptation correctly."""

    def test_respects_cooldown(self):
        avt, _ = _make_avt(step=0.02, cooldown=20, max_adaptations=5, floor=0.40)
        avt.tick(CLASS_NEAR_THRESHOLD, 0.55)  # fires, sets cooldown=20
        result = avt.tick(CLASS_NEAR_THRESHOLD, 0.53)
        assert result is None

    def test_cooldown_decrements_on_each_tick(self):
        avt, _ = _make_avt(step=0.02, cooldown=5, max_adaptations=5, floor=0.40)
        avt.tick(CLASS_NEAR_THRESHOLD, 0.55)  # fires, cooldown=5
        for _ in range(5):
            avt.tick("MODERATE", 0.53)
        assert avt.get_status()["cooldown_remaining"] == 0

    def test_respects_max_adaptations(self):
        avt, _ = _make_avt(step=0.02, cooldown=0, max_adaptations=2, floor=0.40)
        avt.tick(CLASS_NEAR_THRESHOLD, 0.55)
        avt.tick(CLASS_NEAR_THRESHOLD, 0.53)
        result = avt.tick(CLASS_NEAR_THRESHOLD, 0.51)
        assert result is None
        assert avt.get_status()["adaptation_count"] == 2

    def test_respects_floor_guard(self):
        avt, _ = _make_avt(step=0.02, cooldown=0, max_adaptations=5, floor=0.40)
        result = avt.tick(CLASS_NEAR_THRESHOLD, current_threshold=0.41)
        assert result is None
        assert avt.get_status()["adaptation_count"] == 0

    def test_exactly_at_floor_boundary_allowed(self):
        avt, _ = _make_avt(step=0.02, cooldown=0, max_adaptations=5, floor=0.40)
        result = avt.tick(CLASS_NEAR_THRESHOLD, current_threshold=0.42)
        assert result == pytest.approx(0.40, abs=1e-5)

    def test_non_near_threshold_returns_none(self):
        avt, _ = _make_avt()
        for cls in ("FAR_BELOW", "MODERATE", "INSUFFICIENT_DATA", ""):
            result = avt.tick(cls, 0.55)
            assert result is None, f"Expected None for classification={cls!r}"


class TestAdaptiveVoteThreshold_Status:
    """get_status() returns all required fields."""

    def test_get_status_keys(self):
        avt, _ = _make_avt()
        status = avt.get_status()
        required = {
            "adaptation_count", "cooldown_remaining", "max_adaptations",
            "floor", "step", "total_reduction", "last_adapted_threshold",
        }
        for key in required:
            assert key in status, f"Missing status key: {key}"

    def test_get_status_initial_values(self):
        avt, _ = _make_avt(step=0.02, cooldown=20, max_adaptations=5, floor=0.40)
        status = avt.get_status()
        assert status["adaptation_count"] == 0
        assert status["cooldown_remaining"] == 0
        assert status["total_reduction"] == 0.0
        assert status["last_adapted_threshold"] is None


class TestAdaptiveVoteThreshold_Persistence:
    """save_state() / _load_state() round-trip."""

    def test_save_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        path.unlink()

        avt1 = AdaptiveVoteThreshold(step=0.02, cooldown=0, max_adaptations=5, floor=0.40, state_path=path)
        avt1.tick(CLASS_NEAR_THRESHOLD, 0.55)
        avt1.tick(CLASS_NEAR_THRESHOLD, 0.53)
        avt1.save_state()

        avt2 = AdaptiveVoteThreshold(state_path=path)
        status = avt2.get_status()
        assert status["adaptation_count"] == 2
        assert status["total_reduction"] == pytest.approx(0.04, abs=1e-5)
        path.unlink(missing_ok=True)

    def test_save_state_atomic_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        path.unlink()

        avt = AdaptiveVoteThreshold(state_path=path)
        avt.tick(CLASS_NEAR_THRESHOLD, 0.55)
        avt.save_state()

        with open(path) as fh:
            state = json.load(fh)
        assert "adaptation_count" in state
        assert "cooldown_remaining" in state
        path.unlink(missing_ok=True)


# ── Integration Tests — Wiring Greps ────────────────────────────────────────


class TestPhase66WiringGreps:
    """Per Live Wiring Verification Gates: grep engine.py for call sites."""

    def _engine(self) -> str:
        return ENGINE_SRC.read_text()

    def _continuous(self) -> str:
        return CONTINUOUS_SRC.read_text()

    def _synthesizer(self) -> str:
        return SYNTHESIZER_SRC.read_text()

    def test_vote_recorder_init_block(self):
        assert "_vote_recorder = None" in self._engine()

    def test_vote_recorder_import_in_engine(self):
        assert "VoteScoreRecorder" in self._engine()

    def test_vote_recorder_record_call(self):
        assert "_vote_recorder.record(" in self._engine()

    def test_vote_recorder_save_state_call(self):
        assert "_vote_recorder.save_state()" in self._engine()

    def test_adaptive_vote_threshold_init_block(self):
        assert "_adaptive_vote_threshold = None" in self._engine()

    def test_adaptive_vote_threshold_import_in_engine(self):
        assert "AdaptiveVoteThreshold" in self._engine()

    def test_adaptive_vote_threshold_tick_call(self):
        assert "_adaptive_vote_threshold.tick(" in self._engine()

    def test_adaptive_vote_threshold_save_state_call(self):
        assert "_adaptive_vote_threshold.save_state()" in self._engine()

    def test_config_mutation_log_present(self):
        assert "Phase 66 (US-390): AdaptiveVoteThreshold" in self._engine()

    def test_vote_gap_analyzer_import_in_engine(self):
        assert "vote_gap_analyzer" in self._engine()

    def test_vote_recorder_wired_in_continuous(self):
        assert "vote_recorder" in self._continuous()

    def test_vote_threshold_wired_in_continuous(self):
        assert "weighted_vote_threshold" in self._continuous()

    def test_vote_bottleneck_in_synthesizer(self):
        assert "BN_VOTE_NEAR_MISS" in self._synthesizer()

    def test_vote_kill_bottleneck_in_synthesizer(self):
        assert "BN_VOTE_KILL" in self._synthesizer()

    def test_vote_signals_in_synthesizer(self):
        assert "vote_classification" in self._synthesizer()


# ── Behavioral Simulations ──────────────────────────────────────────────────


class TestPhase66BehavioralSimulations:
    """End-to-end behavioral scenarios per PRD acceptance criteria."""

    def test_full_adaptation_lifecycle(self):
        """NEAR_THRESHOLD → adapt → cooldown → adapt again after cooldown."""
        avt, _ = _make_avt(step=0.02, cooldown=3, max_adaptations=5, floor=0.40)
        r1 = avt.tick(CLASS_NEAR_THRESHOLD, 0.55)
        assert r1 == pytest.approx(0.53, abs=1e-4)
        for _ in range(2):
            assert avt.tick(CLASS_NEAR_THRESHOLD, 0.53) is None
        r2 = avt.tick(CLASS_NEAR_THRESHOLD, 0.53)
        assert r2 == pytest.approx(0.51, abs=1e-4)

    def test_max_adaptations_exhausted_scenario(self):
        avt, _ = _make_avt(step=0.02, cooldown=0, max_adaptations=5, floor=0.40)
        threshold = 0.55
        for _ in range(5):
            result = avt.tick(CLASS_NEAR_THRESHOLD, threshold)
            threshold = result
        # threshold should be 0.45 after 5 reductions of 0.02
        assert threshold == pytest.approx(0.45, abs=1e-4)
        result = avt.tick(CLASS_NEAR_THRESHOLD, threshold)
        assert result is None

    def test_floor_guard_scenario(self):
        avt, _ = _make_avt(step=0.02, cooldown=0, max_adaptations=5, floor=0.40)
        result = avt.tick(CLASS_NEAR_THRESHOLD, 0.41)
        assert result is None

    def test_far_below_never_adapts(self):
        avt, _ = _make_avt(step=0.02, cooldown=0, max_adaptations=5, floor=0.40)
        for _ in range(10):
            assert avt.tick("FAR_BELOW", 0.55) is None

    def test_importability(self):
        from src.scanner.adaptive_vote_threshold import AdaptiveVoteThreshold
        assert AdaptiveVoteThreshold is not None

    def test_phase66_modules_all_importable(self):
        from src.scanner.vote_score_recorder import VoteScoreRecorder
        from src.scanner.vote_gap_analyzer import VoteGapAnalyzer
        from src.scanner.adaptive_vote_threshold import AdaptiveVoteThreshold
        assert VoteScoreRecorder is not None
        assert VoteGapAnalyzer is not None
        assert AdaptiveVoteThreshold is not None
