"""Phase 67 US-392/393 — SubInferenceGapAnalyzer + AdaptiveSubInferenceThreshold tests.

Tests cover:
- Gap analyzer classification (NEAR_THRESHOLD, MODERATE, FAR_BELOW, INSUFFICIENT_DATA)
- Near-miss counting
- Adaptive threshold trigger, cooldown, max_adaptations, floor guard
- Save/load round-trip
- Wiring greps in engine.py
- Full adaptation lifecycle simulation
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.scanner.sub_inference_recorder import SubInferenceRecorder
from src.scanner.sub_inference_gap_analyzer import (
    analyze,
    CLASS_NEAR_THRESHOLD,
    CLASS_MODERATE,
    CLASS_FAR_BELOW,
    CLASS_INSUFFICIENT_DATA,
)
from src.scanner.adaptive_sub_inference_threshold import (
    AdaptiveSubInferenceThreshold,
    CLASS_NEAR_THRESHOLD as AVT_NEAR,
)

ENGINE_SRC = Path("src/scanner/engine.py")


def _make_recorder(**kwargs) -> SubInferenceRecorder:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    path.unlink()
    return SubInferenceRecorder(state_path=path, **kwargs)


def _populate_fails(rec, ratios, threshold=0.66):
    """Record fails with given consensus_ratios (votes/total approximated)."""
    for i, r in enumerate(ratios):
        # Approximate votes/total from ratio
        total = 3
        votes = round(r * total)
        rec.record(pair=f"PAIR_{i}", votes=votes, total=total, agent_passed=False)


def _make_asit(**kwargs):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    path.unlink()
    return AdaptiveSubInferenceThreshold(state_path=path, **kwargs), path


# ── Gap Analyzer Tests ─────────────────────────────────────────────────────


class TestSubInferenceGapAnalyzer_Classification:
    def test_near_threshold(self):
        rec = _make_recorder()
        # threshold=0.66, fails at ratio ~0.60 → gap=0.06 ≤ 0.10
        for _ in range(5):
            rec.record(pair="EUR_USD", votes=2, total=3, agent_passed=False)  # ratio=0.667 but "failed"
        # Actually need ratios BELOW threshold. votes=1/3=0.33
        rec2 = _make_recorder()
        for _ in range(5):
            rec2.record(pair="EUR_USD", votes=2, total=3, agent_passed=False)  # 0.667 - but labeled as fail
        result = analyze(rec2, sub_inference_vote_threshold=0.70)
        # p50 of fails = 0.667, gap = 0.70 - 0.667 = 0.033 → NEAR_THRESHOLD
        assert result["classification"] == CLASS_NEAR_THRESHOLD

    def test_moderate(self):
        rec = _make_recorder()
        for _ in range(5):
            rec.record(pair="EUR_USD", votes=1, total=3, agent_passed=False)  # ratio=0.333
        result = analyze(rec, sub_inference_vote_threshold=0.50)
        # gap = 0.50 - 0.333 = 0.167 → MODERATE
        assert result["classification"] == CLASS_MODERATE

    def test_far_below(self):
        rec = _make_recorder()
        for _ in range(5):
            rec.record(pair="EUR_USD", votes=0, total=3, agent_passed=False)  # ratio=0.0
        result = analyze(rec, sub_inference_vote_threshold=0.66)
        # gap = 0.66 - 0.0 = 0.66 → FAR_BELOW
        assert result["classification"] == CLASS_FAR_BELOW

    def test_insufficient_data(self):
        rec = _make_recorder()
        rec.record(pair="EUR_USD", votes=1, total=3, agent_passed=False)
        result = analyze(rec, sub_inference_vote_threshold=0.66)
        assert result["classification"] == CLASS_INSUFFICIENT_DATA

    def test_empty_recorder(self):
        rec = _make_recorder()
        result = analyze(rec, sub_inference_vote_threshold=0.66)
        assert result["classification"] == CLASS_INSUFFICIENT_DATA


class TestSubInferenceGapAnalyzer_NearMiss:
    def test_near_miss_count(self):
        rec = _make_recorder()
        # threshold=0.70, near-miss gap=0.10
        rec.record(pair="P1", votes=2, total=3, agent_passed=False)  # ratio=0.667, gap=0.033 (near)
        rec.record(pair="P2", votes=2, total=3, agent_passed=False)  # ratio=0.667 (near)
        rec.record(pair="P3", votes=0, total=3, agent_passed=False)  # ratio=0.0, gap=0.70 (far)
        result = analyze(rec, sub_inference_vote_threshold=0.70)
        assert result["near_miss_count"] == 2


# ── Adaptive Threshold Tests ──────────────────────────────────────────────


class TestAdaptiveSubInference_Trigger:
    def test_triggers_on_near_threshold(self):
        asit, _ = _make_asit(step=0.05, cooldown=30, max_adaptations=3, floor=0.30)
        result = asit.tick(AVT_NEAR, current_threshold=0.66)
        assert result == pytest.approx(0.3323, abs=1e-4)

    def test_adaptation_count_increments(self):
        asit, _ = _make_asit(floor=0.30)
        asit.tick(AVT_NEAR, 0.66)
        assert asit.get_status()["adaptation_count"] == 1

    def test_cooldown_set_after_adaptation(self):
        asit, _ = _make_asit(cooldown=30, floor=0.30)
        asit.tick(AVT_NEAR, 0.66)
        assert asit.get_status()["cooldown_remaining"] == 30


class TestAdaptiveSubInference_Guards:
    def test_respects_cooldown(self):
        asit, _ = _make_asit(step=0.05, cooldown=30, max_adaptations=3, floor=0.30)
        asit.tick(AVT_NEAR, 0.66)
        assert asit.tick(AVT_NEAR, 0.3323) is None

    def test_respects_max_adaptations(self):
        asit, _ = _make_asit(step=0.05, cooldown=0, max_adaptations=2, floor=0.40)
        r1 = asit.tick(AVT_NEAR, 0.66, total_agents=10)
        r2 = asit.tick(AVT_NEAR, r1, total_agents=10)
        assert asit.tick(AVT_NEAR, r2 if r2 else 0.50, total_agents=10) is None

    def test_respects_floor(self):
        asit, _ = _make_asit(step=0.05, cooldown=0, max_adaptations=5, floor=0.40)
        assert asit.tick(AVT_NEAR, 0.42) is None  # 0.42-0.05=0.37 < 0.40

    def test_non_near_threshold_returns_none(self):
        asit, _ = _make_asit()
        for cls in ("FAR_BELOW", "MODERATE", "INSUFFICIENT_DATA"):
            assert asit.tick(cls, 0.66) is None


class TestAdaptiveSubInference_Persistence:
    def test_save_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        path.unlink()
        asit1 = AdaptiveSubInferenceThreshold(step=0.05, cooldown=0, floor=0.30, state_path=path)
        asit1.tick(AVT_NEAR, 0.66)
        asit1.save_state()

        asit2 = AdaptiveSubInferenceThreshold(state_path=path)
        assert asit2.get_status()["adaptation_count"] == 1
        path.unlink(missing_ok=True)


# ── Wiring Greps ─────────────────────────────────────────────────────────


class TestPhase67WiringGreps:
    def _src(self) -> str:
        return ENGINE_SRC.read_text()

    def test_sub_inference_recorder_init(self):
        assert "_sub_inference_recorder = None" in self._src()

    def test_sub_inference_recorder_import(self):
        assert "SubInferenceRecorder" in self._src()

    def test_sub_inference_recorder_record_call(self):
        assert "_sub_inference_recorder.record(" in self._src()

    def test_sub_inference_recorder_save_state(self):
        assert "_sub_inference_recorder.save_state()" in self._src()

    def test_adaptive_sub_inference_init(self):
        assert "_adaptive_sub_inference_threshold = None" in self._src()

    def test_adaptive_sub_inference_import(self):
        assert "AdaptiveSubInferenceThreshold" in self._src()

    def test_adaptive_sub_inference_tick_call(self):
        assert "_adaptive_sub_inference_threshold.tick(" in self._src()

    def test_adaptive_sub_inference_save_state(self):
        assert "_adaptive_sub_inference_threshold.save_state()" in self._src()

    def test_sub_inference_gap_analyzer_import(self):
        assert "sub_inference_gap_analyzer" in self._src()

    def test_phase67_log_present(self):
        assert "Phase 67 (US-393): AdaptiveSubInferenceThreshold" in self._src()


# ── Behavioral Simulations ──────────────────────────────────────────────────


class TestPhase67BehavioralSimulations:
    def test_full_adaptation_lifecycle(self):
        asit, _ = _make_asit(step=0.05, cooldown=3, max_adaptations=3, floor=0.30)
        r1 = asit.tick(AVT_NEAR, 0.66)
        assert r1 == pytest.approx(0.3323, abs=1e-4)
        for _ in range(2):
            assert asit.tick(AVT_NEAR, 0.3323) is None
        # After cooldown expires, second adaptation from 0.3323:
        # candidate=0.2823 < floor=0.30, blocked. Returns None.
        assert asit.tick(AVT_NEAR, 0.3323) is None

    def test_max_reductions(self):
        # With 10 agents and floor=0.30, 3 adaptations are possible
        asit, _ = _make_asit(step=0.05, cooldown=0, max_adaptations=3, floor=0.30)
        t = 0.66
        for _ in range(3):
            result = asit.tick(AVT_NEAR, t, total_agents=10)
            assert result is not None
            t = result
        assert asit.tick(AVT_NEAR, t, total_agents=10) is None  # exhausted

    def test_all_modules_importable(self):
        from src.scanner.sub_inference_recorder import SubInferenceRecorder
        from src.scanner.sub_inference_gap_analyzer import analyze
        from src.scanner.adaptive_sub_inference_threshold import AdaptiveSubInferenceThreshold
        assert SubInferenceRecorder is not None
        assert analyze is not None
        assert AdaptiveSubInferenceThreshold is not None
