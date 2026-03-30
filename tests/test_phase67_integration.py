"""Phase 66/67 Integration Test — Full adaptive loop simulation.

Bypasses TF model inference (segfaults on Apple Silicon) and instead simulates
the scan data flow to verify:
1. VoteScoreRecorder + SubInferenceRecorder capture data correctly
2. Gap analyzers classify NEAR_THRESHOLD when data warrants it
3. Adaptive thresholds fire and mutate config values
4. ScanHealthSynthesizer reports vote bottlenecks
5. Full lifecycle: 50-scan cycle triggers analysis → adaptation

This is the Phase 68 verification that the Phase 66/67 adaptive mechanisms
work end-to-end without requiring a live TF-based scan.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scanner.vote_score_recorder import VoteScoreRecorder
from src.scanner.vote_gap_analyzer import analyze as vote_analyze
from src.scanner.adaptive_vote_threshold import AdaptiveVoteThreshold

from src.scanner.sub_inference_recorder import SubInferenceRecorder
from src.scanner.sub_inference_gap_analyzer import analyze as sub_analyze
from src.scanner.adaptive_sub_inference_threshold import AdaptiveSubInferenceThreshold

from src.scanner.scan_health_synthesizer import (
    ScanHealthSynthesizer,
    BN_VOTE_NEAR_MISS,
    BN_VOTE_KILL,
)


def _tmppath(suffix):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        p = Path(f.name)
    p.unlink()
    return p


# ── End-to-End: weighted_vote_threshold adaptive loop ───────────────────────


class TestPhase66EndToEnd:
    """Simulate 50 scans → gap analysis → adaptive vote threshold fires."""

    def test_adaptive_vote_threshold_fires_on_near_threshold_data(self):
        """When vote scores cluster just below threshold, adaptation fires."""
        rec = VoteScoreRecorder(state_path=_tmppath(".jsonl"))
        avt = AdaptiveVoteThreshold(
            step=0.02, cooldown=0, max_adaptations=5, floor=0.40,
            state_path=_tmppath(".json"),
        )

        # Simulate 50 scans: vote scores cluster at 0.51-0.54 (just below 0.55)
        for i in range(50):
            for pair in ["EUR_USD", "GBP_USD", "USD_JPY"]:
                score = 0.51 + (i % 4) * 0.01  # 0.51, 0.52, 0.53, 0.54
                rec.record(pair=pair, vote_score=score, agent_passed=False)

        # Run gap analysis
        result = vote_analyze(rec, weighted_vote_threshold=0.55)
        assert result["classification"] == "NEAR_THRESHOLD"
        assert result["fail_count"] >= 3

        # Adaptive mechanism should fire
        new_thresh = avt.tick(result["classification"], 0.55)
        assert new_thresh is not None
        assert new_thresh == pytest.approx(0.53, abs=1e-4)
        assert avt.get_status()["adaptation_count"] == 1

    def test_multiple_adaptations_lower_threshold(self):
        """Multiple NEAR_THRESHOLD cycles progressively lower threshold."""
        rec = VoteScoreRecorder(state_path=_tmppath(".jsonl"))
        avt = AdaptiveVoteThreshold(
            step=0.02, cooldown=0, max_adaptations=5, floor=0.40,
            state_path=_tmppath(".json"),
        )

        threshold = 0.55
        for cycle in range(3):
            # Populate fails just below current threshold
            for i in range(10):
                rec.record(
                    pair="EUR_USD",
                    vote_score=threshold - 0.03,
                    agent_passed=False,
                )
            result = vote_analyze(rec, weighted_vote_threshold=threshold)
            new_thresh = avt.tick(result["classification"], threshold)
            if new_thresh is not None:
                threshold = new_thresh

        assert threshold == pytest.approx(0.49, abs=1e-4)  # 3 reductions of 0.02
        assert avt.get_status()["adaptation_count"] == 3


# ── End-to-End: sub_inference_vote_threshold adaptive loop ───────────────────


class TestPhase67EndToEnd:
    """Simulate 50 scans → gap analysis → adaptive sub-inference fires."""

    def test_adaptive_sub_inference_fires_on_near_threshold(self):
        """When consensus ratios cluster just below threshold, adaptation fires."""
        rec = SubInferenceRecorder(state_path=_tmppath(".jsonl"))
        asit = AdaptiveSubInferenceThreshold(
            step=0.05, cooldown=0, max_adaptations=3, floor=0.40,
            state_path=_tmppath(".json"),
        )

        # Simulate: most pairs get 2/3 votes (ratio=0.667) but threshold is 0.70
        for i in range(50):
            for pair in ["EUR_USD", "GBP_USD"]:
                rec.record(pair=pair, votes=2, total=3, agent_passed=False)

        result = sub_analyze(rec, sub_inference_vote_threshold=0.70)
        assert result["classification"] == "NEAR_THRESHOLD"
        # gap = 0.70 - 0.667 = 0.033 ≤ 0.10

        new_thresh = asit.tick(result["classification"], 0.70)
        assert new_thresh is not None
        assert new_thresh == pytest.approx(0.65, abs=1e-4)

    def test_sub_inference_far_below_no_adaptation(self):
        """When consensus ratios are 0/3, FAR_BELOW — no adaptation."""
        rec = SubInferenceRecorder(state_path=_tmppath(".jsonl"))
        asit = AdaptiveSubInferenceThreshold(
            step=0.05, cooldown=0, max_adaptations=3, floor=0.40,
            state_path=_tmppath(".json"),
        )

        for i in range(20):
            rec.record(pair="EUR_USD", votes=0, total=3, agent_passed=False)

        result = sub_analyze(rec, sub_inference_vote_threshold=0.66)
        assert result["classification"] == "FAR_BELOW"

        new_thresh = asit.tick(result["classification"], 0.66)
        assert new_thresh is None  # FAR_BELOW doesn't trigger adaptation


# ── End-to-End: ScanHealthSynthesizer with vote signals ──────────────────────


class TestPhase66SynthesizerIntegration:
    """ScanHealthSynthesizer produces vote bottlenecks when wired."""

    def test_vote_near_miss_bottleneck_appears(self):
        """When vote recorder shows NEAR_THRESHOLD, synthesizer reports it."""
        rec = VoteScoreRecorder(state_path=_tmppath(".jsonl"))
        for _ in range(10):
            rec.record(pair="EUR_USD", vote_score=0.52, agent_passed=False)

        # Create synthesizer with vote recorder
        synth = ScanHealthSynthesizer(
            vote_recorder=rec,
            weighted_vote_threshold=0.55,
        )

        # Mock signal_funnel to report agent_consensus as dominant kill
        mock_funnel = MagicMock()
        mock_funnel.get_funnel_summary.return_value = {
            "dominant_kill_stage": "agent_consensus",
            "tradeable_pct": 0.0,
        }
        synth._signal_funnel = mock_funnel

        report = synth.synthesize()
        bottleneck_names = [b["name"] for b in report.get("bottlenecks", [])]
        assert BN_VOTE_NEAR_MISS in bottleneck_names or BN_VOTE_KILL in bottleneck_names

    def test_no_vote_bottleneck_without_recorder(self):
        """Backward compat: no vote_recorder → no vote bottlenecks."""
        synth = ScanHealthSynthesizer()
        report = synth.synthesize()
        bottleneck_names = [b["name"] for b in report.get("bottlenecks", [])]
        assert BN_VOTE_NEAR_MISS not in bottleneck_names
        assert BN_VOTE_KILL not in bottleneck_names


# ── End-to-End: Combined adaptive loop simulation ───────────────────────────


class TestCombinedAdaptiveLoop:
    """Simulate the actual engine scan loop pattern for both Phase 66+67."""

    def test_50_scan_periodic_block_simulation(self):
        """Simulate the engine.py periodic block at scan_cycle_count % 50."""
        # Phase 66 modules
        vote_rec = VoteScoreRecorder(state_path=_tmppath(".jsonl"))
        avt = AdaptiveVoteThreshold(
            step=0.02, cooldown=0, max_adaptations=5, floor=0.40,
            state_path=_tmppath(".json"),
        )
        # Phase 67 modules
        si_rec = SubInferenceRecorder(state_path=_tmppath(".jsonl"))
        asit = AdaptiveSubInferenceThreshold(
            step=0.05, cooldown=0, max_adaptations=3, floor=0.30,
            state_path=_tmppath(".json"),
        )

        weighted_vote_threshold = 0.55
        sub_inference_vote_threshold = 0.66

        # Simulate 50 scans
        for scan in range(50):
            for pair in ["EUR_USD", "GBP_USD", "USD_JPY"]:
                # Vote scores just below threshold
                vote_rec.record(pair=pair, vote_score=0.52, agent_passed=False)
                # Sub-inference: 2/3 votes (ratio=0.667, just below 0.70)
                si_rec.record(pair=pair, votes=2, total=3, agent_passed=False)

        # Periodic block (every 50 scans)
        # Phase 66
        vga_result = vote_analyze(vote_rec, weighted_vote_threshold)
        vga_cls = vga_result["classification"]
        new_vt = avt.tick(vga_cls, weighted_vote_threshold)
        if new_vt is not None:
            weighted_vote_threshold = new_vt

        # Phase 67
        siga_result = sub_analyze(si_rec, sub_inference_vote_threshold)
        siga_cls = siga_result["classification"]
        new_sit = asit.tick(siga_cls, sub_inference_vote_threshold)
        if new_sit is not None:
            sub_inference_vote_threshold = new_sit

        # Verify both adapted
        assert weighted_vote_threshold == pytest.approx(0.53, abs=1e-4)
        assert sub_inference_vote_threshold == pytest.approx(0.3323, abs=1e-4)

        # Verify persistence works
        vote_rec.save_state()
        si_rec.save_state()
        avt.save_state()
        asit.save_state()

        assert avt.get_status()["adaptation_count"] == 1
        assert asit.get_status()["adaptation_count"] == 1
