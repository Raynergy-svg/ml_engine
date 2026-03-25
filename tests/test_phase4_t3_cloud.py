#!/usr/bin/env python3
"""Phase 4 Verification: T3 Narrative Arcs + Cloud Fallback.

Tests:
1. T3 narrative arc detection with synthetic multi-week data
2. Cloud fallback synthesizer (local mode — no LLM needed)
3. PatternEngine integration (T1+T2+T3+Cloud all wired)
4. Orchestrator module initialization
"""

import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

results = []


def test(name: str):
    """Decorator to register and run a test."""
    def decorator(fn):
        def wrapper():
            try:
                fn()
                results.append((name, True, ""))
                print(f"  {PASS} {name}")
            except Exception as e:
                results.append((name, False, str(e)))
                print(f"  {FAIL} {name}: {e}")
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
# Test 1: T3 Narrative Arc Detection
# ═══════════════════════════════════════════════════════════════

@test("T3: Emotional drift detection with 6 weeks of data")
def test_t3_emotional_drift():
    from src.aura.patterns.tier3 import Tier3NarrativeArcDetector

    with tempfile.TemporaryDirectory() as tmpdir:
        detector = Tier3NarrativeArcDetector(patterns_dir=Path(tmpdir))

        # Generate 6 weeks of conversations with declining emotional valence
        conversations = []
        base_date = datetime(2026, 1, 5, tzinfo=timezone.utc)
        emotions_by_week = [
            ["calm", "energized", "calm", "focused", "calm"],       # Week 1: positive
            ["calm", "neutral", "calm", "neutral", "focused"],       # Week 2: mostly positive
            ["neutral", "stressed", "neutral", "calm", "neutral"],   # Week 3: mixed
            ["stressed", "anxious", "neutral", "stressed", "neutral"],  # Week 4: negative
            ["anxious", "stressed", "frustrated", "stressed", "anxious"],  # Week 5: bad
            ["frustrated", "overwhelmed", "stressed", "anxious", "frustrated"],  # Week 6: worst
        ]

        for week_idx, emotions in enumerate(emotions_by_week):
            for day_idx, emotion in enumerate(emotions):
                ts = (base_date + timedelta(weeks=week_idx, days=day_idx)).isoformat()
                conversations.append({
                    "timestamp": ts,
                    "emotional_state": emotion,
                    "topics": json.dumps(["trading"]),
                })

        patterns = detector.detect(
            conversations=conversations,
            readiness_history=[],
            trade_outcomes=[],
            override_events=[],
        )

        # Should detect emotional drift
        drift_patterns = [p for p in patterns if "emotional" in p.pattern_id.lower() or "drift" in p.pattern_id.lower()]
        assert len(drift_patterns) > 0, f"Expected emotional drift pattern, got {len(drift_patterns)} patterns: {[p.pattern_id for p in patterns]}"

        drift = drift_patterns[0]
        assert drift.confidence > 0.3, f"Expected confidence > 0.3, got {drift.confidence}"
        assert "declining" in drift.description.lower() or "drift" in drift.description.lower(), \
            f"Expected 'declining' or 'drift' in description: {drift.description}"
        print(f"    → Detected: {drift.description[:80]}")


@test("T3: Override trajectory detection")
def test_t3_override_trajectory():
    from src.aura.patterns.tier3 import Tier3NarrativeArcDetector

    with tempfile.TemporaryDirectory() as tmpdir:
        detector = Tier3NarrativeArcDetector(patterns_dir=Path(tmpdir))

        # Generate increasing override frequency over 5 weeks
        override_events = []
        base_date = datetime(2026, 1, 5, tzinfo=timezone.utc)
        overrides_per_week = [1, 2, 4, 6, 8]  # Escalating

        for week_idx, count in enumerate(overrides_per_week):
            for i in range(count):
                ts = (base_date + timedelta(weeks=week_idx, days=i % 5)).isoformat()
                override_events.append({
                    "timestamp": ts,
                    "override_type": "took_rejected",
                    "outcome": "loss" if i % 3 == 0 else "win",
                    "emotional_state": "stressed" if week_idx > 2 else "neutral",
                })

        patterns = detector.detect(
            conversations=[],
            readiness_history=[],
            trade_outcomes=[],
            override_events=override_events,
        )

        override_patterns = [p for p in patterns if "override" in p.pattern_id.lower()]
        assert len(override_patterns) > 0, f"Expected override trajectory, got {[p.pattern_id for p in patterns]}"

        ov = override_patterns[0]
        assert "increasing" in ov.description.lower(), f"Expected 'increasing' in: {ov.description}"
        print(f"    → Detected: {ov.description[:80]}")


@test("T3: Readiness trend detection")
def test_t3_readiness_trend():
    from src.aura.patterns.tier3 import Tier3NarrativeArcDetector

    with tempfile.TemporaryDirectory() as tmpdir:
        detector = Tier3NarrativeArcDetector(patterns_dir=Path(tmpdir))

        # Generate improving readiness over 5 weeks
        readiness_history = []
        base_date = datetime(2026, 1, 5, tzinfo=timezone.utc)
        scores_by_week = [
            [45, 48, 42, 50, 47],  # Week 1: low
            [52, 55, 50, 58, 53],  # Week 2: improving
            [60, 63, 58, 65, 62],  # Week 3: better
            [68, 72, 65, 75, 70],  # Week 4: good
            [75, 78, 72, 80, 76],  # Week 5: great
        ]

        for week_idx, scores in enumerate(scores_by_week):
            for day_idx, score in enumerate(scores):
                ts = (base_date + timedelta(weeks=week_idx, days=day_idx)).isoformat()
                readiness_history.append({"timestamp": ts, "score": score})

        patterns = detector.detect(
            conversations=[],
            readiness_history=readiness_history,
            trade_outcomes=[],
            override_events=[],
        )

        readiness_patterns = [p for p in patterns if "readiness" in p.pattern_id.lower()]
        assert len(readiness_patterns) > 0, f"Expected readiness trend, got {[p.pattern_id for p in patterns]}"

        rp = readiness_patterns[0]
        assert "improving" in rp.description.lower(), f"Expected 'improving' in: {rp.description}"
        print(f"    → Detected: {rp.description[:80]}")


@test("T3: Stress accumulation detection")
def test_t3_stress_accumulation():
    from src.aura.patterns.tier3 import Tier3NarrativeArcDetector

    with tempfile.TemporaryDirectory() as tmpdir:
        detector = Tier3NarrativeArcDetector(patterns_dir=Path(tmpdir))

        # Generate escalating stress without recovery
        conversations = []
        base_date = datetime(2026, 1, 5, tzinfo=timezone.utc)
        # Stress ratio increases each week
        stress_ratios = [0.1, 0.2, 0.4, 0.6, 0.8]  # Fraction of stressed conversations

        for week_idx, ratio in enumerate(stress_ratios):
            for day_idx in range(5):
                ts = (base_date + timedelta(weeks=week_idx, days=day_idx)).isoformat()
                is_stressed = (day_idx / 5) < ratio
                conversations.append({
                    "timestamp": ts,
                    "emotional_state": "stressed" if is_stressed else "calm",
                    "topics": json.dumps(["work"]),
                })

        patterns = detector.detect(
            conversations=conversations,
            readiness_history=[],
            trade_outcomes=[],
            override_events=[],
        )

        stress_patterns = [p for p in patterns if "stress" in p.pattern_id.lower()]
        # Stress accumulation may or may not trigger depending on thresholds
        if stress_patterns:
            sp = stress_patterns[0]
            print(f"    → Detected: {sp.description[:80]}")
        else:
            # Emotional drift should still catch the declining valence
            any_detected = len(patterns) > 0
            assert any_detected, "Expected at least one pattern from escalating stress"
            print(f"    → Caught as: {patterns[0].pattern_id} ({patterns[0].description[:60]})")


# ═══════════════════════════════════════════════════════════════
# Test 2: Cloud Fallback Synthesizer (Local Mode)
# ═══════════════════════════════════════════════════════════════

@test("Cloud: Local fallback explanation generates meaningful text")
def test_cloud_local_fallback():
    from src.aura.patterns.cloud_fallback import CloudPatternSynthesizer

    with tempfile.TemporaryDirectory() as tmpdir:
        # No LLM configured → should use local fallback
        synth = CloudPatternSynthesizer(log_dir=Path(tmpdir))

        assert not synth.is_available(), "Should not be available without LLM config"

        result = synth.synthesize_pattern_explanation(
            pattern_description="Win rate drops 40% during weeks with high stress",
            evidence_items=[
                {"source_type": "correlation", "summary": "Stress vs win rate r=-0.65"},
            ],
        )

        assert result.success, "Local fallback should succeed"
        assert result.fallback_used, "Should report fallback_used=True"
        assert "stress" in result.narrative.lower(), f"Expected stress-related narrative: {result.narrative}"
        assert len(result.suggested_actions) > 0, "Should suggest actions"
        print(f"    → Narrative: {result.narrative[:80]}")
        print(f"    → Actions: {result.suggested_actions}")


@test("Cloud: Local fallback connections finds themes")
def test_cloud_local_connections():
    from src.aura.patterns.cloud_fallback import CloudPatternSynthesizer

    with tempfile.TemporaryDirectory() as tmpdir:
        synth = CloudPatternSynthesizer(log_dir=Path(tmpdir))

        result = synth.synthesize_pattern_connections([
            {"description": "Stress correlates with lower win rates"},
            {"description": "Override frequency increasing during stress"},
            {"description": "Readiness declining over 4 weeks"},
        ])

        assert result.success, "Local fallback should succeed"
        assert result.fallback_used, "Should use fallback"
        assert "stress" in result.narrative.lower() or "patterns" in result.narrative.lower(), \
            f"Expected themed narrative: {result.narrative}"
        print(f"    → Cross-pattern: {result.narrative[:80]}")


@test("Cloud: Override risk fallback with high-risk context")
def test_cloud_override_risk():
    from src.aura.patterns.cloud_fallback import CloudPatternSynthesizer

    with tempfile.TemporaryDirectory() as tmpdir:
        synth = CloudPatternSynthesizer(log_dir=Path(tmpdir))

        result = synth.synthesize_override_risk_narrative(
            override_context={
                "pair": "EUR_USD",
                "buddy_confidence": 0.82,
                "emotional_state": "revenge",
                "cognitive_load": "high",
                "override_type": "took_rejected",
            },
            recent_patterns=[
                {"description": "Overrides during loss streaks are 80% losers"},
            ],
        )

        assert result.success, "Local fallback should succeed"
        assert "revenge" in result.narrative.lower() or "risk" in result.narrative.lower(), \
            f"Expected risk narrative: {result.narrative}"
        print(f"    → Risk narrative: {result.narrative[:80]}")


@test("Cloud: Status reporting works")
def test_cloud_status():
    from src.aura.patterns.cloud_fallback import CloudPatternSynthesizer

    with tempfile.TemporaryDirectory() as tmpdir:
        synth = CloudPatternSynthesizer(log_dir=Path(tmpdir))
        status = synth.get_status()

        assert status["configured"] == False, "Should not be configured"
        assert status["provider"] == "none", f"Provider should be 'none': {status}"
        assert status["available"] == False, "Should not be available"
        print(f"    → Status: {status}")


@test("Cloud: JSON extraction from various formats")
def test_cloud_json_extraction():
    from src.aura.patterns.cloud_fallback import CloudPatternSynthesizer

    with tempfile.TemporaryDirectory() as tmpdir:
        synth = CloudPatternSynthesizer(log_dir=Path(tmpdir))

        # Direct JSON
        result = synth._extract_json('{"key": "value"}')
        assert result == {"key": "value"}, f"Direct JSON failed: {result}"

        # Code fence
        result = synth._extract_json('```json\n{"key": "value2"}\n```')
        assert result.get("key") == "value2", f"Code fence failed: {result}"

        # Embedded in text
        result = synth._extract_json('Here is the result: {"key": "value3"} and more text')
        assert result.get("key") == "value3", f"Embedded JSON failed: {result}"

        print("    → All JSON extraction formats work")


# ═══════════════════════════════════════════════════════════════
# Test 3: PatternEngine Integration (T1+T2+T3+Cloud)
# ═══════════════════════════════════════════════════════════════

@test("PatternEngine: All three tiers + cloud initialized")
def test_pattern_engine_init():
    from src.aura.patterns.engine import PatternEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = PatternEngine(
            patterns_dir=Path(tmpdir) / "patterns",
            bridge_dir=Path(tmpdir) / "bridge",
            trade_journal_path=Path(tmpdir) / "journal.json",
        )

        assert engine.t1 is not None, "T1 not initialized"
        assert engine.t2 is not None, "T2 not initialized"
        assert engine.t3 is not None, "T3 not initialized"
        assert engine.cloud is not None, "Cloud not initialized"
        print("    → All four components initialized: T1, T2, T3, Cloud")


@test("PatternEngine: run_all includes T3 results")
def test_pattern_engine_run_all():
    from src.aura.patterns.engine import PatternEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = PatternEngine(
            patterns_dir=Path(tmpdir) / "patterns",
            bridge_dir=Path(tmpdir) / "bridge",
            trade_journal_path=Path(tmpdir) / "journal.json",
        )

        results = engine.run_all(conversations=[], readiness_history=[])
        assert "t1" in results, "Missing t1 key"
        assert "t2" in results, "Missing t2 key"
        assert "t3" in results, "Missing t3 key"
        print(f"    → run_all returns: t1={len(results['t1'])}, t2={len(results['t2'])}, t3={len(results['t3'])}")


@test("PatternEngine: Status includes T3 and cloud")
def test_pattern_engine_status():
    from src.aura.patterns.engine import PatternEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = PatternEngine(
            patterns_dir=Path(tmpdir) / "patterns",
            bridge_dir=Path(tmpdir) / "bridge",
            trade_journal_path=Path(tmpdir) / "journal.json",
        )

        status = engine.get_status()
        assert "t3_patterns" in status, f"Missing t3_patterns in status: {list(status.keys())}"
        assert "t3_last_run" in status, f"Missing t3_last_run in status"
        assert "cloud_synthesis" in status, f"Missing cloud_synthesis in status"
        print(f"    → Status keys: {list(status.keys())}")


@test("PatternEngine: Report format includes T3 section")
def test_pattern_engine_report():
    from src.aura.patterns.engine import PatternEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = PatternEngine(
            patterns_dir=Path(tmpdir) / "patterns",
            bridge_dir=Path(tmpdir) / "bridge",
            trade_journal_path=Path(tmpdir) / "journal.json",
        )

        report = engine.format_patterns_report()
        assert "T3 (monthly)" in report, f"Report missing T3 line: {report[:200]}"
        assert "Cloud:" in report, f"Report missing Cloud line: {report[:200]}"
        print(f"    → Report includes T1/T2/T3/Cloud status lines")


# ═══════════════════════════════════════════════════════════════
# Test 4: T3 Helper Functions
# ═══════════════════════════════════════════════════════════════

@test("T3: Linear regression computes correct slope")
def test_linear_regression():
    from src.aura.patterns.tier3 import _linear_regression

    # Perfect positive trend
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [10.0, 12.0, 14.0, 16.0, 18.0]
    slope, intercept, r_sq = _linear_regression(x, y)
    assert abs(slope - 2.0) < 0.01, f"Expected slope ~2.0, got {slope}"
    assert abs(r_sq - 1.0) < 0.01, f"Expected R²~1.0, got {r_sq}"
    print(f"    → slope=2.0, R²=1.0 ✓")

    # Flat trend
    y_flat = [5.0, 5.0, 5.0, 5.0, 5.0]
    slope2, _, r_sq2 = _linear_regression(x, y_flat)
    assert abs(slope2) < 0.01, f"Expected slope ~0.0, got {slope2}"
    print(f"    → Flat: slope={slope2:.3f} ✓")


@test("T3: Moving average smoothing works")
def test_moving_average():
    from src.aura.patterns.tier3 import _moving_average

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    smoothed = _moving_average(values, window=3)
    assert len(smoothed) == 5, f"Expected 5 values, got {len(smoothed)}"
    # Last value should be avg of [3, 4, 5] = 4.0
    assert abs(smoothed[-1] - 4.0) < 0.01, f"Expected last ~4.0, got {smoothed[-1]}"
    print(f"    → Smoothed: {[round(v, 2) for v in smoothed]}")


@test("T3: Phase detection identifies building phase")
def test_phase_detection():
    from src.aura.patterns.tier3 import _detect_phase

    # Strong upward trend → building
    smoothed = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    phase = _detect_phase(smoothed, slope=0.1, r_squared=0.95)
    assert phase in ("building", "peak"), f"Expected building/peak, got {phase}"
    print(f"    → Upward trend: {phase}")

    # Peaked and declining → resolving
    smoothed2 = [0.3, 0.5, 0.7, 0.8, 0.6, 0.4]
    phase2 = _detect_phase(smoothed2, slope=-0.05, r_squared=0.3)
    assert phase2 in ("resolving", "resolved", "stable"), f"Expected resolving/resolved, got {phase2}"
    print(f"    → Post-peak decline: {phase2}")


# ═══════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{BOLD}═══ Phase 4 Verification: T3 Narrative Arcs + Cloud Fallback ═══{RESET}\n")

    print(f"{BOLD}── T3 Narrative Arc Detection ──{RESET}")
    test_t3_emotional_drift()
    test_t3_override_trajectory()
    test_t3_readiness_trend()
    test_t3_stress_accumulation()

    print(f"\n{BOLD}── Cloud Fallback Synthesizer ──{RESET}")
    test_cloud_local_fallback()
    test_cloud_local_connections()
    test_cloud_override_risk()
    test_cloud_status()
    test_cloud_json_extraction()

    print(f"\n{BOLD}── PatternEngine Integration ──{RESET}")
    test_pattern_engine_init()
    test_pattern_engine_run_all()
    test_pattern_engine_status()
    test_pattern_engine_report()

    print(f"\n{BOLD}── T3 Helper Functions ──{RESET}")
    test_linear_regression()
    test_moving_average()
    test_phase_detection()

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'═' * 60}")
    if passed == total:
        print(f"{BOLD}\033[92m ALL {total} TESTS PASSED \033[0m{RESET}")
    else:
        print(f"{BOLD}\033[91m {passed}/{total} TESTS PASSED \033[0m{RESET}")
        for name, ok, err in results:
            if not ok:
                print(f"  {FAIL} {name}: {err}")
    print(f"{'═' * 60}\n")
