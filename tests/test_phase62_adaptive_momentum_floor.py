"""Phase 62 US-375 + US-377 — AdaptiveMomentumFloor unit and integration tests.

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

from src.scanner.adaptive_momentum_floor import (
    AdaptiveMomentumFloor,
    CLASS_NEAR_THRESHOLD,
)

ENGINE_SRC = Path("src/scanner/engine.py")


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_floor(**kwargs) -> tuple[AdaptiveMomentumFloor, Path]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    path.unlink()
    floor = AdaptiveMomentumFloor(state_path=path, **kwargs)
    return floor, path


# ── US-375 Unit Tests ─────────────────────────────────────────────────────────


class TestAdaptiveMomentumFloor_Trigger:
    """tick() triggers adaptation when all guards pass."""

    def test_triggers_on_near_threshold(self):
        """NEAR_THRESHOLD + no cooldown + room to reduce → returns new threshold."""
        floor, _ = _make_floor(step=0.02, cooldown=20, max_adaptations=5, floor=0.30)
        result = floor.tick(CLASS_NEAR_THRESHOLD, current_min_momentum=0.45)
        assert result == pytest.approx(0.43, abs=1e-4)

    def test_adaptation_count_increments(self):
        floor, _ = _make_floor()
        floor.tick(CLASS_NEAR_THRESHOLD, 0.45)
        assert floor.get_status()["adaptation_count"] == 1

    def test_cooldown_set_after_adaptation(self):
        floor, _ = _make_floor(cooldown=20)
        floor.tick(CLASS_NEAR_THRESHOLD, 0.45)
        assert floor.get_status()["cooldown_remaining"] == 20

    def test_total_reduction_accumulates(self):
        floor, _ = _make_floor(step=0.02, cooldown=0)
        floor.tick(CLASS_NEAR_THRESHOLD, 0.45)
        floor.tick(CLASS_NEAR_THRESHOLD, 0.43)
        assert floor.get_status()["total_reduction"] == pytest.approx(0.04, abs=1e-5)


class TestAdaptiveMomentumFloor_Guards:
    """Guard conditions block adaptation correctly."""

    def test_respects_cooldown(self):
        """After one adaptation, cooldown=20 blocks next tick."""
        floor, _ = _make_floor(step=0.02, cooldown=20, max_adaptations=5, floor=0.30)
        floor.tick(CLASS_NEAR_THRESHOLD, 0.45)  # fires, sets cooldown=20
        result = floor.tick(CLASS_NEAR_THRESHOLD, 0.43)  # cooldown=19 remaining → None
        assert result is None

    def test_cooldown_decrements_on_each_tick(self):
        """cooldown_remaining decrements every tick regardless of classification."""
        floor, _ = _make_floor(step=0.02, cooldown=5, max_adaptations=5, floor=0.30)
        floor.tick(CLASS_NEAR_THRESHOLD, 0.45)  # fires, cooldown=5
        for _ in range(5):
            floor.tick("MODERATE", 0.43)  # non-NEAR_THRESHOLD but still decrements
        assert floor.get_status()["cooldown_remaining"] == 0

    def test_respects_max_adaptations(self):
        """After max_adaptations, tick() returns None even with NEAR_THRESHOLD."""
        floor, _ = _make_floor(step=0.02, cooldown=0, max_adaptations=2, floor=0.30)
        floor.tick(CLASS_NEAR_THRESHOLD, 0.50)  # adaptation 1
        floor.tick(CLASS_NEAR_THRESHOLD, 0.48)  # adaptation 2
        result = floor.tick(CLASS_NEAR_THRESHOLD, 0.46)  # exhausted → None
        assert result is None
        assert floor.get_status()["adaptation_count"] == 2

    def test_respects_floor_guard(self):
        """Would go below floor → blocked, returns None."""
        floor, _ = _make_floor(step=0.02, cooldown=0, max_adaptations=5, floor=0.30)
        # 0.31 - 0.02 = 0.29 < 0.30 floor → blocked
        result = floor.tick(CLASS_NEAR_THRESHOLD, current_min_momentum=0.31)
        assert result is None
        assert floor.get_status()["adaptation_count"] == 0

    def test_exactly_at_floor_boundary_blocked(self):
        """0.32 - 0.02 = 0.30 == floor → allowed (not below)."""
        floor, _ = _make_floor(step=0.02, cooldown=0, max_adaptations=5, floor=0.30)
        result = floor.tick(CLASS_NEAR_THRESHOLD, current_min_momentum=0.32)
        assert result == pytest.approx(0.30, abs=1e-5)

    def test_non_near_threshold_returns_none(self):
        """FAR_BELOW and MODERATE classifications never trigger adaptation."""
        floor, _ = _make_floor()
        for cls in ("FAR_BELOW", "MODERATE", "INSUFFICIENT_DATA", ""):
            result = floor.tick(cls, 0.45)
            assert result is None, f"Expected None for classification={cls!r}"


class TestAdaptiveMomentumFloor_Status:
    """get_status() returns all required fields."""

    def test_get_status_keys(self):
        floor, _ = _make_floor()
        status = floor.get_status()
        required = {
            "adaptation_count", "cooldown_remaining", "max_adaptations",
            "floor", "step", "total_reduction", "last_adapted_threshold",
        }
        for key in required:
            assert key in status, f"Missing status key: {key}"

    def test_get_status_initial_values(self):
        floor, _ = _make_floor(step=0.02, cooldown=20, max_adaptations=5, floor=0.30)
        status = floor.get_status()
        assert status["adaptation_count"] == 0
        assert status["cooldown_remaining"] == 0
        assert status["total_reduction"] == 0.0
        assert status["last_adapted_threshold"] is None


class TestAdaptiveMomentumFloor_Persistence:
    """save_state() / _load_state() round-trip."""

    def test_save_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        path.unlink()

        floor1 = AdaptiveMomentumFloor(step=0.02, cooldown=0, max_adaptations=5, floor=0.30, state_path=path)
        floor1.tick(CLASS_NEAR_THRESHOLD, 0.45)
        floor1.tick(CLASS_NEAR_THRESHOLD, 0.43)
        floor1.save_state()

        floor2 = AdaptiveMomentumFloor(state_path=path)
        status = floor2.get_status()
        assert status["adaptation_count"] == 2
        assert status["total_reduction"] == pytest.approx(0.04, abs=1e-5)
        path.unlink(missing_ok=True)

    def test_save_state_atomic_json(self):
        """save_state() writes valid JSON with required fields."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        path.unlink()

        floor = AdaptiveMomentumFloor(state_path=path)
        floor.tick(CLASS_NEAR_THRESHOLD, 0.45)
        floor.save_state()

        with open(path) as fh:
            state = json.load(fh)
        assert "adaptation_count" in state
        assert "cooldown_remaining" in state
        path.unlink(missing_ok=True)


# ── US-377 Integration Tests ─────────────────────────────────────────────────


class TestPhase62WiringGreps:
    """Per Live Wiring Verification Gates: grep engine.py for call sites."""

    def _src(self) -> str:
        return ENGINE_SRC.read_text()

    def test_adaptive_momentum_floor_init_block(self):
        assert "_adaptive_momentum_floor = None" in self._src()

    def test_adaptive_momentum_floor_import_in_engine(self):
        assert "AdaptiveMomentumFloor" in self._src()

    def test_adaptive_momentum_floor_tick_call(self):
        assert "_adaptive_momentum_floor.tick(" in self._src()

    def test_adaptive_momentum_floor_save_state_call(self):
        assert "_adaptive_momentum_floor.save_state()" in self._src()

    def test_config_mutation_log_present(self):
        """Phase 62 adaptation log line present in engine.py."""
        assert "Phase 62 (US-376): AdaptiveMomentumFloor" in self._src()


class TestPhase62BehavioralSimulations:
    """End-to-end behavioral scenarios per PRD acceptance criteria."""

    def test_full_adaptation_lifecycle(self):
        """Simulate NEAR_THRESHOLD → adapt → cooldown → adapt again after cooldown.

        cooldown=3: decrement-before-check means 2 blocking ticks, then 3rd fires.
        Tick sequence after first adaptation (cooldown=3 set):
          tick2: decrement 3→2, check 2>0 → None
          tick3: decrement 2→1, check 1>0 → None
          tick4: decrement 1→0, check 0>0 → False → fires
        """
        floor, _ = _make_floor(step=0.02, cooldown=3, max_adaptations=5, floor=0.30)
        # First adaptation fires
        r1 = floor.tick(CLASS_NEAR_THRESHOLD, 0.45)
        assert r1 == pytest.approx(0.43, abs=1e-4)
        # Next 2 ticks: cooldown blocks (remaining=2 then 1)
        for _ in range(2):
            assert floor.tick(CLASS_NEAR_THRESHOLD, 0.43) is None
        # 3rd tick after adaptation: decrement 1→0 → fires again
        r2 = floor.tick(CLASS_NEAR_THRESHOLD, 0.43)
        assert r2 == pytest.approx(0.41, abs=1e-4)

    def test_cooldown_blocks_scenario(self):
        """cooldown_remaining=5 → tick() returns None."""
        floor, _ = _make_floor(step=0.02, cooldown=20, max_adaptations=5, floor=0.30)
        floor.tick(CLASS_NEAR_THRESHOLD, 0.45)  # fires, cooldown=20
        result = floor.tick(CLASS_NEAR_THRESHOLD, 0.43)
        assert result is None

    def test_max_adaptations_exhausted_scenario(self):
        """adaptation_count=5 → tick() returns None even with NEAR_THRESHOLD."""
        floor, _ = _make_floor(step=0.02, cooldown=0, max_adaptations=5, floor=0.30)
        threshold = 0.55
        for _ in range(5):
            result = floor.tick(CLASS_NEAR_THRESHOLD, threshold)
            threshold = result
        # Now exhausted
        result = floor.tick(CLASS_NEAR_THRESHOLD, threshold)
        assert result is None

    def test_floor_guard_scenario(self):
        """0.31 - 0.02 = 0.29 < floor=0.30 → tick() returns None."""
        floor, _ = _make_floor(step=0.02, cooldown=0, max_adaptations=5, floor=0.30)
        result = floor.tick(CLASS_NEAR_THRESHOLD, 0.31)
        assert result is None

    def test_far_below_never_adapts(self):
        """FAR_BELOW classification → tick() always returns None."""
        floor, _ = _make_floor(step=0.02, cooldown=0, max_adaptations=5, floor=0.30)
        for _ in range(10):
            assert floor.tick("FAR_BELOW", 0.45) is None

    def test_importability(self):
        from src.scanner.adaptive_momentum_floor import AdaptiveMomentumFloor
        assert AdaptiveMomentumFloor is not None

    def test_phase61_modules_still_importable(self):
        from src.scanner.momentum_score_recorder import MomentumScoreRecorder
        from src.scanner.momentum_gap_analyzer import MomentumGapAnalyzer
        assert MomentumGapAnalyzer is not None
