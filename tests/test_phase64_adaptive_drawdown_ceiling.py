"""Phase 64 US-382 + US-383 + US-384: AdaptiveDrawdownCeiling tests.

Unit tests (US-382) + integration/wiring tests (US-383/384).
Minimum 8 unit + 10 integration = 18 tests required.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.scanner.adaptive_drawdown_ceiling import AdaptiveDrawdownCeiling

ENGINE_PATH = Path(__file__).parent.parent / "src" / "scanner" / "engine.py"

CLASS_NEAR = "NEAR_THRESHOLD"
CLASS_MODERATE = "MODERATE"
CLASS_FAR_ABOVE = "FAR_ABOVE"
CLASS_INSUFFICIENT = "INSUFFICIENT_DATA"


def make_ceiling(tmp_path: Path, **kwargs) -> AdaptiveDrawdownCeiling:
    defaults = dict(
        step=0.001,
        cooldown=30,
        max_adaptations=3,
        ceiling=0.040,
        state_path=tmp_path / "state.json",
    )
    defaults.update(kwargs)
    return AdaptiveDrawdownCeiling(**defaults)


# ================================================================== #
# US-382: Unit tests — AdaptiveDrawdownCeiling class                  #
# ================================================================== #

def test_triggers_correctly_on_near_threshold(tmp_path):
    """NEAR_THRESHOLD + cooldown=0 + count=0 + drawdown=0.025 → returns 0.026."""
    c = make_ceiling(tmp_path)
    result = c.tick(CLASS_NEAR, 0.025)
    assert result is not None
    assert abs(result - 0.026) < 1e-7
    assert c._adaptation_count == 1


def test_direction_is_increase_not_decrease(tmp_path):
    """Ceiling raises threshold — returned value must be > input."""
    c = make_ceiling(tmp_path)
    result = c.tick(CLASS_NEAR, 0.025)
    assert result > 0.025, f"Expected increase, got {result} (input was 0.025)"


def test_non_near_threshold_returns_none(tmp_path):
    """FAR_ABOVE, MODERATE, INSUFFICIENT_DATA all return None."""
    c = make_ceiling(tmp_path)
    assert c.tick(CLASS_FAR_ABOVE, 0.025) is None
    assert c.tick(CLASS_MODERATE, 0.025) is None
    assert c.tick(CLASS_INSUFFICIENT, 0.025) is None


def test_cooldown_blocks_after_adaptation(tmp_path):
    """After adaptation, cooldown=30 blocks next N-1 ticks (decrement-before-check)."""
    c = make_ceiling(tmp_path, cooldown=3)
    # First tick — fires adaptation, sets cooldown=3
    r1 = c.tick(CLASS_NEAR, 0.025)
    assert r1 == 0.026
    # tick+1: decrement→2, 2>0 → block
    assert c.tick(CLASS_NEAR, 0.026) is None
    # tick+2: decrement→1, 1>0 → block
    assert c.tick(CLASS_NEAR, 0.026) is None
    # tick+3: decrement→0, 0>0 → False → fires
    r4 = c.tick(CLASS_NEAR, 0.026)
    assert r4 == 0.027


def test_cooldown_remaining_decrements_on_non_near(tmp_path):
    """Cooldown decrements even when classification != NEAR_THRESHOLD."""
    c = make_ceiling(tmp_path, cooldown=2)
    c.tick(CLASS_NEAR, 0.025)     # fires, cooldown=2
    c.tick(CLASS_MODERATE, 0.026)  # decrements cooldown to 1
    assert c._cooldown_remaining == 1


def test_max_adaptations_respected(tmp_path):
    """After max_adaptations=3 fires, tick() always returns None."""
    c = make_ceiling(tmp_path, cooldown=0, max_adaptations=3)
    # Fire 3 adaptations (cooldown=0 → immediate re-trigger each tick)
    r1 = c.tick(CLASS_NEAR, 0.025)
    assert r1 == 0.026
    r2 = c.tick(CLASS_NEAR, 0.026)
    assert r2 == 0.027
    r3 = c.tick(CLASS_NEAR, 0.027)
    assert r3 == 0.028
    # 4th tick: adaptation_count=3 → exhausted
    assert c.tick(CLASS_NEAR, 0.028) is None
    assert c._adaptation_count == 3


def test_ceiling_guard_respected(tmp_path):
    """If candidate would exceed ceiling, return None."""
    # ceiling=0.040, current=0.040, step=0.001 → candidate=0.041 > ceiling
    c = make_ceiling(tmp_path, ceiling=0.040, step=0.001)
    result = c.tick(CLASS_NEAR, 0.040)
    assert result is None


def test_ceiling_guard_boundary_exact(tmp_path):
    """candidate == ceiling exactly is allowed (0.039 + 0.001 = 0.040)."""
    c = make_ceiling(tmp_path, ceiling=0.040, step=0.001)
    result = c.tick(CLASS_NEAR, 0.039)
    assert result is not None
    assert abs(result - 0.040) < 1e-7


def test_save_load_roundtrip(tmp_path):
    """State persists across instantiation."""
    state_path = tmp_path / "state.json"
    c1 = AdaptiveDrawdownCeiling(state_path=state_path, cooldown=5)
    c1.tick(CLASS_NEAR, 0.025)   # adaptation_count=1, cooldown=5
    c1.tick(CLASS_NEAR, 0.026)   # decrements to 4, blocked
    c1.save_state()

    c2 = AdaptiveDrawdownCeiling(state_path=state_path, cooldown=5)
    assert c2._adaptation_count == 1
    assert c2._cooldown_remaining == 4


def test_get_status_returns_all_keys(tmp_path):
    """get_status() must return all 7 required fields."""
    c = make_ceiling(tmp_path)
    status = c.get_status()
    for key in ["adaptation_count", "cooldown_remaining", "max_adaptations",
                "ceiling", "step", "total_increase", "last_adapted_threshold"]:
        assert key in status, f"Missing key: {key}"


def test_total_increase_tracks_cumulative(tmp_path):
    """total_increase accumulates correctly across multiple adaptations."""
    c = make_ceiling(tmp_path, cooldown=0, step=0.001)
    c.tick(CLASS_NEAR, 0.025)
    c.tick(CLASS_NEAR, 0.026)
    assert abs(c._total_increase - 0.002) < 1e-7


def test_adaptation_after_full_cooldown_cycle(tmp_path):
    """Full lifecycle: fire → cooldown → fire again.

    cooldown=3 with decrement-before-check semantics:
      After adapt sets cooldown=3:
        tick+1: decrement→2, 2>0 → block
        tick+2: decrement→1, 1>0 → block
        tick+3: decrement→0, 0>0 → False → fires
      So 2 blocking ticks then the 3rd fires.
    """
    c = make_ceiling(tmp_path, cooldown=3, max_adaptations=2)
    # Adaptation 1
    r1 = c.tick(CLASS_NEAR, 0.025)
    assert r1 == 0.026
    # 2 blocking ticks (cooldown goes 3→2→1)
    assert c.tick(CLASS_NEAR, 0.026) is None
    assert c.tick(CLASS_NEAR, 0.026) is None
    # 3rd tick: decrement 1→0 → fires
    r2 = c.tick(CLASS_NEAR, 0.026)
    assert r2 == 0.027
    assert c._adaptation_count == 2


# ================================================================== #
# US-383 + US-384: Wiring and integration tests                       #
# ================================================================== #

def get_engine_src() -> str:
    return ENGINE_PATH.read_text()


def test_wiring_init_block():
    """_adaptive_drawdown_ceiling = None init block must exist."""
    src = get_engine_src()
    assert "_adaptive_drawdown_ceiling = None" in src


def test_wiring_adaptive_drawdown_ceiling_import():
    """AdaptiveDrawdownCeiling must be imported in engine.py."""
    src = get_engine_src()
    assert "AdaptiveDrawdownCeiling" in src


def test_wiring_tick_call():
    """_adaptive_drawdown_ceiling.tick( must be called in engine.py."""
    src = get_engine_src()
    assert "_adaptive_drawdown_ceiling.tick(" in src


def test_wiring_save_state_call():
    """_adaptive_drawdown_ceiling.save_state() must be called in engine.py."""
    src = get_engine_src()
    assert "_adaptive_drawdown_ceiling.save_state()" in src


def test_wiring_config_mutation_log():
    """Phase 64 log tag and max_drawdown_pct config mutation must exist."""
    src = get_engine_src()
    assert "Phase 64" in src
    assert "max_drawdown_pct raised" in src


def test_simulate_adaptation_fires(tmp_path):
    """NEAR_THRESHOLD + cooldown=0 + count=0 + drawdown=0.025 → 0.026."""
    c = make_ceiling(tmp_path)
    result = c.tick(CLASS_NEAR, 0.025)
    assert result is not None
    assert abs(result - 0.026) < 1e-7
    assert c._adaptation_count == 1
    assert c._cooldown_remaining == 30


def test_simulate_cooldown_block(tmp_path):
    """cooldown_remaining=5 → tick() returns None."""
    c = make_ceiling(tmp_path)
    c._cooldown_remaining = 5
    result = c.tick(CLASS_NEAR, 0.025)
    assert result is None
    assert c._cooldown_remaining == 4  # decremented even though blocked


def test_simulate_max_adaptations_exhausted(tmp_path):
    """adaptation_count=3 → tick() returns None even with NEAR_THRESHOLD."""
    c = make_ceiling(tmp_path, max_adaptations=3)
    c._adaptation_count = 3
    result = c.tick(CLASS_NEAR, 0.025)
    assert result is None


def test_simulate_ceiling_guard(tmp_path):
    """max_drawdown=0.040, step=0.001 → tick() returns None."""
    c = make_ceiling(tmp_path, ceiling=0.040, step=0.001)
    result = c.tick(CLASS_NEAR, 0.040)
    assert result is None


def test_simulate_non_near_threshold_far_above(tmp_path):
    """FAR_ABOVE never adapts."""
    c = make_ceiling(tmp_path)
    for _ in range(10):
        assert c.tick(CLASS_FAR_ABOVE, 0.025) is None


def test_simulate_save_load_two_adaptations(tmp_path):
    """After 2 adaptations, loaded instance has adaptation_count=2."""
    state_path = tmp_path / "state.json"
    c1 = AdaptiveDrawdownCeiling(state_path=state_path, cooldown=0, max_adaptations=3)
    c1.tick(CLASS_NEAR, 0.025)
    c1.tick(CLASS_NEAR, 0.026)
    c1.save_state()

    c2 = AdaptiveDrawdownCeiling(state_path=state_path, cooldown=0, max_adaptations=3)
    assert c2._adaptation_count == 2
    assert abs(c2._total_increase - 0.002) < 1e-7


def test_all_phase64_modules_importable():
    """Phase 64 module imports alongside Phase 56-63 modules."""
    from src.scanner.adaptive_drawdown_ceiling import AdaptiveDrawdownCeiling
    from src.scanner.risk_reject_recorder import RiskRejectRecorder
    from src.scanner.risk_gate_analyzer import analyze
    from src.scanner.adaptive_momentum_floor import AdaptiveMomentumFloor
    assert True
