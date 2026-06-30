"""US-022 — Gated live path tests.

Every test uses real classes against real disk via ``tmp_path``. No
``unittest.mock``, no ``MagicMock``, no ``patch``. Anywhere the gate
needs a kill-switch / drawdown-guardian factory we pass a real callable
(or a real factory that raises) — never a test double.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.equity.live_gate import (
    AUDIT_PATH_DEFAULT,
    LIVE_CONFIRMATION_TOKEN,
    LOG_TAG_ARM,
    LOG_TAG_DISARM,
    LOG_TAG_REFUSE,
    MAX_INITIAL_NAV_FRACTION,
    MAX_PORTFOLIO_RISK_FRACTION,
    SHIP_GATE_PATH_DEFAULT,
    STATE_PATH_DEFAULT,
    ArmResult,
    LiveGate,
    LiveGateConfig,
    LiveGateError,
    LiveGateState,
    enforce_ship_gate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_ship_gate(
    path: Path,
    *,
    gate_pass: bool = True,
    universe_hash: str = "univ-abc-123",
    extra: dict | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate_pass": gate_pass,
        "net_sharpe": 0.92,
        "max_dd": 0.229,
        "positive_years": 12,
        "universe_hash": universe_hash,
        "asof": "2026-06-21",
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


@pytest.fixture()
def ship_gate_path(tmp_path: Path) -> Path:
    return tmp_path / "trained_data" / "backtests" / "SHIP_GATE.json"


@pytest.fixture()
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "trained_data" / "equity" / "live_gate_state.json"


@pytest.fixture()
def audit_path(tmp_path: Path) -> Path:
    return tmp_path / "trained_data" / "equity" / "live_gate_audit.jsonl"


@pytest.fixture()
def gate(state_path: Path, audit_path: Path) -> LiveGate:
    return LiveGate(
        LiveGateConfig(),
        state_path=state_path,
        audit_path=audit_path,
        clock=lambda: 1_700_000_000.0,
    )


# ---------------------------------------------------------------------------
# Public contract constants
# ---------------------------------------------------------------------------


def test_live_confirmation_token_is_exact_literal() -> None:
    """The TUI mode_modal.py pins on this same literal. Don't drift."""
    assert LIVE_CONFIRMATION_TOKEN == "LIVE"


def test_path_defaults_are_under_trained_data() -> None:
    assert SHIP_GATE_PATH_DEFAULT == "trained_data/backtests/SHIP_GATE.json"
    assert STATE_PATH_DEFAULT.startswith("trained_data/")
    assert AUDIT_PATH_DEFAULT.startswith("trained_data/")


def test_max_portfolio_risk_is_15_percent_invariant() -> None:
    """CLAUDE.md 'Trading invariants': max portfolio risk 15% NAV."""
    assert MAX_PORTFOLIO_RISK_FRACTION == 0.15


def test_max_initial_nav_fraction_bounded() -> None:
    assert 0.0 < MAX_INITIAL_NAV_FRACTION <= 1.0


# ---------------------------------------------------------------------------
# LiveGateConfig validation
# ---------------------------------------------------------------------------


def test_config_default_is_small_initial_nav_fraction() -> None:
    cfg = LiveGateConfig()
    assert 0.0 < cfg.initial_nav_fraction <= MAX_INITIAL_NAV_FRACTION
    assert cfg.max_portfolio_risk_fraction == MAX_PORTFOLIO_RISK_FRACTION


@pytest.mark.parametrize("frac", [0.0, -0.01, 0.26, 0.5, 1.0])
def test_config_rejects_out_of_range_initial_nav(frac: float) -> None:
    with pytest.raises(LiveGateError):
        LiveGateConfig(initial_nav_fraction=frac)


@pytest.mark.parametrize("frac", [0.0, -0.01, 0.16, 0.5, 1.0])
def test_config_rejects_above_15pct_portfolio_risk(frac: float) -> None:
    with pytest.raises(LiveGateError):
        LiveGateConfig(max_portfolio_risk_fraction=frac)


# ---------------------------------------------------------------------------
# enforce_ship_gate guard (module refuses to run)
# ---------------------------------------------------------------------------


def test_enforce_ship_gate_blocks_when_missing(ship_gate_path: Path) -> None:
    with pytest.raises(LiveGateError, match=r"ship gate not found"):
        enforce_ship_gate(ship_gate_path, "univ-abc-123")


def test_enforce_ship_gate_blocks_when_gate_pass_false(
    ship_gate_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, gate_pass=False)
    with pytest.raises(LiveGateError, match=r"gate_pass=False"):
        enforce_ship_gate(ship_gate_path, "univ-abc-123")


def test_enforce_ship_gate_blocks_on_universe_hash_mismatch(
    ship_gate_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")
    with pytest.raises(LiveGateError, match=r"universe_hash mismatch"):
        enforce_ship_gate(ship_gate_path, "univ-xyz-999")


def test_enforce_ship_gate_blocks_on_blank_universe_hash(
    ship_gate_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="")
    with pytest.raises(LiveGateError, match=r"universe_hash missing/blank"):
        enforce_ship_gate(ship_gate_path, "univ-abc-123")


def test_enforce_ship_gate_blocks_on_unreadable_json(
    ship_gate_path: Path,
) -> None:
    ship_gate_path.parent.mkdir(parents=True, exist_ok=True)
    ship_gate_path.write_text("not json {")
    with pytest.raises(LiveGateError, match=r"cannot read"):
        enforce_ship_gate(ship_gate_path, "univ-abc-123")


def test_enforce_ship_gate_blocks_on_non_object_payload(
    ship_gate_path: Path,
) -> None:
    ship_gate_path.parent.mkdir(parents=True, exist_ok=True)
    ship_gate_path.write_text("[1, 2, 3]")
    with pytest.raises(LiveGateError, match=r"not a JSON object"):
        enforce_ship_gate(ship_gate_path, "univ-abc-123")


def test_enforce_ship_gate_passes_on_clear_gate(
    ship_gate_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")
    # No exception raised
    enforce_ship_gate(ship_gate_path, "univ-abc-123")


def test_enforce_ship_gate_blocks_on_gate_pass_string_true(
    ship_gate_path: Path,
) -> None:
    """``gate_pass: 'true'`` (string) MUST refuse — only Python True passes."""
    ship_gate_path.parent.mkdir(parents=True, exist_ok=True)
    ship_gate_path.write_text(
        json.dumps({"gate_pass": "true", "universe_hash": "univ-abc-123"})
    )
    with pytest.raises(LiveGateError, match=r"gate_pass="):
        enforce_ship_gate(ship_gate_path, "univ-abc-123")


# ---------------------------------------------------------------------------
# LiveGate.arm — happy path
# ---------------------------------------------------------------------------


def test_arm_succeeds_with_typed_confirmation_and_clear_gate(
    gate: LiveGate, ship_gate_path: Path, state_path: Path, audit_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")
    assert gate.is_armed() is False
    result = gate.arm(
        ship_gate_path=ship_gate_path,
        expected_universe_hash="univ-abc-123",
        confirmation_token=LIVE_CONFIRMATION_TOKEN,
    )
    assert isinstance(result, ArmResult)
    assert result.armed is True
    assert result.universe_hash == "univ-abc-123"
    assert result.initial_nav_fraction == gate.config.initial_nav_fraction
    assert (
        result.max_portfolio_risk_fraction
        == gate.config.max_portfolio_risk_fraction
    )
    assert gate.is_armed() is True

    # Persisted to disk atomically
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["armed"] is True
    assert payload["universe_hash"] == "univ-abc-123"
    assert payload["last_event"] == "arm"

    # Audit log appended
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "arm"
    assert record["armed"] is True


def test_arm_persists_across_gate_reconstruction(
    ship_gate_path: Path, state_path: Path, audit_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")
    g1 = LiveGate(
        LiveGateConfig(),
        state_path=state_path,
        audit_path=audit_path,
        clock=lambda: 1_700_000_000.0,
    )
    g1.arm(
        ship_gate_path=ship_gate_path,
        expected_universe_hash="univ-abc-123",
        confirmation_token=LIVE_CONFIRMATION_TOKEN,
    )

    # Fresh instance loads the persisted armed=True
    g2 = LiveGate(
        LiveGateConfig(),
        state_path=state_path,
        audit_path=audit_path,
    )
    assert g2.is_armed() is True
    assert g2.state.universe_hash == "univ-abc-123"


# ---------------------------------------------------------------------------
# LiveGate.arm — refuse paths (live CANNOT fire)
# ---------------------------------------------------------------------------


def test_arm_refuses_when_ship_gate_missing(
    gate: LiveGate, ship_gate_path: Path,
) -> None:
    with pytest.raises(LiveGateError, match=r"ship gate not found"):
        gate.arm(
            ship_gate_path=ship_gate_path,
            expected_universe_hash="univ-abc-123",
            confirmation_token=LIVE_CONFIRMATION_TOKEN,
        )
    assert gate.is_armed() is False


def test_arm_refuses_when_gate_pass_false(
    gate: LiveGate, ship_gate_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, gate_pass=False)
    with pytest.raises(LiveGateError, match=r"gate_pass=False"):
        gate.arm(
            ship_gate_path=ship_gate_path,
            expected_universe_hash="univ-abc-123",
            confirmation_token=LIVE_CONFIRMATION_TOKEN,
        )
    assert gate.is_armed() is False


def test_arm_refuses_on_universe_hash_mismatch(
    gate: LiveGate, ship_gate_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")
    with pytest.raises(LiveGateError, match=r"universe_hash mismatch"):
        gate.arm(
            ship_gate_path=ship_gate_path,
            expected_universe_hash="univ-different",
            confirmation_token=LIVE_CONFIRMATION_TOKEN,
        )
    assert gate.is_armed() is False


@pytest.mark.parametrize(
    "token",
    ["live", "Live", "LIVe", "", " LIVE", "LIVE ", "YES", "CONFIRM"],
)
def test_arm_refuses_on_wrong_typed_confirmation(
    gate: LiveGate, ship_gate_path: Path, token: str,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")
    with pytest.raises(LiveGateError, match=r"typed confirmation"):
        gate.arm(
            ship_gate_path=ship_gate_path,
            expected_universe_hash="univ-abc-123",
            confirmation_token=token,
        )
    assert gate.is_armed() is False


def test_arm_refuses_when_token_not_string(
    gate: LiveGate, ship_gate_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")
    with pytest.raises(LiveGateError, match=r"must be a string"):
        gate.arm(
            ship_gate_path=ship_gate_path,
            expected_universe_hash="univ-abc-123",
            confirmation_token=None,  # type: ignore[arg-type]
        )
    assert gate.is_armed() is False


def test_arm_refuses_when_kill_switch_factory_raises(
    gate: LiveGate, ship_gate_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")

    def kill_switch_factory():
        raise RuntimeError("ship gate refuses: kill switch unhappy")

    with pytest.raises(LiveGateError, match=r"kill switch \(US-013\) refused"):
        gate.arm(
            ship_gate_path=ship_gate_path,
            expected_universe_hash="univ-abc-123",
            confirmation_token=LIVE_CONFIRMATION_TOKEN,
            kill_switch_factory=kill_switch_factory,
        )
    assert gate.is_armed() is False


def test_arm_refuses_when_drawdown_guardian_factory_raises(
    gate: LiveGate, ship_gate_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")

    def guardian_factory():
        raise RuntimeError("ship gate refuses: drawdown guardian unhappy")

    with pytest.raises(
        LiveGateError, match=r"drawdown guardian \(US-008\) refused"
    ):
        gate.arm(
            ship_gate_path=ship_gate_path,
            expected_universe_hash="univ-abc-123",
            confirmation_token=LIVE_CONFIRMATION_TOKEN,
            drawdown_guardian_factory=guardian_factory,
        )
    assert gate.is_armed() is False


def test_arm_succeeds_when_both_factories_construct(
    gate: LiveGate, ship_gate_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")
    ks_calls = []
    dd_calls = []

    def ks():
        ks_calls.append(1)
        return object()

    def dd():
        dd_calls.append(1)
        return object()

    result = gate.arm(
        ship_gate_path=ship_gate_path,
        expected_universe_hash="univ-abc-123",
        confirmation_token=LIVE_CONFIRMATION_TOKEN,
        kill_switch_factory=ks,
        drawdown_guardian_factory=dd,
    )
    assert result.armed is True
    assert ks_calls == [1]
    assert dd_calls == [1]


# ---------------------------------------------------------------------------
# Refuse audit + log behavior
# ---------------------------------------------------------------------------


def test_refuse_appends_audit_record_and_persists_state(
    gate: LiveGate,
    ship_gate_path: Path,
    state_path: Path,
    audit_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")
    with pytest.raises(LiveGateError):
        gate.arm(
            ship_gate_path=ship_gate_path,
            expected_universe_hash="univ-abc-123",
            confirmation_token="not LIVE",
        )

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["armed"] is False
    assert payload["last_event"] == "refuse"
    assert payload["last_event_reason"]

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "refuse"
    assert record["armed"] is False


def test_refuse_logs_with_pinned_tag(
    gate: LiveGate, ship_gate_path: Path, caplog,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")
    caplog.set_level("WARNING", logger="src.equity.live_gate")
    with pytest.raises(LiveGateError):
        gate.arm(
            ship_gate_path=ship_gate_path,
            expected_universe_hash="mismatch",
            confirmation_token=LIVE_CONFIRMATION_TOKEN,
        )
    assert any(LOG_TAG_REFUSE in r.message for r in caplog.records)


def test_arm_logs_with_pinned_tag(
    gate: LiveGate, ship_gate_path: Path, caplog,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")
    caplog.set_level("INFO", logger="src.equity.live_gate")
    gate.arm(
        ship_gate_path=ship_gate_path,
        expected_universe_hash="univ-abc-123",
        confirmation_token=LIVE_CONFIRMATION_TOKEN,
    )
    assert any(LOG_TAG_ARM in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Disarm
# ---------------------------------------------------------------------------


def test_disarm_flips_armed_to_false(
    gate: LiveGate, ship_gate_path: Path, state_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")
    gate.arm(
        ship_gate_path=ship_gate_path,
        expected_universe_hash="univ-abc-123",
        confirmation_token=LIVE_CONFIRMATION_TOKEN,
    )
    assert gate.is_armed() is True

    gate.disarm(reason="operator requested disarm")
    assert gate.is_armed() is False
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["armed"] is False
    assert payload["last_event"] == "disarm"
    assert payload["last_event_reason"] == "operator requested disarm"


def test_disarm_is_idempotent_and_logged(
    gate: LiveGate, audit_path: Path, caplog,
) -> None:
    caplog.set_level("INFO", logger="src.equity.live_gate")
    gate.disarm(reason="first")
    gate.disarm(reason="second")

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["reason"] == "first"
    assert json.loads(lines[1])["reason"] == "second"
    assert any(LOG_TAG_DISARM in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Atomic writes (tmp + os.replace) — no half-written state
# ---------------------------------------------------------------------------


def test_state_write_leaves_no_temp_files(
    gate: LiveGate, ship_gate_path: Path, state_path: Path,
) -> None:
    _write_ship_gate(ship_gate_path, universe_hash="univ-abc-123")
    gate.arm(
        ship_gate_path=ship_gate_path,
        expected_universe_hash="univ-abc-123",
        confirmation_token=LIVE_CONFIRMATION_TOKEN,
    )
    leftover = list(state_path.parent.glob(".*.tmp"))
    assert leftover == []


# ---------------------------------------------------------------------------
# State recovery — malformed/missing state files
# ---------------------------------------------------------------------------


def test_construct_recovers_from_missing_state(
    state_path: Path, audit_path: Path,
) -> None:
    assert not state_path.exists()
    g = LiveGate(
        LiveGateConfig(), state_path=state_path, audit_path=audit_path,
    )
    assert g.is_armed() is False
    assert g.state.universe_hash is None


def test_construct_recovers_from_corrupt_state(
    state_path: Path, audit_path: Path,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("not json")
    g = LiveGate(
        LiveGateConfig(), state_path=state_path, audit_path=audit_path,
    )
    assert g.is_armed() is False


def test_live_gate_state_from_dict_rejects_non_mapping() -> None:
    with pytest.raises(LiveGateError):
        LiveGateState.from_dict([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# No-mock canary — module/tests never IMPORT unittest.mock / MagicMock / patch
# (docstring prose mentioning "mock" is fine — we only block real imports).
# ---------------------------------------------------------------------------


_BANNED_IMPORT_RES = [
    re.compile(r"^\s*from\s+unittest\.mock\b", re.MULTILINE),
    re.compile(r"^\s*from\s+unittest\s+import\s+mock\b", re.MULTILINE),
    re.compile(r"^\s*import\s+unittest\.mock\b", re.MULTILINE),
    re.compile(r"^\s*from\s+mock\b", re.MULTILINE),
    re.compile(r"^\s*import\s+mock\b", re.MULTILINE),
    re.compile(r"^\s*from\s+[^#\n]*\bimport\b[^#\n]*\bMagicMock\b", re.MULTILINE),
    re.compile(r"^\s*from\s+[^#\n]*\bimport\b[^#\n]*\bpatch\b", re.MULTILINE),
]


def _assert_no_mock_imports(src: str) -> None:
    for pattern in _BANNED_IMPORT_RES:
        assert pattern.search(src) is None, (
            f"forbidden mock import matched by {pattern.pattern!r}"
        )


def test_module_does_not_import_unittest_mock() -> None:
    src = Path("src/equity/live_gate.py").read_text(encoding="utf-8")
    _assert_no_mock_imports(src)


def test_test_module_does_not_import_unittest_mock() -> None:
    src = Path("tests/test_equity_live_gate.py").read_text(encoding="utf-8")
    _assert_no_mock_imports(src)
