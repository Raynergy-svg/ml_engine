"""US-020 — Acceptance tests for the FX directional path retirement.

Real classes + real disk via ``tmp_path``. No ``unittest.mock`` /
``MagicMock`` / ``patch``. The test exercises:

* :func:`disable_fx_directional_path` flips every directional agent flag.
* :func:`enforce_harvester_ship_gate` blocks on **missing / false /
  mismatched** ship-gate state.
* :func:`enforce_harvester_ship_gate` permits a valid + matching gate.
* The retirement roster stays in sync with
  :data:`TeamEvaluator._BASE_WEIGHTS` (so a future agent addition can't
  silently bypass the guard).
* The equity harvester engine path does NOT import any of the FX
  directional agent evaluators — preserved as a canary so the runtime
  re-pointing stays honest.
* :func:`atomic_write_json` round-trips and uses a ``.tmp`` staging
  file (atomic-replace semantics).
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

from src.scanner.agents._team import ScannerAgentTeam
from src.scanner.config import ScannerConfig
from src.scanner.fx_retired import (
    FX_DIRECTIONAL_AGENT_FLAGS,
    FX_DIRECTIONAL_AGENT_NAMES,
    FXPathRetiredError,
    SHIP_GATE_PATH_DEFAULT,
    atomic_write_json,
    disable_fx_directional_path,
    enforce_harvester_ship_gate,
)


VALID_HASH = (
    "8bfe419a0c1c06306ff3fa35c39132df522b17ae31718a1dd5463097b467899c"
)


def _write_ship_gate(path: Path, payload: dict) -> Path:
    """Helper — write a ship-gate JSON file atomically via the module API."""
    return atomic_write_json(payload, path)


# ---------------------------------------------------------------------------
# Roster sync — the canonical 15 agents
# ---------------------------------------------------------------------------


def test_retired_roster_matches_team_base_weights() -> None:
    """The retirement roster must stay in sync with ``_BASE_WEIGHTS``."""
    base_weight_keys = set(ScannerAgentTeam._BASE_WEIGHTS.keys())
    retired_names = set(FX_DIRECTIONAL_AGENT_NAMES)
    assert base_weight_keys == retired_names, (
        "FX retirement roster drifted from TeamEvaluator._BASE_WEIGHTS. "
        f"Only in _BASE_WEIGHTS: {base_weight_keys - retired_names}. "
        f"Only in FX_DIRECTIONAL_AGENT_NAMES: "
        f"{retired_names - base_weight_keys}"
    )
    # 15 agents — invariant in the PRD.
    assert len(retired_names) == 15


def test_retirement_flag_set_covers_every_agent() -> None:
    """Every retired agent name must have at least one toggle flag."""
    flags = set(FX_DIRECTIONAL_AGENT_FLAGS)
    for agent_name in FX_DIRECTIONAL_AGENT_NAMES:
        expected = f"enable_{agent_name}_agent"
        assert expected in flags, (
            f"Agent {agent_name!r} has no matching toggle flag "
            f"{expected!r} in FX_DIRECTIONAL_AGENT_FLAGS"
        )


# ---------------------------------------------------------------------------
# disable_fx_directional_path
# ---------------------------------------------------------------------------


def test_disable_fx_directional_path_real_scanner_config() -> None:
    """Real ``ScannerConfig`` — every directional flag flips to False."""
    cfg = ScannerConfig()
    flipped = disable_fx_directional_path(cfg)
    # ``trader_readiness`` was already False; the rest should have flipped.
    assert "enable_trader_readiness_agent" not in flipped
    assert "enable_trend_agent" in flipped
    for flag in FX_DIRECTIONAL_AGENT_FLAGS:
        # Every directional flag now reads False on the real config.
        if hasattr(cfg, flag):
            assert getattr(cfg, flag) is False, (
                f"{flag} did not flip to False after retirement"
            )


def test_disable_fx_directional_path_is_idempotent() -> None:
    """Calling twice is a no-op the second time."""
    cfg = ScannerConfig()
    first = disable_fx_directional_path(cfg)
    second = disable_fx_directional_path(cfg)
    assert len(first) > 0
    assert second == tuple()


def test_disable_fx_directional_path_skips_missing_attrs() -> None:
    """Duck-typed object missing some flags is handled cleanly."""

    @dataclass
    class TinyConfig:
        enable_trend_agent: bool = True
        enable_momentum_agent: bool = True
        # No other flags present.

    tiny = TinyConfig()
    flipped = disable_fx_directional_path(tiny)
    assert tiny.enable_trend_agent is False
    assert tiny.enable_momentum_agent is False
    assert set(flipped) == {"enable_trend_agent", "enable_momentum_agent"}


# ---------------------------------------------------------------------------
# enforce_harvester_ship_gate — blocks
# ---------------------------------------------------------------------------


def test_ship_gate_blocks_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_gate.json"
    with pytest.raises(FXPathRetiredError) as exc:
        enforce_harvester_ship_gate(VALID_HASH, ship_gate_path=missing)
    assert "file not found" in str(exc.value)


def test_ship_gate_blocks_when_gate_pass_false(tmp_path: Path) -> None:
    gate_path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(
        gate_path,
        {
            "gate_pass": False,
            "universe_hash": VALID_HASH,
        },
    )
    with pytest.raises(FXPathRetiredError) as exc:
        enforce_harvester_ship_gate(VALID_HASH, ship_gate_path=gate_path)
    assert "gate_pass=" in str(exc.value)
    assert "False" in str(exc.value)


def test_ship_gate_blocks_when_gate_pass_truthy_but_not_True(
    tmp_path: Path,
) -> None:
    """``gate_pass`` must be exactly ``True``, not just truthy."""
    gate_path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(
        gate_path,
        {
            "gate_pass": 1,  # truthy but not bool True
            "universe_hash": VALID_HASH,
        },
    )
    with pytest.raises(FXPathRetiredError):
        enforce_harvester_ship_gate(VALID_HASH, ship_gate_path=gate_path)


def test_ship_gate_blocks_when_universe_hash_mismatch(
    tmp_path: Path,
) -> None:
    gate_path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(
        gate_path,
        {
            "gate_pass": True,
            "universe_hash": "different_hash_value_for_a_different_universe",
        },
    )
    with pytest.raises(FXPathRetiredError) as exc:
        enforce_harvester_ship_gate(VALID_HASH, ship_gate_path=gate_path)
    assert "universe_hash mismatch" in str(exc.value)


def test_ship_gate_blocks_when_universe_hash_blank(tmp_path: Path) -> None:
    gate_path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(
        gate_path,
        {
            "gate_pass": True,
            "universe_hash": "",
        },
    )
    with pytest.raises(FXPathRetiredError) as exc:
        enforce_harvester_ship_gate(VALID_HASH, ship_gate_path=gate_path)
    assert "universe_hash missing or blank" in str(exc.value)


def test_ship_gate_blocks_when_universe_hash_missing(tmp_path: Path) -> None:
    gate_path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate_path, {"gate_pass": True})
    with pytest.raises(FXPathRetiredError):
        enforce_harvester_ship_gate(VALID_HASH, ship_gate_path=gate_path)


def test_ship_gate_blocks_when_payload_not_object(tmp_path: Path) -> None:
    gate_path = tmp_path / "SHIP_GATE.json"
    # Write a JSON list, not an object.
    gate_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(FXPathRetiredError) as exc:
        enforce_harvester_ship_gate(VALID_HASH, ship_gate_path=gate_path)
    assert "not a JSON object" in str(exc.value)


def test_ship_gate_blocks_when_malformed_json(tmp_path: Path) -> None:
    gate_path = tmp_path / "SHIP_GATE.json"
    gate_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(FXPathRetiredError) as exc:
        enforce_harvester_ship_gate(VALID_HASH, ship_gate_path=gate_path)
    assert "malformed JSON" in str(exc.value)


def test_ship_gate_rejects_blank_expected_hash(tmp_path: Path) -> None:
    gate_path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(
        gate_path,
        {"gate_pass": True, "universe_hash": VALID_HASH},
    )
    with pytest.raises(FXPathRetiredError) as exc:
        enforce_harvester_ship_gate("", ship_gate_path=gate_path)
    assert "non-empty string" in str(exc.value)


# ---------------------------------------------------------------------------
# enforce_harvester_ship_gate — permits
# ---------------------------------------------------------------------------


def test_ship_gate_permits_when_valid_and_matching(tmp_path: Path) -> None:
    gate_path = tmp_path / "SHIP_GATE.json"
    payload = {
        "gate_pass": True,
        "universe_hash": VALID_HASH,
        "asof": "2026-06-18",
        "net_sharpe": 0.92,
        "max_dd": 0.229,
        "positive_years": 13,
    }
    _write_ship_gate(gate_path, payload)
    returned = enforce_harvester_ship_gate(
        VALID_HASH, ship_gate_path=gate_path
    )
    assert returned["gate_pass"] is True
    assert returned["universe_hash"] == VALID_HASH
    assert returned["net_sharpe"] == pytest.approx(0.92)


def test_ship_gate_default_path_constant_matches_prd() -> None:
    """The default path must point at the canonical PRD location."""
    assert SHIP_GATE_PATH_DEFAULT == "trained_data/backtests/SHIP_GATE.json"


# ---------------------------------------------------------------------------
# Atomic-write contract
# ---------------------------------------------------------------------------


def test_atomic_write_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "subdir" / "out.json"
    payload = {"gate_pass": True, "universe_hash": VALID_HASH, "n": 7}
    out = atomic_write_json(payload, target)
    assert out == target
    assert target.exists()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == payload


def test_atomic_write_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    target.write_text(json.dumps({"old": "value"}), encoding="utf-8")
    payload = {"new": "value"}
    atomic_write_json(payload, target)
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_atomic_write_leaves_no_temp_file_on_success(tmp_path: Path) -> None:
    """``.tmp`` staging files must not survive a successful write."""
    target = tmp_path / "out.json"
    atomic_write_json({"k": "v"}, target)
    surviving_tmps = sorted(tmp_path.glob(".*.tmp"))
    assert surviving_tmps == [], (
        f"atomic_write_json left orphan tmp files: {surviving_tmps}"
    )


# ---------------------------------------------------------------------------
# Canary — harvester engine path doesn't import the FX directional agents
# ---------------------------------------------------------------------------


_EQUITY_DIR = Path(__file__).resolve().parent.parent / "src" / "equity"


_FORBIDDEN_FX_RUNTIME_SYMBOLS = (
    # The whole-team scanner runtime — never reachable from src/equity.
    "src.scanner.engine",
    "src.scanner.agents.directional",
    # NOTE: ``src.scanner.agents._team`` IS allowed for the
    # ``AgentVerdict`` dataclass shape (see US-008). The forbidden bit
    # is constructing the TeamEvaluator runtime, which would happen via
    # ``src.scanner.engine`` — already blocked above.
)


def _collect_imports(path: Path) -> List[str]:
    """Return every ``import`` / ``from ... import`` target in a file."""
    targets: List[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        pytest.fail(f"unparseable file {path}: {exc}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            targets.append(module)
    return targets


def test_equity_path_does_not_import_fx_runtime() -> None:
    """No FX directional runtime symbol is reachable from ``src/equity/``."""
    offenders = []
    for py in sorted(_EQUITY_DIR.rglob("*.py")):
        for target in _collect_imports(py):
            for forbidden in _FORBIDDEN_FX_RUNTIME_SYMBOLS:
                if target == forbidden or target.startswith(forbidden + "."):
                    offenders.append((py.relative_to(_EQUITY_DIR), target))
    assert offenders == [], (
        "Harvester engine path leaks into the retired FX runtime: "
        + ", ".join(f"{p}:{t}" for p, t in offenders)
    )


def test_no_mock_imports_in_this_test_file() -> None:
    """Self-check — the acceptance suite must not import ``unittest.mock``.

    Uses AST so the docstring's literal mention of ``unittest.mock``
    (explaining the no-mock rule) does not trip the check.
    """
    this_file = Path(__file__)
    tree = ast.parse(this_file.read_text(encoding="utf-8"), filename=str(this_file))
    forbidden_modules = {"unittest.mock", "mock"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_modules, (
                    f"forbidden import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module not in forbidden_modules, (
                f"forbidden from-import: {module}"
            )


# ---------------------------------------------------------------------------
# End-to-end — disable + ship-gate together, the way the operator
# would actually wire them.
# ---------------------------------------------------------------------------


def test_end_to_end_retirement_and_handoff(tmp_path: Path) -> None:
    """Realistic narrative: retire FX, then verify harvester gate clears."""
    # 1. Retire the FX runtime on a real config.
    cfg = ScannerConfig()
    flipped = disable_fx_directional_path(cfg)
    assert len(flipped) >= 1
    for flag in FX_DIRECTIONAL_AGENT_FLAGS:
        if hasattr(cfg, flag):
            assert getattr(cfg, flag) is False

    # 2. Write a passing ship gate on disk (atomic) and verify it clears.
    gate_path = tmp_path / "SHIP_GATE.json"
    payload = {
        "gate_pass": True,
        "universe_hash": VALID_HASH,
        "net_sharpe": 0.921,
        "max_dd": 0.229,
        "positive_years": 13,
        "total_years": 17,
        "asof": "2026-05-29",
    }
    atomic_write_json(payload, gate_path)
    cleared = enforce_harvester_ship_gate(
        VALID_HASH, ship_gate_path=gate_path
    )
    assert cleared["gate_pass"] is True

    # 3. Now corrupt the gate and assert the same guard refuses.
    bad_payload = dict(payload)
    bad_payload["gate_pass"] = False
    atomic_write_json(bad_payload, gate_path)
    with pytest.raises(FXPathRetiredError):
        enforce_harvester_ship_gate(VALID_HASH, ship_gate_path=gate_path)

    # 4. Sanity — file still exists; retirement is non-destructive.
    assert gate_path.exists()
    assert os.path.getsize(gate_path) > 0
