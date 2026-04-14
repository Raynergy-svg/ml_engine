"""Tests for ScannerAgentTeam.apply_proposed_weights — Tier 3 safety-gated path.

Key behaviors tested:
  - Schema validation (missing/bad proposed_at, unknown regime/agent)
  - Staleness rejection (> 2h old)
  - Per-agent delta clamp (±5% max per cycle)
  - Range clamp (0.05–10.0 hard bounds)
  - Blend rate dampening
  - Proposal archived after apply
  - File unlinked after apply so we don't re-process
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def isolated_team(tmp_path, monkeypatch):
    """Instantiate ScannerAgentTeam in an isolated CWD with mocked weights file."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude").mkdir()
    (tmp_path / "trained_data" / "models").mkdir(parents=True)

    # Import lazily so monkeypatched cwd takes effect
    from src.scanner.agents._team import ScannerAgentTeam

    # Seed a weights file so team constructs cleanly — minimal valid structure.
    base_weights = {
        "trend": 1.0,
        "mean_reversion": 1.0,
        "volatility": 1.0,
        "risk_sentinel": 1.0,
        "momentum": 1.0,
    }
    weights = {
        "_global": dict(base_weights),
        "NORMAL": dict(base_weights),
        "HIGH": dict(base_weights),
        "EXTREME": dict(base_weights),
        "LOW": dict(base_weights),
        "_meta": {"total_trades": 100, "last_trade_per_agent": {}},
    }
    (tmp_path / "trained_data" / "models" / "agent_weights.json").write_text(
        json.dumps(weights)
    )

    # ScannerAgentTeam has complex init; build a minimal instance by bypassing.
    team = ScannerAgentTeam.__new__(ScannerAgentTeam)
    team._learned_weights = weights
    team._regime_weights = {}
    # Monkey-patch class constants used by the method
    return team


def _now_iso(offset_seconds: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


def _write_proposal(path: Path, proposal: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proposal))


# ── Missing file ───────────────────────────────────────────────────────


def test_apply_returns_noop_when_file_missing(isolated_team):
    result = isolated_team.apply_proposed_weights()
    assert result["applied"] is False
    assert result["skipped_reason"] == "no_proposal"
    assert result["changes"] == {}


# ── Schema validation ──────────────────────────────────────────────────


def test_apply_rejects_malformed_json(isolated_team, tmp_path):
    path = tmp_path / ".claude" / "proposed_weights.json"
    path.write_text("{not valid json")
    result = isolated_team.apply_proposed_weights()
    assert result["applied"] is False
    assert result["skipped_reason"].startswith("parse_error")


def test_apply_rejects_missing_proposed_at(isolated_team, tmp_path):
    _write_proposal(
        tmp_path / ".claude" / "proposed_weights.json",
        {"regime": "NORMAL", "deltas": {"trend": {"target_weight": 1.1}}},
    )
    result = isolated_team.apply_proposed_weights()
    assert result["applied"] is False
    assert "missing proposed_at" in result["skipped_reason"]


def test_apply_rejects_unknown_regime(isolated_team, tmp_path):
    _write_proposal(
        tmp_path / ".claude" / "proposed_weights.json",
        {
            "proposed_at": _now_iso(),
            "regime": "MARS_ATMOSPHERE",
            "deltas": {"trend": {"target_weight": 1.1}},
        },
    )
    result = isolated_team.apply_proposed_weights()
    assert result["applied"] is False
    assert "unknown regime" in result["skipped_reason"]


def test_apply_rejects_stale_proposal(isolated_team, tmp_path):
    """A proposal > 2h old must be rejected — stale means the trade context moved on."""
    _write_proposal(
        tmp_path / ".claude" / "proposed_weights.json",
        {
            "proposed_at": _now_iso(offset_seconds=-3 * 3600),  # 3h old
            "regime": "NORMAL",
            "deltas": {"trend": {"target_weight": 1.2}},
        },
    )
    result = isolated_team.apply_proposed_weights(max_age_seconds=7200)
    assert result["applied"] is False
    assert "stale" in result["skipped_reason"]


def test_apply_silently_skips_unknown_agents(isolated_team, tmp_path):
    """Unknown agent name must NOT crash the apply — skipped with warning."""
    _write_proposal(
        tmp_path / ".claude" / "proposed_weights.json",
        {
            "proposed_at": _now_iso(),
            "regime": "NORMAL",
            "deltas": {"ghost_agent_9000": {"target_weight": 5.0}},
        },
    )
    result = isolated_team.apply_proposed_weights()
    # No known agents → no effective changes → noop + archived
    assert result["applied"] is False
    assert result["skipped_reason"] == "no_effective_changes"


# ── Clamping ───────────────────────────────────────────────────────────


def test_apply_clamps_to_max_delta_pct(isolated_team, tmp_path):
    """Even if target is 10x current, per-cycle clamp caps at ±5%."""
    # Current trend weight = 1.0, target = 5.0 → raw blended = 1.0 + 0.3*4.0 = 2.2
    # But max_delta_pct = 0.05 → max_step = 0.05 → clamped to 1.05
    _write_proposal(
        tmp_path / ".claude" / "proposed_weights.json",
        {
            "proposed_at": _now_iso(),
            "regime": "NORMAL",
            "deltas": {"trend": {"target_weight": 5.0, "reason": "test clamp"}},
        },
    )
    with patch.object(isolated_team, "_save_learned_weights"):
        result = isolated_team.apply_proposed_weights()
    assert result["applied"] is True
    change = result["changes"]["trend"]
    assert change["old"] == pytest.approx(1.0)
    # With max_delta_pct=0.05 of current (1.0), max step is 0.05
    assert change["new"] == pytest.approx(1.05, abs=0.0001)


def test_apply_clamps_to_valid_range(isolated_team, tmp_path):
    """Target < 0.05 or > 10.0 must be clamped before blending."""
    # Set trend to 0.06 so a downward push can test the 0.05 floor via delta cap
    isolated_team._learned_weights["NORMAL"]["trend"] = 0.06
    _write_proposal(
        tmp_path / ".claude" / "proposed_weights.json",
        {
            "proposed_at": _now_iso(),
            "regime": "NORMAL",
            "deltas": {"trend": {"target_weight": 0.001}},  # Below floor
        },
    )
    with patch.object(isolated_team, "_save_learned_weights"):
        result = isolated_team.apply_proposed_weights()
    assert result["applied"] is True
    # target clamped to 0.05, blend = 0.06 + 0.3*(0.05-0.06) = 0.057
    # Then per-cycle cap ±5% = ±0.003 → 0.057 is within so stays 0.057
    new_val = result["changes"]["trend"]["new"]
    assert new_val >= 0.05
    assert new_val < 0.06


def test_apply_soft_blend_dampens_swings(isolated_team, tmp_path):
    """Blend rate 0.3 means we move 30% toward target, not to it directly."""
    # Target within the 5% cap to test the pure blend behavior
    # current=1.0, target=1.03, blend=0.3 → 1.0 + 0.3*0.03 = 1.009
    _write_proposal(
        tmp_path / ".claude" / "proposed_weights.json",
        {
            "proposed_at": _now_iso(),
            "regime": "NORMAL",
            "deltas": {"trend": {"target_weight": 1.03}},
        },
    )
    with patch.object(isolated_team, "_save_learned_weights"):
        result = isolated_team.apply_proposed_weights()
    assert result["applied"] is True
    assert result["changes"]["trend"]["new"] == pytest.approx(1.009, abs=0.0001)


# ── Archive behavior ───────────────────────────────────────────────────


def test_apply_archives_proposal_after_success(isolated_team, tmp_path):
    """After apply, the live file is deleted and an archive entry is written."""
    proposal_path = tmp_path / ".claude" / "proposed_weights.json"
    _write_proposal(
        proposal_path,
        {
            "proposed_at": _now_iso(),
            "regime": "NORMAL",
            "deltas": {"trend": {"target_weight": 1.03}},
        },
    )
    with patch.object(isolated_team, "_save_learned_weights"):
        isolated_team.apply_proposed_weights()
    # Live file removed
    assert not proposal_path.exists()
    # Archive directory populated
    archive_dir = tmp_path / ".claude" / "proposed_weights_archive"
    assert archive_dir.exists()
    entries = list(archive_dir.glob("*.json"))
    assert len(entries) == 1
    archived = json.loads(entries[0].read_text())
    assert archived["status"] == "applied"
    assert "trend" in archived["changes"]


def test_apply_archives_noop_proposals_too(isolated_team, tmp_path):
    """Unknown-agent-only proposals should still be archived so we don't loop."""
    proposal_path = tmp_path / ".claude" / "proposed_weights.json"
    _write_proposal(
        proposal_path,
        {
            "proposed_at": _now_iso(),
            "regime": "NORMAL",
            "deltas": {"ghost_agent": {"target_weight": 5.0}},
        },
    )
    with patch.object(isolated_team, "_save_learned_weights"):
        isolated_team.apply_proposed_weights()
    assert not proposal_path.exists()  # archived even on noop
    archive_dir = tmp_path / ".claude" / "proposed_weights_archive"
    archived_entries = list(archive_dir.glob("*_noop.json"))
    assert len(archived_entries) == 1
