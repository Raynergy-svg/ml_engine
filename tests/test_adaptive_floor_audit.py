"""Tests for Phase 90 US-405: adaptive_floor_audit.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.scanner.tools.adaptive_floor_audit import run


class TestMomentumFloorEmptyState:
    """AC: AdaptiveMomentumFloor with empty state file → activation_count == 0 and status DORMANT."""

    def test_empty_state_gives_dormant(self, tmp_path: Path) -> None:
        # Empty state file (valid JSON, no adaptations)
        state_file = tmp_path / "momentum_state.json"
        state_file.write_text("{}")

        result = run(
            confidence_floor_path=tmp_path / "conf.json",  # non-existent = fresh
            momentum_floor_path=state_file,
            drawdown_ceiling_path=tmp_path / "dd.json",  # non-existent = fresh
        )

        momentum = result["momentum_floor"]
        assert momentum["activation_count"] == 0
        assert momentum["status"] == "DORMANT"


class TestConfidenceFloorWithPriorAdaptation:
    """AC: AdaptiveConfidenceFloor with state containing 1 prior adaptation → activation_count == 1."""

    def test_one_adaptation_gives_count_1(self, tmp_path: Path) -> None:
        state_file = tmp_path / "conf_state.json"
        state_file.write_text(json.dumps({
            "version": "1.0",
            "current_threshold": 57.0,
            "total_adaptations": 1,
            "cooldown_remaining": 5,
            "last_adaptation_ts": "2026-04-01T00:00:00+00:00",
        }, indent=2))

        result = run(
            confidence_floor_path=state_file,
            momentum_floor_path=tmp_path / "mom.json",
            drawdown_ceiling_path=tmp_path / "dd.json",
        )

        conf = result["confidence_floor"]
        assert conf["activation_count"] == 1


class TestRunReturnsExpectedKeys:
    """AC: audit run() returns dict with keys 'confidence_floor', 'momentum_floor', 'drawdown_ceiling'
    each containing 'activation_count' and 'status'."""

    def test_run_keys_and_subkeys(self, tmp_path: Path) -> None:
        result = run(
            confidence_floor_path=tmp_path / "c.json",
            momentum_floor_path=tmp_path / "m.json",
            drawdown_ceiling_path=tmp_path / "d.json",
        )

        assert set(result.keys()) == {"confidence_floor", "momentum_floor", "drawdown_ceiling"}

        for key in ("confidence_floor", "momentum_floor", "drawdown_ceiling"):
            assert "activation_count" in result[key], f"Missing activation_count in {key}"
            assert "status" in result[key], f"Missing status in {key}"
            assert result[key]["status"] in ("ACTIVE", "DORMANT", "EXHAUSTED")
