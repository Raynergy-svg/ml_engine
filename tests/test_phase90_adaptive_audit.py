"""Phase 90 US-405: Tests for adaptive_floor_audit.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.scanner.tools.adaptive_floor_audit import run, _audit_confidence_floor, _audit_momentum_floor, _audit_drawdown_ceiling


# ── Helpers ────────────────────────────────────────────────────────────────


def _write_state(path: Path, adaptation_count: int = 0, total_adaptations: int | None = None) -> None:
    """Write a minimal state file for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": "1.0",
        "adaptation_count": adaptation_count,
        "total_adaptations": adaptation_count if total_adaptations is None else total_adaptations,
        "current_threshold": 58.0,
        "cooldown_remaining": 0,
        "last_adaptation_ts": None,
    }
    path.write_text(json.dumps(data, indent=2))


# ── AdaptiveMomentumFloor with empty state ─────────────────────────────────


def test_momentum_floor_empty_state_is_dormant(tmp_path):
    """AdaptiveMomentumFloor with empty state file → activation_count=0 and DORMANT."""
    state_path = tmp_path / "adaptive_momentum_floor_state.json"
    result = _audit_momentum_floor(state_path)
    assert result["activation_count"] == 0
    assert result["status"] == "DORMANT"
    assert result["state_file_exists"] is False


def test_momentum_floor_with_prior_adaptation(tmp_path):
    """AdaptiveMomentumFloor with 1 prior adaptation → activation_count=1 and ACTIVE."""
    state_path = tmp_path / "adaptive_momentum_floor_state.json"
    _write_state(state_path, adaptation_count=1)
    result = _audit_momentum_floor(state_path)
    assert result["activation_count"] == 1
    assert result["status"] == "ACTIVE"
    assert result["state_file_exists"] is True


# ── AdaptiveConfidenceFloor ────────────────────────────────────────────────


def test_confidence_floor_empty_state_is_dormant(tmp_path):
    """AdaptiveConfidenceFloor with empty state file → activation_count=0 and DORMANT."""
    state_path = tmp_path / "adaptive_floor_state.json"
    result = _audit_confidence_floor(state_path)
    assert result["activation_count"] == 0
    assert result["status"] == "DORMANT"


def test_confidence_floor_with_one_adaptation(tmp_path):
    """AdaptiveConfidenceFloor with state containing 1 prior adaptation → activation_count=1."""
    state_path = tmp_path / "adaptive_floor_state.json"
    _write_state(state_path, total_adaptations=1)
    result = _audit_confidence_floor(state_path)
    assert result["activation_count"] == 1
    assert result["status"] == "ACTIVE"


# ── AdaptiveDrawdownCeiling ────────────────────────────────────────────────


def test_drawdown_ceiling_empty_state_is_dormant(tmp_path):
    """AdaptiveDrawdownCeiling with empty state file → activation_count=0 and DORMANT."""
    state_path = tmp_path / "adaptive_drawdown_ceiling_state.json"
    result = _audit_drawdown_ceiling(state_path)
    assert result["activation_count"] == 0
    assert result["status"] == "DORMANT"


# ── run() integration ──────────────────────────────────────────────────────


def test_run_returns_all_three_component_keys(tmp_path):
    """run() returns dict with keys 'confidence_floor', 'momentum_floor', 'drawdown_ceiling'."""
    confidence_path = tmp_path / "adaptive_floor_state.json"
    momentum_path = tmp_path / "adaptive_momentum_floor_state.json"
    drawdown_path = tmp_path / "adaptive_drawdown_ceiling_state.json"

    result = run(
        confidence_floor_path=confidence_path,
        momentum_floor_path=momentum_path,
        drawdown_ceiling_path=drawdown_path,
    )
    assert "confidence_floor" in result
    assert "momentum_floor" in result
    assert "drawdown_ceiling" in result


def test_run_each_component_has_activation_count_and_status(tmp_path):
    """Each component result in run() contains 'activation_count' and 'status'."""
    confidence_path = tmp_path / "adaptive_floor_state.json"
    momentum_path = tmp_path / "adaptive_momentum_floor_state.json"
    drawdown_path = tmp_path / "adaptive_drawdown_ceiling_state.json"

    result = run(
        confidence_floor_path=confidence_path,
        momentum_floor_path=momentum_path,
        drawdown_ceiling_path=drawdown_path,
    )
    for key in ("confidence_floor", "momentum_floor", "drawdown_ceiling"):
        assert "activation_count" in result[key], f"{key} missing activation_count"
        assert "status" in result[key], f"{key} missing status"
        assert result[key]["status"] in ("ACTIVE", "DORMANT", "EXHAUSTED"), (
            f"{key} status invalid: {result[key]['status']}"
        )


def test_audit_imports_all_three_components():
    """Grep: adaptive_floor_audit.py imports AdaptiveConfidenceFloor, AdaptiveMomentumFloor, AdaptiveDrawdownCeiling."""
    import subprocess
    for name in ("AdaptiveConfidenceFloor", "AdaptiveMomentumFloor", "AdaptiveDrawdownCeiling"):
        result = subprocess.run(
            ["grep", "-c", name, "/Users/buddy/Documents/ml_engine/src/scanner/tools/adaptive_floor_audit.py"],
            capture_output=True, text=True,
        )
        assert int(result.stdout.strip()) >= 1, f"adaptive_floor_audit.py should import {name}"


def test_audit_references_trained_data_paths():
    """Grep: adaptive_floor_audit.py references trained_data/ paths for all three state files."""
    import subprocess
    result = subprocess.run(
        ["grep", "-c", "trained_data/", "/Users/buddy/Documents/ml_engine/src/scanner/tools/adaptive_floor_audit.py"],
        capture_output=True, text=True,
    )
    count = int(result.stdout.strip())
    assert count >= 3, f"Expected >=3 trained_data/ references, found {count}"
