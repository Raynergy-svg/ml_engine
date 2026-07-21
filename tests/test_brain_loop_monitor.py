"""No-mock tests for src.brain_loop.monitor.

Reuses the real decide_cycle() drawdown/ship-gate/halt rails (no re-derived
math) plus a freshness-gated read of the existing .claude/tier7_state.json
(informational only — the brain never competes with Tier 7's own self-heal).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.brain_loop.monitor import evaluate_rails
from src.scanner.config import ScannerConfig


def _write_state(project_root, halted: bool) -> None:
    claude_dir = project_root / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "state.json").write_text(json.dumps({"halted": halted}))


def _write_ship_gate(project_root) -> None:
    sg_dir = project_root / "trained_data" / "backtests"
    sg_dir.mkdir(parents=True, exist_ok=True)
    (sg_dir / "SHIP_GATE.json").write_text(json.dumps({"gate_pass": True, "universe_hash": "abc"}))


def test_no_breach_on_clean_state(tmp_path):
    _write_state(tmp_path, halted=False)
    _write_ship_gate(tmp_path)
    report = evaluate_rails(
        config=ScannerConfig(), project_root=tmp_path, data_asof=pd.Timestamp.now(tz="UTC")
    )
    assert report.decision == "continue"
    assert report.breach is False
    assert report.breach_reasons == ()


def test_breach_on_drawdown(tmp_path):
    _write_state(tmp_path, halted=False)
    portfolio_dir = tmp_path / "trained_data" / "equity"
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    (portfolio_dir / "portfolio_state.json").write_text(
        json.dumps({"nav": 70.0, "peak_nav": 100.0})
    )
    report = evaluate_rails(
        config=ScannerConfig(), project_root=tmp_path, data_asof=pd.Timestamp.now(tz="UTC")
    )
    assert report.decision == "halt"
    assert report.breach is True
    assert any("drawdown_breach" in r for r in report.breach_reasons)


def test_no_breach_when_already_halted(tmp_path):
    """REFUSE (already halted) is not a NEW breach requiring de-risk action —
    the system is already stopped."""
    _write_state(tmp_path, halted=True)
    report = evaluate_rails(
        config=ScannerConfig(), project_root=tmp_path, data_asof=pd.Timestamp.now(tz="UTC")
    )
    assert report.decision == "refuse"
    assert report.breach is False


def test_tier7_state_fresh_is_trusted(tmp_path):
    now = datetime.now(timezone.utc)
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "state.json").write_text(json.dumps({"halted": False}))
    (claude_dir / "tier7_state.json").write_text(
        json.dumps({"running": True, "halted": False, "generated_at": now.isoformat()})
    )
    report = evaluate_rails(
        config=ScannerConfig(), project_root=tmp_path, now=now,
        data_asof=pd.Timestamp.now(tz="UTC"),
    )
    assert report.tier7_running is True
    assert report.tier7_halted is False


def test_tier7_state_stale_is_not_trusted(tmp_path):
    now = datetime.now(timezone.utc)
    stale_ts = now - timedelta(hours=2)
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "state.json").write_text(json.dumps({"halted": False}))
    (claude_dir / "tier7_state.json").write_text(
        json.dumps({"running": True, "halted": False, "generated_at": stale_ts.isoformat()})
    )
    report = evaluate_rails(
        config=ScannerConfig(), project_root=tmp_path, now=now,
        data_asof=pd.Timestamp.now(tz="UTC"),
    )
    assert report.tier7_running is None
    assert report.tier7_halted is None


def test_tier7_state_missing_is_unknown(tmp_path):
    _write_state(tmp_path, halted=False)
    report = evaluate_rails(
        config=ScannerConfig(), project_root=tmp_path, data_asof=pd.Timestamp.now(tz="UTC")
    )
    assert report.tier7_running is None
    assert report.tier7_state_age_s is None
