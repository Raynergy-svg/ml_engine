"""P0 (2026-07-06): running_status.py extended to cover every real lane.

Before this fix, the oracle only knew about two things: the LIVE lane (OANDA
trend + Tier 7) and a single "HARVESTER LANE" bucket that mixed the CURRENT,
actually-running OANDA-based equity shadow harvester (`run_equity_harvester.py`)
in with genuinely retired IBKR-era code (`AutonomousLoop`, `embedded_scanner`,
`continuous.py`, `buddy_scanner`) and labeled the whole bucket "dormant/legacy
IBKR — SUPERSEDED; running:NO is EXPECTED". With 6 real daemons live on
2026-07-06 (oanda trend, tier7, equity harvester, brain loop, crypto_momentum
shadow, track_b shadow — confirmed via `ps`), that bucket was actively lying:
an operator reading the oracle would see the CURRENT equity lane reported as
expected-dormant, and would see NOTHING AT ALL about brain/crypto_momentum/
track_b, three lanes `.claude/state.json.halted_lanes` shows as unhalted.

Reproduce-then-resolve, using synthetic `ps` lines + tmp_path artifacts (no
mocks, real module, real disk) — no need to touch the actual live daemons.
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "running_status", _REPO / ".claude" / "loop" / "running_status.py")
rs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rs)


# Real command lines as they actually appear in `ps axo command` output today.
_PS_ALL_SIX_DAEMONS = [
    "/opt/homebrew/Caskroom/miniforge/base/bin/python3 /Users/buddy/Documents/ml_engine/scripts/run_tier7_loop.py",
    "/opt/homebrew/Caskroom/miniforge/base/bin/python3 scripts/run_oanda_trend.py --loop 3600",
    "python3 scripts/run_crypto_momentum_shadow.py --loop --interval 86400",
    "python3 scripts/run_track_b_shadow.py --loop --interval 86400",
    "python3 scripts/run_brain_loop.py --loop --max-cycles 8760 --interval 3600",
    "python3 scripts/run_equity_harvester.py --broker shadow --loop 3600",
]


def test_regression_equity_harvester_is_a_shadow_lane_not_legacy():
    """Locks the fix: run_equity_harvester must route to the shadow-lane
    bucket, never the legacy/dormant bucket — the exact prior mislabeling."""
    shadow_needles = {needle for _, needle, _, _ in rs._SHADOW_LANES}
    assert "run_equity_harvester" in shadow_needles
    assert not any("run_equity_harvester" in n for n in rs._LEGACY_NEEDLES)
    # And the inverse: legacy needles must not accidentally match a real
    # shadow-lane command line.
    equity_line = [ln for ln in _PS_ALL_SIX_DAEMONS if "run_equity_harvester" in ln]
    assert not rs._proc_alive(rs._LEGACY_NEEDLES, equity_line)


def test_all_four_shadow_lanes_registered():
    names = {name for name, _, _, _ in rs._SHADOW_LANES}
    assert names == {"equity", "brain", "crypto_momentum", "track_b"}


def test_shadow_lanes_detects_all_processes_alive(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "ROOT", tmp_path)  # no artifacts on disk
    status = rs.shadow_lanes_running(lines=_PS_ALL_SIX_DAEMONS)
    for name in ("equity", "brain", "crypto_momentum", "track_b"):
        assert status[name]["running"] is True, f"{name} should be detected as running via process"
        assert status[name]["process_alive"] is True
        assert status[name]["artifact_fresh"] is False  # no artifact in tmp_path


def test_shadow_lane_dead_when_neither_process_nor_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    status = rs.shadow_lanes_running(lines=["some unrelated process --flag"])
    for name in ("equity", "brain", "crypto_momentum", "track_b"):
        assert status[name]["running"] is False


def test_shadow_lane_running_via_fresh_artifact_alone(tmp_path, monkeypatch):
    """Artifact freshness is a valid secondary signal even with no matching ps line
    (e.g. ps unavailable, or the process forked under a different argv)."""
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    ledger = tmp_path / "trained_data" / "equity"
    ledger.mkdir(parents=True)
    (ledger / "cycle_ledger.jsonl").write_text(json.dumps({"cycle": 1}) + "\n")

    status = rs.shadow_lanes_running(lines=["no matching process here"])
    assert status["equity"]["running"] is True
    assert status["equity"]["process_alive"] is False
    assert status["equity"]["artifact_fresh"] is True


def test_shadow_lane_artifact_stale_beyond_tolerance(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    ledger_dir = tmp_path / "trained_data" / "equity"
    ledger_dir.mkdir(parents=True)
    f = ledger_dir / "cycle_ledger.jsonl"
    f.write_text(json.dumps({"cycle": 1}) + "\n")
    import os
    old = time.time() - 100000  # well past the 7200s tolerance for "equity"
    os.utime(f, (old, old))

    status = rs.shadow_lanes_running(lines=["no matching process"])
    assert status["equity"]["running"] is False
    assert status["equity"]["artifact_fresh"] is False


def test_live_lane_shape_unchanged():
    """The safety-critical LIVE lane detection (OANDA trend + Tier7) must be
    untouched by this extension — same keys, same semantics."""
    d = rs.live_lane_running()
    for k in ("running", "oanda_trend_proc", "account_state_fresh",
              "tier7_heartbeat_fresh", "tier7_pid_alive"):
        assert k in d


def test_main_assert_running_still_gates_on_live_lane_only(monkeypatch):
    """--assert-running must still key off the LIVE lane exclusively — a
    down shadow lane must never trip the exit-3 alarm meant for the trader."""
    down_live = {
        "running": False, "oanda_trend_proc": False, "account_state_fresh": False,
        "account_state_age_s": None, "tier7_heartbeat_fresh": False,
        "tier7_heartbeat_age_s": None, "tier7_pid_alive": False,
    }
    up_live = {
        "running": True, "oanda_trend_proc": True, "account_state_fresh": True,
        "account_state_age_s": 1.0, "tier7_heartbeat_fresh": True,
        "tier7_heartbeat_age_s": 1.0, "tier7_pid_alive": True,
    }
    monkeypatch.setattr(rs, "live_lane_running", lambda: down_live)
    monkeypatch.setattr(rs, "shadow_lanes_running", lambda lines=None: {
        n: {"running": True, "process_alive": True, "artifact_fresh": True, "artifact_age_s": 1.0}
        for n, _, _, _ in rs._SHADOW_LANES
    })
    assert rs.main(["--assert-running"]) == 3

    monkeypatch.setattr(rs, "live_lane_running", lambda: up_live)
    assert rs.main(["--assert-running"]) == 0
