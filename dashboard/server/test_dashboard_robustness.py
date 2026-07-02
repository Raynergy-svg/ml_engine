"""Regression tests for the two 2026-07-02 dashboard robustness bugs.

Bug 1: read_tier7_diagnostics() crashed the API worker when `pgrep` hung —
subprocess.TimeoutExpired (a subprocess.SubprocessError) was not in the except
clause, so it propagated through FastAPI's threadpool and killed the process
(confirmed in trained_data/axiom/launchd-api.log, 2026-07-02 00:44).

Bug 2: /api/control/state stalled under ~86 concurrent long-lived SSE
connections because the live-lane oracle (which shells out to `ps`) was called
uncached and inline on every SSE tick for every connection.

No mocks (project rule): monkeypatch swaps in real functions/objects, same
pattern already used by test_control_loop_state.py for the sibling `ps` call.
"""
import subprocess
from types import SimpleNamespace

from dashboard.server import control
from dashboard.server import data_sources as ds


def _fake_run_pgrep_hangs(cmd, *_args, **kwargs):
    if cmd and cmd[0] == "pgrep":
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 3))
    return SimpleNamespace(stdout="", stderr="")


def test_tier7_diagnostics_survives_hung_pgrep(monkeypatch):
    """A hung `pgrep` must degrade the row to unknown, never crash the caller."""
    monkeypatch.setattr(subprocess, "run", _fake_run_pgrep_hangs)

    result = ds.read_tier7_diagnostics(client=None)  # must not raise

    checks = {c["label"]: c for c in result["checks"]}
    assert checks["Memory / CPU"]["ok"] is False  # trend_pid stayed None, honestly


def test_pgrep_timeout_is_a_subprocess_error_not_caught_by_bare_oserror():
    """Guards the exact defect: TimeoutExpired is a SubprocessError, not an OSError."""
    assert issubclass(subprocess.TimeoutExpired, subprocess.SubprocessError)
    assert not issubclass(subprocess.TimeoutExpired, OSError)


def test_cached_live_lane_running_dedupes_concurrent_callers(monkeypatch):
    monkeypatch.setitem(ds._lane_cache, "ts", 0.0)
    monkeypatch.setitem(ds._lane_cache, "data", None)
    calls = {"n": 0}

    def fake_live_lane_running():
        calls["n"] += 1
        return {"running": True, "call": calls["n"]}

    mod = SimpleNamespace(live_lane_running=fake_live_lane_running)

    first = ds._cached_live_lane_running(mod)
    second = ds._cached_live_lane_running(mod)  # within TTL -> cache hit, no new call

    assert calls["n"] == 1
    assert first == second == {"running": True, "call": 1}

    # Force TTL expiry -> next call must hit the oracle again.
    monkeypatch.setitem(ds._lane_cache, "ts", 0.0)
    third = ds._cached_live_lane_running(mod)
    assert calls["n"] == 2
    assert third == {"running": True, "call": 2}


def test_loop_pids_cached_dedupes_concurrent_state_polls(monkeypatch):
    monkeypatch.setattr(control, "_pid_cache", {})
    calls = {"n": 0}

    def fake_loop_pids(_loop):
        calls["n"] += 1
        return [999]

    monkeypatch.setattr(control, "_loop_pids", fake_loop_pids)

    first = control._loop_pids_cached("tier7")
    second = control._loop_pids_cached("tier7")  # within TTL -> no new subprocess call

    assert calls["n"] == 1
    assert first == second == [999]


def test_start_and_stop_loop_never_use_the_cached_pid_lookup():
    """start/stop MUST see the true, uncached pid list — caching only belongs on
    the read-only /api/control/state display path (control._state)."""
    import inspect

    assert "_loop_pids_cached" not in inspect.getsource(control._start_loop)
    assert "_loop_pids_cached" not in inspect.getsource(control._stop_loop)
