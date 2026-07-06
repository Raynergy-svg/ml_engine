"""Singleton-lock guard for the Tier 7 self-heal supervisor (2026-07-03).

Reproduce-then-resolve: on 2026-07-03 a launchd-spawned supervisor (PID 88167)
and a leftover nohup supervisor (PID 16644) ran CONCURRENTLY for the whole debug
window — both writing the shared heartbeat / tier7_state / self-heal event log
every 30s with NO mutual exclusion (unlike run_oanda_trend, which already had a
flock singleton). Two self-heal loops racing on the same config/weight files is a
two-writer collision (the class of bug this project has been bitten by before).

These tests use REAL flock on REAL files via tmp_path — no mocks. flock is keyed
to the open-file-description, so two independent opens of the SAME path model two
separate OS processes within one test process.
"""
import scripts.run_tier7_loop as loop


def test_singleton_lock_refuses_a_second_concurrent_holder(tmp_path):
    lock_path = tmp_path / "tier7_loop.singleton.lock"
    first = loop._acquire_singleton_lock(lock_path)
    assert first is not None
    second = loop._acquire_singleton_lock(lock_path)
    assert second is None  # refused — a live holder already has it
    first.close()
    third = loop._acquire_singleton_lock(lock_path)
    assert third is not None  # released on close -> a new acquire succeeds
    third.close()


def test_lock_records_pid_for_observability(tmp_path):
    lock_path = tmp_path / "tier7_loop.singleton.lock"
    fh = loop._acquire_singleton_lock(lock_path)
    assert fh is not None
    fh.close()
    # The holder writes its pid into the file so an operator inspecting the lock
    # can see who owns it (mirrors run_oanda_trend).
    assert lock_path.read_text().strip().isdigit()


def test_release_frees_lock_for_reacquire_across_reexec(tmp_path, monkeypatch):
    """The code-freshness self-restart uses os.execv (same PID, in-place image
    replace) — UNLIKE run_oanda_trend, which exits for a launchd respawn. If the
    supervisor holds the flock and the re-exec'd image tries to re-acquire before
    the old lock is gone, it would be DENIED BY ITS OWN inherited lock and refuse
    to start — a self-inflicted outage on every committed fix. The loop must
    release the lock immediately before execv so the fresh image re-acquires
    cleanly. Simulate the boundary: acquire (old image) -> release (pre-execv) ->
    acquire (new image) must succeed."""
    monkeypatch.setattr(loop, "REPO_ROOT", tmp_path)
    lock_path = loop._singleton_lock_path()

    old_image = loop._acquire_singleton_lock(lock_path)
    assert old_image is not None
    loop._SINGLETON_LOCK_FH = old_image

    loop._release_singleton_lock()  # what the loop calls right before os.execv

    new_image = loop._acquire_singleton_lock(lock_path)
    assert new_image is not None  # re-exec'd process is NOT blocked by the old lock
    new_image.close()
    loop._SINGLETON_LOCK_FH = None


def test_singleton_lock_path_is_outside_documents_tcc_free_dir(tmp_path, monkeypatch):
    """Lock lives under trained_data/axiom next to the trend lock. The RUNNING
    process writes it (proven to have ~/Documents write access); only launchd's
    pre-exec xpcproxy is TCC-blocked there, and it never touches the lock file."""
    monkeypatch.setattr(loop, "REPO_ROOT", tmp_path)
    assert loop._singleton_lock_path().name == "tier7_loop.singleton.lock"
    assert loop._singleton_lock_path().parent.name == "axiom"
