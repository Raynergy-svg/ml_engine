import json

import scripts.run_oanda_trend as runner


def test_read_control_leverage_prefers_persisted_axiom_override(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    override = tmp_path / "trained_data" / "axiom" / "control_overrides.json"
    override.parent.mkdir(parents=True)
    override.write_text(json.dumps({"gross_leverage": 7.0}), encoding="utf-8")

    assert runner._read_control_leverage(3.0) == 7.0


def test_read_control_leverage_falls_back_when_override_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)

    assert runner._read_control_leverage(3.0) == 3.0


def test_singleton_lock_refuses_a_second_concurrent_holder(tmp_path):
    """Reproduce-then-resolve (verifier finding, 2026-07-01): two concurrent
    run_oanda_trend processes could each independently pass the risk gate off a
    stale snapshot and each place an order — the same duplicate-ticket failure
    via a process-concurrency route. Two independent file opens of the SAME
    path (simulating two separate OS processes, since flock is per open-file-
    description, not per-process) prove the second is refused while the first
    is held, and the lock releases cleanly on close."""
    lock_path = tmp_path / "trend_loop.singleton.lock"
    first = runner._acquire_singleton_lock(lock_path)
    assert first is not None
    second = runner._acquire_singleton_lock(lock_path)
    assert second is None  # refused — a live holder already has it
    first.close()
    third = runner._acquire_singleton_lock(lock_path)
    assert third is not None  # released on close -> a new acquire succeeds
    third.close()
