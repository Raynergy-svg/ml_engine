"""Repo-root pytest conftest — operator-state pollution guard.

Why this exists (2026-07-18 e2e audit): a fresh-clone test run rewrote the real
``.claude/state.json`` (flipped ``halted`` and restamped ``last_updated``).
``StateEngine.STATE_PATH`` is deliberately cwd-relative (the launchd daemons and
the ``monkeypatch.chdir(tmp_path)`` test convention both depend on that), which
means any code path that instantiates ``StateEngine()`` WITHOUT first chdir'ing
into a tmp dir silently writes the operator's live runtime state — a Hard NO #2
surface (respect ``halted``; a test run must never be able to flip it).

This guard makes that failure mode loud and self-healing: every test snapshots
the live state file before it runs; if the file's bytes changed afterwards, the
original content is restored and THAT test fails with a pollution message that
names it. No mocks, real disk, negligible cost (~4 KB read per test).
"""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent
_OPERATOR_STATE = _REPO_ROOT / ".claude" / "state.json"


@pytest.fixture(autouse=True)
def _guard_operator_state(request):
    """Fail (and revert) any test that mutates the repo's live .claude/state.json."""
    before = _OPERATOR_STATE.read_bytes() if _OPERATOR_STATE.exists() else None
    yield
    after = _OPERATOR_STATE.read_bytes() if _OPERATOR_STATE.exists() else None
    if before == after:
        return
    # Restore the operator's real state first — the guard must never leave the
    # polluted content in place (that's how halted got silently flipped).
    if before is None:
        _OPERATOR_STATE.unlink(missing_ok=True)
    else:
        _OPERATOR_STATE.write_bytes(before)
    pytest.fail(
        f"TEST POLLUTION: {request.node.nodeid} mutated the repo's live "
        f"{_OPERATOR_STATE.relative_to(_REPO_ROOT)} (operator runtime state / Hard NO #2 "
        "surface). The original content has been restored. Fix the test: chdir into "
        "tmp_path (monkeypatch.chdir) or pass an explicit state_path before touching "
        "StateEngine or anything that writes .claude/state.json.",
        pytrace=False,
    )
