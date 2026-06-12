"""Unit tests for ClaudeReflectionHandler and claude_subprocess helpers.

Covers CLAUDE.md test-coverage gate (≥5 tests per new subsystem) with
focused isolation — every external boundary (Claude CLI, filesystem,
git) is mocked. No real subprocess spawns in these tests.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scanner.automation.claude_subprocess import (
    ReflectionBudget,
    ReflectionResult,
    SingleFlightLock,
    _parse_result_block,
    invoke_claude_reflection,
)
from src.scanner.automation.event_handlers import (
    ClaudeReflectionHandler,
    PER_TRADE_HANDLERS,
    SyncContext,
)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    """Isolate filesystem writes to a tmp dir — no pollution of real .claude/."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude").mkdir()
    (tmp_path / "logs").mkdir()
    return tmp_path


@pytest.fixture
def sample_sync_context():
    """A minimal SyncContext matching what execution.sync_closed_trades_rl produces."""
    entry = {
        "trade_id": "T-1001",
        "pair": "EUR_USD",
        "direction": "LONG",
        "confidence": 0.72,
        "sl_pips": 10.0,
        "tp_pips": 20.0,
        "rr_ratio": 2.0,
        "gates": {"momentum_passed": True, "confidence_passed": True, "risk_passed": True},
        "agents": {
            "weighted_vote_score": 0.68,
            "agent_reasons": [
                {"name": "trend", "score": 0.75, "weight": 1.2, "reason": "uptrend"},
            ],
        },
        "regime": {"volatility_regime": "NORMAL", "atr_pips": 8.5},
    }
    outcome = {"exit_reason": "TP", "duration_minutes": 45}
    return SyncContext(
        entry=entry,
        outcome_data=outcome,
        closed_trades={"T-1001": {"realizedPL": 12.5, "initialUnits": 10000}},
        entries=[entry],
        synced_entries=[entry],
        rl_updates=[],
        scanner=None,
        execution_manager=None,
        synced_count=1,
        trade_id="T-1001",
        pair="EUR_USD",
        trade_won=True,
        realized_pl=12.5,
        pnl_pips=20.0,
    )


# ── Parse helpers ──────────────────────────────────────────────────────


def test_parse_result_block_yaml_like():
    stdout = """Some analysis text...

<reflection-result>
artifacts_written:
  - .claude/learnings.md
  - .claude/proposed_weights.json
cost_usd: 0.08
hypothesis: "EUR_USD shorts at London open hit TP faster than expected"
confidence: 0.7
</reflection-result>
<promise>REFLECTION_COMPLETE</promise>
"""
    block = _parse_result_block(stdout)
    assert block["cost_usd"] == 0.08
    assert block["confidence"] == 0.7
    assert block["hypothesis"].startswith("EUR_USD shorts")
    assert block["artifacts_written"] == [
        ".claude/learnings.md",
        ".claude/proposed_weights.json",
    ]


def test_parse_result_block_json():
    stdout = """
<reflection-result>
{"cost_usd": 0.12, "hypothesis": "noise", "artifacts_written": []}
</reflection-result>
"""
    block = _parse_result_block(stdout)
    assert block["cost_usd"] == 0.12
    assert block["artifacts_written"] == []


def test_parse_result_block_missing_returns_empty():
    assert _parse_result_block("") == {}
    assert _parse_result_block("no tags here") == {}
    # Malformed (unclosed tag) — returns empty, does not raise
    assert _parse_result_block("<reflection-result>hypothesis: x") == {}


# ── Budget ─────────────────────────────────────────────────────────────


def test_budget_rollover_on_new_day(tmp_workspace):
    budget = ReflectionBudget(path=tmp_workspace / ".claude" / "budget.json", daily_cap_usd=5.0)
    # Write an old-dated record
    budget.path.write_text(json.dumps({"date": "1999-01-01", "spent_usd": 99.0, "spawn_count": 50, "deep_count": 50}))
    state = budget.load()
    assert state["spent_usd"] == 0.0
    assert state["spawn_count"] == 0
    assert state["deep_count"] == 0


def test_budget_blocks_when_cap_hit(tmp_workspace):
    budget = ReflectionBudget(path=tmp_workspace / ".claude" / "budget.json", daily_cap_usd=1.0)
    budget.record(cost_usd=1.5, mode="deep")
    assert budget.allows("lightweight") is False
    assert budget.allows("deep") is False


def test_budget_blocks_deep_cap_separately(tmp_workspace):
    budget = ReflectionBudget(path=tmp_workspace / ".claude" / "budget.json", daily_cap_usd=100.0)
    # Trigger the deep_count cap
    for _ in range(20):
        budget.record(cost_usd=0.01, mode="deep")
    assert budget.allows("deep") is False
    # Lightweight still allowed under $ cap
    assert budget.allows("lightweight") is True


# ── Lock ───────────────────────────────────────────────────────────────


def test_single_flight_lock_blocks_second_acquire(tmp_workspace):
    lock1 = SingleFlightLock(path=tmp_workspace / ".claude" / "lock")
    lock2 = SingleFlightLock(path=tmp_workspace / ".claude" / "lock")
    assert lock1.acquire() is True
    assert lock2.acquire() is False
    lock1.release()
    assert lock2.acquire() is True
    lock2.release()


def test_single_flight_lock_clears_stale(tmp_workspace):
    lock_path = tmp_workspace / ".claude" / "lock"
    lock = SingleFlightLock(path=lock_path, stale_after_seconds=0)
    # Simulate a stale lock from a dead prior process
    lock_path.write_text("99999")
    # stale_after_seconds=0 means anything is stale — should clear + acquire
    import time
    time.sleep(0.01)
    assert lock.acquire() is True
    lock.release()


# ── Handler: decision logic ────────────────────────────────────────────


def test_handler_mode_lightweight_for_small_win(sample_sync_context):
    handler = ClaudeReflectionHandler()
    # 12.5 profit, 20 pnl_pips, 10 sl_pips → 20 pips = 2x SL (not 3x), so lightweight
    mode = handler._decide_mode(sample_sync_context)
    assert mode == "lightweight"


def test_handler_mode_deep_for_big_loss(sample_sync_context):
    sample_sync_context.trade_won = False
    sample_sync_context.realized_pl = -75.0
    handler = ClaudeReflectionHandler()
    assert handler._decide_mode(sample_sync_context) == "deep"


def test_handler_mode_deep_for_big_r_win(sample_sync_context):
    # pnl_pips = 3.5 * sl_pips → deep
    sample_sync_context.pnl_pips = 35.0
    sample_sync_context.entry["sl_pips"] = 10.0
    sample_sync_context.trade_won = True
    handler = ClaudeReflectionHandler()
    assert handler._decide_mode(sample_sync_context) == "deep"


def test_handler_mode_deep_for_three_consecutive_losses(sample_sync_context):
    # Reset class-level state to avoid test-order dependency
    ClaudeReflectionHandler._recent_outcomes = [False, False, False]
    sample_sync_context.trade_won = False
    sample_sync_context.realized_pl = -10.0  # small loss, but 3 in a row
    handler = ClaudeReflectionHandler()
    assert handler._decide_mode(sample_sync_context) == "deep"
    # Clean up to avoid polluting other tests
    ClaudeReflectionHandler._recent_outcomes = []


# ── Handler: integration boundaries (mocked) ───────────────────────────


def test_handler_skips_when_budget_exhausted(tmp_workspace, sample_sync_context, monkeypatch):
    """If budget.allows() returns False, handler must not spawn subprocess."""
    ClaudeReflectionHandler._recent_outcomes = []
    handler = ClaudeReflectionHandler()

    fake_invoker = MagicMock()
    fake_budget = MagicMock()
    fake_budget.allows.return_value = False
    fake_lock_cls = MagicMock()

    # Inject fakes directly (skip _ensure_deps import)
    handler._invoker = fake_invoker
    handler._budget = fake_budget
    handler._lock_cls = fake_lock_cls

    result = handler.handle({}, sample_sync_context)
    assert result is True  # Never breaks chain
    fake_invoker.assert_not_called()


def test_handler_skips_when_lock_held(tmp_workspace, sample_sync_context):
    """If single-flight lock already held, handler must return True and not spawn."""
    ClaudeReflectionHandler._recent_outcomes = []
    handler = ClaudeReflectionHandler()

    fake_invoker = MagicMock()
    fake_budget = MagicMock()
    fake_budget.allows.return_value = True
    fake_lock = MagicMock()
    fake_lock.acquire.return_value = False
    fake_lock_cls = MagicMock(return_value=fake_lock)

    handler._invoker = fake_invoker
    handler._budget = fake_budget
    handler._lock_cls = fake_lock_cls

    assert handler.handle({}, sample_sync_context) is True
    fake_invoker.assert_not_called()
    fake_lock.release.assert_not_called()  # we never acquired, nothing to release


def test_handler_spawns_claude_and_records_budget(tmp_workspace, sample_sync_context):
    """Happy path: budget OK, lock acquired, subprocess invoked, budget recorded."""
    ClaudeReflectionHandler._recent_outcomes = []
    handler = ClaudeReflectionHandler()

    fake_result = ReflectionResult(
        success=True,
        trade_id="T-1001",
        mode="lightweight",
        duration_seconds=1.5,
        stdout_len=1000,
        promise_found=True,
        artifacts_written=[".claude/learnings.md"],
        hypothesis="test hypothesis",
        cost_usd=0.04,
        returncode=0,
    )
    fake_invoker = MagicMock(return_value=fake_result)
    fake_budget = MagicMock()
    fake_budget.allows.return_value = True
    fake_lock = MagicMock()
    fake_lock.acquire.return_value = True
    fake_lock_cls = MagicMock(return_value=fake_lock)

    handler._invoker = fake_invoker
    handler._budget = fake_budget
    handler._lock_cls = fake_lock_cls

    assert handler.handle({}, sample_sync_context) is True
    fake_invoker.assert_called_once()
    call_kwargs = fake_invoker.call_args.kwargs
    assert call_kwargs["trade_id"] == "T-1001"
    assert call_kwargs["mode"] == "lightweight"
    # Lightweight timeout raised 60 -> 90 in commit 96fb666 (2026-04-16).
    assert call_kwargs["timeout_seconds"] == 90
    fake_budget.record.assert_called_once_with(cost_usd=0.04, mode="lightweight")
    fake_lock.release.assert_called_once()


def test_handler_returns_true_even_when_subprocess_fails(sample_sync_context):
    """Handler-chain isolation: reflection failure MUST NOT poison the chain."""
    ClaudeReflectionHandler._recent_outcomes = []
    handler = ClaudeReflectionHandler()

    fake_result = ReflectionResult(
        success=False,
        trade_id="T-1001",
        mode="lightweight",
        duration_seconds=60.0,
        stdout_len=0,
        error="claude timed out",
        returncode=-1,
    )
    fake_invoker = MagicMock(return_value=fake_result)
    fake_budget = MagicMock()
    fake_budget.allows.return_value = True
    fake_lock = MagicMock()
    fake_lock.acquire.return_value = True

    handler._invoker = fake_invoker
    handler._budget = fake_budget
    handler._lock_cls = MagicMock(return_value=fake_lock)

    # Must not raise — must return True
    assert handler.handle({}, sample_sync_context) is True
    fake_lock.release.assert_called_once()  # Always release even on failure


# ── Registry wiring ────────────────────────────────────────────────────


def test_claude_reflection_handler_registered_in_per_trade():
    """ClaudeReflectionHandler must be present in PER_TRADE_HANDLERS registry.

    Without this, the handler is defined but never invoked — CLAUDE.md live-wiring
    verification gate prevents shipping write-only dead code.
    """
    assert ClaudeReflectionHandler in PER_TRADE_HANDLERS
    # Priority must be highest so it runs AFTER all other per-trade handlers
    assert ClaudeReflectionHandler.priority >= max(
        h.priority for h in PER_TRADE_HANDLERS if h is not ClaudeReflectionHandler
    )


def test_invoke_claude_missing_cli_captured_as_error(tmp_workspace, monkeypatch):
    """FileNotFoundError (claude CLI not on PATH) must be captured, not raised."""
    # Get past the no-LLM policy chokepoint (default BLOCKED) so the
    # subprocess error-capture path under test is actually reached.
    monkeypatch.setenv("BUDDY_META_USE_LLM", "1")
    with patch("src.scanner.automation.claude_subprocess.subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("claude not found")
        result = invoke_claude_reflection(
            prompt="test", trade_id="T-X", mode="lightweight", timeout_seconds=1
        )
    assert result.success is False
    assert "claude CLI not found" in (result.error or "")


def test_invoke_claude_timeout_captured_as_error(tmp_workspace, monkeypatch):
    """Subprocess timeout must be captured as structured error, not raised."""
    import subprocess as _sub

    # Get past the no-LLM policy chokepoint (default BLOCKED) so the
    # subprocess error-capture path under test is actually reached.
    monkeypatch.setenv("BUDDY_META_USE_LLM", "1")
    with patch("src.scanner.automation.claude_subprocess.subprocess.run") as mock_run:
        mock_run.side_effect = _sub.TimeoutExpired(cmd="claude", timeout=1)
        result = invoke_claude_reflection(
            prompt="test", trade_id="T-Y", mode="deep", timeout_seconds=1
        )
    assert result.success is False
    assert "timed out" in (result.error or "")
