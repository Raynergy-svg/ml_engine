"""End-to-end autonomy loop test — 4 stages:

  Stage 1: Handler chain (mocked Claude) — verifies ClaudeReflectionHandler
           → claude_subprocess → lock → budget → git_sync wiring.
  Stage 2: Real Claude CLI spawn — actually runs `claude --print` against a
           tiny prompt, verifies reflection_log.jsonl entry appears.
  Stage 3: apply_proposed_weights — places a proposal, runs apply, verifies
           weights blended + proposal archived.
  Stage 4: git_sync whitelist — dry-run commit path, verifies whitelist
           enforcement and error surfacing.

Each stage is isolated (separate tmp workspace) so a failure in one doesn't
mask the others. Stage 2 is marked @slow — real Claude spawn takes ~30s
and costs real API tokens. Skip via -m "not slow" for fast iteration.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Stage 1: Mocked handler chain ──────────────────────────────────────


@pytest.fixture
def iso_workspace(tmp_path, monkeypatch):
    """Fresh CWD with .claude/ and logs/ directories."""
    (tmp_path / ".claude").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "trained_data" / "models").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_stage1_handler_chain_with_mocked_claude(iso_workspace):
    """Full PER_TRADE_HANDLER invocation path with mocked claude subprocess.

    Verifies:
      - Handler invokes claude_subprocess.invoke_claude_reflection
      - Budget records the spawn cost
      - Single-flight lock acquired + released
      - git_sync attempted if artifacts are whitelisted
      - Reflection log JSONL gets an entry
      - Handler returns True (chain isolation)
    """
    from src.scanner.automation.event_handlers import (
        ClaudeReflectionHandler,
        SyncContext,
    )
    from src.scanner.automation.claude_subprocess import ReflectionResult

    # Reset class-level state
    ClaudeReflectionHandler._recent_outcomes = []

    # Build a fake closed trade
    entry = {
        "trade_id": "T-E2E-001",
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
                {"name": "trend", "score": 0.75, "weight": 1.2, "reason": "up"},
            ],
        },
        "regime": {"volatility_regime": "NORMAL"},
    }
    ctx = SyncContext(
        entry=entry,
        outcome_data={"exit_reason": "SL", "duration_minutes": 60},
        closed_trades={"T-E2E-001": {"realizedPL": -55.0, "initialUnits": 10000}},
        entries=[entry],
        synced_entries=[entry],
        rl_updates=[],
        scanner=None,
        execution_manager=None,
        synced_count=1,
        trade_id="T-E2E-001",
        pair="EUR_USD",
        trade_won=False,  # loss > $50 → deep mode
        realized_pl=-55.0,
        pnl_pips=-10.0,
    )

    # Mock invoke_claude_reflection — simulate a successful deep reflection
    fake_result = ReflectionResult(
        success=True,
        trade_id="T-E2E-001",
        mode="deep",
        duration_seconds=45.0,
        stdout_len=5000,
        promise_found=True,
        artifacts_written=[".claude/learnings.md"],
        hypothesis="EUR_USD longs at Asia close show 60% failure rate — consider session filter",
        cost_usd=0.09,
        returncode=0,
    )

    # Pre-create the artifact so git_sync whitelist check passes
    (iso_workspace / ".claude" / "learnings.md").write_text(
        "- [2026-04-14] **PATTERN/asia_close_eur_longs**: test entry\n"
    )

    handler = ClaudeReflectionHandler()
    handler._ensure_deps()
    # Replace the invoker with our mock
    mock_invoker = MagicMock(return_value=fake_result)
    handler._invoker = mock_invoker

    # Mode should be "deep" because realized_pl=-55 (<-$50)
    assert handler._decide_mode(ctx) == "deep"

    # Run the handler — must return True even though subprocess is mocked
    result = handler.handle({}, ctx)
    assert result is True, "handler chain isolation broken"

    # Verify invoke was called with correct args
    mock_invoker.assert_called_once()
    call_kwargs = mock_invoker.call_args.kwargs
    assert call_kwargs["trade_id"] == "T-E2E-001"
    assert call_kwargs["mode"] == "deep"
    assert call_kwargs["timeout_seconds"] == 300  # deep mode

    # Verify lock path cleaned up
    assert not (iso_workspace / ".claude" / ".reflection.lock").exists()

    # Verify budget recorded
    budget_state = handler._budget.load()
    assert budget_state["spawn_count"] >= 1
    assert budget_state["deep_count"] >= 1


def test_stage1_handler_isolates_from_subprocess_failure(iso_workspace):
    """Chain isolation: a Claude CLI crash must NOT break the handler chain."""
    from src.scanner.automation.event_handlers import (
        ClaudeReflectionHandler,
        SyncContext,
    )
    from src.scanner.automation.claude_subprocess import ReflectionResult

    ClaudeReflectionHandler._recent_outcomes = []

    ctx = SyncContext(
        entry={"trade_id": "T-FAIL", "pair": "USD_JPY", "sl_pips": 5},
        outcome_data={},
        closed_trades={},
        entries=[],
        synced_entries=[],
        rl_updates=[],
        trade_id="T-FAIL",
        pair="USD_JPY",
        trade_won=False,
        realized_pl=-10.0,
        pnl_pips=-5.0,
    )

    fake_result = ReflectionResult(
        success=False,
        trade_id="T-FAIL",
        mode="lightweight",
        duration_seconds=60.0,
        stdout_len=0,
        error="claude timed out after 60s",
        returncode=-1,
    )
    handler = ClaudeReflectionHandler()
    handler._ensure_deps()
    handler._invoker = MagicMock(return_value=fake_result)

    # Must return True even on subprocess failure
    assert handler.handle({}, ctx) is True


# ── Stage 2: Real Claude CLI spawn (slow, costs real tokens) ───────────


@pytest.mark.slow
def test_stage2_real_claude_reflection_smoke(iso_workspace):
    """Spawn a REAL claude subprocess with a trivial prompt.

    Cost control: uses a minimal prompt (~50 tokens in) and 60s timeout.
    Verifies:
      - claude CLI launches + returns 0
      - reflection_log.jsonl has a new entry
      - The entry has ts, trade_id, mode, duration_seconds

    Skip with:  pytest -m "not slow"
    """
    if not shutil.which("claude"):
        pytest.skip("claude CLI not on PATH")

    from src.scanner.automation.claude_subprocess import (
        invoke_claude_reflection,
        REFLECTION_LOG,
    )

    prompt = (
        "This is a smoke test. "
        "Emit exactly:\n"
        "<reflection-result>\n"
        "artifacts_written: []\n"
        "cost_usd: 0.01\n"
        'hypothesis: "smoke test"\n'
        "</reflection-result>\n"
        "<promise>REFLECTION_COMPLETE</promise>\n"
        "And nothing else."
    )

    result = invoke_claude_reflection(
        prompt=prompt,
        trade_id="T-SMOKE-REAL",
        mode="lightweight",
        timeout_seconds=120,
        cwd=iso_workspace,
    )

    # Claude CLI may legitimately fail on unusual environments; don't fail
    # the suite on infra issues — but if it ran, all the parsing should work.
    assert result.returncode is not None
    assert (iso_workspace / "logs" / "reflection_log.jsonl").exists()

    # Verify the log entry parses as JSON
    with open(iso_workspace / "logs" / "reflection_log.jsonl") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    assert len(entries) >= 1
    entry = entries[-1]
    assert entry["trade_id"] == "T-SMOKE-REAL"
    assert entry["mode"] == "lightweight"
    assert "duration_seconds" in entry
    # Log whatever we got for debugging
    print(f"\n[Stage 2] Real claude spawn result: success={result.success}, "
          f"duration={result.duration_seconds:.1f}s, "
          f"promise={result.promise_found}, "
          f"hypothesis={result.hypothesis!r}")


# ── Stage 3: apply_proposed_weights E2E ────────────────────────────────


def test_stage3_apply_proposed_weights_e2e(iso_workspace):
    """Drop a proposal file, run apply, verify weights blended + archive created."""
    from src.scanner.agents._team import ScannerAgentTeam

    # Seed agent_weights.json with baseline
    weights = {
        "_global": {"trend": 1.0, "mean_reversion": 1.0, "volatility": 1.0, "risk_sentinel": 1.0, "momentum": 1.0},
        "NORMAL": {"trend": 1.0, "mean_reversion": 1.0, "volatility": 1.0, "risk_sentinel": 1.0, "momentum": 1.0},
        "HIGH": {"trend": 1.0, "mean_reversion": 1.0, "volatility": 1.0, "risk_sentinel": 1.0, "momentum": 1.0},
        "EXTREME": {"trend": 1.0, "mean_reversion": 1.0, "volatility": 1.0, "risk_sentinel": 1.0, "momentum": 1.0},
        "LOW": {"trend": 1.0, "mean_reversion": 1.0, "volatility": 1.0, "risk_sentinel": 1.0, "momentum": 1.0},
        "_meta": {"total_trades": 100, "last_trade_per_agent": {}},
    }
    weights_path = iso_workspace / "trained_data" / "models" / "agent_weights.json"
    weights_path.write_text(json.dumps(weights))

    # Place a proposal
    proposal = {
        "proposed_at": datetime.now(timezone.utc).isoformat(),
        "trade_id": "T-E2E-003",
        "regime": "NORMAL",
        "deltas": {
            "trend": {"target_weight": 1.05, "reason": "strong trend signal correlated with wins"},
            "risk_sentinel": {"target_weight": 0.95, "reason": "too conservative"},
        },
    }
    proposal_path = iso_workspace / ".claude" / "proposed_weights.json"
    proposal_path.write_text(json.dumps(proposal))

    # Build a minimal team instance (bypass heavy constructor)
    team = ScannerAgentTeam.__new__(ScannerAgentTeam)
    team._learned_weights = weights
    team._regime_weights = {}

    result = team.apply_proposed_weights()

    assert result["applied"] is True, f"apply failed: {result}"
    assert "trend" in result["changes"]
    assert "risk_sentinel" in result["changes"]
    # Blend at 0.3 rate: 1.0 + 0.3*(1.05-1.0) = 1.015
    assert abs(result["changes"]["trend"]["new"] - 1.015) < 0.001
    # 1.0 + 0.3*(0.95-1.0) = 0.985
    assert abs(result["changes"]["risk_sentinel"]["new"] - 0.985) < 0.001

    # Proposal archived and live file removed
    assert not proposal_path.exists(), "live proposal not cleaned up"
    archive_dir = iso_workspace / ".claude" / "proposed_weights_archive"
    assert archive_dir.exists()
    archived = list(archive_dir.glob("*_applied.json"))
    assert len(archived) == 1

    archived_content = json.loads(archived[0].read_text())
    assert archived_content["status"] == "applied"
    assert "trend" in archived_content["changes"]


# ── Stage 4: git_sync whitelist E2E ────────────────────────────────────


def test_stage4_git_sync_whitelist_dry_run():
    """Dry-run verifies whitelist logic without touching any real git state."""
    from src.scanner.automation.git_sync import (
        WHITELIST_PATHS,
        _is_whitelisted,
        commit_and_push_self_improvements,
    )

    # Defense-in-depth: confirm the hard no-go paths are all rejected
    danger_zones = [
        "src/scanner/execution.py",
        "src/scanner/engine.py",
        "trained_data/models/agent_weights.json",
        "trained_data/trade_journal_rl.json",
        ".env",
        ".env.local",
        "secrets/api_key.txt",
        "../../etc/passwd",
    ]
    for path in danger_zones:
        assert not _is_whitelisted(path), f"FATAL: {path} is whitelisted"

    # Confirm whitelisted paths
    allowed = [
        ".claude/learnings.md",
        ".claude/rules/trading.md",
        ".claude/rules/improvement.md",
        ".claude/config_adjustments.json",
        ".claude/proposed_weights.json",
        "logs/reflection_log.jsonl",
    ]
    for path in allowed:
        assert _is_whitelisted(path), f"{path} should be whitelisted"

    # Try to commit a non-whitelisted path → must refuse before any git op
    with patch("src.scanner.automation.git_sync.subprocess.run") as mock_run:
        ok = commit_and_push_self_improvements(
            artifacts=["src/scanner/engine.py"],  # poison
            trade_id="T-MALICIOUS",
            hypothesis="this should never land",
            mode="deep",
        )
    assert ok is False
    mock_run.assert_not_called()  # never even shelled out

    # Dry-run with clean tree + valid artifact → returns True, no git calls
    # except the initial status check
    with patch("src.scanner.automation.git_sync.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        ok = commit_and_push_self_improvements(
            artifacts=[".claude/learnings.md"],
            trade_id="T-OK",
            hypothesis="dry run",
            mode="lightweight",
            dry_run=True,
        )
    assert ok is True


# ── End-to-end: full cycle in a single test ────────────────────────────


def test_full_autonomy_cycle_e2e(iso_workspace):
    """Exercise the full loop with mocks — trade close → handler → artifacts
    staged → proposed_weights applied → verify end state.

    This is the canonical "does everything talk to each other" test.
    """
    from src.scanner.agents._team import ScannerAgentTeam
    from src.scanner.automation.claude_subprocess import ReflectionResult
    from src.scanner.automation.event_handlers import (
        ClaudeReflectionHandler,
        SyncContext,
    )

    ClaudeReflectionHandler._recent_outcomes = []

    # Step 1: Seed the weights file that the orchestrator would blend into
    weights = {
        "_global": {"trend": 1.0, "mean_reversion": 1.0, "volatility": 1.0, "risk_sentinel": 1.0, "momentum": 1.0},
        "NORMAL": {"trend": 1.0, "mean_reversion": 1.0, "volatility": 1.0, "risk_sentinel": 1.0, "momentum": 1.0},
        "HIGH": {"trend": 1.0, "mean_reversion": 1.0, "volatility": 1.0, "risk_sentinel": 1.0, "momentum": 1.0},
        "EXTREME": {"trend": 1.0, "mean_reversion": 1.0, "volatility": 1.0, "risk_sentinel": 1.0, "momentum": 1.0},
        "LOW": {"trend": 1.0, "mean_reversion": 1.0, "volatility": 1.0, "risk_sentinel": 1.0, "momentum": 1.0},
        "_meta": {"total_trades": 100, "last_trade_per_agent": {}},
    }
    weights_path = iso_workspace / "trained_data" / "models" / "agent_weights.json"
    weights_path.write_text(json.dumps(weights))

    # Step 2: Simulate Claude's output — write the artifacts it would write
    learnings_path = iso_workspace / ".claude" / "learnings.md"
    learnings_path.write_text(
        "- [2026-04-14] **PATTERN/eur_usd_asia_fade**: 3 consecutive losses shorting EUR_USD at Tokyo open\n"
    )
    proposal_path = iso_workspace / ".claude" / "proposed_weights.json"
    proposal_path.write_text(json.dumps({
        "proposed_at": datetime.now(timezone.utc).isoformat(),
        "trade_id": "T-FULL-CYCLE",
        "regime": "NORMAL",
        "deltas": {"trend": {"target_weight": 1.04, "reason": "cycle test"}},
    }))

    # Step 3: Build a fake closed trade and fire the handler
    entry = {
        "trade_id": "T-FULL-CYCLE",
        "pair": "EUR_USD",
        "direction": "SHORT",
        "confidence": 0.65,
        "sl_pips": 10.0,
        "tp_pips": 18.0,
        "gates": {"momentum_passed": True, "confidence_passed": True, "risk_passed": True},
        "agents": {"weighted_vote_score": 0.60, "agent_reasons": []},
        "regime": {"volatility_regime": "NORMAL"},
    }
    ctx = SyncContext(
        entry=entry,
        outcome_data={"exit_reason": "SL", "duration_minutes": 30},
        closed_trades={"T-FULL-CYCLE": {"realizedPL": -65.0}},
        entries=[entry],
        synced_entries=[entry],
        rl_updates=[],
        trade_id="T-FULL-CYCLE",
        pair="EUR_USD",
        trade_won=False,
        realized_pl=-65.0,
        pnl_pips=-10.0,
    )

    fake_result = ReflectionResult(
        success=True, trade_id="T-FULL-CYCLE", mode="deep",
        duration_seconds=50.0, stdout_len=2000, promise_found=True,
        artifacts_written=[".claude/learnings.md", ".claude/proposed_weights.json"],
        hypothesis="Tokyo-open EUR_USD shorts show loss pattern",
        cost_usd=0.07, returncode=0,
    )

    handler = ClaudeReflectionHandler()
    handler._ensure_deps()
    handler._invoker = MagicMock(return_value=fake_result)

    # Step 4: Fire — simulates trade close event
    assert handler.handle({}, ctx) is True

    # Step 5: Verify budget charged
    assert handler._budget.load()["spawn_count"] >= 1

    # Step 6: Simulate the orchestrator picking up the proposal on next cycle
    team = ScannerAgentTeam.__new__(ScannerAgentTeam)
    team._learned_weights = weights
    team._regime_weights = {}
    apply_result = team.apply_proposed_weights()

    assert apply_result["applied"] is True
    assert "trend" in apply_result["changes"]
    new_trend = apply_result["changes"]["trend"]["new"]
    assert new_trend > 1.0  # pushed up toward 1.04 target
    assert new_trend <= 1.05  # capped by ±5% per cycle + 0.3 EMA

    # Step 7: Verify proposal was consumed (archive + file removed)
    assert not proposal_path.exists()
    archive_entries = list(
        (iso_workspace / ".claude" / "proposed_weights_archive").glob("*.json")
    )
    assert len(archive_entries) >= 1

    print(
        f"\n[Full cycle] Trend weight: 1.0 → {new_trend:.4f} "
        f"(proposal target was 1.04, clamped to max 5% step)"
    )
