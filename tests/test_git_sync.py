"""Unit tests for git_sync.py — autonomous self-improvement push path.

Every git command is mocked via patching subprocess.run. No real git
operations are performed during tests, which keeps them hermetic and fast.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scanner.automation.git_sync import (
    WHITELIST_PATHS,
    _is_whitelisted,
    commit_and_push_self_improvements,
    get_current_branch,
    has_disallowed_changes,
    pull_latest,
)


# ── Whitelist ──────────────────────────────────────────────────────────


def test_whitelist_allows_exact_files():
    assert _is_whitelisted(".claude/learnings.md") is True
    assert _is_whitelisted(".claude/config_adjustments.json") is True
    assert _is_whitelisted(".claude/proposed_weights.json") is True
    assert _is_whitelisted(".claude/state.json") is True


def test_whitelist_allows_directory_prefix():
    assert _is_whitelisted(".claude/rules/trading.md") is True
    assert _is_whitelisted(".claude/rules/improvement.md") is True
    assert _is_whitelisted(".claude/brain/briefing.md") is True
    assert _is_whitelisted("logs/reflection_log.jsonl") is True


def test_whitelist_rejects_source_code():
    """Critical safety test: src/ paths MUST NOT be allowed."""
    assert _is_whitelisted("src/scanner/engine.py") is False
    assert _is_whitelisted("src/scanner/execution.py") is False
    assert _is_whitelisted("trained_data/models/agent_weights.json") is False
    assert _is_whitelisted("trained_data/trade_journal_rl.json") is False
    assert _is_whitelisted(".env") is False
    assert _is_whitelisted(".env.local") is False
    assert _is_whitelisted("CLAUDE.md") is False  # Not in whitelist


# ── Helpers ────────────────────────────────────────────────────────────


def _fake_proc(stdout="", stderr="", returncode=0):
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


def test_get_current_branch_returns_name():
    with patch("src.scanner.automation.git_sync.subprocess.run") as mock_run:
        mock_run.return_value = _fake_proc(stdout="claude/autonomous-trading-agent-XysP0\n")
        branch = get_current_branch()
    assert branch == "claude/autonomous-trading-agent-XysP0"


def test_get_current_branch_returns_none_on_error():
    with patch("src.scanner.automation.git_sync.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(128, "git")
        assert get_current_branch() is None


def test_has_disallowed_changes_detects_src_modification():
    """When git status shows src/ files modified, we must surface them."""
    fake_status = " M src/scanner/engine.py\n M .claude/learnings.md\n?? src/new_file.py\n"
    with patch("src.scanner.automation.git_sync.subprocess.run") as mock_run:
        mock_run.return_value = _fake_proc(stdout=fake_status)
        disallowed = has_disallowed_changes()
    assert "src/scanner/engine.py" in disallowed
    assert "src/new_file.py" in disallowed
    assert ".claude/learnings.md" not in disallowed  # whitelisted


def test_has_disallowed_changes_empty_when_only_whitelisted():
    fake_status = " M .claude/learnings.md\n M .claude/rules/trading.md\n"
    with patch("src.scanner.automation.git_sync.subprocess.run") as mock_run:
        mock_run.return_value = _fake_proc(stdout=fake_status)
        assert has_disallowed_changes() == []


# ── commit_and_push_self_improvements ──────────────────────────────────


def test_commit_rejects_non_whitelisted_artifacts():
    """If Claude hallucinates a src/ path, we must refuse to stage it."""
    with patch("src.scanner.automation.git_sync.subprocess.run") as mock_run:
        ok = commit_and_push_self_improvements(
            artifacts=["src/scanner/engine.py"],
            trade_id="T-1",
            hypothesis="bad",
            mode="deep",
        )
    assert ok is False
    mock_run.assert_not_called()  # Never even ran git


def test_commit_skips_when_tree_dirty_outside_whitelist():
    with patch("src.scanner.automation.git_sync.subprocess.run") as mock_run:
        # status output shows a src/ modification alongside our artifact
        mock_run.side_effect = [
            _fake_proc(stdout=" M src/scanner/execution.py\n M .claude/learnings.md\n"),
        ]
        ok = commit_and_push_self_improvements(
            artifacts=[".claude/learnings.md"],
            trade_id="T-2",
            hypothesis="won't commit src/",
            mode="lightweight",
        )
    assert ok is False


def test_commit_happy_path(tmp_path):
    """Whitelisted artifacts + clean tree + successful push = True."""
    # Create the artifact on disk so Path(path).exists() passes
    (tmp_path / ".claude").mkdir()
    artifact = tmp_path / ".claude" / "learnings.md"
    artifact.write_text("test content\n")

    with patch("src.scanner.automation.git_sync.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _fake_proc(stdout=""),                      # git status (clean)
            _fake_proc(stdout=""),                      # git add
            _fake_proc(stdout=".claude/learnings.md\n"),  # diff --cached (has staged)
            _fake_proc(stdout=""),                      # git commit
            _fake_proc(stdout="main\n"),                # rev-parse (branch)
            _fake_proc(stdout=""),                      # git push
        ]
        ok = commit_and_push_self_improvements(
            artifacts=[".claude/learnings.md"],
            trade_id="T-3",
            hypothesis="happy path",
            mode="lightweight",
            cwd=tmp_path,
        )
    assert ok is True


def test_commit_dry_run_does_not_invoke_git(tmp_path):
    with patch("src.scanner.automation.git_sync.subprocess.run") as mock_run:
        mock_run.return_value = _fake_proc(stdout="")  # only status check
        ok = commit_and_push_self_improvements(
            artifacts=[".claude/learnings.md"],
            trade_id="T-4",
            hypothesis="dry run",
            mode="lightweight",
            dry_run=True,
        )
    assert ok is True
    # Only the initial status check was called, no add/commit/push
    assert mock_run.call_count == 1


def test_commit_handles_push_failure_with_retries(tmp_path):
    """Push failure must trigger retry backoff, then give up gracefully."""
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "learnings.md").write_text("x")

    with patch("src.scanner.automation.git_sync.subprocess.run") as mock_run, \
         patch("src.scanner.automation.git_sync.time.sleep") as mock_sleep:
        # Build a sequence: status, add, diff, commit, branch, then N push failures
        push_err = subprocess.CalledProcessError(1, "git", stderr="fatal: network down")
        mock_run.side_effect = [
            _fake_proc(stdout=""),                      # status
            _fake_proc(stdout=""),                      # add
            _fake_proc(stdout=".claude/learnings.md\n"),  # diff
            _fake_proc(stdout=""),                      # commit
            _fake_proc(stdout="main\n"),                # branch
            push_err, push_err, push_err, push_err, push_err,  # 5 push attempts
        ]
        ok = commit_and_push_self_improvements(
            artifacts=[".claude/learnings.md"],
            trade_id="T-5",
            hypothesis="retry",
            mode="lightweight",
            cwd=tmp_path,
            retries=4,
        )
    assert ok is False
    # 4 retries = 4 sleeps (the last failure has no sleep after it)
    assert mock_sleep.call_count == 4


def test_pull_latest_skips_on_dirty_tree():
    """pull_latest must not run on a dirty tree with src/ changes."""
    with patch("src.scanner.automation.git_sync.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _fake_proc(stdout="main\n"),                                  # branch
            _fake_proc(stdout=" M src/scanner/execution.py\n"),           # status
        ]
        ok = pull_latest()
    # Returned True (skipped is not an error) but never invoked `git pull`
    assert ok is True
    assert mock_run.call_count == 2  # branch + status, no pull


def test_pull_latest_success():
    with patch("src.scanner.automation.git_sync.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _fake_proc(stdout="main\n"),  # branch
            _fake_proc(stdout=""),        # status (clean)
            _fake_proc(stdout=""),        # pull
        ]
        assert pull_latest() is True
