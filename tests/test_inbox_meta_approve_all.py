"""Mythos audit 2026-04-30 — Inbox approve-all handler for meta packages.

Operator-reported bug: "approve all button doesn't work in inbox" when
the [🧠 Meta] filter was active. Disk diagnosis:

  * inbox_screen.py:1249 action_approve_all had two loops:
    - homework: iterates _current_rows where entry_type=="homework"
    - adjustments: AdjustmentApprover.approve_all() (filter-blind)
  * The [Meta] filter (commit 18a273f) added entry_type="meta" rows
    but action_approve_all has NO branch for them.
  * Pressing approve-all with [Meta] filter active → silent no-op.

Plus a deeper issue: even if a meta package were approved (moved
pending → approved by ApprovalQueue.approve), nothing in the TUI
runtime calls MetaManager.drain() to actuate the package. Only the
standalone scripts/cybernetic_promote.py drains. So packages sit in
.claude/approval_queue/approved/ until... never.

This commit's fix:
  1. Add _approve_meta_packages helper that iterates visible meta
     rows and calls ApprovalQueue.approve(change_id) for each.
  2. After approving, call MetaManager.drain() inline so packages
     advance to DEPLOYED_SHADOW immediately. The operator sees the
     state change in the inbox without waiting for a separate drain
     trigger.
  3. action_approve_all rolls the meta count into its summary notification.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class _FakeRow:
    entry_type: str
    raw_entry: dict


def _make_inbox(meta_rows: list):
    """Build a bare InboxScreen instance for unit testing the helper.
    Bypasses Textual lifecycle entirely; we test the helper as a
    plain method."""
    from src.tui.screens.inbox_screen import InboxScreen
    inbox = InboxScreen.__new__(InboxScreen)
    inbox._current_rows = meta_rows
    inbox.notify = MagicMock()
    return inbox


def test_no_meta_rows_returns_zero_zero():
    inbox = _make_inbox([])
    result = inbox._approve_meta_packages()
    assert result == (0, 0)


def test_single_meta_row_approved():
    """Happy path: one meta row → ApprovalQueue.approve called once
    with the right change_id, returns (1, 0)."""
    rows = [_FakeRow("meta", {"change_id": "abcdef123456"})]
    inbox = _make_inbox(rows)

    fake_pkg = MagicMock()
    with patch("src.scanner.automation.approval_queue.ApprovalQueue") as MockQ, \
         patch("src.scanner.automation.meta_manager.is_enabled", return_value=False):
        MockQ.return_value.approve.return_value = fake_pkg
        approved, failed = inbox._approve_meta_packages()

    assert (approved, failed) == (1, 0)
    MockQ.return_value.approve.assert_called_once()
    args, kwargs = MockQ.return_value.approve.call_args
    assert args[0] == "abcdef123456"


def test_multiple_meta_rows_all_approved():
    rows = [
        _FakeRow("meta", {"change_id": "aa1"}),
        _FakeRow("meta", {"change_id": "bb2"}),
        _FakeRow("meta", {"change_id": "cc3"}),
    ]
    inbox = _make_inbox(rows)

    with patch("src.scanner.automation.approval_queue.ApprovalQueue") as MockQ, \
         patch("src.scanner.automation.meta_manager.is_enabled", return_value=False):
        MockQ.return_value.approve.return_value = MagicMock()
        approved, failed = inbox._approve_meta_packages()

    assert (approved, failed) == (3, 0)
    assert MockQ.return_value.approve.call_count == 3


def test_skips_homework_and_adjustment_rows():
    """Only entry_type=='meta' rows trigger the approval call."""
    rows = [
        _FakeRow("homework", {"trade_id": "1234"}),
        _FakeRow("adjustment", {"key": "min_confidence"}),
        _FakeRow("meta", {"change_id": "the_only_meta"}),
    ]
    inbox = _make_inbox(rows)

    with patch("src.scanner.automation.approval_queue.ApprovalQueue") as MockQ, \
         patch("src.scanner.automation.meta_manager.is_enabled", return_value=False):
        MockQ.return_value.approve.return_value = MagicMock()
        approved, failed = inbox._approve_meta_packages()

    assert (approved, failed) == (1, 0)
    args, _ = MockQ.return_value.approve.call_args
    assert args[0] == "the_only_meta"


def test_missing_change_id_counted_as_failed():
    """A meta row with no change_id (corrupt blob) must be counted
    as failed, not silently skipped."""
    rows = [
        _FakeRow("meta", {}),  # no change_id
        _FakeRow("meta", {"change_id": "good"}),
    ]
    inbox = _make_inbox(rows)

    with patch("src.scanner.automation.approval_queue.ApprovalQueue") as MockQ, \
         patch("src.scanner.automation.meta_manager.is_enabled", return_value=False):
        MockQ.return_value.approve.return_value = MagicMock()
        approved, failed = inbox._approve_meta_packages()

    assert approved == 1
    assert failed == 1


def test_approve_returns_none_counts_as_failed():
    """ApprovalQueue.approve returns None when pending file is missing
    (already approved or orphan). Must count as failed so the operator
    notification is honest about partial success."""
    rows = [_FakeRow("meta", {"change_id": "stale"})]
    inbox = _make_inbox(rows)

    with patch("src.scanner.automation.approval_queue.ApprovalQueue") as MockQ, \
         patch("src.scanner.automation.meta_manager.is_enabled", return_value=False):
        MockQ.return_value.approve.return_value = None  # missing pending
        approved, failed = inbox._approve_meta_packages()

    assert (approved, failed) == (0, 1)


def test_approve_exception_counts_as_failed_does_not_raise():
    rows = [
        _FakeRow("meta", {"change_id": "boom"}),
        _FakeRow("meta", {"change_id": "ok"}),
    ]
    inbox = _make_inbox(rows)

    def fake_approve(change_id, **kw):
        if change_id == "boom":
            raise RuntimeError("approval store down")
        return MagicMock()

    with patch("src.scanner.automation.approval_queue.ApprovalQueue") as MockQ, \
         patch("src.scanner.automation.meta_manager.is_enabled", return_value=False):
        MockQ.return_value.approve.side_effect = fake_approve
        approved, failed = inbox._approve_meta_packages()

    assert (approved, failed) == (1, 1)


def test_drain_called_when_meta_enabled():
    """When meta is enabled, drain runs inline so the operator sees
    DEPLOYED_SHADOW state immediately, not after a separate trigger."""
    rows = [_FakeRow("meta", {"change_id": "trigger"})]
    inbox = _make_inbox(rows)

    drain_calls = []
    with patch("src.scanner.automation.approval_queue.ApprovalQueue") as MockQ, \
         patch("src.scanner.automation.meta_manager.is_enabled", return_value=True):
        MockQ.return_value.approve.return_value = MagicMock()
        # Mock the singleton + its drain method
        from src.scanner.automation import meta_manager as _mm
        original_singleton = _mm._PRODUCTION_MGR
        try:
            _mm._PRODUCTION_MGR = MagicMock()
            _mm._PRODUCTION_MGR.drain.return_value = {"approved_advanced": 1}
            inbox._approve_meta_packages()
            drain_calls = _mm._PRODUCTION_MGR.drain.call_args_list
        finally:
            _mm._PRODUCTION_MGR = original_singleton

    assert len(drain_calls) == 1


def test_drain_skipped_when_meta_disabled():
    """When BUDDY_META_MANAGER_ENABLED is off, don't call drain — the
    operator opted out of the pipeline entirely."""
    rows = [_FakeRow("meta", {"change_id": "x"})]
    inbox = _make_inbox(rows)

    with patch("src.scanner.automation.approval_queue.ApprovalQueue") as MockQ, \
         patch("src.scanner.automation.meta_manager.is_enabled", return_value=False):
        MockQ.return_value.approve.return_value = MagicMock()
        from src.scanner.automation import meta_manager as _mm
        original_singleton = _mm._PRODUCTION_MGR
        try:
            _mm._PRODUCTION_MGR = MagicMock()
            inbox._approve_meta_packages()
            drain_called = _mm._PRODUCTION_MGR.drain.call_count
        finally:
            _mm._PRODUCTION_MGR = original_singleton

    assert drain_called == 0


def test_drain_failure_does_not_raise_or_invalidate_approvals():
    """If drain raises, the approval count should still reflect what
    succeeded — operator sees "N approved, but deploy didn't trigger
    yet" rather than "0 approved, error" which would be misleading."""
    rows = [_FakeRow("meta", {"change_id": "deploy_breaks"})]
    inbox = _make_inbox(rows)

    with patch("src.scanner.automation.approval_queue.ApprovalQueue") as MockQ, \
         patch("src.scanner.automation.meta_manager.is_enabled", return_value=True):
        MockQ.return_value.approve.return_value = MagicMock()
        from src.scanner.automation import meta_manager as _mm
        original_singleton = _mm._PRODUCTION_MGR
        try:
            _mm._PRODUCTION_MGR = MagicMock()
            _mm._PRODUCTION_MGR.drain.side_effect = RuntimeError("deploy down")
            # Must not raise.
            approved, failed = inbox._approve_meta_packages()
        finally:
            _mm._PRODUCTION_MGR = original_singleton

    assert approved == 1
    assert failed == 0
