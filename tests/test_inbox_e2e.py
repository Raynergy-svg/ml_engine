"""
Integration test: InboxScreen (US-509) — Adjustment Inbox.

Seeds pending_adjustments.json with 3 proposals:
  1. valid pending — min_confidence = 55.0
  2. orphan key — unknown_field (invalid, field not in ScannerConfig)
  3. out-of-bounds — atr_sl_multiplier = 99.9 (max is 3.0)

Asserts:
- Table shows 3 rows with correct validation display
- Approve action on the valid proposal writes to config_adjustments.json
- Invalid proposals have error_message in their validation dict
"""
from __future__ import annotations

import json
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proposal(
    key: str,
    value,
    source: str = "test",
    reason: str = "test reason",
    status: str = "pending",
    valid: bool = True,
    error_message: str = "",
    field_exists: bool = True,
    type_matches: bool = True,
    within_bounds: bool = True,
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "key": key,
        "current_value": None,
        "proposed_value": value,
        "reason": reason,
        "source": source,
        "validation": {
            "valid": valid,
            "field_exists": field_exists,
            "type_matches": type_matches,
            "within_bounds": within_bounds,
            "error_message": error_message,
        },
        "status": status,
        "snooze_until": None,
    }


def _seed_proposals(pending_path: Path) -> tuple[str, str, str]:
    """Write 3 proposals to pending_adjustments.json. Returns (valid_id, orphan_id, oob_id)."""
    valid = _make_proposal(
        key="min_confidence",
        value=55.0,
        source="threshold_optimizer",
        reason="win rate below target",
        valid=True,
    )
    orphan = _make_proposal(
        key="unknown_field_xyz",
        value=0.5,
        source="drift_monitor",
        reason="drift detected",
        valid=False,
        field_exists=False,
        type_matches=False,
        within_bounds=False,
        error_message="Unknown field 'unknown_field_xyz' — not in ScannerConfig.",
    )
    out_of_bounds = _make_proposal(
        key="atr_sl_multiplier",
        value=99.9,
        source="adaptive_lr",
        reason="aggressive multiplier test",
        valid=False,
        field_exists=True,
        type_matches=True,
        within_bounds=False,
        error_message="Value 99.9 for 'atr_sl_multiplier' out of safe bounds [0.3, 3.0]",
    )
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(json.dumps({"proposals": [valid, orphan, out_of_bounds]}, indent=2))
    return valid["id"], orphan["id"], out_of_bounds["id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInboxProposalLoading:
    """Unit-level tests — directly exercise InboxScreen without full TUI."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pending_path = Path(self.tmpdir) / ".claude" / "pending_adjustments.json"
        self.approved_path = Path(self.tmpdir) / ".claude" / "config_adjustments.json"
        self.valid_id, self.orphan_id, self.oob_id = _seed_proposals(self.pending_path)

    def test_reads_three_proposals(self):
        """InboxScreen loads all 3 seeded proposals."""
        from src.tui.screens.inbox_screen import InboxScreen
        screen = InboxScreen.__new__(InboxScreen)
        screen._pending_path = self.pending_path
        screen._approved_path = self.approved_path
        proposals = screen._read_proposals()
        assert len(proposals) == 3

    def test_valid_proposal_has_correct_validation(self):
        """The valid proposal has validation.valid == True."""
        from src.tui.screens.inbox_screen import InboxScreen
        screen = InboxScreen.__new__(InboxScreen)
        screen._pending_path = self.pending_path
        proposals = screen._read_proposals()
        valid_proposals = [p for p in proposals if p["id"] == self.valid_id]
        assert len(valid_proposals) == 1
        assert valid_proposals[0]["validation"]["valid"] is True

    def test_orphan_proposal_has_invalid_validation(self):
        """The orphan-key proposal has validation.valid == False and field_exists == False."""
        from src.tui.screens.inbox_screen import InboxScreen
        screen = InboxScreen.__new__(InboxScreen)
        screen._pending_path = self.pending_path
        proposals = screen._read_proposals()
        orphan = next(p for p in proposals if p["id"] == self.orphan_id)
        assert orphan["validation"]["valid"] is False
        assert orphan["validation"]["field_exists"] is False
        assert "unknown_field_xyz" in orphan["validation"]["error_message"]

    def test_out_of_bounds_proposal_fails_bounds_check(self):
        """The out-of-bounds proposal has validation.valid == False and within_bounds == False."""
        from src.tui.screens.inbox_screen import InboxScreen
        screen = InboxScreen.__new__(InboxScreen)
        screen._pending_path = self.pending_path
        proposals = screen._read_proposals()
        oob = next(p for p in proposals if p["id"] == self.oob_id)
        assert oob["validation"]["valid"] is False
        assert oob["validation"]["within_bounds"] is False
        assert "99.9" in oob["validation"]["error_message"]

    def test_approve_valid_proposal_writes_to_approved(self):
        """Approving the valid proposal writes an entry to config_adjustments.json."""
        from src.scanner.automation.adjustment_approver import AdjustmentApprover

        approver = AdjustmentApprover(
            pending_path=self.pending_path,
            approved_path=self.approved_path,
        )
        result = approver.approve(self.valid_id)
        assert result is True

        # Verify config_adjustments.json was written
        assert self.approved_path.exists()
        data = json.loads(self.approved_path.read_text())
        history = data.get("history", [])
        assert any(e.get("proposal_id") == self.valid_id for e in history), (
            f"Expected proposal_id {self.valid_id} in approved history"
        )

    def test_approve_invalid_proposal_rejected(self):
        """Approving an invalid proposal (orphan key) returns False and does not write to approved."""
        from src.scanner.automation.adjustment_approver import AdjustmentApprover

        approver = AdjustmentApprover(
            pending_path=self.pending_path,
            approved_path=self.approved_path,
        )
        result = approver.approve(self.orphan_id)
        assert result is False

        # config_adjustments.json should not exist or have no history for orphan
        if self.approved_path.exists():
            data = json.loads(self.approved_path.read_text())
            history = data.get("history", [])
            assert not any(e.get("proposal_id") == self.orphan_id for e in history)

    def test_reject_proposal_updates_status(self):
        """Rejecting a proposal sets status='rejected' in pending_adjustments.json."""
        from src.scanner.automation.adjustment_approver import AdjustmentApprover

        approver = AdjustmentApprover(
            pending_path=self.pending_path,
            approved_path=self.approved_path,
        )
        ok = approver.reject(self.valid_id, "operator says no")
        assert ok is True

        data = json.loads(self.pending_path.read_text())
        proposal = next(p for p in data["proposals"] if p["id"] == self.valid_id)
        assert proposal["status"] == "rejected"
        assert proposal["rejection_reason"] == "operator says no"

    def test_snooze_proposal_sets_snooze_until(self):
        """Snoozing a proposal sets status='snoozed' and snooze_until 24h from now."""
        from src.scanner.automation.adjustment_approver import AdjustmentApprover
        from datetime import datetime, timezone

        approver = AdjustmentApprover(
            pending_path=self.pending_path,
            approved_path=self.approved_path,
        )
        ok = approver.snooze(self.valid_id, hours=24)
        assert ok is True

        data = json.loads(self.pending_path.read_text())
        proposal = next(p for p in data["proposals"] if p["id"] == self.valid_id)
        assert proposal["status"] == "snoozed"
        assert proposal["snooze_until"] is not None
        # Should be roughly 24h from now
        snooze_dt = datetime.fromisoformat(proposal["snooze_until"])
        now = datetime.now(timezone.utc)
        delta_hours = (snooze_dt - now).total_seconds() / 3600
        assert 23.9 < delta_hours < 24.1, f"Expected ~24h snooze, got {delta_hours:.2f}h"

    def test_approve_all_approves_valid_and_skips_invalid(self):
        """Bulk approve moves valid proposals only; invalid rows stay actionable."""
        from src.scanner.automation.adjustment_approver import AdjustmentApprover

        approver = AdjustmentApprover(
            pending_path=self.pending_path,
            approved_path=self.approved_path,
        )
        result = approver.approve_all()

        assert result["approved"] == 1
        assert result["skipped"] == 2
        assert result["failed"] == []

        pending_data = json.loads(self.pending_path.read_text())
        statuses = {p["id"]: p["status"] for p in pending_data["proposals"]}
        assert statuses[self.valid_id] == "approved"
        assert statuses[self.orphan_id] == "invalid"
        assert statuses[self.oob_id] == "invalid"

        approved_data = json.loads(self.approved_path.read_text())
        history = approved_data.get("history", [])
        assert [e.get("proposal_id") for e in history] == [self.valid_id]

    def test_reject_all_rejects_every_actionable_proposal(self):
        """Bulk reject clears valid and invalid rows from the active inbox."""
        from src.scanner.automation.adjustment_approver import AdjustmentApprover

        approver = AdjustmentApprover(
            pending_path=self.pending_path,
            approved_path=self.approved_path,
        )
        result = approver.reject_all("operator bulk reject")

        assert result["rejected"] == 3
        assert result["failed"] == []

        pending_data = json.loads(self.pending_path.read_text())
        assert {p["status"] for p in pending_data["proposals"]} == {"rejected"}
        assert all(
            p.get("rejection_reason") == "operator bulk reject"
            for p in pending_data["proposals"]
        )

    def test_validation_column_invalid_marker(self):
        """InboxScreen._read_proposals returns proposals with correct validation.valid flags."""
        from src.tui.screens.inbox_screen import InboxScreen
        screen = InboxScreen.__new__(InboxScreen)
        screen._pending_path = self.pending_path
        proposals = screen._read_proposals()

        valid_count = sum(1 for p in proposals if p["validation"]["valid"])
        invalid_count = sum(1 for p in proposals if not p["validation"]["valid"])
        assert valid_count == 1, f"Expected 1 valid proposal, got {valid_count}"
        assert invalid_count == 2, f"Expected 2 invalid proposals, got {invalid_count}"

    def test_preview_diff_includes_valid_pending_only(self):
        """PreviewModal._build_diff lists only the valid pending proposal."""
        from src.tui.screens.inbox_screen import PreviewModal

        # Load all 3 proposals
        data = json.loads(self.pending_path.read_text())
        proposals = data["proposals"]

        modal = PreviewModal.__new__(PreviewModal)
        modal._proposals = proposals
        lines = modal._build_diff()

        combined = "\n".join(lines)
        # Should include the valid field
        assert "min_confidence" in combined
        # Should NOT include the invalid fields
        assert "unknown_field_xyz" not in combined
        assert "99.9" not in combined or "atr_sl_multiplier" not in combined


class TestInboxE2E:
    """End-to-end: seed proposals, approve valid one, verify config_adjustments.json."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pending_path = Path(self.tmpdir) / ".claude" / "pending_adjustments.json"
        self.approved_path = Path(self.tmpdir) / ".claude" / "config_adjustments.json"
        self.valid_id, self.orphan_id, self.oob_id = _seed_proposals(self.pending_path)

    def test_full_approval_flow(self):
        """Seed 3 proposals → approve valid one → appears in config_adjustments.json."""
        from src.scanner.automation.adjustment_approver import AdjustmentApprover

        # Verify seeded correctly
        data = json.loads(self.pending_path.read_text())
        assert len(data["proposals"]) == 3

        # Approve the valid proposal
        approver = AdjustmentApprover(
            pending_path=self.pending_path,
            approved_path=self.approved_path,
        )
        ok = approver.approve(self.valid_id)
        assert ok is True, "Approval should succeed for valid proposal"

        # config_adjustments.json must exist and contain the approved entry
        assert self.approved_path.exists(), "config_adjustments.json must be created on approve"
        approved_data = json.loads(self.approved_path.read_text())
        history = approved_data.get("history", [])
        ids_in_history = [e.get("proposal_id") for e in history]
        assert self.valid_id in ids_in_history, (
            f"valid_id {self.valid_id} not in approved history: {ids_in_history}"
        )

        # The valid proposal's key must match
        entry = next(e for e in history if e.get("proposal_id") == self.valid_id)
        assert entry["key"] == "min_confidence"
        assert entry["new_value"] == 55.0

        # Invalid proposals must NOT be in history
        assert self.orphan_id not in ids_in_history
        assert self.oob_id not in ids_in_history

    def test_invalid_proposals_cannot_be_approved(self):
        """Both invalid proposals fail approval and stay in pending."""
        from src.scanner.automation.adjustment_approver import AdjustmentApprover

        approver = AdjustmentApprover(
            pending_path=self.pending_path,
            approved_path=self.approved_path,
        )

        # Orphan key — fails re-validation in approver
        result_orphan = approver.approve(self.orphan_id)
        assert result_orphan is False

        # Out-of-bounds — fails re-validation in approver
        result_oob = approver.approve(self.oob_id)
        assert result_oob is False

        # Neither should appear in approved store
        if self.approved_path.exists():
            data = json.loads(self.approved_path.read_text())
            history = data.get("history", [])
            ids = [e.get("proposal_id") for e in history]
            assert self.orphan_id not in ids
            assert self.oob_id not in ids

    def test_table_row_count_matches_proposals(self):
        """All 3 seeded proposals are loaded as active rows (status=pending or invalid)."""
        from src.tui.screens.inbox_screen import InboxScreen

        screen = InboxScreen.__new__(InboxScreen)
        screen._pending_path = self.pending_path
        screen._approved_path = self.approved_path
        proposals = screen._read_proposals()

        assert len(proposals) == 3, f"Expected 3 rows, got {len(proposals)}"
        keys = [p["key"] for p in proposals]
        assert "min_confidence" in keys
        assert "unknown_field_xyz" in keys
        assert "atr_sl_multiplier" in keys
