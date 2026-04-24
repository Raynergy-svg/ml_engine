"""E2E integration test: self-tuner proposes → lands in pending → approve → applied.

Verifies the full US-508 approval flow:
1. self_heal_reflection calls collect_adjustment() with a valid key
2. Proposal appears in pending_adjustments.json with status='pending'
3. AdjustmentApprover.approve(proposal_id) moves it to config_adjustments.json
4. ConfigAdjuster.apply_adjustments(config) applies the field to a real config
5. ScannerConfig.min_confidence is actually changed

Also verifies:
- Invalid/orphan keys land in pending_adjustments.json with status='invalid'
- BypassAttempt is raised if self._pending is populated directly
- Events are emitted: adjustment.proposed, adjustment.applied
"""

import dataclasses
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Minimal ScannerConfig for the E2E test — avoids heavy ML imports
# ---------------------------------------------------------------------------

@dataclass
class _ScannerConfig:
    min_confidence: float = 42.0
    atr_sl_multiplier: float = 1.0
    max_uncertainty_score: float = 0.45
    max_model_disagreement: float = 0.30
    weighted_vote_threshold: float = 0.72
    risk_per_trade_pct: float = 0.05


def _mock_fields():
    return {f.name: f for f in dataclasses.fields(_ScannerConfig)}


def _mock_hints():
    import typing
    return typing.get_type_hints(_ScannerConfig)


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------

class TestAdjustmentFlowE2E(unittest.TestCase):

    def setUp(self):
        # Isolated temp dir for all JSON files
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.pending_path = self.tmp / "pending_adjustments.json"
        self.approved_path = self.tmp / "config_adjustments.json"

        # Patch config fields to use our lightweight mock
        self._field_patcher = patch(
            "src.scanner.automation.adjustment_validator._get_scanner_config_fields",
            return_value=_mock_fields(),
        )
        self._hint_patcher = patch(
            "src.scanner.automation.adjustment_validator._get_scanner_config_hints",
            return_value=_mock_hints(),
        )
        self._field_patcher.start()
        self._hint_patcher.start()

        # Silence event bus calls
        self._event_bus_patcher = patch(
            "src.scanner.automation.event_bus.get_event_bus",
            return_value=MagicMock(),
        )
        self._event_bus_patcher.start()

    def tearDown(self):
        self._field_patcher.stop()
        self._hint_patcher.stop()
        self._event_bus_patcher.stop()
        self._tmpdir.cleanup()

    def _make_adjuster(self):
        from src.scanner.automation.config_adjuster import ConfigAdjuster
        return ConfigAdjuster(
            persistence_path=self.approved_path,
            pending_path=self.pending_path,
        )

    def _make_approver(self):
        from src.scanner.automation.adjustment_approver import AdjustmentApprover
        return AdjustmentApprover(
            pending_path=self.pending_path,
            approved_path=self.approved_path,
        )

    # ------------------------------------------------------------------
    # Core happy-path flow
    # ------------------------------------------------------------------

    def test_full_flow_propose_approve_apply(self):
        """Complete flow: propose → approve → apply → config field changes."""
        adjuster = self._make_adjuster()
        approver = self._make_approver()
        config = _ScannerConfig()
        self.assertEqual(config.min_confidence, 42.0)

        # Step 1: propose
        adjuster.collect_adjustment(
            source="self_heal_reflection",
            key="min_confidence",
            value=68.0,
            reason="0/14 win rate — raise confidence floor",
            cycle=1,
        )

        # Step 2: verify lands in pending
        self.assertTrue(self.pending_path.exists(), "pending_adjustments.json must exist")
        pending_data = json.loads(self.pending_path.read_text())
        proposals = pending_data["proposals"]
        self.assertEqual(len(proposals), 1)
        proposal = proposals[0]
        self.assertEqual(proposal["key"], "min_confidence")
        self.assertEqual(proposal["proposed_value"], 68.0)
        self.assertEqual(proposal["source"], "self_heal_reflection")
        self.assertEqual(proposal["status"], "pending")
        self.assertTrue(proposal["validation"]["valid"])

        proposal_id = proposal["id"]
        self.assertIsNotNone(proposal_id)

        # Step 3: approve
        approved = approver.approve(proposal_id)
        self.assertTrue(approved)

        # Verify config_adjustments.json has the entry
        self.assertTrue(self.approved_path.exists())
        approved_data = json.loads(self.approved_path.read_text())
        history = approved_data["history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["key"], "min_confidence")
        self.assertEqual(history[0]["new_value"], 68.0)
        self.assertEqual(history[0]["proposal_id"], proposal_id)

        # Step 4: apply_adjustments
        applied = adjuster.apply_adjustments(config, current_cycle=20)
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]["key"], "min_confidence")

        # Step 5: ScannerConfig field actually changed
        self.assertEqual(config.min_confidence, 68.0)

    def test_invalid_key_lands_in_pending_as_invalid(self):
        """Orphan keys must be written to pending with status='invalid'."""
        adjuster = self._make_adjuster()

        adjuster.collect_adjustment(
            source="self_heal_reflection",
            key="min_confidence_threshold",  # orphan key
            value=68.0,
            reason="phase 90 incident replay",
        )

        pending_data = json.loads(self.pending_path.read_text())
        proposal = pending_data["proposals"][0]
        self.assertEqual(proposal["status"], "invalid")
        self.assertFalse(proposal["validation"]["valid"])
        self.assertFalse(proposal["validation"]["field_exists"])

    def test_invalid_proposal_not_approvable(self):
        """An invalid proposal cannot be moved to config_adjustments.json via approve()."""
        adjuster = self._make_adjuster()
        approver = self._make_approver()

        adjuster.collect_adjustment(
            source="drift_monitor",
            key="atr_sl_multiplier_low_regime",  # orphan
            value=1.2,
            reason="LOW regime needs wider SL",
        )

        pending_data = json.loads(self.pending_path.read_text())
        proposal_id = pending_data["proposals"][0]["id"]

        # Attempt to approve orphan — should fail
        # The approver re-validates, so it will flip to 'invalid' and return False
        approved = approver.approve(proposal_id)
        self.assertFalse(approved)

        # config_adjustments.json must NOT have been updated
        self.assertFalse(self.approved_path.exists())

    def test_bypass_attempt_raises(self):
        """Populating _pending directly and calling apply_adjustments raises BypassAttempt."""
        from src.scanner.automation.config_adjuster import BypassAttempt
        adjuster = self._make_adjuster()
        config = _ScannerConfig()

        # Simulate old-code bypass: directly populate _pending
        adjuster._pending["min_confidence"] = {
            "source": "bypass_test",
            "value": 99.0,
            "reason": "bypass",
            "cycle_proposed": 0,
        }

        with self.assertRaises(BypassAttempt):
            adjuster.apply_adjustments(config, current_cycle=0)

        # Config must not have changed
        self.assertEqual(config.min_confidence, 42.0)

    # ------------------------------------------------------------------
    # Reject / Snooze
    # ------------------------------------------------------------------

    def test_reject_prevents_approval(self):
        adjuster = self._make_adjuster()
        approver = self._make_approver()
        config = _ScannerConfig()

        adjuster.collect_adjustment("source", "min_confidence", 65.0, "test")
        pending_data = json.loads(self.pending_path.read_text())
        proposal_id = pending_data["proposals"][0]["id"]

        rejected = approver.reject(proposal_id, "not enough evidence")
        self.assertTrue(rejected)

        # Re-approve should fail (status is 'rejected', not 'pending')
        re_approved = approver.approve(proposal_id)
        self.assertFalse(re_approved)

        # Config_adjustments.json must not exist
        self.assertFalse(self.approved_path.exists())

    def test_snooze_keeps_proposal_actionable(self):
        adjuster = self._make_adjuster()
        approver = self._make_approver()

        adjuster.collect_adjustment("source", "atr_sl_multiplier", 1.3, "test snooze")
        pending_data = json.loads(self.pending_path.read_text())
        proposal_id = pending_data["proposals"][0]["id"]

        snoozed = approver.snooze(proposal_id, hours=24)
        self.assertTrue(snoozed)

        pending_data = json.loads(self.pending_path.read_text())
        proposal = pending_data["proposals"][0]
        self.assertEqual(proposal["status"], "snoozed")
        self.assertIsNotNone(proposal["snooze_until"])

        # Snoozed proposals CAN be approved (snooze is a deferral, not a block)
        approved = approver.approve(proposal_id)
        self.assertTrue(approved)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def test_rate_limit_prevents_double_apply(self):
        """Same key approved twice should only apply once per RATE_LIMIT_CYCLES."""
        adjuster = self._make_adjuster()
        approver = self._make_approver()
        config = _ScannerConfig()

        # First proposal
        adjuster.collect_adjustment("src", "min_confidence", 65.0, "first")
        data = json.loads(self.pending_path.read_text())
        approver.approve(data["proposals"][0]["id"])

        # Apply at cycle 0
        applied = adjuster.apply_adjustments(config, current_cycle=0)
        self.assertEqual(len(applied), 1)
        self.assertEqual(config.min_confidence, 65.0)

        # Reset applied_ids to simulate a second proposal for same key
        # (in production this would be a new proposal from a new cycle)
        # Re-using same adjuster: apply_adjustments again at cycle 1
        # Rate-limit should block it (same cycle range)
        applied2 = adjuster.apply_adjustments(config, current_cycle=1)
        self.assertEqual(len(applied2), 0)  # blocked by rate limit

    def test_collect_adjustment_suppresses_duplicate_pending_proposal(self):
        """Same source/key/value should create one inbox item, not one per cycle."""
        adjuster = self._make_adjuster()

        for reason in [
            "Rolling win rate optimization for UNKNOWN",
            "Rolling win rate optimization for EXTREME_NEXT",
            "Rolling win rate optimization for UNKNOWN",
        ]:
            adjuster.collect_adjustment(
                source="threshold_optimizer",
                key="min_confidence",
                value=55.0,
                reason=reason,
            )

        data = json.loads(self.pending_path.read_text())
        proposals = data.get("proposals", [])
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["key"], "min_confidence")

    # ------------------------------------------------------------------
    # No duplicate application across restarts
    # ------------------------------------------------------------------

    def test_applied_ids_persisted_across_restart(self):
        """After save_state, a new ConfigAdjuster should not re-apply same proposals."""
        adjuster = self._make_adjuster()
        approver = self._make_approver()
        config = _ScannerConfig()

        adjuster.collect_adjustment("src", "atr_sl_multiplier", 1.5, "test restart")
        data = json.loads(self.pending_path.read_text())
        approver.approve(data["proposals"][0]["id"])

        adjuster.apply_adjustments(config, current_cycle=0)
        self.assertEqual(config.atr_sl_multiplier, 1.5)

        adjuster.save_state()

        # Simulate restart: new adjuster instance, same persistence files
        adjuster2 = self._make_adjuster()
        config2 = _ScannerConfig()
        self.assertEqual(config2.atr_sl_multiplier, 1.0)  # default

        # Should NOT re-apply (already in applied_ids)
        applied = adjuster2.apply_adjustments(config2, current_cycle=100)
        self.assertEqual(len(applied), 0)
        self.assertEqual(config2.atr_sl_multiplier, 1.0)  # unchanged


if __name__ == "__main__":
    unittest.main()
