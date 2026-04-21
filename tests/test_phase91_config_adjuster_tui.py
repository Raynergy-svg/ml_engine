"""Tests for ConfigAdjuster US-508 approval flow and SELFHEAL-C1 regression.

US-508 changed the flow:
  collect_adjustment() → pending_adjustments.json (requires operator approval)
  AdjustmentApprover.approve() → config_adjustments.json  ["history"]
  apply_adjustments() → reads config_adjustments.json["history"], applies {key, new_value}

Key invariants:
  - self._pending is the BYPASS-DETECTION guard, not an apply queue — always starts empty
  - apply_adjustments() raises BypassAttempt if self._pending is non-empty
  - Approved entries live in persistence_path["history"] as {proposal_id, key, new_value, ...}

SELFHEAL-C1 regression (2026-04-16): _load_state() must NOT restore _pending —
the fix confirmed this is intentional in US-508. The regression guard is now:
"apply_adjustments() raises BypassAttempt if _pending is injected directly."
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from src.scanner.automation.config_adjuster import BypassAttempt, ConfigAdjuster
from src.scanner.config import ScannerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approved_history_file(path: Path, entries: list[dict]) -> None:
    """Write a config_adjustments.json containing pre-approved history entries."""
    data = {
        "version": 2,
        "history": entries,
        "last_applied": {},
        "applied_ids": [],
        "pending": {},
    }
    path.write_text(json.dumps(data, indent=2))


def _approved_entry(key: str, new_value: float, source: str = "test") -> dict:
    """Minimal approved-history entry that apply_adjustments() can consume."""
    return {
        "proposal_id": str(uuid.uuid4()),
        "key": key,
        "new_value": new_value,
        "source": source,
        "reason": "round-trip test",
    }


# ---------------------------------------------------------------------------
# Round-trip: approved history → apply_adjustments → config updated
# ---------------------------------------------------------------------------

class TestConfigAdjusterRoundTrip:
    def test_apply_adjustments_updates_config_fields(self, tmp_path: Path) -> None:
        """Approved history entries are applied to ScannerConfig attributes."""
        adj_file = tmp_path / "config_adjustments.json"
        _approved_history_file(adj_file, [
            _approved_entry("max_model_disagreement", 0.25),
            _approved_entry("min_confidence", 62.0),  # 0-100 scale per ScannerConfig
        ])

        adjuster = ConfigAdjuster(persistence_path=adj_file)
        config = ScannerConfig()

        # cycle=999 bypasses the RATE_LIMIT_CYCLES=10 guard
        adjuster.apply_adjustments(config, current_cycle=999)

        assert config.max_model_disagreement == pytest.approx(0.25)
        assert config.min_confidence == pytest.approx(62.0)

    def test_applied_entries_appear_in_history(self, tmp_path: Path) -> None:
        """After apply, applied keys appear in _history and _applied_ids."""
        adj_file = tmp_path / "config_adjustments.json"
        _approved_history_file(adj_file, [
            _approved_entry("max_model_disagreement", 0.25),
        ])

        adjuster = ConfigAdjuster(persistence_path=adj_file)
        config = ScannerConfig()
        adjuster.apply_adjustments(config, current_cycle=999)

        applied_keys = {r["key"] for r in adjuster._history}
        assert "max_model_disagreement" in applied_keys
        assert len(adjuster._applied_ids) >= 1

    def test_already_applied_id_not_reapplied(self, tmp_path: Path) -> None:
        """An entry whose proposal_id is already in _applied_ids is skipped."""
        proposal_id = str(uuid.uuid4())
        adj_file = tmp_path / "config_adjustments.json"
        entry = {
            "proposal_id": proposal_id,
            "key": "max_model_disagreement",
            "new_value": 0.20,
            "source": "test",
            "reason": "dedup test",
        }
        _approved_history_file(adj_file, [entry])

        adjuster = ConfigAdjuster(persistence_path=adj_file)
        # Simulate the ID already being applied
        adjuster._applied_ids.add(proposal_id)
        config = ScannerConfig()
        adjuster.apply_adjustments(config, current_cycle=999)

        # Value must NOT have changed — entry was skipped
        assert config.max_model_disagreement != pytest.approx(0.20)

    def test_fresh_instance_applies_from_approved_file(self, tmp_path: Path) -> None:
        """Round-trip: write approved file → fresh instance → apply → config updated."""
        adj_file = tmp_path / "config_adjustments.json"
        _approved_history_file(adj_file, [
            _approved_entry("max_model_disagreement", 0.22),
        ])

        # Fresh instance — _load_state() reads last_applied / applied_ids from disk
        adjuster2 = ConfigAdjuster(persistence_path=adj_file)
        config = ScannerConfig()
        adjuster2.apply_adjustments(config, current_cycle=999)

        assert config.max_model_disagreement == pytest.approx(0.22)


# ---------------------------------------------------------------------------
# SELFHEAL-C1 regression — bypass-detection guard
# ---------------------------------------------------------------------------

class TestBypassDetectionGuard:
    def test_pending_not_restored_from_disk(self, tmp_path: Path) -> None:
        """_load_state() must NOT restore _pending — it must always start empty.

        US-508 intentional design: self._pending is a bypass-detection guard,
        not an apply queue. Loading it from disk would defeat the guard.
        """
        adj_file = tmp_path / "config_adjustments.json"
        # Write a file that has data in "pending" key (old format or attacker injection)
        data = {
            "version": 2,
            "history": [],
            "pending": {"max_model_disagreement": {"value": 0.10, "source": "injected"}},
            "last_applied": {},
            "applied_ids": [],
        }
        adj_file.write_text(json.dumps(data))

        adjuster = ConfigAdjuster(persistence_path=adj_file)

        # _pending must be empty regardless of what was on disk
        assert adjuster._pending == {}, (
            "_pending must not be restored from disk — it is a bypass-detection guard"
        )

    def test_apply_raises_bypass_attempt_when_pending_injected(self, tmp_path: Path) -> None:
        """apply_adjustments() raises BypassAttempt if self._pending is non-empty.

        This is the SELFHEAL-C1 regression: the old path allowed direct writes
        to self._pending that bypassed the approval flow. BypassAttempt makes
        this visible and auditable.
        """
        adj_file = tmp_path / "config_adjustments.json"
        _approved_history_file(adj_file, [])

        adjuster = ConfigAdjuster(persistence_path=adj_file)
        # Simulate old-path bypass: inject directly into _pending
        adjuster._pending["max_model_disagreement"] = {
            "source": "bypass_attempt",
            "value": 0.10,
            "reason": "injected without approval",
        }
        config = ScannerConfig()

        with pytest.raises(BypassAttempt):
            adjuster.apply_adjustments(config, current_cycle=999)
