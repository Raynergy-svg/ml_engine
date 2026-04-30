"""Mythos audit 2026-04-30 — write-side tripwire on AdjustmentApprover.

Pre-fix forensics from 2026-04-29 production state-loss:
    Working tree had ~200 entries in config_adjustments.json earlier
    in the day. By 12:35:32 EDT the disk file was 5 entries (matching
    git HEAD's 04-16 snapshot — meaning some path overwrote runtime
    state with a shorter version. The architectural enabler:
    `_load_approved` had a bare `except: return {history: []}` that
    silently produced an empty load on any JSON parse failure; the
    next `_save_approved` then wrote 1 entry + truncated against an
    empty list, wiping the rest.

This test pins the new shrinkage tripwire:
  * If a save would write FEWER history entries than what's already
    on disk, the writer aborts.
  * The proposed payload is dumped to a `.recovered` shadow file so
    the operator can reconcile (no data is lost — just relocated).
  * Disk file is left untouched until the operator intervenes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.automation.adjustment_approver import AdjustmentApprover


def _make(approved_path: Path, history: list[dict]) -> dict:
    blob = {
        "version": 1,
        "history": history,
        "pending": {},
        "last_applied": {},
        "last_updated": "2026-04-30T11:00:00+00:00",
        "total_adjustments": len(history),
    }
    approved_path.write_text(json.dumps(blob, indent=2))
    return blob


@pytest.fixture
def approver(tmp_path):
    return AdjustmentApprover(
        pending_path=tmp_path / "pending.json",
        approved_path=tmp_path / "approved.json",
    )


def test_save_passes_when_history_grows(approver):
    """Append-then-save (the normal case) must succeed unchanged."""
    _make(approver.approved_path, [{"key": "k1", "new_value": 1}])
    new_blob = {
        "version": 1,
        "history": [
            {"key": "k1", "new_value": 1},
            {"key": "k2", "new_value": 2},
        ],
        "pending": {},
        "last_applied": {},
    }
    approver._save_approved(new_blob)
    on_disk = json.loads(approver.approved_path.read_text())
    assert len(on_disk["history"]) == 2


def test_save_passes_when_history_unchanged(approver):
    """Same-size rewrite (idempotent re-save) is allowed."""
    history = [{"key": "k1", "new_value": 1}]
    _make(approver.approved_path, history)
    approver._save_approved({"version": 1, "history": history})
    on_disk = json.loads(approver.approved_path.read_text())
    assert len(on_disk["history"]) == 1


def test_save_refuses_to_shrink_history(approver, caplog):
    """The actual production bug: disk has 200 entries, proposed save
    has 1. Must abort + dump proposed to .recovered shadow file."""
    big_history = [{"key": f"k{i}", "new_value": i} for i in range(200)]
    _make(approver.approved_path, big_history)
    on_disk_before = json.loads(approver.approved_path.read_text())

    proposed = {
        "version": 1,
        "history": [{"key": "single", "new_value": 999}],
    }
    with caplog.at_level("CRITICAL"):
        approver._save_approved(proposed)

    # Disk is untouched.
    on_disk_after = json.loads(approver.approved_path.read_text())
    assert on_disk_after == on_disk_before
    assert len(on_disk_after["history"]) == 200

    # Shadow recovery file got the proposed payload.
    recovered = approver.approved_path.with_suffix(".json.recovered")
    assert recovered.exists()
    rec_blob = json.loads(recovered.read_text())
    assert rec_blob["history"][0]["key"] == "single"

    # CRITICAL log was emitted.
    crits = [r for r in caplog.records if r.levelname == "CRITICAL"]
    assert any("REFUSED to shrink" in r.message for r in crits)


def test_save_handles_legacy_list_root_on_disk(approver):
    """Legacy on-disk format is a bare list (not dict). The tripwire
    must still measure its length — refuse if shrinking."""
    legacy_list = [{"key": f"k{i}"} for i in range(50)]
    approver.approved_path.write_text(json.dumps(legacy_list))
    proposed = {"version": 1, "history": [{"key": "tiny"}]}

    approver._save_approved(proposed)

    on_disk = json.loads(approver.approved_path.read_text())
    # Disk should still be the legacy list (untouched).
    assert isinstance(on_disk, list)
    assert len(on_disk) == 50


def test_save_passes_when_disk_is_corrupt(approver):
    """If the on-disk file is unparseable, treat disk_n=0 and let the
    save proceed — better to have valid runtime state than to lock
    forever on corrupt JSON."""
    approver.approved_path.write_text("{not json at all")

    proposed = {"version": 1, "history": [{"key": "k1"}]}
    approver._save_approved(proposed)

    on_disk = json.loads(approver.approved_path.read_text())
    assert len(on_disk["history"]) == 1


def test_save_passes_when_disk_missing(approver):
    """First-ever write (no prior file) must succeed."""
    assert not approver.approved_path.exists()
    proposed = {"version": 1, "history": [{"key": "k1"}]}
    approver._save_approved(proposed)
    on_disk = json.loads(approver.approved_path.read_text())
    assert len(on_disk["history"]) == 1


def test_load_approved_logs_warning_on_corrupt(approver, caplog):
    """The bare-except was replaced with a logged warning. Verify the
    operator gets a visible signal of corruption."""
    approver.approved_path.write_text("{not json")
    with caplog.at_level("WARNING"):
        result = approver._load_approved()
    assert result["history"] == []
    warns = [
        r for r in caplog.records
        if "corrupt/unreadable" in r.message and r.levelname == "WARNING"
    ]
    assert warns, "Expected a WARNING log for the corrupt-file path"


def test_save_refuses_then_real_writer_can_fix(approver):
    """Sequencing: a buggy writer attempts a shrinking save and gets
    refused. A subsequent legitimate writer (with the full history)
    can still succeed — the tripwire shouldn't permanently lock the
    file."""
    big = [{"key": f"k{i}"} for i in range(200)]
    _make(approver.approved_path, big)

    # Buggy writer attempts shrink.
    approver._save_approved({"version": 1, "history": [{"key": "ouch"}]})
    on_disk = json.loads(approver.approved_path.read_text())
    assert len(on_disk["history"]) == 200

    # Legitimate writer with full + 1 new entry succeeds.
    approver._save_approved({
        "version": 1,
        "history": big + [{"key": "k200", "new_value": 200}],
    })
    on_disk = json.loads(approver.approved_path.read_text())
    assert len(on_disk["history"]) == 201
