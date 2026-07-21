"""Durable config overlay (P1, 2026-07-03) — no-mock tests.

Covers the headless consumer for approved config adjustments:
- consume: approved history → overlay values + provenance, ids durably marked
  consumed (idempotent on re-run), orphan keys skipped loudly;
- apply: overlay replays onto a real ScannerConfig (restart survival), orphan
  keys never setattr'd;
- status: read-only summary lands in the maintenance report.

Real ConfigAdjuster + real disk under tmp_path. No unittest.mock.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scanner.config import ScannerConfig  # noqa: E402
from src.scanner.automation.config_overlay import (  # noqa: E402
    apply_overlay,
    consume_approved_adjustments,
    overlay_status,
)
from src.scanner.automation.maintenance_report import build_maintenance_report  # noqa: E402

OVERLAY = ".claude/config_overlay.json"


def _seed_approved(root: Path, entries) -> None:
    claude = root / ".claude"
    claude.mkdir(parents=True, exist_ok=True)
    (claude / "config_adjustments.json").write_text(json.dumps({
        "version": 2, "history": entries, "applied_ids": [],
    }))


def test_consume_moves_approved_into_overlay_and_marks_consumed(tmp_path) -> None:
    _seed_approved(tmp_path, [
        {"proposal_id": "p1", "key": "min_confidence", "new_value": 0.48,
         "source": "self_heal", "reason": "r", "timestamp": "2026-07-02T00:00:00+00:00"},
    ])
    out = consume_approved_adjustments(tmp_path)
    assert out["error"] is None
    assert out["consumed"] == ["min_confidence"]

    overlay = json.loads((tmp_path / OVERLAY).read_text())
    assert overlay["values"]["min_confidence"] == 0.48
    assert overlay["provenance"]["min_confidence"]["source"] == "self_heal"

    # ids durably consumed — a second run is a no-op (idempotent).
    out2 = consume_approved_adjustments(tmp_path)
    assert out2["consumed"] == []
    adj = json.loads((tmp_path / ".claude" / "config_adjustments.json").read_text())
    assert "p1" in adj["applied_ids"]


def test_orphan_keys_are_skipped_never_landed(tmp_path) -> None:
    _seed_approved(tmp_path, [
        {"proposal_id": "p2", "key": "not_a_real_field_xyz", "new_value": 1,
         "source": "s", "reason": "r", "timestamp": "2026-07-02T00:00:00+00:00"},
        {"proposal_id": "p3", "key": "min_confidence", "new_value": 0.5,
         "source": "s", "reason": "r", "timestamp": "2026-07-02T00:00:01+00:00"},
    ])
    out = consume_approved_adjustments(tmp_path)
    assert out["orphans_skipped"] == ["not_a_real_field_xyz"]
    assert out["consumed"] == ["min_confidence"]
    overlay = json.loads((tmp_path / OVERLAY).read_text())
    assert "not_a_real_field_xyz" not in overlay["values"]


def test_apply_overlay_survives_restart_and_skips_orphans(tmp_path) -> None:
    (tmp_path / ".claude").mkdir(parents=True, exist_ok=True)
    (tmp_path / OVERLAY).write_text(json.dumps({
        "version": 1, "cycle": 3,
        "values": {"min_confidence": 0.51, "orphan_key_zzz": 9},
        "provenance": {"min_confidence": {"source": "s"}},
    }))
    cfg = ScannerConfig()  # a fresh "restarted process" config with defaults
    assert cfg.min_confidence != 0.51
    applied = apply_overlay(cfg, tmp_path)
    assert cfg.min_confidence == 0.51            # adjustment survived the restart
    assert [a["key"] for a in applied] == ["min_confidence"]
    assert not hasattr(cfg, "orphan_key_zzz") or getattr(cfg, "orphan_key_zzz", None) != 9

    # Re-apply is a no-op (values already equal).
    assert apply_overlay(cfg, tmp_path) == []


def test_missing_or_corrupt_state_degrades_gracefully(tmp_path) -> None:
    # Nothing on disk at all.
    out = consume_approved_adjustments(tmp_path)
    assert out["consumed"] == [] and out["error"] is None
    assert apply_overlay(ScannerConfig(), tmp_path) == []
    assert overlay_status(tmp_path) is None

    # Corrupt overlay file: treated as empty, never raises.
    (tmp_path / ".claude").mkdir(parents=True, exist_ok=True)
    (tmp_path / OVERLAY).write_text("{corrupt")
    assert apply_overlay(ScannerConfig(), tmp_path) == []


def test_protected_field_refused_at_every_layer(tmp_path) -> None:
    """oanda_environment (Hard NO, L-003) must be refused unconditionally:
    at proposal validation, at overlay consumption, and at overlay apply."""
    from src.scanner.automation.adjustment_validator import validate_adjustment

    # Layer 1: the validator refuses the proposal outright.
    v = validate_adjustment("oanda_environment", "live", "malicious or mistaken")
    assert v.valid is False
    assert "PROTECTED" in v.error_message

    # Layers 2+4: a hand-crafted approved entry never lands in the overlay.
    # (Which layer fires first is an implementation detail — ConfigAdjuster's
    # own PROTECTED_FIELDS check now refuses before the overlay's consume-loop
    # guard even sees it. Assert the OUTCOME: refused everywhere, overlay clean.)
    _seed_approved(tmp_path, [
        {"proposal_id": "evil", "key": "oanda_environment", "new_value": "live",
         "source": "x", "reason": "x", "timestamp": "2026-07-02T00:00:00+00:00"},
    ])
    out = consume_approved_adjustments(tmp_path)
    assert out["consumed"] == []
    assert out["error"] is None
    overlay_path = tmp_path / OVERLAY
    if overlay_path.exists():
        assert "oanda_environment" not in json.loads(
            overlay_path.read_text()).get("values", {})

    # Layer 3: a tampered overlay carrying it is refused at apply.
    (tmp_path / ".claude").mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(json.dumps({
        "version": 1, "cycle": 1,
        "values": {"oanda_environment": "live"}, "provenance": {},
    }))
    cfg = ScannerConfig()
    before = cfg.oanda_environment
    assert before == "practice"
    applied = apply_overlay(cfg, tmp_path)
    assert applied == []
    assert cfg.oanda_environment == "practice"


def test_overlay_applies_after_profile_and_wins_overlapping_keys(tmp_path) -> None:
    """The engine-seam ordering contract (operator-approved wiring 2026-07-03):
    apply_profile first, apply_overlay second — the overlay must win on
    overlapping keys, and re-applying the profile afterwards would clobber
    (which is exactly why the engine seam sits AFTER _load_yaml_config)."""
    (tmp_path / ".claude").mkdir(parents=True, exist_ok=True)

    cfg = ScannerConfig()
    cfg.apply_profile("smart")
    profile_value = cfg.min_confidence

    overlay_value = float(profile_value) + 7.0  # guaranteed different
    (tmp_path / OVERLAY).write_text(json.dumps({
        "version": 1, "cycle": 1,
        "values": {"min_confidence": overlay_value},
        "provenance": {"min_confidence": {"source": "test"}},
    }))

    applied = apply_overlay(cfg, tmp_path)
    assert [a["key"] for a in applied] == ["min_confidence"]
    assert cfg.min_confidence == overlay_value          # overlay wins post-profile

    # Documenting the hazard the ordering protects against: a later profile
    # re-apply clobbers the overlay value (so overlay must always come last).
    cfg.apply_profile("smart")
    assert cfg.min_confidence == profile_value


def test_overlay_status_lands_in_maintenance_report(tmp_path) -> None:
    _seed_approved(tmp_path, [
        {"proposal_id": "p4", "key": "min_confidence", "new_value": 0.49,
         "source": "s", "reason": "r", "timestamp": "2026-07-02T00:00:00+00:00"},
    ])
    consume_approved_adjustments(tmp_path)
    report = build_maintenance_report(tmp_path)
    assert report["config_overlay"]["n_values"] == 1
    assert report["config_overlay"]["keys"] == ["min_confidence"]
