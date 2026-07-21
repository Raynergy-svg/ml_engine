"""Verifier-identified two-consumer gap (2026-07-03) — no-mock regression test.

`EmbeddedScanner._reload_config_now` (the TUI's dirty-flag-triggered reload
path) used to call `ConfigAdjuster.apply_adjustments` directly: the approved
value landed on the live in-memory `ScannerConfig` but was never routed
through the durable overlay (`config_overlay.consume_approved_adjustments` +
`apply_overlay`). A restart threw the adjustment away even though
`ConfigAdjuster` had already marked it "applied" (so it would never be
re-offered) — the exact restart-loss defect the overlay was built to close,
left open for every TUI-approved adjustment.

Real ConfigAdjuster + real ScannerConfig + real disk under tmp_path. No
unittest.mock, per .claude/rules/improvement.md No-Mock Rule.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scanner.config import ScannerConfig  # noqa: E402
from src.scanner.automation.config_adjuster import ConfigAdjuster  # noqa: E402
from src.scanner.automation.config_overlay import apply_overlay  # noqa: E402
from src.tui.embedded_scanner import EmbeddedScanner  # noqa: E402

OVERLAY = ".claude/config_overlay.json"
ADJUSTMENTS = ".claude/config_adjustments.json"


def _seed_approved(root: Path, entries) -> None:
    claude = root / ".claude"
    claude.mkdir(parents=True, exist_ok=True)
    (claude / "config_adjustments.json").write_text(json.dumps({
        "version": 2, "history": entries, "applied_ids": [],
    }))


def _make_embedded(tmp_path: Path) -> EmbeddedScanner:
    """Bare EmbeddedScanner: bypasses __init__ (heavy deps), wires only the
    real collaborators `_reload_config_now` touches."""
    es = EmbeddedScanner.__new__(EmbeddedScanner)
    es._project_root = tmp_path
    es._config = ScannerConfig()
    es._config_adjuster = ConfigAdjuster(
        persistence_path=tmp_path / ADJUSTMENTS,
        pending_path=tmp_path / ".claude" / "pending_adjustments.json",
    )
    es._scan_count = 0
    return es


def test_reload_config_now_lands_in_durable_overlay(tmp_path: Path) -> None:
    """The TUI reload path must consume through the overlay, not raw-apply."""
    _seed_approved(tmp_path, [
        {"proposal_id": "p1", "key": "min_confidence", "new_value": 0.48,
         "source": "self_heal", "reason": "r",
         "timestamp": "2026-07-03T00:00:00+00:00"},
    ])
    es = _make_embedded(tmp_path)
    assert es._config.min_confidence != 0.48

    es._reload_config_now()

    # In-memory effect preserved — the live session sees the new value.
    assert es._config.min_confidence == 0.48

    # Durable effect — this is the actual fix. Bypassing the overlay left the
    # adjustment ONLY in-memory even though ConfigAdjuster marked it consumed.
    overlay_path = tmp_path / OVERLAY
    assert overlay_path.exists(), (
        "_reload_config_now must route approved adjustments through "
        "config_overlay.consume_approved_adjustments so they survive a "
        "restart, not just apply them to the live in-memory config."
    )
    overlay = json.loads(overlay_path.read_text())
    assert overlay["values"]["min_confidence"] == 0.48


def test_reload_config_now_adjustment_survives_simulated_restart(tmp_path: Path) -> None:
    """The exact restart-loss scenario: TUI approves + reloads, process dies,
    a freshly-constructed ScannerConfig at the next startup must still see it."""
    _seed_approved(tmp_path, [
        {"proposal_id": "p2", "key": "min_confidence", "new_value": 0.47,
         "source": "self_heal", "reason": "r",
         "timestamp": "2026-07-03T00:00:00+00:00"},
    ])
    es = _make_embedded(tmp_path)
    es._reload_config_now()

    # Simulate restart: brand-new config, brand-new process, same disk.
    fresh_cfg = ScannerConfig()
    assert fresh_cfg.min_confidence != 0.47
    applied = apply_overlay(fresh_cfg, tmp_path)

    assert [a["key"] for a in applied] == ["min_confidence"]
    assert fresh_cfg.min_confidence == 0.47


def test_reload_config_now_is_idempotent_across_calls(tmp_path: Path) -> None:
    """Re-running the reload with nothing new approved must be a clean no-op —
    no re-application, no crash, no duplicate overlay writes."""
    _seed_approved(tmp_path, [
        {"proposal_id": "p3", "key": "min_confidence", "new_value": 0.46,
         "source": "self_heal", "reason": "r",
         "timestamp": "2026-07-03T00:00:00+00:00"},
    ])
    es = _make_embedded(tmp_path)
    es._reload_config_now()
    assert es._config.min_confidence == 0.46

    es._reload_config_now()  # nothing new pending
    assert es._config.min_confidence == 0.46


def test_reload_config_now_still_noop_when_no_adjuster_or_config() -> None:
    """Guard clause preserved: no crash when adjuster/config aren't wired yet
    (e.g. during early init or in tests)."""
    es = EmbeddedScanner.__new__(EmbeddedScanner)
    es._config_adjuster = None
    es._config = None
    es._reload_config_now()  # must not raise
