"""Front-end fix regression tests (cluster 1).

Reproduce-then-resolve tests for four verified TUI display-layer bugs:

  C1/C2/C5  src/tui/snapshot.py — the 'c'/copy-state snapshot crashed entirely
            when a single section hit edge data (numeric close timestamp,
            JSON-null brain msg, non-numeric agent weight). The module
            docstring promises inline 'ERR:' isolation that build_snapshot
            never implemented. Fixed: _safe_section wrapper + value coercions.

  C3        StateStrip floored model age via int() (7.9 -> 7) and painted
            GREEN while StalenessBanner (days > 7.0 on the raw float) showed
            RED for the same snapshot. Fixed: keep the float, round on display.

  B3        BrainEditorScreen used the human section title ('Next Actions')
            as a Textual widget id -> BadIdentifier crash on mount (Ctrl+E).
            Fixed: positional slug ids mapped back to titles.

  A1        Global 'r' binding targeted 'action_refresh_rules' (doubled
            'action_' prefix) -> resolved to a nonexistent action. Fixed.

No mocks per CLAUDE.md No-Mock Rule: real ScannerConfig/DashboardSnapshot,
real disk via tmp_path, monkeypatch only repoints a module *path constant*
(not behavior) and swaps a real function for a real raising function.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from textual.app import App
from textual.binding import Binding
from textual.widgets import ListView, TextArea

import src.tui.snapshot as snapshot
from src.tui.data_provider import DashboardSnapshot
from src.tui.screens.brain_editor_screen import BrainEditorScreen
from src.tui.widgets.staleness_banner import StalenessBanner
from src.tui.widgets.state_strip import StateStrip


# ---------------------------------------------------------------------------
# Snapshot robustness (C1, C2, C5 + build_snapshot section isolation)
# ---------------------------------------------------------------------------


def test_trades_section_survives_numeric_close_timestamp(tmp_path, monkeypatch):
    """C1: a numeric epoch close_time must not crash _trades_section."""
    monkeypatch.setattr(snapshot, "PROJECT_ROOT", tmp_path)
    journal = tmp_path / "trained_data" / "trade_journal_rl.json"
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps([
        {
            "trade_id": "t1",
            "pair": "EUR_USD",
            "outcome": {"close_time": 1700000000, "trade_won": True, "pnl_pips": 5},
        }
    ]))
    out = snapshot._trades_section()  # pre-fix: TypeError (int not subscriptable)
    assert "t1" in out
    assert "1700000000"[:19] in out


def test_brain_feed_section_survives_null_msg(tmp_path, monkeypatch):
    """C2: a JSON-null 'msg' field must not crash _brain_feed_section."""
    monkeypatch.setattr(snapshot, "PROJECT_ROOT", tmp_path)
    feed = tmp_path / ".claude" / "brain" / "feed.jsonl"
    feed.parent.mkdir(parents=True)
    feed.write_text(
        json.dumps({"ts": "2026-06-24T00:00:00Z", "msg": None}) + "\n"
        + json.dumps({"ts": "2026-06-24T00:01:00Z", "msg": "hello world"}) + "\n"
    )
    out = snapshot._brain_feed_section()  # pre-fix: None[:140] TypeError (uncaught)
    assert "hello world" in out


def test_agents_section_survives_non_numeric_weight(tmp_path, monkeypatch):
    """C5: a non-numeric agent weight must not crash _agents_section sort/format."""
    monkeypatch.setattr(snapshot, "PROJECT_ROOT", tmp_path)
    weights = tmp_path / "trained_data" / "models" / "agent_weights.json"
    weights.parent.mkdir(parents=True)
    weights.write_text(json.dumps({"NORMAL": {"trend": 1.15, "broken": "abc"}}))
    out = snapshot._agents_section()  # pre-fix: TypeError on float<str comparison
    assert "trend" in out


def test_build_snapshot_isolates_a_failing_section(tmp_path, monkeypatch):
    """A raising section degrades to an inline ERR line; the snapshot survives."""
    monkeypatch.setattr(snapshot, "PROJECT_ROOT", tmp_path)

    def _boom() -> str:
        raise RuntimeError("section exploded")

    monkeypatch.setattr(snapshot, "_trades_section", _boom)
    out = snapshot.build_snapshot()
    assert "ERR:" in out                       # failure surfaced inline
    assert "Buddy TUI Snapshot" in out          # header still produced
    assert "## Logs" in out                     # build continued past the failure


# ---------------------------------------------------------------------------
# C3 — StateStrip / StalenessBanner staleness consistency
# ---------------------------------------------------------------------------


def test_state_strip_not_green_while_banner_red():
    """C3: at 7.9d the strip must not be GREEN while the banner is RED."""
    snap = DashboardSnapshot()
    snap.max_component_age_days = 7.9

    banner = StalenessBanner()
    banner.display = False
    banner.update_from_snapshot(snap)
    assert banner.display is True, "banner must fire for 7.9d (> 7.0)"

    strip = StateStrip()
    strip.update_from_snapshot(snap)
    assert strip.model_staleness_days == pytest.approx(7.9)  # no int() floor

    text = strip.render()
    assert "8d old" in text.plain, "strip number must round like the banner (.0f)"
    model_styles = [
        str(s.style) for s in text.spans if "d old" in text.plain[s.start:s.end]
    ]
    assert model_styles, "MODELS staleness segment not rendered"
    assert "#00ff41" not in model_styles[0], "strip must not be GREEN while banner is RED"


def test_state_strip_green_when_fresh_matches_banner_hidden():
    """Consistency on the other side: 3d -> banner hidden AND strip GREEN."""
    snap = DashboardSnapshot()
    snap.max_component_age_days = 3.0

    banner = StalenessBanner()
    banner.display = False
    banner.update_from_snapshot(snap)
    assert banner.display is False

    strip = StateStrip()
    strip.update_from_snapshot(snap)
    text = strip.render()
    assert "3d old" in text.plain
    model_styles = [
        str(s.style) for s in text.spans if "d old" in text.plain[s.start:s.end]
    ]
    assert "#00ff41" in model_styles[0], "fresh models must render GREEN"


# ---------------------------------------------------------------------------
# B3 — BrainEditorScreen mounts with spaced section titles (no BadIdentifier)
# ---------------------------------------------------------------------------


_THEME = Path(__file__).resolve().parent.parent / "src" / "tui" / "theme.tcss"

_BRIEFING = """# Briefing

## Current Situation

Halted on a loss streak.

## Next Actions

- [ ] Retrain the sizer

## Open Questions

Why 52%?
"""


class _BrainApp(App):
    CSS_PATH = str(_THEME)


def test_brain_editor_mounts_with_spaced_titles(tmp_path):
    """B3: opening the editor on a real briefing (spaced titles) must not crash,
    and selecting a section must load its body via the slug->title map."""
    briefing = tmp_path / "briefing.md"
    briefing.write_text(_BRIEFING)
    app = _BrainApp()

    async def _run() -> None:
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            # Pre-fix this raised textual.css.errors.BadIdentifier on mount.
            await app.push_screen(BrainEditorScreen(briefing_path=briefing))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, BrainEditorScreen)
            lst = screen.query_one("#brain_sections", ListView)
            # Three spaced-title sections mounted with valid slug ids.
            assert len(lst.children) == 3
            ids = [item.id for item in lst.children]
            assert ids == ["brain-sec-0", "brain-sec-1", "brain-sec-2"]

            # Selecting the spaced-title section recovers the real title + body.
            lst.index = 1
            lst.action_select_cursor()
            await pilot.pause()
            assert screen._selected_title == "Next Actions"
            body = screen.query_one("#brain_body", TextArea)
            assert "Retrain the sizer" in body.text

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# A1 — 'r' binding resolves to a real action (no doubled 'action_' prefix)
# ---------------------------------------------------------------------------


def test_r_binding_targets_existing_action():
    """A1: the global 'r' binding must resolve to action_refresh_rules, and no
    binding may carry a doubled 'action_' prefix (the bug class)."""
    from src.tui.app import BuddyApp

    actions = []
    for b in BuddyApp.BINDINGS:
        if isinstance(b, Binding):
            actions.append(b.action)
        elif isinstance(b, tuple) and len(b) >= 2:
            actions.append(b[1])

    assert "refresh_rules" in actions
    assert hasattr(BuddyApp, "action_refresh_rules")
    doubled = [a for a in actions if isinstance(a, str) and a.startswith("action_")]
    assert not doubled, f"bindings with doubled 'action_' prefix: {doubled}"
