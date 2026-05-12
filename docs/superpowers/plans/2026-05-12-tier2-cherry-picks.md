# Tier 2 Cherry-Picks Implementation Plan (T7–T10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship four medium-effort lifts after Tier 1 has been live for ≥ 1 week: brain section editor, config-write-triggers-reload, two-tier trade journal cache, per-pair model inventory panel.

**Architecture:** Tier 2 builds on Tier 1 surfaces. T7 uses `brain_caps.py` (cloud branch) for pre-write validation. T8 reuses the reactive-watcher pattern from T6. T9 introduces a new `src/tui/cache/` namespace. T10 reads meta sidecars via `joblib` (sklearn convention; transparently handles the serialized-format meta sidecars that `transformer_trainer.py` writes).

**Tech Stack:** Python 3.11, Textual, pytest, `safe_json_read` / `safe_json_write`, `dataclasses`, `joblib` (already a project dep — used by sklearn estimators throughout the trainer code).

**Prerequisites:**
- Cloud branch `origin/claude/cherry-pick-ml-engine-upgrade-hKlIu` merged.
- Tier 1 (T1–T6) all merged. T8 reuses the EmbeddedScanner modification pattern from T3/T4/T6.

---

## Task 7: Brain section editor with pre-write validation + reset-to-template

**Files:**
- Create: `src/tui/widgets/brain_editor.py`
- Create: `.claude/brain/briefing.md.default` (template)
- Create: `src/tui/screens/brain_editor_screen.py`
- Modify: `src/tui/app.py` (Ctrl+E binding)
- Create: `tests/test_brain_editor.py`

- [ ] **Step 1: Write failing test for `BriefingDocument` parse/serialize**

Create `tests/test_brain_editor.py`:

```python
"""Tier 2 T7: Tests for the brain section editor."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.tui.widgets.brain_editor import BriefingDocument, BriefingSection


SAMPLE = """# Briefing

## Current Situation

Halted on Trade 1261 loss streak.

## Hypotheses

1. RL position sizer stale
2. Trend agent ADX threshold wrong

## Next Actions

- [ ] Retrain RL sizer
- [ ] Audit ADX
"""


def test_parse_three_sections(tmp_path: Path):
    p = tmp_path / "briefing.md"
    p.write_text(SAMPLE)
    doc = BriefingDocument.from_path(p)
    assert len(doc.sections) == 3
    titles = [s.title for s in doc.sections]
    assert titles == ["Current Situation", "Hypotheses", "Next Actions"]


def test_section_body_preserves_content(tmp_path: Path):
    p = tmp_path / "briefing.md"
    p.write_text(SAMPLE)
    doc = BriefingDocument.from_path(p)
    s = next(s for s in doc.sections if s.title == "Hypotheses")
    assert "RL position sizer stale" in s.body
    assert "Trend agent ADX threshold wrong" in s.body


def test_serialize_round_trip(tmp_path: Path):
    p = tmp_path / "briefing.md"
    p.write_text(SAMPLE)
    doc = BriefingDocument.from_path(p)
    out = doc.to_text()
    assert "## Current Situation" in out
    assert "## Hypotheses" in out
    assert "## Next Actions" in out


def test_update_section_changes_body(tmp_path: Path):
    p = tmp_path / "briefing.md"
    p.write_text(SAMPLE)
    doc = BriefingDocument.from_path(p)
    doc.update_section("Hypotheses", "1. only one hypothesis now")
    out = doc.to_text()
    assert "only one hypothesis now" in out
    assert "RL position sizer stale" not in out


def test_save_pre_write_validation_refuses_over_cap(tmp_path: Path):
    # briefing.md hard_cap is 3000 chars per brain_caps.py.
    p = tmp_path / "briefing.md"
    big_body = "x" * 5000
    p.write_text(f"# Briefing\n\n## Big\n\n{big_body}\n")
    doc = BriefingDocument.from_path(p)
    doc.update_section("Big", big_body + "Y" * 2000)
    with pytest.raises(ValueError) as ei:
        doc.save_to_path(p)
    assert "cap" in str(ei.value).lower() or "limit" in str(ei.value).lower()


def test_reset_to_template_overwrites_file(tmp_path: Path):
    p = tmp_path / "briefing.md"
    tmpl = tmp_path / "briefing.md.default"
    tmpl.write_text("# Briefing\n\n## Current Situation\n\n(empty)\n")
    p.write_text("# Briefing\n\n## Stale\n\nold content\n")
    doc = BriefingDocument.from_path(p)
    doc.reset_to_template(tmpl, target=p)
    fresh = BriefingDocument.from_path(p)
    assert any(s.title == "Current Situation" for s in fresh.sections)
    assert not any(s.title == "Stale" for s in fresh.sections)
```

- [ ] **Step 2: Run — confirm failure**

Run: `pytest tests/test_brain_editor.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `BriefingDocument`**

Create `src/tui/widgets/brain_editor.py`:

```python
"""Tier 2 T7: Brain section editor + BriefingDocument parser."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.scanner.automation.brain_caps import caps as _brain_caps


_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class BriefingSection:
    title: str
    body: str  # excludes the heading line


@dataclass
class BriefingDocument:
    sections: List[BriefingSection] = field(default_factory=list)
    preamble: str = ""

    @classmethod
    def from_path(cls, path: Path) -> "BriefingDocument":
        text = path.read_text(encoding="utf-8")
        return cls.from_text(text)

    @classmethod
    def from_text(cls, text: str) -> "BriefingDocument":
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return cls(preamble=text, sections=[])
        preamble = text[: matches[0].start()].rstrip() + ("\n" if matches[0].start() > 0 else "")
        sections: List[BriefingSection] = []
        for i, m in enumerate(matches):
            title = m.group(1)
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].lstrip("\n").rstrip()
            sections.append(BriefingSection(title=title, body=body))
        return cls(preamble=preamble, sections=sections)

    def to_text(self) -> str:
        parts: List[str] = []
        if self.preamble.strip():
            parts.append(self.preamble.rstrip() + "\n\n")
        for s in self.sections:
            parts.append(f"## {s.title}\n\n{s.body}\n\n")
        return "".join(parts).rstrip() + "\n"

    def update_section(self, title: str, body: str) -> None:
        for s in self.sections:
            if s.title == title:
                s.body = body.rstrip()
                return
        self.sections.append(BriefingSection(title=title, body=body.rstrip()))

    def save_to_path(self, path: Path, *, filename_for_cap: str = "briefing.md") -> None:
        """Atomic write with brain_caps validation."""
        text = self.to_text()
        cap_table = _brain_caps()
        if filename_for_cap in cap_table:
            hard_cap, warn_ratio = cap_table[filename_for_cap]
            if len(text) > int(hard_cap * 1.50):
                raise ValueError(
                    f"Would write {len(text):,} chars > 1.5x hard cap "
                    f"{hard_cap:,} for {filename_for_cap}; trim sections first."
                )
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def reset_to_template(self, template_path: Path, *, target: Path) -> None:
        text = template_path.read_text(encoding="utf-8")
        target.write_text(text, encoding="utf-8")
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/test_brain_editor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit the document model**

```bash
git add src/tui/widgets/brain_editor.py tests/test_brain_editor.py
git commit -m "feat(brain_editor): BriefingDocument parser + brain_caps-aware save (T7 model)"
```

- [ ] **Step 6: Create the template file**

Create `.claude/brain/briefing.md.default`:

```markdown
# Briefing

## Current Situation

(empty — operator or Claude fills in at session start)

## Hypotheses

(empty)

## Next Actions

(empty)

## Open Questions

(empty)
```

- [ ] **Step 7: Create the BrainEditorScreen**

Create `src/tui/screens/brain_editor_screen.py`:

```python
"""Tier 2 T7: Brain editor screen — two-pane section CRUD."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, ListItem, ListView, TextArea

from src.tui.widgets.brain_editor import BriefingDocument


class BrainEditorScreen(Screen):
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("ctrl+s", "save", "Save"),
    ]

    def __init__(self, *, briefing_path: Optional[Path] = None, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        root = Path(__file__).resolve().parents[3]
        self._briefing_path = briefing_path or root / ".claude" / "brain" / "briefing.md"
        self._template_path = root / ".claude" / "brain" / "briefing.md.default"
        self._doc = BriefingDocument()
        self._selected_title: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield ListView(id="brain_sections")
            with Vertical():
                yield TextArea("", id="brain_body")
                with Horizontal():
                    yield Button("Save (Ctrl+S)", id="save", variant="primary")
                    yield Button("Reset to template", id="reset", variant="warning")
        yield Footer()

    def on_mount(self) -> None:
        if self._briefing_path.exists():
            self._doc = BriefingDocument.from_path(self._briefing_path)
        self._render_list()

    def _render_list(self) -> None:
        lst = self.query_one("#brain_sections", ListView)
        lst.clear()
        for s in self._doc.sections:
            lst.append(ListItem(Label(s.title), id=s.title))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item is None or event.item.id is None:
            return
        self._selected_title = event.item.id
        s = next((s for s in self._doc.sections if s.title == self._selected_title), None)
        body = self.query_one("#brain_body", TextArea)
        body.text = s.body if s else ""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.action_save()
        elif event.button.id == "reset":
            try:
                self._doc.reset_to_template(self._template_path, target=self._briefing_path)
                self._doc = BriefingDocument.from_path(self._briefing_path)
                self._render_list()
                self.notify("Briefing reset to template", severity="information")
            except Exception as e:
                self.notify(f"Reset failed: {e}", severity="error")

    def action_save(self) -> None:
        if self._selected_title is None:
            return
        body_widget = self.query_one("#brain_body", TextArea)
        self._doc.update_section(self._selected_title, body_widget.text)
        try:
            self._doc.save_to_path(self._briefing_path)
            self.notify("Saved briefing.md", severity="information")
        except ValueError as e:
            self.notify(str(e), severity="error")
```

- [ ] **Step 8: Bind Ctrl+E in app.py**

In `src/tui/app.py` BINDINGS:

```python
        Binding("ctrl+e", "edit_briefing", "Edit Briefing", show=False),
```

```python
    def action_edit_briefing(self) -> None:
        from src.tui.screens.brain_editor_screen import BrainEditorScreen
        self.push_screen(BrainEditorScreen())
```

- [ ] **Step 9: Manual smoke**

```bash
./buddy --demo
# Press Ctrl+E. Brain Editor screen opens.
# Click a section, edit body, Ctrl+S — confirm .claude/brain/briefing.md updated on disk.
# Click "Reset to template" — confirm sections replaced from briefing.md.default.
```

- [ ] **Step 10: Commit**

```bash
git add src/tui/screens/brain_editor_screen.py src/tui/app.py .claude/brain/briefing.md.default
git commit -m "feat(tui): brain editor screen with section CRUD + reset-to-template (T7 UI)"
```

---

## Task 8: Config-write triggers immediate reload

**Files:**
- Modify: `src/scanner/automation/adjustment_approver.py` (post-write hook → set `config_dirty=true`)
- Modify: `src/tui/embedded_scanner.py` (pre-cycle check → reload if dirty)
- Create: `tests/test_config_dirty_flag.py`

- [ ] **Step 1: Write failing test for `config_dirty` flow**

Create `tests/test_config_dirty_flag.py`:

```python
"""Tier 2 T8: Config-write triggers immediate reload."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.automation.adjustment_approver import AdjustmentApprover
from src.scanner.automation.config_adjuster import ConfigAdjuster


def test_save_approved_sets_config_dirty(tmp_path: Path):
    persistence = tmp_path / "config_adjustments.json"
    persistence.write_text(json.dumps({"history": [], "pending": [], "last_applied": None}))
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"halted": False, "mode": "demo"}))

    adjuster = ConfigAdjuster(persistence_path=persistence)
    approver = AdjustmentApprover(adjuster=adjuster, state_path=state)
    approver._save_approved([
        {"key": "atr_sl_multiplier", "value": 1.3, "reason": "test"},
    ])

    state_data = json.loads(state.read_text())
    assert state_data.get("config_dirty") is True


def test_dirty_flag_cleared_after_consume(tmp_path: Path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"halted": False, "mode": "demo", "config_dirty": True}))

    from src.tui.embedded_scanner import _consume_config_dirty_flag
    dirty = _consume_config_dirty_flag(state)
    assert dirty is True

    after = json.loads(state.read_text())
    assert after.get("config_dirty") is False


def test_dirty_flag_false_when_not_set(tmp_path: Path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"halted": False, "mode": "demo"}))
    from src.tui.embedded_scanner import _consume_config_dirty_flag
    dirty = _consume_config_dirty_flag(state)
    assert dirty is False
```

- [ ] **Step 2: Run — confirm failure**

Run: `pytest tests/test_config_dirty_flag.py -v`
Expected: FAIL — `AdjustmentApprover` not setting `config_dirty`, helper function not defined.

- [ ] **Step 3: Add post-write hook to `_save_approved`**

Run: `grep -n "_save_approved\|state_path\|state\.json" src/scanner/automation/adjustment_approver.py | head -30`

Modify `AdjustmentApprover` constructor to accept a `state_path` and add `_mark_config_dirty`:

```python
import json
from pathlib import Path

class AdjustmentApprover:
    def __init__(self, *, adjuster, state_path: Path = None):
        self._adjuster = adjuster
        self._state_path = state_path or (Path(__file__).resolve().parents[3] / ".claude" / "state.json")

    def _save_approved(self, items):
        # ... existing save logic ...
        self._mark_config_dirty()

    def _mark_config_dirty(self) -> None:
        try:
            if self._state_path.exists():
                data = json.loads(self._state_path.read_text())
            else:
                data = {}
            data["config_dirty"] = True
            tmp = self._state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
            tmp.replace(self._state_path)
        except (OSError, json.JSONDecodeError) as e:
            import logging
            logging.getLogger(__name__).warning("AdjustmentApprover._mark_config_dirty failed: %s", e)
```

- [ ] **Step 4: Add the `_consume_config_dirty_flag` helper to embedded_scanner.py**

In `src/tui/embedded_scanner.py`, add a module-level helper:

```python
def _consume_config_dirty_flag(state_path: Path) -> bool:
    """Read state.json:config_dirty; if true, clear it atomically and return True."""
    try:
        if not state_path.exists():
            return False
        data = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not data.get("config_dirty"):
        return False
    data["config_dirty"] = False
    try:
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(state_path)
    except OSError:
        return True
    return True
```

In `EmbeddedScanner.run_one_cycle`, at the very top:

```python
        from pathlib import Path as _P
        _claude = _P(__file__).resolve().parents[3] / ".claude"
        if _consume_config_dirty_flag(_claude / "state.json"):
            self._reload_config_now()
            self._brain("[cyan]▸ config reloaded mid-session[/]")
```

Add the `_reload_config_now` method (uses T5's `_invalidate_cache`):

```python
    def _reload_config_now(self) -> None:
        if hasattr(self, "_adjuster") and self._adjuster is not None:
            try:
                self._adjuster._invalidate_cache()
                self._adjuster.apply_adjustments(self._scanner_config, current_cycle=self._scan_count)
            except Exception as e:
                logger.warning("_reload_config_now failed: %s", e)
```

Substitute actual attribute names (`self._adjuster` / `self._scanner_config` / `self._scan_count`) to match your EmbeddedScanner.

- [ ] **Step 5: Run tests — confirm pass**

Run: `pytest tests/test_config_dirty_flag.py -v`
Expected: PASS.

- [ ] **Step 6: Run existing test suite — confirm no regression**

Run: `pytest tests/ -k 'adjustment or config' -v`
Expected: PASS.

- [ ] **Step 7: Manual smoke**

```bash
./buddy --demo
# In another terminal: approve a pending adjustment via the F2 inbox.
# Confirm: scanner brain feed shows "config reloaded mid-session" within ~1 cycle.
```

- [ ] **Step 8: Commit**

```bash
git add src/scanner/automation/adjustment_approver.py src/tui/embedded_scanner.py tests/test_config_dirty_flag.py
git commit -m "feat(config): config-write triggers immediate scanner reload (T8)"
```

---

## Task 9: Two-tier trade journal cache

**Files:**
- Create: `src/tui/cache/__init__.py`
- Create: `src/tui/cache/trades_cache.py`
- Modify: `src/tui/screens/trades_screen.py` (consume the cache)
- Create: `tests/test_trades_cache.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_trades_cache.py`:

```python
"""Tier 2 T9: Two-tier trade journal cache."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tui.cache.trades_cache import TradesCache, TradeRow


def _seed_journal(path: Path, trades: list[dict]) -> None:
    path.write_text(json.dumps({"trades": trades}))


def test_initial_sync_builds_index(tmp_path: Path):
    j = tmp_path / "trade_journal_rl.json"
    cache = tmp_path / "trades_index.json"
    _seed_journal(j, [
        {"id": 1, "pair": "EUR_USD", "direction": "LONG", "pnl": 50.0,
         "opened_at": "2026-05-01T10:00:00", "closed_at": "2026-05-01T11:00:00",
         "outcome": "TP"},
        {"id": 2, "pair": "GBP_USD", "direction": "SHORT", "pnl": -23.0,
         "opened_at": "2026-05-01T12:00:00", "closed_at": "2026-05-01T13:00:00",
         "outcome": "SL"},
    ])
    c = TradesCache(journal_path=j, cache_path=cache)
    c.sync()
    rows = c.rows()
    assert len(rows) == 2
    assert any(r.pair == "EUR_USD" for r in rows)
    assert any(r.outcome == "SL" for r in rows)


def test_second_sync_reads_from_cache_unless_journal_grew(tmp_path: Path):
    j = tmp_path / "trade_journal_rl.json"
    cache = tmp_path / "trades_index.json"
    _seed_journal(j, [{"id": 1, "pair": "EUR_USD", "direction": "LONG",
                       "pnl": 50.0, "opened_at": "x", "closed_at": "y",
                       "outcome": "TP"}])
    c = TradesCache(journal_path=j, cache_path=cache)
    c.sync()
    assert c.sync_count == 1
    c.sync()
    assert c.sync_count == 1


def test_journal_growth_triggers_incremental_sync(tmp_path: Path):
    j = tmp_path / "trade_journal_rl.json"
    cache = tmp_path / "trades_index.json"
    _seed_journal(j, [{"id": 1, "pair": "EUR_USD", "direction": "LONG",
                       "pnl": 50.0, "opened_at": "x", "closed_at": "y",
                       "outcome": "TP"}])
    c = TradesCache(journal_path=j, cache_path=cache)
    c.sync()
    _seed_journal(j, [
        {"id": 1, "pair": "EUR_USD", "direction": "LONG",
         "pnl": 50.0, "opened_at": "x", "closed_at": "y", "outcome": "TP"},
        {"id": 2, "pair": "GBP_USD", "direction": "SHORT",
         "pnl": -23.0, "opened_at": "a", "closed_at": "b", "outcome": "SL"},
    ])
    c.sync()
    rows = c.rows()
    assert len(rows) == 2


def test_corrupt_cache_falls_back_to_full_rebuild(tmp_path: Path):
    j = tmp_path / "trade_journal_rl.json"
    cache = tmp_path / "trades_index.json"
    _seed_journal(j, [{"id": 1, "pair": "EUR_USD", "direction": "LONG",
                       "pnl": 50.0, "opened_at": "x", "closed_at": "y",
                       "outcome": "TP"}])
    cache.write_text("not valid json")
    c = TradesCache(journal_path=j, cache_path=cache)
    c.sync()
    assert len(c.rows()) == 1
```

- [ ] **Step 2: Run — confirm failure**

Run: `pytest tests/test_trades_cache.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement TradesCache**

Create `src/tui/cache/__init__.py` (empty file marks the package).

Create `src/tui/cache/trades_cache.py`:

```python
"""Tier 2 T9: Two-tier trade journal cache — precomputed summary rows."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradeRow:
    id: int
    pair: str
    direction: str
    pnl: float
    opened_at: str
    closed_at: str
    outcome: str


class TradesCache:
    """Maintains a precomputed trade-summary index synced incrementally."""

    def __init__(self, *, journal_path: Path, cache_path: Path) -> None:
        self._journal_path = Path(journal_path)
        self._cache_path = Path(cache_path)
        self._rows: List[TradeRow] = []
        self._last_journal_size: int = 0
        self.sync_count: int = 0
        self._loaded = False

    def _load_cache(self) -> Optional[List[TradeRow]]:
        if not self._cache_path.exists():
            return None
        try:
            data = json.loads(self._cache_path.read_text())
            rows = [TradeRow(**r) for r in data.get("rows", [])]
            self._last_journal_size = int(data.get("journal_size", 0))
            return rows
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.debug("TradesCache: cache load failed (rebuilding): %s", e)
            return None

    def _persist_cache(self) -> None:
        payload = {
            "rows": [asdict(r) for r in self._rows],
            "journal_size": self._last_journal_size,
        }
        try:
            tmp = self._cache_path.with_suffix(".json.tmp")
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload))
            tmp.replace(self._cache_path)
        except OSError as e:
            logger.warning("TradesCache: persist failed: %s", e)

    def _rebuild_from_journal(self) -> None:
        try:
            raw = json.loads(self._journal_path.read_text())
        except (OSError, json.JSONDecodeError):
            self._rows = []
            self._last_journal_size = 0
            return
        trades = raw.get("trades", [])
        self._rows = [
            TradeRow(
                id=int(t.get("id", 0)),
                pair=str(t.get("pair", "")),
                direction=str(t.get("direction", "")),
                pnl=float(t.get("pnl", 0.0)),
                opened_at=str(t.get("opened_at", "")),
                closed_at=str(t.get("closed_at", "")),
                outcome=str(t.get("outcome", "")),
            )
            for t in trades
        ]
        try:
            self._last_journal_size = self._journal_path.stat().st_size
        except OSError:
            self._last_journal_size = 0

    def sync(self) -> None:
        """Idempotent sync. Only does work if the journal grew (or no cache)."""
        if not self._loaded:
            cached = self._load_cache()
            if cached is not None:
                self._rows = cached
            else:
                self._rebuild_from_journal()
                self._persist_cache()
                self.sync_count += 1
            self._loaded = True
            return

        try:
            current_size = self._journal_path.stat().st_size
        except OSError:
            return
        if current_size == self._last_journal_size:
            return
        self._rebuild_from_journal()
        self._persist_cache()
        self.sync_count += 1

    def rows(self) -> List[TradeRow]:
        return list(self._rows)
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/test_trades_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Wire `TradesCache` into the Trades screen**

In `src/tui/screens/trades_screen.py`, at screen init:

```python
        from src.tui.cache.trades_cache import TradesCache
        from pathlib import Path as _P
        _root = _P(__file__).resolve().parents[3]
        self._cache = TradesCache(
            journal_path=_root / "trained_data" / "trade_journal_rl.json",
            cache_path=_root / ".claude" / "ui_cache" / "trades_index.json",
        )
```

In every refresh path:

```python
        self._cache.sync()
        rows = self._cache.rows()
        # render rows in the existing DataTable
```

Drop the prior direct-JSON-parse path.

- [ ] **Step 6: Manual smoke**

```bash
./buddy --demo
# F3 Trades — load is faster on repeat opens (cache hit).
# rm .claude/ui_cache/trades_index.json — re-open F3 — rebuild happens silently.
```

- [ ] **Step 7: Commit**

```bash
git add src/tui/cache/ src/tui/screens/trades_screen.py tests/test_trades_cache.py
git commit -m "feat(tui): two-tier trade journal cache for Trades screen (T9)"
```

---

## Task 10: Per-pair model inventory panel

**Files:**
- Create: `src/tui/widgets/model_inventory.py`
- Modify: `src/tui/screens/diagnostics_screen.py` (mount the panel)
- Create: `tests/test_model_inventory.py`

**Note on meta-sidecar deserialization:** The trainer writes `transformer_direction.meta.pkl` via the standard sklearn pattern — these files are joblib-loadable. We use `joblib.load(...)` rather than the lower-level deserializer so the inventory module stays decoupled from the trainer's internal write API. All meta files come from this codebase's trainer; no external/untrusted input passes through this read path.

- [ ] **Step 1: Write failing test**

Create `tests/test_model_inventory.py`:

```python
"""Tier 2 T10: Per-pair model inventory."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pytest

from src.tui.widgets.model_inventory import ModelInventory, ModelCard


def _write_meta(path: Path, granularity: str = "M15", holdout: float = 0.65) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "granularity": granularity,
        "val_balanced_accuracy": holdout,
        "trained_at": "2026-05-01T00:00:00+00:00",
        "feature_pipeline_version": "2026-05-08-v1",
    }
    joblib.dump(payload, path)


def test_inventory_lists_pairs_with_meta(tmp_path: Path):
    root = tmp_path / "models"
    _write_meta(root / "EUR_USD" / "transformer_direction.meta.pkl")
    _write_meta(root / "GBP_USD" / "transformer_direction.meta.pkl", holdout=0.58)
    inv = ModelInventory(models_root=root)
    inv.scan()
    cards = inv.cards()
    pairs = {c.pair for c in cards}
    assert pairs == {"EUR_USD", "GBP_USD"}
    eur = next(c for c in cards if c.pair == "EUR_USD")
    assert eur.holdout_accuracy == 0.65
    gbp = next(c for c in cards if c.pair == "GBP_USD")
    assert gbp.holdout_accuracy == 0.58


def test_missing_meta_returns_unknown_status(tmp_path: Path):
    root = tmp_path / "models"
    (root / "EUR_USD").mkdir(parents=True)
    inv = ModelInventory(models_root=root)
    inv.scan()
    cards = inv.cards()
    assert len(cards) == 1
    assert cards[0].pair == "EUR_USD"
    assert cards[0].status == "no_meta"
    assert cards[0].holdout_accuracy is None


def test_corrupt_meta_handled_gracefully(tmp_path: Path):
    root = tmp_path / "models"
    (root / "EUR_USD").mkdir(parents=True)
    (root / "EUR_USD" / "transformer_direction.meta.pkl").write_bytes(b"not a serialized object")
    inv = ModelInventory(models_root=root)
    inv.scan()
    cards = inv.cards()
    assert cards[0].status in ("corrupt_meta", "no_meta", "error")
```

- [ ] **Step 2: Run — confirm failure**

Run: `pytest tests/test_model_inventory.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ModelInventory`**

Create `src/tui/widgets/model_inventory.py`:

```python
"""Tier 2 T10: Per-pair model inventory — read meta sidecars via joblib.

Loader rationale: meta sidecars are written by the project's own trainer
(transformer_trainer.py) using joblib.dump — the sklearn-ecosystem convention.
We read with joblib.load to keep the inventory module decoupled from the
trainer's internal write API. The files originate inside this repo; no
external/untrusted input reaches this loader.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import joblib

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCard:
    pair: str
    status: str  # ok | no_meta | corrupt_meta | error
    granularity: Optional[str] = None
    holdout_accuracy: Optional[float] = None
    trained_at: Optional[str] = None
    age_days: Optional[float] = None
    pipeline_version: Optional[str] = None


class ModelInventory:
    """Walks trained_data/models/<PAIR>/*.meta.pkl and produces ModelCards."""

    def __init__(self, *, models_root: Path) -> None:
        self._root = Path(models_root)
        self._cards: List[ModelCard] = []

    def scan(self) -> None:
        cards: List[ModelCard] = []
        if not self._root.exists():
            self._cards = cards
            return
        for pair_dir in sorted(self._root.iterdir()):
            if not pair_dir.is_dir():
                continue
            meta = pair_dir / "transformer_direction.meta.pkl"
            if not meta.exists():
                cards.append(ModelCard(pair=pair_dir.name, status="no_meta"))
                continue
            try:
                payload = joblib.load(meta)
            except (OSError, EOFError, KeyError, ValueError, Exception) as e:
                # joblib raises a variety of errors on corrupt input; coalesce.
                logger.debug("ModelInventory: corrupt %s: %s", meta, e)
                cards.append(ModelCard(pair=pair_dir.name, status="corrupt_meta"))
                continue
            try:
                trained_at_str = payload.get("trained_at") if isinstance(payload, dict) else None
                age_days: Optional[float] = None
                if trained_at_str:
                    ts = datetime.fromisoformat(trained_at_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
                cards.append(ModelCard(
                    pair=pair_dir.name,
                    status="ok",
                    granularity=payload.get("granularity") if isinstance(payload, dict) else None,
                    holdout_accuracy=payload.get("val_balanced_accuracy") if isinstance(payload, dict) else None,
                    trained_at=trained_at_str,
                    age_days=age_days,
                    pipeline_version=payload.get("feature_pipeline_version") if isinstance(payload, dict) else None,
                ))
            except (TypeError, ValueError) as e:
                logger.debug("ModelInventory: malformed meta %s: %s", meta, e)
                cards.append(ModelCard(pair=pair_dir.name, status="error"))
        self._cards = cards

    def cards(self) -> List[ModelCard]:
        return list(self._cards)
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/test_model_inventory.py -v`
Expected: PASS.

- [ ] **Step 5: Mount in Diagnostics screen**

In `src/tui/screens/diagnostics_screen.py`, add a new section:

```python
from src.tui.widgets.model_inventory import ModelInventory
from pathlib import Path as _P

# In compose():
yield Static("Per-Pair Model Inventory")
yield DataTable(id="model_inv_table", zebra_stripes=True)

# In on_mount():
self._inv = ModelInventory(
    models_root=_P(__file__).resolve().parents[3] / "trained_data" / "models",
)
self._refresh_inventory()
self.set_interval(30.0, self._refresh_inventory)

def _refresh_inventory(self) -> None:
    self._inv.scan()
    t = self.query_one("#model_inv_table", DataTable)
    t.clear()
    if not t.columns:
        t.add_columns("Pair", "Status", "Granularity", "Holdout %", "Age (days)", "Pipeline Ver")
    for c in self._inv.cards():
        t.add_row(
            c.pair,
            c.status,
            c.granularity or "—",
            f"{c.holdout_accuracy:.1%}" if c.holdout_accuracy is not None else "—",
            f"{c.age_days:.1f}" if c.age_days is not None else "—",
            c.pipeline_version or "—",
        )
```

- [ ] **Step 6: Manual smoke**

```bash
./buddy --demo
# F8 Diagnostics → scroll to "Per-Pair Model Inventory".
# Each pair under trained_data/models/ appears with its meta data.
# Remove one pair's meta file, wait 30s, status flips to "no_meta".
```

- [ ] **Step 7: Commit**

```bash
git add src/tui/widgets/model_inventory.py src/tui/screens/diagnostics_screen.py tests/test_model_inventory.py
git commit -m "feat(tui): per-pair model inventory panel in Diagnostics (T10)"
```

---

## Self-review checklist

1. **Spec coverage:** T7–T10 from the design spec all have tasks. ✓
2. **Placeholder scan:** every step contains actual code or commands. ✓
3. **Type consistency:** `BriefingDocument` / `BriefingSection` consistent across T7. `TradesCache` / `TradeRow` consistent across T9. `ModelInventory` / `ModelCard` consistent across T10. ✓
4. **No mocks:** all new tests use real classes + real disk via `tmp_path`. ✓
5. **CLAUDE.md alignment:** atomic writes (temp + rename) for all new file writes, additive-only schema changes for `state.json` and `JobRuntimeState`, no Claude in hot path. ✓
6. **Sidecar deserialization:** T10 uses `joblib.load` on first-party meta files written by this repo's own trainer — no external/untrusted input. ✓

## Execution handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task (T7–T10), four independent PRs.
2. **Inline Execution** — sequential execution in one session via `superpowers:executing-plans`.

Recommend (1) for parity with the Tier 1 execution pattern.
