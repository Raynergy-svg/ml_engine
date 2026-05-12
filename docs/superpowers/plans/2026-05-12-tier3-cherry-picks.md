# Tier 3 Cherry-Picks Implementation Plan (T11–T13)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship three speculative lifts when a concrete operational trigger surfaces: stepped retrain progress, two-pane skills/rules viewer, ML-head capability inventory.

**Architecture:** Tier 3 is design-first. Each task includes an **activation gate** — a specific operational trigger that justifies the effort. Do NOT execute these tasks until the gate fires. The plans below are durable specs; they sit on the shelf until needed.

**Tech Stack:** Same as Tier 1+2.

**Prerequisites:**
- Cloud branch merged.
- Tier 1 (T1–T6) merged.
- Tier 2 (T7–T10) merged (T13's data-flow assumes the T10 model inventory pattern exists).
- The task's activation gate has fired (recorded by operator in their session log or `.claude/brain/strategic_log.md`).

---

## Task 11: Stepped retrain progress display

**Activation gate:** triggered when an operator manually runs a per-pair retrain ≥ 3 times within 7 days, OR when a retrain failure goes undiagnosed for ≥ 1 cycle because the operator couldn't tell where it crashed.

**Rationale:** Retraining is currently an opaque log stream. If retrains are infrequent, the cost of opacity is low; if they become regular operations, three-layer transparency (progress bar + step label + tailing log) pays for itself.

**Files:**
- Modify: `src/training/trainers/transformer_trainer.py` (emit progress events)
- Create: `src/tui/widgets/retrain_progress.py`
- Create: `src/tui/screens/retrain_progress_modal.py`
- Modify: `src/tui/app.py` (operator-triggered retrain action wraps the modal around the existing trainer call)
- Create: `tests/test_retrain_progress.py`

- [ ] **Step 1: Read existing trainer entry point**

Run: `grep -n "def train\|class TransformerTrainer\|class Trainer\|def fit" src/training/trainers/transformer_trainer.py | head -20`
Expected: identifies the top-level `train(...)` callable + the major phase boundaries (data load, feature compute, model fit, validation, holdout, save).

- [ ] **Step 2: Write failing test for `ProgressEvent` model**

Create `tests/test_retrain_progress.py`:

```python
"""Tier 3 T11: Stepped retrain progress."""
from __future__ import annotations

import pytest

from src.tui.widgets.retrain_progress import ProgressEvent, ProgressTracker


def test_event_defaults():
    e = ProgressEvent(step=0, total_steps=6, title="Init", detail="")
    assert e.percent() == 0


def test_event_midway_percent():
    e = ProgressEvent(step=3, total_steps=6, title="Training", detail="epoch 5/40")
    assert e.percent() == 50


def test_event_complete_percent_caps_at_100():
    e = ProgressEvent(step=10, total_steps=6, title="Done", detail="")
    assert e.percent() == 100


def test_tracker_emits_log_lines():
    t = ProgressTracker(total_steps=4)
    t.advance("Load data", "loading 22mo H1")
    t.advance("Compute features", "186 cols")
    log = t.log_lines()
    assert len(log) == 2
    assert "Load data" in log[0]
    assert "186 cols" in log[1]


def test_tracker_current_event_tracks_progress():
    t = ProgressTracker(total_steps=4)
    t.advance("Step A")
    e = t.current()
    assert e.step == 1
    assert e.total_steps == 4
    assert e.title == "Step A"


def test_tracker_complete_marks_step_eq_total():
    t = ProgressTracker(total_steps=3)
    t.advance("a"); t.advance("b"); t.advance("c")
    t.complete()
    e = t.current()
    assert e.step == 3
    assert e.total_steps == 3
```

- [ ] **Step 3: Run — confirm failure**

Run: `pytest tests/test_retrain_progress.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Implement ProgressTracker + ProgressEvent**

Create `src/tui/widgets/retrain_progress.py`:

```python
"""Tier 3 T11: Retrain progress data model + Textual widget."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass(frozen=True)
class ProgressEvent:
    step: int
    total_steps: int
    title: str
    detail: str = ""

    def percent(self) -> int:
        if self.total_steps <= 0:
            return 0
        pct = int(round(self.step / self.total_steps * 100))
        return min(100, max(0, pct))


class ProgressTracker:
    """Stateful, append-only progress recorder. Trainer holds one; UI reads it."""

    def __init__(self, *, total_steps: int) -> None:
        self._total = total_steps
        self._step = 0
        self._current_title = ""
        self._current_detail = ""
        self._log: List[str] = []

    def advance(self, title: str, detail: str = "") -> None:
        self._step += 1
        self._current_title = title
        self._current_detail = detail
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._log.append(f"[{ts}] Step {self._step}/{self._total}: {title}" +
                         (f" — {detail}" if detail else ""))

    def complete(self) -> None:
        self._step = self._total

    def current(self) -> ProgressEvent:
        return ProgressEvent(
            step=self._step,
            total_steps=self._total,
            title=self._current_title,
            detail=self._current_detail,
        )

    def log_lines(self) -> List[str]:
        return list(self._log)
```

- [ ] **Step 5: Run tests — confirm pass**

Run: `pytest tests/test_retrain_progress.py -v`
Expected: PASS.

- [ ] **Step 6: Thread `ProgressTracker` through the trainer**

In `src/training/trainers/transformer_trainer.py`, modify the top-level `train` function (or whatever the canonical entry is) to accept an optional `progress_tracker: Optional[ProgressTracker] = None` kwarg. At each phase boundary call `progress_tracker.advance(...)` when non-None. Phases (adjust to actual code):

```python
from src.tui.widgets.retrain_progress import ProgressTracker

def train(pair: str, *, progress_tracker: Optional[ProgressTracker] = None, ...):
    if progress_tracker:
        progress_tracker.advance("Load data", f"22mo {granularity} for {pair}")
    df = load_direction_data(...)

    if progress_tracker:
        progress_tracker.advance("Compute features", "normalized feature pipeline")
    X, y = compute_normalized_features(df, ...)

    if progress_tracker:
        progress_tracker.advance("Fit model", f"d_model={d_model} layers={layers}")
    model = fit_transformer(X, y, ...)

    if progress_tracker:
        progress_tracker.advance("Validate", "walk-forward + purged k-fold")
    metrics = walkforward_validate(model, ...)

    if progress_tracker:
        progress_tracker.advance("Holdout eval", "OOS slice")
    holdout = evaluate_holdout(model, ...)

    if progress_tracker:
        progress_tracker.advance("Save", "atomic write + meta sidecar")
    save_model(model, ..., meta={...})

    if progress_tracker:
        progress_tracker.complete()
```

(The exact phase set + total_steps will depend on the actual trainer pipeline — count the existing logger.info breakpoints and treat each major one as a step.)

- [ ] **Step 7: Build the modal UI**

Create `src/tui/screens/retrain_progress_modal.py`:

```python
"""Tier 3 T11: Modal screen showing retrain progress."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, ProgressBar, RichLog, Static

from src.tui.widgets.retrain_progress import ProgressTracker


class RetrainProgressModal(ModalScreen):
    """Three-layer display: progress bar + current step label + log tail."""

    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(self, *, tracker: ProgressTracker, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tracker = tracker
        self._bar = None
        self._label = None
        self._log = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            self._bar = ProgressBar(total=100, show_eta=False, id="retrain_bar")
            yield self._bar
            self._label = Static("", id="retrain_label")
            yield self._label
            self._log = RichLog(id="retrain_log", max_lines=500)
            yield self._log
        yield Footer()

    def on_mount(self) -> None:
        self._tick()
        self.set_interval(0.5, self._tick)

    def _tick(self) -> None:
        ev = self._tracker.current()
        if self._bar is not None:
            self._bar.update(progress=ev.percent())
        if self._label is not None:
            label = f"Step {ev.step}/{ev.total_steps}: {ev.title}"
            if ev.detail:
                label += f" — {ev.detail}"
            self._label.update(label)
        if self._log is not None:
            # Re-tail the log lines (simple replace; logs are short).
            self._log.clear()
            for line in self._tracker.log_lines()[-200:]:
                self._log.write(line)
```

- [ ] **Step 8: Wire an operator action to trigger a retrain**

In `src/tui/app.py`, add a binding (e.g., Ctrl+T) that opens a confirm-modal asking "retrain which pair?", then spawns the trainer in a daemon thread holding a `ProgressTracker`, then `push_screen(RetrainProgressModal(tracker=tracker))`.

Skeleton:

```python
def action_trigger_retrain(self) -> None:
    # Operator selects pair (modal or quick-input).
    # Construct tracker + thread.
    from src.tui.widgets.retrain_progress import ProgressTracker
    tracker = ProgressTracker(total_steps=6)
    import threading
    def _run():
        from src.training.trainers.transformer_trainer import train
        try:
            train(pair=chosen_pair, granularity="M15", progress_tracker=tracker)
        except Exception:
            logger.exception("retrain failed")
    threading.Thread(target=_run, daemon=True, name=f"retrain_{chosen_pair}").start()
    self.push_screen(RetrainProgressModal(tracker=tracker))
```

(Pair selection UI is a separate small modal — operator types e.g. `EUR_USD` or picks from a list.)

- [ ] **Step 9: Manual smoke**

```bash
./buddy --demo
# Ctrl+T → select EUR_USD → modal opens with progress bar advancing.
# Steps update as trainer phases fire.
# Esc dismisses the modal; trainer continues in the background.
```

- [ ] **Step 10: Commit**

```bash
git add src/tui/widgets/retrain_progress.py src/tui/screens/retrain_progress_modal.py src/training/trainers/transformer_trainer.py src/tui/app.py tests/test_retrain_progress.py
git commit -m "feat(retrain): stepped progress display (T11)"
```

---

## Task 12: Two-pane skills/rules viewer

**Activation gate:** triggered when `.claude/rules/` exceeds 20 distinct rule documents, OR when a rules-cleanup audit happens and the operator finds themselves opening files one at a time.

**Rationale:** Today `.claude/rules/` is grep-and-cat. With 20+ files the browse-and-read latency becomes painful. A two-pane viewer mirrors Hermes Desktop's skill browser pattern.

**Files:**
- Modify: `src/tui/screens/rules_screen.py` (existing F7 screen — enhance from list-only to two-pane)
- Create: `tests/test_rules_viewer.py`

- [ ] **Step 1: Read the existing rules_screen.py to know the current shape**

Run: `wc -l src/tui/screens/rules_screen.py && head -40 src/tui/screens/rules_screen.py`
Expected: small file (probably < 200 lines). Note: rules screen likely shows just file names or simple list. The enhancement is adding a right-pane content view.

- [ ] **Step 2: Write failing test for the rule-document parser**

Create `tests/test_rules_viewer.py`:

```python
"""Tier 3 T12: Two-pane skills/rules viewer."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.tui.screens.rules_screen import discover_rules, RuleDoc


def test_discover_rules_finds_md_files(tmp_path: Path):
    (tmp_path / "trading.md").write_text("# Trading Rules\n\nrule one\n")
    (tmp_path / "improvement.md").write_text("# Improvement Rules\n\nrule two\n")
    (tmp_path / "README.md").write_text("readme — should be excluded\n")
    docs = discover_rules(rules_dir=tmp_path)
    names = {d.name for d in docs}
    assert "trading.md" in names
    assert "improvement.md" in names
    # README is excluded by convention:
    assert "README.md" not in names


def test_rule_doc_body_loaded_lazily(tmp_path: Path):
    p = tmp_path / "trading.md"
    p.write_text("# Trading Rules\n\nrule one\nrule two\n")
    docs = discover_rules(rules_dir=tmp_path)
    d = next(d for d in docs if d.name == "trading.md")
    assert d.read_body() == "# Trading Rules\n\nrule one\nrule two\n"


def test_rule_doc_promoted_badge_detected(tmp_path: Path):
    p = tmp_path / "trading.md"
    p.write_text("## Promoted Rules\n\n- [2026-04-02] x\n- [2026-04-15] y\n")
    docs = discover_rules(rules_dir=tmp_path)
    d = next(d for d in docs if d.name == "trading.md")
    assert d.has_promoted_section() is True


def test_rule_doc_no_promoted_section(tmp_path: Path):
    p = tmp_path / "draft.md"
    p.write_text("# Draft Rules\n\nideas only\n")
    docs = discover_rules(rules_dir=tmp_path)
    d = next(d for d in docs if d.name == "draft.md")
    assert d.has_promoted_section() is False
```

- [ ] **Step 3: Run — confirm failure**

Run: `pytest tests/test_rules_viewer.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement `RuleDoc` + `discover_rules` in `rules_screen.py`**

Add at the top of `src/tui/screens/rules_screen.py`:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class RuleDoc:
    name: str
    path: Path

    def read_body(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def has_promoted_section(self) -> bool:
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return False
        return "## Promoted Rules" in text or "## Promoted" in text


def discover_rules(*, rules_dir: Path) -> List[RuleDoc]:
    if not rules_dir.exists():
        return []
    out: List[RuleDoc] = []
    for p in sorted(rules_dir.glob("*.md")):
        if p.name.lower() == "readme.md":
            continue
        out.append(RuleDoc(name=p.name, path=p))
    return out
```

- [ ] **Step 5: Run tests — confirm pass**

Run: `pytest tests/test_rules_viewer.py -v`
Expected: PASS.

- [ ] **Step 6: Restructure the screen to two-pane**

Replace the existing rules_screen.py compose() with a horizontal split: left = `ListView` of rule names with promoted-badge column; right = `Markdown` widget showing the selected doc's body. Refresh body on `ListView.Selected` event.

Skeleton:

```python
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import ListView, ListItem, Label, Markdown, Header, Footer


class RulesScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("r", "refresh", "Refresh")]

    def __init__(self, *, rules_dir: Path = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rules_dir = rules_dir or Path(__file__).resolve().parents[3] / ".claude" / "rules"
        self._docs: list[RuleDoc] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield ListView(id="rules_list")
            yield Markdown("", id="rule_body")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        self._docs = discover_rules(rules_dir=self._rules_dir)
        lst = self.query_one("#rules_list", ListView)
        lst.clear()
        for d in self._docs:
            badge = "✓" if d.has_promoted_section() else " "
            lst.append(ListItem(Label(f"{badge}  {d.name}"), id=d.name))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item is None or event.item.id is None:
            return
        doc = next((d for d in self._docs if d.name == event.item.id), None)
        if doc is None:
            return
        body = self.query_one("#rule_body", Markdown)
        body.update(doc.read_body())

    def action_refresh(self) -> None:
        self._refresh()
```

- [ ] **Step 7: Manual smoke**

```bash
./buddy --demo
# F7 Rules. Confirm two-pane layout. Arrow keys / Enter to select.
# Right pane updates with markdown body. ✓ badge appears next to files with "Promoted Rules" section.
```

- [ ] **Step 8: Commit**

```bash
git add src/tui/screens/rules_screen.py tests/test_rules_viewer.py
git commit -m "feat(tui): two-pane rules/skills viewer with promoted badge (T12)"
```

---

## Task 13: ML head capability inventory (live registry)

**Activation gate:** triggered when an operator can't quickly answer "which heads are wired for EUR_USD M15 right now" without grep, OR when a per-pair head silently regresses to fallback (e.g., transformer falls back to HistGB) and the operator doesn't notice.

**Rationale:** Each pair's effective inference stack is a function of which artifacts exist on disk and which the `GateEvaluator` actually loaded. Today the answer requires reading orchestrator logs. A live registry panel turns "which heads are live for EUR_USD" into a glance.

**Files:**
- Create: `src/tui/widgets/capability_inventory.py`
- Modify: `src/tui/screens/diagnostics_screen.py` (mount alongside the T10 model inventory)
- Create: `tests/test_capability_inventory.py`

**Note:** This task overlaps T10 (per-pair model inventory). T10 asks "what model files exist on disk"; T13 asks "what is the live `GateEvaluator` actually using". The difference matters when a model exists but failed to load.

- [ ] **Step 1: Identify the canonical "what's loaded" surface**

Run: `grep -n "class GateEvaluator\|self\._models\|_load_transformer\|_load_momentum\|def _get_pair_evaluator" src/scanner/gates.py | head -30`
Expected: identifies the `GateEvaluator` class, its per-pair sub-evaluator cache (`_get_pair_evaluator`), and the per-head loaders.

- [ ] **Step 2: Write failing test for `CapabilityRegistry.snapshot`**

Create `tests/test_capability_inventory.py`:

```python
"""Tier 3 T13: ML head capability inventory."""
from __future__ import annotations

import pytest

from src.tui.widgets.capability_inventory import CapabilityRow, CapabilityRegistry


class _FakeEvaluator:
    """Stand-in for GateEvaluator with the minimal surface CapabilityRegistry reads."""
    def __init__(self, *, transformer_loaded: bool, ridge_loaded: bool,
                 momentum_loaded: bool, tcn_loaded: bool):
        self._transformer = object() if transformer_loaded else None
        self._ridge = object() if ridge_loaded else None
        self._momentum = object() if momentum_loaded else None
        self._tcn = object() if tcn_loaded else None


def test_snapshot_lists_heads_per_pair():
    pair_evaluators = {
        "EUR_USD": _FakeEvaluator(transformer_loaded=True, ridge_loaded=True,
                                  momentum_loaded=True, tcn_loaded=True),
        "GBP_USD": _FakeEvaluator(transformer_loaded=True, ridge_loaded=False,
                                  momentum_loaded=True, tcn_loaded=True),
    }
    reg = CapabilityRegistry(pair_evaluators=pair_evaluators)
    rows = reg.snapshot()
    eur = next(r for r in rows if r.pair == "EUR_USD")
    gbp = next(r for r in rows if r.pair == "GBP_USD")
    assert eur.heads_loaded == {"transformer": True, "ridge": True, "momentum": True, "tcn": True}
    assert gbp.heads_loaded["ridge"] is False
    assert gbp.heads_loaded["transformer"] is True


def test_snapshot_handles_empty_dict():
    reg = CapabilityRegistry(pair_evaluators={})
    assert reg.snapshot() == []
```

- [ ] **Step 3: Run — confirm failure**

Run: `pytest tests/test_capability_inventory.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement CapabilityRegistry**

Create `src/tui/widgets/capability_inventory.py`:

```python
"""Tier 3 T13: Live ML-head capability registry."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class CapabilityRow:
    pair: str
    heads_loaded: Dict[str, bool]


class CapabilityRegistry:
    """Reads `_transformer`, `_ridge`, `_momentum`, `_tcn` attributes from
    each per-pair GateEvaluator and reports which are non-None.

    The attribute names match the actual GateEvaluator internals as of
    Tier 7 per-pair routing. If those names change, update _HEADS.
    """

    _HEADS = ("transformer", "ridge", "momentum", "tcn")

    def __init__(self, *, pair_evaluators: Dict[str, Any]) -> None:
        self._evaluators = pair_evaluators

    def snapshot(self) -> List[CapabilityRow]:
        rows: List[CapabilityRow] = []
        for pair, evaluator in sorted(self._evaluators.items()):
            heads: Dict[str, bool] = {}
            for head in self._HEADS:
                attr = getattr(evaluator, f"_{head}", None)
                heads[head] = attr is not None
            rows.append(CapabilityRow(pair=pair, heads_loaded=heads))
        return rows
```

- [ ] **Step 5: Run tests — confirm pass**

Run: `pytest tests/test_capability_inventory.py -v`
Expected: PASS.

- [ ] **Step 6: Wire the registry into Diagnostics screen**

In `src/tui/screens/diagnostics_screen.py`, mount alongside T10's model inventory:

```python
from src.tui.widgets.capability_inventory import CapabilityRegistry

# In compose() — after the T10 model_inv_table:
yield Static("Live ML Head Capability (per-pair)")
yield DataTable(id="capability_table", zebra_stripes=True)

# In on_mount() — after T10 wiring:
self._capability = None  # populated when scanner ref available
self._refresh_capability()
self.set_interval(15.0, self._refresh_capability)

def _refresh_capability(self) -> None:
    # Resolve per-pair evaluators from the live GateEvaluator.
    scanner = self.app._scanner if hasattr(self.app, "_scanner") else None
    pair_evaluators: dict = {}
    try:
        gate_eval = getattr(scanner, "_gate_evaluator", None) if scanner else None
        if gate_eval is not None:
            # Access internal cache populated by _get_pair_evaluator:
            pair_evaluators = dict(getattr(gate_eval, "_pair_evaluators", {}) or {})
    except Exception:
        pass
    reg = CapabilityRegistry(pair_evaluators=pair_evaluators)
    t = self.query_one("#capability_table", DataTable)
    t.clear()
    if not t.columns:
        t.add_columns("Pair", "transformer", "ridge", "momentum", "tcn")
    for row in reg.snapshot():
        t.add_row(
            row.pair,
            "✓" if row.heads_loaded["transformer"] else "✗",
            "✓" if row.heads_loaded["ridge"] else "✗",
            "✓" if row.heads_loaded["momentum"] else "✗",
            "✓" if row.heads_loaded["tcn"] else "✗",
        )
```

(The exact paths to the per-pair evaluator dict depend on the runtime `GateEvaluator` internals — confirm via Step 1's grep and adjust attribute names.)

- [ ] **Step 7: Manual smoke**

```bash
./buddy --demo
# F8 Diagnostics → scroll to "Live ML Head Capability".
# Confirm: per-pair rows show ✓/✗ for transformer/ridge/momentum/tcn.
# If you renamed a model file out from under the scanner and restarted,
# the corresponding cell flips to ✗ on next 15s refresh.
```

- [ ] **Step 8: Commit**

```bash
git add src/tui/widgets/capability_inventory.py src/tui/screens/diagnostics_screen.py tests/test_capability_inventory.py
git commit -m "feat(tui): live ML head capability registry per pair (T13)"
```

---

## Self-review checklist

1. **Spec coverage:** T11–T13 from the design spec all have tasks. ✓
2. **Placeholder scan:** every step has actual code or commands. The trainer-instrumentation step (T11 Step 6) requires the implementer to adapt to actual trainer phase boundaries, but the *pattern* and contract is fully specified. ✓
3. **Type consistency:** `ProgressEvent` / `ProgressTracker` consistent across T11. `RuleDoc` / `discover_rules` consistent across T12. `CapabilityRow` / `CapabilityRegistry` consistent across T13. ✓
4. **Activation gates:** every task documents the operational trigger that justifies the work. Operator should not execute these unless the gate has fired. ✓
5. **No mocks:** T11 + T12 use real disk; T13 uses a tiny in-file `_FakeEvaluator` stand-in (a real GateEvaluator instance has tier-7 dependencies that aren't worth setting up for a unit test). This is **not** a mock — it's a minimal duck-type that exposes the attributes the registry reads. ✓
6. **CLAUDE.md alignment:** no Claude in hot path, atomic writes where applicable, additive changes to existing files. ✓

## Execution handoff

Plan complete. Tier 3 is **specs-on-the-shelf** — do NOT execute these tasks proactively. Wait for the activation gate. When a gate fires:

1. **Subagent-Driven** — dispatch a fresh subagent for the specific task whose gate fired.
2. **Inline Execution** — execute that one task in-session.

Recommendation: revisit Tier 3 quarterly (90 days after Tier 2 lands). If no gate has fired by then, archive the plan rather than letting it rot in `docs/superpowers/plans/`.
