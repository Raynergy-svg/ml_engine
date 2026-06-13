"""
RULES / LEARNINGS / BRAIN VIEWER (F7) — read-only window into Buddy's reasoning layer.

Surfaces the promoted rules that actively gate behavior, the raw learnings awaiting
promotion, and Claude's working-memory brain files. File tree on the left, rendered
markdown on the right. '/' searches in the current view, 'g' greps across all files.

This screen is strictly read-only — no write operations of any kind.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from rich.text import Text

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, Markdown, Static, Tree
from textual.widgets.tree import TreeNode

logger = logging.getLogger(__name__)


# ── File set ────────────────────────────────────────────────────────────


def _collect_files(project_root: Path) -> dict[str, list[Path]]:
    """Return the canonical file map: section → list of file paths.

    Always returns the section keys; missing files are simply omitted from
    their section so the tree stays clean.
    """
    claude = project_root / ".claude"
    rules_dir = claude / "rules"
    brain_dir = claude / "brain"

    rules_files = sorted(rules_dir.glob("*.md")) if rules_dir.exists() else []

    learnings = []
    learnings_path = claude / "learnings.md"
    if learnings_path.exists():
        learnings.append(learnings_path)

    brain_names = [
        "briefing.md",
        "open_questions.md",
        "session_handoff.md",
        "strategic_log.md",
        "trade_narrative.md",
    ]
    brain_files = [brain_dir / n for n in brain_names if (brain_dir / n).exists()]

    return {
        "rules": rules_files,
        "learnings": learnings,
        "brain": brain_files,
    }


# ── Search Modal ('/') ─────────────────────────────────────────────────


class SearchModal(ModalScreen[Optional[str]]):
    """In-view search — returns the search term or None on cancel."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=True)]

    DEFAULT_CSS = """
    SearchModal { align: center middle; }
    #search-dialog {
        width: 60; height: auto; background: #0a0a0f;
        border: double #00ffcc; padding: 1 2;
    }
    #search-title { text-align: center; color: #00ffcc; text-style: bold; padding: 0 0 1 0; }
    #search-input { margin: 1 0; border: solid #00ffcc; background: #131320; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="search-dialog"):
            yield Label("◈ SEARCH IN VIEW", id="search-title")
            yield Input(placeholder="term…", id="search-input")

    def on_mount(self) -> None:
        self.query_one("#search-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── Grep Modal ('g') ────────────────────────────────────────────────────


class GrepModal(ModalScreen[Optional[tuple[Path, int]]]):
    """Cross-file grep over .claude/rules + learnings.md + brain/.

    Returns (file_path, line_number) on selection, or None on cancel.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "open_selected", "Open", show=True),
    ]

    DEFAULT_CSS = """
    GrepModal { align: center middle; }
    #grep-dialog {
        width: 100; height: 30; background: #0a0a0f;
        border: double #ff00ff; padding: 1 2;
    }
    #grep-title { text-align: center; color: #ff00ff; text-style: bold; padding: 0 0 1 0; }
    #grep-input { margin: 1 0; border: solid #ff00ff; background: #131320; }
    #grep-results { height: 1fr; background: #0a0a0f; color: #cccccc; }
    """

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self._project_root = project_root
        self._results: list[tuple[Path, int, str]] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="grep-dialog"):
            yield Label("◈ GREP RULES / LEARNINGS / BRAIN", id="grep-title")
            yield Input(placeholder="regex…", id="grep-input")
            yield Static("", id="grep-results", expand=True)

    def on_mount(self) -> None:
        self.query_one("#grep-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        term = (event.value or "").strip()
        if not term or len(term) < 2:
            self._results = []
            self.query_one("#grep-results", Static).update("[dim]type ≥2 chars to search…[/dim]")
            return
        self._results = self._grep(term)
        self._render_results()

    def _grep(self, term: str) -> list[tuple[Path, int, str]]:
        try:
            pat = re.compile(term, re.IGNORECASE)
        except re.error:
            return []
        out: list[tuple[Path, int, str]] = []
        files = _collect_files(self._project_root)
        all_files = files["rules"] + files["learnings"] + files["brain"]
        for fp in all_files:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if pat.search(line):
                    out.append((fp, i, line.rstrip()))
                    if len(out) >= 200:
                        return out
        return out

    def _render_results(self) -> None:
        if not self._results:
            self.query_one("#grep-results", Static).update("[dim]no matches[/dim]")
            return
        t = Text()
        for idx, (fp, ln, line) in enumerate(self._results[:50]):
            rel = fp.relative_to(self._project_root) if fp.is_absolute() else fp
            marker = "▶ " if idx == 0 else "  "
            t.append(f"{marker}{rel}:{ln}: ", style="bold cyan")
            t.append(line[:120] + "\n", style="white")
        self.query_one("#grep-results", Static).update(t)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_open_selected()

    def action_open_selected(self) -> None:
        if not self._results:
            self.dismiss(None)
            return
        fp, ln, _ = self._results[0]
        self.dismiss((fp, ln))

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── Main Screen ─────────────────────────────────────────────────────────


class RulesScreen(Static):
    """Tree + Markdown viewer for Buddy's reasoning artifacts (read-only)."""

    BINDINGS = [
        Binding("slash", "open_search", "Search", show=True),
        Binding("g", "open_grep", "Grep", show=True),
        Binding("r", "refresh_files", "Refresh", show=True),
    ]

    DEFAULT_CSS = """
    RulesScreen { layout: vertical; height: 1fr; }
    #rules-body { height: 1fr; }
    #rules-tree-pane { width: 36; border-right: solid #2a2a4a; }
    #rules-md-pane { width: 1fr; padding: 0 1; }
    #rules-tree { background: #0a0a0f; color: #cccccc; }
    #rules-status { height: 1; color: #6666aa; padding: 0 1; }
    #rules-search-bar { height: 1; color: #ffab00; padding: 0 1; background: #131320; }
    """

    def __init__(self, project_root: str | Path | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._project_root = Path(project_root) if project_root else \
            Path(__file__).resolve().parent.parent.parent.parent
        self._path_by_node: dict[int, Path] = {}
        self._current_path: Optional[Path] = None
        self._current_text: str = ""
        self._search_term: str = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="rules-body"):
            with Vertical(id="rules-tree-pane"):
                yield Label("⟨ REASONING ARTIFACTS ⟩", classes="panel-title")
                tree: Tree[str] = Tree("◈ .claude", id="rules-tree")
                tree.show_root = True
                tree.guide_depth = 2
                yield tree
            with Vertical(id="rules-md-pane"):
                yield Static("", id="rules-search-bar")
                yield Markdown("", id="rules-md")
                yield Static("", id="rules-status")

    def on_mount(self) -> None:
        self._populate_tree()
        self._set_status("ready — '/' search · 'g' grep · 'r' refresh")

    def on_show(self) -> None:
        """Re-read files each time the tab is activated."""
        self._populate_tree()
        if self._current_path is not None and self._current_path.exists():
            self._render_file(self._current_path)

    # ── Tree population ──────────────────────────────────────────────

    def _populate_tree(self) -> None:
        try:
            tree = self.query_one("#rules-tree", Tree)
        except Exception:
            return
        tree.clear()
        self._path_by_node.clear()

        files = _collect_files(self._project_root)

        rules_node = tree.root.add("rules/", expand=True)
        for fp in files["rules"]:
            leaf = rules_node.add_leaf(fp.name)
            self._path_by_node[leaf.id] = fp

        learn_node = tree.root.add("learnings/", expand=True)
        for fp in files["learnings"]:
            leaf = learn_node.add_leaf(fp.name)
            self._path_by_node[leaf.id] = fp

        brain_node = tree.root.add("brain/", expand=True)
        for fp in files["brain"]:
            leaf = brain_node.add_leaf(fp.name)
            self._path_by_node[leaf.id] = fp

        tree.root.expand()

    # ── Selection → render ───────────────────────────────────────────

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        node: TreeNode = event.node
        fp = self._path_by_node.get(node.id)
        if fp is None:
            return
        self._render_file(fp)

    def _render_file(self, fp: Path) -> None:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            self._set_status(f"[red]error reading {fp.name}: {e}[/red]")
            return
        self._current_path = fp
        self._current_text = text
        try:
            md = self.query_one("#rules-md", Markdown)
            md.update(text)
        except Exception:
            pass
        rel = fp.relative_to(self._project_root) if fp.is_absolute() else fp
        self._set_status(f"viewing {rel}  ({len(text.splitlines())} lines)")
        # Reapply search highlight if term active
        if self._search_term:
            self._apply_search_highlight(self._search_term)

    def _set_status(self, msg: str) -> None:
        try:
            self.query_one("#rules-status", Static).update(msg)
        except Exception:
            pass

    # ── Search ('/') ─────────────────────────────────────────────────

    def action_open_search(self) -> None:
        def _on_done(term: Optional[str]) -> None:
            if not term:
                self._search_term = ""
                self._clear_search_bar()
                return
            self._search_term = term
            self._apply_search_highlight(term)

        self.app.push_screen(SearchModal(), _on_done)

    def _apply_search_highlight(self, term: str) -> None:
        """Count matches in current file, surface in the search bar.

        We can't easily restyle Markdown widget's rendered output, so we
        report match counts + first-match line as a status line. Operator
        can then page to the line.
        """
        if not self._current_text:
            self._clear_search_bar()
            return
        try:
            pat = re.compile(re.escape(term), re.IGNORECASE)
        except re.error:
            self._clear_search_bar()
            return
        matches: list[tuple[int, str]] = []
        for i, line in enumerate(self._current_text.splitlines(), start=1):
            if pat.search(line):
                matches.append((i, line.strip()[:80]))
        try:
            bar = self.query_one("#rules-search-bar", Static)
            if not matches:
                bar.update(f"[yellow]/ {term}[/yellow]  no matches")
            else:
                first_ln, first_text = matches[0]
                bar.update(
                    f"[yellow]/ {term}[/yellow]  {len(matches)} match(es) — "
                    f"first @ line {first_ln}: {first_text}"
                )
        except Exception:
            pass

    def _clear_search_bar(self) -> None:
        try:
            self.query_one("#rules-search-bar", Static).update("")
        except Exception:
            pass

    # ── Grep ('g') ───────────────────────────────────────────────────

    def action_open_grep(self) -> None:
        def _on_done(result: Optional[tuple[Path, int]]) -> None:
            if result is None:
                return
            fp, ln = result
            self._render_file(fp)
            self._set_status(f"jumped to {fp.name}:{ln}")

        self.app.push_screen(GrepModal(self._project_root), _on_done)

    def action_refresh_files(self) -> None:
        self._populate_tree()
        if self._current_path is not None and self._current_path.exists():
            self._render_file(self._current_path)
        self._set_status("refreshed file list")
