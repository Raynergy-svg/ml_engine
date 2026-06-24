"""Tier 2 T7: Brain editor screen — two-pane section CRUD.

Two-pane Textual screen for editing ``.claude/brain/briefing.md``:
- Left pane: list of ``## ``-headed sections.
- Right pane: ``TextArea`` bound to the selected section's body, with
  ``Save`` (Ctrl+S) and ``Reset to template`` buttons.

The screen mounts the underlying ``BriefingDocument`` on entry, refreshes
the list on every reset, and surfaces ``brain_caps`` ValueErrors via the
Textual ``notify`` toast. Escape returns to the prior screen.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    TextArea,
)

from src.tui.widgets.brain_editor import BriefingDocument


class BrainEditorScreen(Screen):
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("ctrl+s", "save", "Save"),
    ]

    def __init__(
        self,
        *,
        briefing_path: Optional[Path] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        root = Path(__file__).resolve().parents[3]
        self._briefing_path = (
            briefing_path or root / ".claude" / "brain" / "briefing.md"
        )
        self._template_path = root / ".claude" / "brain" / "briefing.md.default"
        self._doc = BriefingDocument()
        self._selected_title: Optional[str] = None
        # ListItem ids must be valid Textual identifiers ([A-Za-z0-9_-]); a
        # section title like "Next Actions" contains a space and raises
        # BadIdentifier. We give each row a positional slug id and map it back
        # to the human title here.
        self._id_to_title: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield ListView(id="brain_sections")
            with Vertical():
                yield TextArea("", id="brain_body")
                with Horizontal():
                    yield Button("Save (Ctrl+S)", id="save", variant="primary")
                    yield Button(
                        "Reset to template", id="reset", variant="warning"
                    )
        yield Footer()

    def on_mount(self) -> None:
        if self._briefing_path.exists():
            try:
                self._doc = BriefingDocument.from_path(self._briefing_path)
            except OSError as e:
                self.notify(
                    f"Failed to load briefing: {e}", severity="error"
                )
                self._doc = BriefingDocument()
        self._render_list()

    def _render_list(self) -> None:
        lst = self.query_one("#brain_sections", ListView)
        lst.clear()
        # Positional slug id (see __init__): the human title can't be a widget
        # id. Map slug -> title so selection can recover the real section.
        self._id_to_title = {}
        for i, s in enumerate(self._doc.sections):
            item_id = f"brain-sec-{i}"
            self._id_to_title[item_id] = s.title
            lst.append(ListItem(Label(s.title), id=item_id))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item is None or event.item.id is None:
            return
        self._selected_title = self._id_to_title.get(event.item.id)
        if self._selected_title is None:
            return
        s = next(
            (s for s in self._doc.sections if s.title == self._selected_title),
            None,
        )
        body = self.query_one("#brain_body", TextArea)
        body.text = s.body if s else ""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.action_save()
        elif event.button.id == "reset":
            try:
                self._doc.reset_to_template(
                    self._template_path, target=self._briefing_path
                )
                self._doc = BriefingDocument.from_path(self._briefing_path)
                self._render_list()
                self.notify(
                    "Briefing reset to template", severity="information"
                )
            except (OSError, ValueError) as e:
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
        except OSError as e:
            self.notify(f"Save failed: {e}", severity="error")
