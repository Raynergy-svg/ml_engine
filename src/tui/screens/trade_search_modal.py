"""TradeSearchModal — `/` from Trades screen, FTS5 search over closed trades.

Lets the operator find any trade (not just the last 20 shown on the screen)
by pair, outcome, regime, agent reason, or trade_id. Dismisses with the
selected trade_id (or None on Escape).
"""
from __future__ import annotations

from typing import Any, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Label, Static

from src.tui.trade_search_index import TradeSearchIndex


class TradeSearchModal(ModalScreen[Optional[str]]):
    """Search modal. Pass in a pre-built TradeSearchIndex.

    Dismisses with the chosen trade_id (str) or None on Escape.
    """

    BINDINGS = [
        Binding("escape", "dismiss_cancel", "Cancel", show=True),
        Binding("up", "move_selection(-1)", "Up", show=False),
        Binding("down", "move_selection(1)", "Down", show=False),
        Binding("enter", "select_current", "Open", show=True),
    ]

    DEFAULT_CSS = """
    TradeSearchModal {
        align: center middle;
    }

    #trade-search-dialog {
        width: 96;
        height: auto;
        max-height: 30;
        background: #0a0a0f;
        border: double #00ffcc;
        padding: 1 2;
    }

    #trade-search-title {
        text-align: center;
        color: #00ffcc;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #trade-search-input {
        margin: 0 0 1 0;
        border: solid #00ffcc;
        background: #131320;
    }

    #trade-search-results {
        height: auto;
        max-height: 18;
        background: #07070b;
    }

    #trade-search-hint {
        color: #6666aa;
        text-align: center;
        padding: 1 0 0 0;
    }

    #trade-search-empty {
        color: #6666aa;
        text-align: center;
        padding: 0;
    }
    """

    DEFAULT_LIMIT = 100

    def __init__(self, index: TradeSearchIndex, *, limit: int = DEFAULT_LIMIT) -> None:
        super().__init__()
        self._index = index
        self._limit = limit
        self._results: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="trade-search-dialog"):
            yield Label(f"⌕  TRADE SEARCH  ({len(self._index)} indexed)", id="trade-search-title")
            yield Input(placeholder="e.g.  eur_aud loss   |   1220   |   ranging   |   trend", id="trade-search-input")
            yield DataTable(id="trade-search-results", cursor_type="row")
            yield Static("[dim]↑↓ move · Enter open · Esc cancel[/]", id="trade-search-hint", markup=True)

    def on_mount(self) -> None:
        table = self.query_one("#trade-search-results", DataTable)
        table.add_columns("ID", "Time", "Pair", "Dir", "Outcome", "P/L", "Regime")
        self._refresh("")
        self.query_one("#trade-search-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "trade-search-input":
            self._refresh(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "trade-search-input":
            self.action_select_current()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_select_current()

    def _refresh(self, query: str) -> None:
        self._results = self._index.search(query, limit=self._limit)
        table = self.query_one("#trade-search-results", DataTable)
        table.clear()
        for entry in self._results:
            table.add_row(*_row_cells(entry))
        if self._results:
            try:
                table.move_cursor(row=0)
            except Exception:
                pass

    def action_move_selection(self, delta: int) -> None:
        table = self.query_one("#trade-search-results", DataTable)
        if not self._results:
            return
        new = (table.cursor_row or 0) + delta
        new = max(0, min(len(self._results) - 1, new))
        try:
            table.move_cursor(row=new)
        except Exception:
            pass

    def action_select_current(self) -> None:
        table = self.query_one("#trade-search-results", DataTable)
        if not self._results:
            return
        row = table.cursor_row or 0
        if 0 <= row < len(self._results):
            tid = self._results[row].get("trade_id")
            self.dismiss(str(tid) if tid is not None else None)

    def action_dismiss_cancel(self) -> None:
        self.dismiss(None)


def _row_cells(entry: dict[str, Any]) -> list[str]:
    """Render one journal entry as a 7-tuple for the DataTable."""
    outcome = entry.get("outcome") if isinstance(entry.get("outcome"), dict) else None
    regime = entry.get("regime") if isinstance(entry.get("regime"), dict) else None

    if outcome and "realized_pl" in outcome:
        if outcome.get("trade_won") is True:
            outcome_str = "✓ win"
        else:
            outcome_str = "✗ loss"
        try:
            pl = float(outcome.get("realized_pl", 0.0))
            pl_str = f"+${pl:,.2f}" if pl >= 0 else f"-${abs(pl):,.2f}"
        except (TypeError, ValueError):
            pl_str = "—"
    else:
        outcome_str = "open"
        pl_str = "—"

    ts = str(entry.get("timestamp", ""))[:19].replace("T", " ")
    return [
        str(entry.get("trade_id", "")),
        ts,
        str(entry.get("pair", "")).replace("_", "/"),
        str(entry.get("direction", "")),
        outcome_str,
        pl_str,
        str(regime.get("volatility_regime", "")) if regime else "",
    ]
