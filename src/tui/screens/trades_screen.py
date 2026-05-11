"""
TRADES SCREEN -- "The Trading Floor"

Three-panel layout:
  1. OPEN POSITIONS -- DataTable of live trades with P/L coloring
  2. PAIR DRILL-DOWN -- Expanded detail for the selected row
  3. RECENT CLOSED -- DataTable of recently closed trades from the journal

Drop-in replacement for PlaceholderContent in the Trades TabPane.
"""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.text import Text

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Label, Static

from src.tui.data_provider import DashboardSnapshot, TradeRow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (matches theme.tcss)
# ---------------------------------------------------------------------------
_BG = "#131320"
_BORDER = "#2a2a4a"
_PRIMARY = "#00ffcc"
_SECONDARY = "#ff00ff"
_POSITIVE = "#00ff41"
_NEGATIVE = "#ff1744"
_WARNING = "#ffab00"
_TEXT = "#e0e0ff"
_DIM = "#6666aa"
_DATA = "#7c4dff"

# Base weights (same order as data_provider)
_BASE_WEIGHTS: dict[str, float] = {
    "trend": 1.15, "mean_reversion": 0.90, "volatility": 1.00,
    "risk_sentinel": 1.25, "uncertainty": 1.10, "execution_quality": 1.05,
    "momentum": 1.05, "news_risk": 0.95, "multi_timeframe": 1.10,
    "pair_performance": 0.85, "session_timing": 0.80,
    "support_resistance": 1.00,
}


# ---------------------------------------------------------------------------
# Sparkline helper
# ---------------------------------------------------------------------------

def _sparkline_str(prices: list[float], width: int = 20) -> str:
    """Convert a list of prices into a unicode sparkline string."""
    if not prices or len(prices) < 2:
        return "░" * width
    mn, mx = min(prices), max(prices)
    rng = mx - mn if mx != mn else 0.001
    blocks = "▁▂▃▄▅▆▇█"
    subset = prices[-width:]
    return "".join(blocks[min(7, int((p - mn) / rng * 7))] for p in subset)


def _format_duration(minutes: float) -> str:
    """Format duration in minutes to 'Xh Ym' display string."""
    if minutes <= 0:
        return "--"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if hours > 0:
        return f"{hours}h {mins:02d}m"
    return f"{mins}m"


# ---------------------------------------------------------------------------
# Drill-Down Panel
# ---------------------------------------------------------------------------

class DrillDownPanel(Static):
    """Expanded detail view for a selected open trade.

    Shows entry/SL/TP, agent breakdown with bar charts, and H1 sparkline.
    Hidden when no trade is selected.
    """

    DEFAULT_CSS = """
    DrillDownPanel {
        height: auto;
        min-height: 3;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._trade: TradeRow | None = None
        self._agent_scores: list[dict[str, Any]] = []
        self._candles_h1: list[float] = []
        self._confidence: float = 0.0
        self._rr_ratio: float = 0.0
        self._regime: str = "NORMAL"
        self._spread: float = 0.0
        self._atr: float = 0.0

    def set_trade(
        self,
        trade: TradeRow | None,
        agent_scores: list[dict[str, Any]] | None = None,
        candles_h1: list[float] | None = None,
        confidence: float = 0.0,
        rr_ratio: float = 0.0,
        regime: str = "NORMAL",
        spread: float = 0.0,
        atr: float = 0.0,
    ) -> None:
        self._trade = trade
        self._agent_scores = agent_scores or []
        self._candles_h1 = candles_h1 or []
        self._confidence = confidence
        self._rr_ratio = rr_ratio
        self._regime = regime
        self._spread = spread
        self._atr = atr
        self.refresh()

    def render(self) -> Text:
        t = Text()
        tr = self._trade

        if tr is None:
            t.append("  Select a trade row above to see details", style=_DIM)
            return t

        # -- Header line --
        pnl_color = _POSITIVE if tr.pnl_pips >= 0 else _NEGATIVE
        pnl_sign = "+" if tr.pnl_pips >= 0 else ""
        t.append(f"  PAIR DRILL-DOWN: ", style=_DIM)
        t.append(f"{tr.pair}", style=f"bold {_PRIMARY}")
        t.append(f"  {pnl_sign}${tr.pnl_pips:,.0f}", style=f"bold {pnl_color}")
        t.append("\n")

        # -- Row 1: Direction, Confidence, R:R --
        dir_color = _POSITIVE if tr.direction == "BUY" else _NEGATIVE
        t.append("  Direction: ", style=_DIM)
        t.append(f"{tr.direction}", style=f"bold {dir_color}")
        t.append("  │  ", style=_BORDER)
        t.append("Confidence: ", style=_DIM)
        conf_color = _POSITIVE if self._confidence >= 0.70 else _WARNING if self._confidence >= 0.55 else _NEGATIVE
        t.append(f"{self._confidence:.2f}", style=f"bold {conf_color}")
        t.append("  │  ", style=_BORDER)
        t.append("R:R: ", style=_DIM)
        rr_color = _POSITIVE if self._rr_ratio >= 1.5 else _WARNING if self._rr_ratio >= 1.2 else _NEGATIVE
        t.append(f"{self._rr_ratio:.1f}:1", style=f"bold {rr_color}")
        t.append("\n")

        # -- Row 2: Entry, SL, TP --
        t.append("  Entry: ", style=_DIM)
        t.append(f"{tr.entry_price:.5f}", style=f"bold {_TEXT}")
        t.append("  │  ", style=_BORDER)
        t.append("SL: ", style=_DIM)
        sl_str = f"{tr.sl_price:.5f}" if tr.sl_price else "---"
        t.append(sl_str, style=f"bold {_NEGATIVE}")
        t.append("  │  ", style=_BORDER)
        t.append("TP: ", style=_DIM)
        tp_str = f"{tr.tp_price:.5f}" if tr.tp_price else "---"
        t.append(tp_str, style=f"bold {_POSITIVE}")
        t.append("\n")

        # -- Row 3: ATR, Regime, Spread --
        t.append("  ATR: ", style=_DIM)
        t.append(f"{self._atr:.1f} pips", style=_TEXT)
        t.append("  │  ", style=_BORDER)
        t.append("Regime: ", style=_DIM)
        t.append(f"{self._regime}", style=f"bold {_DATA}")
        t.append("  │  ", style=_BORDER)
        t.append("Spread: ", style=_DIM)
        t.append(f"{self._spread:.1f} pips", style=_TEXT)
        t.append("\n\n")

        # -- Agent Breakdown --
        if self._agent_scores:
            t.append("  Agent Breakdown:\n", style=f"bold {_DIM}")
            # Display agents in two columns
            agents = self._agent_scores
            mid = (len(agents) + 1) // 2
            col1 = agents[:mid]
            col2 = agents[mid:]

            for i in range(max(len(col1), len(col2))):
                # Left column
                if i < len(col1):
                    a = col1[i]
                    name = a.get("name", "???")[:6]
                    score = float(a.get("score", 0.0))
                    s = max(0.0, min(1.0, score))
                    filled = int(s * 10)
                    empty = 10 - filled
                    color = _POSITIVE if s >= 0.7 else _WARNING if s >= 0.5 else _NEGATIVE
                    t.append(f"  {name:<6}", style=_DIM)
                    t.append(f" {s:.2f} ", style=f"bold {color}")
                    t.append("█" * filled, style=color)
                    t.append("░" * empty, style=_BORDER)
                else:
                    t.append("                          ")

                t.append("  │  ", style=_BORDER)

                # Right column
                if i < len(col2):
                    a = col2[i]
                    name = a.get("name", "???")[:6]
                    score = float(a.get("score", 0.0))
                    s = max(0.0, min(1.0, score))
                    filled = int(s * 10)
                    empty = 10 - filled
                    color = _POSITIVE if s >= 0.7 else _WARNING if s >= 0.5 else _NEGATIVE
                    t.append(f"{name:<6}", style=_DIM)
                    t.append(f" {s:.2f} ", style=f"bold {color}")
                    t.append("█" * filled, style=color)
                    t.append("░" * empty, style=_BORDER)

                t.append("\n")

        # -- H1 Candles Sparkline --
        t.append("\n")
        t.append("  H1 Candles: ", style=_DIM)
        spark = _sparkline_str(self._candles_h1, 30)
        t.append(spark, style=_PRIMARY)
        t.append("\n")

        return t


# ---------------------------------------------------------------------------
# TradeCloseModal — per-trade market-close confirmation
# ---------------------------------------------------------------------------

class TradeCloseModal(ModalScreen[bool]):
    """Confirm market-close for a single open trade.

    Returns True on confirm, False on cancel/escape.
    """

    BINDINGS = [Binding("escape", "dismiss_cancel", "Cancel", show=True)]

    DEFAULT_CSS = """
    TradeCloseModal {
        align: center middle;
    }

    #close-dialog {
        width: 58;
        height: auto;
        background: #0a0a0f;
        border: double #ff00ff;
        padding: 1 2;
    }

    #close-title {
        text-align: center;
        color: #ff00ff;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #close-details {
        padding: 0 0 1 0;
        color: #e0e0ff;
    }

    #close-warning {
        text-align: center;
        color: #ffab00;
        padding: 0 0 1 0;
    }

    #close-buttons {
        align: center middle;
        height: 3;
    }

    #close-confirm {
        margin: 0 1;
    }

    #close-cancel {
        margin: 0 1;
    }
    """

    def __init__(self, trade: TradeRow) -> None:
        super().__init__()
        self._trade = trade

    def compose(self) -> ComposeResult:
        tr = self._trade
        pnl_sign = "+" if tr.pnl_pips >= 0 else ""
        pnl_str = f"{pnl_sign}${tr.pnl_pips:,.2f}"
        dir_label = "BUY (Long)" if tr.direction == "BUY" else "SELL (Short)"

        with Vertical(id="close-dialog"):
            yield Label("✕  CLOSE TRADE  ✕", id="close-title")
            yield Static(
                f"  Pair:          {tr.pair}\n"
                f"  Direction:     {dir_label}\n"
                f"  Entry:         {tr.entry_price:.5f}\n"
                f"  SL / TP:       {tr.sl_price:.5f} / {tr.tp_price:.5f}\n"
                f"  Qty:           {tr.quantity:,.0f} units\n"
                f"  Current P/L:   {pnl_str}\n"
                f"  Trade ID:      {tr.trade_id}",
                id="close-details",
            )
            yield Label(
                "Market-close at current bid/ask — cannot be undone.",
                id="close-warning",
            )
            with Horizontal(id="close-buttons"):
                yield Button("✓ Confirm Close", id="close-confirm", variant="error")
                yield Button("✕ Cancel", id="close-cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-confirm":
            self.dismiss(True)
        elif event.button.id == "close-cancel":
            self.dismiss(False)

    def action_dismiss_cancel(self) -> None:
        self.dismiss(False)


# ---------------------------------------------------------------------------
# TradesScreen -- Main Container
# ---------------------------------------------------------------------------

class TradesScreen(Container):
    """The Trading Floor -- F2 screen.

    Three-panel layout:
      - Open positions DataTable (top)
      - Drill-down detail panel (middle, conditional)
      - Recent closed trades DataTable (bottom)
    """

    BINDINGS = [
        Binding("c", "close_selected_trade", "Close Trade", show=True),
        Binding("slash", "open_search", "Search", show=True),
    ]

    DEFAULT_CSS = """
    TradesScreen {
        height: 1fr;
        layout: vertical;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._trades: list[TradeRow] = []
        self._closed_trades: list[dict[str, Any]] = []
        self._agent_scores: list[dict[str, Any]] = []
        self._candles_h1: list[float] = []
        self._selected_index: int | None = None
        self._project_root: Path = Path(__file__).resolve().parents[3]

    def compose(self) -> ComposeResult:
        with Vertical(id="trades-open-panel", classes="panel"):
            yield Label("  OPEN POSITIONS", classes="panel-title")
            yield DataTable(id="trades-open-table", cursor_type="row")

        yield DrillDownPanel(id="trades-drilldown")

        with Vertical(id="trades-closed-panel", classes="panel"):
            yield Label("  RECENT CLOSED", classes="panel-title")
            yield DataTable(id="trades-closed-table", cursor_type="row")

    def on_mount(self) -> None:
        # Set up open positions table
        open_table = self.query_one("#trades-open-table", DataTable)
        open_table.add_columns("Pair", "Dir", "Entry", "SL", "TP", "Qty", "P/L")
        open_table.zebra_stripes = True

        # Set up closed trades table
        closed_table = self.query_one("#trades-closed-table", DataTable)
        closed_table.add_columns("Pair", "Dir", "Entry", "Exit", "P/L", "Result", "Time")
        closed_table.zebra_stripes = True

        # Load initial closed trades from journal
        self._load_closed_trades()

    # ------------------------------------------------------------------
    # DataTable row selection -> drill-down
    # ------------------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the open positions table.

        US-512: Enter on a row pushes a GateTraceModal — answers
        "why did this fire?" for executed/rejected/vetoed signals.
        Closed-trades table uses the same handler with its own trade_id list.
        """
        try:
            open_table = self.query_one("#trades-open-table", DataTable)
            closed_table = self.query_one("#trades-closed-table", DataTable)
        except Exception:
            return

        from src.tui.screens.gate_trace_modal import GateTraceModal

        if event.data_table is open_table:
            row_idx = event.cursor_row
            if 0 <= row_idx < len(self._trades):
                self._selected_index = row_idx
                trade = self._trades[row_idx]
                self._update_drilldown(trade)
                if trade.trade_id:
                    self.app.push_screen(GateTraceModal(trade_id=str(trade.trade_id)))
        elif event.data_table is closed_table:
            row_idx = event.cursor_row
            if 0 <= row_idx < len(self._closed_trades):
                tid = self._closed_trades[row_idx].get("trade_id")
                if tid:
                    self.app.push_screen(GateTraceModal(trade_id=str(tid)))

    # ------------------------------------------------------------------
    # Public: update from snapshot
    # ------------------------------------------------------------------

    def update_from_snapshot(self, snap: DashboardSnapshot) -> None:
        """Refresh all sub-widgets from a new DashboardSnapshot."""
        self._trades = list(snap.trades) if snap.trades else []

        # Cache agent scores for drill-down
        if snap.agents:
            self._agent_scores = [
                {"name": a.name, "score": a.score, "signal": a.signal, "weight": a.weight}
                for a in snap.agents
            ]

        # Cache candles for drill-down sparkline
        self._candles_h1 = list(snap.candles_h1) if snap.candles_h1 else []

        self._refresh_open_table()

        # Re-apply drill-down if we had a selection
        if self._selected_index is not None and self._selected_index < len(self._trades):
            self._update_drilldown(self._trades[self._selected_index])
        elif self._trades:
            self._selected_index = 0
            self._update_drilldown(self._trades[0])
        else:
            self._clear_drilldown()

        # Reload closed trades
        self._load_closed_trades()

    # ------------------------------------------------------------------
    # Private: table refresh
    # ------------------------------------------------------------------

    def _refresh_open_table(self) -> None:
        """Rebuild the open positions DataTable from current trades."""
        try:
            table = self.query_one("#trades-open-table", DataTable)
        except Exception:
            return

        table.clear()

        for tr in self._trades:
            pnl_val = tr.pnl_pips
            pnl_sign = "+" if pnl_val >= 0 else ""
            pnl_str = f"{pnl_sign}${pnl_val:,.0f}"

            # Build styled cells
            pair_text = Text(tr.pair, style=_TEXT)
            dir_color = _POSITIVE if tr.direction == "BUY" else _NEGATIVE
            dir_text = Text(tr.direction, style=f"bold {dir_color}")
            entry_text = Text(f"{tr.entry_price:.5f}", style=_TEXT)
            sl_text = Text(f"{tr.sl_price:.5f}" if tr.sl_price else "---", style=_NEGATIVE)
            tp_text = Text(f"{tr.tp_price:.5f}" if tr.tp_price else "---", style=_POSITIVE)
            qty_text = Text(f"{tr.quantity:,.0f}", style=_TEXT)
            pnl_color = _POSITIVE if pnl_val >= 0 else _NEGATIVE
            pnl_text = Text(pnl_str, style=f"bold {pnl_color}")

            table.add_row(pair_text, dir_text, entry_text, sl_text, tp_text, qty_text, pnl_text)

    def _update_drilldown(self, trade: TradeRow) -> None:
        """Update the drill-down panel for a given trade."""
        try:
            panel = self.query_one("#trades-drilldown", DrillDownPanel)
        except Exception:
            return

        # Estimate confidence and R:R from trade SL/TP distances
        sl_dist = abs(trade.entry_price - trade.sl_price) if trade.sl_price else 0
        tp_dist = abs(trade.tp_price - trade.entry_price) if trade.tp_price else 0
        rr = tp_dist / sl_dist if sl_dist > 0 else 0.0

        # Use weighted vote score from agent data if available
        confidence = 0.0
        if self._agent_scores:
            total_score = sum(a.get("score", 0.0) for a in self._agent_scores)
            confidence = total_score / len(self._agent_scores) if self._agent_scores else 0.0

        panel.set_trade(
            trade=trade,
            agent_scores=self._agent_scores,
            candles_h1=self._candles_h1,
            confidence=confidence,
            rr_ratio=rr,
            regime="NORMAL",
            spread=random.uniform(0.8, 2.5),
            atr=sl_dist * 10000 if sl_dist < 1 else sl_dist,  # rough ATR from SL
        )

    def _clear_drilldown(self) -> None:
        """Clear the drill-down panel."""
        try:
            panel = self.query_one("#trades-drilldown", DrillDownPanel)
            panel.set_trade(None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private: closed trades
    # ------------------------------------------------------------------

    def _load_closed_trades(self) -> None:
        """Load recently closed trades from trade_journal_rl.json."""
        try:
            closed_table = self.query_one("#trades-closed-table", DataTable)
        except Exception:
            return

        closed_table.clear()

        journal_path = self._project_root / "trained_data" / "trade_journal_rl.json"
        entries: list[dict[str, Any]] = []

        if journal_path.exists():
            try:
                raw = json.loads(journal_path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    entries = raw
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("Failed to read trade journal: %s", exc)

        # Filter to closed trades (those with outcome.realized_pl)
        closed = [
            e for e in entries
            if isinstance(e.get("outcome"), dict) and "realized_pl" in e.get("outcome", {})
        ]

        # Take last 20, most recent first
        recent = list(reversed(closed[-20:]))
        self._closed_trades = recent

        for entry in recent:
            outcome = entry.get("outcome", {})
            pair = str(entry.get("pair", "---")).replace("_", "/")
            direction = str(entry.get("direction", "---"))
            entry_price = float(entry.get("entry_price", 0.0))
            exit_price = float(outcome.get("exit_price", outcome.get("close_price", 0.0)))
            realized_pl = float(outcome.get("realized_pl", 0.0))
            trade_won = bool(outcome.get("trade_won", False))
            duration_min = float(outcome.get("duration_minutes", 0.0))

            pnl_sign = "+" if realized_pl >= 0 else ""
            pnl_str = f"{pnl_sign}${realized_pl:,.0f}"
            result_str = "WIN" if trade_won else "LOSS"
            time_str = _format_duration(duration_min)

            # Styled cells
            pair_text = Text(pair, style=_TEXT)

            dir_color = _POSITIVE if direction in ("BUY", "LONG") else _NEGATIVE
            dir_text = Text(direction[:4], style=f"bold {dir_color}")

            entry_text = Text(f"{entry_price:.5f}", style=_TEXT)
            exit_text = Text(f"{exit_price:.5f}", style=_TEXT)

            pnl_color = _POSITIVE if realized_pl >= 0 else _NEGATIVE
            pnl_text = Text(pnl_str, style=f"bold {pnl_color}")

            result_color = _POSITIVE if trade_won else _NEGATIVE
            result_text = Text(result_str, style=f"bold {result_color}")

            time_text = Text(time_str, style=_DIM)

            closed_table.add_row(
                pair_text, dir_text, entry_text, exit_text,
                pnl_text, result_text, time_text,
            )

    # ------------------------------------------------------------------
    # US-510: Per-trade close action
    # ------------------------------------------------------------------

    def action_open_search(self) -> None:
        """Hotkey /: open FTS5 trade-journal search modal.

        Rebuilds the index from disk each time so newly-closed trades are
        searchable without a TUI restart. Selection from the modal scrolls
        the closed-trades panel to the picked trade and surfaces an info
        notification if the trade lies outside the last-20 window.
        """
        from src.tui.screens.trade_search_modal import TradeSearchModal
        from src.tui.trade_search_index import TradeSearchIndex

        journal_path = self._project_root / "trained_data" / "trade_journal_rl.json"
        index = TradeSearchIndex.from_journal_file(journal_path)
        if len(index) == 0:
            self.app.notify(
                "Trade journal empty or unreadable.",
                title="Trade Search",
                severity="warning",
            )
            index.close()
            return

        def _on_pick(trade_id: str | None) -> None:
            try:
                if trade_id is None:
                    return
                self._jump_to_closed_trade(trade_id)
            finally:
                index.close()

        self.app.push_screen(TradeSearchModal(index), _on_pick)

    def _jump_to_closed_trade(self, trade_id: str) -> None:
        """Scroll the closed-trades table to the row matching `trade_id`.

        If the trade isn't in the last-20 window currently rendered, surface
        a hint to the operator rather than silently doing nothing.
        """
        for idx, entry in enumerate(self._closed_trades):
            if str(entry.get("trade_id")) == str(trade_id):
                try:
                    table = self.query_one("#trades-closed-table", DataTable)
                    table.move_cursor(row=idx)
                    table.focus()
                except Exception as exc:
                    logger.debug("Trade jump scroll failed for %s: %s", trade_id, exc)
                return
        self.app.notify(
            f"Trade {trade_id} is older than the last 20 closed shown here.",
            title="Trade Search",
            severity="information",
        )

    def action_close_selected_trade(self) -> None:
        """Hotkey C: push TradeCloseModal for the currently selected open trade."""
        if self._selected_index is None or self._selected_index >= len(self._trades):
            self.app.notify("No trade selected.", title="Close Trade", severity="warning")
            return

        trade = self._trades[self._selected_index]
        if not trade.trade_id:
            self.app.notify("Selected trade has no ID.", title="Close Trade", severity="warning")
            return

        def _on_modal_result(confirmed: bool) -> None:
            if confirmed:
                self._do_close_trade(trade.trade_id)

        self.app.push_screen(TradeCloseModal(trade), _on_modal_result)

    @work
    async def _do_close_trade(self, trade_id: str) -> None:
        """Async worker: call ExecutionManager.close_trade and show result toast."""
        from src.scanner.execution import ExecutionManager

        em = ExecutionManager()
        result = await em.close_trade(trade_id, reason="operator")

        if result.get("success"):
            pl = result.get("realized_pl", 0.0)
            pl_sign = "+" if pl >= 0 else ""
            self.app.notify(
                f"Trade {trade_id} closed. P/L: {pl_sign}${pl:,.2f}",
                title="Trade Closed",
                severity="information",
            )
        else:
            err = result.get("error", "Unknown error")
            self.app.notify(
                f"Close failed: {err}",
                title="Close Trade Error",
                severity="error",
            )

        # Refresh both tables so the closed trade moves from open → closed
        self._load_closed_trades()
        # Remove from local trades list so open table reflects reality
        self._trades = [t for t in self._trades if t.trade_id != trade_id]
        if self._selected_index is not None and self._selected_index >= len(self._trades):
            self._selected_index = max(0, len(self._trades) - 1) if self._trades else None
        self._refresh_open_table()
        if self._trades and self._selected_index is not None:
            self._update_drilldown(self._trades[self._selected_index])
        else:
            self._clear_drilldown()
