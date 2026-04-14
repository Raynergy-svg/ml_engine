"""
AGENTS SCREEN -- "The Agent Lab"

Four-panel layout:
  1. AGENT WEIGHT MATRIX -- DataTable: base vs learned vs effective weights
  2. ACCURACY BY AGENT -- Per-agent win rate bars (left column)
  3. VOTING HISTORY -- Last scan agent vote breakdown (right column)
  4. MODEL DISAGREEMENT -- Sparkline of recent disagreement with status

Drop-in replacement for PlaceholderContent in the Agents TabPane.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from rich.text import Text

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Label, Static

from src.tui.data_provider import AgentScore, DashboardSnapshot

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

# Agent display name mapping (full name -> short 6-char label)
_AGENT_SHORT_NAMES: dict[str, str] = {
    "trend": "trend",
    "mean_reversion": "m_rev",
    "volatility": "volat",
    "risk_sentinel": "risk",
    "uncertainty": "uncrt",
    "execution_quality": "exec",
    "momentum": "momt",
    "news_risk": "news",
    "multi_timeframe": "mtf",
    "pair_performance": "pair",
    "session_timing": "sess",
    "support_resistance": "sr",
    "trader_readiness": "ready",
    "devil_advocate": "devil",
}

# Canonical base weights
_BASE_WEIGHTS: dict[str, float] = {
    "trend": 1.15, "mean_reversion": 0.90, "volatility": 1.00,
    "risk_sentinel": 1.25, "uncertainty": 1.10, "execution_quality": 1.05,
    "momentum": 1.05, "news_risk": 0.95, "multi_timeframe": 1.10,
    "pair_performance": 0.85, "session_timing": 0.80,
    "support_resistance": 1.00,
}


# ---------------------------------------------------------------------------
# Helper: sparkline from floats
# ---------------------------------------------------------------------------

def _sparkline_str(values: list[float], width: int = 30) -> str:
    """Convert a list of values into a unicode sparkline string."""
    if not values or len(values) < 2:
        return "░" * width
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 0.001
    blocks = "▁▂▃▄▅▆▇█"
    subset = values[-width:]
    return "".join(blocks[min(7, int((v - mn) / rng * 7))] for v in subset)


def _bar_str(value: float, max_val: float = 1.5, width: int = 14) -> str:
    """Render a small horizontal bar from a weight value."""
    normalized = max(0.0, min(1.0, value / max_val))
    filled = int(normalized * width)
    empty = width - filled
    return "█" * filled + "░" * empty


# ---------------------------------------------------------------------------
# AccuracyPanel -- per-agent win rate
# ---------------------------------------------------------------------------

class AccuracyPanel(Static):
    """Per-agent accuracy (win rate when that agent voted PASS).

    Data comes from JournalReader.get_agent_accuracy() or demo defaults.
    """

    DEFAULT_CSS = """
    AccuracyPanel {
        height: auto;
        min-height: 8;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._accuracy: dict[str, float] = {}

    def set_accuracy(self, accuracy: dict[str, float]) -> None:
        self._accuracy = dict(accuracy)
        self.refresh()

    def render(self) -> Text:
        t = Text()

        if not self._accuracy:
            t.append("  No accuracy data yet\n", style=_DIM)
            t.append("  (need closed trades)\n", style=_DIM)
            return t

        for agent_name, win_rate in sorted(self._accuracy.items(), key=lambda x: -x[1]):
            short = _AGENT_SHORT_NAMES.get(agent_name, agent_name[:6])
            pct = win_rate * 100
            color = _POSITIVE if pct >= 70 else _WARNING if pct >= 50 else _NEGATIVE
            filled = int(win_rate * 10)
            empty = 10 - filled

            t.append(f"  {short:<6}", style=_DIM)
            t.append(f" {pct:3.0f}% ", style=f"bold {color}")
            t.append("█" * filled, style=color)
            t.append("░" * empty, style=_BORDER)
            t.append("\n")

        # Average accuracy line
        if self._accuracy:
            avg = sum(self._accuracy.values()) / len(self._accuracy) * 100
            avg_color = _POSITIVE if avg >= 65 else _WARNING if avg >= 50 else _NEGATIVE
            t.append("  ─────────────────────────\n", style=_BORDER)
            t.append("  Avg accuracy: ", style=_DIM)
            t.append(f"{avg:.0f}%", style=f"bold {avg_color}")

        return t


# ---------------------------------------------------------------------------
# VotingHistoryPanel -- last scan's agent votes
# ---------------------------------------------------------------------------

class VotingHistoryPanel(Static):
    """Shows the breakdown of agent votes from the most recent scan/trade."""

    DEFAULT_CSS = """
    VotingHistoryPanel {
        height: auto;
        min-height: 8;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._votes: list[dict[str, Any]] = []
        self._pair: str = ""
        self._scan_id: int = 0
        self._result: str = ""

    def set_votes(
        self,
        votes: list[dict[str, Any]],
        pair: str = "",
        scan_id: int = 0,
        result: str = "",
    ) -> None:
        self._votes = list(votes)
        self._pair = pair
        self._scan_id = scan_id
        self._result = result
        self.refresh()

    def render(self) -> Text:
        t = Text()

        if not self._votes:
            t.append("  No voting data yet\n", style=_DIM)
            t.append("  (waiting for scan)\n", style=_DIM)
            return t

        # Header
        if self._scan_id > 0 or self._pair:
            t.append(f"  Scan", style=_DIM)
            if self._scan_id:
                t.append(f" #{self._scan_id}", style=f"bold {_DATA}")
            if self._pair:
                t.append(f": {self._pair.replace('_', '/')}", style=f"bold {_PRIMARY}")
            t.append("\n")

        # Vote rows
        for vote in self._votes:
            name = str(vote.get("name", "???"))
            short = _AGENT_SHORT_NAMES.get(name, name[:6])
            passed = bool(vote.get("passed", False))
            score = float(vote.get("score", 0.0))
            reason_code = str(vote.get("reason_code", ""))

            # Determine signal text from reason code or direction
            signal = "PASS" if passed else "FAIL"
            if "bullish" in reason_code or "buy" in reason_code.lower():
                signal = "BUY" if passed else "SKIP"
            elif "bearish" in reason_code or "sell" in reason_code.lower():
                signal = "SELL" if passed else "SKIP"
            elif "ok" in reason_code or "safe" in reason_code:
                signal = "PASS"
            elif "neutral" in reason_code:
                signal = "NEUT"

            check = "  ✓" if passed else "  ✗"
            check_color = _POSITIVE if passed else _NEGATIVE

            t.append(f"  {short:<6}", style=_DIM)
            t.append(f" {signal:<5}", style=f"bold {_TEXT}")
            t.append(check, style=check_color)
            t.append("\n")

        # Result footer
        if self._result:
            result_upper = self._result.upper()
            if "WIN" in result_upper:
                t.append(f"  Result: ", style=_DIM)
                t.append(f"TRADE -> WIN", style=f"bold {_POSITIVE}")
            elif "LOSS" in result_upper:
                t.append(f"  Result: ", style=_DIM)
                t.append(f"TRADE -> LOSS", style=f"bold {_NEGATIVE}")
            elif "SKIP" in result_upper or "REJECT" in result_upper:
                t.append(f"  Result: ", style=_DIM)
                t.append(f"REJECTED", style=f"bold {_WARNING}")
            else:
                t.append(f"  Result: ", style=_DIM)
                t.append(f"{result_upper}", style=f"bold {_DATA}")

        return t


# ---------------------------------------------------------------------------
# DisagreementPanel -- model disagreement sparkline + status
# ---------------------------------------------------------------------------

class DisagreementPanel(Static):
    """Shows model disagreement sparkline and current status."""

    DEFAULT_CSS = """
    DisagreementPanel {
        height: auto;
        min-height: 4;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._history: list[float] = []
        self._current: float = 0.0
        self._threshold: float = 0.30

    def set_disagreement(
        self,
        history: list[float],
        current: float = 0.0,
        threshold: float = 0.30,
    ) -> None:
        self._history = list(history)
        self._current = current
        self._threshold = threshold
        self.refresh()

    def render(self) -> Text:
        t = Text()

        # Sparkline
        t.append("  Last 20 scans disagreement:  ", style=_DIM)
        spark = _sparkline_str(self._history, 20)
        spark_color = _POSITIVE if self._current < self._threshold else _NEGATIVE
        t.append(spark, style=spark_color)
        t.append("\n")

        # Metrics line
        avg = sum(self._history) / len(self._history) if self._history else 0.0
        cur_color = _POSITIVE if self._current < self._threshold else _NEGATIVE

        t.append("  Current: ", style=_DIM)
        t.append(f"{self._current:.2f}", style=f"bold {cur_color}")
        t.append("  │  ", style=_BORDER)
        t.append("Avg: ", style=_DIM)
        t.append(f"{avg:.2f}", style=_TEXT)
        t.append("  │  ", style=_BORDER)
        t.append("Threshold: ", style=_DIM)
        t.append(f"{self._threshold:.2f}", style=f"bold {_WARNING}")
        t.append("\n")

        # Status
        within = self._current < self._threshold
        status_text = "WITHIN LIMITS" if within else "EXCEEDS THRESHOLD"
        status_icon = "✓" if within else "✗"
        status_color = _POSITIVE if within else _NEGATIVE

        t.append("  Status: ", style=_DIM)
        t.append(f"{status_icon} {status_text}", style=f"bold {status_color}")

        return t


# ---------------------------------------------------------------------------
# AgentsScreen -- Main Container
# ---------------------------------------------------------------------------

class AgentsScreen(Container):
    """The Agent Lab -- F3 screen.

    Four-panel layout:
      - Agent weight matrix DataTable (top, full width)
      - Accuracy panel (middle-left) + Voting history (middle-right)
      - Model disagreement sparkline (bottom, full width)
    """

    DEFAULT_CSS = """
    AgentsScreen {
        height: 1fr;
        layout: vertical;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._agents: list[AgentScore] = []
        self._learned_weights: dict[str, float] = {}
        self._regime: str = "NORMAL"
        self._project_root: Path = Path(__file__).resolve().parents[3]
        self._disagreement_history: list[float] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="agents-weight-panel", classes="panel"):
            yield Label("  AGENT WEIGHT MATRIX", classes="panel-title")
            yield DataTable(id="agents-weight-table", cursor_type="row")

        with Horizontal(id="agents-middle-row"):
            with Vertical(id="agents-accuracy-panel", classes="panel"):
                yield Label("  ACCURACY BY AGENT", classes="panel-title")
                yield AccuracyPanel(id="agents-accuracy")

            with Vertical(id="agents-voting-panel", classes="panel"):
                yield Label("  VOTING HISTORY", classes="panel-title")
                yield VotingHistoryPanel(id="agents-voting")

        with Vertical(id="agents-disagreement-panel", classes="panel"):
            yield Label("  MODEL DISAGREEMENT", classes="panel-title")
            yield DisagreementPanel(id="agents-disagreement")

    def on_mount(self) -> None:
        # Set up weight matrix table columns
        weight_table = self.query_one("#agents-weight-table", DataTable)
        weight_table.add_columns("Agent", "Base", "Learned", "Effective", "Bar")
        weight_table.zebra_stripes = True

        # Load initial data
        self._load_weights()
        self._load_journal_data()

    # ------------------------------------------------------------------
    # Public: update from snapshot
    # ------------------------------------------------------------------

    def update_from_snapshot(self, snap: DashboardSnapshot) -> None:
        """Refresh all sub-widgets from a new DashboardSnapshot."""
        if snap.agents:
            self._agents = list(snap.agents)

        self._load_weights()
        self._refresh_weight_table()
        self._load_journal_data()

    # ------------------------------------------------------------------
    # Private: weight matrix
    # ------------------------------------------------------------------

    def _load_weights(self) -> None:
        """Load agent weights from trained_data/models/agent_weights.json."""
        weights_path = self._project_root / "trained_data" / "models" / "agent_weights.json"
        if not weights_path.exists():
            return

        try:
            data = json.loads(weights_path.read_text(encoding="utf-8"))
            # Prefer current regime, fall back to _global
            regime_data = data.get(self._regime, data.get("_global", {}))
            if isinstance(regime_data, dict):
                self._learned_weights = {
                    k: float(v) for k, v in regime_data.items()
                    if isinstance(v, (int, float))
                }

            # Detect regime from keys
            regimes = [k for k in data.keys() if k not in ("_global", "_meta")]
            if regimes and "NORMAL" in regimes:
                self._regime = "NORMAL"
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Failed to load agent weights: %s", exc)

    def _refresh_weight_table(self) -> None:
        """Rebuild the agent weight matrix DataTable."""
        try:
            table = self.query_one("#agents-weight-table", DataTable)
        except Exception:
            return

        table.clear()

        # Merge base + learned weights for all known agents
        all_agents = set(_BASE_WEIGHTS.keys()) | set(self._learned_weights.keys())
        # Exclude non-core agents from display
        display_agents = sorted(
            [a for a in all_agents if a in _AGENT_SHORT_NAMES],
            key=lambda a: _BASE_WEIGHTS.get(a, 1.0),
            reverse=True,
        )

        for agent_key in display_agents:
            short_name = _AGENT_SHORT_NAMES.get(agent_key, agent_key[:6])
            base_w = _BASE_WEIGHTS.get(agent_key, 1.0)
            learned_w = self._learned_weights.get(agent_key, base_w)
            effective_w = learned_w  # In current architecture, effective == learned

            # Color: green if learned > base (improving), red if declining
            if learned_w > base_w + 0.01:
                weight_color = _POSITIVE
            elif learned_w < base_w - 0.01:
                weight_color = _NEGATIVE
            else:
                weight_color = _TEXT

            # Build styled cells
            name_text = Text(f" {short_name:<15}", style=_TEXT)
            base_text = Text(f" {base_w:.2f}", style=_DIM)
            learned_text = Text(f" {learned_w:.3f}", style=f"bold {weight_color}")
            effective_text = Text(f" {effective_w:.3f}", style=f"bold {weight_color}")

            # Bar visualization
            bar = _bar_str(effective_w, max_val=1.5, width=14)
            bar_text = Text(f" {bar}", style=weight_color)

            table.add_row(name_text, base_text, learned_text, effective_text, bar_text)

        # Add regime label as table footer info
        try:
            regime_label = self.query_one("#agents-weight-panel .panel-title", Label)
            regime_label.update(
                Text.from_markup(
                    f"  [bold #00ffcc]AGENT WEIGHT MATRIX[/]"
                    f"  [#6666aa]Regime:[/] [bold #7c4dff]{self._regime}[/]"
                )
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private: journal-based panels
    # ------------------------------------------------------------------

    def _load_journal_data(self) -> None:
        """Load accuracy, voting history, and disagreement from journal."""
        try:
            from src.tui.journal_reader import JournalReader
            reader = JournalReader(
                journal_path=self._project_root / "trained_data" / "trade_journal_rl.json"
            )
            entries = reader.load()
        except Exception as exc:
            logger.debug("Failed to load journal for agents screen: %s", exc)
            self._load_demo_data()
            return

        # -- Accuracy panel --
        try:
            accuracy = reader.get_agent_accuracy()
            accuracy_panel = self.query_one("#agents-accuracy", AccuracyPanel)
            if accuracy:
                accuracy_panel.set_accuracy(accuracy)
            else:
                self._set_demo_accuracy()
        except Exception:
            self._set_demo_accuracy()

        # -- Voting history (last trade's agent votes) --
        try:
            voting_panel = self.query_one("#agents-voting", VotingHistoryPanel)
            if entries:
                last = entries[-1]
                votes = [
                    {
                        "name": v.name,
                        "score": v.score,
                        "passed": v.passed,
                        "reason_code": v.reason_code,
                    }
                    for v in last.agent_votes
                ]
                result = ""
                if last.outcome is not None:
                    result = "WIN" if last.outcome.trade_won else "LOSS"
                elif last.is_closed:
                    result = "CLOSED"
                else:
                    result = "OPEN"

                voting_panel.set_votes(
                    votes=votes,
                    pair=last.pair,
                    scan_id=int(last.trade_id) if last.trade_id.isdigit() else 0,
                    result=result,
                )
            else:
                self._set_demo_voting()
        except Exception:
            self._set_demo_voting()

        # -- Disagreement panel --
        self._update_disagreement(entries)

    def _update_disagreement(self, entries: list | None = None) -> None:
        """Update the disagreement panel from journal entries or demo data."""
        try:
            panel = self.query_one("#agents-disagreement", DisagreementPanel)
        except Exception:
            return

        # Compute disagreement from agent vote variance in recent entries
        history: list[float] = []
        if entries:
            for entry in entries[-20:]:
                if hasattr(entry, "agent_votes") and entry.agent_votes:
                    scores = [v.score for v in entry.agent_votes]
                    if len(scores) >= 2:
                        mean_s = sum(scores) / len(scores)
                        variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
                        disagreement = variance ** 0.5  # std dev as proxy
                        history.append(round(disagreement, 3))

        if not history:
            # Generate plausible demo data
            history = [round(random.uniform(0.10, 0.28), 3) for _ in range(20)]

        self._disagreement_history = history
        current = history[-1] if history else 0.18

        panel.set_disagreement(
            history=history,
            current=current,
            threshold=0.30,
        )

    # ------------------------------------------------------------------
    # Demo data fallbacks
    # ------------------------------------------------------------------

    def _load_demo_data(self) -> None:
        """Populate all panels with demo data when journal is unavailable."""
        self._set_demo_accuracy()
        self._set_demo_voting()
        self._update_disagreement(None)

    def _set_demo_accuracy(self) -> None:
        """Set demo accuracy data."""
        try:
            panel = self.query_one("#agents-accuracy", AccuracyPanel)
            demo = {
                "trend": 0.67, "risk_sentinel": 0.82, "momentum": 0.61,
                "uncertainty": 0.74, "volatility": 0.58, "mean_reversion": 0.53,
                "execution_quality": 0.71, "news_risk": 0.65,
                "multi_timeframe": 0.69, "support_resistance": 0.63,
                "session_timing": 0.55, "pair_performance": 0.48,
            }
            panel.set_accuracy(demo)
        except Exception:
            pass

    def _set_demo_voting(self) -> None:
        """Set demo voting history data."""
        try:
            panel = self.query_one("#agents-voting", VotingHistoryPanel)
            demo_votes = [
                {"name": "trend", "score": 0.82, "passed": True, "reason_code": "trend_bullish"},
                {"name": "risk_sentinel", "score": 0.91, "passed": True, "reason_code": "risk_ok"},
                {"name": "momentum", "score": 0.74, "passed": True, "reason_code": "momentum_strong"},
                {"name": "uncertainty", "score": 0.31, "passed": True, "reason_code": "uncertainty_ok"},
                {"name": "volatility", "score": 0.65, "passed": True, "reason_code": "volatility_support"},
                {"name": "mean_reversion", "score": 0.45, "passed": False, "reason_code": "mean_reversion_neutral"},
            ]
            panel.set_votes(
                votes=demo_votes,
                pair="EUR_USD",
                scan_id=427,
                result="WIN",
            )
        except Exception:
            pass
