"""
GATE TRACE MODAL — US-512

Read-only modal that answers "why did this fire?" or "why was this rejected?"
for any signal in the journal. Displays five sections:
  1. CONTEXT  — pair / direction / timestamp / regime / ATR / broker
  2. GATES    — confidence | momentum | risk (PASS/FAIL color-coded)
  3. AGENTS   — 15 agent votes with score bars + reason codes (see ``ScannerAgentTeam._BASE_WEIGHTS``)
  4. MATH     — weighted_vote_score, agent_passed/total, uncertainty,
                model disagreement, confidence
  5. R:R      — sl_pips, tp_pips, rr_ratio, entry/SL/TP

Pushed by Trades screen (Enter on row), Journal screen (Enter on row),
or any caller that has a trade_id.
"""
from __future__ import annotations

from typing import Any

from rich.text import Text

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

# Theme palette (mirrors theme.tcss)
_PRIMARY = "#00ffcc"
_SECONDARY = "#ff00ff"
_POSITIVE = "#00ff41"
_NEGATIVE = "#ff1744"
_WARNING = "#ffab00"
_TEXT = "#e0e0ff"
_DIM = "#6666aa"
_DATA = "#7c4dff"
_BORDER = "#2a2a4a"


# ---------------------------------------------------------------------------
# Renderable section widgets
# ---------------------------------------------------------------------------


class _ContextSection(Static):
    def __init__(self, ctx: dict[str, Any], **kw: Any) -> None:
        super().__init__(**kw)
        self._ctx = ctx

    def render(self) -> Text:
        c = self._ctx
        t = Text()
        t.append("⟨ CONTEXT ⟩\n", style=f"bold {_PRIMARY}")
        t.append("  Trade ID: ", style=_DIM)
        t.append(f"{c.get('trade_id', '---')}", style=f"bold {_TEXT}")
        t.append("    Pair: ", style=_DIM)
        t.append(f"{c.get('pair', '---')}", style=f"bold {_TEXT}")
        direction = c.get("direction", "---")
        dir_color = _POSITIVE if direction == "BUY" else _NEGATIVE if direction == "SELL" else _DIM
        t.append("    Dir: ", style=_DIM)
        t.append(f"{direction}\n", style=f"bold {dir_color}")
        t.append("  Time: ", style=_DIM)
        ts = c.get("timestamp", "")[:19].replace("T", " ")
        t.append(f"{ts or '---'}", style=_TEXT)
        t.append("    Broker: ", style=_DIM)
        t.append(f"{c.get('broker', '---')}", style=_TEXT)
        t.append("    Asset: ", style=_DIM)
        t.append(f"{c.get('asset_class', 'FX')}\n", style=_TEXT)
        t.append("  Regime: ", style=_DIM)
        t.append(f"{c.get('volatility_regime', 'UNKNOWN')}", style=f"bold {_DATA}")
        t.append("    ATR: ", style=_DIM)
        t.append(f"{c.get('atr_pips', 0.0):.1f} pips", style=_TEXT)
        t.append("    Risk Mod: ", style=_DIM)
        t.append(f"{c.get('regime_risk_modifier', 1.0):.2f}x", style=_TEXT)
        return t


class _GatesSection(Static):
    def __init__(self, gates: dict[str, dict[str, Any]], summary: str, **kw: Any) -> None:
        super().__init__(**kw)
        self._gates = gates
        self._summary = summary

    def render(self) -> Text:
        t = Text()
        t.append("⟨ GATES ⟩  3-of-3 must pass to execute\n", style=f"bold {_PRIMARY}")
        for key in ("confidence", "momentum", "risk"):
            g = self._gates.get(key, {})
            passed = bool(g.get("passed", False))
            label = "PASS" if passed else "FAIL"
            color = _POSITIVE if passed else _NEGATIVE
            t.append(f"  {key.upper():<11}", style=_DIM)
            t.append(f"  [{label}]\n", style=f"bold {color}")
        if self._summary:
            t.append("  Summary: ", style=_DIM)
            t.append(f"{self._summary}\n", style=_TEXT)
        return t


class _AgentsSection(Static):
    def __init__(self, agents: list[dict[str, Any]], **kw: Any) -> None:
        super().__init__(**kw)
        self._agents = agents

    def render(self) -> Text:
        t = Text()
        n = len(self._agents)
        t.append(f"⟨ AGENTS ⟩  {n} votes\n", style=f"bold {_PRIMARY}")
        if not self._agents:
            t.append("  (no agent votes recorded)\n", style=_DIM)
            return t
        for a in self._agents:
            name = (a.get("name") or "???")[:18]
            score = max(0.0, min(1.0, float(a.get("score", 0.0))))
            passed = bool(a.get("passed", False))
            weight = float(a.get("weight", 1.0))
            reason = a.get("reason") or a.get("reason_code") or ""

            # Color by score with NEAR-MISS amber band 0.45-0.55
            if passed:
                score_color = _POSITIVE
            elif 0.45 <= score <= 0.55:
                score_color = _WARNING  # NEAR-MISS
            else:
                score_color = _NEGATIVE

            label = "PASS" if passed else ("NEAR" if 0.45 <= score <= 0.55 else "FAIL")

            filled = int(score * 10)
            empty = 10 - filled

            t.append(f"  {name:<18}", style=_TEXT)
            t.append(f" {score:.2f} ", style=f"bold {score_color}")
            t.append("█" * filled, style=score_color)
            t.append("░" * empty, style=_BORDER)
            t.append(f"  [{label}]", style=f"bold {score_color}")
            t.append(f"  w={weight:.2f}", style=_DIM)
            if reason:
                t.append(f"  — {reason}", style=_DIM)
            t.append("\n")
        return t


class _MathSection(Static):
    def __init__(self, math: dict[str, Any], **kw: Any) -> None:
        super().__init__(**kw)
        self._math = math

    def render(self) -> Text:
        m = self._math
        t = Text()
        t.append("⟨ MATH ⟩  weighted vote + uncertainty\n", style=f"bold {_PRIMARY}")
        wvs = float(m.get("weighted_vote_score", 0.0))
        wvs_color = _POSITIVE if wvs >= 0.65 else _WARNING if wvs >= 0.50 else _NEGATIVE
        t.append("  Weighted Vote Score: ", style=_DIM)
        t.append(f"{wvs:.3f}\n", style=f"bold {wvs_color}")
        t.append("  Agents passed:       ", style=_DIM)
        t.append(f"{int(m.get('agent_passed', 0))} / {int(m.get('agent_total', 0))}\n",
                 style=f"bold {_TEXT}")
        conf = float(m.get("confidence", 0.0))
        conf_color = _POSITIVE if conf >= 0.70 else _WARNING if conf >= 0.55 else _NEGATIVE
        t.append("  Confidence:          ", style=_DIM)
        t.append(f"{conf:.3f}\n", style=f"bold {conf_color}")
        unc = float(m.get("uncertainty_score", 0.0))
        unc_color = _NEGATIVE if unc >= 0.45 else _WARNING if unc >= 0.35 else _POSITIVE
        t.append("  Uncertainty:         ", style=_DIM)
        t.append(f"{unc:.3f}\n", style=f"bold {unc_color}")
        dis = float(m.get("model_disagreement", 0.0))
        dis_color = _NEGATIVE if dis >= 0.30 else _WARNING if dis >= 0.20 else _POSITIVE
        t.append("  Model Disagreement:  ", style=_DIM)
        t.append(f"{dis:.3f}\n", style=f"bold {dis_color}")
        return t


class _RRSection(Static):
    def __init__(self, rr: dict[str, Any], **kw: Any) -> None:
        super().__init__(**kw)
        self._rr = rr

    def render(self) -> Text:
        r = self._rr
        t = Text()
        t.append("⟨ R:R ⟩  risk/reward computation\n", style=f"bold {_PRIMARY}")
        rr_val = float(r.get("rr_ratio", 0.0))
        rr_color = _POSITIVE if rr_val >= 1.5 else _WARNING if rr_val >= 1.2 else _NEGATIVE
        t.append("  R:R ratio: ", style=_DIM)
        t.append(f"{rr_val:.2f}:1", style=f"bold {rr_color}")
        rr_label = "PASS" if rr_val >= 1.2 else "FAIL"
        t.append(f"  [{rr_label}]\n", style=f"bold {rr_color}")
        t.append("  SL: ", style=_DIM)
        t.append(f"{float(r.get('sl_pips', 0.0)):.1f} pips", style=f"bold {_NEGATIVE}")
        t.append("    TP: ", style=_DIM)
        t.append(f"{float(r.get('tp_pips', 0.0)):.1f} pips\n", style=f"bold {_POSITIVE}")
        t.append("  Entry: ", style=_DIM)
        t.append(f"{float(r.get('entry_price', 0.0)):.5f}", style=_TEXT)
        t.append("    SL: ", style=_DIM)
        t.append(f"{float(r.get('sl_price', 0.0)):.5f}", style=_NEGATIVE)
        t.append("    TP: ", style=_DIM)
        t.append(f"{float(r.get('tp_price', 0.0)):.5f}\n", style=_POSITIVE)
        t.append("  Lots: ", style=_DIM)
        t.append(f"{float(r.get('lots', 0.0)):.4f}\n", style=_TEXT)
        return t


# ---------------------------------------------------------------------------
# Modal
# ---------------------------------------------------------------------------


class GateTraceModal(ModalScreen[None]):
    """Read-only gate-trace modal pushed by Trades / Journal screens.

    Pass either:
      - trade_id (str) and let the modal load via JournalReader.get_gate_trace, or
      - trace (dict) directly (used by tests + already-loaded callers)
    """

    BINDINGS = [
        Binding("escape", "dismiss_modal", "Close", show=True),
        Binding("q", "dismiss_modal", "Close", show=False),
    ]

    DEFAULT_CSS = """
    GateTraceModal {
        align: center middle;
    }

    #trace-dialog {
        width: 84;
        max-width: 95%;
        height: 38;
        max-height: 90%;
        background: #0a0a0f;
        border: double #00ffcc;
        padding: 1 2;
    }

    #trace-title {
        text-align: center;
        color: #00ffcc;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #trace-scroll {
        height: 1fr;
        background: #0d0d18;
    }

    #trace-fallback {
        color: #ffab00;
        text-align: center;
        padding: 2;
    }

    #trace-buttons {
        align: center middle;
        height: 3;
        padding: 1 0 0 0;
    }

    #trace-close-btn {
        background: #2a2a4a;
        color: #00ffcc;
        min-width: 16;
    }

    #trace-close-btn:hover {
        background: #00ffcc;
        color: #0a0a0f;
    }

    .trace-section {
        padding: 1 1;
        margin: 0 0 1 0;
        background: #131320;
        border: solid #2a2a4a;
    }
    """

    def __init__(
        self,
        trade_id: str | None = None,
        trace: dict[str, Any] | None = None,
        journal_path: str | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self._trade_id = trade_id
        self._trace = trace
        self._journal_path = journal_path

    def _load_trace(self) -> dict[str, Any] | None:
        if self._trace is not None:
            return self._trace
        if not self._trade_id:
            return None
        # Lazy import — keeps this module importable in headless tests
        from src.tui.journal_reader import JournalReader

        reader = JournalReader(journal_path=self._journal_path)
        return reader.get_gate_trace(self._trade_id)

    def compose(self) -> ComposeResult:
        trace = self._load_trace()

        with Vertical(id="trace-dialog"):
            title = "⟨ GATE TRACE ⟩  why did this fire?"
            if trace:
                ctx = trace.get("context") or {}
                title = (
                    f"⟨ GATE TRACE ⟩  {ctx.get('pair', '---')} "
                    f"{ctx.get('direction', '---')}  #{ctx.get('trade_id', '---')}"
                )
            yield Label(title, id="trace-title")

            if not trace:
                yield Static(
                    "Trace unavailable — journal entry predates phase 91",
                    id="trace-fallback",
                )
            else:
                with VerticalScroll(id="trace-scroll"):
                    yield _ContextSection(trace.get("context") or {}, classes="trace-section")
                    yield _GatesSection(
                        trace.get("gates") or {},
                        (trace.get("math") or {}).get("gate_summary", ""),
                        classes="trace-section",
                    )
                    yield _AgentsSection(trace.get("agents") or [], classes="trace-section")
                    yield _MathSection(trace.get("math") or {}, classes="trace-section")
                    yield _RRSection(trace.get("rr") or {}, classes="trace-section")

            with Container(id="trace-buttons"):
                yield Button("Close (Esc)", id="trace-close-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "trace-close-btn":
            self.dismiss(None)

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)
