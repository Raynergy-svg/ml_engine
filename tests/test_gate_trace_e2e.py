"""US-512 — E2E: launch a minimal TUI app, push GateTraceModal, verify rows."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Static

from src.tui.screens.gate_trace_modal import (
    GateTraceModal,
    _AgentsSection,
    _GatesSection,
)


def _trace_with_12_agents() -> dict:
    """Fully-populated trace dict mirroring a real closed trade."""
    return {
        "context": {
            "trade_id": "E2E-1",
            "pair": "GBP_USD",
            "direction": "SELL",
            "timestamp": "2026-04-15T14:30:00Z",
            "broker": "oanda",
            "asset_class": "FX",
            "volatility_regime": "HIGH",
            "atr_pips": 22.0,
            "regime_risk_modifier": 0.85,
        },
        "agents": [
            {
                "name": name,
                "score": 0.6,
                "passed": (i % 2 == 0),
                "weight": 1.0,
                "reason": f"r-{i}",
                "reason_code": f"c-{i}",
                "block_trade": False,
            }
            for i, name in enumerate([
                "trend", "mean_reversion", "volatility", "risk_sentinel",
                "uncertainty", "execution_quality", "momentum", "news_risk",
                "multi_timeframe", "pair_performance", "session_timing",
                "support_resistance",
            ])
        ],
        "gates": {
            "confidence": {"passed": True, "label": "confidence"},
            "momentum": {"passed": True, "label": "momentum"},
            "risk": {"passed": True, "label": "risk"},
        },
        "math": {
            "weighted_vote_score": 0.71,
            "agent_total": 12,
            "agent_passed": 6,
            "confidence": 0.72,
            "uncertainty_score": 0.28,
            "model_disagreement": 0.15,
            "gate_summary": "all_pass",
        },
        "rr": {
            "sl_pips": 18.0, "tp_pips": 27.0, "rr_ratio": 1.5,
            "entry_price": 1.25000, "sl_price": 1.25180, "tp_price": 1.24730,
            "lots": 0.10,
        },
        "outcome": None,
    }


class _ModalHostApp(App):
    """Tiny harness app — pushes the modal on mount."""

    def __init__(self, trace: dict | None = None, trade_id: str | None = None,
                 journal_path: str | None = None) -> None:
        super().__init__()
        self._trace = trace
        self._trade_id = trade_id
        self._journal_path = journal_path

    def compose(self) -> ComposeResult:  # pragma: no cover - trivial
        yield Static("host", id="host")

    async def on_mount(self) -> None:
        await self.push_screen(
            GateTraceModal(
                trade_id=self._trade_id,
                trace=self._trace,
                journal_path=self._journal_path,
            )
        )


def test_gate_trace_modal_renders_12_agents_and_3_gates() -> None:
    """Press Enter equivalent: push the modal directly and assert content."""
    trace = _trace_with_12_agents()

    async def _run() -> None:
        app = _ModalHostApp(trace=trace)
        async with app.run_test() as pilot:
            await pilot.pause()

            assert isinstance(app.screen, GateTraceModal)

            # 12 agent rows
            agents_widget = app.screen.query_one(_AgentsSection)
            text = agents_widget.render().plain
            for name in [
                "trend", "mean_reversion", "volatility", "risk_sentinel",
                "uncertainty", "execution_quality", "momentum", "news_risk",
                "multi_timeframe", "pair_performance", "session_timing",
                "support_resistance",
            ]:
                assert name in text, f"agent {name!r} missing from modal"

            # 3 gate rows
            gates_widget = app.screen.query_one(_GatesSection)
            gtext = gates_widget.render().plain
            for gate in ("CONFIDENCE", "MOMENTUM", "RISK"):
                assert gate in gtext, f"gate {gate} missing"
            assert gtext.count("[PASS]") == 3

            # Escape dismisses the modal
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, GateTraceModal)

    asyncio.run(_run())


def test_gate_trace_modal_fallback_for_missing_data(tmp_path: Path) -> None:
    """Trade ID with no journal match shows the predates-phase-91 fallback."""
    journal = tmp_path / "j.json"
    journal.write_text(json.dumps([]))

    async def _run() -> None:
        app = _ModalHostApp(trade_id="ghost", journal_path=str(journal))
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, GateTraceModal)
            fallback = app.screen.query_one("#trace-fallback", Static)
            rendered = str(fallback.render())
            assert "predates phase 91" in rendered

    asyncio.run(_run())
