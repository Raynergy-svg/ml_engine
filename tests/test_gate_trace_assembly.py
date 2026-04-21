"""US-512 — gate trace assembly: synthetic journal entry → 5-section dict."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.agents._team import ScannerAgentTeam
from src.tui.journal_reader import JournalReader

_BASE_WEIGHTS = ScannerAgentTeam._BASE_WEIGHTS


def _synthetic_entry() -> dict:
    """Build a journal entry that exercises every section the modal needs."""
    return {
        "trade_id": "TEST-512",
        "pair": "EUR_USD",
        "direction": "BUY",
        "timestamp": "2026-04-16T20:00:00Z",
        "broker": "oanda",
        "asset_class": "FX",
        "confidence": 0.71,
        "entry_price": 1.08500,
        "sl_price": 1.08300,
        "tp_price": 1.08800,
        "sl_pips": 20.0,
        "tp_pips": 30.0,
        "rr_ratio": 1.5,
        "lots": 0.25,
        "gates": {
            "momentum_passed": True,
            "confidence_passed": True,
            "risk_passed": False,
            "gate_summary": "risk_block:portfolio_exposure",
        },
        "agents": {
            "agent_total": 12,
            "agent_passed": 9,
            "weighted_vote_score": 0.68,
            "agent_reasons": [
                {
                    "name": name,
                    "score": 0.55 + (i * 0.03),
                    "passed": (i % 2 == 0),
                    "weight": 1.0,
                    "reason": f"reason-{i}",
                    "reason_code": f"code-{i}",
                    "block_trade": False,
                }
                for i, name in enumerate([
                    "trend", "mean_reversion", "volatility", "risk_sentinel",
                    "uncertainty", "execution_quality", "momentum", "news_risk",
                    "multi_timeframe", "pair_performance", "session_timing",
                    "support_resistance",
                ])
            ],
        },
        "regime": {
            "volatility_regime": "NORMAL",
            "atr_pips": 18.4,
            "uncertainty_score": 0.32,
            "model_disagreement": 0.18,
            "regime_risk_modifier": 1.0,
        },
        "outcome": {
            "trade_won": True,
            "realized_pl": 45.20,
            "exit_reason": "tp_hit",
        },
    }


@pytest.fixture
def journal_path(tmp_path: Path) -> Path:
    p = tmp_path / "journal.json"
    p.write_text(json.dumps([_synthetic_entry()]))
    return p


def test_get_gate_trace_assembles_all_five_sections(journal_path: Path) -> None:
    reader = JournalReader(journal_path=journal_path)
    trace = reader.get_gate_trace("TEST-512")

    assert trace is not None
    # 5 sections required by US-512
    for section in ("context", "agents", "gates", "math", "rr"):
        assert section in trace, f"missing section: {section}"
        assert trace[section], f"section empty: {section}"

    # Context populated
    assert trace["context"]["pair"] == "EUR_USD"
    assert trace["context"]["direction"] == "BUY"
    assert trace["context"]["volatility_regime"] == "NORMAL"
    assert trace["context"]["atr_pips"] == pytest.approx(18.4)

    # Agent votes carried through — anchor to _BASE_WEIGHTS (source of truth)
    # so growth from 12 -> 15 agents (+order_flow, +trader_readiness,
    # +devil_advocate) doesn't silently break this assertion.
    # Note: synthetic entry exercises only the core 12; real traces carry all
    # enabled agents, which is why we assert >= not ==.
    assert len(trace["agents"]) >= min(len(_BASE_WEIGHTS), 12)
    names = {a["name"] for a in trace["agents"]}
    assert "trend" in names and "support_resistance" in names

    # 3 gates with PASS/FAIL flags
    assert trace["gates"]["confidence"]["passed"] is True
    assert trace["gates"]["momentum"]["passed"] is True
    assert trace["gates"]["risk"]["passed"] is False

    # Math section
    assert trace["math"]["weighted_vote_score"] == pytest.approx(0.68)
    assert trace["math"]["agent_passed"] == 9
    assert trace["math"]["agent_total"] == 12
    assert trace["math"]["uncertainty_score"] == pytest.approx(0.32)
    assert trace["math"]["model_disagreement"] == pytest.approx(0.18)

    # R:R section
    assert trace["rr"]["rr_ratio"] == pytest.approx(1.5)
    assert trace["rr"]["sl_pips"] == pytest.approx(20.0)
    assert trace["rr"]["tp_pips"] == pytest.approx(30.0)


def test_gate_trace_missing_id_returns_none(journal_path: Path) -> None:
    reader = JournalReader(journal_path=journal_path)
    assert reader.get_gate_trace("does-not-exist") is None
    assert reader.get_gate_trace("") is None


def test_gate_trace_missing_journal_returns_none(tmp_path: Path) -> None:
    reader = JournalReader(journal_path=tmp_path / "no_such_file.json")
    assert reader.get_gate_trace("TEST-512") is None


def test_gate_trace_corrupt_json_returns_none(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    reader = JournalReader(journal_path=bad)
    assert reader.get_gate_trace("TEST-512") is None
