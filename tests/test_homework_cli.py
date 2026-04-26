"""Tests for the homework CLI subcommand: --generate-batch with selection flags."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def journal_with_closed_trades(tmp_path: Path) -> Path:
    """Synthetic journal with 3 closed trades."""
    journal = tmp_path / "trade_journal_rl.json"
    entries = []
    for i, (pair, dir_, reason, pl) in enumerate([
        ("EUR_AUD", "SHORT", "SL", -354.56),
        ("USD_CHF", "LONG", "SL", -663.08),
        ("EUR_USD", "LONG", "TP",  261.00),
    ]):
        entries.append({
            "trade_id": str(1000 + i),
            "pair": pair,
            "direction": dir_,
            "entry_price": 1.0, "sl_price": 0.99, "tp_price": 1.02,
            "sl_pips": 10.0, "tp_pips": 20.0, "rr_ratio": 2.0,
            "confidence": 0.65, "weighted_vote_score": 0.70,
            "regime": "NORMAL",
            "agents": [{"name": "trend", "passed": True, "score": 0.6, "weight": 1.15}],
            "gate_details": {"adx": 20.0, "rsi": 50.0, "atr_pips": 10.0,
                             "model_disagreement": 0.20,
                             "disagreement_hard_floor": 0.50},
            "outcome": {
                "close_time": "2026-04-15T02:46:03Z",
                "close_price": 0.99 if reason == "SL" else 1.02,
                "realized_pl": pl,
                "close_reason": reason,
                "duration_minutes": 32,
                "mfe_pips": 2.0, "mae_pips": 12.0,
            },
        })
    journal.write_text(json.dumps(entries))
    return journal


def _run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "buddy_scanner.py", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


class TestHomeworkCLI:
    def test_help_flag_lists_homework_subcommand(self) -> None:
        result = _run_cli("--help")
        assert "homework" in result.stdout.lower()

    def test_homework_help_lists_generate_batch(self) -> None:
        result = _run_cli("homework", "--help")
        assert "--generate-batch" in result.stdout
        assert "--last" in result.stdout

    def test_generate_batch_last_3_creates_pending_entries(
        self, journal_with_closed_trades: Path, tmp_path: Path
    ) -> None:
        """Set BUDDY_HOMEWORK_PENDING_PATH + BUDDY_TRADE_JOURNAL_PATH to tmp."""
        import os
        env = {
            **os.environ,
            "BUDDY_TRADE_JOURNAL_PATH": str(journal_with_closed_trades),
            "BUDDY_HOMEWORK_PENDING_PATH": str(tmp_path / "homework_pending.jsonl"),
            "BUDDY_HOMEWORK_HISTORY_PATH": str(tmp_path / "homework_history.jsonl"),
        }
        result = _run_cli("homework", "--generate-batch", "--last", "3", env=env)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Pending file exists with 3 entries
        pending = tmp_path / "homework_pending.jsonl"
        assert pending.exists()
        lines = [l for l in pending.read_text().splitlines() if l.strip()]
        assert len(lines) == 3
