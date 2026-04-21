"""Integration test for ExecutionManager.flatten_all() — US-503.

Requires a live OANDA practice account with valid credentials in env:
  OANDA_API_TOKEN, OANDA_ACCOUNT_ID

This test:
1. Opens 3 small market trades on EUR_USD.
2. Calls flatten_all() and verifies all 3 close in < 2 seconds.
3. Asserts state.halted=True in .claude/state.json.
4. Asserts .claude/brain/strategic_log.md contains the kill entry.

Skip automatically when OANDA_API_TOKEN is not set.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.integration


def _has_oanda_creds() -> bool:
    return bool(os.getenv("OANDA_API_TOKEN")) and bool(os.getenv("OANDA_ACCOUNT_ID"))


@pytest.fixture(autouse=True)
def require_oanda_creds():
    if not _has_oanda_creds():
        pytest.skip("OANDA_API_TOKEN / OANDA_ACCOUNT_ID not set — skipping integration test")


def _oanda_headers() -> dict:
    return {
        "Authorization": f"Bearer {os.getenv('OANDA_API_TOKEN')}",
        "Content-Type": "application/json",
    }


def _base() -> str:
    return os.getenv("OANDA_API_URL", "https://api-fxpractice.oanda.com")


def _acct() -> str:
    return os.getenv("OANDA_ACCOUNT_ID", "")


def _open_trade(units: int = 100) -> str:
    """Open a tiny EUR_USD market trade and return trade ID."""
    resp = requests.post(
        f"{_base()}/v3/accounts/{_acct()}/orders",
        headers=_oanda_headers(),
        json={"order": {"type": "MARKET", "instrument": "EUR_USD", "units": str(units)}},
        timeout=10,
    )
    assert resp.status_code == 201, f"Failed to open trade: {resp.text}"
    trade_id = resp.json()["orderFillTransaction"]["tradeOpened"]["tradeID"]
    return str(trade_id)


def _get_open_trade_ids() -> list[str]:
    resp = requests.get(
        f"{_base()}/v3/accounts/{_acct()}/openTrades",
        headers=_oanda_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    return [str(t["id"]) for t in resp.json().get("trades", [])]


class TestFlattenAllIntegration:
    def test_flatten_closes_all_trades(self, tmp_path):
        """Open 3 trades, flatten_all, confirm all closed < 2 s, halted=True, log written."""
        from src.scanner.execution import ExecutionManager, KillSwitchPartialFailure
        from src.scanner.automation.state_engine import StateEngine

        # Open 3 tiny trades
        trade_ids = []
        for _ in range(3):
            tid = _open_trade(units=100)
            trade_ids.append(tid)

        # Confirm trades are open
        open_before = _get_open_trade_ids()
        for tid in trade_ids:
            assert tid in open_before, f"Trade {tid} not found in openTrades before flatten"

        mgr = ExecutionManager()

        # Measure wall-clock time
        t0 = time.monotonic()
        try:
            result = asyncio.run(mgr.flatten_all(reason="integration-test"))
        except KillSwitchPartialFailure as exc:
            elapsed = (time.monotonic() - t0) * 1000
            pytest.fail(
                f"flatten_all raised KillSwitchPartialFailure after {elapsed:.0f}ms: "
                f"remaining={exc.remaining_trades}"
            )
        elapsed_ms = (time.monotonic() - t0) * 1000

        # ── AC: all 3 trades closed ───────────────────────────────────
        open_after = _get_open_trade_ids()
        for tid in trade_ids:
            assert tid not in open_after, f"Trade {tid} still open after flatten_all"

        # ── AC: completed in < 2 000 ms ──────────────────────────────
        assert elapsed_ms < 2000, f"flatten_all took {elapsed_ms:.0f}ms (> 2000ms limit)"

        # ── AC: FlattenResult fields ──────────────────────────────────
        assert result.positions_closed >= 3
        assert result.positions_failed == 0
        assert result.duration_ms > 0

        # ── AC: state.halted = True ───────────────────────────────────
        se = StateEngine()
        assert se.get_halted() is True, "StateEngine.halted should be True after flatten_all"

        # Clean up — resume so other tests aren't blocked
        se.set_halted(False)

        # ── AC: strategic_log.md has entry ───────────────────────────
        log_path = Path(".claude/brain/strategic_log.md")
        assert log_path.exists(), "strategic_log.md does not exist"
        content = log_path.read_text()
        assert "TUI_KILL" in content, "strategic_log.md missing TUI_KILL actor entry"
        assert "integration-test" in content, "strategic_log.md missing reason field"
