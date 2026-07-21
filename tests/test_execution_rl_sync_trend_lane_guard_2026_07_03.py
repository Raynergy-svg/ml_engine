"""sync_closed_trades_rl must skip trend-lane (rl_eligible=False) entries.

Root cause: trend_journal_sync._build_open_entry writes rl_eligible=False as
"the explicit signal that sync_closed_trades_rl ... must skip this trade"
(its own docstring), but sync_closed_trades_rl never read the flag. Legacy
trend entries on disk carry a bare regime=None (not a dict) — pending trend
entries used to reach `entry.get("regime", {}).get("volatility_regime")`,
which raises AttributeError on a None value (dict.get's default only applies
when the key is ABSENT, not when its value is None), aborting the whole
RL-sync batch mid-loop. 3 such entries are live in trained_data/
trade_journal_rl.json as of 2026-07-03.

NO MOCKS per CLAUDE.md No-Mock Rule. Real ExecutionManager, real disk via
tmp_path. Both cases below resolve before any network call is made, so no
OANDA stub is needed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.scanner.execution import ExecutionConfig, ExecutionManager


def _write_journal(tmp_path: Path, entries: list) -> None:
    (tmp_path / "trained_data").mkdir(exist_ok=True)
    (tmp_path / "trained_data" / "trade_journal_rl.json").write_text(json.dumps(entries))


def _run_in(tmp_path: Path, mgr: ExecutionManager):
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return mgr.sync_closed_trades_rl()
    finally:
        os.chdir(original_cwd)


_TREND_ENTRY = {
    "trade_id": "9001",
    "pair": "USD_JPY",
    "lane": "trend",
    "rl_eligible": False,
    "direction": "LONG",
    "timestamp": "2026-06-01T00:00:00Z",
    "entry_price": 145.0,
    "outcome": None,
    "agents": {},
    "regime": None,  # legacy on-disk shape — bare None, not a dict
}


def test_trend_lane_pending_entries_never_reach_the_none_regime_crash_site(
    tmp_path, monkeypatch,
):
    """All-trend-lane pending batch resolves to a clean no-op — the
    AttributeError-prone code path (past the pending filter) is never
    reached, so this holds regardless of OANDA credentials/network."""
    _write_journal(tmp_path, [_TREND_ENTRY, dict(_TREND_ENTRY, trade_id="9002")])
    mgr = ExecutionManager(config=ExecutionConfig())

    result = _run_in(tmp_path, mgr)

    assert result == {
        "trades_synced": 0,
        "weights_updated": False,
        "detail": "no pending trades",
    }


def test_scanner_lane_pending_entry_without_rl_eligible_key_still_included(
    tmp_path, monkeypatch,
):
    """Scanner-lane entries never set rl_eligible — the default (True) must
    keep them in the pending batch, i.e. no regression for the normal path.
    Credentials forced to empty strings (not deleted) so python-dotenv's
    override=False default can't backfill real values from .env.local."""
    for key in ("OANDA_API_TOKEN", "OANDA_API_KEY", "OANDA_ACCOUNT_ID"):
        monkeypatch.setenv(key, "")
    scanner_entry = {
        "trade_id": "9003",
        "pair": "EUR_USD",
        "direction": "LONG",
        "timestamp": "2026-06-01T00:00:00Z",
        "entry_price": 1.1,
        "outcome": None,
        "agents": {},
        "regime": {},
    }
    _write_journal(tmp_path, [scanner_entry])
    mgr = ExecutionManager(config=ExecutionConfig())

    result = _run_in(tmp_path, mgr)

    # Reached the credentials check (i.e. was NOT filtered out) — proves the
    # rl_eligible default doesn't regress scanner-lane sync.
    assert result == {
        "trades_synced": 0,
        "weights_updated": False,
        "detail": "missing oanda credentials",
    }


def test_mixed_batch_only_trend_lane_is_filtered(tmp_path, monkeypatch):
    for key in ("OANDA_API_TOKEN", "OANDA_API_KEY", "OANDA_ACCOUNT_ID"):
        monkeypatch.setenv(key, "")
    scanner_entry = {
        "trade_id": "9004",
        "pair": "GBP_USD",
        "direction": "SHORT",
        "timestamp": "2026-06-01T00:00:00Z",
        "entry_price": 1.25,
        "outcome": None,
        "agents": {},
        "regime": {},
    }
    _write_journal(tmp_path, [_TREND_ENTRY, scanner_entry])
    mgr = ExecutionManager(config=ExecutionConfig())

    result = _run_in(tmp_path, mgr)

    # The trend entry was dropped silently; the scanner entry alone drove the
    # batch to the credentials check (not a no-op skip of the whole batch).
    assert result == {
        "trades_synced": 0,
        "weights_updated": False,
        "detail": "missing oanda credentials",
    }
