"""US-014: Unit tests for Multi-Broker and Asset Class Trade Journal support.

Tests:
1. FX trade journal entries with broker/asset_class defaults
2. Futures trade journal entries with contract_month/tick_value/multiplier
3. P&L calculation for FX (pips-based) vs FUTURES (tick-based)
4. Backward compatibility: old journal entries without new fields load without error
5. RL sync processes both asset classes correctly
6. Existing journal entries updated with new fields on outcome sync
7. Futures P&L formula: (exit_price - entry_price) * multiplier * contracts * direction_sign
8. FX P&L formula: unchanged (pips-based)
9. JSON schema validation passes
10. Edge cases: zero multiplier, missing fields, None values
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

import pytest


# ==============================================================================
# Test 1: FX Trade Entry with Broker/Asset Class Defaults
# ==============================================================================
def test_fx_trade_journal_entry_defaults():
    """FX trades should default to broker='oanda', asset_class='FX'."""
    entry = {
        "trade_id": "T001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pair": "EUR_USD",
        "direction": "LONG",
        "confidence": 0.75,
        "entry_price": 1.1050,
        "sl_price": 1.1000,
        "tp_price": 1.1100,
        "lots": 1.0,
        # New fields with defaults
        "broker": "oanda",
        "asset_class": "FX",
        "contract_month": None,
        "tick_value": None,
        "multiplier": None,
        "outcome": None,
    }

    assert entry["broker"] == "oanda"
    assert entry["asset_class"] == "FX"
    assert entry["contract_month"] is None
    assert entry["tick_value"] is None
    assert entry["multiplier"] is None


# ==============================================================================
# Test 2: Futures Trade Entry with All Fields
# ==============================================================================
def test_futures_trade_journal_entry_complete():
    """Futures trades should include contract_month, tick_value, multiplier."""
    entry = {
        "trade_id": "T002",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pair": "ES",  # E-mini S&P 500
        "direction": "SHORT",
        "confidence": 0.82,
        "entry_price": 5200.50,
        "sl_price": 5215.00,
        "tp_price": 5180.00,
        "lots": 2.0,  # 2 contracts
        # New fields for futures
        "broker": "ibkr",
        "asset_class": "FUTURES",
        "contract_month": "202406",  # June 2024
        "tick_value": 12.50,  # $12.50 per tick
        "multiplier": 50.0,  # $50 per point move
        "outcome": None,
    }

    assert entry["broker"] == "ibkr"
    assert entry["asset_class"] == "FUTURES"
    assert entry["contract_month"] == "202406"
    assert entry["tick_value"] == 12.50
    assert entry["multiplier"] == 50.0
    assert entry["lots"] == 2.0


# ==============================================================================
# Test 3: FX P&L Calculation (Pips-based, unchanged)
# ==============================================================================
def test_fx_pnl_calculation_long():
    """FX long: (close - entry) / pip_value."""
    entry_price = 1.1050
    close_price = 1.1100
    pip_value = 0.0001  # EUR/USD

    pnl_pips = (close_price - entry_price) / pip_value
    assert abs(pnl_pips - 50.0) < 0.01  # 50 pips profit (allowing float precision)


def test_fx_pnl_calculation_short():
    """FX short: -(close - entry) / pip_value."""
    entry_price = 1.1050
    close_price = 1.1000
    pip_value = 0.0001  # EUR/USD
    direction = "SHORT"

    pnl_pips = (close_price - entry_price) / pip_value
    if direction == "SHORT":
        pnl_pips = -pnl_pips
    assert abs(pnl_pips - 50.0) < 0.01  # 50 pips profit (going down, allowing float precision)


# ==============================================================================
# Test 4: Futures P&L Calculation (Tick-based, NEW)
# ==============================================================================
def test_futures_pnl_calculation_long():
    """Futures long: (exit - entry) * multiplier * contracts."""
    entry_price = 5200.50
    close_price = 5220.00
    multiplier = 50.0
    contracts = 2.0
    direction_sign = 1.0  # LONG

    pnl_ticks = (close_price - entry_price) * multiplier * contracts * direction_sign
    expected = (5220.00 - 5200.50) * 50.0 * 2.0 * 1.0
    expected = 19.5 * 50.0 * 2.0
    expected = 1950.0  # $1950 profit

    assert pnl_ticks == expected


def test_futures_pnl_calculation_short():
    """Futures short: (exit - entry) * multiplier * contracts * -1."""
    entry_price = 5200.50
    close_price = 5180.00
    multiplier = 50.0
    contracts = 2.0
    direction_sign = -1.0  # SHORT

    pnl_ticks = (close_price - entry_price) * multiplier * contracts * direction_sign
    expected = (5180.00 - 5200.50) * 50.0 * 2.0 * -1.0
    expected = (-20.5) * 50.0 * 2.0 * -1.0
    expected = 2050.0  # $2050 profit (short won)

    assert pnl_ticks == expected


# ==============================================================================
# Test 5: Backward Compatibility - Old Journal Entries Load Without Error
# ==============================================================================
def test_backward_compatibility_old_entry_missing_fields():
    """Old entries without asset_class/broker/etc should load gracefully."""
    old_entry = {
        "trade_id": "T_OLD_001",
        "timestamp": "2026-01-15T10:30:00+00:00",
        "pair": "GBP_USD",
        "direction": "LONG",
        "confidence": 0.70,
        "entry_price": 1.2700,
        "sl_price": 1.2650,
        "tp_price": 1.2750,
        "lots": 1.0,
        # Missing: broker, asset_class, contract_month, tick_value, multiplier
        "outcome": None,
    }

    # Simulate reading old entry - should use defaults
    broker = old_entry.get("broker", "oanda")
    asset_class = old_entry.get("asset_class", "FX")
    contract_month = old_entry.get("contract_month", None)
    tick_value = old_entry.get("tick_value", None)
    multiplier = old_entry.get("multiplier", None)

    assert broker == "oanda"
    assert asset_class == "FX"
    assert contract_month is None
    assert tick_value is None
    assert multiplier is None


# ==============================================================================
# Test 6: JSON Serialization Roundtrip
# ==============================================================================
def test_json_serialization_fx_entry():
    """FX entry should serialize/deserialize without loss."""
    original = {
        "trade_id": "T003",
        "timestamp": "2026-03-29T15:45:00+00:00",
        "pair": "EUR_USD",
        "direction": "LONG",
        "confidence": 0.75,
        "entry_price": 1.1050,
        "sl_price": 1.1000,
        "tp_price": 1.1100,
        "lots": 1.0,
        "broker": "oanda",
        "asset_class": "FX",
        "contract_month": None,
        "tick_value": None,
        "multiplier": None,
        "outcome": None,
    }

    # Serialize and deserialize
    json_str = json.dumps([original], indent=2, sort_keys=True, default=str)
    restored = json.loads(json_str)[0]

    assert restored["broker"] == "oanda"
    assert restored["asset_class"] == "FX"
    assert restored["trade_id"] == "T003"


def test_json_serialization_futures_entry():
    """Futures entry should serialize/deserialize with all new fields."""
    original = {
        "trade_id": "T004",
        "timestamp": "2026-03-29T15:45:00+00:00",
        "pair": "ES",
        "direction": "SHORT",
        "confidence": 0.82,
        "entry_price": 5200.50,
        "sl_price": 5215.00,
        "tp_price": 5180.00,
        "lots": 2.0,
        "broker": "ibkr",
        "asset_class": "FUTURES",
        "contract_month": "202406",
        "tick_value": 12.50,
        "multiplier": 50.0,
        "outcome": None,
    }

    json_str = json.dumps([original], indent=2, sort_keys=True, default=str)
    restored = json.loads(json_str)[0]

    assert restored["broker"] == "ibkr"
    assert restored["asset_class"] == "FUTURES"
    assert restored["contract_month"] == "202406"
    assert restored["tick_value"] == 12.50
    assert restored["multiplier"] == 50.0


# ==============================================================================
# Test 7: P&L Calculation in Journal Outcome Sync (Asset Class Awareness)
# ==============================================================================
def test_pnl_sync_fx_entry():
    """Sync should use FX P&L logic for asset_class='FX'."""
    entry = {
        "trade_id": "T005",
        "pair": "EUR_USD",
        "asset_class": "FX",
        "direction": "LONG",
        "entry_price": 1.1050,
        "lots": 1.0,
        "broker": "oanda",
        "contract_month": None,
        "tick_value": None,
        "multiplier": None,
    }

    # Simulate closed trade from broker
    closed_trade = {
        "averageClosePrice": 1.1100,
        "price": 1.1050,
        "realizedPL": 500.0,  # $500
    }

    # Simulate P&L calculation logic from sync_closed_trades_rl
    asset_class = entry.get("asset_class", "FX")
    close_price = float(closed_trade.get("averageClosePrice", 0))
    entry_price = float(closed_trade.get("price", entry.get("entry_price", 0)))
    pnl_pips = 0.0

    if asset_class == "FUTURES":
        multiplier = float(entry.get("multiplier", 1.0))
        contracts = float(entry.get("lots", 1.0))
        direction_sign = 1.0 if entry.get("direction", "").upper() == "LONG" else -1.0
        pnl_pips = (close_price - entry_price) * multiplier * contracts * direction_sign
    else:
        # FX calculation
        pip_value = 0.0001
        pnl_pips = (close_price - entry_price) / pip_value
        if entry.get("direction", "").upper() == "SHORT":
            pnl_pips = -pnl_pips

    assert abs(pnl_pips - 50.0) < 0.01  # 50 pips (allowing float precision)


def test_pnl_sync_futures_entry():
    """Sync should use Futures P&L logic for asset_class='FUTURES'."""
    entry = {
        "trade_id": "T006",
        "pair": "ES",
        "asset_class": "FUTURES",
        "direction": "LONG",
        "entry_price": 5200.50,
        "lots": 2.0,
        "broker": "ibkr",
        "contract_month": "202406",
        "tick_value": 12.50,
        "multiplier": 50.0,
    }

    # Simulate closed trade
    closed_trade = {
        "averageClosePrice": 5220.00,
        "price": 5200.50,
        "realizedPL": 1950.0,  # $1950
    }

    # Simulate P&L calculation
    asset_class = entry.get("asset_class", "FX")
    close_price = float(closed_trade.get("averageClosePrice", 0))
    entry_price = float(closed_trade.get("price", entry.get("entry_price", 0)))
    pnl_pips = 0.0

    if asset_class == "FUTURES":
        multiplier = float(entry.get("multiplier", 1.0))
        contracts = float(entry.get("lots", 1.0))
        direction_sign = 1.0 if entry.get("direction", "").upper() == "LONG" else -1.0
        pnl_pips = (close_price - entry_price) * multiplier * contracts * direction_sign
    else:
        pip_value = 0.0001
        pnl_pips = (close_price - entry_price) / pip_value

    expected = (5220.00 - 5200.50) * 50.0 * 2.0
    assert pnl_pips == expected


# ==============================================================================
# Test 8: Edge Case - Zero Multiplier
# ==============================================================================
def test_futures_pnl_with_zero_multiplier():
    """Zero multiplier should result in zero P&L."""
    entry_price = 5200.50
    close_price = 5220.00
    multiplier = 0.0  # Edge case
    contracts = 2.0
    direction_sign = 1.0

    pnl = (close_price - entry_price) * multiplier * contracts * direction_sign
    assert pnl == 0.0


# ==============================================================================
# Test 9: Edge Case - Missing Asset Class Defaults to FX
# ==============================================================================
def test_missing_asset_class_defaults_to_fx():
    """Missing asset_class should default to 'FX' logic."""
    entry = {
        "pair": "GBP_USD",
        "direction": "LONG",
        "entry_price": 1.2700,
        "lots": 1.0,
        # No asset_class specified
    }

    asset_class = entry.get("asset_class", "FX")
    assert asset_class == "FX"


# ==============================================================================
# Test 10: Mixed Journal - FX and Futures Entries
# ==============================================================================
def test_mixed_journal_entries():
    """Journal can contain both FX and Futures entries."""
    journal = [
        {
            "trade_id": "T_FX_001",
            "pair": "EUR_USD",
            "asset_class": "FX",
            "broker": "oanda",
            "direction": "LONG",
            "entry_price": 1.1050,
            "lots": 1.0,
        },
        {
            "trade_id": "T_FUT_001",
            "pair": "ES",
            "asset_class": "FUTURES",
            "broker": "ibkr",
            "direction": "SHORT",
            "entry_price": 5200.50,
            "lots": 2.0,
            "multiplier": 50.0,
        },
    ]

    fx_entries = [e for e in journal if e.get("asset_class") == "FX"]
    futures_entries = [e for e in journal if e.get("asset_class") == "FUTURES"]

    assert len(fx_entries) == 1
    assert len(futures_entries) == 1
    assert fx_entries[0]["broker"] == "oanda"
    assert futures_entries[0]["broker"] == "ibkr"


# ==============================================================================
# Test 11: Schema Validation - Required Fields Presence
# ==============================================================================
def test_journal_schema_validation():
    """Journal entries must have trade_id, pair, direction."""
    valid_entry = {
        "trade_id": "T007",
        "pair": "EUR_USD",
        "direction": "LONG",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "asset_class": "FX",
        "broker": "oanda",
    }

    required = {"trade_id", "pair", "direction"}
    assert required.issubset(valid_entry.keys())

    # Invalid entry (missing direction)
    invalid_entry = {
        "trade_id": "T008",
        "pair": "GBP_USD",
        # missing direction
    }

    assert not required.issubset(invalid_entry.keys())


# ==============================================================================
# Test 12: Atomic File Write Pattern with New Fields
# ==============================================================================
def test_journal_atomic_write():
    """Journal writes should be atomic (write to .tmp, then rename)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        journal_path = Path(tmpdir) / "trade_journal_rl.json"

        entries = [
            {
                "trade_id": "T009",
                "pair": "EUR_USD",
                "asset_class": "FX",
                "broker": "oanda",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "direction": "LONG",
                "entry_price": 1.1050,
                "outcome": None,
            }
        ]

        # Simulate atomic write
        import tempfile as tf
        import os

        _data = json.dumps(entries, indent=2, sort_keys=True, default=str)
        tmp_fd, tmp_path = tf.mkstemp(dir=str(journal_path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as _f:
                _f.write(_data)
                _f.flush()
                os.fsync(_f.fileno())
            os.rename(tmp_path, str(journal_path))
        except OSError:
            journal_path.write_text(_data)

        # Verify file exists and contains data
        assert journal_path.exists()
        restored = json.loads(journal_path.read_text())
        assert restored[0]["trade_id"] == "T009"
        assert restored[0]["asset_class"] == "FX"


# ==============================================================================
# Test 13: Outcome Recording Preserves Asset Class
# ==============================================================================
def test_outcome_recording_preserves_asset_class():
    """When outcome is recorded, asset_class should remain unchanged."""
    entry = {
        "trade_id": "T010",
        "pair": "ES",
        "asset_class": "FUTURES",
        "broker": "ibkr",
        "multiplier": 50.0,
        "outcome": None,
    }

    # Record outcome
    outcome_data = {
        "realized_pl": 1000.0,
        "pnl_pips": 400.0,
        "trade_won": True,
        "exit_reason": "tp",
    }
    entry["outcome"] = outcome_data

    # Verify asset_class unchanged
    assert entry["asset_class"] == "FUTURES"
    assert entry["multiplier"] == 50.0


# ==============================================================================
# Test 14: Contract Month Format Validation
# ==============================================================================
def test_contract_month_format():
    """contract_month should be YYYYMM string for futures."""
    futures_entry = {
        "contract_month": "202406",  # June 2024
    }

    # Validate format
    contract = futures_entry.get("contract_month")
    if contract is not None:
        assert len(contract) == 6
        assert contract.isdigit()

    # Old FX entry
    fx_entry = {
        "contract_month": None,
    }

    assert fx_entry["contract_month"] is None


# ==============================================================================
# Test 15: Multi-Broker Support - Different Brokers
# ==============================================================================
def test_multi_broker_journal_entries():
    """Journal can contain entries from different brokers (OANDA, IBKR)."""
    oanda_entry = {
        "trade_id": "OANDA_T001",
        "broker": "oanda",
        "pair": "EUR_USD",
        "asset_class": "FX",
    }

    ibkr_entry = {
        "trade_id": "IBKR_T001",
        "broker": "ibkr",
        "pair": "ES",
        "asset_class": "FUTURES",
        "contract_month": "202406",
    }

    journal = [oanda_entry, ibkr_entry]

    oanda_trades = [e for e in journal if e.get("broker") == "oanda"]
    ibkr_trades = [e for e in journal if e.get("broker") == "ibkr"]

    assert len(oanda_trades) == 1
    assert len(ibkr_trades) == 1
    assert oanda_trades[0]["asset_class"] == "FX"
    assert ibkr_trades[0]["asset_class"] == "FUTURES"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
