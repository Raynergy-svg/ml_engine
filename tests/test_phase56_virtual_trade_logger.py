"""Tests for Phase 56 US-348: Virtual Trade Logger.

Covers:
  - log_virtual_trade() writes to JSONL file
  - Entries include all required fields
  - get_recent(n) returns last n entries
  - Append behavior: each call adds a new line
  - Low confidence entries are skipped (< 0.10)
  - HOLD direction entries are skipped
  - get_stats() returns correct total, avg_confidence, top_pairs, top_failures
  - File created on first write
  - get_recent() returns empty list when no file
  - Counter rebuilt correctly from existing file on init
  - Wiring: VirtualTradeLogger initialized in engine.py
  - Wiring: log_virtual_trade called in engine.py
"""

import json
from pathlib import Path

import pytest

from src.scanner.virtual_trade_logger import VirtualTradeLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def logger(tmp_path):
    return VirtualTradeLogger(log_path=tmp_path / "virtual_trades.jsonl")


# ---------------------------------------------------------------------------
# 1. log_virtual_trade
# ---------------------------------------------------------------------------

class TestLogVirtualTrade:

    def test_returns_true_on_success(self, logger):
        result = logger.log_virtual_trade("EUR_USD", "LONG", 0.35, ["confidence < 0.60"])
        assert result is True

    def test_file_created_on_first_write(self, logger):
        assert not logger.log_path.exists()
        logger.log_virtual_trade("EUR_USD", "LONG", 0.35, [])
        assert logger.log_path.exists()

    def test_entry_contains_required_fields(self, logger):
        logger.log_virtual_trade(
            pair="EUR_USD",
            direction="LONG",
            confidence=0.42,
            gate_failures=["confidence=0.42 < 0.60"],
            features={"momentum": 0.5},
            agent_scores={"trend": 0.7},
        )
        lines = logger.log_path.read_text().strip().split("\n")
        entry = json.loads(lines[0])
        for key in ("timestamp", "pair", "direction", "raw_confidence", "gate_failures", "features", "agent_scores"):
            assert key in entry, f"Missing key: {key}"

    def test_pair_and_direction_stored_correctly(self, logger):
        logger.log_virtual_trade("GBP_USD", "SHORT", 0.30, [])
        entry = json.loads(logger.log_path.read_text().strip())
        assert entry["pair"] == "GBP_USD"
        assert entry["direction"] == "SHORT"

    def test_low_confidence_skipped(self, logger):
        result = logger.log_virtual_trade("EUR_USD", "LONG", 0.05, [])
        assert result is False
        assert not logger.log_path.exists()

    def test_hold_direction_skipped(self, logger):
        result = logger.log_virtual_trade("EUR_USD", "HOLD", 0.50, [])
        assert result is False

    def test_append_behavior_multiple_entries(self, logger):
        logger.log_virtual_trade("EUR_USD", "LONG", 0.35, [])
        logger.log_virtual_trade("GBP_USD", "SHORT", 0.28, [])
        logger.log_virtual_trade("USD_JPY", "LONG", 0.40, [])
        lines = [l for l in logger.log_path.read_text().split("\n") if l.strip()]
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# 2. get_recent
# ---------------------------------------------------------------------------

class TestGetRecent:

    def test_get_recent_returns_last_n(self, logger):
        for i in range(10):
            logger.log_virtual_trade(f"PAIR_{i}", "LONG", 0.30 + i * 0.01, [])
        recent = logger.get_recent(n=3)
        assert len(recent) == 3
        assert recent[-1]["pair"] == "PAIR_9"

    def test_get_recent_returns_all_when_less_than_n(self, logger):
        logger.log_virtual_trade("EUR_USD", "LONG", 0.35, [])
        logger.log_virtual_trade("GBP_USD", "SHORT", 0.28, [])
        recent = logger.get_recent(n=50)
        assert len(recent) == 2

    def test_get_recent_returns_empty_when_no_file(self, tmp_path):
        logger_fresh = VirtualTradeLogger(log_path=tmp_path / "nonexistent.jsonl")
        assert logger_fresh.get_recent() == []


# ---------------------------------------------------------------------------
# 3. get_stats
# ---------------------------------------------------------------------------

class TestGetStats:

    def test_stats_total_logged_correct(self, logger):
        logger.log_virtual_trade("EUR_USD", "LONG", 0.40, ["confidence"])
        logger.log_virtual_trade("GBP_USD", "SHORT", 0.30, ["momentum"])
        stats = logger.get_stats()
        assert stats["total_logged"] == 2

    def test_stats_avg_confidence_correct(self, logger):
        logger.log_virtual_trade("EUR_USD", "LONG", 0.40, [])
        logger.log_virtual_trade("GBP_USD", "SHORT", 0.20, [])
        stats = logger.get_stats()
        assert stats["avg_confidence"] == pytest.approx(0.30, abs=0.01)

    def test_stats_has_required_keys(self, logger):
        stats = logger.get_stats()
        for key in ("total_logged", "avg_confidence", "top_rejected_pairs", "top_gate_failures"):
            assert key in stats, f"Missing key: {key}"

    def test_top_rejected_pairs_ordered(self, logger):
        for _ in range(3):
            logger.log_virtual_trade("EUR_USD", "LONG", 0.35, [])
        logger.log_virtual_trade("GBP_USD", "SHORT", 0.30, [])
        stats = logger.get_stats()
        top_pair, top_count = stats["top_rejected_pairs"][0]
        assert top_pair == "EUR_USD"
        assert top_count == 3


# ---------------------------------------------------------------------------
# 4. Counter rebuild on init
# ---------------------------------------------------------------------------

class TestCounterRebuildOnInit:

    def test_counters_rebuilt_from_file(self, tmp_path):
        path = tmp_path / "vt.jsonl"
        l1 = VirtualTradeLogger(log_path=path)
        l1.log_virtual_trade("EUR_USD", "LONG", 0.35, ["confidence"])
        l1.log_virtual_trade("EUR_USD", "LONG", 0.28, ["confidence"])

        # New instance reads the file and rebuilds counters
        l2 = VirtualTradeLogger(log_path=path)
        assert l2._total_logged == 2
        assert l2._pair_counts["EUR_USD"] == 2
        assert l2._gate_failure_counts["confidence"] == 2


# ---------------------------------------------------------------------------
# 5. Wiring verification
# ---------------------------------------------------------------------------

class TestWiringVerification:

    def test_virtual_trade_logger_initialized_in_engine(self):
        with open("src/scanner/engine.py") as f:
            content = f.read()
        assert "VirtualTradeLogger" in content
        assert "_virtual_trade_logger = None" in content

    def test_log_virtual_trade_called_in_engine(self):
        with open("src/scanner/engine.py") as f:
            content = f.read()
        assert "_virtual_trade_logger.log_virtual_trade(" in content
