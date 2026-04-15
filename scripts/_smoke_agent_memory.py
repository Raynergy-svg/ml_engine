"""Smoke test: agent memory write → outcome backfill → query round-trip.

Verifies the three paths:
  1. cache_verdicts_for_trade_open + TRADE_OPENED -> rows land in Chroma
  2. TRADE_CLOSED -> outcome backfill updates metadata
  3. query_and_score on a similar context returns hits with correct
     confidence_delta sign (negative avg pnl -> negative delta)

Uses tmp Chroma path + tmp JSONL so it's safe to run while other processes
are live.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Enable memory + redirect paths BEFORE imports
os.environ["BUDDY_AGENT_MEMORY_ENABLED"] = "1"
_tmp_root = Path(tempfile.mkdtemp(prefix="buddy_mem_smoke_"))
_tmp_store = _tmp_root / "chroma"
_tmp_history = _tmp_root / "events.jsonl"

import src.scanner.agents.memory as _mem_mod  # noqa: E402
_mem_mod._STORE_PATH = _tmp_store

import src.scanner.automation.trading_event_bus as _bus_mod  # noqa: E402
_bus_mod._HISTORY_PATH = _tmp_history

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

from src.scanner.automation.trading_event_bus import get_trading_event_bus  # noqa: E402
from src.scanner.agents.memory import (  # noqa: E402
    cache_verdicts_for_trade_open,
    get_memory_store,
    register_memory_handlers,
)

register_memory_handlers()
store = get_memory_store()
assert store._ready, "memory store failed to initialize"
print(f"store path: {_tmp_store}")
print(f"initial count: {store.count()}")

bus = get_trading_event_bus()


class FakeVerdict:
    def __init__(self, name, score, reason):
        self.name = name
        self.score = score
        self.reason = reason


PAIR = "EUR_USD"
DIRECTION = "LONG"
REGIME = "NORMAL"

# Feed 20 synthetic setups that ALL lost (-12 to -20 pips) so the memory
# knows "when trend=0.65 on EUR_USD LONG at these features, we LOSE".
print("\n=== Phase 1: simulate 20 historical losing setups ===")
for i in range(20):
    trade_id = f"hist-{i:02d}"
    # Seed the cache (what would happen at evaluate time)
    cache_verdicts_for_trade_open(
        pair=PAIR,
        direction=DIRECTION,
        regime=REGIME,
        context={
            "adx": 18.0 + (i % 5),
            "rsi": 55.0 + (i % 3),
            "atr_pips": 12.0 + (i % 4),
            "spread_pips": 0.8,
            "uncertainty_score": 0.40,
            "model_disagreement": 0.28,
            "weighted_vote_score": 0.62,
            "confidence": 0.65,
        },
        verdicts=[
            FakeVerdict("trend", 0.65, "ADX ok"),
            FakeVerdict("momentum", 0.60, "RSI neutral"),
            FakeVerdict("risk_sentinel", 0.55, "exposure ok"),
        ],
    )
    # TRADE_OPENED -> writes cached verdicts to memory
    bus.emit({
        "event_id": str(uuid.uuid4()),
        "event_type": "trade_opened",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "smoke", "session_id": "mem_smoke",
        "correlation_id": str(uuid.uuid4()),
        "payload": {
            "trade_id": trade_id, "pair": PAIR, "direction": DIRECTION,
            "entry_price": 1.18, "sl_pips": 15, "tp_pips": 25, "lots": 1.0,
            "confidence": 0.65, "regime": REGIME, "model": "test",
        },
        "payload_version": 1,
    })
    # TRADE_CLOSED with -15 pips loss -> outcome backfill
    bus.emit({
        "event_id": str(uuid.uuid4()),
        "event_type": "trade_closed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "smoke", "session_id": "mem_smoke",
        "correlation_id": str(uuid.uuid4()),
        "payload": {
            "trade_id": trade_id, "pair": PAIR, "direction": DIRECTION,
            "entry_price": 1.18, "exit_price": 1.178,
            "realized_pl": -15.0, "pnl_pips": -15.0 - (i % 6),
            "trade_won": False, "exit_reason": "sl_hit", "duration_minutes": 42,
            "confidence": 0.65, "regime": REGIME,
            "agent_reasons": {}, "model": "test", "analysis_context": {},
            "close_time": datetime.now(timezone.utc).isoformat(),
            "sl_pips": 15, "tp_pips": 25,
            "mfe_pips": 0.5,
        },
        "payload_version": 1,
    })

print(f"after 20 trades: count={store.count()}")
assert store.count() == 60, f"expected 60 rows (20 trades × 3 agents), got {store.count()}"

# Now query with a similar context — expect a negative confidence_delta
print("\n=== Phase 2: query memory for a similar setup ===")
result = store.query_and_score(
    agent_name="trend",
    pair=PAIR,
    context={
        "score": 0.65,
        "adx": 20.0, "rsi": 56.0, "atr_pips": 13.0,
        "spread_pips": 0.8, "uncertainty_score": 0.40,
        "model_disagreement": 0.28, "weighted_vote_score": 0.62,
        "confidence": 0.65, "regime": REGIME, "direction": DIRECTION,
    },
    n_results=20,
    scope="self",
)
print(f"hits: n={result.n_hits} with_outcome={result.n_with_outcome}")
print(f"hit_rate={result.hit_rate:.2f} avg_pnl={result.avg_pnl_pips:.2f} delta={result.confidence_delta:+.3f}")
assert result.n_with_outcome >= 15, f"expected >=15 outcomes, got {result.n_with_outcome}"
assert result.hit_rate == 0.0, f"all trades lost, hit_rate should be 0, got {result.hit_rate}"
assert result.avg_pnl_pips < -10, f"expected avg<-10, got {result.avg_pnl_pips}"
assert result.confidence_delta <= -0.10, f"expected strong penalty, got {result.confidence_delta}"

print("\n=== Phase 3: query a DIFFERENT agent on same pair ===")
# The "momentum" agent should also have memories (same 20 trades)
result2 = store.query_and_score(
    agent_name="momentum",
    pair=PAIR,
    context={
        "score": 0.60,
        "adx": 20.0, "rsi": 56.0, "atr_pips": 13.0,
        "spread_pips": 0.8, "confidence": 0.65,
        "regime": REGIME, "direction": DIRECTION,
    },
    n_results=20, scope="self",
)
print(f"momentum hits: n={result2.n_hits} avg_pnl={result2.avg_pnl_pips}")
assert result2.n_with_outcome >= 15

print("\n=== Phase 4: query a FRESH context (different features) ===")
# A very different setup should have fewer similar hits (but still some —
# L2 kNN returns the N closest regardless)
result3 = store.query_and_score(
    agent_name="trend",
    pair=PAIR,
    context={
        "score": 0.80, "adx": 45.0, "rsi": 75.0, "atr_pips": 28.0,
        "spread_pips": 2.0, "uncertainty_score": 0.20, "model_disagreement": 0.10,
        "weighted_vote_score": 0.85, "confidence": 0.82,
        "regime": "HIGH", "direction": DIRECTION,
    },
    n_results=5, scope="self",
)
print(f"fresh-context hits: n={result3.n_hits}")
# Chroma still returns nearest neighbors, but they're far in feature space

print("\n✓ memory loop verified: write -> outcome -> query -> confidence_delta")
print(f"cleanup: rm -rf {_tmp_root}")
sys.exit(0)
