"""Smoke test: TRADE_CLOSED → retrain_agent → TRAINING_PROPOSED → worker → status.

Runs the full autonomy-loop event chain in isolation. Uses a tmp JSONL path so
it does not deadlock against any live runtime (TUI, scanner) holding fcntl
LOCK_EX on trained_data/trading_events.jsonl.

Exit code:
    0  — all 3 trades emitted, >=1 proposal fired, no exceptions
    1  — any handler raised or chain broke
"""
import logging
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Isolate bus history so the smoke test doesn't fight the running TUI.
_tmp_history = Path(tempfile.mkdtemp(prefix="buddy_smoke_")) / "events.jsonl"
import src.scanner.automation.trading_event_bus as _bus_mod  # noqa: E402
_bus_mod._HISTORY_PATH = _tmp_history

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

from src.scanner.automation.trading_event_bus import get_trading_event_bus  # noqa: E402
from src.training.retrain_agent import get_retrain_agent, register_retrain_agent  # noqa: E402
from src.training.retrain_worker import register_retrain_worker  # noqa: E402
from src.training.training_events import register_training_handlers  # noqa: E402

register_retrain_agent()
register_retrain_worker()
register_training_handlers()

bus = get_trading_event_bus()
print("\nSubscribers:")
for et in ("trade_closed", "training_proposed", "training_completed"):
    print(f"  {et}: {[h.name for h in bus._handlers.get(et, [])]}")


def trade_closed(trade_id, pair="EUR_USD", won=False, mfe=0.2):
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "trade_closed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "test",
        "session_id": "smoke",
        "correlation_id": str(uuid.uuid4()),
        "payload": {
            "trade_id": trade_id,
            "pair": pair,
            "trade_won": won,
            "pnl_pips": -15.0,
            "mfe_pips": mfe,
            "confidence": 0.62,
            "uncertainty_score": 0.55,
            "model_disagreement": 0.35,
            "regime": "NORMAL",
            "close_time": datetime.now(timezone.utc).isoformat(),
        },
        "payload_version": 1,
    }


print("\n=== Firing 3 losing trades with MFE≈0 on EUR_USD ===")
for i, tid in enumerate(["t-1", "t-2", "t-3"], 1):
    print(f"\n-- trade {i}: {tid} --")
    bus.emit(trade_closed(tid))

proposals = get_retrain_agent().recent_proposals()
print("\n=== Proposals ===")
for p in proposals:
    print(f"  reason={p.reason} pairs={len(p.pairs)} trigger={p.details.get('triggering_pair')} event={p.event_id[:8]}")

assert len(proposals) >= 1, "expected at least 1 TRAINING_PROPOSED event"
print("\n✓ chain verified")
sys.exit(0)
