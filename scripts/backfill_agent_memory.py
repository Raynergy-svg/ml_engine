"""Backfill agent memory from historical trade journal data.

Seeds the Chroma memory store with all past trades that have:
  - Agent verdicts (agents.agent_reasons[])
  - Realized outcomes (outcome.pnl_pips, outcome.trade_won)

Sources:
  - trained_data/trade_journal_rl.json (main journal, ~15 trades)
  - trained_data/trade_journal_low_rr_archive.json (rejected-but-tracked, ~4 trades)

After this script runs, agents have ~80 verdict memories with real outcomes from
day 1, so the confidence_delta nudge activates immediately instead of needing 5+
new trades to warm up.

Usage:
    BUDDY_AGENT_MEMORY_ENABLED=1 python scripts/backfill_agent_memory.py
"""
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("BUDDY_AGENT_MEMORY_ENABLED", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

from src.scanner.agents.memory import get_memory_store, is_enabled  # noqa: E402


def _safe_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _extract_features_from_trade(trade: dict) -> dict:
    """Build a feature context dict from a journal entry.

    Uses whatever fields are available — agent metadata, gate details, etc.
    Missing fields → 0.0 (the feature encoder normalizes anyway).
    """
    agents_data = trade.get("agents", {})
    # Pull uncertainty/disagreement from the uncertainty agent's metadata
    uncertainty_meta = {}
    for ar in agents_data.get("agent_reasons", []):
        if ar.get("name") == "uncertainty":
            uncertainty_meta = ar.get("metadata", {})
            break

    # Pull execution quality metadata
    exec_meta = {}
    for ar in agents_data.get("agent_reasons", []):
        if ar.get("name") == "execution_quality":
            exec_meta = ar.get("metadata", {})
            break

    return {
        "confidence": _safe_float(trade.get("confidence")),
        "adx": 0.0,  # not stored in journal (only in df_feat at scan time)
        "rsi": 0.0,
        "atr_pips": _safe_float(trade.get("sl_pips")) / 1.5 if trade.get("sl_pips") else 0.0,  # rough ATR proxy
        "spread_pips": _safe_float(exec_meta.get("spread_pips")),
        "uncertainty_score": _safe_float(uncertainty_meta.get("uncertainty_score")),
        "model_disagreement": _safe_float(uncertainty_meta.get("model_disagreement")),
        "weighted_vote_score": _safe_float(agents_data.get("weighted_vote_score")),
        "regime": str(trade.get("regime_at_entry") or trade.get("regime") or "UNKNOWN").upper(),
        "direction": str(trade.get("direction") or "HOLD").upper(),
    }


def backfill_from_journal(path: Path) -> int:
    """Read a trade journal JSON file and write all verdicts with outcomes to memory."""
    if not path.exists():
        logger.warning("Journal not found: %s", path)
        return 0

    with open(path) as f:
        journal = json.load(f)

    if not isinstance(journal, list):
        logger.warning("Journal is not a list: %s", path)
        return 0

    store = get_memory_store()
    written = 0
    skipped = 0

    for trade in journal:
        outcome = trade.get("outcome")
        if not outcome or outcome.get("pnl_pips") is None:
            skipped += 1
            continue

        trade_id = str(trade.get("trade_id", ""))
        pair = str(trade.get("pair", "UNKNOWN"))
        pnl_pips = _safe_float(outcome.get("pnl_pips"))
        trade_won = bool(outcome.get("trade_won", False))
        mfe_pips = _safe_float(outcome.get("max_favorable_excursion"))

        features = _extract_features_from_trade(trade)
        agent_reasons = trade.get("agents", {}).get("agent_reasons", [])

        for ar in agent_reasons:
            agent_name = ar.get("name", "unknown")
            score = _safe_float(ar.get("score"))
            reason = str(ar.get("reason", ""))[:200]

            ctx = dict(features)
            ctx["score"] = score

            row_id = store.record_verdict(
                agent_name=agent_name,
                pair=pair,
                verdict_score=score,
                verdict_reason=reason,
                context=ctx,
                trade_id=trade_id,
            )
            if row_id:
                store.record_outcome(
                    trade_id=trade_id,
                    pnl_pips=pnl_pips,
                    trade_won=trade_won,
                    mfe_pips=mfe_pips,
                )
                written += 1

    logger.info("Backfilled %d verdicts from %s (%d trades skipped — no outcome)", written, path.name, skipped)
    return written


def main():
    if not is_enabled():
        logger.error("BUDDY_AGENT_MEMORY_ENABLED is not set. Run with: BUDDY_AGENT_MEMORY_ENABLED=1")
        sys.exit(1)

    store = get_memory_store()
    initial = store.count()
    logger.info("Memory store count before backfill: %d", initial)

    total = 0
    sources = [
        PROJECT_ROOT / "trained_data" / "trade_journal_rl.json",
        PROJECT_ROOT / "trained_data" / "trade_journal_low_rr_archive.json",
    ]
    for src in sources:
        total += backfill_from_journal(src)

    final = store.count()
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("  sources: %s", [s.name for s in sources])
    logger.info("  verdicts written: %d", total)
    logger.info("  store count: %d → %d", initial, final)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
