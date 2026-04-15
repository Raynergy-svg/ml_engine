"""AgentMemoryStore — typed experience log for the 12 scoring agents.

Each agent's prior verdict + the trade outcome that followed it is persisted to
an embedded Chroma collection so later calls can answer:

    "The last N times I (the trend agent) scored EUR_USD in a LOW regime with
     ADX<15, what did I say and what actually happened?"

Storage: ONE Chroma collection (`buddy_agent_memory`) keyed by agent_name in
metadata. Cross-agent pattern mining works via `where={"agent_name": {"$in":
[...]}}` filters. Per-pair 10k soft cap with summary-compression of older rows.

Integration:
- Called by ScannerAgentTeam._evaluate_* via `memory.query_and_score(...)` to
  get a confidence delta based on similar past outcomes. Returns a MemoryHit
  bundle with avg_pnl_pips, hit_rate, and n_hits.
- Subscribes to TRADE_CLOSED on the trading event bus; backfills outcome
  metadata into the rows written at trade entry.
- Env-gated: BUDDY_AGENT_MEMORY_ENABLED. When off, all methods no-op.
- Storage path: trained_data/agent_memory/ (SQLite + parquet under the hood).

Embedding strategy (per user decision): raw numeric feature vector. No LLM, no
sentence-transformer. We use Chroma's "bring your own embedding" path so the
stored vector IS the feature snapshot — similarity search becomes a
well-defined numeric kNN over trading features.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.observability import op as weave_op

logger = logging.getLogger(__name__)


# ── Env gates + config ──────────────────────────────────────────
_ENV_FLAG = "BUDDY_AGENT_MEMORY_ENABLED"
_STORE_PATH = Path("trained_data/agent_memory")
_COLLECTION_NAME = "buddy_agent_memory"
_PER_PAIR_SOFT_CAP = 10_000  # rows per pair before compression
_DEFAULT_FEATURE_DIM = 16   # padded feature vector size; see _encode_features


def _env_truthy(value: Optional[str]) -> bool:
    return bool(value) and value.strip().lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    return _env_truthy(os.getenv(_ENV_FLAG))


# ── Data shapes ─────────────────────────────────────────────────

@dataclass
class MemoryHit:
    """A single retrieved past decision + its outcome."""
    id: str
    agent_name: str
    pair: str
    regime: str
    verdict_score: float
    verdict_reason: str
    outcome_pnl_pips: Optional[float] = None
    outcome_won: Optional[bool] = None
    outcome_mfe_pips: Optional[float] = None
    distance: float = 0.0
    timestamp: str = ""


@dataclass
class MemoryQueryResult:
    """Aggregate summary returned to a calling agent.

    The agent uses this to adjust its verdict's `confidence_delta` — e.g.
    penalize when historical similar setups averaged negative P/L.
    """
    n_hits: int
    n_with_outcome: int              # subset that have been closed
    hit_rate: float                  # fraction of closed outcomes that won
    avg_pnl_pips: Optional[float]    # mean realized P/L on closed outcomes
    median_pnl_pips: Optional[float]
    confidence_delta: float          # suggested adjustment in [-0.20, +0.10]
    hits: List[MemoryHit] = field(default_factory=list)


# ── Feature encoding ────────────────────────────────────────────

def _encode_features(context: Dict[str, Any], dim: int = _DEFAULT_FEATURE_DIM) -> List[float]:
    """Encode a verdict context into a fixed-length numeric vector.

    This is the "embedding" — no NLP, no LLM. Each slot is a normalized
    trading feature. Unknown/missing → 0.0. Chroma will k-NN on L2 distance.

    Order is stable: changing it requires clearing the memory store.
    """
    slots = [0.0] * dim
    slots[0] = _safe_float(context.get("score"))
    slots[1] = _safe_float(context.get("confidence"))
    slots[2] = _safe_float(context.get("adx")) / 60.0     # ADX 0-60 typical
    slots[3] = _safe_float(context.get("rsi")) / 100.0    # RSI 0-100
    slots[4] = _safe_float(context.get("atr_pips")) / 30.0  # typical 0-30 pips
    slots[5] = _safe_float(context.get("spread_pips")) / 5.0
    slots[6] = _safe_float(context.get("uncertainty_score"))
    slots[7] = _safe_float(context.get("model_disagreement"))
    slots[8] = _safe_float(context.get("weighted_vote_score"))
    # Regime one-hot (LOW/NORMAL/HIGH/EXTREME)
    regime = str(context.get("regime") or "").upper()
    slots[9] = 1.0 if regime == "LOW" else 0.0
    slots[10] = 1.0 if regime == "NORMAL" else 0.0
    slots[11] = 1.0 if regime == "HIGH" else 0.0
    slots[12] = 1.0 if regime == "EXTREME" else 0.0
    # Direction
    direction = str(context.get("direction") or "").upper()
    slots[13] = 1.0 if direction == "LONG" else (-1.0 if direction == "SHORT" else 0.0)
    # Session hour (UTC) normalized to [0,1]
    try:
        hour = datetime.now(timezone.utc).hour
        slots[14] = hour / 24.0
    except Exception:
        pass
    # Reserved slot for future features (e.g. correlation_score)
    slots[15] = _safe_float(context.get("reserved_0"))
    return slots


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# ── The store ───────────────────────────────────────────────────

class AgentMemoryStore:
    """Chroma-backed experience log. Thread-safe, env-gated, no-op when off."""

    def __init__(self, path: Path = _STORE_PATH) -> None:
        self._path = path
        self._client = None
        self._collection = None
        self._ready = False
        self._lock = threading.RLock()
        self._outcome_index: Dict[str, List[str]] = {}  # trade_id -> row_ids
        if is_enabled():
            self._lazy_init()

    # ── Initialization ──────────────────────────────────────────

    def _lazy_init(self) -> None:
        if self._ready or not is_enabled():
            return
        try:
            import chromadb
            from chromadb.config import Settings

            self._path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self._path),
                settings=Settings(anonymized_telemetry=False, allow_reset=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "l2"},
            )
            self._ready = True
            count = self._collection.count()
            logger.info("AgentMemoryStore ready at %s (%d rows)", self._path, count)
        except Exception as exc:  # noqa: BLE001
            logger.error("AgentMemoryStore init failed (will run disabled): %s", exc)
            self._ready = False

    # ── Write path ──────────────────────────────────────────────

    def record_verdict(
        self,
        agent_name: str,
        pair: str,
        verdict_score: float,
        verdict_reason: str,
        context: Dict[str, Any],
        trade_id: Optional[str] = None,
    ) -> Optional[str]:
        """Persist an agent's verdict for future retrieval.

        Returns the assigned row_id, or None if disabled/failed.
        Outcome fields are filled later by `record_outcome(trade_id, ...)`.
        """
        if not self._ready:
            return None
        row_id = f"{agent_name}:{pair}:{int(time.time()*1000)}:{verdict_score:.4f}"
        try:
            embedding = _encode_features(context)
            regime = str(context.get("regime") or "UNKNOWN").upper()
            metadata = {
                "agent_name": agent_name,
                "pair": pair,
                "regime": regime,
                "direction": str(context.get("direction") or "HOLD").upper(),
                "verdict_score": float(verdict_score),
                "verdict_reason": str(verdict_reason)[:200],
                "confidence": _safe_float(context.get("confidence")),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trade_id": str(trade_id) if trade_id else "",
                # Outcome fields stay empty until backfill
                "outcome_pnl_pips": 0.0,
                "outcome_won": False,
                "outcome_mfe_pips": 0.0,
                "has_outcome": False,
            }
            with self._lock:
                self._collection.add(
                    ids=[row_id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[verdict_reason[:500]],
                )
                if trade_id:
                    self._outcome_index.setdefault(str(trade_id), []).append(row_id)
            return row_id
        except Exception as exc:  # noqa: BLE001
            logger.debug("record_verdict failed for %s/%s: %s", agent_name, pair, exc)
            return None

    def record_outcome(
        self,
        trade_id: str,
        pnl_pips: float,
        trade_won: bool,
        mfe_pips: Optional[float] = None,
    ) -> int:
        """Backfill outcome metadata onto all rows linked to this trade.

        Called by the TRADE_CLOSED subscriber. Returns number of rows updated.
        """
        if not self._ready:
            return 0
        try:
            with self._lock:
                row_ids = self._outcome_index.pop(str(trade_id), [])
                if not row_ids:
                    # Fallback: query by metadata (may miss if index was cleared)
                    found = self._collection.get(where={"trade_id": str(trade_id)})
                    row_ids = found.get("ids", []) if found else []
                if not row_ids:
                    return 0
                existing = self._collection.get(ids=row_ids)
                updated_metas = []
                for m in existing.get("metadatas") or []:
                    m = dict(m or {})
                    m["outcome_pnl_pips"] = float(pnl_pips)
                    m["outcome_won"] = bool(trade_won)
                    m["outcome_mfe_pips"] = float(mfe_pips) if mfe_pips is not None else 0.0
                    m["has_outcome"] = True
                    updated_metas.append(m)
                if updated_metas:
                    self._collection.update(ids=row_ids, metadatas=updated_metas)
                return len(row_ids)
        except Exception as exc:  # noqa: BLE001
            logger.debug("record_outcome failed for trade_id=%s: %s", trade_id, exc)
            return 0

    # ── Read path (the interesting one) ────────────────────────

    @weave_op
    def query_and_score(
        self,
        agent_name: str,
        pair: str,
        context: Dict[str, Any],
        n_results: int = 20,
        scope: str = "self",  # "self" | "pair" | "cross"
    ) -> MemoryQueryResult:
        """Retrieve similar past decisions and compute a confidence adjustment.

        scope:
            "self"  - only this agent's own past verdicts (default)
            "pair"  - any agent on the same pair (cross-agent mining)
            "cross" - any agent, any pair — use sparingly

        The returned `confidence_delta` is the suggested nudge to the current
        verdict's `confidence_delta` field. Conservative clamps: [-0.20, +0.10].
        """
        empty = MemoryQueryResult(
            n_hits=0, n_with_outcome=0, hit_rate=0.0,
            avg_pnl_pips=None, median_pnl_pips=None,
            confidence_delta=0.0, hits=[],
        )
        if not self._ready:
            return empty
        try:
            where_filter: Dict[str, Any] = {"pair": pair}
            if scope == "self":
                where_filter = {"$and": [{"agent_name": agent_name}, {"pair": pair}]}
            elif scope == "pair":
                where_filter = {"pair": pair}
            # "cross" leaves filter unconstrained

            embedding = _encode_features(context)
            with self._lock:
                raw = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=n_results,
                    where=where_filter,
                )

            hits = _parse_hits(raw)
            if not hits:
                return empty

            outcomes = [h for h in hits if h.outcome_pnl_pips is not None]
            n_with_outcome = len(outcomes)
            if n_with_outcome == 0:
                return MemoryQueryResult(
                    n_hits=len(hits), n_with_outcome=0, hit_rate=0.0,
                    avg_pnl_pips=None, median_pnl_pips=None,
                    confidence_delta=0.0, hits=hits,
                )

            pnls = sorted([h.outcome_pnl_pips for h in outcomes if h.outcome_pnl_pips is not None])
            avg_pnl = sum(pnls) / len(pnls) if pnls else None
            median_pnl = pnls[len(pnls) // 2] if pnls else None
            wins = sum(1 for h in outcomes if h.outcome_won)
            hit_rate = wins / n_with_outcome

            delta = _confidence_delta_from_outcomes(
                avg_pnl=avg_pnl,
                hit_rate=hit_rate,
                n_with_outcome=n_with_outcome,
            )
            return MemoryQueryResult(
                n_hits=len(hits),
                n_with_outcome=n_with_outcome,
                hit_rate=hit_rate,
                avg_pnl_pips=avg_pnl,
                median_pnl_pips=median_pnl,
                confidence_delta=delta,
                hits=hits,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("query_and_score failed: %s", exc)
            return empty

    # ── Maintenance ─────────────────────────────────────────────

    def compact_if_needed(self, pair: str) -> int:
        """If `pair` exceeds the per-pair soft cap, drop the oldest rows.

        Returns number of rows removed. Called opportunistically; safe to skip.
        A future improvement is to replace drops with a summary row ("prior
        calibration: avg_pnl_over_5k_samples=..., wr=...") but for now we just
        drop to keep the collection bounded.
        """
        if not self._ready:
            return 0
        try:
            with self._lock:
                count = self._collection.count()
                if count < _PER_PAIR_SOFT_CAP:
                    return 0
                # Pull all for pair, sort by timestamp, drop oldest half.
                found = self._collection.get(where={"pair": pair}, limit=_PER_PAIR_SOFT_CAP * 2)
                ids = found.get("ids") or []
                metas = found.get("metadatas") or []
                if len(ids) <= _PER_PAIR_SOFT_CAP:
                    return 0
                ranked = sorted(
                    zip(ids, metas),
                    key=lambda t: (t[1] or {}).get("timestamp", ""),
                )
                to_drop = [i for i, _ in ranked[: len(ids) - _PER_PAIR_SOFT_CAP // 2]]
                if to_drop:
                    self._collection.delete(ids=to_drop)
                    logger.info("compact_if_needed(%s): dropped %d rows", pair, len(to_drop))
                return len(to_drop)
        except Exception as exc:  # noqa: BLE001
            logger.debug("compact failed for %s: %s", pair, exc)
            return 0

    def count(self) -> int:
        if not self._ready:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0


# ── Helpers ─────────────────────────────────────────────────────

def _parse_hits(raw: Dict[str, Any]) -> List[MemoryHit]:
    """Chroma returns lists of lists (one per query). Flatten + parse."""
    ids = (raw.get("ids") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    out: List[MemoryHit] = []
    for i, m, d in zip(ids, metas, distances):
        m = m or {}
        pnl = m.get("outcome_pnl_pips")
        has_out = bool(m.get("has_outcome"))
        out.append(MemoryHit(
            id=str(i),
            agent_name=str(m.get("agent_name", "")),
            pair=str(m.get("pair", "")),
            regime=str(m.get("regime", "")),
            verdict_score=_safe_float(m.get("verdict_score")),
            verdict_reason=str(m.get("verdict_reason", "")),
            outcome_pnl_pips=float(pnl) if (has_out and pnl is not None) else None,
            outcome_won=bool(m.get("outcome_won")) if has_out else None,
            outcome_mfe_pips=float(m.get("outcome_mfe_pips") or 0.0) if has_out else None,
            distance=float(d) if d is not None else 0.0,
            timestamp=str(m.get("timestamp", "")),
        ))
    return out


def _confidence_delta_from_outcomes(
    avg_pnl: Optional[float],
    hit_rate: float,
    n_with_outcome: int,
) -> float:
    """Map observed history to a confidence nudge.

    Rules (tunable, tight clamps):
      - < 5 outcomes: no signal, delta=0
      - avg_pnl > +5 pips AND hit_rate > 0.55: boost +0.05
      - avg_pnl between -2 and +5: neutral 0
      - avg_pnl between -5 and -2 OR hit_rate < 0.45: penalize -0.05
      - avg_pnl < -5 OR hit_rate < 0.35: penalize -0.10
      - avg_pnl < -15 (catastrophic cluster): penalize -0.20 (max clamp)

    Values picked from .claude/rules/trading.md intuitions: "+0.05 boost on
    confirmed edge, up to -0.20 penalty when history screams don't do it".
    """
    if n_with_outcome < 5 or avg_pnl is None:
        return 0.0
    if avg_pnl < -15:
        return -0.20
    if avg_pnl < -5 or hit_rate < 0.35:
        return -0.10
    if avg_pnl < -2 or hit_rate < 0.45:
        return -0.05
    if avg_pnl > 5 and hit_rate > 0.55:
        return 0.05
    return 0.0


# ── Module-level singleton + bus subscription ──────────────────

_instance: Optional[AgentMemoryStore] = None
_instance_lock = threading.RLock()
_subscribed = False


def get_memory_store() -> AgentMemoryStore:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = AgentMemoryStore()
    return _instance


def register_memory_handlers() -> None:
    """Subscribe the memory store to TRADE_CLOSED for outcome backfill."""
    global _subscribed
    if _subscribed or not is_enabled():
        return
    with _instance_lock:
        if _subscribed:
            return
        try:
            from src.scanner.automation.trading_event_bus import get_trading_event_bus
            bus = get_trading_event_bus()
            bus.subscribe(
                event_type="trade_closed",
                handler_fn=_on_trade_closed,
                priority=60,  # after RL sync (50), before retrain_agent (80)
                handler_name="agent_memory_outcome_backfill",
            )
            bus.subscribe(
                event_type="trade_opened",
                handler_fn=_on_trade_opened,
                priority=60,
                handler_name="agent_memory_verdict_writer",
            )
            _subscribed = True
            logger.info("agent_memory: subscribed to TRADE_OPENED + TRADE_CLOSED")
        except Exception as exc:  # noqa: BLE001
            logger.error("agent_memory bus subscription failed: %s", exc)


def _on_trade_closed(event: Dict[str, Any]) -> None:
    """Bus handler — updates memory rows with realized outcomes."""
    payload = event.get("payload") or {}
    trade_id = payload.get("trade_id")
    if not trade_id:
        return
    store = get_memory_store()
    updated = store.record_outcome(
        trade_id=str(trade_id),
        pnl_pips=_safe_float(payload.get("pnl_pips")),
        trade_won=bool(payload.get("trade_won", False)),
        mfe_pips=_safe_float(payload.get("mfe_pips")) if payload.get("mfe_pips") is not None else None,
    )
    if updated:
        logger.info("agent_memory: backfilled %d rows for trade_id=%s", updated, trade_id)


# ── Verdict cache + TRADE_OPENED write path ────────────────────
#
# Problem: at evaluate() time we don't know if the scan will become a trade.
# Recording all verdicts would flood memory with non-traded noise.
# Solution: cache recent verdicts per (pair, direction) during scan; when
# TRADE_OPENED fires for that pair, flush the matching cache entry into memory
# tagged with the real trade_id. Bounded cache so restarts lose at most a few.

_VERDICT_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_CACHE_LOCK = threading.RLock()
_CACHE_TTL_SECONDS = 600  # 10-minute entry TTL; older entries drop on next write


def cache_verdicts_for_trade_open(
    pair: str,
    direction: str,
    regime: str,
    context: Dict[str, Any],
    verdicts: List[Any],
) -> None:
    """Stash verdicts keyed by (pair, direction) for later memory write.

    Called by ScannerAgentTeam._evaluate_body after verdicts are finalized.
    Only the most recent verdict-set per (pair, direction) is kept.
    """
    if not is_enabled():
        return
    key = (pair.upper(), direction.upper())
    with _CACHE_LOCK:
        _VERDICT_CACHE[key] = {
            "ts": time.time(),
            "pair": pair,
            "direction": direction,
            "regime": regime,
            "context": dict(context),
            "verdicts": [
                {
                    "name": getattr(v, "name", "unknown"),
                    "score": float(getattr(v, "score", 0.0)),
                    "reason": str(getattr(v, "reason", ""))[:200],
                }
                for v in verdicts
            ],
        }
        _evict_stale_cache_entries()


def _evict_stale_cache_entries() -> None:
    """Drop cache entries older than TTL. Caller holds _CACHE_LOCK."""
    now = time.time()
    stale = [k for k, v in _VERDICT_CACHE.items() if now - v.get("ts", 0) > _CACHE_TTL_SECONDS]
    for k in stale:
        _VERDICT_CACHE.pop(k, None)


def _on_trade_opened(event: Dict[str, Any]) -> None:
    """Bus handler — flushes cached verdicts into memory tagged with trade_id."""
    payload = event.get("payload") or {}
    pair = str(payload.get("pair") or "").upper()
    direction = str(payload.get("direction") or "").upper()
    trade_id = payload.get("trade_id")
    if not (pair and direction and trade_id):
        return

    with _CACHE_LOCK:
        entry = _VERDICT_CACHE.pop((pair, direction), None)

    if entry is None:
        logger.debug("agent_memory: no cached verdicts for %s/%s — trade_id=%s not indexed",
                     pair, direction, trade_id)
        return

    store = get_memory_store()
    written = 0
    for vd in entry["verdicts"]:
        row_ctx = dict(entry["context"])
        row_ctx["score"] = vd["score"]
        row_ctx["regime"] = entry["regime"]
        row_ctx["direction"] = direction
        row_id = store.record_verdict(
            agent_name=vd["name"],
            pair=pair,
            verdict_score=vd["score"],
            verdict_reason=vd["reason"],
            context=row_ctx,
            trade_id=str(trade_id),
        )
        if row_id:
            written += 1
    if written:
        logger.info("agent_memory: recorded %d verdicts for trade_id=%s (%s/%s)",
                    written, trade_id, pair, direction)
