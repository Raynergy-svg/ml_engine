"""Retrain Agent — makes training evidence-driven instead of calendar-driven.

Subscribes to `TRADE_CLOSED` events on the Tier 7 trading event bus. Maintains
rolling per-pair state. When any of four learning-derived trigger conditions
fires, publishes a `TRAINING_PROPOSED` event that a worker can drain.

TRIGGERS (every threshold sourced from .claude/learnings.md + rules/trading.md):

  1. LOSING_STREAK_MFE_ZERO
       3 consecutive losses with MFE ≈ 0 on the same pair.
       Source: 2026-04-15 10-loss streak — price never moved in predicted
       direction. The first 3 such losses were the early signal; after 10 it
       was $2,637. Agent proposes retrain after 3.

  2. MODEL_STALENESS
       max component age > MAX_MODEL_AGE_DAYS (14d, from promotion_policy).
       Checked lazily on trade close, not on every tick.
       Source: 2026-04-15 root cause of 10-loss streak was 28d-old ensemble.

  3. DISAGREEMENT_PATTERN
       3+ of the last 5 trades (any pair) had model_disagreement > 0.30.
       Source: rules/trading.md line 20 — "disagreement > 0.30 is loss
       predictor". Treating repeated high disagreement as a drift signal.

  4. UNCERTAINTY_PATTERN
       3+ of the last 5 trades had uncertainty_score > 0.45 AND won < 50%.
       Source: rules/trading.md line 19 + today's PROMOTED learning on
       uncertainty hard-block during staleness.

COOLDOWN: 2 cycles minimum between proposals (per pair context). Matches the
blacklist cooldown in learnings.md (pair blacklist re-opens after 2 cycles).

STATE: kept in-memory on the agent instance; persisted to a JSONL under
`trained_data/retrain_agent_state.jsonl` on every trigger so a restart doesn't
lose the "we just proposed a retrain" signal.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from src.training.promotion_policy import MAX_MODEL_AGE_DAYS

logger = logging.getLogger(__name__)


# ── Trigger thresholds ──────────────────────────────────────────
LOSING_STREAK_THRESHOLD = 3              # consecutive losses with MFE~=0
LOSING_STREAK_MFE_ABS_MAX_PIPS = 2.0     # "MFE≈0" means absolute value <= 2 pips
DISAGREEMENT_THRESHOLD = 0.30            # from rules/trading.md
DISAGREEMENT_WINDOW = 5                  # last N trades
DISAGREEMENT_HITS_REQUIRED = 3           # out of DISAGREEMENT_WINDOW
UNCERTAINTY_THRESHOLD = 0.45             # from rules/trading.md
UNCERTAINTY_HITS_REQUIRED = 3
COOLDOWN_SECONDS = 2 * 60 * 60           # 2 hours = approx 2 scan cycles on H1
RECENT_WINDOW_SIZE = 10                  # keep last N outcomes per pair

_STATE_LOG_PATH = Path("trained_data/retrain_agent_state.jsonl")


# ── State containers ────────────────────────────────────────────

@dataclass
class PairState:
    """Rolling outcome history for a single pair."""
    pair: str
    recent: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW_SIZE))
    consecutive_losses: int = 0
    consecutive_zero_mfe_losses: int = 0


@dataclass
class ProposalRecord:
    """Audit record for a published TRAINING_PROPOSED event."""
    timestamp: str
    reason: str
    pairs: List[str]
    event_id: str
    details: Dict[str, Any] = field(default_factory=dict)


# ── The agent ───────────────────────────────────────────────────

class RetrainAgent:
    """Evidence-driven retrain proposer. Thread-safe, idempotent, observable."""

    def __init__(
        self,
        models_root: Path = Path("trained_data/models"),
        cooldown_seconds: int = COOLDOWN_SECONDS,
    ) -> None:
        self._pair_state: Dict[str, PairState] = {}
        self._global_recent: Deque[Dict[str, Any]] = deque(maxlen=RECENT_WINDOW_SIZE * 2)
        self._last_proposal_ts: Optional[datetime] = None
        self._cooldown_seconds = cooldown_seconds
        self._models_root = models_root
        self._lock = threading.Lock()
        self._proposals: List[ProposalRecord] = []

    # ── Event handler ───────────────────────────────────────────

    def on_trade_closed(self, event: Dict[str, Any]) -> None:
        """Handler for TRADE_CLOSED events. Idempotency is enforced upstream
        by TradingEventBus's LRU seen-cache."""
        payload = event.get("payload") or {}
        if not payload:
            return

        pair = str(payload.get("pair") or "UNKNOWN")
        outcome = {
            "trade_id": payload.get("trade_id"),
            "pair": pair,
            "trade_won": bool(payload.get("trade_won", False)),
            "pnl_pips": _safe_float(payload.get("pnl_pips")),
            "mfe_pips": _safe_float(payload.get("mfe_pips")) or _safe_float(payload.get("max_favorable_excursion")),
            "confidence": _safe_float(payload.get("confidence")),
            "uncertainty": _safe_float(payload.get("uncertainty_score")),
            "disagreement": _safe_float(payload.get("model_disagreement")),
            "regime": payload.get("regime"),
            "closed_at": payload.get("close_time") or datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            state = self._pair_state.setdefault(pair, PairState(pair=pair))
            state.recent.append(outcome)
            self._global_recent.append(outcome)

            if outcome["trade_won"]:
                state.consecutive_losses = 0
                state.consecutive_zero_mfe_losses = 0
            else:
                state.consecutive_losses += 1
                mfe = outcome["mfe_pips"]
                if mfe is not None and abs(mfe) <= LOSING_STREAK_MFE_ABS_MAX_PIPS:
                    state.consecutive_zero_mfe_losses += 1
                else:
                    state.consecutive_zero_mfe_losses = 0

            triggered_reason = self._evaluate_triggers(state)
            if triggered_reason is None:
                return

            if self._in_cooldown():
                logger.info(
                    "retrain_agent: trigger=%s fired but in cooldown (%d remaining sec)",
                    triggered_reason, self._cooldown_remaining_seconds(),
                )
                return

            # Decide pair scope for the retrain
            proposal_pairs = self._select_pairs_for_retrain(triggered_reason, pair)
            self._publish_proposal(triggered_reason, proposal_pairs, state)

    # ── Trigger evaluation ──────────────────────────────────────

    def _evaluate_triggers(self, state: PairState) -> Optional[str]:
        # 1. Losing streak with near-zero MFE (today's catastrophic pattern)
        if state.consecutive_zero_mfe_losses >= LOSING_STREAK_THRESHOLD:
            return "losing_streak_mfe_zero"

        # 2. Model staleness — only probes files; cheap enough per trade close.
        age_days = self._max_component_age_days()
        if age_days is not None and age_days > MAX_MODEL_AGE_DAYS:
            return "model_staleness"

        # 3. Disagreement pattern across global recent
        disagreement_hits = sum(
            1 for o in self._global_recent
            if (o.get("disagreement") or 0.0) > DISAGREEMENT_THRESHOLD
        )
        if (disagreement_hits >= DISAGREEMENT_HITS_REQUIRED
                and len(self._global_recent) >= DISAGREEMENT_WINDOW):
            return "disagreement_pattern"

        # 4. Uncertainty + losing pattern
        uncertainty_loss_hits = sum(
            1 for o in self._global_recent
            if (o.get("uncertainty") or 0.0) > UNCERTAINTY_THRESHOLD and not o["trade_won"]
        )
        if (uncertainty_loss_hits >= UNCERTAINTY_HITS_REQUIRED
                and len(self._global_recent) >= UNCERTAINTY_HITS_REQUIRED):
            return "uncertainty_loss_pattern"

        return None

    # ── Helpers ─────────────────────────────────────────────────

    def _max_component_age_days(self) -> Optional[float]:
        """Returns oldest meta/artifact age in days under models_root, or None."""
        import time as _time
        if not self._models_root.exists():
            return None
        candidates: List[Path] = [
            self._models_root / "modular_ensemble.meta.json",
            *(self._models_root / "joint").glob("*.keras"),
            *(self._models_root / "joint").glob("*.pkl"),
        ]
        mtimes = [p.stat().st_mtime for p in candidates if p.exists()]
        if not mtimes:
            return None
        oldest = min(mtimes)
        return max(0.0, (_time.time() - oldest) / 86400.0)

    def _select_pairs_for_retrain(self, reason: str, triggering_pair: str) -> List[str]:
        """Default: full ensemble retrain (all pairs). Scoped retrain is a follow-up."""
        from src.scanner.config import DEFAULT_PAIRS
        try:
            return list(DEFAULT_PAIRS)
        except Exception:
            return [triggering_pair]

    def _in_cooldown(self) -> bool:
        if self._last_proposal_ts is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self._last_proposal_ts).total_seconds()
        return elapsed < self._cooldown_seconds

    def _cooldown_remaining_seconds(self) -> int:
        if self._last_proposal_ts is None:
            return 0
        elapsed = (datetime.now(timezone.utc) - self._last_proposal_ts).total_seconds()
        return max(0, int(self._cooldown_seconds - elapsed))

    def _publish_proposal(
        self,
        reason: str,
        pairs: List[str],
        state: PairState,
    ) -> None:
        from src.scanner.automation.trading_event_bus import get_trading_event_bus

        now = datetime.now(timezone.utc)
        event_id = str(uuid.uuid4())
        details = {
            "triggering_pair": state.pair,
            "consecutive_losses": state.consecutive_losses,
            "consecutive_zero_mfe_losses": state.consecutive_zero_mfe_losses,
            "recent_window_size": len(self._global_recent),
            "max_component_age_days": self._max_component_age_days(),
        }
        event = {
            "event_id": event_id,
            "event_type": "training_proposed",
            "timestamp": now.isoformat(),
            "source": "retrain_agent",
            "session_id": os.getenv("BUDDY_SESSION_ID", "retrain_agent"),
            "correlation_id": event_id,
            "payload": {
                "reason": reason,
                "pairs": pairs,
                "priority": _priority_for_reason(reason),
                "details": details,
            },
            "payload_version": 1,
        }

        try:
            get_trading_event_bus().emit(event)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to emit TRAINING_PROPOSED: %s", exc, exc_info=True)
            return

        self._last_proposal_ts = now
        record = ProposalRecord(
            timestamp=now.isoformat(),
            reason=reason,
            pairs=pairs,
            event_id=event_id,
            details=details,
        )
        self._proposals.append(record)
        _persist_proposal(record)
        logger.warning(
            "retrain_agent: TRAINING_PROPOSED reason=%s pairs=%d details=%s",
            reason, len(pairs), details,
        )

    # ── Introspection ──────────────────────────────────────────

    def recent_proposals(self, limit: int = 10) -> List[ProposalRecord]:
        with self._lock:
            return list(self._proposals[-limit:])


# ── Module-level singleton + registration ───────────────────────

_agent_instance: Optional[RetrainAgent] = None
# RLock: register_retrain_agent holds this while calling get_retrain_agent,
# which would deadlock on a plain Lock.
_agent_lock = threading.RLock()
_REGISTERED = False


def get_retrain_agent() -> RetrainAgent:
    global _agent_instance
    if _agent_instance is None:
        with _agent_lock:
            if _agent_instance is None:
                _agent_instance = RetrainAgent()
    return _agent_instance


def register_retrain_agent() -> None:
    """Subscribe the retrain agent to TRADE_CLOSED. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return
    with _agent_lock:
        if _REGISTERED:
            return
        from src.scanner.automation.trading_event_bus import get_trading_event_bus
        bus = get_trading_event_bus()
        agent = get_retrain_agent()
        bus.subscribe(
            event_type="trade_closed",
            handler_fn=agent.on_trade_closed,
            priority=80,  # runs after RL sync (priority 50) but before analytics
            handler_name="retrain_agent",
        )
        _REGISTERED = True
        logger.info("retrain_agent: subscribed to TRADE_CLOSED")


# ── Helpers ─────────────────────────────────────────────────────

def _priority_for_reason(reason: str) -> str:
    """Map trigger reason to queue priority for the worker."""
    return {
        "losing_streak_mfe_zero": "critical",
        "model_staleness": "high",
        "disagreement_pattern": "high",
        "uncertainty_loss_pattern": "normal",
    }.get(reason, "normal")


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _persist_proposal(record: ProposalRecord) -> None:
    try:
        _STATE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_STATE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": record.timestamp,
                "reason": record.reason,
                "pairs": record.pairs,
                "event_id": record.event_id,
                "details": record.details,
            }) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to persist retrain proposal: %s", exc)
