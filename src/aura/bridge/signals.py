"""Bridge signal contracts between Aura and Buddy.

Defines the JSON schemas for bidirectional data flow:
- ReadinessSignal: Human → Domain (already defined in aura.core.readiness)
- OutcomeSignal: Domain → Human (Buddy writes, Aura reads)
- OverrideEvent: Bidirectional (logged by both systems)

All signals are exchanged via JSON files in .aura/bridge/
Both systems can run independently — signals are read when available,
gracefully ignored when not.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BRIDGE_DIR = Path(".aura/bridge")


@dataclass
class OutcomeSignal:
    """Domain → Human: Buddy's daily trading summary for Aura's pattern engine.

    Written by Buddy after each trade cycle.
    Read by Aura's Tier 2 cross-domain pattern engine.
    """

    pnl_today: float = 0.0
    win_rate_7d: float = 0.0
    override_events: List[Dict[str, Any]] = field(default_factory=list)
    regime: str = "NORMAL"
    streak: str = "neutral"  # "winning", "losing", "neutral"
    trades_today: int = 0
    open_positions: int = 0
    max_drawdown_today: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pnl_today": round(self.pnl_today, 2),
            "win_rate_7d": round(self.win_rate_7d, 4),
            "override_events": self.override_events,
            "regime": self.regime,
            "streak": self.streak,
            "trades_today": self.trades_today,
            "open_positions": self.open_positions,
            "max_drawdown_today": round(self.max_drawdown_today, 2),
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class OverrideEvent:
    """Bidirectional: logged when trader overrides Buddy's signal.

    Both systems learn from this:
    - Buddy logs the market context (pair, direction, outcome)
    - Aura logs the human context (emotional state, conversation topics)
    """

    timestamp: str
    pair: str
    override_type: str          # "took_rejected", "skipped_recommended", "closed_early", "modified_sl_tp"
    buddy_recommendation: str   # What Buddy recommended
    trader_action: str          # What the trader actually did
    outcome: Optional[str] = None       # "win", "loss", or None if still open
    pnl_pips: float = 0.0
    # Human context (filled by Aura)
    emotional_state: str = ""
    cognitive_load: str = ""
    conversation_context: str = ""  # Summary of recent conversation
    # Market context (filled by Buddy)
    regime: str = ""
    confidence_at_time: float = 0.0
    weighted_vote_at_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "pair": self.pair,
            "override_type": self.override_type,
            "buddy_recommendation": self.buddy_recommendation,
            "trader_action": self.trader_action,
            "outcome": self.outcome,
            "pnl_pips": self.pnl_pips,
            "emotional_state": self.emotional_state,
            "cognitive_load": self.cognitive_load,
            "conversation_context": self.conversation_context,
            "regime": self.regime,
            "confidence_at_time": self.confidence_at_time,
            "weighted_vote_at_time": self.weighted_vote_at_time,
        }


@dataclass
class AuraQuery:
    """Buddy → Aura: A pre-trade question requiring a response.

    Buddy submits this before high-risk trades. Aura reads pending queries
    and writes back an AuraQueryResponse. If Aura is offline or slow (>25s),
    Buddy proceeds without blocking.

    query_types:
    - "readiness_check" — Am I ready to take a high-risk trade?
    - "override_pattern" — Have I been overriding impulsively?
    - "session_state"    — What's my emotional/cognitive state?
    - "risk_posture"     — Should I reduce risk given mental state?
    """
    query_id: str                            # uuid4 hex string
    query_type: str
    context: Dict[str, Any]                  # pair, confidence, regime, rr_ratio, etc.
    timestamp: str                           # ISO UTC
    expires_at: str                          # ISO UTC (+30s default)
    version: str = "1.0"
    answered: bool = False
    answer: Optional[Dict[str, Any]] = None
    answer_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_type": self.query_type,
            "context": self.context,
            "timestamp": self.timestamp,
            "expires_at": self.expires_at,
            "version": self.version,
            "answered": self.answered,
            "answer": self.answer,
            "answer_timestamp": self.answer_timestamp,
        }


@dataclass
class AuraQueryResponse:
    """Aura → Buddy: Response to an AuraQuery.

    proceed=True  → trade allowed (with optional confidence modifier)
    proceed=False → trade discouraged (confidence penalty applied)
    hard_veto=True → trade BLOCKED (only for critical emotional states)

    On timeout (no response in 25s): Buddy proceeds as if proceed=True.
    """
    query_id: str
    proceed: bool
    readiness_score: float                   # 0.0 - 1.0
    confidence_modifier: float               # clamp to [-0.3, +0.1]
    message: str
    hard_veto: bool = False
    version: str = "1.0"
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "proceed": self.proceed,
            "readiness_score": round(self.readiness_score, 4),
            "confidence_modifier": round(max(-0.3, min(0.1, self.confidence_modifier)), 4),
            "message": self.message,
            "hard_veto": self.hard_veto,
            "version": self.version,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }


class FeedbackBridge:
    """Manages bidirectional signal flow between Aura and Buddy.

    File-based bridge — both systems read/write JSON files in .aura/bridge/.
    This keeps systems decoupled while enabling data sharing.
    """

    def __init__(self, bridge_dir: Optional[Path] = None):
        self.bridge_dir = bridge_dir or BRIDGE_DIR
        self.bridge_dir.mkdir(parents=True, exist_ok=True)
        self._outcome_path = self.bridge_dir / "outcome_signal.json"
        self._override_log_path = self.bridge_dir / "override_events.jsonl"
        self._readiness_path = self.bridge_dir / "readiness_signal.json"

    # --- US-202: File-locking helpers for concurrent access safety ---

    @staticmethod
    def _locked_write(path: Path, data: str) -> None:
        """Atomic write with exclusive file lock (fcntl.LOCK_EX).

        Uses temp-file + rename for crash safety so readers never see
        partial content.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".bridge_"
            )
            os.write(fd, data.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            fd = None  # mark as closed
            # Atomic rename (POSIX guarantees atomicity on same filesystem)
            os.rename(tmp_path, str(path))
            tmp_path = None  # mark as renamed
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _locked_read(path: Path) -> Optional[str]:
        """Read with shared file lock (fcntl.LOCK_SH)."""
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    return f.read()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (OSError, IOError) as e:
            logger.warning(f"Bridge: locked read failed for {path.name}: {e}")
            return None

    @staticmethod
    def _locked_append(path: Path, line: str) -> None:
        """Append with exclusive file lock (fcntl.LOCK_EX)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (OSError, IOError) as e:
            logger.error(f"Bridge: locked append failed for {path.name}: {e}")

    # --- Outcome Signal (Buddy → Aura) ---

    def write_outcome(self, signal: OutcomeSignal) -> None:
        """Buddy writes its trading summary for Aura to read."""
        try:
            self._locked_write(
                self._outcome_path,
                json.dumps(signal.to_dict(), indent=2, default=str),
            )
            logger.debug(f"Bridge: wrote outcome signal (PnL: {signal.pnl_today:+.2f})")
        except Exception as e:
            logger.error(f"Bridge: failed to write outcome signal: {e}")

    def read_outcome(self) -> Optional[OutcomeSignal]:
        """Aura reads Buddy's latest trading summary."""
        raw = self._locked_read(self._outcome_path)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return OutcomeSignal(**{
                k: v for k, v in data.items()
                if k in OutcomeSignal.__dataclass_fields__
            })
        except Exception as e:
            logger.warning(f"Bridge: failed to read outcome signal: {e}")
            return None

    # --- Override Events (Bidirectional) ---

    def log_override(self, event: OverrideEvent) -> None:
        """Log an override event (appendable JSONL format)."""
        try:
            self._locked_append(
                self._override_log_path,
                json.dumps(event.to_dict(), default=str) + "\n",
            )
            logger.info(
                f"Bridge: override logged — {event.pair} {event.override_type} "
                f"(emotional: {event.emotional_state or 'unknown'})"
            )
        except Exception as e:
            logger.error(f"Bridge: failed to log override: {e}")

    def get_recent_overrides(self, limit: int = 20) -> List[OverrideEvent]:
        """Read recent override events."""
        raw = self._locked_read(self._override_log_path)
        if raw is None:
            return []
        try:
            events = []
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        events.append(OverrideEvent(**{
                            k: v for k, v in data.items()
                            if k in OverrideEvent.__dataclass_fields__
                        }))
                    except Exception:
                        continue
            return events[-limit:]
        except Exception as e:
            logger.warning(f"Bridge: failed to read overrides: {e}")
            return []

    # --- Readiness Signal (Aura → Buddy) ---

    def read_readiness(self) -> Optional[Dict[str, Any]]:
        """Read Aura's readiness signal (convenience method)."""
        raw = self._locked_read(self._readiness_path)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Bridge: failed to read readiness signal: {e}")
            return None

    # --- Bridge Statistics ---

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get the current state of all bridge signals."""
        readiness = self.read_readiness()
        outcome = self.read_outcome()
        overrides = self.get_recent_overrides(limit=5)

        return {
            "readiness_signal": {
                "available": readiness is not None,
                "score": readiness.get("readiness_score") if readiness else None,
                "timestamp": readiness.get("timestamp") if readiness else None,
            },
            "outcome_signal": {
                "available": outcome is not None,
                "pnl_today": outcome.pnl_today if outcome else None,
                "timestamp": outcome.timestamp if outcome else None,
            },
            "override_events": {
                "total_recent": len(overrides),
                "last_override": overrides[-1].to_dict() if overrides else None,
            },
        }

    # --- AuraQuery Protocol (Tier 6 Co-Cognition) ---

    def submit_query(self, query: "AuraQuery") -> None:
        """Buddy submits a pre-trade query for Aura to answer."""
        queries_dir = self.bridge_dir / "queries"
        queries_dir.mkdir(parents=True, exist_ok=True)
        path = queries_dir / f"{query.query_id}.json"
        try:
            self._locked_write(path, json.dumps(query.to_dict(), indent=2, sort_keys=True))
            logger.info("AuraQuery submitted: %s type=%s", query.query_id[:8], query.query_type)
        except Exception as e:
            logger.error("AuraQuery: failed to submit query %s: %s", query.query_id[:8], e)

    def read_response(self, query_id: str) -> Optional["AuraQueryResponse"]:
        """Read Aura's response to a query. Returns None if not answered yet."""
        responses_dir = self.bridge_dir / "responses"
        path = responses_dir / f"{query_id}.json"
        raw = self._locked_read(path)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return AuraQueryResponse(**{
                k: v for k, v in data.items()
                if k in AuraQueryResponse.__dataclass_fields__
            })
        except Exception as e:
            logger.warning("AuraQuery: failed to parse response %s: %s", query_id[:8], e)
            return None

    def wait_for_response(
        self, query_id: str, timeout_sec: float = 25.0
    ) -> Optional["AuraQueryResponse"]:
        """Poll for Aura's response. Returns None on timeout (non-blocking by design)."""
        import time
        deadline = time.monotonic() + timeout_sec
        poll_interval = 0.5
        while time.monotonic() < deadline:
            response = self.read_response(query_id)
            if response is not None:
                logger.info(
                    "AuraQuery: response received for %s in %.1fs (proceed=%s)",
                    query_id[:8], timeout_sec - (deadline - time.monotonic()), response.proceed
                )
                return response
            time.sleep(poll_interval)
        logger.info("AuraQuery: timeout waiting for %s — proceeding without Aura", query_id[:8])
        return None

    def get_pending_queries(self) -> List["AuraQuery"]:
        """Aura calls this to read all unanswered queries from Buddy."""
        queries_dir = self.bridge_dir / "queries"
        if not queries_dir.exists():
            return []
        results: List["AuraQuery"] = []
        for path in sorted(queries_dir.glob("*.json")):
            raw = self._locked_read(path)
            if raw is None:
                continue
            try:
                data = json.loads(raw)
                if not data.get("answered", False):
                    query = AuraQuery(**{
                        k: v for k, v in data.items()
                        if k in AuraQuery.__dataclass_fields__
                    })
                    results.append(query)
            except Exception as e:
                logger.warning("AuraQuery: failed to parse query %s: %s", path.name, e)
        return results

    def submit_response(self, response: "AuraQueryResponse") -> None:
        """Aura writes its response so Buddy can read it."""
        responses_dir = self.bridge_dir / "responses"
        responses_dir.mkdir(parents=True, exist_ok=True)
        path = responses_dir / f"{response.query_id}.json"
        try:
            resp_data = response.to_dict()
            if not resp_data.get("timestamp"):
                resp_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            self._locked_write(path, json.dumps(resp_data, indent=2, sort_keys=True))
            logger.info(
                "AuraQuery: response written for %s (proceed=%s, veto=%s)",
                response.query_id[:8], response.proceed, response.hard_veto
            )
        except Exception as e:
            logger.error("AuraQuery: failed to write response %s: %s", response.query_id[:8], e)

    def cleanup_stale_queries(self, max_age_sec: float = 120.0) -> int:
        """Remove expired query/response files. Returns count of files removed."""
        import time
        removed = 0
        now = time.time()
        for subdir_name in ("queries", "responses"):
            subdir = self.bridge_dir / subdir_name
            if not subdir.exists():
                continue
            for path in list(subdir.glob("*.json")):
                try:
                    if (now - path.stat().st_mtime) > max_age_sec:
                        path.unlink(missing_ok=True)
                        removed += 1
                except OSError as e:
                    logger.warning("AuraQuery: cleanup failed for %s: %s", path.name, e)
        if removed:
            logger.info("AuraQuery: cleaned up %d stale query/response files", removed)
        return removed
