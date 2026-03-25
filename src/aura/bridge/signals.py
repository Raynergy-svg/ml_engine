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
