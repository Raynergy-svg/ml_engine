"""Virtual trade logger — capture rejected-but-promising setups for offline RL.

Phase 56 US-348: When a pair generates a valid signal direction but fails
execution gates, log it as a "virtual trade" with entry price, gate failure
reasons, and agent scores. These entries provide pseudo-data for offline
RL analysis and help diagnose why no real trades are executing.

Uses append-only JSONL format to avoid full-file rewrites on each scan.

Usage:
    logger = VirtualTradeLogger()
    logger.log_virtual_trade(
        pair="EUR_USD",
        direction="LONG",
        confidence=0.42,
        gate_failures=["confidence=0.42 < 0.60", "agent_passed=False"],
        features={"momentum": 0.3, "trend_strength": 0.5},
        agent_scores={"trend": 0.7, "risk_sentinel": 0.3},
    )
    recent = logger.get_recent(n=20)
    stats = logger.get_stats()
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_LOG_PATH = _PROJECT_ROOT / "trained_data" / "virtual_trades.jsonl"

# Minimum confidence to log as virtual trade (very low signals not worth capturing)
MIN_CONFIDENCE_TO_LOG = 0.10


class VirtualTradeLogger:
    """Log rejected trade setups for offline RL analysis.

    Writes append-only JSONL so each scan adds new lines without touching
    old entries. get_recent() reads the tail of the file. get_stats()
    aggregates counters from all logged entries.

    Usage:
        logger = VirtualTradeLogger()
        logger.log_virtual_trade("EUR_USD", "LONG", 0.42, ["confidence < 0.60"], {}, {})
        recent = logger.get_recent(20)
        stats = logger.get_stats()
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        min_confidence: float = MIN_CONFIDENCE_TO_LOG,
    ):
        self.log_path = Path(log_path or _DEFAULT_LOG_PATH)
        self.min_confidence = float(min_confidence)

        # In-memory counters (rebuilt from file on init)
        self._total_logged: int = 0
        self._pair_counts: Counter = Counter()
        self._gate_failure_counts: Counter = Counter()

        self._load_counts()

    # ── Public API ──────────────────────────────────────────────────────────

    def log_virtual_trade(
        self,
        pair: str,
        direction: str,
        confidence: float,
        gate_failures: Optional[List[str]] = None,
        features: Optional[Dict[str, float]] = None,
        agent_scores: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Log a rejected setup as a virtual trade.

        Args:
            pair: Currency pair (e.g. "EUR_USD")
            direction: Signal direction ("LONG" or "SHORT")
            confidence: Raw confidence score at time of rejection
            gate_failures: List of gate failure reason strings
            features: Feature snapshot dict at scan time
            agent_scores: Agent scores dict (agent_name → score)

        Returns:
            True if entry was written, False if skipped (confidence too low).
        """
        if float(confidence) < self.min_confidence:
            return False

        if direction not in {"LONG", "SHORT"}:
            return False

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "direction": direction,
            "raw_confidence": round(float(confidence), 4),
            "gate_failures": list(gate_failures or []),
            "features": {k: round(float(v), 4) for k, v in (features or {}).items()},
            "agent_scores": {k: round(float(v), 4) for k, v in (agent_scores or {}).items()},
        }

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            # Update in-memory counters
            self._total_logged += 1
            self._pair_counts[pair] += 1
            for failure in gate_failures or []:
                # Extract gate name prefix before '=' or first space
                gate_name = failure.split("=")[0].strip().split(" ")[0].lower()
                self._gate_failure_counts[gate_name] += 1

            logger.debug(
                "VirtualTradeLogger: Logged %s %s conf=%.3f failures=%s",
                pair, direction, confidence, gate_failures,
            )
            return True

        except Exception as e:
            logger.debug("VirtualTradeLogger: Write failed: %s", e)
            return False

    def get_recent(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return the last n virtual trades as dicts.

        Args:
            n: Number of recent records to return.

        Returns:
            List of trade entry dicts, newest last.
        """
        if not self.log_path.exists():
            return []
        try:
            lines = self.log_path.read_text(encoding="utf-8").strip().split("\n")
            valid_lines = [l for l in lines if l.strip()]
            tail = valid_lines[-n:] if len(valid_lines) > n else valid_lines
            return [json.loads(l) for l in tail]
        except Exception as e:
            logger.debug("VirtualTradeLogger: get_recent failed: %s", e)
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics across all logged virtual trades.

        Returns:
            Dict with:
                total_logged: int
                avg_confidence: float
                top_rejected_pairs: list of (pair, count) tuples
                top_gate_failures: list of (gate, count) tuples
        """
        avg_conf = 0.0
        if self._total_logged > 0 and self.log_path.exists():
            try:
                lines = [l for l in self.log_path.read_text().split("\n") if l.strip()]
                confs = []
                for l in lines:
                    try:
                        d = json.loads(l)
                        confs.append(float(d.get("raw_confidence", 0.0)))
                    except Exception:
                        continue
                if confs:
                    avg_conf = round(sum(confs) / len(confs), 4)
            except Exception:
                pass

        return {
            "total_logged": self._total_logged,
            "avg_confidence": avg_conf,
            "top_rejected_pairs": self._pair_counts.most_common(5),
            "top_gate_failures": self._gate_failure_counts.most_common(5),
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _load_counts(self) -> None:
        """Rebuild in-memory counters from existing log file on startup."""
        if not self.log_path.exists():
            return
        try:
            lines = [l for l in self.log_path.read_text(encoding="utf-8").split("\n") if l.strip()]
            for line in lines:
                try:
                    entry = json.loads(line)
                    self._total_logged += 1
                    pair = entry.get("pair", "")
                    if pair:
                        self._pair_counts[pair] += 1
                    for failure in entry.get("gate_failures", []):
                        gate_name = failure.split("=")[0].strip().split(" ")[0].lower()
                        self._gate_failure_counts[gate_name] += 1
                except Exception:
                    continue
            logger.debug(
                "VirtualTradeLogger: Loaded %d existing entries", self._total_logged
            )
        except Exception as e:
            logger.debug("VirtualTradeLogger: Count load failed: %s", e)
