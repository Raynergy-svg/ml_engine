"""Episodic Market Memory System.

Records trade setups with full market context, queries similar past setups when
evaluating a new signal, and emits a pattern suppression signal when a similar
historical pattern has consistently failed.

This module implements pattern-based market memory that learns from trade outcomes
and gates new trades based on historical performance of similar setups.

File structure: `trained_data/episodic_memory.json`
- Each episode records the full context of a trade setup
- Episodes are completed with outcome/pnl when trades close
- Query engine finds similar past episodes to evaluate pattern risk
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.scanner.automation.safe_json import safe_json_read, safe_json_write

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_MEMORY_PATH = _PROJECT_ROOT / "trained_data" / "episodic_memory.json"


@dataclass
class Episode:
    """Single episodic trade setup record."""

    episode_id: str
    timestamp: str
    pair: str
    direction: str
    regime: str
    session: str
    news_risk_score: float
    uncertainty_score: float
    atr_normalized: float
    spread_pips: float
    confidence: float
    weighted_vote_score: float
    rr_ratio: float
    outcome: Optional[str] = None  # WIN, LOSS, BREAKEVEN
    pnl_pips: Optional[float] = None
    closed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dict for JSON serialization."""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Episode:
        """Create episode from dict."""
        return Episode(**data)


class EpisodicMemory:
    """Records and queries episodic trade patterns with pattern suppression logic.

    Thread-safe storage with atomic file writes. Limits in-memory episode count
    to prevent unbounded growth (LRU eviction of oldest).

    Usage:
        memory = EpisodicMemory()
        episode_id = memory.record_setup(pair="EUR_USD", direction="LONG", ...)
        # ... trade executes ...
        memory.record_outcome(episode_id, outcome="WIN", pnl_pips=15.5)

        # Query before entry
        signal = memory.get_suppression_signal(pair="EUR_USD", direction="LONG", ...)
        if signal:
            logger.warning("Pattern suppression active — skip this trade")
    """

    def __init__(self, memory_path: Optional[Path] = None, max_episodes: int = 500):
        """Initialize episodic memory.

        Args:
            memory_path: Path to episodic_memory.json file. Defaults to
                trained_data/episodic_memory.json
            max_episodes: Maximum episodes to keep in memory (LRU eviction above this)
        """
        self.memory_path = memory_path or _MEMORY_PATH
        self.max_episodes = max_episodes
        self._episodes: List[Episode] = []
        self._lock = threading.Lock()

        # Load existing episodes from disk
        self.load()

    def load(self) -> None:
        """Load episodes from persistent storage.

        Handles missing file and corrupt JSON gracefully (logs warning, starts empty).
        Called automatically on __init__.
        """
        with self._lock:
            self._episodes = []

        try:
            data = safe_json_read(self.memory_path, default=[])
            if not isinstance(data, list):
                logger.warning(
                    "EpisodicMemory: Invalid JSON structure (expected list), starting fresh"
                )
                return

            loaded_count = 0
            for item in data:
                try:
                    if isinstance(item, dict):
                        episode = Episode.from_dict(item)
                        with self._lock:
                            self._episodes.append(episode)
                        loaded_count += 1
                except Exception as e:
                    logger.debug(f"EpisodicMemory: Failed to load episode: {e}")

            logger.info(f"EpisodicMemory: Loaded {loaded_count} episodes from disk")
        except Exception as e:
            logger.warning(f"EpisodicMemory: Failed to load episodes: {e}")

    def save(self) -> None:
        """Persist episodes to disk (atomic write with temp file + rename).

        Called internally after record_setup and record_outcome.
        """
        with self._lock:
            data = [ep.to_dict() for ep in self._episodes]

        success = safe_json_write(self.memory_path, data, indent=2)
        if not success:
            logger.error(f"EpisodicMemory: Failed to persist episodes to {self.memory_path}")

    def record_setup(
        self,
        pair: str,
        direction: str,
        regime: str,
        session: str,
        news_risk_score: float,
        uncertainty_score: float,
        atr_normalized: float,
        spread_pips: float,
        confidence: float,
        weighted_vote_score: float,
        rr_ratio: float,
    ) -> str:
        """Record a new trade setup episode.

        Creates episode with outcome=None (not yet closed). Returns episode_id
        for later outcome recording.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            direction: Trade direction ("LONG" or "SHORT")
            regime: Market regime (e.g., "TRENDING", "MEAN_REVERSION")
            session: Trading session (e.g., "london", "new_york")
            news_risk_score: News risk score (0.0-1.0)
            uncertainty_score: Model uncertainty (0.0-1.0)
            atr_normalized: Normalized ATR value
            spread_pips: Spread in pips
            confidence: Confidence score (0.0-1.0)
            weighted_vote_score: Agent consensus score (0.0-1.0)
            rr_ratio: Risk-reward ratio (TP/SL)

        Returns:
            episode_id: UUID string for later outcome recording
        """
        episode_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        episode = Episode(
            episode_id=episode_id,
            timestamp=timestamp,
            pair=pair,
            direction=direction,
            regime=regime,
            session=session,
            news_risk_score=float(news_risk_score),
            uncertainty_score=float(uncertainty_score),
            atr_normalized=float(atr_normalized),
            spread_pips=float(spread_pips),
            confidence=float(confidence),
            weighted_vote_score=float(weighted_vote_score),
            rr_ratio=float(rr_ratio),
            outcome=None,
            pnl_pips=None,
            closed_at=None,
        )

        with self._lock:
            self._episodes.append(episode)
            # Evict oldest if needed
            while len(self._episodes) > self.max_episodes:
                removed = self._episodes.pop(0)
                logger.debug(
                    f"EpisodicMemory: Evicted oldest episode {removed.episode_id}"
                )

        self.save()
        logger.debug(f"EpisodicMemory: Recorded setup {episode_id} for {pair} {direction}")
        return episode_id

    def record_outcome(
        self, episode_id: str, outcome: str, pnl_pips: float
    ) -> bool:
        """Record trade outcome for a previously recorded setup.

        Finds episode by id, fills in outcome, pnl_pips, closed_at timestamp.
        Returns True if found and updated, False otherwise.

        Args:
            episode_id: UUID returned from record_setup
            outcome: Trade result ("WIN", "LOSS", or "BREAKEVEN")
            pnl_pips: Realized P/L in pips

        Returns:
            True if episode was found and updated, False if not found
        """
        with self._lock:
            for episode in self._episodes:
                if episode.episode_id == episode_id:
                    episode.outcome = outcome
                    episode.pnl_pips = float(pnl_pips)
                    episode.closed_at = datetime.now(timezone.utc).isoformat()

        self.save()

        if any(ep.episode_id == episode_id for ep in self._episodes):
            logger.debug(
                f"EpisodicMemory: Recorded outcome {outcome} ({pnl_pips} pips) for {episode_id}"
            )
            return True
        else:
            logger.warning(f"EpisodicMemory: Episode {episode_id} not found for outcome recording")
            return False

    def query_similar(
        self,
        pair: str,
        direction: str,
        regime: str,
        session: str,
        news_risk_score: float,
        uncertainty_score: float,
    ) -> Dict[str, Any]:
        """Query similar past episodes to assess pattern risk.

        Finds episodes matching exact pair/direction/regime/session, then filters
        to those with similar news_risk_score (±0.20) and uncertainty_score (±0.15).
        Only considers episodes with outcomes already recorded (completed trades).

        Returns a dict with:
        - similar_count: Number of matching episodes
        - loss_rate: Fraction of losses
        - win_rate: Fraction of wins
        - avg_pnl_pips: Average P/L across all matches
        - suppression_active: True if loss_rate >= 0.70 AND similar_count >= 3
        - suppression_reason: Human-readable explanation
        - episodes_sampled: How many completed episodes were checked

        Args:
            pair: Currency pair
            direction: Trade direction
            regime: Market regime
            session: Trading session
            news_risk_score: News risk score to match
            uncertainty_score: Uncertainty score to match

        Returns:
            Dict with query results and suppression signal
        """
        with self._lock:
            episodes_list = list(self._episodes)

        try:
            # Filter exact matches on pair/direction/regime/session with completed outcomes
            similar: List[Episode] = []
            for ep in episodes_list:
                if (
                    ep.pair == pair
                    and ep.direction == direction
                    and ep.regime == regime
                    and ep.session == session
                    and ep.outcome is not None  # Only completed trades
                ):
                    # Check news_risk_score tolerance (±0.20)
                    if abs(ep.news_risk_score - news_risk_score) > 0.20:
                        continue
                    # Check uncertainty_score tolerance (±0.15)
                    if abs(ep.uncertainty_score - uncertainty_score) > 0.15:
                        continue

                    similar.append(ep)

            similar_count = len(similar)
            if similar_count == 0:
                logger.debug(
                    f"EpisodicMemory: No similar episodes found for {pair} {direction}"
                )
                return {
                    "similar_count": 0,
                    "loss_rate": 0.0,
                    "win_rate": 0.0,
                    "avg_pnl_pips": 0.0,
                    "suppression_active": False,
                    "suppression_reason": "No historical data",
                    "episodes_sampled": len(episodes_list),
                }

            # Calculate statistics
            wins = sum(1 for ep in similar if ep.outcome == "WIN")
            losses = sum(1 for ep in similar if ep.outcome == "LOSS")
            breakevens = sum(1 for ep in similar if ep.outcome == "BREAKEVEN")

            win_rate = wins / similar_count if similar_count > 0 else 0.0
            loss_rate = losses / similar_count if similar_count > 0 else 0.0

            total_pnl = sum(ep.pnl_pips or 0.0 for ep in similar if ep.pnl_pips is not None)
            avg_pnl_pips = total_pnl / similar_count if similar_count > 0 else 0.0

            # Suppression logic: loss_rate >= 0.70 AND similar_count >= 3
            suppression_active = loss_rate >= 0.70 and similar_count >= 3

            suppression_reason = (
                f"{losses} of {similar_count} similar setups lost ({loss_rate*100:.0f}%)"
            )

            result = {
                "similar_count": similar_count,
                "loss_rate": loss_rate,
                "win_rate": win_rate,
                "avg_pnl_pips": avg_pnl_pips,
                "suppression_active": suppression_active,
                "suppression_reason": suppression_reason,
                "episodes_sampled": len(episodes_list),
            }

            logger.debug(f"EpisodicMemory query result: {result}")
            return result

        except Exception as e:
            logger.error(f"EpisodicMemory: query_similar failed: {e}", exc_info=True)
            return {
                "similar_count": 0,
                "loss_rate": 0.0,
                "win_rate": 0.0,
                "avg_pnl_pips": 0.0,
                "suppression_active": False,
                "suppression_reason": "Query error (fail open)",
                "episodes_sampled": 0,
            }

    def get_suppression_signal(
        self,
        pair: str,
        direction: str,
        regime: str,
        session: str,
        news_risk_score: float,
        uncertainty_score: float,
    ) -> bool:
        """Get pattern suppression signal for a proposed trade setup.

        Returns True if a similar historical pattern has consistently failed
        (loss_rate >= 0.70 with at least 3 similar episodes).
        Returns False on any error (fail open — do not suppress due to code bugs).

        Args:
            pair: Currency pair
            direction: Trade direction
            regime: Market regime
            session: Trading session
            news_risk_score: News risk score
            uncertainty_score: Uncertainty score

        Returns:
            True if suppression should block this trade, False otherwise
        """
        try:
            result = self.query_similar(
                pair=pair,
                direction=direction,
                regime=regime,
                session=session,
                news_risk_score=news_risk_score,
                uncertainty_score=uncertainty_score,
            )
            return bool(result.get("suppression_active", False))
        except Exception as e:
            logger.warning(f"EpisodicMemory: get_suppression_signal error: {e}")
            return False  # Fail open
