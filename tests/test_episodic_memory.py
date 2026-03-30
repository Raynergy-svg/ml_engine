"""Unit tests for Episodic Market Memory system.

Tests the recording, querying, and pattern suppression logic of the episodic memory
module, with special attention to edge cases, JSON corruption, and thread safety.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pytest

from src.scanner.automation.episodic_memory import Episode, EpisodicMemory


@pytest.fixture
def tmp_memory_path() -> Path:
    """Create a temporary memory file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "episodic_memory.json"


@pytest.fixture
def memory(tmp_memory_path: Path) -> EpisodicMemory:
    """Create a fresh EpisodicMemory instance with temp storage."""
    return EpisodicMemory(memory_path=tmp_memory_path, max_episodes=500)


class TestRecordSetup:
    """Test recording of new trade setups."""

    def test_record_setup_creates_episode_with_outcome_none(
        self, memory: EpisodicMemory
    ) -> None:
        """Test that record_setup creates episode with outcome=None."""
        episode_id = memory.record_setup(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.42,
            uncertainty_score=0.31,
            atr_normalized=0.0012,
            spread_pips=1.2,
            confidence=0.67,
            weighted_vote_score=0.71,
            rr_ratio=1.45,
        )

        assert episode_id is not None
        assert isinstance(episode_id, str)
        assert len(episode_id) > 0

        # Check that episode was recorded
        with memory._lock:
            assert len(memory._episodes) == 1
            ep = memory._episodes[0]
            assert ep.episode_id == episode_id
            assert ep.pair == "EUR_USD"
            assert ep.direction == "LONG"
            assert ep.outcome is None
            assert ep.pnl_pips is None
            assert ep.closed_at is None

    def test_record_setup_persists_to_disk(
        self, tmp_memory_path: Path
    ) -> None:
        """Test that record_setup persists episode to disk."""
        memory = EpisodicMemory(memory_path=tmp_memory_path)

        episode_id = memory.record_setup(
            pair="GBP_USD",
            direction="SHORT",
            regime="MEAN_REVERSION",
            session="new_york",
            news_risk_score=0.15,
            uncertainty_score=0.25,
            atr_normalized=0.0010,
            spread_pips=1.5,
            confidence=0.58,
            weighted_vote_score=0.65,
            rr_ratio=1.30,
        )

        # Load fresh memory instance from same path
        memory2 = EpisodicMemory(memory_path=tmp_memory_path)
        assert len(memory2._episodes) == 1
        assert memory2._episodes[0].episode_id == episode_id

    def test_record_setup_multiple_episodes(self, memory: EpisodicMemory) -> None:
        """Test recording multiple episodes."""
        id1 = memory.record_setup(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.1,
            uncertainty_score=0.1,
            atr_normalized=0.001,
            spread_pips=1.0,
            confidence=0.6,
            weighted_vote_score=0.7,
            rr_ratio=1.5,
        )

        id2 = memory.record_setup(
            pair="GBP_USD",
            direction="SHORT",
            regime="MEAN_REVERSION",
            session="new_york",
            news_risk_score=0.2,
            uncertainty_score=0.2,
            atr_normalized=0.0012,
            spread_pips=1.2,
            confidence=0.65,
            weighted_vote_score=0.75,
            rr_ratio=1.4,
        )

        assert id1 != id2
        with memory._lock:
            assert len(memory._episodes) == 2


class TestRecordOutcome:
    """Test recording of trade outcomes."""

    def test_record_outcome_updates_episode(
        self, memory: EpisodicMemory
    ) -> None:
        """Test that record_outcome updates episode with outcome and pnl."""
        episode_id = memory.record_setup(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.42,
            uncertainty_score=0.31,
            atr_normalized=0.0012,
            spread_pips=1.2,
            confidence=0.67,
            weighted_vote_score=0.71,
            rr_ratio=1.45,
        )

        # Record outcome
        result = memory.record_outcome(episode_id, outcome="WIN", pnl_pips=15.5)
        assert result is True

        # Verify episode was updated
        with memory._lock:
            ep = memory._episodes[0]
            assert ep.outcome == "WIN"
            assert ep.pnl_pips == 15.5
            assert ep.closed_at is not None

    def test_record_outcome_nonexistent_episode(
        self, memory: EpisodicMemory
    ) -> None:
        """Test that record_outcome returns False for nonexistent episode."""
        result = memory.record_outcome(
            "nonexistent_id", outcome="LOSS", pnl_pips=-10.0
        )
        assert result is False

    def test_record_outcome_persists(self, tmp_memory_path: Path) -> None:
        """Test that outcome is persisted to disk."""
        memory = EpisodicMemory(memory_path=tmp_memory_path)
        episode_id = memory.record_setup(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.42,
            uncertainty_score=0.31,
            atr_normalized=0.0012,
            spread_pips=1.2,
            confidence=0.67,
            weighted_vote_score=0.71,
            rr_ratio=1.45,
        )

        memory.record_outcome(episode_id, outcome="LOSS", pnl_pips=-20.0)

        # Load fresh instance
        memory2 = EpisodicMemory(memory_path=tmp_memory_path)
        assert len(memory2._episodes) == 1
        assert memory2._episodes[0].outcome == "LOSS"
        assert memory2._episodes[0].pnl_pips == -20.0


class TestQuerySimilar:
    """Test querying similar historical episodes."""

    def _create_episode_with_outcome(
        self,
        memory: EpisodicMemory,
        pair: str,
        direction: str,
        regime: str,
        session: str,
        outcome: str,
        pnl_pips: float,
    ) -> str:
        """Helper to create and close an episode."""
        episode_id = memory.record_setup(
            pair=pair,
            direction=direction,
            regime=regime,
            session=session,
            news_risk_score=0.40,
            uncertainty_score=0.30,
            atr_normalized=0.0012,
            spread_pips=1.2,
            confidence=0.67,
            weighted_vote_score=0.71,
            rr_ratio=1.45,
        )
        memory.record_outcome(episode_id, outcome=outcome, pnl_pips=pnl_pips)
        return episode_id

    def test_query_similar_returns_correct_loss_rate(
        self, memory: EpisodicMemory
    ) -> None:
        """Test that query_similar returns correct loss rate."""
        # Create 3 losses, 1 win
        self._create_episode_with_outcome(
            memory,
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            outcome="LOSS",
            pnl_pips=-15.0,
        )
        self._create_episode_with_outcome(
            memory,
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            outcome="LOSS",
            pnl_pips=-10.0,
        )
        self._create_episode_with_outcome(
            memory,
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            outcome="LOSS",
            pnl_pips=-5.0,
        )
        self._create_episode_with_outcome(
            memory,
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            outcome="WIN",
            pnl_pips=20.0,
        )

        result = memory.query_similar(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.40,
            uncertainty_score=0.30,
        )

        assert result["similar_count"] == 4
        assert result["loss_rate"] == 0.75  # 3 of 4
        assert result["win_rate"] == 0.25  # 1 of 4

    def test_query_similar_avg_pnl(self, memory: EpisodicMemory) -> None:
        """Test that query_similar calculates correct average P&L."""
        self._create_episode_with_outcome(
            memory,
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            outcome="WIN",
            pnl_pips=10.0,
        )
        self._create_episode_with_outcome(
            memory,
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            outcome="WIN",
            pnl_pips=20.0,
        )
        self._create_episode_with_outcome(
            memory,
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            outcome="LOSS",
            pnl_pips=-30.0,
        )

        result = memory.query_similar(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.40,
            uncertainty_score=0.30,
        )

        assert result["similar_count"] == 3
        expected_avg = (10.0 + 20.0 - 30.0) / 3
        assert abs(result["avg_pnl_pips"] - expected_avg) < 0.01

    def test_query_similar_no_matches_returns_zero_count(
        self, memory: EpisodicMemory
    ) -> None:
        """Test that query returns zero count when no episodes match."""
        self._create_episode_with_outcome(
            memory,
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            outcome="WIN",
            pnl_pips=15.0,
        )

        # Query for different pair
        result = memory.query_similar(
            pair="GBP_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.40,
            uncertainty_score=0.30,
        )

        assert result["similar_count"] == 0
        assert result["suppression_active"] is False

    def test_query_similar_filters_by_news_risk_tolerance(
        self, memory: EpisodicMemory
    ) -> None:
        """Test that query filters by news_risk_score ±0.20."""
        # Create with news_risk=0.40
        self._create_episode_with_outcome(
            memory,
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            outcome="WIN",
            pnl_pips=15.0,
        )

        # Query with same score — should match
        result = memory.query_similar(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.40,  # Exact match
            uncertainty_score=0.30,
        )
        assert result["similar_count"] == 1

        # Query with +0.20 — should match
        result = memory.query_similar(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.60,  # +0.20
            uncertainty_score=0.30,
        )
        assert result["similar_count"] == 1

        # Query with +0.21 — should NOT match
        result = memory.query_similar(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.61,  # +0.21
            uncertainty_score=0.30,
        )
        assert result["similar_count"] == 0

    def test_query_similar_filters_by_uncertainty_tolerance(
        self, memory: EpisodicMemory
    ) -> None:
        """Test that query filters by uncertainty_score ±0.15."""
        # Create with uncertainty=0.30
        self._create_episode_with_outcome(
            memory,
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            outcome="WIN",
            pnl_pips=15.0,
        )

        # Query with +0.14 — should match (within ±0.15)
        result = memory.query_similar(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.40,
            uncertainty_score=0.44,  # +0.14
        )
        assert result["similar_count"] == 1

        # Query with +0.16 — should NOT match (exceeds ±0.15)
        result = memory.query_similar(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.40,
            uncertainty_score=0.46,  # +0.16
        )
        assert result["similar_count"] == 0

    def test_query_similar_only_completed_trades(
        self, memory: EpisodicMemory
    ) -> None:
        """Test that query only considers episodes with outcomes (completed trades)."""
        # Create one completed trade
        self._create_episode_with_outcome(
            memory,
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            outcome="WIN",
            pnl_pips=15.0,
        )

        # Create one incomplete setup (no outcome recorded)
        memory.record_setup(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.40,
            uncertainty_score=0.30,
            atr_normalized=0.0012,
            spread_pips=1.2,
            confidence=0.67,
            weighted_vote_score=0.71,
            rr_ratio=1.45,
        )

        result = memory.query_similar(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.40,
            uncertainty_score=0.30,
        )

        # Should only count the completed trade
        assert result["similar_count"] == 1


class TestSuppressionLogic:
    """Test pattern suppression signal logic."""

    def _create_episode_with_outcome(
        self,
        memory: EpisodicMemory,
        pair: str,
        direction: str,
        outcome: str,
        pnl_pips: float,
    ) -> str:
        """Helper to create and close an episode."""
        episode_id = memory.record_setup(
            pair=pair,
            direction=direction,
            regime="TRENDING",
            session="london",
            news_risk_score=0.40,
            uncertainty_score=0.30,
            atr_normalized=0.0012,
            spread_pips=1.2,
            confidence=0.67,
            weighted_vote_score=0.71,
            rr_ratio=1.45,
        )
        memory.record_outcome(episode_id, outcome=outcome, pnl_pips=pnl_pips)
        return episode_id

    def test_suppression_active_when_loss_rate_gte_70_and_count_gte_3(
        self, memory: EpisodicMemory
    ) -> None:
        """Test suppression activates when loss_rate >= 0.70 AND similar_count >= 3."""
        # Create 3 losses
        self._create_episode_with_outcome(
            memory, "EUR_USD", "LONG", "LOSS", -10.0
        )
        self._create_episode_with_outcome(
            memory, "EUR_USD", "LONG", "LOSS", -10.0
        )
        self._create_episode_with_outcome(
            memory, "EUR_USD", "LONG", "LOSS", -10.0
        )

        result = memory.query_similar(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.40,
            uncertainty_score=0.30,
        )

        assert result["suppression_active"] is True
        assert result["similar_count"] == 3
        assert result["loss_rate"] == 1.0

    def test_suppression_inactive_when_count_less_than_3(
        self, memory: EpisodicMemory
    ) -> None:
        """Test suppression stays inactive when similar_count < 3."""
        # Create 2 losses (below threshold)
        self._create_episode_with_outcome(
            memory, "EUR_USD", "LONG", "LOSS", -10.0
        )
        self._create_episode_with_outcome(
            memory, "EUR_USD", "LONG", "LOSS", -10.0
        )

        result = memory.query_similar(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.40,
            uncertainty_score=0.30,
        )

        assert result["suppression_active"] is False
        assert result["similar_count"] == 2
        assert result["loss_rate"] == 1.0  # 100%, but count too low

    def test_suppression_inactive_when_loss_rate_less_than_70(
        self, memory: EpisodicMemory
    ) -> None:
        """Test suppression stays inactive when loss_rate < 0.70."""
        # Create 3 trades: 2 wins, 1 loss = 33% loss rate
        self._create_episode_with_outcome(
            memory, "EUR_USD", "LONG", "WIN", 20.0
        )
        self._create_episode_with_outcome(
            memory, "EUR_USD", "LONG", "WIN", 20.0
        )
        self._create_episode_with_outcome(
            memory, "EUR_USD", "LONG", "LOSS", -10.0
        )

        result = memory.query_similar(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.40,
            uncertainty_score=0.30,
        )

        assert result["suppression_active"] is False
        assert result["similar_count"] == 3
        assert abs(result["loss_rate"] - 0.333) < 0.01

    def test_get_suppression_signal_returns_bool(
        self, memory: EpisodicMemory
    ) -> None:
        """Test that get_suppression_signal returns True/False."""
        self._create_episode_with_outcome(
            memory, "EUR_USD", "LONG", "LOSS", -10.0
        )
        self._create_episode_with_outcome(
            memory, "EUR_USD", "LONG", "LOSS", -10.0
        )
        self._create_episode_with_outcome(
            memory, "EUR_USD", "LONG", "LOSS", -10.0
        )

        signal = memory.get_suppression_signal(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.40,
            uncertainty_score=0.30,
        )

        assert isinstance(signal, bool)
        assert signal is True

    def test_get_suppression_signal_fails_open(
        self, memory: EpisodicMemory
    ) -> None:
        """Test that get_suppression_signal fails open (returns False on error)."""
        # Intentionally cause an error by passing invalid types
        signal = memory.get_suppression_signal(
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score="invalid",  # type: ignore
            uncertainty_score=0.30,
        )

        # Should fail gracefully and return False
        assert signal is False


class TestCorruption:
    """Test handling of corrupt JSON and missing files."""

    def test_load_missing_file_starts_empty(self, tmp_memory_path: Path) -> None:
        """Test that loading from missing file starts with empty episodes."""
        memory = EpisodicMemory(memory_path=tmp_memory_path)
        assert len(memory._episodes) == 0

    def test_load_corrupt_json_graceful_fallback(
        self, tmp_memory_path: Path
    ) -> None:
        """Test that corrupt JSON is handled gracefully."""
        # Write invalid JSON
        tmp_memory_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_memory_path.write_text("{ invalid json }")

        # Should log warning but not crash
        memory = EpisodicMemory(memory_path=tmp_memory_path)
        assert len(memory._episodes) == 0

    def test_load_wrong_structure_graceful_fallback(
        self, tmp_memory_path: Path
    ) -> None:
        """Test that wrong JSON structure (dict instead of list) is handled."""
        tmp_memory_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_memory_path.write_text(json.dumps({"wrong": "structure"}))

        # Should log warning but not crash
        memory = EpisodicMemory(memory_path=tmp_memory_path)
        assert len(memory._episodes) == 0


class TestEviction:
    """Test LRU eviction when max_episodes exceeded."""

    def test_eviction_when_max_episodes_exceeded(
        self, tmp_memory_path: Path
    ) -> None:
        """Test that oldest episodes are evicted when max exceeded."""
        memory = EpisodicMemory(memory_path=tmp_memory_path, max_episodes=3)

        # Record 4 episodes (should evict oldest)
        for i in range(4):
            memory.record_setup(
                pair=f"EUR_USD_{i}",
                direction="LONG",
                regime="TRENDING",
                session="london",
                news_risk_score=0.40,
                uncertainty_score=0.30,
                atr_normalized=0.0012,
                spread_pips=1.2,
                confidence=0.67,
                weighted_vote_score=0.71,
                rr_ratio=1.45,
            )

        # Should only have 3 episodes
        with memory._lock:
            assert len(memory._episodes) == 3
            # Oldest (EUR_USD_0) should be gone
            pairs = [ep.pair for ep in memory._episodes]
            assert "EUR_USD_0" not in pairs
            assert "EUR_USD_1" in pairs
            assert "EUR_USD_2" in pairs
            assert "EUR_USD_3" in pairs


class TestEpisodeDataclass:
    """Test Episode dataclass functionality."""

    def test_episode_to_dict(self) -> None:
        """Test Episode.to_dict serialization."""
        ep = Episode(
            episode_id="test-id",
            timestamp="2026-03-27T10:00:00+00:00",
            pair="EUR_USD",
            direction="LONG",
            regime="TRENDING",
            session="london",
            news_risk_score=0.4,
            uncertainty_score=0.3,
            atr_normalized=0.0012,
            spread_pips=1.2,
            confidence=0.67,
            weighted_vote_score=0.71,
            rr_ratio=1.45,
            outcome="WIN",
            pnl_pips=15.5,
            closed_at="2026-03-27T11:00:00+00:00",
        )

        data = ep.to_dict()
        assert isinstance(data, dict)
        assert data["episode_id"] == "test-id"
        assert data["pair"] == "EUR_USD"
        assert data["outcome"] == "WIN"

    def test_episode_from_dict(self) -> None:
        """Test Episode.from_dict deserialization."""
        data = {
            "episode_id": "test-id",
            "timestamp": "2026-03-27T10:00:00+00:00",
            "pair": "EUR_USD",
            "direction": "LONG",
            "regime": "TRENDING",
            "session": "london",
            "news_risk_score": 0.4,
            "uncertainty_score": 0.3,
            "atr_normalized": 0.0012,
            "spread_pips": 1.2,
            "confidence": 0.67,
            "weighted_vote_score": 0.71,
            "rr_ratio": 1.45,
            "outcome": "WIN",
            "pnl_pips": 15.5,
            "closed_at": "2026-03-27T11:00:00+00:00",
        }

        ep = Episode.from_dict(data)
        assert ep.episode_id == "test-id"
        assert ep.pair == "EUR_USD"
        assert ep.outcome == "WIN"
        assert ep.pnl_pips == 15.5
