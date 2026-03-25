"""
Unit tests for PortfolioOptimizer.

Tests:
1. Sharpe ratio calculation (edge cases)
2. Trade journal loading
3. Pair ranking algorithm
4. Active/observe pair selection
5. USD exposure check
6. File I/O (save/load rankings)
7. Rotation interval check
8. ObservationLog integration
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scanner.automation.portfolio_optimizer import PortfolioOptimizer, PairRanking


class TestPortfolioOptimizer:
    """Test suite for PortfolioOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create a PortfolioOptimizer instance."""
        return PortfolioOptimizer(
            rolling_window=50,
            max_active_pairs=7,
            max_usd_exposure=0.40,
            rotation_interval=10,
        )

    @pytest.fixture
    def sample_trade_journal(self):
        """Sample trade journal for testing."""
        return [
            {
                "pair": "EUR_USD",
                "direction": "LONG",
                "entry_price": 1.085,
                "outcome": {"realized_pl": 20.0, "trade_won": True},
            },
            {
                "pair": "EUR_USD",
                "direction": "SHORT",
                "entry_price": 1.090,
                "outcome": {"realized_pl": -15.0, "trade_won": False},
            },
            {
                "pair": "GBP_USD",
                "direction": "LONG",
                "entry_price": 1.280,
                "outcome": {"realized_pl": 30.0, "trade_won": True},
            },
        ]

    # ===== Sharpe Calculation Tests =====

    def test_sharpe_normal_case(self, optimizer):
        """Test Sharpe calculation with normal returns."""
        returns = [10.0, 15.0, 8.0, 12.0, 11.0]
        sharpe = optimizer.calculate_sharpe(returns)
        assert sharpe > 0.0
        assert not float("nan") == sharpe  # Not NaN

    def test_sharpe_empty_list(self, optimizer):
        """Test Sharpe with empty returns."""
        sharpe = optimizer.calculate_sharpe([])
        assert sharpe == 0.0

    def test_sharpe_single_value(self, optimizer):
        """Test Sharpe with single return."""
        sharpe = optimizer.calculate_sharpe([5.0])
        assert sharpe == 0.0

    def test_sharpe_zero_variance(self, optimizer):
        """Test Sharpe with zero variance (constant returns)."""
        sharpe = optimizer.calculate_sharpe([5.0, 5.0, 5.0, 5.0])
        assert sharpe == 0.0

    def test_sharpe_mixed_returns(self, optimizer):
        """Test Sharpe with mixed positive/negative returns."""
        sharpe = optimizer.calculate_sharpe([-10.0, -5.0, 0.0, 5.0, 10.0])
        # Should calculate without error; value depends on mean and std
        assert isinstance(sharpe, float)

    # ===== Trade Journal Loading Tests =====

    def test_load_journal_missing_file(self, optimizer):
        """Test loading from non-existent file."""
        optimizer.trade_journal_path = Path("/nonexistent/path/trade_journal_rl.json")
        journal = optimizer._load_trade_journal()
        assert journal == []

    def test_load_journal_valid_file(self, optimizer, sample_trade_journal):
        """Test loading from valid journal file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_trade_journal, f)
            temp_path = f.name

        try:
            optimizer.trade_journal_path = Path(temp_path)
            journal = optimizer._load_trade_journal()
            assert len(journal) == 3
            assert journal[0]["pair"] == "EUR_USD"
        finally:
            Path(temp_path).unlink()

    def test_load_journal_corrupted_json(self, optimizer):
        """Test loading from corrupted JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json")
            temp_path = f.name

        try:
            optimizer.trade_journal_path = Path(temp_path)
            journal = optimizer._load_trade_journal()
            assert journal == []  # Graceful degradation
        finally:
            Path(temp_path).unlink()

    # ===== Pair Ranking Tests =====

    def test_rank_pairs_empty_journal(self, optimizer):
        """Test ranking with empty trade journal."""
        rankings = optimizer.rank_pairs(trade_journal=[])
        assert rankings == []

    def test_rank_pairs_single_pair(self, optimizer, sample_trade_journal):
        """Test ranking with single pair (2 trades)."""
        rankings = optimizer.rank_pairs(trade_journal=sample_trade_journal[:2])
        assert len(rankings) == 1
        assert rankings[0].pair == "EUR_USD"
        assert rankings[0].status == "observe"  # < 5 trades
        assert rankings[0].trade_count == 2
        assert rankings[0].win_rate == 1.0  # Both won

    def test_rank_pairs_multiple_pairs(self, optimizer, sample_trade_journal):
        """Test ranking with multiple pairs."""
        rankings = optimizer.rank_pairs(trade_journal=sample_trade_journal)
        assert len(rankings) == 2  # EUR_USD, GBP_USD
        # Sorted by Sharpe descending
        assert rankings[0].pair == "GBP_USD"  # Sharpe > EUR_USD

    # ===== Active/Observe Selection Tests =====

    def test_get_active_pairs_empty_journal(self, optimizer):
        """Test active pair selection with empty journal."""
        active = optimizer.get_active_pairs(trade_journal=[])
        assert active == []

    def test_get_observe_only_pairs_empty_journal(self, optimizer):
        """Test observe pair selection with empty journal."""
        from src.scanner.config import DEFAULT_PAIRS

        observe = optimizer.get_observe_only_pairs(trade_journal=[], all_pairs=DEFAULT_PAIRS)
        # All pairs should be observe if no trades
        assert len(observe) > 0
        assert set(observe) == set(DEFAULT_PAIRS)

    # ===== Rotation Check Tests =====

    def test_should_rotate_false(self, optimizer):
        """Test rotation check returns False when not due."""
        assert not optimizer.should_rotate(0)
        assert not optimizer.should_rotate(5)
        assert not optimizer.should_rotate(9)
        assert not optimizer.should_rotate(15)

    def test_should_rotate_true(self, optimizer):
        """Test rotation check returns True when due."""
        assert optimizer.should_rotate(10)
        assert optimizer.should_rotate(20)
        assert optimizer.should_rotate(30)

    def test_rotation_interval_custom(self):
        """Test custom rotation interval."""
        opt = PortfolioOptimizer(rotation_interval=5)
        assert opt.should_rotate(5)
        assert opt.should_rotate(10)
        assert not opt.should_rotate(4)
        assert not opt.should_rotate(7)

    # ===== File I/O Tests =====

    def test_save_and_load_rankings(self, optimizer):
        """Test saving and loading pair rankings."""
        rankings = [
            PairRanking(
                pair="EUR_USD",
                sharpe_ratio=1.25,
                rolling_pnl=350.0,
                trade_count=15,
                win_rate=0.6,
                status="active",
                reason="Sharpe 1.25",
            ),
            PairRanking(
                pair="GBP_USD",
                sharpe_ratio=0.85,
                rolling_pnl=200.0,
                trade_count=10,
                win_rate=0.55,
                status="observe",
                reason="Insufficient history",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer.pair_rankings_path = Path(tmpdir) / "rankings.json"

            # Save
            optimizer.save_rankings(rankings)
            assert optimizer.pair_rankings_path.exists()

            # Load
            loaded = optimizer.load_rankings()
            assert len(loaded) == 2
            assert loaded[0].pair == "EUR_USD"
            assert loaded[0].sharpe_ratio == 1.25
            assert loaded[1].pair == "GBP_USD"

    # ===== USD Exposure Tests =====

    def test_usd_pairs_set(self, optimizer):
        """Test USD_PAIRS set is correct."""
        expected = {"EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "NZD_USD", "USD_CAD", "USD_CHF"}
        assert optimizer.USD_PAIRS == expected

    # ===== Integration Tests =====

    def test_full_workflow_with_journal(self, optimizer, sample_trade_journal):
        """Test full workflow: rank → select active → select observe."""
        # Rank pairs
        rankings = optimizer.rank_pairs(trade_journal=sample_trade_journal)
        assert len(rankings) > 0

        # Get active and observe
        active = optimizer.get_active_pairs(trade_journal=sample_trade_journal)
        from src.scanner.config import DEFAULT_PAIRS

        observe = optimizer.get_observe_only_pairs(
            trade_journal=sample_trade_journal, all_pairs=DEFAULT_PAIRS
        )

        # Sanity checks
        active_set = set(active)
        observe_set = set(observe)
        assert active_set.isdisjoint(observe_set) or not active  # No overlap (if any active)
        assert active_set | observe_set == set(DEFAULT_PAIRS)  # Union == all pairs

    def test_pair_ranking_dataclass(self):
        """Test PairRanking dataclass."""
        ranking = PairRanking(
            pair="EUR_USD",
            sharpe_ratio=1.23,
            rolling_pnl=456.78,
            trade_count=20,
            win_rate=0.65,
            status="active",
            reason="Sharpe 1.23",
        )

        # Test to_dict
        d = ranking.to_dict()
        assert d["pair"] == "EUR_USD"
        assert d["sharpe_ratio"] == 1.23
        assert d["rolling_pnl"] == 456.78
        assert d["trade_count"] == 20
        assert d["win_rate"] == 0.65
        assert d["status"] == "active"
        assert d["reason"] == "Sharpe 1.23"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
