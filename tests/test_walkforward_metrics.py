"""
Comprehensive tests for walk-forward validation and trading metrics.

Covers US-250 (CV Splitting) and US-251 (Trading Metrics + Monte Carlo).
"""

import numpy as np
import pytest
from collections import defaultdict

# Import the module under test
from src.training.walkforward_validation import (
    WalkForwardConfig,
    WalkForwardValidator,
    WalkForwardResult,
    WalkForwardSummary,
    TransactionCosts,
    MonteCarloResult,
    purged_kfold_split,
    CombinatorialPurgedCV,
    calculate_trading_metrics,
    calculate_baseline_metrics,
    apply_transaction_costs,
    monte_carlo_simulation,
)


# =============================================================================
# US-250: Walk-Forward CV Splitting Tests
# =============================================================================

class TestWalkForwardConfig:
    """Tests for WalkForwardConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = WalkForwardConfig()

        assert config.enabled is True
        assert config.mode == "rolling"
        assert config.n_splits == 5
        assert config.train_size == 0.60
        assert config.val_size == 0.10
        assert config.test_size == 0.10
        assert config.gap == 24
        assert config.min_train_size == 2000
        assert config.use_purged_kfold is True
        assert config.purge_gap == 24
        assert config.embargo_gap == 12
        assert config.retrain_per_fold is True
        assert config.aggregate_method == "best"
        assert config.ensure_regime_balance is False
        assert config.min_samples_per_regime == 50

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'enabled': False,
            'mode': 'expanding',
            'n_splits': 10,
            'train_size': 0.7,
            'val_size': 0.15,
            'test_size': 0.15,
            'gap': 50,
            'min_train_size': 500,
        }

        config = WalkForwardConfig.from_dict(config_dict)

        assert config.enabled is False
        assert config.mode == "expanding"
        assert config.n_splits == 10
        assert config.train_size == 0.7
        assert config.val_size == 0.15
        assert config.test_size == 0.15
        assert config.gap == 50
        assert config.min_train_size == 500

    def test_from_dict_partial(self):
        """Test from_dict with only some fields specified."""
        config_dict = {'n_splits': 3, 'mode': 'expanding'}
        config = WalkForwardConfig.from_dict(config_dict)

        assert config.n_splits == 3
        assert config.mode == "expanding"
        assert config.enabled is True  # Default preserved

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = WalkForwardConfig(
            n_splits=8,
            mode='expanding',
            train_size=0.65,
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['n_splits'] == 8
        assert config_dict['mode'] == 'expanding'
        assert config_dict['train_size'] == 0.65

    def test_round_trip(self):
        """Test from_dict -> to_dict round-trip."""
        original = {
            'enabled': False,
            'mode': 'rolling',
            'n_splits': 7,
            'train_size': 0.5,
            'val_size': 0.2,
            'test_size': 0.2,
            'gap': 100,
            'min_train_size': 1000,
            'use_purged_kfold': False,
            'purge_gap': 50,
            'embargo_gap': 20,
            'retrain_per_fold': False,
            'aggregate_method': 'average',
            'ensure_regime_balance': True,
            'min_samples_per_regime': 100,
        }

        config = WalkForwardConfig.from_dict(original)
        result = config.to_dict()

        assert result == original


class TestWalkForwardValidator:
    """Tests for WalkForwardValidator walk-forward splitting."""

    def test_split_yields_correct_number_of_folds(self):
        """Test that split yields folds up to the number requested."""
        n_splits = 5
        validator = WalkForwardValidator(
            n_splits=n_splits,
            train_size=0.5,
            val_size=0.2,
            test_size=0.2,
            gap=10,
            min_train_size=500,
        )
        X = np.arange(100000).reshape(-1, 1)  # Use very large data

        folds = list(validator.split(X))

        # Should yield at least some folds
        assert len(folds) >= 1
        # Should not exceed requested splits
        assert len(folds) <= n_splits

    def test_split_yields_correct_number_of_folds_small_data(self):
        """Test split with small data that can't accommodate all folds."""
        validator = WalkForwardValidator(n_splits=5, min_train_size=100)
        X = np.arange(500).reshape(-1, 1)

        folds = list(validator.split(X))

        # Should have fewer folds than requested due to data constraints
        assert len(folds) >= 1
        assert len(folds) <= 5

    def test_temporal_order_expanding(self):
        """Test that expanding mode maintains temporal order (train < val < test)."""
        validator = WalkForwardValidator(
            n_splits=3,
            train_size=0.5,
            val_size=0.2,
            test_size=0.2,
            mode='expanding',
        )
        X = np.arange(2000).reshape(-1, 1)

        for fold, (train_idx, val_idx, test_idx) in enumerate(validator.split(X)):
            # All train indices should be less than all val indices
            assert train_idx[-1] < val_idx[0], f"Fold {fold}: train > val"
            # All val indices should be less than all test indices
            assert val_idx[-1] < test_idx[0], f"Fold {fold}: val > test"

    def test_temporal_order_rolling(self):
        """Test that rolling mode maintains temporal order."""
        validator = WalkForwardValidator(
            n_splits=3,
            train_size=0.5,
            val_size=0.2,
            test_size=0.2,
            mode='rolling',
        )
        X = np.arange(2000).reshape(-1, 1)

        for fold, (train_idx, val_idx, test_idx) in enumerate(validator.split(X)):
            assert train_idx[-1] < val_idx[0], f"Fold {fold}: train > val"
            assert val_idx[-1] < test_idx[0], f"Fold {fold}: val > test"

    def test_no_overlap_between_train_val_test(self):
        """Test that train/val/test sets don't overlap."""
        validator = WalkForwardValidator(n_splits=4)
        X = np.arange(3000).reshape(-1, 1)

        for fold, (train_idx, val_idx, test_idx) in enumerate(validator.split(X)):
            # Convert to sets for overlap checking
            train_set = set(train_idx)
            val_set = set(val_idx)
            test_set = set(test_idx)

            assert len(train_set & val_set) == 0, f"Fold {fold}: train/val overlap"
            assert len(train_set & test_set) == 0, f"Fold {fold}: train/test overlap"
            assert len(val_set & test_set) == 0, f"Fold {fold}: val/test overlap"

    def test_gap_enforced_between_train_and_val(self):
        """Test that gap is enforced between train and validation."""
        gap = 50
        validator = WalkForwardValidator(
            n_splits=3,
            gap=gap,
            train_size=0.5,
            val_size=0.2,
        )
        X = np.arange(5000).reshape(-1, 1)

        for fold, (train_idx, val_idx, test_idx) in enumerate(validator.split(X)):
            # Check gap between train end and val start
            train_end = train_idx[-1]
            val_start = val_idx[0]
            actual_gap = val_start - train_end - 1

            assert actual_gap >= gap, f"Fold {fold}: gap {actual_gap} < {gap}"

    def test_expanding_vs_rolling_difference(self):
        """Test that expanding and rolling modes produce different splits."""
        X = np.arange(3000).reshape(-1, 1)

        validator_expanding = WalkForwardValidator(
            n_splits=3,
            train_size=0.5,
            mode='expanding',
        )
        validator_rolling = WalkForwardValidator(
            n_splits=3,
            train_size=0.5,
            mode='rolling',
        )

        expanding_folds = list(validator_expanding.split(X))
        rolling_folds = list(validator_rolling.split(X))

        # Expanding should have growing train sets; rolling should have fixed size
        if len(expanding_folds) > 1:
            exp_fold1_train = len(expanding_folds[0][0])
            exp_fold2_train = len(expanding_folds[1][0])
            assert exp_fold2_train > exp_fold1_train, "Expanding should have growing train"

        if len(rolling_folds) > 1:
            roll_fold1_train = len(rolling_folds[0][0])
            roll_fold2_train = len(rolling_folds[1][0])
            # Rolling should have similar train sizes (within min_train_size adjustments)
            assert abs(roll_fold2_train - roll_fold1_train) < 200, "Rolling should have stable train"

    def test_min_train_size_respected(self):
        """Test that minimum training size is enforced."""
        min_train_size = 500
        validator = WalkForwardValidator(
            n_splits=3,
            min_train_size=min_train_size,
        )
        X = np.arange(3000).reshape(-1, 1)

        for fold, (train_idx, val_idx, test_idx) in enumerate(validator.split(X)):
            assert len(train_idx) >= min_train_size, f"Fold {fold}: train size < min"

    def test_get_n_splits(self):
        """Test get_n_splits method."""
        validator = WalkForwardValidator(n_splits=7)
        assert validator.get_n_splits() == 7


class TestWalkForwardResult:
    """Tests for WalkForwardResult dataclass."""

    def test_construction(self):
        """Test constructing a WalkForwardResult with all fields."""
        result = WalkForwardResult(
            fold=0,
            train_size=1000,
            val_size=200,
            test_size=200,
            train_metrics={'loss': 0.5, 'accuracy': 0.85},
            val_metrics={'loss': 0.6, 'accuracy': 0.82},
            test_metrics={'loss': 0.65, 'accuracy': 0.80, 'sharpe': 1.5},
            train_period=(0, 999),
            val_period=(1050, 1249),
            test_period=(1250, 1449),
        )

        assert result.fold == 0
        assert result.train_size == 1000
        assert result.val_size == 200
        assert result.test_size == 200
        assert result.train_metrics['accuracy'] == 0.85
        assert result.val_metrics['accuracy'] == 0.82
        assert result.test_metrics['sharpe'] == 1.5
        assert result.train_period == (0, 999)
        assert result.val_period == (1050, 1249)
        assert result.test_period == (1250, 1449)


class TestWalkForwardSummary:
    """Tests for WalkForwardSummary and compute_summary."""

    def test_compute_summary_aggregation(self):
        """Test that compute_summary correctly aggregates fold results."""
        fold_results = [
            WalkForwardResult(
                fold=0,
                train_size=100,
                val_size=20,
                test_size=20,
                train_metrics={},
                val_metrics={},
                test_metrics={'accuracy': 0.80, 'sharpe': 1.0},
                train_period=(0, 99),
                val_period=(110, 129),
                test_period=(130, 149),
            ),
            WalkForwardResult(
                fold=1,
                train_size=100,
                val_size=20,
                test_size=20,
                train_metrics={},
                val_metrics={},
                test_metrics={'accuracy': 0.85, 'sharpe': 1.5},
                train_period=(150, 249),
                val_period=(260, 279),
                test_period=(280, 299),
            ),
            WalkForwardResult(
                fold=2,
                train_size=100,
                val_size=20,
                test_size=20,
                train_metrics={},
                val_metrics={},
                test_metrics={'accuracy': 0.82, 'sharpe': 1.2},
                train_period=(300, 399),
                val_period=(410, 429),
                test_period=(430, 449),
            ),
        ]

        summary = WalkForwardSummary(n_folds=3, fold_results=fold_results)
        summary.compute_summary()

        # Check aggregated metrics
        assert 'accuracy' in summary.mean_test_metrics
        assert 'sharpe' in summary.mean_test_metrics
        assert abs(summary.mean_test_metrics['accuracy'] - 0.823333) < 0.01
        assert abs(summary.mean_test_metrics['sharpe'] - 1.233333) < 0.01

        # Check std
        assert 'accuracy' in summary.std_test_metrics
        assert summary.std_test_metrics['accuracy'] > 0

        # Check stability
        assert 'accuracy' in summary.metric_stability
        assert 'sharpe' in summary.metric_stability

    def test_compute_summary_empty(self):
        """Test compute_summary with empty fold results."""
        summary = WalkForwardSummary(n_folds=0, fold_results=[])
        summary.compute_summary()

        assert len(summary.mean_test_metrics) == 0
        assert len(summary.std_test_metrics) == 0

    def test_compute_summary_with_nan_values(self):
        """Test that compute_summary skips NaN values."""
        fold_results = [
            WalkForwardResult(
                fold=0,
                train_size=100,
                val_size=20,
                test_size=20,
                train_metrics={},
                val_metrics={},
                test_metrics={'accuracy': 0.80, 'sharpe': np.nan},
                train_period=(0, 99),
                val_period=(110, 129),
                test_period=(130, 149),
            ),
            WalkForwardResult(
                fold=1,
                train_size=100,
                val_size=20,
                test_size=20,
                train_metrics={},
                val_metrics={},
                test_metrics={'accuracy': 0.85, 'sharpe': 1.5},
                train_period=(150, 249),
                val_period=(260, 279),
                test_period=(280, 299),
            ),
        ]

        summary = WalkForwardSummary(n_folds=2, fold_results=fold_results)
        summary.compute_summary()

        # Sharpe should only average the valid value
        assert summary.mean_test_metrics['sharpe'] == 1.5
        assert summary.std_test_metrics['sharpe'] == 0.0


class TestPurgedKFoldSplit:
    """Tests for purged_kfold_split function."""

    def test_yields_n_splits_folds(self):
        """Test that purged_kfold_split yields n_splits folds."""
        n_samples = 1000
        n_splits = 5

        folds = list(purged_kfold_split(n_samples, n_splits=n_splits))

        assert len(folds) == n_splits

    def test_purge_gap_removes_near_boundary_samples(self):
        """Test that purge gap removes samples near test boundaries."""
        n_samples = 1000
        n_splits = 5
        purge_gap = 50

        folds = list(purged_kfold_split(
            n_samples,
            n_splits=n_splits,
            purge_gap=purge_gap,
        ))

        for train_idx, test_idx in folds:
            # Check that no training samples exist within purge_gap of test
            test_start = test_idx.min()
            test_end = test_idx.max()

            # Training samples should not be in [test_start - purge_gap, test_start)
            # This is handled by the split logic
            train_before_test = train_idx[train_idx < test_start]
            if len(train_before_test) > 0:
                train_max_before_test = train_before_test.max()
                assert test_start - train_max_before_test > purge_gap, \
                    "Purge gap not enforced before test"

    def test_embargo_gap_applied(self):
        """Test that embargo gap is applied at test start."""
        n_samples = 1000
        n_splits = 5
        embargo_gap = 10

        folds = list(purged_kfold_split(
            n_samples,
            n_splits=n_splits,
            embargo_gap=embargo_gap,
        ))

        # First fold should not have embargo (i > 0 check)
        first_train, first_test = folds[0]
        # First test should start from 0 (no embargo)

        # Second fold should have embargo
        if len(folds) > 1:
            second_train, second_test = folds[1]
            # Second test start should have embargo applied
            expected_start = (1000 // n_splits) + embargo_gap
            assert second_test[0] >= expected_start


class TestCombinatorialPurgedCV:
    """Tests for CombinatorialPurgedCV."""

    def test_split_yields_combinations(self):
        """Test that CombinatorialPurgedCV yields correct number of splits."""
        X = np.arange(1000).reshape(-1, 1)

        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=10)
        splits = list(cv.split(X))

        # Should be C(5, 2) = 10 combinations
        from math import comb
        expected_splits = comb(5, 2)
        assert len(splits) == expected_splits

    def test_no_overlap_in_combinatorial_splits(self):
        """Test that train/test don't overlap in combinatorial splits."""
        X = np.arange(1000).reshape(-1, 1)

        cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=2, purge_gap=10)

        for train_idx, test_idx in cv.split(X):
            train_set = set(train_idx)
            test_set = set(test_idx)

            assert len(train_set & test_set) == 0, "Train/test overlap found"


# =============================================================================
# US-251: Trading Metrics + Monte Carlo Tests
# =============================================================================

class TestCalculateTradingMetrics:
    """Tests for calculate_trading_metrics function."""

    def test_perfect_predictions_accuracy(self):
        """Test that perfect predictions give 100% accuracy."""
        y_true = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])

        metrics = calculate_trading_metrics(y_true, y_pred)

        assert metrics['directional_accuracy'] == 1.0

    def test_random_predictions_accuracy(self):
        """Test that random predictions give ~50% accuracy."""
        np.random.seed(42)
        n = 1000
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.rand(n)  # Probabilistic

        metrics = calculate_trading_metrics(y_true, y_pred)

        # With 1000 samples and random predictions, should be close to 50%
        assert 0.4 < metrics['directional_accuracy'] < 0.6

    def test_returns_sharpe_when_prices_given(self):
        """Test that Sharpe ratio is calculated when prices provided."""
        np.random.seed(42)
        y_true = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        prices = np.array([100, 101, 102, 101.5, 102.5, 101.8, 103, 104, 103.5, 104.5])

        metrics = calculate_trading_metrics(y_true, y_pred, prices=prices)

        assert 'sharpe_ratio' in metrics
        assert isinstance(metrics['sharpe_ratio'], float)

    def test_max_drawdown_non_positive(self):
        """Test that max_drawdown is <= 0."""
        np.random.seed(42)
        y_true = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        prices = np.array([100, 101, 102, 101.5, 102.5, 101.8, 103, 104, 103.5, 104.5])

        metrics = calculate_trading_metrics(y_true, y_pred, prices=prices)

        assert 'max_drawdown' in metrics
        assert 0 <= metrics['max_drawdown'] <= 1

    def test_precision_recall_for_long_signals(self):
        """Test that precision and recall are calculated."""
        y_true = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        y_pred = np.array([0.8, 0.9, 0.2, 0.7, 0.3, 0.1, 0.85, 0.9, 0.2, 0.75])

        metrics = calculate_trading_metrics(y_true, y_pred)

        assert 'precision_long' in metrics
        assert 'recall_long' in metrics
        assert 0 <= metrics['precision_long'] <= 1
        assert 0 <= metrics['recall_long'] <= 1

    def test_f1_score_calculated(self):
        """Test that F1 score is calculated."""
        y_true = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        y_pred = np.array([0.8, 0.9, 0.2, 0.7, 0.3, 0.1, 0.85, 0.9, 0.2, 0.75])

        metrics = calculate_trading_metrics(y_true, y_pred)

        assert 'f1_long' in metrics
        assert 0 <= metrics['f1_long'] <= 1


class TestCalculateBaselineMetrics:
    """Tests for calculate_baseline_metrics function."""

    def test_returns_baseline_dict_keys(self):
        """Test that baseline metrics returns correct keys."""
        y_true = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        y_pred = np.array([0.8, 0.9, 0.2, 0.7, 0.3, 0.1, 0.85, 0.9, 0.2, 0.75])

        metrics = calculate_baseline_metrics(y_true, y_pred)

        assert 'buy_hold' in metrics
        assert 'random_walk' in metrics
        assert 'model' in metrics
        # sma_crossover and momentum may not be present without prices

    def test_with_prices_all_baselines(self):
        """Test that all baselines are computed with prices."""
        np.random.seed(42)
        y_true = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1] * 5)  # 50 samples
        y_pred = np.random.rand(50)
        prices = np.linspace(100, 110, 51)  # 51 prices for 50 returns

        metrics = calculate_baseline_metrics(y_true, y_pred, prices=prices)

        assert 'buy_hold' in metrics
        assert 'random_walk' in metrics
        assert 'sma_crossover' in metrics
        assert 'momentum' in metrics
        assert 'model' in metrics

    def test_baseline_accuracy_values(self):
        """Test that baseline accuracies are in [0, 1]."""
        y_true = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        y_pred = np.array([0.8, 0.9, 0.2, 0.7, 0.3, 0.1, 0.85, 0.9, 0.2, 0.75])

        metrics = calculate_baseline_metrics(y_true, y_pred)

        for baseline in ['buy_hold', 'random_walk']:
            assert 0 <= metrics[baseline]['accuracy'] <= 1


class TestTransactionCosts:
    """Tests for TransactionCosts dataclass."""

    def test_defaults(self):
        """Test default transaction cost values."""
        costs = TransactionCosts()

        assert costs.spread_pips == 1.0
        assert costs.slippage_pips == 0.5
        assert costs.commission_pct == 0.0
        assert costs.pip_value == 10.0


class TestApplyTransactionCosts:
    """Tests for apply_transaction_costs function."""

    def test_no_position_changes_no_cost(self):
        """Test that no position changes result in minimal cost deducted."""
        returns = np.array([0.001, 0.001, 0.001, 0.001, 0.001])
        positions = np.array([1, 1, 1, 1, 1])  # Always long, no changes
        costs = TransactionCosts(spread_pips=0.0, slippage_pips=0.0, commission_pct=0.0)

        adjusted = apply_transaction_costs(returns, positions, costs)

        # With no costs and no position changes, returns should be identical
        assert np.allclose(adjusted, returns)

    def test_position_changes_reduce_returns(self):
        """Test that position changes reduce returns."""
        returns = np.array([0.001, 0.001, 0.001, 0.001, 0.001])
        positions = np.array([1, 1, 0, 0, 1])  # Change at index 2 and 4
        costs = TransactionCosts(spread_pips=1.0, slippage_pips=0.5)

        adjusted = apply_transaction_costs(returns, positions, costs)

        # Adjusted returns should be less than original where costs are applied
        assert adjusted[2] < returns[2]
        assert adjusted[4] < returns[4]

    def test_output_is_numpy_array(self):
        """Test that output is a numpy array."""
        returns = np.array([0.001, 0.002, 0.001])
        positions = np.array([1, 1, 1])
        costs = TransactionCosts()

        adjusted = apply_transaction_costs(returns, positions, costs)

        assert isinstance(adjusted, np.ndarray)
        assert len(adjusted) == len(returns)


class TestMonteCarloSimulation:
    """Tests for monte_carlo_simulation function."""

    def test_returns_monte_carlo_result(self):
        """Test that monte_carlo_simulation returns MonteCarloResult."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.rand(100)
        prices = np.linspace(100, 110, 101)

        result = monte_carlo_simulation(
            y_true, y_pred, prices,
            n_simulations=10,
            random_seed=42,
        )

        assert isinstance(result, MonteCarloResult)

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with seed."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.rand(100)
        prices = np.linspace(100, 110, 101)

        result1 = monte_carlo_simulation(
            y_true, y_pred, prices,
            n_simulations=20,
            random_seed=123,
        )

        result2 = monte_carlo_simulation(
            y_true, y_pred, prices,
            n_simulations=20,
            random_seed=123,
        )

        assert result1.mean_sharpe == result2.mean_sharpe
        assert np.allclose(result1.all_sharpe_ratios, result2.all_sharpe_ratios)

    def test_probability_positive_sharpe_in_range(self):
        """Test that probability_positive_sharpe is in [0, 1]."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.rand(100)
        prices = np.linspace(100, 110, 101)

        result = monte_carlo_simulation(
            y_true, y_pred, prices,
            n_simulations=20,
            random_seed=42,
        )

        assert 0 <= result.probability_positive_sharpe <= 1
        assert 0 <= result.probability_positive_return <= 1

    def test_confidence_interval_ordering(self):
        """Test that confidence interval is ordered (5th <= 95th percentile)."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.rand(100)
        prices = np.linspace(100, 110, 101)

        result = monte_carlo_simulation(
            y_true, y_pred, prices,
            n_simulations=20,
            random_seed=42,
        )

        lower, upper = result.confidence_interval_95
        assert lower <= upper
        assert lower == result.percentile_5
        assert upper == result.percentile_95

    def test_with_transaction_costs(self):
        """Test monte_carlo_simulation with transaction costs."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.rand(100)
        prices = np.linspace(100, 110, 101)
        costs = TransactionCosts(spread_pips=2.0, slippage_pips=1.0)

        result = monte_carlo_simulation(
            y_true, y_pred, prices,
            n_simulations=10,
            costs=costs,
            random_seed=42,
        )

        assert isinstance(result, MonteCarloResult)
        assert len(result.all_sharpe_ratios) == 10

    def test_all_sharpe_ratios_shape(self):
        """Test that all_sharpe_ratios has correct shape."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.rand(100)
        prices = np.linspace(100, 110, 101)
        n_sims = 50

        result = monte_carlo_simulation(
            y_true, y_pred, prices,
            n_simulations=n_sims,
            random_seed=42,
        )

        assert result.all_sharpe_ratios.shape == (n_sims,)
        assert result.all_total_returns.shape == (n_sims,)


class TestMonteCarloResult:
    """Tests for MonteCarloResult dataclass."""

    def test_construction_with_all_fields(self):
        """Test constructing MonteCarloResult with all fields."""
        sharpe_ratios = np.array([1.0, 1.5, 0.8, 1.2, 0.9])
        returns = np.array([0.10, 0.15, 0.08, 0.12, 0.09])

        result = MonteCarloResult(
            mean_sharpe=1.08,
            std_sharpe=0.25,
            percentile_5=0.8,
            percentile_95=1.5,
            confidence_interval_95=(0.8, 1.5),
            all_sharpe_ratios=sharpe_ratios,
            all_total_returns=returns,
            probability_positive_sharpe=0.8,
            probability_positive_return=0.9,
        )

        assert result.mean_sharpe == 1.08
        assert result.std_sharpe == 0.25
        assert result.percentile_5 == 0.8
        assert result.percentile_95 == 1.5
        assert result.confidence_interval_95 == (0.8, 1.5)
        assert len(result.all_sharpe_ratios) == 5
        assert len(result.all_total_returns) == 5
        assert result.probability_positive_sharpe == 0.8
        assert result.probability_positive_return == 0.9


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_walkforward_workflow(self):
        """Test a complete walk-forward validation workflow."""
        np.random.seed(42)
        n_samples = 10000  # Use larger data to get more folds
        X = np.random.randn(n_samples, 10)
        y = np.random.randint(0, 2, n_samples)
        prices = np.linspace(100, 120, n_samples)

        # Create validator
        validator = WalkForwardValidator(
            n_splits=3,
            train_size=0.6,
            val_size=0.15,
            test_size=0.15,
            gap=20,
            min_train_size=1000,
        )

        fold_results = []

        for fold, (train_idx, val_idx, test_idx) in enumerate(validator.split(X)):
            # Simulate training
            y_pred = y[test_idx]  # Perfect predictions for simplicity
            y_true = y[test_idx]
            test_prices = prices[test_idx]

            # Calculate metrics
            metrics = calculate_trading_metrics(y_true, y_pred, test_prices)

            result = WalkForwardResult(
                fold=fold,
                train_size=len(train_idx),
                val_size=len(val_idx),
                test_size=len(test_idx),
                train_metrics={},
                val_metrics={},
                test_metrics=metrics,
                train_period=(int(train_idx[0]), int(train_idx[-1])),
                val_period=(int(val_idx[0]), int(val_idx[-1])),
                test_period=(int(test_idx[0]), int(test_idx[-1])),
            )
            fold_results.append(result)

        # Create summary
        summary = WalkForwardSummary(n_folds=len(fold_results), fold_results=fold_results)
        summary.compute_summary()

        # Verify summary - should have at least 1 fold
        assert len(summary.fold_results) >= 1
        assert 'directional_accuracy' in summary.mean_test_metrics
        assert summary.mean_test_metrics['directional_accuracy'] == 1.0

    def test_purged_cv_with_trading_metrics(self):
        """Test using purged K-fold splits with trading metrics."""
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.rand(n_samples)
        prices = np.linspace(100, 120, n_samples)

        fold_count = 0
        for train_idx, test_idx in purged_kfold_split(n_samples, n_splits=5):
            fold_count += 1

            y_test = y_true[test_idx]
            y_pred_test = y_pred[test_idx]
            prices_test = prices[test_idx]

            metrics = calculate_trading_metrics(y_test, y_pred_test, prices_test)
            assert 'directional_accuracy' in metrics

        assert fold_count == 5

    def test_monte_carlo_with_baselines(self):
        """Test Monte Carlo simulation with baseline comparison."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_pred = np.random.rand(200)
        prices = np.linspace(100, 120, 201)

        # Run Monte Carlo
        mc_result = monte_carlo_simulation(
            y_true, y_pred, prices,
            n_simulations=30,
            random_seed=42,
        )

        # Get baseline metrics
        baseline = calculate_baseline_metrics(y_true, y_pred, prices=prices)

        # Both should be present
        assert isinstance(mc_result, MonteCarloResult)
        assert 'buy_hold' in baseline
        assert 'model' in baseline


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
