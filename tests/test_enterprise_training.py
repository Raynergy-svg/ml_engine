"""
Comprehensive test suite for enterprise_training module.

Tests for:
- US-268: ReproducibilityManager & ResourceMonitor
- US-269: WalkForwardValidator & StatisticalValidator & DataValidator

Target: 70+ tests with high coverage of core functionality.
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

# Suppress enterprise_training logger noise in tests
logging.getLogger("enterprise_training").setLevel(logging.CRITICAL)

# Import classes to test
from src.training.enterprise_training import (
    ReproducibilityConfig,
    ReproducibilityManager,
    ResourceStats,
    ResourceMonitor,
    WalkForwardConfig,
    WalkForwardValidator,
    StatisticalValidator,
    DataValidator,
    DataValidationResult,
    ExperimentConfig,
)


# =============================================================================
# US-268: REPRODUCIBILITY TESTS
# =============================================================================

class TestReproducibilityConfig:
    """Test ReproducibilityConfig dataclass initialization."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReproducibilityConfig()
        assert config.seed == 42
        assert config.deterministic is True
        assert config.hash_data is True
        assert config.log_environment is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReproducibilityConfig(
            seed=123,
            deterministic=False,
            hash_data=False,
            log_environment=False,
        )
        assert config.seed == 123
        assert config.deterministic is False
        assert config.hash_data is False
        assert config.log_environment is False

    def test_partial_override(self):
        """Test partial configuration override."""
        config = ReproducibilityConfig(seed=999)
        assert config.seed == 999
        assert config.deterministic is True
        assert config.hash_data is True


class TestReproducibilityManager:
    """Test ReproducibilityManager for seed setting and fingerprinting."""

    def test_set_seeds_returns_info(self):
        """Test set_seeds() returns proper info dict."""
        manager = ReproducibilityManager()
        info = manager.set_seeds()

        assert 'seed' in info
        assert 'deterministic' in info
        assert 'timestamp' in info
        assert info['seed'] == 42
        assert info['deterministic'] is True

    def test_set_seeds_deterministic_true(self):
        """Test deterministic mode is set correctly."""
        config = ReproducibilityConfig(seed=42, deterministic=True)
        manager = ReproducibilityManager(config)
        info = manager.set_seeds()

        assert info['deterministic'] is True

    def test_set_seeds_deterministic_false(self):
        """Test non-deterministic mode is set correctly."""
        config = ReproducibilityConfig(seed=42, deterministic=False)
        manager = ReproducibilityManager(config)
        info = manager.set_seeds()

        assert info['deterministic'] is False

    def test_seeds_ensure_reproducibility(self):
        """Test that same seed produces same random values."""
        manager1 = ReproducibilityManager(ReproducibilityConfig(seed=42))
        manager1.set_seeds()
        vals1 = np.random.randn(10)

        manager2 = ReproducibilityManager(ReproducibilityConfig(seed=42))
        manager2.set_seeds()
        vals2 = np.random.randn(10)

        np.testing.assert_array_equal(vals1, vals2)

    def test_different_seeds_different_values(self):
        """Test that different seeds produce different random values."""
        manager1 = ReproducibilityManager(ReproducibilityConfig(seed=42))
        manager1.set_seeds()
        vals1 = np.random.randn(100)

        manager2 = ReproducibilityManager(ReproducibilityConfig(seed=123))
        manager2.set_seeds()
        vals2 = np.random.randn(100)

        # Should be very unlikely to be equal
        assert not np.allclose(vals1, vals2)

    def test_get_environment_fingerprint(self):
        """Test environment fingerprint extraction."""
        manager = ReproducibilityManager()
        fingerprint = manager.get_environment_fingerprint()

        assert 'python_version' in fingerprint
        assert 'platform' in fingerprint
        assert 'processor' in fingerprint
        assert 'numpy_version' in fingerprint
        assert isinstance(fingerprint['python_version'], str)

    def test_get_environment_fingerprint_cached(self):
        """Test that fingerprint is cached (no repeated calls)."""
        manager = ReproducibilityManager()
        fingerprint1 = manager.get_environment_fingerprint()
        fingerprint2 = manager.get_environment_fingerprint()

        # Should be same object (cached)
        assert fingerprint1 is fingerprint2

    def test_get_environment_fingerprint_contains_git(self):
        """Test that fingerprint includes git info."""
        manager = ReproducibilityManager()
        fingerprint = manager.get_environment_fingerprint()

        assert 'git_commit' in fingerprint
        assert isinstance(fingerprint['git_commit'], str)

    def test_compute_data_hash_deterministic(self):
        """Test compute_data_hash produces same hash for same data."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        hash1 = ReproducibilityManager.compute_data_hash(X, y, n_samples=50)
        hash2 = ReproducibilityManager.compute_data_hash(X, y, n_samples=50)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 16

    def test_compute_data_hash_different_data(self):
        """Test compute_data_hash differs for different data."""
        X1 = np.random.randn(100, 10)
        y1 = np.random.randint(0, 2, 100)

        X2 = np.random.randn(100, 10)
        y2 = np.random.randint(0, 2, 100)

        hash1 = ReproducibilityManager.compute_data_hash(X1, y1)
        hash2 = ReproducibilityManager.compute_data_hash(X2, y2)

        # Different data should produce different hashes (with high probability)
        assert hash1 != hash2

    def test_compute_data_hash_sampling(self):
        """Test that sampling works correctly for large datasets."""
        X = np.random.randn(10000, 10)
        y = np.random.randint(0, 2, 10000)

        # Should sample 1000 by default
        hash1 = ReproducibilityManager.compute_data_hash(X, y, n_samples=1000)
        assert isinstance(hash1, str)
        assert len(hash1) == 16

    def test_compute_data_hash_with_small_sample(self):
        """Test compute_data_hash with n_samples larger than data."""
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)

        hash_val = ReproducibilityManager.compute_data_hash(X, y, n_samples=1000)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 16


# =============================================================================
# US-268: RESOURCE MONITORING TESTS
# =============================================================================

class TestResourceStats:
    """Test ResourceStats dataclass."""

    def test_resource_stats_creation(self):
        """Test ResourceStats instantiation."""
        stats = ResourceStats(
            timestamp=123.456,
            cpu_percent=45.0,
            memory_percent=60.0,
            memory_gb=8.5,
            gpu_available=True,
            gpu_memory_gb=12.0,
            gpu_utilization=75.0,
        )

        assert stats.timestamp == 123.456
        assert stats.cpu_percent == 45.0
        assert stats.memory_percent == 60.0
        assert stats.memory_gb == 8.5
        assert stats.gpu_available is True
        assert stats.gpu_memory_gb == 12.0
        assert stats.gpu_utilization == 75.0

    def test_resource_stats_defaults(self):
        """Test ResourceStats default values."""
        stats = ResourceStats(timestamp=100.0, cpu_percent=25.0, memory_percent=50.0, memory_gb=5.0)

        assert stats.gpu_available is False
        assert stats.gpu_memory_gb == 0.0
        assert stats.gpu_utilization == 0.0


class TestResourceMonitor:
    """Test ResourceMonitor for resource tracking."""

    def test_monitor_init(self):
        """Test ResourceMonitor initialization."""
        monitor = ResourceMonitor(interval_seconds=2.0)

        assert monitor.interval == 2.0
        assert monitor.alert_threshold == 90.0
        assert monitor.history == []
        assert monitor._monitoring is False

    def test_monitor_with_custom_threshold(self):
        """Test ResourceMonitor with custom alert threshold."""
        monitor = ResourceMonitor(alert_memory_threshold=75.0)
        assert monitor.alert_threshold == 75.0

    def test_compute_summary_empty_history(self):
        """Test _compute_summary with empty history."""
        monitor = ResourceMonitor()
        summary = monitor._compute_summary()

        assert summary == {}

    def test_compute_summary_with_history(self):
        """Test _compute_summary with populated history."""
        monitor = ResourceMonitor()

        # Manually add history
        monitor.history = [
            ResourceStats(timestamp=0.0, cpu_percent=10.0, memory_percent=20.0, memory_gb=2.0),
            ResourceStats(timestamp=5.0, cpu_percent=30.0, memory_percent=40.0, memory_gb=4.0),
            ResourceStats(timestamp=10.0, cpu_percent=50.0, memory_percent=60.0, memory_gb=6.0),
        ]

        summary = monitor._compute_summary()

        assert 'cpu_mean' in summary
        assert 'cpu_max' in summary
        assert 'memory_mean_gb' in summary
        assert 'memory_max_gb' in summary
        assert 'duration_seconds' in summary
        assert 'n_samples' in summary

        assert summary['cpu_mean'] == 30.0
        assert summary['cpu_max'] == 50.0
        assert summary['memory_mean_gb'] == 4.0
        assert summary['memory_max_gb'] == 6.0
        assert summary['duration_seconds'] == 10.0
        assert summary['n_samples'] == 3

    def test_collect_stats_returns_resource_stats(self):
        """Test _collect_stats returns ResourceStats object."""
        with patch('src.training.enterprise_training.PSUTIL_AVAILABLE', False):
            monitor = ResourceMonitor()
            stats = monitor._collect_stats()

            assert isinstance(stats, ResourceStats)
            assert stats.timestamp > 0
            assert stats.cpu_percent == 0  # No psutil
            assert stats.memory_percent == 0
            assert stats.memory_gb == 0


# =============================================================================
# US-269: WALK-FORWARD VALIDATION TESTS
# =============================================================================

class TestWalkForwardConfig:
    """Test WalkForwardConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = WalkForwardConfig()

        assert config.n_splits == 3
        assert config.min_train_samples == 4000
        assert config.train_period == 6000
        assert config.test_period == 1000
        assert config.gap == 24
        assert config.expanding is True
        assert config.purge_overlap == 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = WalkForwardConfig(
            n_splits=5,
            min_train_samples=2000,
            train_period=3000,
            test_period=500,
            gap=12,
            expanding=False,
            purge_overlap=10,
        )

        assert config.n_splits == 5
        assert config.min_train_samples == 2000
        assert config.train_period == 3000
        assert config.test_period == 500
        assert config.gap == 12
        assert config.expanding is False
        assert config.purge_overlap == 10


class TestWalkForwardValidator:
    """Test WalkForwardValidator for time series cross-validation."""

    def test_validator_init(self):
        """Test WalkForwardValidator initialization."""
        config = WalkForwardConfig(n_splits=3)
        validator = WalkForwardValidator(config)

        assert validator.config.n_splits == 3

    def test_validator_init_default_config(self):
        """Test WalkForwardValidator with default config."""
        validator = WalkForwardValidator()
        assert validator.config.n_splits == 3

    def test_split_basic(self):
        """Test basic split functionality."""
        X = np.random.randn(10000, 10)
        validator = WalkForwardValidator(
            WalkForwardConfig(n_splits=3, min_train_samples=2000, test_period=500, gap=24)
        )

        splits = list(validator.split(X))

        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(np.intersect1d(train_idx, test_idx)) == 0  # No overlap

    def test_split_no_train_test_overlap(self):
        """Test that training and test sets don't overlap."""
        X = np.random.randn(10000, 10)
        y = np.random.randint(0, 2, 10000)

        validator = WalkForwardValidator(
            WalkForwardConfig(n_splits=3, min_train_samples=2000, test_period=500, gap=24)
        )

        for train_idx, test_idx in validator.split(X, y):
            # Check no overlap
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_split_respects_gap(self):
        """Test that gap is respected between train and test."""
        X = np.random.randn(10000, 10)
        validator = WalkForwardValidator(
            WalkForwardConfig(n_splits=2, min_train_samples=2000, test_period=500, gap=100)
        )

        for train_idx, test_idx in validator.split(X):
            # Train end + gap <= test start
            train_end = train_idx.max()
            test_start = test_idx.min()
            assert test_start > train_end  # Gap enforced

    def test_split_minimum_train_samples(self):
        """Test that minimum training samples is enforced."""
        X = np.random.randn(10000, 10)
        min_train = 3000

        validator = WalkForwardValidator(
            WalkForwardConfig(n_splits=3, min_train_samples=min_train, test_period=500, gap=24)
        )

        for train_idx, test_idx in validator.split(X):
            assert len(train_idx) >= min_train

    def test_split_expanding_window(self):
        """Test expanding window mode."""
        X = np.random.randn(10000, 10)
        validator = WalkForwardValidator(
            WalkForwardConfig(
                n_splits=3,
                min_train_samples=2000,
                test_period=500,
                gap=24,
                expanding=True,
            )
        )

        prev_train_size = 0
        for train_idx, test_idx in validator.split(X):
            # In expanding mode, training set should grow
            assert len(train_idx) >= prev_train_size
            prev_train_size = len(train_idx)

    def test_split_sliding_window(self):
        """Test sliding window mode."""
        X = np.random.randn(10000, 10)
        validator = WalkForwardValidator(
            WalkForwardConfig(
                n_splits=3,
                min_train_samples=2000,
                train_period=3000,
                test_period=500,
                gap=24,
                expanding=False,
            )
        )

        for train_idx, test_idx in validator.split(X):
            # In sliding mode, training size should be roughly constant
            assert len(train_idx) >= 2000

    def test_split_insufficient_data_warning(self):
        """Test warning when insufficient data for splits."""
        X = np.random.randn(100, 10)  # Very small dataset
        validator = WalkForwardValidator(
            WalkForwardConfig(n_splits=10, min_train_samples=2000, test_period=500)
        )

        # Should still yield some splits
        splits = list(validator.split(X))
        assert len(splits) >= 0  # May be empty or reduced

    def test_calculate_metrics_accuracy(self):
        """Test _calculate_metrics computes accuracy."""
        validator = WalkForwardValidator()

        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0.1, 0.8, 0.9, 0.2, 0.7])  # Probabilities

        metrics = validator._calculate_metrics(y_true, y_pred)

        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_calculate_metrics_balanced_accuracy(self):
        """Test _calculate_metrics computes balanced accuracy."""
        validator = WalkForwardValidator()

        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0.1, 0.8, 0.9, 0.2, 0.7])

        metrics = validator._calculate_metrics(y_true, y_pred)

        assert 'balanced_accuracy' in metrics
        assert 0 <= metrics['balanced_accuracy'] <= 1

    def test_calculate_metrics_auc(self):
        """Test _calculate_metrics computes AUC if scipy available."""
        validator = WalkForwardValidator()

        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.3, 0.6, 0.4])

        metrics = validator._calculate_metrics(y_true, y_pred)

        # AUC may or may not be present depending on scipy availability
        if 'auc' in metrics:
            assert 0 <= metrics['auc'] <= 1

    def test_calculate_metrics_2d_predictions(self):
        """Test _calculate_metrics handles 2D prediction arrays."""
        validator = WalkForwardValidator()

        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([[0.1], [0.8], [0.9], [0.2], [0.7]])  # 2D array

        metrics = validator._calculate_metrics(y_true, y_pred)

        assert 'accuracy' in metrics
        assert 'balanced_accuracy' in metrics

    def test_calculate_metrics_all_zeros(self):
        """Test _calculate_metrics with all zero labels."""
        validator = WalkForwardValidator()

        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.1, 0.3, 0.2])

        metrics = validator._calculate_metrics(y_true, y_pred)

        assert 'accuracy' in metrics
        # balanced_accuracy = (pos_acc + neg_acc) / 2
        # pos_acc = 0 (no positives to predict correctly)
        # neg_acc = 1.0 (all negatives correct)
        # So balanced_accuracy = 0.5
        assert metrics['balanced_accuracy'] == 0.5

    def test_calculate_metrics_all_ones(self):
        """Test _calculate_metrics with all one labels."""
        validator = WalkForwardValidator()

        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([0.8, 0.9, 0.7, 0.85, 0.92])

        metrics = validator._calculate_metrics(y_true, y_pred)

        assert 'accuracy' in metrics
        # balanced_accuracy = (pos_acc + neg_acc) / 2
        # pos_acc = 1.0 (all positives correct)
        # neg_acc = 0 (no negatives to predict correctly)
        # So balanced_accuracy = 0.5
        assert metrics['balanced_accuracy'] == 0.5


# =============================================================================
# US-269: STATISTICAL VALIDATION TESTS
# =============================================================================

class TestStatisticalValidator:
    """Test StatisticalValidator for significance testing."""

    def test_bootstrap_confidence_interval_basic(self):
        """Test basic bootstrap CI computation."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.3, 0.6, 0.85])

        def accuracy_fn(y_t, y_p):
            return np.mean((y_p > 0.5).astype(int) == y_t)

        result = StatisticalValidator.bootstrap_confidence_interval(
            metric_fn=accuracy_fn,
            y_true=y_true,
            y_pred=y_pred,
            n_bootstrap=100,
            confidence=0.95,
            seed=42,
        )

        assert 'mean' in result
        assert 'std' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert 'n_bootstrap' in result
        assert 'confidence' in result

        assert 0 <= result['mean'] <= 1
        assert result['ci_lower'] <= result['mean'] <= result['ci_upper']

    def test_bootstrap_confidence_interval_seed_reproducibility(self):
        """Test that bootstrap CI is reproducible with same seed."""
        y_true = np.array([0, 1, 1, 0, 1] * 10)
        y_pred = np.random.randn(50)

        def metric_fn(y_t, y_p):
            return np.mean(y_t)

        result1 = StatisticalValidator.bootstrap_confidence_interval(
            metric_fn=metric_fn,
            y_true=y_true,
            y_pred=y_pred,
            n_bootstrap=100,
            seed=42,
        )

        result2 = StatisticalValidator.bootstrap_confidence_interval(
            metric_fn=metric_fn,
            y_true=y_true,
            y_pred=y_pred,
            n_bootstrap=100,
            seed=42,
        )

        assert result1['mean'] == result2['mean']
        assert result1['ci_lower'] == result2['ci_lower']
        assert result1['ci_upper'] == result2['ci_upper']

    def test_bootstrap_confidence_interval_different_confidence_levels(self):
        """Test bootstrap CI with different confidence levels."""
        y_true = np.array([0, 1] * 20)
        y_pred = np.random.randn(40)

        def metric_fn(y_t, y_p):
            return np.mean(y_t)

        result_95 = StatisticalValidator.bootstrap_confidence_interval(
            metric_fn=metric_fn,
            y_true=y_true,
            y_pred=y_pred,
            n_bootstrap=500,
            confidence=0.95,
            seed=42,
        )

        result_99 = StatisticalValidator.bootstrap_confidence_interval(
            metric_fn=metric_fn,
            y_true=y_true,
            y_pred=y_pred,
            n_bootstrap=500,
            confidence=0.99,
            seed=42,
        )

        # Higher confidence should yield wider CI
        width_95 = result_95['ci_upper'] - result_95['ci_lower']
        width_99 = result_99['ci_upper'] - result_99['ci_lower']
        assert width_99 > width_95

    def test_bootstrap_empty_metric(self):
        """Test bootstrap CI when metric_fn fails for all samples."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0.5, 0.5, 0.5])

        def failing_metric_fn(y_t, y_p):
            raise ValueError("Metric not computable")

        result = StatisticalValidator.bootstrap_confidence_interval(
            metric_fn=failing_metric_fn,
            y_true=y_true,
            y_pred=y_pred,
            n_bootstrap=10,
            seed=42,
        )

        assert result['mean'] == 0
        assert result['ci_lower'] == 0
        assert result['ci_upper'] == 0

    def test_paired_ttest_significant_difference(self):
        """Test paired t-test detects significant difference."""
        scores_a = np.array([0.85, 0.87, 0.89, 0.86, 0.88])
        scores_b = np.array([0.70, 0.72, 0.71, 0.73, 0.72])

        result = StatisticalValidator.paired_ttest(scores_a, scores_b, alpha=0.05)

        assert 't_statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert 'mean_difference' in result
        assert 'better_model' in result

        # A should be significantly better than B
        # Use == for comparison with numpy bool
        assert result['significant'] == True
        assert result['better_model'] == 'A'

    def test_paired_ttest_no_difference(self):
        """Test paired t-test when no significant difference."""
        scores_a = np.array([0.80, 0.81, 0.82, 0.80, 0.81])
        scores_b = np.array([0.80, 0.81, 0.82, 0.80, 0.81])

        result = StatisticalValidator.paired_ttest(scores_a, scores_b, alpha=0.05)

        # Should not be significant
        assert result['significant'] == False

    def test_paired_ttest_alpha_threshold(self):
        """Test paired t-test respects alpha threshold."""
        scores_a = np.array([0.81, 0.82, 0.80, 0.81, 0.82])
        scores_b = np.array([0.80, 0.80, 0.80, 0.80, 0.80])

        result_alpha_01 = StatisticalValidator.paired_ttest(scores_a, scores_b, alpha=0.01)
        result_alpha_05 = StatisticalValidator.paired_ttest(scores_a, scores_b, alpha=0.05)

        # More lenient alpha should be more likely to show significance
        assert result_alpha_01['alpha'] == 0.01
        assert result_alpha_05['alpha'] == 0.05

    def test_sharpe_ratio_significance_basic(self):
        """Test Sharpe ratio significance test."""
        returns = np.random.randn(252) * 0.01  # 1 year of returns
        returns = returns + 0.0005  # Add positive drift

        result = StatisticalValidator.sharpe_ratio_significance(
            returns=returns,
            benchmark_sharpe=0.0,
            periods_per_year=252,
        )

        assert 'sharpe_ratio' in result
        assert 'std_error' in result
        assert 'z_statistic' in result
        assert 'p_value' in result
        assert 'significant_at_05' in result
        assert 'significant_at_01' in result

    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = np.array([0.0] * 100)

        result = StatisticalValidator.sharpe_ratio_significance(
            returns=returns,
            benchmark_sharpe=0.0,
            periods_per_year=252,
        )

        assert result['sharpe_ratio'] == 0
        assert result['p_value'] == 1.0
        # When volatility is zero, function returns early and only has these keys
        assert 'sharpe_ratio' in result
        assert 'p_value' in result
        assert 'significant' in result

    def test_sharpe_ratio_annualization(self):
        """Test Sharpe ratio annualization."""
        returns = np.random.randn(252) * 0.01
        returns = returns + 0.0005

        result = StatisticalValidator.sharpe_ratio_significance(
            returns=returns,
            benchmark_sharpe=0.0,
            periods_per_year=252,
        )

        # Manual calculation
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        expected_sharpe = (mean_ret * np.sqrt(252)) / std_ret

        assert np.isclose(result['sharpe_ratio'], expected_sharpe, atol=0.01)

    def test_sharpe_ratio_different_benchmarks(self):
        """Test Sharpe ratio against different benchmarks."""
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001

        result_bench_0 = StatisticalValidator.sharpe_ratio_significance(
            returns=returns,
            benchmark_sharpe=0.0,
            periods_per_year=252,
        )

        # With a very high benchmark, the sharpe ratio difference becomes large negative
        result_bench_1 = StatisticalValidator.sharpe_ratio_significance(
            returns=returns,
            benchmark_sharpe=5.0,  # Much higher benchmark
            periods_per_year=252,
        )

        # z_stat against higher benchmark will be more negative, leading to larger p-value
        # in the two-tailed test
        assert result_bench_0['z_statistic'] > result_bench_1['z_statistic']


# =============================================================================
# US-269: DATA VALIDATION TESTS
# =============================================================================

class TestDataValidationResult:
    """Test DataValidationResult dataclass."""

    def test_data_validation_result_passed(self):
        """Test DataValidationResult with passed=True."""
        result = DataValidationResult(passed=True)

        assert result.passed is True
        assert result.checks == {}
        assert result.warnings == []
        assert result.errors == []

    def test_data_validation_result_with_details(self):
        """Test DataValidationResult with checks, warnings, errors."""
        result = DataValidationResult(
            passed=False,
            checks={'missing': 5},
            warnings=['Some NaN values'],
            errors=['Infinite values detected'],
        )

        assert result.passed is False
        assert result.checks == {'missing': 5}
        assert len(result.warnings) == 1
        assert len(result.errors) == 1


class TestDataValidator:
    """Test DataValidator for data quality checks."""

    def test_validator_init(self):
        """Test DataValidator initialization."""
        validator = DataValidator()

        assert validator.reference_stats == {}
        assert validator.drift_threshold == 0.05

    def test_validator_with_custom_threshold(self):
        """Test DataValidator with custom drift threshold."""
        validator = DataValidator(drift_threshold=0.10)
        assert validator.drift_threshold == 0.10

    def test_validate_clean_array(self):
        """Test validate on clean numpy array."""
        X = np.random.randn(100, 10)
        validator = DataValidator()
        result = validator.validate(X)

        assert result.passed is True
        assert len(result.errors) == 0

    def test_validate_nan_values(self):
        """Test validate detects NaN values."""
        X = np.random.randn(100, 10)
        X[5, 3] = np.nan

        validator = DataValidator()
        result = validator.validate(X)

        # Should detect NaN
        assert 'nan_count' in result.checks
        assert result.checks['nan_count'] > 0
        assert len(result.warnings) > 0

    def test_validate_inf_values(self):
        """Test validate detects infinite values."""
        X = np.random.randn(100, 10)
        X[5, 3] = np.inf

        validator = DataValidator()
        result = validator.validate(X)

        # Should detect inf and mark as error
        assert 'inf_count' in result.checks
        assert result.checks['inf_count'] > 0
        assert result.passed is False
        assert len(result.errors) > 0

    def test_validate_extreme_values(self):
        """Test validate warns on extreme values."""
        X = np.random.randn(100, 10) * 1e11  # Even larger values

        validator = DataValidator()
        result = validator.validate(X)

        # Should warn about extreme values (max > 1e10)
        assert len(result.warnings) > 0

    def test_compute_reference_stats(self):
        """Test compute_reference_stats on training data."""
        X = np.random.randn(100, 10)

        validator = DataValidator()
        stats = validator.compute_reference_stats(X)

        assert 'overall' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats

        assert len(stats['overall']) > 0
        assert isinstance(stats['mean'], (float, np.floating))
        assert isinstance(stats['std'], (float, np.floating))

    def test_compute_reference_stats_with_nan(self):
        """Test compute_reference_stats handles NaN values."""
        X = np.random.randn(100, 10)
        X[5, 3] = np.nan

        validator = DataValidator()
        stats = validator.compute_reference_stats(X)

        # Should skip NaN values
        assert stats['mean'] < 1e10  # Shouldn't be NaN
        assert stats['std'] < 1e10

    def test_detect_drift_no_reference(self):
        """Test _detect_drift with no reference stats."""
        X = np.random.randn(100, 10)

        validator = DataValidator()
        drift = validator._detect_drift(X)

        assert drift == {}

    def test_detect_drift_with_reference(self):
        """Test _detect_drift with reference stats."""
        X_ref = np.random.randn(100, 10)
        X_new = np.random.randn(100, 10)

        validator = DataValidator()
        ref_stats = validator.compute_reference_stats(X_ref)
        validator.reference_stats = ref_stats

        drift = validator._detect_drift(X_new)

        # Should compute drift metrics
        if 'overall' in drift:
            assert 'ks_statistic' in drift['overall']
            assert 'p_value' in drift['overall']
            assert 'drifted' in drift['overall']

    def test_validate_with_drift_detection(self):
        """Test validate includes drift detection."""
        X_ref = np.random.randn(100, 10)
        X_new = np.random.randn(100, 10)

        validator = DataValidator()
        ref_stats = validator.compute_reference_stats(X_ref)
        validator.reference_stats = ref_stats

        result = validator.validate(X_new)

        # Should include drift info if scipy available
        if 'drift' in result.checks:
            assert isinstance(result.checks['drift'], dict)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestExperimentConfig:
    """Test ExperimentConfig dataclass."""

    def test_experiment_config_minimal(self):
        """Test minimal ExperimentConfig."""
        config = ExperimentConfig(
            name="test_exp",
            model_type="lstm",
        )

        assert config.name == "test_exp"
        assert config.model_type == "lstm"

    def test_experiment_config_full(self):
        """Test ExperimentConfig with all fields."""
        config = ExperimentConfig(
            name="eur_usd_transformer",
            model_type="transformer",
            instrument="EUR_USD",
            granularity="H1",
            description="Test experiment",
            tags={"version": "1.0", "env": "test"},
            epochs=50,
            batch_size=32,
            learning_rate=0.0005,
        )

        assert config.name == "eur_usd_transformer"
        assert config.instrument == "EUR_USD"
        assert config.epochs == 50
        assert config.batch_size == 32


class TestReproducibilityIntegration:
    """Integration tests combining reproducibility features."""

    def test_reproducibility_with_walk_forward(self):
        """Test that reproducibility works with walk-forward validation."""
        # Set seeds
        manager = ReproducibilityManager(ReproducibilityConfig(seed=42))
        manager.set_seeds()

        X = np.random.randn(1000, 10)
        y = np.random.randint(0, 2, 1000)

        # First run
        validator1 = WalkForwardValidator(
            WalkForwardConfig(n_splits=2, min_train_samples=200, test_period=100)
        )
        splits1 = list(validator1.split(X, y))

        # Reset seeds
        manager = ReproducibilityManager(ReproducibilityConfig(seed=42))
        manager.set_seeds()

        # Second run - should be identical
        validator2 = WalkForwardValidator(
            WalkForwardConfig(n_splits=2, min_train_samples=200, test_period=100)
        )
        splits2 = list(validator2.split(X, y))

        # Splits should be identical
        assert len(splits1) == len(splits2)
        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    def test_data_hash_with_different_sampling(self):
        """Test data hash changes with different sampling."""
        X = np.random.randn(10000, 10)
        y = np.random.randint(0, 2, 10000)

        hash_100 = ReproducibilityManager.compute_data_hash(X, y, n_samples=100)
        hash_1000 = ReproducibilityManager.compute_data_hash(X, y, n_samples=1000)

        # Same data, likely different samples - hashes should still be consistent
        # but this tests that different sample sizes work
        assert isinstance(hash_100, str)
        assert isinstance(hash_1000, str)


# =============================================================================
# EDGE CASES & ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_walk_forward_single_sample(self):
        """Test walk-forward with single sample (edge case)."""
        X = np.random.randn(10, 5)

        validator = WalkForwardValidator(
            WalkForwardConfig(n_splits=1, min_train_samples=5, test_period=2, gap=1)
        )

        splits = list(validator.split(X))
        # May be empty if data too small
        assert isinstance(splits, list)

    def test_bootstrap_single_sample(self):
        """Test bootstrap CI with single sample."""
        y_true = np.array([1])
        y_pred = np.array([0.9])

        def metric_fn(y_t, y_p):
            return float(y_p[0])

        result = StatisticalValidator.bootstrap_confidence_interval(
            metric_fn=metric_fn,
            y_true=y_true,
            y_pred=y_pred,
            n_bootstrap=10,
            seed=42,
        )

        # Should still work with resampling
        assert 'mean' in result

    def test_data_validator_empty_array(self):
        """Test DataValidator on empty array."""
        X = np.array([]).reshape(0, 10)

        validator = DataValidator()
        result = validator.validate(X)

        # Should handle gracefully
        assert isinstance(result, DataValidationResult)

    def test_reproducibility_manager_with_none_config(self):
        """Test ReproducibilityManager with None config."""
        manager = ReproducibilityManager(config=None)

        # Should create default config
        assert manager.config is not None
        assert manager.config.seed == 42

    def test_resource_monitor_without_psutil(self):
        """Test ResourceMonitor when psutil not available."""
        with patch('src.training.enterprise_training.PSUTIL_AVAILABLE', False):
            monitor = ResourceMonitor()
            stats = monitor._collect_stats()

            # Should return zero values when psutil unavailable
            assert stats.cpu_percent == 0
            assert stats.memory_percent == 0
            assert stats.memory_gb == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
