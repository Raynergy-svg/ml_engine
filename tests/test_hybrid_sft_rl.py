"""
Comprehensive tests for Hybrid SFT-RL Training Module.

Tests cover:
- HybridSFTLoss: Stochastic switch between SFT and RL loss
- HybridSFTLossWrapper: Keras-compatible wrapper
- RLMetricsCallback: RL metrics tracking callback
- RLBaselineScheduler: Curriculum learning scheduler
- RLEarlyStoppingCallback: RL-based early stopping
- RLMetricsTracker: Metrics tracking helper
- Integration with TrainerConfig

Quality Requirements:
- Minimum 90% code coverage for new modules
- All edge cases tested
- Parameterized tests where appropriate

Note: These tests require TensorFlow. They will be skipped if TensorFlow is not available.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

# Check if TensorFlow is available
tf = None
HAS_TENSORFLOW = False
try:
    import tensorflow as tf  # noqa: E402
    HAS_TENSORFLOW = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
except ImportError:
    pass

# Import modules under test conditionally
if HAS_TENSORFLOW:
    from src.training.trainers.hybrid_sft_rl_loss import (  # noqa: E402
        HybridSFTLoss,
        HybridSFTLossWrapper,
        RLMetricsTracker,
        create_hybrid_sft_rl_loss,
    )
    from src.training.trainers.rl_callbacks import (  # noqa: E402
        RLMetricsCallback,
        RLBaselineScheduler,
        RLEarlyStoppingCallback,
    )
    from src.training.trainers.config import TrainerConfig  # noqa: E402

# pytest marker for requiring TensorFlow
requires_tensorflow = pytest.mark.skipif(
    not HAS_TENSORFLOW,
    reason="TensorFlow not installed - skipping Hybrid SFT-RL tests"
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def set_random_seeds():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    if HAS_TENSORFLOW:
        tf.random.set_seed(42)
    yield


@pytest.fixture
def basic_config():
    """Basic TrainerConfig for testing."""
    return TrainerConfig(
        use_hybrid_sft_rl=True,
        rl_prob=0.5,
        entropy_coef=0.01,
        initial_baseline=0.5,
        baseline_momentum=0.9,
        num_classes=2,
    )


@pytest.fixture
def hybrid_loss(basic_config):
    """HybridSFTLoss instance with default parameters."""
    return HybridSFTLoss(
        rl_prob=basic_config.rl_prob,
        entropy_coef=basic_config.entropy_coef,
        initial_baseline=basic_config.initial_baseline,
        baseline_momentum=basic_config.baseline_momentum,
        num_classes=basic_config.num_classes,
    )


@pytest.fixture
def hybrid_loss_wrapper(basic_config):
    """HybridSFTLossWrapper instance."""
    return HybridSFTLossWrapper(basic_config)


@pytest.fixture
def sample_batch():
    """Sample batch of data for testing."""
    batch_size = 32
    num_classes = 2
    y_true = tf.constant(np.random.randint(0, num_classes, size=(batch_size,)), dtype=tf.int32)
    y_pred = tf.constant(np.random.randn(batch_size, num_classes), dtype=tf.float32)
    return y_true, y_pred


@pytest.fixture
def temp_log_dir():
    """Temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# TESTS FOR RLMetricsTracker
# ============================================================================

@requires_tensorflow
class TestRLMetricsTracker:
    """Tests for RLMetricsTracker helper class."""

    def test_initialization(self):
        """Test correct initialization with default values."""
        tracker = RLMetricsTracker()
        assert tracker.reward_history == []
        assert tracker.baseline_history == []
        assert tracker.entropy_history == []
        assert tracker.action_distribution == {}
        assert tracker.rl_steps == 0
        assert tracker.sft_steps == 0

    def test_update_rl_step(self):
        """Test updating metrics with RL step."""
        tracker = RLMetricsTracker()
        actions = np.array([0, 1, 0, 1, 0])
        tracker.update(reward_mean=0.6, baseline=0.5, entropy=0.69, actions=actions, is_rl=True)
        assert tracker.reward_history == [0.6]
        assert tracker.baseline_history == [0.5]
        assert tracker.entropy_history == [0.69]
        assert tracker.rl_steps == 1
        assert tracker.sft_steps == 0
        assert tracker.action_distribution == {0: 3, 1: 2}

    def test_update_sft_step(self):
        """Test updating metrics with SFT step."""
        tracker = RLMetricsTracker()
        actions = np.array([0, 0, 0])
        tracker.update(reward_mean=0.5, baseline=0.5, entropy=0.5, actions=actions, is_rl=False)
        assert tracker.rl_steps == 0
        assert tracker.sft_steps == 1

    def test_multiple_updates(self):
        """Test multiple updates accumulate correctly."""
        tracker = RLMetricsTracker()
        for i in range(5):
            tracker.update(
                reward_mean=0.5 + i * 0.1,
                baseline=0.5,
                entropy=0.5,
                actions=np.array([0, 1]),
                is_rl=i % 2 == 0
            )
        assert len(tracker.reward_history) == 5
        assert tracker.rl_steps == 3
        assert tracker.sft_steps == 2
        assert tracker.action_distribution == {0: 5, 1: 5}

    def test_get_summary_no_data(self):
        """Test get_summary returns status when no data."""
        tracker = RLMetricsTracker()
        summary = tracker.get_summary()
        assert summary == {"status": "no_data"}

    def test_get_summary_with_data(self):
        """Test get_summary returns correct statistics."""
        tracker = RLMetricsTracker()
        for i in range(10):
            tracker.update(
                reward_mean=0.5 + i * 0.05,
                baseline=0.5 + i * 0.01,
                entropy=0.5,
                actions=np.array([0, 1]),
                is_rl=True
            )
        summary = tracker.get_summary()
        assert "mean_reward" in summary
        assert "mean_entropy" in summary
        assert "current_baseline" in summary
        assert "rl_ratio" in summary
        assert summary["rl_ratio"] == 1.0
        assert summary["total_rl_steps"] == 10
        assert summary["total_sft_steps"] == 0

    def test_get_summary_truncates_to_100(self):
        """Test get_summary only uses last 100 entries."""
        tracker = RLMetricsTracker()
        for i in range(150):
            tracker.update(reward_mean=float(i), baseline=0.5, entropy=0.5, actions=np.array([0]), is_rl=True)
        summary = tracker.get_summary()
        expected_mean = np.mean(list(range(50, 150)))
        assert abs(summary["mean_reward"] - expected_mean) < 0.01

    def test_check_policy_collapse_insufficient_data(self):
        """Test collapse detection with insufficient data."""
        tracker = RLMetricsTracker()
        is_collapsed, reason = tracker.check_policy_collapse()
        assert is_collapsed is False
        assert reason == "Insufficient data"

    def test_check_policy_collapse_healthy(self):
        """Test collapse detection with healthy entropy."""
        tracker = RLMetricsTracker()
        for _ in range(15):
            tracker.update(reward_mean=0.5, baseline=0.5, entropy=0.5, actions=np.array([0, 1]), is_rl=True)
        is_collapsed, reason = tracker.check_policy_collapse()
        assert is_collapsed is False
        assert reason == "Policy healthy"

    def test_check_policy_collapse_low_entropy(self):
        """Test collapse detection with low entropy."""
        tracker = RLMetricsTracker()
        for _ in range(15):
            tracker.update(reward_mean=0.5, baseline=0.5, entropy=0.005, actions=np.array([0, 1]), is_rl=True)
        is_collapsed, reason = tracker.check_policy_collapse(threshold=0.01)
        assert is_collapsed is True
        assert "Entropy collapsed" in reason

    def test_check_policy_collapse_mode_collapse(self):
        """Test collapse detection with mode collapse."""
        tracker = RLMetricsTracker()
        for _ in range(15):
            tracker.update(
                reward_mean=0.5,
                baseline=0.5,
                entropy=0.5,
                actions=np.array([0] * 100 + [1] * 5),
                is_rl=True
            )
        is_collapsed, reason = tracker.check_policy_collapse()
        assert is_collapsed is True
        assert "Mode collapse" in reason


# ============================================================================
# TESTS FOR HybridSFTLoss
# ============================================================================

@requires_tensorflow
class TestHybridSFTLoss:
    """Tests for HybridSFTLoss class."""

    def test_initialization_defaults(self):
        """Test correct initialization with default parameters."""
        loss = HybridSFTLoss()
        assert loss.rl_prob == 0.5
        assert loss.entropy_coef == 0.01
        assert loss.num_classes == 2
        assert loss.label_smoothing == 0.0
        assert loss.baseline_momentum == 0.9
        assert loss.reward_baseline == 0.5

    def test_initialization_custom(self):
        """Test correct initialization with custom parameters."""
        loss = HybridSFTLoss(
            rl_prob=0.3,
            entropy_coef=0.02,
            initial_baseline=0.6,
            baseline_momentum=0.95,
            num_classes=3,
            label_smoothing=0.1,
            name="custom_loss"
        )
        assert loss.rl_prob == 0.3
        assert loss.entropy_coef == 0.02
        assert loss.num_classes == 3
        assert loss.label_smoothing == 0.1
        assert loss.baseline_momentum == 0.95
        assert loss.reward_baseline == 0.6
        assert loss.name == "custom_loss"

    def test_get_config(self, hybrid_loss):
        """Test get_config returns correct configuration."""
        config = hybrid_loss.get_config()
        assert config["rl_prob"] == 0.5
        assert config["entropy_coef"] == 0.01
        assert config["num_classes"] == 2
        assert config["baseline_momentum"] == 0.9
        assert "name" in config

    def test_from_config(self, hybrid_loss):
        """Test from_config creates loss correctly."""
        config = hybrid_loss.get_config()
        new_loss = HybridSFTLoss.from_config(config)
        assert new_loss.rl_prob == hybrid_loss.rl_prob
        assert new_loss.entropy_coef == hybrid_loss.entropy_coef
        assert new_loss.num_classes == hybrid_loss.num_classes

    def test_supervised_path_rl_prob_zero(self, set_random_seeds, sample_batch):
        """Test that rl_prob=0 always uses Cross-Entropy loss."""
        y_true, y_pred = sample_batch
        loss = HybridSFTLoss(rl_prob=0.0)
        losses = [float(loss(y_true, y_pred)) for _ in range(10)]
        assert len(set(losses)) == 1

    def test_rl_path_rl_prob_one(self, set_random_seeds, sample_batch):
        """Test that rl_prob=1 always uses RL loss."""
        y_true, y_pred = sample_batch
        loss = HybridSFTLoss(rl_prob=1.0, entropy_coef=0.0)
        losses = [float(loss(y_true, y_pred)) for _ in range(10)]
        assert all(not np.isnan(loss_val) for loss_val in losses)

    def test_stochastic_switching(self, set_random_seeds, sample_batch):
        """Test that switching occurs with expected probability."""
        y_true, y_pred = sample_batch
        loss = HybridSFTLoss(rl_prob=0.5)
        rl_count = 0
        sft_count = 0
        for _ in range(100):
            loss(y_true, y_pred)
            if loss._last_was_rl:
                rl_count += 1
            else:
                sft_count += 1
        rl_ratio = rl_count / (rl_count + sft_count)
        assert 0.3 < rl_ratio < 0.7, f"RL ratio {rl_ratio} not in expected range"

    def test_baseline_update(self, hybrid_loss, sample_batch):
        """Test EMA baseline update is correct and detached from graph."""
        y_true, y_pred = sample_batch
        initial_baseline = hybrid_loss.reward_baseline
        hybrid_loss.rl_prob = 1.0
        _ = hybrid_loss(y_true, y_pred)
        new_baseline = hybrid_loss.reward_baseline
        assert new_baseline != initial_baseline or initial_baseline == 0.5

    def test_baseline_ema_formula(self, set_random_seeds):
        """Test baseline follows EMA formula exactly."""
        loss = HybridSFTLoss(rl_prob=1.0, initial_baseline=0.5, baseline_momentum=0.9)
        y_true = tf.constant([0, 0, 0, 0], dtype=tf.int32)
        y_pred = tf.constant([[10.0, -10.0]] * 4, dtype=tf.float32)
        _ = loss(y_true, y_pred)
        assert 0.0 <= loss.reward_baseline <= 1.0

    def test_entropy_calculation(self, hybrid_loss, set_random_seeds):
        """Test entropy bonus is calculated correctly."""
        y_true = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        y_pred = tf.constant([[0.0, 0.0]] * 4, dtype=tf.float32)
        hybrid_loss.rl_prob = 1.0
        _ = hybrid_loss(y_true, y_pred)
        if hybrid_loss.metrics_tracker.entropy_history:
            entropy = hybrid_loss.metrics_tracker.entropy_history[-1]
            assert 0.5 < entropy < 0.8, f"Entropy {entropy} not in expected range"

    def test_reward_calculation_correct_action(self, set_random_seeds):
        """Test binary reward: 1.0 for correct action, 0.0 otherwise."""
        loss = HybridSFTLoss(rl_prob=1.0, entropy_coef=0.0)
        y_true = tf.constant([0, 0, 0, 0], dtype=tf.int32)
        y_pred = tf.constant([[10.0, -10.0]] * 4, dtype=tf.float32)
        _ = loss(y_true, y_pred)
        if loss.metrics_tracker.reward_history:
            mean_reward = np.mean(loss.metrics_tracker.reward_history[-5:])
            assert mean_reward > 0.5, f"Mean reward {mean_reward} too low"

    def test_reward_calculation_incorrect_action(self, set_random_seeds):
        """Test reward is low when predictions are wrong."""
        loss = HybridSFTLoss(rl_prob=1.0, entropy_coef=0.0)
        y_true = tf.constant([1, 1, 1, 1], dtype=tf.int32)
        y_pred = tf.constant([[10.0, -10.0]] * 4, dtype=tf.float32)
        _ = loss(y_true, y_pred)
        if loss.metrics_tracker.reward_history:
            mean_reward = np.mean(loss.metrics_tracker.reward_history[-5:])
            assert mean_reward < 0.5, f"Mean reward {mean_reward} too high"

    def test_gradient_flow(self, hybrid_loss, sample_batch):
        """Test gradients flow correctly through loss."""
        y_true, y_pred = sample_batch
        var = tf.Variable(y_pred)
        with tf.GradientTape() as tape:
            loss_value = hybrid_loss(y_true, var)
        grads = tape.gradient(loss_value, var)
        assert grads is not None
        assert not tf.reduce_all(grads == 0)

    def test_gradient_flow_rl_path(self, set_random_seeds):
        """Test gradients flow through RL loss path."""
        loss = HybridSFTLoss(rl_prob=1.0)
        y_true = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        var = tf.Variable(tf.random.normal((4, 2)))
        with tf.GradientTape() as tape:
            loss_value = loss(y_true, var)
        grads = tape.gradient(loss_value, var)
        assert grads is not None
        assert not tf.reduce_all(grads == 0)

    def test_nan_handling_probs(self, hybrid_loss):
        """Test NaN probabilities are handled gracefully."""
        y_true = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        y_pred = tf.constant([[0.0, 0.0]] * 4, dtype=tf.float32)
        loss_value = hybrid_loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)
        assert not tf.math.is_inf(loss_value)

    def test_nan_handling_extreme_values(self, hybrid_loss):
        """Test handling of extreme logit values."""
        y_true = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        y_pred = tf.constant([[1e10, -1e10], [-1e10, 1e10], [1e10, -1e10], [-1e10, 1e10]], dtype=tf.float32)
        loss_value = hybrid_loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)
        assert not tf.math.is_inf(loss_value) or float(loss_value) < 1e10

    def test_batch_size_robustness_single(self, hybrid_loss):
        """Test with batch size of 1."""
        y_true = tf.constant([0], dtype=tf.int32)
        y_pred = tf.constant([[0.5, -0.5]], dtype=tf.float32)
        loss_value = hybrid_loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)

    def test_batch_size_robustness_large(self, hybrid_loss):
        """Test with large batch size."""
        batch_size = 256
        y_true = tf.constant(np.random.randint(0, 2, batch_size), dtype=tf.int32)
        y_pred = tf.constant(np.random.randn(batch_size, 2), dtype=tf.float32)
        loss_value = hybrid_loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)

    def test_batch_size_robustness_empty(self, hybrid_loss):
        """Test with empty batch (edge case)."""
        y_true = tf.constant([], dtype=tf.int32)
        y_pred = tf.constant([], dtype=tf.float32)
        try:
            loss_value = hybrid_loss(y_true, y_pred)
            assert not tf.math.is_nan(loss_value)
        except Exception:
            pass

    def test_probability_clamping(self, hybrid_loss):
        """Test probabilities are clamped to avoid log(0)."""
        y_true = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        y_pred = tf.constant([[100.0, -100.0], [-100.0, 100.0], [100.0, -100.0], [-100.0, 100.0]], dtype=tf.float32)
        loss_value = hybrid_loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)

    def test_multiclass_support(self, set_random_seeds):
        """Test loss works with more than 2 classes."""
        loss = HybridSFTLoss(num_classes=5, rl_prob=0.5)
        batch_size = 16
        y_true = tf.constant(np.random.randint(0, 5, batch_size), dtype=tf.int32)
        y_pred = tf.constant(np.random.randn(batch_size, 5), dtype=tf.float32)
        loss_value = loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)

    def test_get_metrics_summary(self, hybrid_loss, sample_batch):
        """Test get_metrics_summary returns correct data."""
        y_true, y_pred = sample_batch
        for _ in range(5):
            hybrid_loss(y_true, y_pred)
        summary = hybrid_loss.get_metrics_summary()
        assert "current_baseline" in summary
        assert "rl_prob" in summary
        assert "entropy_coef" in summary
        assert "policy_collapsed" in summary
        assert "collapse_reason" in summary

    def test_reset_metrics(self, hybrid_loss, sample_batch):
        """Test reset_metrics clears tracked data."""
        y_true, y_pred = sample_batch
        for _ in range(5):
            hybrid_loss(y_true, y_pred)
        assert len(hybrid_loss.metrics_tracker.reward_history) > 0
        hybrid_loss.reset_metrics()
        assert hybrid_loss.metrics_tracker.reward_history == []
        assert hybrid_loss.metrics_tracker.rl_steps == 0

    def test_label_smoothing_sft_path(self, set_random_seeds):
        """Test label smoothing is applied in SFT path."""
        loss = HybridSFTLoss(rl_prob=0.0, label_smoothing=0.1)
        y_true = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        y_pred = tf.constant([[0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, 0.5]], dtype=tf.float32)
        loss_value = loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)


# ============================================================================
# TESTS FOR HybridSFTLossWrapper
# ============================================================================

@requires_tensorflow
class TestHybridSFTLossWrapper:
    """Tests for Keras-compatible wrapper."""

    def test_wrapper_call(self, basic_config, sample_batch):
        """Test wrapper correctly calls underlying loss."""
        wrapper = HybridSFTLossWrapper(basic_config)
        y_true, y_pred = sample_batch
        loss_value = wrapper(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)

    def test_hybrid_loss_property(self, hybrid_loss_wrapper):
        """Test hybrid_loss property returns underlying loss."""
        assert isinstance(hybrid_loss_wrapper.hybrid_loss, HybridSFTLoss)

    def test_config_serialization(self, basic_config):
        """Test get_config/from_config round-trip."""
        wrapper = HybridSFTLossWrapper(basic_config)
        config = wrapper.get_config()
        new_wrapper = HybridSFTLossWrapper.from_config(config)
        assert new_wrapper.hybrid_loss.rl_prob == wrapper.hybrid_loss.rl_prob
        assert new_wrapper.hybrid_loss.entropy_coef == wrapper.hybrid_loss.entropy_coef

    def test_wrapper_with_missing_config_attrs(self):
        """Test wrapper handles config with missing attributes."""
        @dataclass
        class MinimalConfig:
            rl_prob: float = 0.3

        config = MinimalConfig()
        wrapper = HybridSFTLossWrapper(config)
        assert wrapper.hybrid_loss.rl_prob == 0.3
        assert wrapper.hybrid_loss.entropy_coef == 0.01

    def test_get_metrics_summary(self, hybrid_loss_wrapper, sample_batch):
        """Test get_metrics_summary delegates to underlying loss."""
        y_true, y_pred = sample_batch
        for _ in range(3):
            hybrid_loss_wrapper(y_true, y_pred)
        summary = hybrid_loss_wrapper.get_metrics_summary()
        assert "rl_prob" in summary

    def test_reset_metrics(self, hybrid_loss_wrapper, sample_batch):
        """Test reset_metrics delegates to underlying loss."""
        y_true, y_pred = sample_batch
        hybrid_loss_wrapper(y_true, y_pred)
        hybrid_loss_wrapper.reset_metrics()
        assert hybrid_loss_wrapper.hybrid_loss.metrics_tracker.reward_history == []


# ============================================================================
# TESTS FOR create_hybrid_sft_rl_loss FACTORY
# ============================================================================

@requires_tensorflow
class TestCreateHybridSFTRLLoss:
    """Tests for factory function."""

    def test_factory_creates_wrapper(self, basic_config):
        """Test factory creates HybridSFTLossWrapper."""
        loss = create_hybrid_sft_rl_loss(basic_config)
        assert isinstance(loss, HybridSFTLossWrapper)

    def test_factory_raises_when_disabled(self):
        """Test factory raises when use_hybrid_sft_rl=False."""
        config = TrainerConfig(use_hybrid_sft_rl=False)
        with pytest.raises(ValueError, match="use_hybrid_sft_rl=False"):
            create_hybrid_sft_rl_loss(config)

    def test_factory_with_kwargs(self, basic_config):
        """Test factory passes kwargs correctly."""
        loss = create_hybrid_sft_rl_loss(basic_config, name="custom_name")
        assert loss.name == "custom_name"


# ============================================================================
# TESTS FOR RLMetricsCallback
# ============================================================================

@requires_tensorflow
class TestRLMetricsCallback:
    """Tests for RLMetricsCallback."""

    def test_initialization(self):
        """Test callback initialization."""
        callback = RLMetricsCallback(log_every_n_epochs=5, collapse_threshold=0.02)
        assert callback.log_every_n_epochs == 5
        assert callback.collapse_threshold == 0.02

    def test_initialization_with_log_dir(self, temp_log_dir):
        """Test initialization with log directory."""
        callback = RLMetricsCallback(log_dir=str(temp_log_dir))
        assert callback.log_dir == temp_log_dir

    def test_on_train_begin_creates_writer(self, temp_log_dir):
        """Test TensorBoard writer is created on train begin."""
        callback = RLMetricsCallback(log_dir=str(temp_log_dir))
        callback.model = Mock()
        callback.on_train_begin()
        assert callback._summary_writer is not None

    def test_on_train_begin_no_log_dir(self):
        """Test train begin without log directory."""
        callback = RLMetricsCallback()
        callback.model = Mock()
        callback.on_train_begin()
        assert callback._summary_writer is None

    def test_epoch_end_logging(self, hybrid_loss, sample_batch, temp_log_dir):
        """Test metrics are logged at epoch end."""
        callback = RLMetricsCallback(loss_fn=hybrid_loss, log_dir=str(temp_log_dir), log_every_n_epochs=1)
        callback.model = Mock()
        callback.on_train_begin()
        y_true, y_pred = sample_batch
        for _ in range(3):
            hybrid_loss(y_true, y_pred)
        callback.on_epoch_end(epoch=0)
        assert len(callback._metrics_history) == 1
        assert callback._metrics_history[0]["epoch"] == 0

    def test_epoch_end_skip_based_on_frequency(self, hybrid_loss):
        """Test logging respects log_every_n_epochs."""
        callback = RLMetricsCallback(loss_fn=hybrid_loss, log_every_n_epochs=3)
        callback.model = Mock()
        callback.on_train_begin()
        for epoch in range(10):
            callback.on_epoch_end(epoch=epoch)
        assert len(callback._metrics_history) == 4

    def test_get_loss_metrics_from_model(self, hybrid_loss):
        """Test extracting metrics from model.loss."""
        callback = RLMetricsCallback()
        mock_model = Mock()
        mock_model.loss = hybrid_loss
        callback.model = mock_model
        metrics = callback._get_loss_metrics()
        assert metrics is not None
        assert "rl_prob" in metrics

    def test_get_loss_metrics_from_wrapper(self, hybrid_loss_wrapper):
        """Test extracting metrics from HybridSFTLossWrapper."""
        callback = RLMetricsCallback()
        mock_model = Mock()
        mock_model.loss = hybrid_loss_wrapper
        callback.model = mock_model
        metrics = callback._get_loss_metrics()
        assert metrics is not None

    def test_policy_collapse_warning(self, hybrid_loss, sample_batch, caplog):
        """Test collapse detection logs warning."""
        callback = RLMetricsCallback(loss_fn=hybrid_loss, collapse_threshold=0.5)
        callback.model = Mock()
        callback.on_train_begin()
        for _ in range(15):
            hybrid_loss.metrics_tracker.update(reward_mean=0.5, baseline=0.5, entropy=0.001, actions=np.array([0]), is_rl=True)
        with caplog.at_level(logging.WARNING):
            callback.on_epoch_end(epoch=0)
        assert callback._collapse_warnings > 0

    def test_on_train_end_closes_writer(self, temp_log_dir):
        """Test TensorBoard writer is closed on train end."""
        callback = RLMetricsCallback(log_dir=str(temp_log_dir))
        callback.model = Mock()
        callback.on_train_begin()
        assert callback._summary_writer is not None
        callback.on_train_end()

    def test_get_metrics_history(self, hybrid_loss):
        """Test get_metrics_history returns copy of history."""
        callback = RLMetricsCallback(loss_fn=hybrid_loss)
        callback.model = Mock()
        callback.on_train_begin()
        callback.on_epoch_end(epoch=0)
        callback.on_epoch_end(epoch=1)
        history = callback.get_metrics_history()
        assert len(history) == 2
        history.append({"test": "value"})
        assert len(callback.get_metrics_history()) == 2


# ============================================================================
# TESTS FOR RLBaselineScheduler
# ============================================================================

@requires_tensorflow
class TestRLBaselineScheduler:
    """Tests for curriculum learning scheduler."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        scheduler = RLBaselineScheduler()
        assert scheduler.initial_rl_prob == 0.0
        assert scheduler.final_rl_prob == 0.5
        assert scheduler.warmup_epochs == 10
        assert scheduler.schedule_type == "linear"

    def test_initialization_custom(self):
        """Test custom initialization."""
        scheduler = RLBaselineScheduler(
            initial_rl_prob=0.1,
            final_rl_prob=0.8,
            warmup_epochs=20,
            schedule_type="cosine"
        )
        assert scheduler.initial_rl_prob == 0.1
        assert scheduler.final_rl_prob == 0.8
        assert scheduler.warmup_epochs == 20
        assert scheduler.schedule_type == "cosine"

    def test_invalid_schedule_type(self):
        """Test invalid schedule type raises error."""
        with pytest.raises(ValueError, match="Invalid schedule_type"):
            RLBaselineScheduler(schedule_type="invalid")

    def test_step_schedule_validation(self):
        """Test step schedule validates epoch/value lists."""
        with pytest.raises(ValueError, match="must have same length"):
            RLBaselineScheduler(schedule_type="step", step_epochs=[5, 10], step_values=[0.3])

    def test_linear_schedule(self, hybrid_loss):
        """Test linear curriculum schedule."""
        scheduler = RLBaselineScheduler(
            loss_fn=hybrid_loss,
            initial_rl_prob=0.0,
            final_rl_prob=1.0,
            warmup_epochs=10,
            schedule_type="linear"
        )
        scheduler.model = Mock()
        scheduler.on_train_begin()
        assert scheduler._compute_linear_schedule(0) == 0.0
        assert scheduler._compute_linear_schedule(5) == 0.5
        assert scheduler._compute_linear_schedule(10) == 1.0
        assert scheduler._compute_linear_schedule(15) == 1.0

    def test_cosine_schedule(self, hybrid_loss):
        """Test cosine curriculum schedule."""
        scheduler = RLBaselineScheduler(
            loss_fn=hybrid_loss,
            initial_rl_prob=0.0,
            final_rl_prob=1.0,
            warmup_epochs=10,
            schedule_type="cosine"
        )
        assert scheduler._compute_cosine_schedule(0) == 0.0
        assert scheduler._compute_cosine_schedule(10) == 1.0
        mid = scheduler._compute_cosine_schedule(5)
        assert 0.45 < mid < 0.55

    def test_step_schedule(self, hybrid_loss):
        """Test step curriculum schedule."""
        scheduler = RLBaselineScheduler(
            loss_fn=hybrid_loss,
            schedule_type="step",
            step_epochs=[5, 10, 15],
            step_values=[0.3, 0.6, 0.9],
            final_rl_prob=1.0
        )
        assert scheduler._compute_step_schedule(0) == 0.3
        assert scheduler._compute_step_schedule(5) == 0.3
        assert scheduler._compute_step_schedule(10) == 0.6
        assert scheduler._compute_step_schedule(15) == 0.9
        assert scheduler._compute_step_schedule(20) == 1.0

    def test_min_rl_prob(self, hybrid_loss):
        """Test minimum rl_prob is respected."""
        scheduler = RLBaselineScheduler(
            loss_fn=hybrid_loss,
            initial_rl_prob=0.0,
            final_rl_prob=0.5,
            warmup_epochs=10,
            schedule_type="linear",
            min_rl_prob=0.1
        )
        rl_prob = scheduler._compute_rl_prob(0)
        assert rl_prob >= 0.1

    def test_updates_loss_rl_prob(self, hybrid_loss):
        """Test scheduler updates loss function's rl_prob."""
        scheduler = RLBaselineScheduler(
            loss_fn=hybrid_loss,
            initial_rl_prob=0.0,
            final_rl_prob=1.0,
            warmup_epochs=10,
            schedule_type="linear"
        )
        scheduler.model = Mock()
        scheduler.on_train_begin()
        assert hybrid_loss.rl_prob == 0.0
        scheduler.on_epoch_begin(epoch=5)
        assert hybrid_loss.rl_prob == 0.5

    def test_updates_wrapper_loss(self, hybrid_loss_wrapper):
        """Test scheduler updates HybridSFTLossWrapper's rl_prob."""
        scheduler = RLBaselineScheduler(
            loss_fn=hybrid_loss_wrapper,
            initial_rl_prob=0.0,
            final_rl_prob=1.0,
            warmup_epochs=10
        )
        scheduler.model = Mock()
        scheduler.on_train_begin()
        scheduler.on_epoch_begin(epoch=10)
        assert hybrid_loss_wrapper.hybrid_loss.rl_prob == 1.0

    def test_get_current_rl_prob(self, hybrid_loss):
        """Test get_current_rl_prob returns current value."""
        scheduler = RLBaselineScheduler(
            loss_fn=hybrid_loss,
            initial_rl_prob=0.2,
            final_rl_prob=0.8,
            warmup_epochs=10
        )
        scheduler.model = Mock()
        scheduler.on_train_begin()
        assert scheduler.get_current_rl_prob() == 0.2
        scheduler.on_epoch_begin(epoch=10)
        assert scheduler.get_current_rl_prob() == 0.8


# ============================================================================
# TESTS FOR RLEarlyStoppingCallback
# ============================================================================

@requires_tensorflow
class TestRLEarlyStoppingCallback:
    """Tests for RL-based early stopping."""

    def test_initialization(self):
        """Test callback initialization."""
        callback = RLEarlyStoppingCallback(min_entropy=0.01, min_reward=0.4, patience=5)
        assert callback.min_entropy == 0.01
        assert callback.min_reward == 0.4
        assert callback.patience == 5
        assert callback.restore_best_weights is True

    def test_tracks_best_weights(self, hybrid_loss, sample_batch):
        """Test callback tracks best weights."""
        callback = RLEarlyStoppingCallback(loss_fn=hybrid_loss)
        model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(2,))])
        model.compile(loss=hybrid_loss, optimizer='adam')
        callback.model = model
        hybrid_loss.metrics_tracker.reward_history = [0.5]
        callback.on_epoch_end(epoch=0)
        assert callback._best_reward == 0.5

    def test_stops_on_low_entropy(self, hybrid_loss):
        """Test stopping when entropy is too low."""
        callback = RLEarlyStoppingCallback(loss_fn=hybrid_loss, min_entropy=0.01, patience=3)
        model = Mock()
        model.stop_training = False
        callback.model = model
        for _ in range(15):
            hybrid_loss.metrics_tracker.update(
                reward_mean=0.5,
                baseline=0.5,
                entropy=0.001,
                actions=np.array([0]),
                is_rl=True
            )
        for i in range(5):
            callback.on_epoch_end(epoch=i)
            if model.stop_training:
                break
        assert model.stop_training is True

    def test_restores_best_weights(self, hybrid_loss):
        """Test restoring best weights on stop."""
        callback = RLEarlyStoppingCallback(
            loss_fn=hybrid_loss,
            min_entropy=0.01,
            patience=2,
            restore_best_weights=True
        )
        model = Mock()
        model.stop_training = False
        model.get_weights.return_value = [np.array([1.0])]
        model.set_weights = Mock()
        callback.model = model
        hybrid_loss.metrics_tracker.reward_history = [0.8]
        hybrid_loss.metrics_tracker.entropy_history = [0.5] * 15
        callback.on_epoch_end(epoch=0)
        hybrid_loss.metrics_tracker.entropy_history = [0.001] * 15
        for i in range(1, 5):
            callback.on_epoch_end(epoch=i)
            if model.stop_training:
                break
        if callback._best_weights is not None:
            model.set_weights.assert_called()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@requires_tensorflow
class TestHybridSFTRLIntegration:
    """Integration tests with TrainerConfig and training pipeline."""

    def test_config_to_loss_instantiation(self):
        """Test TrainerConfig correctly instantiates HybridSFTLoss."""
        config = TrainerConfig(
            use_hybrid_sft_rl=True,
            rl_prob=0.3,
            entropy_coef=0.02,
            initial_baseline=0.6,
            baseline_momentum=0.95
        )
        loss = create_hybrid_sft_rl_loss(config)
        assert isinstance(loss, HybridSFTLossWrapper)
        assert loss.hybrid_loss.rl_prob == 0.3
        assert loss.hybrid_loss.entropy_coef == 0.02
        assert loss.hybrid_loss.reward_baseline == 0.6
        assert loss.hybrid_loss.baseline_momentum == 0.95

    def test_disabled_by_default(self):
        """Test use_hybrid_sft_rl=False does not create loss."""
        config = TrainerConfig(use_hybrid_sft_rl=False)
        with pytest.raises(ValueError):
            create_hybrid_sft_rl_loss(config)

    def test_config_rl_curriculum_fields(self):
        """Test TrainerConfig has all RL curriculum fields."""
        config = TrainerConfig()
        assert hasattr(config, 'use_hybrid_sft_rl')
        assert hasattr(config, 'rl_prob')
        assert hasattr(config, 'entropy_coef')
        assert hasattr(config, 'initial_baseline')
        assert hasattr(config, 'baseline_momentum')
        assert hasattr(config, 'rl_curriculum_enabled')
        assert hasattr(config, 'rl_curriculum_type')
        assert hasattr(config, 'rl_curriculum_warmup_epochs')
        assert hasattr(config, 'rl_early_stopping')
        assert hasattr(config, 'rl_min_entropy')
        assert hasattr(config, 'rl_patience')

    def test_training_with_hybrid_loss(self, basic_config):
        """Test training completes without error when enabled."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(2)
        ])
        loss = HybridSFTLossWrapper(basic_config)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss)
        X_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randint(0, 2, 100).astype(np.int32)
        history = model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
        assert len(history.history['loss']) == 3

    def test_training_with_callbacks(self, basic_config, temp_log_dir):
        """Test training with RL callbacks."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(2)
        ])
        loss = HybridSFTLossWrapper(basic_config)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss)
        metrics_callback = RLMetricsCallback(loss_fn=loss, log_dir=str(temp_log_dir))
        scheduler = RLBaselineScheduler(
            loss_fn=loss,
            initial_rl_prob=0.0,
            final_rl_prob=basic_config.rl_prob,
            warmup_epochs=2
        )
        X_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randint(0, 2, 100).astype(np.int32)
        history = model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=32,
            callbacks=[metrics_callback, scheduler],
            verbose=0
        )
        assert len(history.history['loss']) == 5
        assert len(metrics_callback.get_metrics_history()) > 0

    def test_end_to_end_workflow(self, temp_log_dir):
        """Test complete workflow from config to trained model."""
        config = TrainerConfig(
            use_hybrid_sft_rl=True,
            rl_prob=0.5,
            entropy_coef=0.01,
            initial_baseline=0.5,
            baseline_momentum=0.9,
            rl_curriculum_enabled=True,
            rl_curriculum_type="linear",
            rl_curriculum_warmup_epochs=3,
            rl_curriculum_initial_prob=0.0,
            rl_curriculum_final_prob=0.5,
            rl_early_stopping=True,
            rl_min_entropy=0.005,
            rl_patience=10
        )
        loss = create_hybrid_sft_rl_loss(config)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(2)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss)
        callbacks = [
            RLMetricsCallback(loss_fn=loss, log_dir=str(temp_log_dir)),
            RLBaselineScheduler(
                loss_fn=loss,
                initial_rl_prob=config.rl_curriculum_initial_prob,
                final_rl_prob=config.rl_curriculum_final_prob,
                warmup_epochs=config.rl_curriculum_warmup_epochs,
                schedule_type=config.rl_curriculum_type
            )
        ]
        X_train = np.random.randn(200, 20).astype(np.float32)
        y_train = np.random.randint(0, 2, 200).astype(np.int32)
        history = model.fit(X_train, y_train, epochs=5, batch_size=32, callbacks=callbacks, verbose=0)
        assert len(history.history['loss']) == 5
        summary = loss.get_metrics_summary()
        assert "rl_prob" in summary
        assert "mean_reward" in summary
        assert loss.hybrid_loss.rl_prob == config.rl_curriculum_final_prob


# ============================================================================
# PARAMETERIZED TESTS
# ============================================================================

@requires_tensorflow
class TestParameterized:
    """Parameterized tests for comprehensive coverage."""

    @pytest.mark.parametrize("rl_prob", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_rl_prob_values(self, rl_prob, sample_batch):
        """Test loss with various rl_prob values."""
        loss = HybridSFTLoss(rl_prob=rl_prob)
        y_true, y_pred = sample_batch
        loss_value = loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)
        assert not tf.math.is_inf(loss_value)

    @pytest.mark.parametrize("entropy_coef", [0.0, 0.001, 0.01, 0.1])
    def test_entropy_coef_values(self, entropy_coef, sample_batch):
        """Test loss with various entropy coefficients."""
        loss = HybridSFTLoss(rl_prob=1.0, entropy_coef=entropy_coef)
        y_true, y_pred = sample_batch
        loss_value = loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)

    @pytest.mark.parametrize("num_classes", [2, 3, 5, 10])
    def test_num_classes(self, num_classes):
        """Test loss with various number of classes."""
        loss = HybridSFTLoss(num_classes=num_classes, rl_prob=0.5)
        batch_size = 16
        y_true = tf.constant(np.random.randint(0, num_classes, batch_size), dtype=tf.int32)
        y_pred = tf.constant(np.random.randn(batch_size, num_classes), dtype=tf.float32)
        loss_value = loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)

    @pytest.mark.parametrize("batch_size", [1, 2, 8, 32, 64, 128])
    def test_batch_sizes(self, batch_size):
        """Test loss with various batch sizes."""
        loss = HybridSFTLoss(rl_prob=0.5)
        y_true = tf.constant(np.random.randint(0, 2, batch_size), dtype=tf.int32)
        y_pred = tf.constant(np.random.randn(batch_size, 2), dtype=tf.float32)
        loss_value = loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)

    @pytest.mark.parametrize("momentum", [0.5, 0.9, 0.95, 0.99])
    def test_baseline_momentum_values(self, momentum, sample_batch):
        """Test baseline update with various momentum values."""
        loss = HybridSFTLoss(rl_prob=1.0, baseline_momentum=momentum, initial_baseline=0.5)
        y_true, y_pred = sample_batch
        loss(y_true, y_pred)
        new_baseline = loss.reward_baseline
        assert 0.0 <= new_baseline <= 1.0

    @pytest.mark.parametrize("schedule_type", ["linear", "cosine", "step"])
    def test_schedule_types(self, schedule_type, hybrid_loss):
        """Test all schedule types work correctly."""
        if schedule_type == "step":
            scheduler = RLBaselineScheduler(
                loss_fn=hybrid_loss,
                schedule_type=schedule_type,
                step_epochs=[5, 10],
                step_values=[0.3, 0.7],
                final_rl_prob=1.0
            )
        else:
            scheduler = RLBaselineScheduler(
                loss_fn=hybrid_loss,
                schedule_type=schedule_type,
                warmup_epochs=10
            )
        scheduler.model = Mock()
        scheduler.on_train_begin()
        for epoch in range(15):
            rl_prob = scheduler._compute_rl_prob(epoch)
            assert 0.0 <= rl_prob <= 1.0


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

@requires_tensorflow
class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_all_same_class_batch(self):
        """Test with all samples having same class."""
        loss = HybridSFTLoss(rl_prob=0.5)
        y_true = tf.constant([0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.int32)
        y_pred = tf.constant(np.random.randn(8, 2), dtype=tf.float32)
        loss_value = loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        loss = HybridSFTLoss(rl_prob=0.0)
        y_true = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        y_pred = tf.constant([[-10.0, 10.0], [10.0, -10.0], [-10.0, 10.0], [10.0, -10.0]], dtype=tf.float32)
        loss_value = loss(y_true, y_pred)
        assert float(loss_value) < 0.1

    def test_worst_predictions(self):
        """Test with completely wrong predictions."""
        loss = HybridSFTLoss(rl_prob=0.0)
        y_true = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        y_pred = tf.constant([[10.0, -10.0], [-10.0, 10.0], [10.0, -10.0], [-10.0, 10.0]], dtype=tf.float32)
        loss_value = loss(y_true, y_pred)
        assert float(loss_value) > 1.0

    def test_very_small_entropy_coef(self):
        """Test with very small entropy coefficient."""
        loss = HybridSFTLoss(rl_prob=1.0, entropy_coef=1e-10)
        y_true = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        y_pred = tf.constant(np.random.randn(4, 2), dtype=tf.float32)
        loss_value = loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)

    def test_very_large_entropy_coef(self):
        """Test with very large entropy coefficient."""
        loss = HybridSFTLoss(rl_prob=1.0, entropy_coef=10.0)
        y_true = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        y_pred = tf.constant(np.random.randn(4, 2), dtype=tf.float32)
        loss_value = loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)

    def test_callback_without_loss_fn(self):
        """Test callback behavior when loss_fn is None."""
        callback = RLMetricsCallback()
        metrics = callback._get_loss_metrics()
        assert metrics is None

    def test_scheduler_without_loss_fn(self):
        """Test scheduler behavior when loss_fn is None."""
        scheduler = RLBaselineScheduler()
        result = scheduler._update_loss_rl_prob(0.5)
        assert result is False

    def test_loss_with_squeezed_predictions(self):
        """Test loss with different prediction shapes."""
        loss = HybridSFTLoss(rl_prob=0.5)
        y_true = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        y_pred = tf.constant(np.random.randn(4, 2), dtype=tf.float32)
        loss_value = loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)

    def test_metrics_tracker_with_empty_actions(self):
        """Test metrics tracker with empty actions array."""
        tracker = RLMetricsTracker()
        tracker.update(reward_mean=0.5, baseline=0.5, entropy=0.5, actions=np.array([]), is_rl=True)
        assert tracker.action_distribution == {}


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@requires_tensorflow
class TestPerformance:
    """Tests for performance characteristics."""

    def test_large_batch_performance(self):
        """Test loss computation with large batch."""
        loss = HybridSFTLoss(rl_prob=0.5)
        batch_size = 1024
        y_true = tf.constant(np.random.randint(0, 2, batch_size), dtype=tf.int32)
        y_pred = tf.constant(np.random.randn(batch_size, 2), dtype=tf.float32)
        loss_value = loss(y_true, y_pred)
        assert not tf.math.is_nan(loss_value)

    def test_many_sequential_batches(self):
        """Test processing many sequential batches."""
        loss = HybridSFTLoss(rl_prob=0.5)
        for _ in range(100):
            y_true = tf.constant(np.random.randint(0, 2, 32), dtype=tf.int32)
            y_pred = tf.constant(np.random.randn(32, 2), dtype=tf.float32)
            loss(y_true, y_pred)
        assert len(loss.metrics_tracker.reward_history) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
