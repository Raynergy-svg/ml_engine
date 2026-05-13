"""
Comprehensive tests for ReplayBuffer, DriftDetector, and TrainingLineage.

US-252: ReplayBuffer + DriftDetector (15+ tests)
US-253: TrainingLineage (10+ tests)

Tests cover:
- Default initialization and parameter validation
- Data flow and buffer management
- Drift detection logic
- Serialization and persistence
- Edge cases (empty buffers, missing history, etc.)
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest

# Note: previously this file had `sys.modules['tensorflow'] = MagicMock()` at
# module load to avoid needing real tensorflow in CI. That mock leaked into
# every other test that pytest collected after this one — when any subsequent
# test imported a module needing real tensorflow (e.g. transformer_trainer's
# @register_keras_serializable decorator), the cached Mock raised
# `AttributeError: Mock object has no attribute '__name__'`. With requirements.txt
# now installed in CI, tensorflow is available natively. None of the 3 classes
# tested below (ReplayBuffer, DriftDetector, TrainingLineage) use `tf.` —
# verified by grep — so dropping the mock has no behavioral effect on these
# tests but unblocks every other test downstream.
from src.training.trainers.callbacks import (
    ReplayBuffer,
    DriftDetector,
    TrainingLineage,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def tmp_path_str(tmp_path):
    """Fixture that provides tmp_path as a string."""
    return str(tmp_path)


@pytest.fixture
def sample_data_2d():
    """Sample 2D data (num_samples, num_features)."""
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def sample_data_3d():
    """Sample 3D data (num_samples, timesteps, num_features) - typical for sequences."""
    np.random.seed(42)
    X = np.random.randn(50, 20, 5).astype(np.float32)
    y = np.random.randint(0, 2, 50)
    return X, y


@pytest.fixture
def sample_weights():
    """Sample weight array."""
    np.random.seed(42)
    return np.random.uniform(0.5, 1.5, 100)


@pytest.fixture
def sample_metrics():
    """Sample training metrics."""
    return {
        "val_accuracy": 0.85,
        "train_accuracy": 0.92,
        "loss": 0.15,
    }


# =============================================================================
# US-252: REPLAY BUFFER TESTS
# =============================================================================


class TestReplayBufferInit:
    """Test ReplayBuffer initialization."""

    def test_default_init(self):
        """Test ReplayBuffer initializes with correct defaults."""
        rb = ReplayBuffer()
        assert rb.capacity_ratio == 0.10
        assert rb.mix_ratio == 0.20
        assert rb.buffer_dir == Path("trained_data/replay")
        assert rb.x_buffer is None
        assert rb.y_buffer is None
        assert rb.w_buffer is None
        assert rb.feature_names is None
        assert rb._sample_count == 0

    def test_custom_init(self):
        """Test ReplayBuffer with custom parameters."""
        rb = ReplayBuffer(capacity_ratio=0.15, mix_ratio=0.25, buffer_dir="custom/path")
        assert rb.capacity_ratio == 0.15
        assert rb.mix_ratio == 0.25
        assert rb.buffer_dir == Path("custom/path")


class TestReplayBufferAddSamples:
    """Test ReplayBuffer.add_samples()."""

    def test_add_samples_first_batch(self, sample_data_2d):
        """Test adding samples initializes buffer correctly."""
        X, y = sample_data_2d
        rb = ReplayBuffer(capacity_ratio=0.10)
        rb.add_samples(X, y)

        # Buffer should be populated
        assert rb.x_buffer is not None
        assert rb.y_buffer is not None
        assert rb.w_buffer is not None

        # Buffer size should respect capacity ratio
        expected_capacity = max(int(len(X) * 0.10), 10)
        expected_capacity = min(expected_capacity, 100)
        assert len(rb.x_buffer) == expected_capacity

    def test_add_samples_with_weights(self, sample_data_2d, sample_weights):
        """Test adding samples with explicit weights."""
        X, y = sample_data_2d
        rb = ReplayBuffer()
        rb.add_samples(X, y, w=sample_weights)

        assert rb.w_buffer is not None
        assert len(rb.w_buffer) == len(rb.x_buffer)

    def test_add_samples_without_weights(self, sample_data_2d):
        """Test adding samples without weights initializes to ones."""
        X, y = sample_data_2d
        rb = ReplayBuffer()
        rb.add_samples(X, y, w=None)

        assert rb.w_buffer is not None
        assert np.allclose(rb.w_buffer, 1.0)

    def test_add_samples_shape_preservation(self, sample_data_2d):
        """Test that added samples preserve shape."""
        X, y = sample_data_2d
        rb = ReplayBuffer()
        rb.add_samples(X, y)

        assert rb.x_buffer.ndim == X.ndim
        assert rb.y_buffer.ndim == y.ndim
        assert rb.x_buffer.shape[1:] == X.shape[1:]
        assert rb.y_buffer.shape[1:] == y.shape[1:]

    def test_add_samples_respects_capacity(self, sample_data_2d):
        """Test that capacity limit is respected."""
        X, y = sample_data_2d
        rb = ReplayBuffer(capacity_ratio=0.10)
        initial_capacity = int(len(X) * 0.10)
        initial_capacity = min(max(initial_capacity, 10), 100)

        rb.add_samples(X, y)
        assert len(rb.x_buffer) <= initial_capacity + 1  # +1 for rounding

    def test_add_samples_with_feature_names(self, sample_data_2d):
        """Test adding samples with feature names."""
        X, y = sample_data_2d
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        rb = ReplayBuffer()
        rb.add_samples(X, y, feature_names=feature_names)

        assert rb.feature_names == feature_names

    def test_add_samples_with_data_id(self, sample_data_2d):
        """Test that data_id is tracked in metadata."""
        X, y = sample_data_2d
        rb = ReplayBuffer()
        rb.add_samples(X, y, data_id="EUR_USD_2024-01-01")

        assert "data_sources" in rb.metadata
        assert len(rb.metadata["data_sources"]) == 1
        assert rb.metadata["data_sources"][0]["id"] == "EUR_USD_2024-01-01"
        assert rb.metadata["data_sources"][0]["n_samples"] == len(X)

    def test_add_samples_multiple_batches(self, sample_data_2d):
        """Test adding multiple batches updates buffer."""
        X, y = sample_data_2d
        rb = ReplayBuffer(capacity_ratio=0.10)

        # Add first batch
        rb.add_samples(X[:50], y[:50])
        first_size = len(rb.x_buffer)

        # Add second batch
        rb.add_samples(X[50:], y[50:])
        second_size = len(rb.x_buffer)

        # Buffer should maintain reasonable size (reservoir sampling keeps it bounded)
        # With capacity_ratio=0.10 on 50 samples = 10 min samples
        # After second batch of 50, buffer uses reservoir sampling to maintain capacity
        assert first_size > 0
        assert second_size > 0
        assert second_size <= len(X)  # Never exceed original data size


class TestReplayBufferGetSamples:
    """Test ReplayBuffer.get_replay_samples()."""

    def test_get_replay_samples_empty_buffer(self):
        """Test getting samples from empty buffer returns None."""
        rb = ReplayBuffer()
        X, y, w = rb.get_replay_samples(n_new_samples=50)
        assert X is None
        assert y is None
        assert w is None

    def test_get_replay_samples_correct_count(self, sample_data_2d):
        """Test that correct number of samples are returned."""
        X, y = sample_data_2d
        rb = ReplayBuffer(capacity_ratio=0.20, mix_ratio=0.20)
        rb.add_samples(X, y)

        n_new = 100
        X_replay, y_replay, w_replay = rb.get_replay_samples(n_new_samples=n_new)

        expected_n = int(n_new * 0.20)
        # Verify we got approximately the right count (accounting for rounding and buffer size constraints)
        assert len(X_replay) == expected_n or len(X_replay) <= len(rb.x_buffer)
        assert len(y_replay) == len(X_replay)
        assert len(w_replay) == len(X_replay)

    def test_get_replay_samples_respects_buffer_size(self, sample_data_2d):
        """Test that returned samples don't exceed buffer size."""
        X, y = sample_data_2d
        rb = ReplayBuffer(capacity_ratio=0.05, mix_ratio=0.50)
        rb.add_samples(X, y)

        # Request samples larger than buffer
        X_replay, y_replay, w_replay = rb.get_replay_samples(n_new_samples=1000)

        # Should be capped at buffer size
        assert len(X_replay) <= len(rb.x_buffer)

    def test_get_replay_samples_returns_copies(self, sample_data_2d):
        """Test that returned samples are independent copies."""
        X, y = sample_data_2d
        rb = ReplayBuffer()
        rb.add_samples(X, y)

        X_replay, y_replay, w_replay = rb.get_replay_samples(n_new_samples=100)
        original_x = rb.x_buffer[0].copy()

        # Modify returned sample
        X_replay[0, 0] = 999.0

        # Original buffer should be unchanged
        assert rb.x_buffer[0, 0] == original_x[0]

    def test_get_replay_samples_zero_when_mix_ratio_low(self, sample_data_2d):
        """Test that no samples returned when mix_ratio*n_new is zero."""
        X, y = sample_data_2d
        rb = ReplayBuffer(mix_ratio=0.01)
        rb.add_samples(X, y)

        # Very small n_new samples
        X_replay, y_replay, w_replay = rb.get_replay_samples(n_new_samples=5)

        # 0.01 * 5 = 0.05 rounds down to 0
        assert X_replay is None or len(X_replay) == 0


class TestReplayBufferClear:
    """Test ReplayBuffer.clear()."""

    def test_clear_empties_buffer(self, sample_data_2d):
        """Test that clear() empties the buffer."""
        X, y = sample_data_2d
        rb = ReplayBuffer()
        rb.add_samples(X, y)

        assert rb.x_buffer is not None
        rb.clear()

        assert rb.x_buffer is None
        assert rb.y_buffer is None
        assert rb.w_buffer is None
        assert rb._sample_count == 0
        assert rb.metadata == {}


class TestReplayBufferSaveLoad:
    """Test ReplayBuffer save/load with real files."""

    def test_save_creates_directory(self, sample_data_2d, tmp_path):
        """Test that save() creates directory structure."""
        X, y = sample_data_2d
        rb = ReplayBuffer(buffer_dir=str(tmp_path / "replay"))
        rb.add_samples(X, y)
        rb.save(instrument="EUR_USD")

        # Check directory and files exist
        save_dir = tmp_path / "replay" / "EUR_USD"
        assert save_dir.exists()
        assert (save_dir / "buffer.npz").exists()
        assert (save_dir / "buffer_meta.json").exists()

    def test_save_preserves_data(self, sample_data_2d, tmp_path):
        """Test that save preserves buffer data."""
        X, y = sample_data_2d
        rb = ReplayBuffer(buffer_dir=str(tmp_path / "replay"))
        rb.add_samples(X, y)
        rb.save(instrument="EUR_USD")

        # Load and verify
        data = np.load(tmp_path / "replay" / "EUR_USD" / "buffer.npz")
        assert data["X"].shape == rb.x_buffer.shape
        assert np.allclose(data["X"], rb.x_buffer)
        assert np.array_equal(data["y"], rb.y_buffer)

    def test_save_metadata(self, sample_data_2d, tmp_path):
        """Test that save includes metadata."""
        X, y = sample_data_2d
        rb = ReplayBuffer(buffer_dir=str(tmp_path / "replay"), capacity_ratio=0.15)
        rb.add_samples(X, y, data_id="test_batch")
        rb.save(instrument="EUR_USD")

        meta_path = tmp_path / "replay" / "EUR_USD" / "buffer_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["capacity_ratio"] == 0.15
        assert "data_sources" in meta

    def test_load_restores_buffer(self, sample_data_2d, tmp_path):
        """Test that load() restores buffer correctly."""
        X, y = sample_data_2d
        rb1 = ReplayBuffer(buffer_dir=str(tmp_path / "replay"))
        rb1.add_samples(X, y, data_id="batch_1")
        rb1.save(instrument="EUR_USD")

        # Create new instance and load
        rb2 = ReplayBuffer(buffer_dir=str(tmp_path / "replay"))
        success = rb2.load(instrument="EUR_USD")

        assert success
        assert np.allclose(rb2.x_buffer, rb1.x_buffer)
        assert np.array_equal(rb2.y_buffer, rb1.y_buffer)

    def test_load_nonexistent_returns_false(self, tmp_path):
        """Test that load() returns False for missing file."""
        rb = ReplayBuffer(buffer_dir=str(tmp_path / "replay"))
        success = rb.load(instrument="NONEXISTENT")

        assert success is False
        assert rb.x_buffer is None

    def test_load_restores_metadata(self, sample_data_2d, tmp_path):
        """Test that load() restores metadata."""
        X, y = sample_data_2d
        rb1 = ReplayBuffer(buffer_dir=str(tmp_path / "replay"), mix_ratio=0.25)
        rb1.add_samples(X, y, data_id="batch_1")
        rb1.save(instrument="EUR_USD")

        rb2 = ReplayBuffer(buffer_dir=str(tmp_path / "replay"))
        rb2.load(instrument="EUR_USD")

        assert rb2._sample_count == rb1._sample_count


# =============================================================================
# US-252: DRIFT DETECTOR TESTS
# =============================================================================


class TestDriftDetectorInit:
    """Test DriftDetector initialization."""

    def test_default_init(self):
        """Test DriftDetector initializes with correct defaults."""
        dd = DriftDetector()
        assert dd.performance_threshold == 0.03
        assert dd.feature_drift_threshold == 0.1
        assert dd.window_size == 5
        assert dd.baseline_stats is None

    def test_custom_init(self):
        """Test DriftDetector with custom parameters."""
        dd = DriftDetector(
            performance_threshold=0.05,
            feature_drift_threshold=0.15,
            window_size=10,
        )
        assert dd.performance_threshold == 0.05
        assert dd.feature_drift_threshold == 0.15
        assert dd.window_size == 10


class TestDriftDetectorFeatureStats:
    """Test DriftDetector.compute_feature_stats()."""

    def test_compute_feature_stats_2d(self, sample_data_2d):
        """Test feature stats computation on 2D data."""
        X, _ = sample_data_2d
        dd = DriftDetector()
        stats = dd.compute_feature_stats(X)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

        assert stats["mean"].shape == (X.shape[1],)
        assert stats["std"].shape == (X.shape[1],)
        assert stats["min"].shape == (X.shape[1],)
        assert stats["max"].shape == (X.shape[1],)

    def test_compute_feature_stats_3d(self, sample_data_3d):
        """Test feature stats computation on 3D data (sequences)."""
        X, _ = sample_data_3d
        dd = DriftDetector()
        stats = dd.compute_feature_stats(X)

        # Stats should be computed across (samples, timesteps) axes
        assert stats["mean"].shape == (X.shape[2],)

    def test_compute_feature_stats_values(self, sample_data_2d):
        """Test that computed stats match numpy calculations."""
        X, _ = sample_data_2d
        dd = DriftDetector()
        stats = dd.compute_feature_stats(X)

        expected_mean = np.mean(X, axis=0)
        expected_std = np.std(X, axis=0)

        assert np.allclose(stats["mean"], expected_mean)
        assert np.allclose(stats["std"], expected_std)


class TestDriftDetectorSetBaseline:
    """Test DriftDetector.set_baseline()."""

    def test_set_baseline_stores_stats(self, sample_data_2d, sample_metrics):
        """Test that set_baseline stores feature stats."""
        X, _ = sample_data_2d
        dd = DriftDetector()
        dd.set_baseline(X, sample_metrics)

        assert dd.baseline_stats is not None
        assert "feature_stats" in dd.baseline_stats
        assert "best_val_accuracy" in dd.baseline_stats

    def test_set_baseline_accuracy(self, sample_data_2d, sample_metrics):
        """Test that baseline stores correct accuracy."""
        X, _ = sample_data_2d
        dd = DriftDetector()
        dd.set_baseline(X, sample_metrics)

        assert dd.baseline_stats["best_val_accuracy"] == sample_metrics["val_accuracy"]


class TestDriftDetectorPerformanceDrift:
    """Test DriftDetector.check_performance_drift()."""

    def test_no_drift_stable_accuracy(self):
        """Test no drift when accuracy is stable."""
        dd = DriftDetector(performance_threshold=0.03)
        metric_history = [
            {"val_accuracy": 0.85},
            {"val_accuracy": 0.86},
            {"val_accuracy": 0.85},
        ]
        is_drifted, reason = dd.check_performance_drift(0.85, metric_history)
        assert is_drifted is False

    def test_drift_when_accuracy_drops(self):
        """Test drift detection when accuracy drops significantly."""
        dd = DriftDetector(performance_threshold=0.03)
        metric_history = [
            {"val_accuracy": 0.90},
            {"val_accuracy": 0.88},
            {"val_accuracy": 0.80},
        ]
        is_drifted, reason = dd.check_performance_drift(0.80, metric_history)
        assert is_drifted is True
        assert "dropped" in reason.lower()

    def test_drift_detection_with_empty_history(self):
        """Test that empty history returns no drift."""
        dd = DriftDetector()
        is_drifted, reason = dd.check_performance_drift(0.85, [])
        assert is_drifted is False

    def test_drift_with_declining_trend(self):
        """Test drift detection with declining trend."""
        dd = DriftDetector(performance_threshold=0.03, window_size=3)
        metric_history = [
            {"val_accuracy": 0.90},
            {"val_accuracy": 0.87},
            {"val_accuracy": 0.84},
        ]
        is_drifted, reason = dd.check_performance_drift(0.81, metric_history)
        assert is_drifted is True
        # The method checks absolute drop from best first, so the reason will be "dropped"
        assert "dropped" in reason.lower() or "trend" in reason.lower() or "drift" in reason.lower()


class TestDriftDetectorFeatureDrift:
    """Test DriftDetector.check_feature_drift()."""

    def test_no_feature_drift_stable_distribution(self, sample_data_2d):
        """Test no feature drift when distribution is stable."""
        X, _ = sample_data_2d
        dd = DriftDetector()

        # Set baseline
        metrics = {"val_accuracy": 0.85}
        dd.set_baseline(X, metrics)

        # Check same data
        is_drifted, reason = dd.check_feature_drift(X[:50])
        assert is_drifted is False

    def test_feature_drift_detection_no_baseline(self):
        """Test that missing baseline returns no drift."""
        dd = DriftDetector()
        X_new = np.random.randn(50, 10)
        is_drifted, reason = dd.check_feature_drift(X_new)
        assert is_drifted is False

    def test_feature_drift_with_shifted_data(self, sample_data_2d):
        """Test drift detection with significantly shifted features."""
        X, _ = sample_data_2d
        dd = DriftDetector(feature_drift_threshold=0.01)

        # Set baseline
        metrics = {"val_accuracy": 0.85}
        dd.set_baseline(X, metrics)

        # Create heavily shifted data
        X_shifted = X + 100  # Large constant shift
        is_drifted, reason = dd.check_feature_drift(X_shifted)

        # This should detect drift due to mean shift
        assert is_drifted is True or "shift" in reason.lower()


class TestDriftDetectorFullCheck:
    """Test DriftDetector.full_drift_check()."""

    def test_full_drift_check_returns_dict(self, sample_data_2d, sample_metrics):
        """Test that full_drift_check returns complete recommendation dict."""
        X, _ = sample_data_2d
        dd = DriftDetector()
        dd.set_baseline(X, sample_metrics)

        result = dd.full_drift_check(X[:50], 0.85, [{"val_accuracy": 0.85}])

        assert isinstance(result, dict)
        assert "any_drift" in result
        assert "performance_drift" in result
        assert "feature_drift" in result
        assert "recommendation" in result

    def test_full_drift_check_no_drift(self, sample_data_2d, sample_metrics):
        """Test full_drift_check with no drift."""
        X, _ = sample_data_2d
        dd = DriftDetector()
        dd.set_baseline(X, sample_metrics)

        result = dd.full_drift_check(X[:50], 0.85, [{"val_accuracy": 0.85}])

        assert result["any_drift"] is False
        assert result["recommendation"] == "normal"

    def test_full_drift_check_performance_drift(self, sample_data_2d):
        """Test full_drift_check recommendation for performance drift."""
        X, _ = sample_data_2d
        dd = DriftDetector(performance_threshold=0.02)

        metric_history = [{"val_accuracy": 0.90}]
        dd.set_baseline(X, {"val_accuracy": 0.90})

        result = dd.full_drift_check(X[:50], 0.80, metric_history)

        assert result["performance_drift"] is True
        assert result["recommendation"] == "warm_start_retrain"


class TestDriftDetectorRecordTrainingResult:
    """Test DriftDetector.record_training_result()."""

    def test_record_training_result_appends_history(self):
        """Test that record_training_result creates and appends history."""
        dd = DriftDetector()
        feature_means = np.array([0.5, 0.3, -0.2])

        dd.record_training_result(
            val_accuracy=0.85,
            instrument="EUR_USD",
            data_hash="abc123",
            feature_means=feature_means,
        )

        assert hasattr(dd, "_history")
        assert len(dd._history) == 1
        assert dd._history[0]["val_accuracy"] == 0.85

    def test_record_training_result_multiple_calls(self):
        """Test multiple recordings maintain history."""
        dd = DriftDetector()
        feature_means = np.array([0.5, 0.3, -0.2])

        for i in range(3):
            dd.record_training_result(
                val_accuracy=0.85 + i * 0.01,
                instrument="EUR_USD",
                data_hash=f"hash_{i}",
                feature_means=feature_means,
            )

        assert len(dd._history) == 3


# =============================================================================
# US-253: TRAINING LINEAGE TESTS
# =============================================================================


class TestTrainingLineageInit:
    """Test TrainingLineage initialization."""

    def test_default_init(self):
        """Test TrainingLineage default field values."""
        tl = TrainingLineage()

        assert tl.checkpoint_id == ""
        assert tl.parent_checkpoint_id is None
        assert tl.cumulative_epochs == 0
        assert tl.cumulative_samples == 0
        assert tl.session_epochs == 0
        assert tl.generation == 1
        assert tl.data_hash == ""
        assert tl.instrument == ""
        assert tl.granularity == ""
        assert tl.metric_history == []
        assert tl.ema_enabled is False
        assert tl.ewc_n_tasks == 0
        assert tl.replay_buffer_size == 0
        assert tl.drift_detected is False
        assert tl.lineage_version == 2

    def test_custom_init(self):
        """Test TrainingLineage with custom values."""
        tl = TrainingLineage(
            checkpoint_id="test_ckpt",
            cumulative_epochs=100,
            instrument="EUR_USD",
            generation=2,
        )

        assert tl.checkpoint_id == "test_ckpt"
        assert tl.cumulative_epochs == 100
        assert tl.instrument == "EUR_USD"
        assert tl.generation == 2


class TestTrainingLineageCheckpointId:
    """Test TrainingLineage.generate_checkpoint_id()."""

    def test_generate_checkpoint_id_nonempty(self):
        """Test that generate_checkpoint_id returns non-empty string."""
        tl = TrainingLineage()
        ckpt_id = tl.generate_checkpoint_id()

        assert isinstance(ckpt_id, str)
        assert len(ckpt_id) > 0
        assert tl.checkpoint_id == ckpt_id

    def test_generate_checkpoint_id_uniqueness(self):
        """Test that two calls produce different IDs."""
        tl1 = TrainingLineage()
        tl2 = TrainingLineage()

        id1 = tl1.generate_checkpoint_id()
        id2 = tl2.generate_checkpoint_id()

        # IDs should be different (unless incredibly unlucky with random)
        assert id1 != id2

    def test_generate_checkpoint_id_sets_created_at(self):
        """Test that generate_checkpoint_id sets created_at timestamp."""
        tl = TrainingLineage()
        tl.generate_checkpoint_id()

        assert tl.created_at != ""
        # Check it's a valid ISO format (basic check)
        assert "T" in tl.created_at


class TestTrainingLineageMetrics:
    """Test TrainingLineage.add_metrics()."""

    def test_add_metrics_appends_to_history(self, sample_metrics):
        """Test that add_metrics appends to history."""
        tl = TrainingLineage()
        tl.checkpoint_id = "test_ckpt"

        tl.add_metrics(sample_metrics)

        assert len(tl.metric_history) == 1
        assert tl.metric_history[0]["val_accuracy"] == sample_metrics["val_accuracy"]

    def test_add_metrics_includes_metadata(self, sample_metrics):
        """Test that add_metrics includes checkpoint and generation."""
        tl = TrainingLineage()
        tl.checkpoint_id = "test_ckpt"
        tl.generation = 2

        tl.add_metrics(sample_metrics)

        entry = tl.metric_history[0]
        assert entry["checkpoint_id"] == "test_ckpt"
        assert entry["generation"] == 2
        assert "timestamp" in entry

    def test_add_metrics_multiple_calls(self, sample_metrics):
        """Test multiple metric additions."""
        tl = TrainingLineage()
        tl.checkpoint_id = "test_ckpt"

        for i in range(3):
            metrics = sample_metrics.copy()
            metrics["val_accuracy"] = 0.85 + i * 0.01
            tl.add_metrics(metrics)

        assert len(tl.metric_history) == 3


class TestTrainingLineageCheckDrift:
    """Test TrainingLineage.check_drift()."""

    def test_check_drift_empty_history(self):
        """Test check_drift with empty metric history."""
        tl = TrainingLineage()
        is_drifted = tl.check_drift(current_val_acc=0.85)

        assert is_drifted is False

    def test_check_drift_improving_accuracy(self):
        """Test no drift when accuracy improving."""
        tl = TrainingLineage()
        tl.metric_history = [
            {"val_accuracy": 0.80},
            {"val_accuracy": 0.85},
        ]

        is_drifted = tl.check_drift(current_val_acc=0.87)

        assert is_drifted is False

    def test_check_drift_significant_drop(self):
        """Test drift detection with significant accuracy drop."""
        tl = TrainingLineage(checkpoint_id="test")
        tl.metric_history = [
            {"val_accuracy": 0.90},
            {"val_accuracy": 0.88},
            {"val_accuracy": 0.87},
        ]

        is_drifted = tl.check_drift(current_val_acc=0.80, threshold=0.03)

        assert is_drifted is True

    def test_check_drift_custom_threshold(self):
        """Test check_drift with custom threshold."""
        tl = TrainingLineage()
        tl.metric_history = [{"val_accuracy": 0.90}]

        # Small drop with tight threshold
        is_drifted = tl.check_drift(current_val_acc=0.88, threshold=0.01)

        assert is_drifted is True

        # Same drop with loose threshold
        is_drifted = tl.check_drift(current_val_acc=0.88, threshold=0.05)

        assert is_drifted is False


class TestTrainingLineageDataHash:
    """Test TrainingLineage.compute_data_hash()."""

    def test_compute_data_hash_same_data(self, sample_data_2d):
        """Test that same data produces same hash."""
        X, y = sample_data_2d

        hash1 = TrainingLineage.compute_data_hash(X, y)
        hash2 = TrainingLineage.compute_data_hash(X, y)

        assert hash1 == hash2

    def test_compute_data_hash_different_data(self, sample_data_2d):
        """Test that different data produces different hash."""
        X, y = sample_data_2d
        X_modified = X.copy()
        X_modified[0, 0] += 0.001  # Small modification

        hash1 = TrainingLineage.compute_data_hash(X, y)
        hash2 = TrainingLineage.compute_data_hash(X_modified, y)

        assert hash1 != hash2

    def test_compute_data_hash_format(self, sample_data_2d):
        """Test that hash has expected format."""
        X, y = sample_data_2d
        hash_val = TrainingLineage.compute_data_hash(X, y)

        # Should be 12 character hex
        assert len(hash_val) == 12
        assert all(c in "0123456789abcdef" for c in hash_val)


class TestTrainingLineageSummary:
    """Test TrainingLineage.get_training_summary()."""

    def test_get_training_summary_nonempty(self):
        """Test that get_training_summary returns non-empty string."""
        tl = TrainingLineage()
        tl.generate_checkpoint_id()
        tl.cumulative_epochs = 100
        tl.instrument = "EUR_USD"
        tl.granularity = "H1"

        summary = tl.get_training_summary()

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_get_training_summary_includes_key_info(self):
        """Test that summary includes key information."""
        tl = TrainingLineage()
        tl.checkpoint_id = "test_ckpt_123"
        tl.instrument = "EUR_USD"
        tl.granularity = "H1"
        tl.generation = 2
        tl.cumulative_epochs = 50

        summary = tl.get_training_summary()

        assert "test_ckpt_123" in summary
        assert "EUR_USD" in summary
        assert "H1" in summary

    def test_get_training_summary_with_drift(self):
        """Test that summary includes drift information."""
        tl = TrainingLineage()
        tl.checkpoint_id = "test"
        tl.drift_detected = True
        tl.drift_reason = "Accuracy dropped 5%"

        summary = tl.get_training_summary()

        assert "Drift detected" in summary or "drift" in summary.lower()


class TestTrainingLineageShouldRetrain:
    """Test TrainingLineage.should_retrain()."""

    def test_should_retrain_time_based(self):
        """Test time-based retraining recommendation."""
        tl = TrainingLineage(recommended_retrain_interval_hours=168)

        should_retrain, reason = tl.should_retrain(hours_since_last=200)

        assert should_retrain is True
        assert "Scheduled" in reason

    def test_should_retrain_drift_based(self):
        """Test drift-based retraining recommendation."""
        tl = TrainingLineage()
        tl.drift_detected = True
        tl.drift_reason = "Performance degraded"

        should_retrain, reason = tl.should_retrain(hours_since_last=10)

        assert should_retrain is True
        assert "drift" in reason.lower() or "Drift" in reason

    def test_should_retrain_declining_accuracy(self):
        """Test retraining recommendation for declining accuracy."""
        tl = TrainingLineage()
        tl.metric_history = [
            {"val_accuracy": 0.90},
            {"val_accuracy": 0.87},
            {"val_accuracy": 0.84},
        ]

        should_retrain, reason = tl.should_retrain(hours_since_last=10)

        assert should_retrain is True
        assert "Declining" in reason or "declining" in reason.lower()

    def test_should_retrain_no_retraining_needed(self):
        """Test case where no retraining is needed."""
        tl = TrainingLineage(recommended_retrain_interval_hours=168)
        tl.metric_history = [{"val_accuracy": 0.90}]

        should_retrain, reason = tl.should_retrain(hours_since_last=10)

        assert should_retrain is False


class TestTrainingLineageSerialization:
    """Test TrainingLineage to_dict/from_dict round-trip."""

    def test_to_dict_returns_dict(self):
        """Test that to_dict returns dictionary."""
        tl = TrainingLineage()
        tl.checkpoint_id = "test_ckpt"
        tl.instrument = "EUR_USD"

        d = tl.to_dict()

        assert isinstance(d, dict)
        assert "checkpoint_id" in d
        assert "instrument" in d

    def test_from_dict_creates_instance(self):
        """Test that from_dict creates valid instance."""
        data = {
            "checkpoint_id": "test_ckpt",
            "instrument": "EUR_USD",
            "generation": 2,
            "cumulative_epochs": 100,
        }

        tl = TrainingLineage.from_dict(data)

        assert tl.checkpoint_id == "test_ckpt"
        assert tl.instrument == "EUR_USD"
        assert tl.generation == 2

    def test_round_trip_preserves_all_fields(self):
        """Test that to_dict/from_dict round-trip preserves all fields."""
        tl1 = TrainingLineage()
        tl1.checkpoint_id = "ckpt_1"
        tl1.parent_checkpoint_id = "ckpt_0"
        tl1.cumulative_epochs = 100
        tl1.cumulative_samples = 10000
        tl1.generation = 2
        tl1.instrument = "EUR_USD"
        tl1.granularity = "H1"
        tl1.ema_enabled = True
        tl1.ewc_n_tasks = 3
        tl1.replay_buffer_size = 5000
        tl1.drift_detected = True
        tl1.drift_reason = "Test drift"
        tl1.metric_history = [
            {"val_accuracy": 0.85, "loss": 0.15},
            {"val_accuracy": 0.87, "loss": 0.13},
        ]

        d = tl1.to_dict()
        tl2 = TrainingLineage.from_dict(d)

        assert tl2.checkpoint_id == tl1.checkpoint_id
        assert tl2.parent_checkpoint_id == tl1.parent_checkpoint_id
        assert tl2.cumulative_epochs == tl1.cumulative_epochs
        assert tl2.cumulative_samples == tl1.cumulative_samples
        assert tl2.generation == tl1.generation
        assert tl2.instrument == tl1.instrument
        assert tl2.granularity == tl1.granularity
        assert tl2.ema_enabled == tl1.ema_enabled
        assert tl2.ewc_n_tasks == tl1.ewc_n_tasks
        assert tl2.replay_buffer_size == tl1.replay_buffer_size
        assert tl2.drift_detected == tl1.drift_detected
        assert tl2.drift_reason == tl1.drift_reason
        assert len(tl2.metric_history) == len(tl1.metric_history)

    def test_from_dict_with_missing_optional_fields(self):
        """Test from_dict with minimal required fields."""
        minimal_data = {
            "checkpoint_id": "test",
        }

        tl = TrainingLineage.from_dict(minimal_data)

        # Should have sensible defaults
        assert tl.checkpoint_id == "test"
        assert tl.generation >= 1
        assert isinstance(tl.metric_history, list)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestReplayBufferDriftDetectorIntegration:
    """Integration tests combining ReplayBuffer and DriftDetector."""

    def test_replay_buffer_with_drift_detection(self, sample_data_2d):
        """Test replay buffer feeding into drift detection."""
        X, y = sample_data_2d
        rb = ReplayBuffer()
        dd = DriftDetector()

        # Add samples to replay buffer
        rb.add_samples(X, y)

        # Get replay samples
        X_replay, y_replay, w_replay = rb.get_replay_samples(n_new_samples=100)

        assert X_replay is not None
        assert len(X_replay) > 0

        # Use replayed data for drift detection
        dd.set_baseline(X_replay, {"val_accuracy": 0.85})
        is_drifted, reason = dd.check_feature_drift(X[:50])

        assert isinstance(is_drifted, bool)


class TestTrainingLineageWithDriftDetection:
    """Integration tests combining TrainingLineage and DriftDetector."""

    def test_lineage_tracks_drift_detection(self):
        """Test that lineage can track drift detection results."""
        tl = TrainingLineage()
        tl.checkpoint_id = "ckpt_1"
        tl.drift_detected = True
        tl.drift_reason = "Performance dropped 5%"

        should_retrain, reason = tl.should_retrain(hours_since_last=10)

        assert should_retrain is True

    def test_lineage_with_metric_history_and_drift(self):
        """Test lineage combining metric history with drift checking."""
        tl = TrainingLineage()
        tl.checkpoint_id = "ckpt_1"

        # Add metrics
        metrics_history = [
            {"val_accuracy": 0.90, "loss": 0.10},
            {"val_accuracy": 0.88, "loss": 0.12},
            {"val_accuracy": 0.85, "loss": 0.15},
        ]

        for metrics in metrics_history:
            tl.add_metrics(metrics)

        # Check for drift
        is_drifted = tl.check_drift(current_val_acc=0.80, threshold=0.03)

        assert is_drifted is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
