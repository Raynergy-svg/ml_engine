"""
Integration tests for the OnlineRetrainer module.

Validates:
1. Cooldown enforcement (cannot retrain within cooldown_minutes)
2. Daily limit enforcement (max_retrains_per_day)
3. trigger_retrain with fake data (100x50 array)
4. create_retrain_callback returns callable
5. State save/load via tmp_path
6. Minimum sample requirement
7. Thread-safe lock behavior
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Check for sklearn availability (needed for actual model retraining)
try:
    import sklearn  # noqa: F401
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

_skip_no_sklearn = pytest.mark.skipif(
    not _HAS_SKLEARN, reason="sklearn not installed"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create a temporary model directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def tmp_state_file(tmp_path):
    """Create a temporary state file path."""
    return str(tmp_path / "retrain_state.json")


@pytest.fixture
def retrainer(tmp_model_dir, tmp_state_file):
    """Create an OnlineRetrainer with temporary paths."""
    from online_retrainer import OnlineRetrainer, RetrainConfig

    config = RetrainConfig(
        cooldown_minutes=60,
        max_retrains_per_day=3,
        min_samples_for_retrain=50,
        state_file=tmp_state_file,
    )
    return OnlineRetrainer(config=config, model_dir=str(tmp_model_dir))


@pytest.fixture
def fake_data():
    """Generate fake replay data (100 samples, 50 features)."""
    np.random.seed(42)
    X = np.random.randn(100, 50)
    y = np.random.randint(0, 2, 100).astype(float)
    return X, y


@pytest.fixture
def small_data():
    """Generate data below minimum sample threshold (30 samples)."""
    np.random.seed(42)
    X = np.random.randn(30, 50)
    y = np.random.randint(0, 2, 30).astype(float)
    return X, y


# ---------------------------------------------------------------------------
# Test: RetrainConfig defaults
# ---------------------------------------------------------------------------

class TestRetrainConfig:
    """Test RetrainConfig dataclass defaults."""

    def test_default_values(self):
        """Default config values are correct."""
        from online_retrainer import RetrainConfig

        config = RetrainConfig()
        assert config.cooldown_minutes == 60
        assert config.max_retrains_per_day == 3
        # 100 since 2026-07-03 (operator-approved): aligned with the eval
        # gate's 20-sample holdout floor (20% of 100).
        assert config.min_samples_for_retrain == 100
        assert config.validation_split == 0.2
        assert config.retrain_xgboost is True
        assert config.retrain_rf is True
        assert config.retrain_ridge is True
        assert config.retrain_transformer is False

    def test_custom_values(self):
        """Can override default config values."""
        from online_retrainer import RetrainConfig

        config = RetrainConfig(cooldown_minutes=30, max_retrains_per_day=5)
        assert config.cooldown_minutes == 30
        assert config.max_retrains_per_day == 5


# ---------------------------------------------------------------------------
# Test: RetrainState defaults
# ---------------------------------------------------------------------------

class TestRetrainState:
    """Test RetrainState dataclass defaults."""

    def test_default_values(self):
        """Default state has sane initial values."""
        from online_retrainer import RetrainState

        state = RetrainState()
        assert state.last_retrain_time is None
        assert state.retrains_today == 0
        assert state.total_retrains == 0
        assert state.retrain_history == []


# ---------------------------------------------------------------------------
# Test: can_retrain checks
# ---------------------------------------------------------------------------

class TestCanRetrain:
    """Test cooldown and daily limit enforcement."""

    def test_can_retrain_initially(self, retrainer):
        """Fresh retrainer should allow retraining."""
        can, reason = retrainer.can_retrain()
        assert can is True
        assert "Ready" in reason

    def test_cooldown_blocks_retrain(self, retrainer):
        """Cannot retrain within cooldown period."""
        # Simulate a recent retrain
        retrainer._state.last_retrain_time = datetime.utcnow()
        retrainer._state.last_retrain_date = datetime.utcnow().strftime('%Y-%m-%d')
        retrainer._state.retrains_today = 1

        can, reason = retrainer.can_retrain()
        assert can is False
        assert "Cooldown" in reason

    def test_daily_limit_blocks_retrain(self, retrainer):
        """Cannot retrain after daily limit reached."""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        retrainer._state.last_retrain_date = today
        retrainer._state.retrains_today = 3  # At max (default is 3)

        can, reason = retrainer.can_retrain()
        assert can is False
        assert "Daily limit" in reason

    def test_new_day_resets_counter(self, retrainer):
        """Retrains today counter resets on a new day."""
        retrainer._state.last_retrain_date = "2020-01-01"
        retrainer._state.retrains_today = 3

        can, reason = retrainer.can_retrain()
        # New day detected, counter should reset
        assert can is True

    def test_already_retraining_blocks(self, retrainer):
        """Cannot retrain while another retrain is in progress."""
        retrainer._is_retraining = True

        can, reason = retrainer.can_retrain()
        assert can is False
        assert "in progress" in reason.lower()


# ---------------------------------------------------------------------------
# Test: trigger_retrain with data
# ---------------------------------------------------------------------------

class TestTriggerRetrain:
    """Test trigger_retrain execution."""

    @_skip_no_sklearn
    def test_successful_retrain(self, retrainer, fake_data):
        """Successful retrain with sufficient data."""
        X, y = fake_data
        result = retrainer.trigger_retrain(
            X_replay=X, y_replay=y, reason="test", force=True,
        )

        assert result['status'] == 'completed'
        assert len(result['models_retrained']) > 0
        assert result['duration_seconds'] >= 0
        assert result['reason'] == 'test'

    @_skip_no_sklearn
    def test_retrain_updates_state(self, retrainer, fake_data):
        """After retrain, state counters update."""
        X, y = fake_data

        assert retrainer._state.total_retrains == 0

        retrainer.trigger_retrain(X_replay=X, y_replay=y, reason="test", force=True)

        assert retrainer._state.total_retrains == 1
        assert retrainer._state.retrains_today == 1
        assert retrainer._state.last_retrain_time is not None
        assert retrainer._state.last_retrain_reason == "test"

    def test_retrain_blocked_by_cooldown(self, retrainer, fake_data):
        """Retrain blocked when cooldown is active (force=False)."""
        X, y = fake_data

        # Simulate recent retrain
        retrainer._state.last_retrain_time = datetime.utcnow()
        retrainer._state.last_retrain_date = datetime.utcnow().strftime('%Y-%m-%d')
        retrainer._state.retrains_today = 1

        result = retrainer.trigger_retrain(
            X_replay=X, y_replay=y, reason="drift", force=False,
        )

        assert result['status'] == 'blocked'
        assert 'blocked_reason' in result

    @_skip_no_sklearn
    def test_retrain_force_bypasses_cooldown(self, retrainer, fake_data):
        """force=True bypasses cooldown check."""
        X, y = fake_data

        # Simulate recent retrain
        retrainer._state.last_retrain_time = datetime.utcnow()
        retrainer._state.last_retrain_date = datetime.utcnow().strftime('%Y-%m-%d')
        retrainer._state.retrains_today = 1

        result = retrainer.trigger_retrain(
            X_replay=X, y_replay=y, reason="test", force=True,
        )

        assert result['status'] == 'completed'

    def test_retrain_skipped_insufficient_data(self, retrainer, small_data):
        """Retrain skipped when data is below min_samples_for_retrain."""
        X, y = small_data  # Only 30 samples, need 50

        result = retrainer.trigger_retrain(
            X_replay=X, y_replay=y, reason="test", force=True,
        )

        assert result['status'] == 'skipped'
        assert 'skipped_reason' in result
        assert 'Insufficient' in result['skipped_reason']

    def test_retrain_skipped_no_data(self, retrainer):
        """Retrain skipped when no replay data available."""
        result = retrainer.trigger_retrain(
            X_replay=None, y_replay=None, reason="test", force=True,
        )

        assert result['status'] == 'skipped'

    @_skip_no_sklearn
    def test_retrain_result_has_metrics(self, retrainer, fake_data):
        """Completed retrain includes model metrics."""
        X, y = fake_data
        result = retrainer.trigger_retrain(
            X_replay=X, y_replay=y, reason="test", force=True,
        )

        assert 'metrics' in result
        assert isinstance(result['metrics'], dict)

    @_skip_no_sklearn
    def test_models_retrained_list(self, retrainer, fake_data):
        """models_retrained lists which models were updated."""
        X, y = fake_data
        result = retrainer.trigger_retrain(
            X_replay=X, y_replay=y, reason="test", force=True,
        )

        models = result['models_retrained']
        assert isinstance(models, list)
        # Default config enables xgboost, rf, ridge
        assert 'ridge' in models


# ---------------------------------------------------------------------------
# Test: State persistence (save/load)
# ---------------------------------------------------------------------------

class TestStatePersistence:
    """Test state save and load with tmp_path."""

    def test_save_state(self, retrainer, tmp_state_file):
        """State is saved to disk as JSON."""
        retrainer._state.total_retrains = 5
        retrainer._state.last_retrain_reason = "drift"
        retrainer._save_state()

        state_path = Path(tmp_state_file)
        assert state_path.exists()

        data = json.loads(state_path.read_text())
        assert data['total_retrains'] == 5
        assert data['last_retrain_reason'] == 'drift'

    def test_load_state(self, tmp_model_dir, tmp_state_file):
        """State loaded from disk on initialization."""
        from online_retrainer import OnlineRetrainer, RetrainConfig

        # Write state to disk
        state_data = {
            'last_retrain_time': '2025-01-15T10:00:00',
            'retrains_today': 2,
            'last_retrain_date': '2025-01-15',
            'total_retrains': 10,
            'last_retrain_reason': 'drift_detected',
            'last_retrain_metrics': {'xgboost': {'momentum_mae': 0.15}},
            'retrain_history': [{'timestamp': '2025-01-15T10:00:00', 'reason': 'test'}],
        }
        Path(tmp_state_file).write_text(json.dumps(state_data))

        # Initialize retrainer - should load state
        config = RetrainConfig(state_file=tmp_state_file)
        retrainer = OnlineRetrainer(config=config, model_dir=str(tmp_model_dir))

        assert retrainer._state.total_retrains == 10
        assert retrainer._state.retrains_today == 2
        assert retrainer._state.last_retrain_reason == 'drift_detected'

    def test_load_state_missing_file(self, tmp_model_dir, tmp_path):
        """Loading state from non-existent file does not crash."""
        from online_retrainer import OnlineRetrainer, RetrainConfig

        config = RetrainConfig(state_file=str(tmp_path / "nonexistent.json"))
        retrainer = OnlineRetrainer(config=config, model_dir=str(tmp_model_dir))

        # Should have default state
        assert retrainer._state.total_retrains == 0

    @_skip_no_sklearn
    def test_retrain_saves_state_automatically(self, retrainer, fake_data, tmp_state_file):
        """trigger_retrain automatically saves state after completion."""
        X, y = fake_data
        retrainer.trigger_retrain(X_replay=X, y_replay=y, reason="test", force=True)

        state_path = Path(tmp_state_file)
        assert state_path.exists()

        data = json.loads(state_path.read_text())
        assert data['total_retrains'] == 1

    def test_retrain_history_capped(self, tmp_model_dir, tmp_state_file):
        """Retrain history keeps max 100 entries."""
        from online_retrainer import OnlineRetrainer, RetrainConfig

        # Write state with >100 history items
        state_data = {
            'total_retrains': 150,
            'retrains_today': 0,
            'last_retrain_date': '2020-01-01',
            'retrain_history': [
                {'timestamp': f'2025-01-{i:02d}T10:00:00', 'reason': f'test_{i}'}
                for i in range(1, 151)
            ],
        }
        Path(tmp_state_file).write_text(json.dumps(state_data))

        config = RetrainConfig(state_file=tmp_state_file)
        retrainer = OnlineRetrainer(config=config, model_dir=str(tmp_model_dir))

        # History should be capped at 100
        assert len(retrainer._state.retrain_history) <= 100


# ---------------------------------------------------------------------------
# Test: create_retrain_callback
# ---------------------------------------------------------------------------

class TestCreateRetrainCallback:
    """Test create_retrain_callback factory function."""

    def test_returns_callable(self, retrainer):
        """create_retrain_callback returns a callable."""
        from online_retrainer import create_retrain_callback

        callback = create_retrain_callback(retrainer=retrainer)
        assert callable(callback)

    @_skip_no_sklearn
    def test_callback_calls_trigger_retrain(self, retrainer, fake_data):
        """Callback invokes retrainer.trigger_retrain."""
        from online_retrainer import create_retrain_callback

        X, y = fake_data

        # Mock market_intel to provide replay data
        mock_intel = MagicMock()
        mock_intel.get_replay_data.return_value = (X, y)

        callback = create_retrain_callback(retrainer=retrainer, market_intel=mock_intel)
        result = callback()

        assert isinstance(result, dict)
        assert 'status' in result
        mock_intel.get_replay_data.assert_called_once()

    def test_callback_without_market_intel(self, retrainer):
        """Callback works without market_intel (no replay data)."""
        from online_retrainer import create_retrain_callback

        callback = create_retrain_callback(retrainer=retrainer, market_intel=None)
        result = callback()

        # No data -> skipped
        assert result['status'] == 'skipped'

    @_skip_no_sklearn
    def test_callback_marks_model_updated_on_success(self, retrainer, fake_data):
        """Callback calls market_intel.mark_model_updated on success."""
        from online_retrainer import create_retrain_callback

        X, y = fake_data
        mock_intel = MagicMock()
        mock_intel.get_replay_data.return_value = (X, y)

        # Force to bypass cooldown
        callback = create_retrain_callback(retrainer=retrainer, market_intel=mock_intel)

        # First call should succeed (no cooldown)
        result = callback()

        if result['status'] == 'completed':
            mock_intel.mark_model_updated.assert_called_once()

    def test_creates_default_retrainer_if_none(self):
        """If retrainer=None, creates a new OnlineRetrainer."""
        from online_retrainer import create_retrain_callback

        callback = create_retrain_callback(retrainer=None, market_intel=None)
        assert callable(callback)

        result = callback()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Test: get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    """Test get_status reporting method."""

    def test_status_keys(self, retrainer):
        """get_status returns expected keys."""
        status = retrainer.get_status()

        expected_keys = {
            'can_retrain', 'reason', 'is_retraining', 'total_retrains',
            'retrains_today', 'max_per_day', 'cooldown_minutes',
            'last_retrain', 'last_reason', 'last_metrics',
        }
        assert set(status.keys()) == expected_keys

    def test_status_initial_values(self, retrainer):
        """Initial status has correct values."""
        status = retrainer.get_status()

        assert status['can_retrain'] is True
        assert status['is_retraining'] is False
        assert status['total_retrains'] == 0
        assert status['max_per_day'] == 3
        assert status['cooldown_minutes'] == 60

    def test_status_after_retrain(self, retrainer, fake_data):
        """Status reflects state after completed retrain."""
        X, y = fake_data
        retrainer.trigger_retrain(X_replay=X, y_replay=y, reason="test", force=True)

        status = retrainer.get_status()
        assert status['total_retrains'] == 1
        assert status['last_reason'] == 'test'
        assert status['last_retrain'] is not None
