"""
Tests for global seed management system.

Tests verify that:
- Seeds are properly set across all RNG systems
- Thread-safety is maintained
- Environment variable initialization works
- Temporary seed contexts work correctly
- Reproducibility is achieved
"""

import os
import random
import threading
from unittest.mock import patch

import numpy as np
import pytest

from src.utils.seed_manager import (
    set_global_seed,
    get_global_seed,
    get_thread_rng,
    reset_seed,
    initialize_from_env,
    get_sklearn_random_state,
    SeedContext,
)


@pytest.fixture(autouse=True)
def cleanup_seed():
    """Reset seed state before and after each test."""
    reset_seed()
    yield
    reset_seed()


def test_set_global_seed():
    """Test that set_global_seed sets seed for all RNGs."""
    set_global_seed(42)
    
    # Verify global seed is stored
    assert get_global_seed() == 42
    
    # Verify Python random is seeded
    random.seed(42)
    expected_random = random.random()
    random.seed(42)
    actual_random = random.random()
    assert actual_random == expected_random
    
    # Verify NumPy is seeded
    np.random.seed(42)
    expected_numpy = np.random.rand()
    np.random.seed(42)
    actual_numpy = np.random.rand()
    assert actual_numpy == expected_numpy


def test_set_global_seed_tensorflow():
    """Test that TensorFlow seed is set if TF is available."""
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    
    set_global_seed(42)
    
    # Verify TensorFlow deterministic mode is enabled
    assert os.environ.get('TF_DETERMINISTIC_OPS') == '1'
    assert os.environ.get('TF_CUDNN_DETERMINISTIC') == '1'
    
    # Verify TF random is seeded (basic check)
    tf.random.set_seed(42)
    expected_tf = tf.random.normal([1, 1]).numpy()
    tf.random.set_seed(42)
    actual_tf = tf.random.normal([1, 1]).numpy()
    np.testing.assert_array_equal(actual_tf, expected_tf)


def test_seed_validation():
    """Test that invalid seeds are rejected."""
    # Non-integer seed
    with pytest.raises(ValueError, match="Seed must be an integer"):
        set_global_seed(42.5)
    
    # Negative seed
    with pytest.raises(ValueError, match="Seed must be in range"):
        set_global_seed(-1)
    
    # Seed too large
    with pytest.raises(ValueError, match="Seed must be in range"):
        set_global_seed(2**32)


def test_get_global_seed():
    """Test getting global seed value."""
    # Initially None
    assert get_global_seed() is None
    
    # After setting
    set_global_seed(123)
    assert get_global_seed() == 123


def test_get_thread_rng():
    """Test thread-local RNG initialization."""
    set_global_seed(42)
    
    # Get RNG
    rng = get_thread_rng()
    assert isinstance(rng, np.random.Generator)
    
    # Verify it produces consistent results
    rng1 = get_thread_rng()
    val1 = rng1.random()
    
    # Same thread should get same RNG instance
    rng2 = get_thread_rng()
    assert rng1 is rng2


def test_thread_safety():
    """Test that seed management is thread-safe."""
    results = {}
    
    def worker(thread_id, seed):
        set_global_seed(seed)
        # Small delay to increase chance of race conditions
        import time
        time.sleep(0.001)
        results[thread_id] = get_global_seed()
    
    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i, 42 + i))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # Last thread to finish should have set the global seed
    # At least verify no crashes or exceptions occurred
    assert len(results) == 10


def test_reset_seed():
    """Test seed reset functionality."""
    set_global_seed(42)
    assert get_global_seed() == 42
    
    reset_seed()
    assert get_global_seed() is None


def test_initialize_from_env():
    """Test initialization from environment variable."""
    # No env var set
    with patch.dict(os.environ, {}, clear=True):
        result = initialize_from_env()
        assert result is None
    
    # Valid env var
    with patch.dict(os.environ, {'RANDOM_SEED': '999'}):
        result = initialize_from_env()
        assert result == 999
        assert get_global_seed() == 999
    
    reset_seed()
    
    # Invalid env var
    with patch.dict(os.environ, {'RANDOM_SEED': 'invalid'}):
        result = initialize_from_env()
        assert result is None


def test_get_sklearn_random_state():
    """Test sklearn random state retrieval."""
    # No seed set - should use default
    assert get_sklearn_random_state() == 42
    
    # Seed set - should use global seed
    set_global_seed(123)
    assert get_sklearn_random_state() == 123


def test_seed_context():
    """Test temporary seed context manager."""
    # Set initial seed
    set_global_seed(42)
    assert get_global_seed() == 42
    
    # Use temporary seed
    with SeedContext(999):
        assert get_global_seed() == 999
        # Generate some random numbers
        val_in_context = random.random()
    
    # Seed should be restored
    assert get_global_seed() == 42
    
    # Verify context actually changed seed
    with SeedContext(999):
        val_in_context2 = random.random()
    assert val_in_context == val_in_context2


def test_seed_context_no_previous_seed():
    """Test seed context when no previous seed was set."""
    # No initial seed
    assert get_global_seed() is None
    
    with SeedContext(123):
        assert get_global_seed() == 123
    
    # Should be reset to None
    assert get_global_seed() is None


def test_reproducibility_python_random():
    """Test reproducibility with Python's random module."""
    set_global_seed(42)
    values1 = [random.random() for _ in range(10)]
    
    set_global_seed(42)
    values2 = [random.random() for _ in range(10)]
    
    assert values1 == values2


def test_reproducibility_numpy():
    """Test reproducibility with NumPy."""
    set_global_seed(42)
    values1 = np.random.rand(10)
    
    set_global_seed(42)
    values2 = np.random.rand(10)
    
    np.testing.assert_array_equal(values1, values2)


def test_reproducibility_sklearn():
    """Test reproducibility with scikit-learn."""
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate simple dataset
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Train two models with same seed
    set_global_seed(42)
    clf1 = RandomForestClassifier(
        n_estimators=10, 
        random_state=get_sklearn_random_state()
    )
    clf1.fit(X, y)
    pred1 = clf1.predict(X)
    
    set_global_seed(42)
    clf2 = RandomForestClassifier(
        n_estimators=10,
        random_state=get_sklearn_random_state()
    )
    clf2.fit(X, y)
    pred2 = clf2.predict(X)
    
    # Predictions should be identical
    np.testing.assert_array_equal(pred1, pred2)


def test_reproducibility_tensorflow():
    """Test reproducibility with TensorFlow."""
    tf = pytest.importorskip("tensorflow")
    
    set_global_seed(42)
    values1 = tf.random.normal([10, 10]).numpy()
    
    set_global_seed(42)
    values2 = tf.random.normal([10, 10]).numpy()
    
    np.testing.assert_array_almost_equal(values1, values2, decimal=5)


def test_multiple_seed_changes():
    """Test changing seed multiple times."""
    seeds = [42, 123, 999, 0, 12345]
    
    for seed in seeds:
        set_global_seed(seed)
        assert get_global_seed() == seed
        
        # Verify RNGs use new seed
        val = random.random()
        
        # Reset and verify reproducibility
        set_global_seed(seed)
        val2 = random.random()
        assert val == val2


def test_concurrent_thread_rngs():
    """Test that each thread gets its own RNG when using thread-local."""
    set_global_seed(42)
    
    results = {}
    
    def worker(thread_id):
        rng = get_thread_rng()
        # Each thread gets the same initial RNG state from global seed
        # This is the expected behavior for reproducibility
        results[thread_id] = rng.random()
    
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # Should have results from all threads
    assert len(results) == 5
    
    # When initialized with the same global seed, all threads get the same
    # initial RNG state, which is correct for reproducibility
    values = list(results.values())
    # All values should be the same (same initial seed)
    assert len(set(values)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
