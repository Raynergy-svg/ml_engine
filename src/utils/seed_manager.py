"""
Global Seed Management for Reproducible ML Pipeline.

This module provides a centralized seed management system to ensure reproducibility
across all random number generators used in the ML pipeline including:
- Python's built-in random module
- NumPy
- TensorFlow
- scikit-learn
- Any other libraries that use random state

Thread-safe implementation with environment variable support.
"""

from __future__ import annotations

import logging
import os
import random
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Thread-local storage for seed state
_thread_local = threading.local()

# Global seed registry
_GLOBAL_SEED: Optional[int] = None
_SEED_LOCK = threading.Lock()


def set_global_seed(seed: int) -> None:
    """
    Set global seed for all random number generators.
    
    This function sets the seed for:
    - Python's random module
    - NumPy
    - TensorFlow (if available)
    - Thread-local random state
    
    Args:
        seed: Integer seed value (0 to 2^32-1)
        
    Raises:
        ValueError: If seed is not a valid integer
        
    Example:
        >>> from src.utils.seed_manager import set_global_seed
        >>> set_global_seed(42)
        >>> # All random operations are now reproducible
    """
    global _GLOBAL_SEED
    
    if not isinstance(seed, int):
        raise ValueError(f"Seed must be an integer, got {type(seed)}")
    
    if seed < 0 or seed >= 2**32:
        raise ValueError(f"Seed must be in range [0, 2^32-1], got {seed}")
    
    with _SEED_LOCK:
        _GLOBAL_SEED = seed
        
        # Set Python's built-in random seed
        random.seed(seed)
        logger.info(f"Set Python random seed: {seed}")
        
        # Set NumPy seed
        np.random.seed(seed)
        logger.info(f"Set NumPy random seed: {seed}")
        
        # Set TensorFlow seed (if available)
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            logger.info(f"Set TensorFlow random seed: {seed}")
            
            # Configure TensorFlow for deterministic operations
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
            
        except ImportError:
            logger.debug("TensorFlow not available, skipping TF seed setting")
        
        # Set thread-local random state
        _thread_local.seed = seed
        _thread_local.rng = np.random.default_rng(seed)
        
        logger.info(f"✅ Global seed set to {seed} (all RNGs synchronized)")


def get_global_seed() -> Optional[int]:
    """
    Get the current global seed value.
    
    Returns:
        Current global seed, or None if not set
        
    Example:
        >>> from src.utils.seed_manager import set_global_seed, get_global_seed
        >>> set_global_seed(42)
        >>> print(get_global_seed())
        42
    """
    return _GLOBAL_SEED


def get_thread_rng() -> np.random.Generator:
    """
    Get thread-local NumPy random generator.
    
    This provides a thread-safe random number generator for each thread.
    If the global seed has been set, the RNG will be initialized with it.
    
    Returns:
        NumPy random Generator instance
        
    Example:
        >>> from src.utils.seed_manager import set_global_seed, get_thread_rng
        >>> set_global_seed(42)
        >>> rng = get_thread_rng()
        >>> random_value = rng.random()
    """
    if not hasattr(_thread_local, 'rng'):
        seed = _GLOBAL_SEED if _GLOBAL_SEED is not None else None
        _thread_local.rng = np.random.default_rng(seed)
        if seed is not None:
            logger.debug(f"Initialized thread-local RNG with seed {seed}")
    return _thread_local.rng


def reset_seed() -> None:
    """
    Reset global seed to None and clear thread-local state.
    
    This is primarily useful for testing to ensure clean state between tests.
    
    Example:
        >>> from src.utils.seed_manager import set_global_seed, reset_seed
        >>> set_global_seed(42)
        >>> reset_seed()
        >>> assert get_global_seed() is None
    """
    global _GLOBAL_SEED
    
    with _SEED_LOCK:
        _GLOBAL_SEED = None
        
        # Clear thread-local state
        if hasattr(_thread_local, 'seed'):
            delattr(_thread_local, 'seed')
        if hasattr(_thread_local, 'rng'):
            delattr(_thread_local, 'rng')
        
        logger.info("Global seed reset")


def initialize_from_env() -> Optional[int]:
    """
    Initialize global seed from environment variable.
    
    Checks for the RANDOM_SEED environment variable and sets the global seed
    if it exists. This allows seed configuration through environment variables.
    
    Returns:
        The seed value that was set, or None if env var not found
        
    Example:
        # In shell:
        # export RANDOM_SEED=42
        
        >>> from src.utils.seed_manager import initialize_from_env
        >>> seed = initialize_from_env()
        >>> print(f"Initialized with seed: {seed}")
        Initialized with seed: 42
    """
    seed_str = os.environ.get('RANDOM_SEED')
    
    if seed_str is None:
        logger.debug("RANDOM_SEED environment variable not set")
        return None
    
    try:
        seed = int(seed_str)
        set_global_seed(seed)
        logger.info(f"Initialized global seed from environment: {seed}")
        return seed
    except ValueError:
        logger.error(f"Invalid RANDOM_SEED value: {seed_str} (must be integer)")
        return None


def get_sklearn_random_state() -> int:
    """
    Get random state value for scikit-learn estimators.
    
    Returns the global seed if set, otherwise returns a default value of 42.
    This ensures scikit-learn models use the global seed when available.
    
    Returns:
        Random state value for sklearn
        
    Example:
        >>> from src.utils.seed_manager import set_global_seed, get_sklearn_random_state
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> set_global_seed(42)
        >>> clf = RandomForestClassifier(random_state=get_sklearn_random_state())
    """
    seed = get_global_seed()
    if seed is None:
        logger.warning("Global seed not set, using default value 42 for sklearn")
        return 42
    return seed


class SeedContext:
    """
    Context manager for temporary seed changes.
    
    This allows temporarily setting a different seed for a specific code block,
    then restoring the previous seed state.
    
    Example:
        >>> from src.utils.seed_manager import set_global_seed, SeedContext
        >>> set_global_seed(42)
        >>> with SeedContext(123):
        ...     # Code here uses seed 123
        ...     pass
        >>> # Seed 42 is restored here
    """
    
    def __init__(self, seed: int):
        """
        Initialize seed context.
        
        Args:
            seed: Temporary seed value to use
        """
        self.temp_seed = seed
        self.previous_seed = None
    
    def __enter__(self):
        """Enter context: save current seed and set temporary seed."""
        self.previous_seed = get_global_seed()
        set_global_seed(self.temp_seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: restore previous seed."""
        if self.previous_seed is not None:
            set_global_seed(self.previous_seed)
        else:
            reset_seed()
        return False


# Auto-initialize from environment on module import
initialize_from_env()
