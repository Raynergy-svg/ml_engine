"""
Utility functions and helpers for ML Engine.

This module provides common utilities including:
- Configuration loading and validation
- Logging setup with multiple handlers
- Function caching and timing decorators
- Configuration merging and management
"""

import time
import logging
from pathlib import Path
from typing import Any, Dict, Callable
import yaml
from joblib import Memory
import functools
import os
import copy
from functools import lru_cache

# Add caching capabilities
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
memory = Memory(location=str(CACHE_DIR), verbose=0)


def cache(func):
    """
    Decorator to cache function results using joblib.
    
    Args:
        func: Function to cache
        
    Returns:
        Cached version of the function
    """
    return memory.cache(func)


def timer(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that logs execution time
        
    Example:
        >>> @timer
        ... def slow_function():
        ...     time.sleep(1)
        >>> slow_function()
        # Logs: Function 'slow_function' took 1.0000 secs
    """
    @functools.wraps(func)
    def wrapper_timer(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        logging.info(f"Function {func.__name__!r} took {run_time:.4f} secs")
        return value

    return wrapper_timer


def setup_logging(
    log_file=None,
    level=logging.INFO,
    enable_logging=True,
    enable_wandb_logging=True,
    enable_checkpointing_logging=True,
    enable_tensorboard_logging=True,
):
    logger = logging.getLogger()
    logger.setLevel(level)
    if logger.handlers:
        logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if enable_logging:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    base_dir = Path(log_file).parent if log_file else Path.cwd()
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if enable_wandb_logging:
        wandb_log_path = base_dir / "wandb.log"
        wh = logging.FileHandler(str(wandb_log_path))
        wh.setLevel(level)
        wh.setFormatter(formatter)
        logger.addHandler(wh)
    if enable_checkpointing_logging:
        cp_log_path = base_dir / "checkpoint.log"
        cph = logging.FileHandler(str(cp_log_path))
        cph.setLevel(level)
        cph.setFormatter(formatter)
        logger.addHandler(cph)
    if enable_tensorboard_logging:
        tb_log_path = base_dir / "tensorboard.log"
        tbh = logging.FileHandler(str(tb_log_path))
        tbh.setLevel(level)
        tbh.setFormatter(formatter)
        logger.addHandler(tbh)
    return logger


@lru_cache(maxsize=1)
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {e}")
        raise


def merge_config(user_config, default_config):
    def recursive_merge(default, override):
        for key, value in override.items():
            if (
                key in default
                and isinstance(default[key], dict)
                and isinstance(value, dict)
            ):
                default[key] = recursive_merge(default[key], value)
            else:
                default[key] = value
        return default

    merged = copy.deepcopy(default_config)
    return recursive_merge(merged, user_config)


def get_config(config_path="config.yaml", default_config=None):
    user_config = load_config(config_path)
    if default_config is not None:
        return merge_config(user_config, default_config)
    return user_config


if __name__ == "__main__":
    logger = setup_logging(log_file="app.log")
    logger.info("Logging is configured.")
    default_cfg = {
        "device": "cpu",
        "num_workers": 4,
        "pin_memory": False,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
    }
    try:
        config = get_config("config.yaml", default_config=default_cfg)
        logger.info(f"Config loaded and merged: {config}")
    except FileNotFoundError as e:
        logger.warning(e)

    @cache
    @timer
    def slow_function(n):
        time.sleep(n)
        return f"Slept for {n} seconds"

    print(slow_function(2))
    print(slow_function(2))
