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
import sys
from pathlib import Path
from typing import Any, Dict, Callable
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None
try:
    from joblib import Memory  # type: ignore
except Exception:  # pragma: no cover
    Memory = None
import functools
import copy
from functools import lru_cache

logger = logging.getLogger(__name__)

# Add caching capabilities
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


if Memory is None:  # pragma: no cover
    class _NoopMemory:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def cache(self, func: Callable) -> Callable:
            return func

    memory = _NoopMemory()
else:
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
        logger.info(f"Function {func.__name__!r} took {run_time:.4f} secs")
        return value

    return wrapper_timer


def setup_logging(
    log_file=None,
    level=logging.INFO,
    console_level=None,
    enable_logging=True,
    enable_wandb_logging=True,
    enable_checkpointing_logging=True,
    enable_tensorboard_logging=True,
    quiet=True,
):
    """
    Setup comprehensive logging configuration for the ML engine.

    Args:
        log_file: Path to main log file (optional)
        level: Logging level (default: INFO)
        console_level: Console handler level (default: WARNING). Set to
            logging.INFO for verbose console output.
        enable_logging: Enable console logging
        enable_wandb_logging: Enable Weights & Biases logging
        enable_checkpointing_logging: Enable checkpoint logging
        enable_tensorboard_logging: Enable TensorBoard logging
        quiet: If True (default), console shows only WARNING+. Full DEBUG logs always go to file.

    Returns:
        Configured logger instance

    Note:
        In quiet mode (default), verbose logs are captured to train.log for debugging failures.
        Use --verbose flag to see full console output.
    """
    logger = logging.getLogger()
    # Root logger always at DEBUG so file handlers get everything
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Enhanced formatter with more context
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler: WARNING in quiet mode, level in verbose mode
    if enable_logging:
        # Use stdout so `conda run` doesn't buffer console logs until process exit.
        # (Default StreamHandler uses stderr, which can appear "after" training finishes.)
        # Console shows WARNING+ only; full detail goes to file handlers.
        ch = logging.StreamHandler(stream=sys.stdout)
        console_level = logging.WARNING if quiet else level
        ch.setLevel(console_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    base_dir = Path(log_file).parent if log_file else Path.cwd()

    # Always create train.log with full DEBUG output for debugging failures
    # This captures verbose output even in quiet mode
    train_log_path = base_dir / "train.log"
    try:
        train_fh = logging.FileHandler(str(train_log_path), mode='a')
        train_fh.setLevel(logging.DEBUG)
        train_fh.setFormatter(formatter)
        logger.addHandler(train_fh)
    except Exception:
        pass  # Skip if can't create log file

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG if quiet else level)  # Full logs to file
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
    """Load configuration from a YAML file with validation.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If configuration is invalid
    """
    logger = logging.getLogger(__name__)

    try:
        config_path_obj = Path(config_path)

        # Check if file exists
        if not config_path_obj.exists():
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if yaml is None:
            raise ModuleNotFoundError(
                "PyYAML is required to load YAML configs. Install it with: pip install PyYAML"
            )

        # Load YAML
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Validate config is not None
        if config is None:
            logger.error(f"Configuration file is empty: {config_path}")
            raise ValueError(f"Configuration file is empty: {config_path}")

        # Validate config is a dictionary
        if not isinstance(config, dict):
            logger.error(f"Configuration must be a dictionary, got {type(config)}")
            raise ValueError(f"Configuration must be a dictionary, got {type(config)}")

        logger.info(f"Successfully loaded configuration from {config_path}")
        return config

    except FileNotFoundError:
        raise
    except Exception as e:
        if yaml is not None and isinstance(e, getattr(yaml, "YAMLError", Exception)):
            logger.error(f"Error parsing YAML file: {e}")
            raise
        logger.error(f"Unexpected error loading configuration: {e}")
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


def _validate_required_fields(
    config: Dict[str, Any],
    required_fields: Dict[str, list],
    logger: logging.Logger,
) -> bool:
    is_valid = True

    for section, fields in required_fields.items():
        section_value = config.get(section)
        if section_value is None:
            logger.warning(f"Missing configuration section: {section}")
            is_valid = False
            continue

        if not isinstance(section_value, dict):
            logger.warning(
                f"Invalid configuration section type: {section} (must be a dict)"
            )
            is_valid = False
            continue

        for field in fields:
            if field not in section_value:
                logger.warning(f"Missing configuration field: {section}.{field}")
                is_valid = False

    return is_valid


def _validate_positive_field(
    config: Dict[str, Any],
    section: str,
    field: str,
    logger: logging.Logger,
) -> bool:
    if section not in config:
        return True

    section_value = config.get(section)
    if not isinstance(section_value, dict):
        logger.warning(f"Invalid configuration section type: {section} (must be a dict)")
        return False

    value = section_value.get(field, 0)
    if not isinstance(value, (int, float)) or value <= 0:
        logger.warning(f"Invalid {field} value: {value} (must be > 0)")
        return False

    return True


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration has required fields and sensible values.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if valid, False otherwise

    Logs warnings for missing or invalid fields.
    """
    logger = logging.getLogger(__name__)

    required_fields = {
        'model': ['hidden_size', 'num_layers'],
        'training': ['epochs'],
        'data': ['sequence_length']
    }

    is_valid = _validate_required_fields(config, required_fields, logger)

    numeric_checks = [
        ("training", "epochs"),
        ("model", "hidden_size"),
        ("model", "num_layers"),
    ]
    for section, field in numeric_checks:
        if not _validate_positive_field(config, section, field, logger):
            is_valid = False

    return is_valid


def get_config(config_path="config_tuned.yaml", default_config=None, validate=True):
    """Load and optionally validate configuration.

    Args:
        config_path: Path to configuration file
        default_config: Default configuration to merge with
        validate: Whether to validate the configuration

    Returns:
        Configuration dictionary
    """
    logger = logging.getLogger(__name__)

    user_config = load_config(config_path)

    if default_config is not None:
        user_config = merge_config(user_config, default_config)

    if validate and not validate_config(user_config):
        logger.warning("Configuration validation found issues, but continuing...")

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
        config = get_config("config_tuned.yaml", default_config=default_cfg)
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
# — Raynergy-svg —
