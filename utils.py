"""Utility functions and helpers"""

import time
import logging
from pathlib import Path
from typing import Any, Dict
import yaml
from joblib import Memory
import functools
import os
import copy
from functools import lru_cache

# Add caching capabilities
cache_dir = Path("./cache")
cache_dir.mkdir(exist_ok=True)
memory = Memory(location=str(cache_dir), verbose=0)
cache = memory.cache

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
memory = Memory(location=str(CACHE_DIR), verbose=0)


def cache(func):
    return memory.cache(func)


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()  # 1. Record start time
        value = func(*args, **kwargs)
        end_time = time.time()  # 2. Record end time
        run_time = end_time - start_time  # 3. Calculate the execution time
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
    """
    Setup comprehensive logging configuration for the ML engine.
    
    Args:
        log_file: Path to main log file (optional)
        level: Logging level (default: INFO)
        enable_logging: Enable console logging
        enable_wandb_logging: Enable Weights & Biases logging
        enable_checkpointing_logging: Enable checkpoint logging
        enable_tensorboard_logging: Enable TensorBoard logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Enhanced formatter with more context
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
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
        
        # Load YAML
        with open(config_path, "r") as f:
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
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
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


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration has required fields and sensible values.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, False otherwise
        
    Logs warnings for missing or invalid fields.
    """
    logger = logging.getLogger(__name__)
    is_valid = True
    
    # Check for common required fields
    required_fields = {
        'model': ['hidden_size', 'num_layers'],
        'training': ['epochs'],
        'data': ['sequence_length']
    }
    
    for section, fields in required_fields.items():
        if section not in config:
            logger.warning(f"Missing configuration section: {section}")
            is_valid = False
            continue
            
        for field in fields:
            if field not in config[section]:
                logger.warning(f"Missing configuration field: {section}.{field}")
                is_valid = False
    
    # Validate numeric ranges
    if 'training' in config:
        epochs = config['training'].get('epochs', 0)
        if epochs <= 0:
            logger.warning(f"Invalid epochs value: {epochs} (must be > 0)")
            is_valid = False
    
    if 'model' in config:
        hidden_size = config['model'].get('hidden_size', 0)
        if hidden_size <= 0:
            logger.warning(f"Invalid hidden_size: {hidden_size} (must be > 0)")
            is_valid = False
            
        num_layers = config['model'].get('num_layers', 0)
        if num_layers <= 0:
            logger.warning(f"Invalid num_layers: {num_layers} (must be > 0)")
            is_valid = False
    
    return is_valid


def get_config(config_path="config.yaml", default_config=None, validate=True):
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
    
    if validate:
        if not validate_config(user_config):
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
