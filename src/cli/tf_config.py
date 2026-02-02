"""TensorFlow and Metal GPU configuration utilities.

This module provides functions for configuring TensorFlow to work
optimally on Apple Silicon (M1/M2/M3) with Metal GPU acceleration.
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Any

logger = logging.getLogger(__name__)


def get_env_flag(name: str) -> str | None:
    """Get environment variable value, returning None if empty.

    Args:
        name: Environment variable name.

    Returns:
        Trimmed value or None if not set/empty.
    """
    v = os.environ.get(name)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def is_truthy(v: str | None) -> bool:
    """Check if string value represents a truthy boolean.

    Args:
        v: Input string.

    Returns:
        True if value is 1/true/yes/y/on (case-insensitive).
    """
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def set_log_level(*, verbose: bool) -> None:
    """Set TensorFlow C++ log level.

    Args:
        verbose: If True, set to level 2 (warnings); otherwise level 3 (errors only).
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2" if verbose else "3")


def should_disable_meta_optimizer() -> bool:
    """Check if TensorFlow meta optimizer should be disabled.

    Disabled by default on Apple Silicon (Darwin/arm64) for stability.

    Returns:
        True if meta optimizer should be disabled.
    """
    v = get_env_flag("BUDDY_DISABLE_META_OPTIMIZER")
    if v is not None:
        return is_truthy(v)
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def try_disable_meta_optimizer(
    tf_mod: Any, *, verbose: bool, console: Any | None = None
) -> None:
    """Attempt to disable TensorFlow meta optimizer.

    Args:
        tf_mod: TensorFlow module.
        verbose: Whether to print status messages.
        console: Rich console for output (optional).
    """
    if not should_disable_meta_optimizer():
        return
    try:
        tf_mod.config.optimizer.set_experimental_options(
            {"disable_meta_optimizer": True}
        )
        if verbose:
            msg = "TensorFlow meta optimizer disabled (BUDDY_DISABLE_META_OPTIMIZER=1)"
            if console:
                console.print(f"[green]{msg}[/green]")
            else:
                logger.info(msg)
    except (AttributeError, RuntimeError):
        pass


def try_force_cpu(tf_mod: Any, *, verbose: bool, console: Any | None = None) -> None:
    """Force TensorFlow to use CPU only.

    Args:
        tf_mod: TensorFlow module.
        verbose: Whether to print status messages.
        console: Rich console for output (optional).
    """
    try:
        tf_mod.config.set_visible_devices([], "GPU")
        if verbose:
            msg = "TensorFlow forced to CPU (--device cpu)"
            if console:
                console.print(f"[yellow]{msg}[/yellow]")
            else:
                logger.info(msg)
    except RuntimeError as e:
        if verbose:
            msg = f"Could not force CPU-only TensorFlow: {e}"
            if console:
                console.print(f"[yellow]{msg}[/yellow]")
            else:
                logger.warning(msg)


def try_apply_gpu_memory_limit(
    tf_mod: Any, gpus: list[Any], *, verbose: bool, console: Any | None = None
) -> None:
    """Apply GPU memory limit if configured.

    Uses BUDDY_GPU_MEMORY_LIMIT_MB environment variable.

    Args:
        tf_mod: TensorFlow module.
        gpus: List of GPU devices.
        verbose: Whether to print status messages.
        console: Rich console for output (optional).
    """
    mem_limit_mb = os.environ.get("BUDDY_GPU_MEMORY_LIMIT_MB")
    if not mem_limit_mb:
        return
    try:
        limit = int(mem_limit_mb)
        if limit <= 0:
            return
        tf_mod.config.set_logical_device_configuration(
            gpus[0],
            [tf_mod.config.LogicalDeviceConfiguration(memory_limit=limit)],
        )
        if verbose:
            msg = f"TensorFlow GPU memory limit set: {limit} MB"
            if console:
                console.print(f"[green]{msg}[/green]")
            else:
                logger.info(msg)
    except (ValueError, RuntimeError, IndexError) as e:
        if verbose:
            msg = f"Could not set GPU memory limit: {e}"
            if console:
                console.print(f"[yellow]{msg}[/yellow]")
            else:
                logger.warning(msg)


def try_enable_memory_growth(tf_mod: Any, gpus: list[Any]) -> None:
    """Enable memory growth for all GPUs.

    Args:
        tf_mod: TensorFlow module.
        gpus: List of GPU devices.
    """
    for gpu in gpus:
        try:
            tf_mod.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # If a logical device limit was set, memory growth may not be supported.
            pass


def configure_tf_metal(
    *, verbose: bool = False, force_cpu: bool = False, console: Any | None = None
) -> None:
    """Enable TensorFlow Metal GPU on Apple Silicon when available.

    Args:
        verbose: Whether to print status messages.
        force_cpu: If True, force CPU-only mode.
        console: Rich console for output (optional).
    """
    set_log_level(verbose=bool(verbose))

    import tensorflow as tf

    try_disable_meta_optimizer(tf, verbose=bool(verbose), console=console)

    if force_cpu:
        try_force_cpu(tf, verbose=bool(verbose), console=console)

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if verbose:
            msg = "TensorFlow GPU not detected (CPU mode)."
            if console:
                console.print(f"[yellow]{msg}[/yellow]")
            else:
                logger.info(msg)
        return

    try:
        try_apply_gpu_memory_limit(tf, gpus, verbose=bool(verbose), console=console)
        try_enable_memory_growth(tf, gpus)
        if verbose:
            msg = f"TensorFlow GPU enabled: {gpus}"
            if console:
                console.print(f"[green]{msg}[/green]")
            else:
                logger.info(msg)
    except RuntimeError as e:
        if verbose:
            msg = f"GPU detected but could not enable memory growth: {e}"
            if console:
                console.print(f"[yellow]{msg}[/yellow]")
            else:
                logger.warning(msg)


# Aliases for backward compatibility with main.py naming convention
_tf_env_flag = get_env_flag
_tf_is_truthy = is_truthy
_tf_set_log_level = set_log_level
_tf_should_disable_meta_optimizer = should_disable_meta_optimizer
_tf_try_disable_meta_optimizer = try_disable_meta_optimizer
_tf_try_force_cpu = try_force_cpu
_tf_try_apply_gpu_memory_limit = try_apply_gpu_memory_limit
_tf_try_enable_memory_growth = try_enable_memory_growth
_configure_tf_metal = configure_tf_metal
