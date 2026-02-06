#!/usr/bin/env python3
"""TensorFlow and Apple Silicon Metal configuration utilities."""
from __future__ import annotations

import os
import logging
import platform

from rich.console import Console

console = Console()


def _configure_predict_output(verbose: bool) -> None:
    """Reduce log/warning noise for interactive predict runs."""
    if verbose:
        return

    import warnings

    # Keep common third-party warnings from drowning out the prediction output.
    # (Seen on macOS system Python where ssl is LibreSSL.)
    warnings.filterwarnings(
        "ignore",
        message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+",
    )
    try:
        import pandas as pd  # noqa: F401
        from pandas.errors import PerformanceWarning

        warnings.filterwarnings("ignore", category=PerformanceWarning)
    except Exception:
        pass

    # Keep CLI output clean by muting INFO logs from internal modules.
    logging.getLogger().setLevel(logging.WARNING)
    for name in (
        "utils",
        "reasoning_enhanced",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _tf_env_flag(name: str) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _tf_is_truthy(v: str | None) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _tf_set_log_level(*, verbose: bool) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2" if verbose else "3")


def _tf_should_disable_meta_optimizer() -> bool:
    v = _tf_env_flag("BUDDY_DISABLE_META_OPTIMIZER")
    if v is not None:
        return _tf_is_truthy(v)
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _tf_try_disable_meta_optimizer(tf_mod, *, verbose: bool) -> None:
    if not _tf_should_disable_meta_optimizer():
        return
    try:
        tf_mod.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
        if verbose:
            console.print("[green]TensorFlow meta optimizer disabled[/green] (BUDDY_DISABLE_META_OPTIMIZER=1)")
    except Exception:
        pass


def _tf_try_force_cpu(tf_mod, *, verbose: bool) -> None:
    try:
        tf_mod.config.set_visible_devices([], "GPU")
        if verbose:
            console.print("[yellow]TensorFlow forced to CPU[/yellow] (--device cpu)")
    except Exception as e:
        if verbose:
            console.print(f"[yellow]Could not force CPU-only TensorFlow[/yellow]: {e}")


def _tf_try_apply_gpu_memory_limit(tf_mod, gpus, *, verbose: bool) -> None:
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
            console.print(f"[green]TensorFlow GPU memory limit set[/green]: {limit} MB")
    except Exception as e:
        if verbose:
            console.print(f"[yellow]Could not set GPU memory limit[/yellow]: {e}")


def _tf_try_enable_memory_growth(tf_mod, gpus) -> None:
    for gpu in gpus:
        try:
            tf_mod.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            # If a logical device limit was set, memory growth may not be supported.
            pass


def _configure_tf_metal(*, verbose: bool = False, force_cpu: bool = False) -> None:
    """Enable TensorFlow Metal GPU on Apple Silicon when available."""
    _tf_set_log_level(verbose=bool(verbose))

    import tensorflow as tf

    _tf_try_disable_meta_optimizer(tf, verbose=bool(verbose))

    if force_cpu:
        _tf_try_force_cpu(tf, verbose=bool(verbose))

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if verbose:
            console.print("[yellow]TensorFlow GPU not detected[/yellow] (CPU mode).")
        return

    try:
        _tf_try_apply_gpu_memory_limit(tf, gpus, verbose=bool(verbose))
        _tf_try_enable_memory_growth(tf, gpus)
        if verbose:
            console.print(f"[green]TensorFlow GPU enabled[/green]: {gpus}")
    except Exception as e:
        if verbose:
            console.print(f"[yellow]GPU detected but could not enable memory growth[/yellow]: {e}")
