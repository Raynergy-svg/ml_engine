"""Enhanced memory management utilities (TensorFlow-only).

The original version of this module contained GPU/mixed-precision helpers tied
to a different framework. Buddy is now TensorFlow-only.

We keep a small compatibility layer so existing imports keep working.
"""

from __future__ import annotations

import gc
import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional

import psutil

logger = logging.getLogger(__name__)


def _has_gpu() -> bool:
    try:
        import tensorflow as tf

        return bool(tf.config.list_physical_devices("GPU"))
    except Exception:
        return False


class MemoryManager:
    """Lightweight CPU memory manager with optional TF GPU awareness."""

    def __init__(
        self,
        device: Optional[str] = None,
        threshold_mb: int = 1000,
        log_level: str = "INFO",
        cleanup_interval: int = 10,
        proactive_cleanup: bool = True,
    ):
        self.device = (device or ("gpu" if _has_gpu() else "cpu")).lower()
        self.threshold_mb = int(threshold_mb)
        self.cleanup_interval = int(cleanup_interval)
        self.proactive_cleanup = bool(proactive_cleanup)
        self.last_cleanup_time = 0.0

        self._setup_logging(log_level)
        self.peak_memory = 0.0

        logger.info("MemoryManager initialized on %s (threshold=%sMB)", self.device, self.threshold_mb)

    def _setup_logging(self, log_level: str) -> None:
        numeric_level = getattr(logging, str(log_level).upper(), logging.INFO)
        logger.setLevel(numeric_level)

    def get_used_memory(self) -> float:
        return psutil.Process().memory_info().rss / (1024 * 1024)

    def get_available_memory(self) -> float:
        return psutil.virtual_memory().available / (1024 * 1024)

    def is_memory_critical(self) -> bool:
        return self.get_available_memory() < float(self.threshold_mb)

    def cleanup(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self.last_cleanup_time) < self.cleanup_interval:
            return
        gc.collect()
        self.last_cleanup_time = now

    def monitor_memory(self) -> dict:
        used = float(self.get_used_memory())
        available = float(self.get_available_memory())
        self.peak_memory = max(self.peak_memory, used)

        if self.is_memory_critical() and self.proactive_cleanup:
            logger.warning("Memory critical: used=%.2fMB available=%.2fMB", used, available)
            self.cleanup(force=True)

        return {
            "used_mb": used,
            "available_mb": available,
            "peak_mb": float(self.peak_memory),
            "is_critical": bool(self.is_memory_critical()),
        }


def memory_efficient(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        try:
            return func(*args, **kwargs)
        finally:
            gc.collect()

    return wrapper


@contextmanager
def mixed_precision_context(enabled: bool = True):
    """No-op compatibility shim.

    Mixed precision in TensorFlow is typically enabled via:
    `tf.keras.mixed_precision.set_global_policy('mixed_float16')`.
    """

    _ = enabled
    yield


def profile_memory(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = psutil.Process().memory_info().rss
        result = func(*args, **kwargs)
        end = psutil.Process().memory_info().rss
        logger.info(
            "%s - Memory: start=%.2fMB end=%.2fMB",
            getattr(func, "__name__", "<fn>"),
            start / 1024**2,
            end / 1024**2,
        )
        return result

    return wrapper
