"""
Enhanced memory management utilities for optimizing GPU and CPU memory usage.
Includes advanced optimization techniques and intelligent caching.
"""

import gc
import time
import logging
import psutil
import torch
from functools import wraps
from typing import Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryManager:
    """Enhanced memory manager for both CPU and GPU operations."""

    def __init__(
        self,
        device: str = None,
        threshold_mb: int = 1000,
        log_level: str = "INFO",
        cleanup_interval: int = 10,
        proactive_cleanup: bool = True,
    ):
        """Initialize the memory manager."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold_mb = threshold_mb
        self.is_cuda = self.device.startswith("cuda")
        self.cleanup_interval = cleanup_interval
        self.proactive_cleanup = proactive_cleanup
        self.last_cleanup_time = 0

        self._setup_logging(log_level)

        self.peak_memory = 0
        self.memory_history = []

        logger.info(
            f"MemoryManager initialized on {self.device} with {threshold_mb}MB threshold"
        )

    def _setup_logging(self, log_level: str):
        """Configure logging."""
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

    def get_used_memory(self) -> float:
        """Get current used memory in MB."""
        if self.is_cuda:
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            return psutil.Process().memory_info().rss / (1024 * 1024)

    def get_available_memory(self) -> float:
        """Get available memory in MB."""
        if self.is_cuda:
            free_memory = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            )
            return free_memory / (1024 * 1024)
        else:
            return psutil.virtual_memory().available / (1024 * 1024)

    def is_memory_critical(self) -> bool:
        """Check if memory usage is above threshold."""
        if self.is_cuda:
            return self.get_used_memory() > self.threshold_mb
        else:
            return self.get_available_memory() < self.threshold_mb

    def cleanup(self, force: bool = False):
        """Perform memory cleanup."""
        current_time = time.time()

        if (
            not force
            and (current_time - self.last_cleanup_time) < self.cleanup_interval
        ):
            return

        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()

        self.last_cleanup_time = current_time

        used = self.get_used_memory()
        logger.debug(f"Memory cleanup completed. Current usage: {used:.2f}MB")

    def monitor_memory(self):
        """Monitor and log memory usage."""
        used = self.get_used_memory()
        available = self.get_available_memory()

        self.peak_memory = max(self.peak_memory, used)

        if self.is_memory_critical():
            logger.warning(
                f"Memory usage critical: {used:.2f}MB used, {available:.2f}MB available"
            )
            if self.proactive_cleanup:
                self.cleanup(force=True)

        return {
            "used_mb": used,
            "available_mb": available,
            "peak_mb": self.peak_memory,
            "is_critical": self.is_memory_critical(),
        }


def memory_efficient(func: Callable) -> Callable:
    """Decorator for memory-efficient function execution."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Clear cache before execution
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Clear cache after execution
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return wrapper


@contextmanager
def mixed_precision_context(enabled: bool = True):
    """Context manager for mixed precision training."""
    if enabled and torch.cuda.is_available():
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


def optimize_memory_allocation(model: torch.nn.Module, device: str = None):
    """Optimize memory allocation for a model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    # Enable cudnn benchmarking for optimal performance
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    return model


def get_optimal_batch_size(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str = None,
    max_memory_fraction: float = 0.8,
) -> int:
    """Estimate optimal batch size based on available memory."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # Start with a small batch size
    batch_size = 1

    try:
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape).to(device)

        # Forward pass to estimate memory
        with torch.no_grad():
            _ = model(dummy_input)

        if device.startswith("cuda"):
            # Get memory usage
            used_memory = torch.cuda.memory_allocated()
            total_memory = torch.cuda.get_device_properties(0).total_memory

            # Calculate optimal batch size
            available_memory = total_memory * max_memory_fraction
            estimated_batch_size = int(available_memory / used_memory)

            return max(1, estimated_batch_size)
        else:
            # For CPU, use a conservative estimate
            return 32

    except Exception as e:
        logger.warning(f"Failed to estimate optimal batch size: {e}")
        return 32
    finally:
        # Cleanup
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def profile_memory(func: Callable) -> Callable:
    """Decorator to profile memory usage of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        else:
            start_memory = psutil.Process().memory_info().rss

        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()

            logger.info(
                f"{func.__name__} - Memory: start={start_memory / 1024**2:.2f}MB, "
                f"end={end_memory / 1024**2:.2f}MB, peak={peak_memory / 1024**2:.2f}MB"
            )
        else:
            end_memory = psutil.Process().memory_info().rss
            logger.info(
                f"{func.__name__} - Memory: start={start_memory / 1024**2:.2f}MB, "
                f"end={end_memory / 1024**2:.2f}MB"
            )

        return result

    return wrapper
