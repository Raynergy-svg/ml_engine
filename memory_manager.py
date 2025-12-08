"""
Memory management utilities for optimizing GPU and CPU memory usage.
Enhanced with additional optimization techniques and better error handling.
"""

import gc
import time
import logging
import psutil
import numpy as np
import torch
from functools import wraps
from typing import Callable, Optional, Union, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Enhanced memory manager for both CPU and GPU operations with intelligent caching,
    proactive memory management, and tensor optimization.
    """
    
    def __init__(
        self, 
        device: str = None, 
        threshold_mb: int = 1000, 
        log_level: str = "INFO",
        cleanup_interval: int = 10,
        proactive_cleanup: bool = True,
        tensor_cache_size: int = 100
    ):
        """
        Initialize the memory manager.
        
        Args:
            device: Device to monitor ('cpu', 'cuda', or None for auto-detection)
            threshold_mb: Memory threshold in MB to trigger cleanup
            log_level: Logging level for memory operations
            cleanup_interval: Minimum seconds between cleanup operations
            proactive_cleanup: Whether to proactively clean up memory
            tensor_cache_size: Maximum number of tensors to cache
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold_mb = threshold_mb
        self.is_cuda = self.device.startswith('cuda')
        self._setup_logging(log_level)
        self.last_cleanup_time = 0
        self.cleanup_interval = cleanup_interval
        self.proactive_cleanup = proactive_cleanup
        
        # Tensor cache for reusing common tensor shapes
        self.tensor_cache = {}
        self.tensor_cache_size = tensor_cache_size
        self.tensor_cache_hits = 0
        self.tensor_cache_misses = 0
        
        # Track memory usage over time
        self.memory_history = []
        self.record_interval = 60  # seconds
        self.last_record_time = 0
        
        # Initialize memory stats
        self.peak_memory = 0
        self.record_memory_usage()
        
        logger.info(f"MemoryManager initialized on {self.device} with {threshold_mb}MB threshold")
        
    def _setup_logging(self, log_level: str):
        """Configure logging for memory operations."""
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
            # For CUDA, get free memory
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            return free_memory / (1024 * 1024)
        else:
            # For CPU, get system available memory
            return psutil.virtual_memory().available / (1024 * 1024)
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is above threshold."""
        if self.is_cuda:
            # For CUDA, check if we're using more than threshold_mb
            return self.get_used_memory() > self.threshold_mb
        else:
            # For CPU, check if available memory is less than threshold_mb
            return self.get_available_memory() < self.threshold_mb
    
    def record_memory_usage(self):
        """Record current memory usage for tracking."""
        current_time = time.time()
        if current_time - self.last_record_time > self.record_interval:
            used_memory = self.get_used_memory()
            self.memory_history.append((current_time, used_memory))
            self.peak_memory = max(self.peak_memory, used_memory)
            self.last_record_time = current_time
            
            # Keep only the last 100 records
            if len(self.memory_history) > 100:
                self.memory_history = self.memory_history[-100:]
    
    def cleanup(self, force: bool = False):
        """
        Perform memory cleanup if needed or forced.
        
        Args:
            force: Force cleanup regardless of threshold
        """
        current_time = time.time()
        
        # Record memory usage
        self.record_memory_usage()
        
        # Only clean up if forced or if we're above threshold and enough time has passed
        if force or (self.is_memory_critical() and 
                    current_time - self.last_cleanup_time > self.cleanup_interval):
            
            # Run Python garbage collector
            gc.collect()
            
            # CUDA-specific cleanup
            if self.is_cuda:
                torch.cuda.empty_cache()
                
            self.last_cleanup_time = current_time
            logger.debug(f"Memory cleanup performed. Current usage: {self.get_used_memory():.2f} MB")
            return True
        
        # Proactive cleanup if enabled and approaching threshold
        elif self.proactive_cleanup:
            used_memory = self.get_used_memory()
            threshold_ratio = used_memory / self.threshold_mb
            
            # If we're using more than 80% of threshold, do a light cleanup
            if threshold_ratio > 0.8 and current_time - self.last_cleanup_time > self.cleanup_interval * 2:
                gc.collect()
                if self.is_cuda:
                    torch.cuda.empty_cache()
                self.last_cleanup_time = current_time
                logger.debug(f"Proactive memory cleanup performed. Current usage: {self.get_used_memory():.2f} MB")
                return True
                
        return False

    def optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize a tensor for the current device with memory-efficient settings.
        
        Args:
            tensor: Input tensor to optimize
            
        Returns:
            Optimized tensor
        """
        # Ensure tensor is on the right device
        if tensor.device.type != self.device:
            tensor = tensor.to(self.device)
            
        # For CPU tensors, we can use more aggressive memory optimizations
        if not self.is_cuda:
            # Use contiguous memory layout for better CPU cache utilization
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
                
        return tensor
    
    def get_cached_tensor(self, shape: Tuple[int, ...], dtype=torch.float32) -> Optional[torch.Tensor]:
        """
        Get a cached tensor of the specified shape and dtype.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            
        Returns:
            Cached tensor or None if not found
        """
        cache_key = (shape, dtype, self.device)
        if cache_key in self.tensor_cache:
            self.tensor_cache_hits += 1
            return self.tensor_cache[cache_key]
        
        self.tensor_cache_misses += 1
        return None
    
    def cache_tensor(self, tensor: torch.Tensor):
        """
        Cache a tensor for future reuse.
        
        Args:
            tensor: Tensor to cache
        """
        # Only cache if we haven't exceeded the cache size
        if len(self.tensor_cache) >= self.tensor_cache_size:
            # Remove oldest entry
            if self.tensor_cache:
                oldest_key = next(iter(self.tensor_cache))
                del self.tensor_cache[oldest_key]
        
        cache_key = (tensor.shape, tensor.dtype, tensor.device.type)
        self.tensor_cache[cache_key] = tensor
    
    def create_tensor(
        self, 
        shape: Tuple[int, ...], 
        dtype=torch.float32, 
        device=None, 
        requires_grad=False
    ) -> torch.Tensor:
        """
        Create a tensor with caching for common shapes.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            device: Device to create tensor on (defaults to manager's device)
            requires_grad: Whether tensor requires gradients
            
        Returns:
            New or cached tensor
        """
        device = device or self.device
        
        # Try to get from cache first
        cached = self.get_cached_tensor(shape, dtype)
        if cached is not None:
            # Clone to avoid modifying cached tensor
            result = cached.clone()
            result.requires_grad = requires_grad
            return result
        
        # Create new tensor
        result = torch.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        
        # Cache for future use
        if not requires_grad:
            self.cache_tensor(result)
            
        return result
    
    def smart_to_device(self, data, non_blocking: bool = True):
        """
        Efficiently move data to device with smart handling of different types.
        
        Args:
            data: Input data (tensor, list, dict, etc.)
            non_blocking: Whether to use non_blocking transfer for CUDA
            
        Returns:
            Data on the target device
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=non_blocking)
        elif isinstance(data, (list, tuple)):
            return [self.smart_to_device(x, non_blocking) for x in data]
        elif isinstance(data, dict):
            return {k: self.smart_to_device(v, non_blocking) for k, v in data.items()}
        else:
            return data
    
    def get_optimal_batch_size(
        self, 
        model: torch.nn.Module, 
        input_shape: Tuple[int, ...], 
        starting_batch_size: int = 32,
        max_memory_usage: float = 0.8
    ) -> int:
        """
        Determine the optimal batch size for the given model and input shape.
        
        Args:
            model: PyTorch model
            input_shape: Shape of a single input (without batch dimension)
            starting_batch_size: Initial batch size to try
            max_memory_usage: Maximum fraction of memory to use
            
        Returns:
            Optimal batch size
        """
        if not self.is_cuda:
            # For CPU, just return the starting batch size
            return starting_batch_size
        
        # Get total available memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        available_memory = total_memory * max_memory_usage
        
        # Start with the provided batch size
        batch_size = starting_batch_size
        
        # Create a sample input
        sample_input = torch.zeros((batch_size,) + input_shape, device=self.device)
        
        try:
            # Record starting memory
            start_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            
            # Do a forward and backward pass
            output = model(sample_input)
            if isinstance(output, tuple):
                output = output[0]
            loss = output.sum()
            loss.backward()
            
            # Record memory after backward pass
            end_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            
            # Calculate memory per sample
            memory_per_sample = (end_memory - start_memory) / batch_size
            
            # Calculate optimal batch size
            optimal_batch_size = int((available_memory - start_memory) / memory_per_sample)
            
            # Ensure batch size is at least 1
            optimal_batch_size = max(1, optimal_batch_size)
            
            # Round to power of 2 for better GPU utilization
            optimal_batch_size = 2 ** int(np.log2(optimal_batch_size))
            
            logger.info(f"Optimal batch size determined: {optimal_batch_size}")
            return optimal_batch_size
            
        except Exception as e:
            logger.warning(f"Error determining optimal batch size: {e}")
            return starting_batch_size
        finally:
            # Clean up
            torch.cuda.empty_cache()
            gc.collect()
    
    def print_memory_stats(self):
        """Print memory usage statistics."""
        used_memory = self.get_used_memory()
        available_memory = self.get_available_memory()
        
        logger.info(f"Memory Stats for {self.device}:")
        logger.info(f"  Current Usage: {used_memory:.2f} MB")
        logger.info(f"  Available: {available_memory:.2f} MB")
        logger.info(f"  Peak Usage: {self.peak_memory:.2f} MB")
        logger.info(f"  Tensor Cache: {len(self.tensor_cache)} tensors")
        logger.info(f"  Cache Hits/Misses: {self.tensor_cache_hits}/{self.tensor_cache_misses}")
        
        if self.is_cuda:
            # Print CUDA-specific stats
            logger.info(f"  CUDA Memory Allocated: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
            logger.info(f"  CUDA Memory Reserved: {torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB")
            logger.info(f"  CUDA Max Memory Allocated: {torch.cuda.max_memory_allocated() / (1024 * 1024):.2f} MB")


def memory_efficient(threshold_mb: int = 1000, device: Optional[str] = None, proactive: bool = True):
    """
    Decorator for memory-efficient function execution.
    
    Args:
        threshold_mb: Memory threshold to trigger cleanup
        device: Target device or None for auto-detection
        proactive: Whether to use proactive memory management
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create memory manager for this function call
            manager = MemoryManager(
                device=device, 
                threshold_mb=threshold_mb,
                proactive_cleanup=proactive
            )
            
            # Check memory before execution
            manager.cleanup(force=False)
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Clean up after execution if needed
                manager.cleanup(force=False)
                
                return result
            except Exception as e:
                # Force cleanup on error
                manager.cleanup(force=True)
                raise e
                
        return wrapper
    return decorator


def mixed_precision_context(enabled: bool = True):
    """
    Context manager for mixed precision training.
    
    Args:
        enabled: Whether to enable mixed precision
        
    Returns:
        Context manager for mixed precision
    """
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast()
    else:
        # Return a dummy context manager if not enabled or CUDA not available
        from contextlib import nullcontext
        return nullcontext()


def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize a model for inference by fusing operations and setting eval mode.
    
    Args:
        model: PyTorch model
        
    Returns:
        Optimized model
    """
    # Set model to evaluation mode
    model.eval()
    
    # Fuse batch normalization layers with convolutions where possible
    if hasattr(torch, 'fx'):
        try:
            # Use torch.fx for more advanced optimizations if available
            from torch.fx import symbolic_trace
            from torch.fx.experimental.optimization import fuse
            
            # Symbolically trace the model
            traced_model = symbolic_trace(model)
            
            # Fuse op<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>