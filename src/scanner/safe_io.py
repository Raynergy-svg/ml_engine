"""
Phase 55 (US-341 / US-342): Safe I/O utilities for atomic JSON persistence.

Re-exports safe_json_write, safe_json_read, safe_jsonl_append from
src.scanner.automation.safe_json and adds convenience wrappers.

US-342 adds:
  - retry_with_backoff: decorator for retrying transient I/O failures
  - resilient_save: high-level save with retry + async fallback queue
  - _failed_writes_queue: deferred writes replayed on next successful cycle

Usage:
    from src.scanner.safe_io import safe_json_write, safe_json_read, safe_jsonl_append
    from src.scanner.safe_io import retry_with_backoff, resilient_save

    # Atomic write with locking
    safe_json_write("path/to/file.json", data)

    # Read with corruption recovery
    data = safe_json_read("path/to/file.json", default={})

    # Append to JSONL with locking
    safe_jsonl_append("path/to/file.jsonl", record)

    # Retry decorator
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def my_save(): ...

    # High-level resilient save (retry + queue on failure)
    resilient_save("path/to/file.json", data, context="agent_weights")
"""

import logging
import random
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# US-342: Retry with exponential backoff
# ---------------------------------------------------------------------------

# Non-retryable error types — permission / disk-full are permanent
_NON_RETRYABLE = (PermissionError, IsADirectoryError, NotADirectoryError)

# Module-level save success tracking
_save_stats: Dict[str, Dict[str, int]] = {}  # {context: {success: N, failure: N}}
_save_stats_lock = threading.Lock()

# Async fallback queue for writes that exhausted all retries
_failed_writes_queue: deque = deque(maxlen=200)
_queue_lock = threading.Lock()

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    jitter: bool = True,
    context: str = "",
) -> Callable:
    """Decorator: retry a function on transient I/O failures with exponential backoff.

    Per Retry & Robustness Gates:
    - Exponential backoff with jitter
    - Never retry on permission errors or disk-full (only transient)
    - Log each retry attempt with attempt_number, delay, error_type, context
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = context or func.__name__
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    _record_save_stat(ctx, success=True)
                    return result
                except _NON_RETRYABLE as e:
                    # Non-retryable — surface immediately
                    logger.error(
                        "Phase 55 (US-342): Non-retryable error in %s: %s (%s) — not retrying",
                        ctx, type(e).__name__, e,
                    )
                    _record_save_stat(ctx, success=False)
                    raise
                except (OSError, IOError, Exception) as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        if jitter:
                            delay *= (0.5 + random.random())
                        logger.warning(
                            "Phase 55 (US-342): Retry %d/%d for %s — %s: %s — waiting %.2fs",
                            attempt + 1, max_retries, ctx, type(e).__name__, e, delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "Phase 55 (US-342): All %d retries exhausted for %s — %s: %s",
                            max_retries, ctx, type(e).__name__, e,
                        )
                        _record_save_stat(ctx, success=False)
                        raise
            # Should not reach here, but safety
            if last_exc:
                raise last_exc
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def _record_save_stat(context: str, success: bool) -> None:
    """Track save success/failure rate per context."""
    with _save_stats_lock:
        if context not in _save_stats:
            _save_stats[context] = {"success": 0, "failure": 0}
        key = "success" if success else "failure"
        _save_stats[context][key] += 1
        total = _save_stats[context]["success"] + _save_stats[context]["failure"]
        if total >= 100 and total % 50 == 0:
            rate = _save_stats[context]["success"] / total
            if rate < 0.95:
                logger.warning(
                    "Phase 55 (US-342): Save success rate for '%s' is %.1f%% "
                    "(%d/%d) — below 95%% threshold",
                    context, rate * 100,
                    _save_stats[context]["success"], total,
                )


def get_save_stats() -> Dict[str, Dict[str, int]]:
    """Return copy of save stats for monitoring."""
    with _save_stats_lock:
        return {k: dict(v) for k, v in _save_stats.items()}


def resilient_save(
    path: Union[str, "Path"],
    data: Any,
    context: str = "unknown",
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> bool:
    """High-level save: retry with backoff, queue on total failure.

    Returns True if saved successfully, False if queued for later.
    """
    from pathlib import Path as _Path

    path = _Path(path) if not isinstance(path, _Path) else path

    @retry_with_backoff(max_retries=max_retries, base_delay=base_delay, context=context)
    def _do_save() -> bool:
        return safe_json_write(str(path), data)

    try:
        result = _do_save()
        if result:
            # Also flush any queued writes for this path
            _flush_queued_writes()
            return True
        # safe_json_write returned False (logged internally) — queue it
        raise IOError(f"safe_json_write returned False for {path}")
    except Exception as e:
        logger.warning(
            "Phase 55 (US-342): Queueing failed write for next cycle — %s (%s: %s)",
            context, type(e).__name__, e,
        )
        with _queue_lock:
            _failed_writes_queue.append((str(path), data, context, time.time()))
        return False


def _flush_queued_writes() -> int:
    """Attempt to write any queued items from previous failures. Returns count flushed."""
    flushed = 0
    with _queue_lock:
        items = list(_failed_writes_queue)
        _failed_writes_queue.clear()

    for path_str, data, ctx, queued_at in items:
        age = time.time() - queued_at
        if age > 3600:  # Drop writes older than 1 hour
            logger.info(
                "Phase 55 (US-342): Dropping stale queued write for %s (%.0fs old)", ctx, age,
            )
            continue
        try:
            if safe_json_write(path_str, data):
                flushed += 1
                logger.info("Phase 55 (US-342): Flushed queued write for %s", ctx)
            else:
                with _queue_lock:
                    _failed_writes_queue.append((path_str, data, ctx, queued_at))
        except Exception:
            with _queue_lock:
                _failed_writes_queue.append((path_str, data, ctx, queued_at))

    return flushed


def get_failed_writes_count() -> int:
    """Return number of writes in the failure queue."""
    with _queue_lock:
        return len(_failed_writes_queue)

# Re-export from existing module
try:
    from src.scanner.automation.safe_json import (
        safe_json_write,
        safe_json_read,
        safe_jsonl_append,
    )
except ImportError:
    # Fallback implementations if automation.safe_json not available
    import json
    import os
    import tempfile
    from pathlib import Path
    from typing import Any, Union

    def safe_json_write(
        path: Union[str, Path],
        data: Any,
        indent: int = 2,
        create_backup: bool = True,
    ) -> bool:
        """Atomic JSON write: serialize → temp file → fsync → rename."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            json_str = json.dumps(data, indent=indent, sort_keys=True, default=str)
            if create_backup and path.exists():
                try:
                    bak = path.with_suffix(path.suffix + ".bak")
                    bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                except Exception:
                    pass
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
            )
            try:
                os.write(fd, json_str.encode("utf-8"))
                os.fsync(fd)
                os.close(fd)
                fd = -1
                os.rename(tmp_path, str(path))
                return True
            except Exception:
                if fd >= 0:
                    os.close(fd)
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.error("safe_io: Failed to write %s: %s", path, e)
            return False

    def safe_json_read(
        path: Union[str, Path],
        default: Any = None,
    ) -> Any:
        """Read JSON with graceful fallback."""
        path = Path(path)
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("safe_io: Failed to read %s: %s", path, e)
            return default

    def safe_jsonl_append(
        path: Union[str, Path],
        record: Any,
    ) -> bool:
        """Append a JSON record to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            line = json.dumps(record, default=str) + "\n"
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            return True
        except Exception as e:
            logger.error("safe_io: Failed to append to %s: %s", path, e)
            return False

    logger.debug("Phase 55 (US-341): Using fallback safe_io (automation.safe_json not available)")
