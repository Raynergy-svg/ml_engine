"""Safe, optional Weave (W&B) tracing integration.

Design rules:
- Never crash the trading loop because of observability.
- Zero overhead when disabled (decorator returns the original callable).
- Env-gated: set `BUDDY_WEAVE_ENABLED=1` (or `=true`) to activate.
- Idempotent `init_weave()` — safe to call from many entry points.

Usage:
    from src.observability import init_weave, op, thread, patch_instance_methods

    init_weave()                                    # once at startup

    @op                                             # trace a function
    def my_fn(x): ...

    with thread("trade-1124"):                      # group nested ops
        ...

    patch_instance_methods(team, prefix="_evaluate_")  # trace every _evaluate_* method
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ── Module-level state ───────────────────────────────────────────
_weave_module: Any = None          # the imported `weave` module (or None)
_initialized: bool = False         # has init_weave() succeeded once?
_enabled: bool = False             # computed from env at init time


def _env_truthy(value: Optional[str]) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """Return True iff weave is imported, initialized, and env-enabled."""
    return _enabled and _initialized and _weave_module is not None


def init_weave(
    project: Optional[str] = None,
    enabled_override: Optional[bool] = None,
) -> bool:
    """Initialize Weave once. Idempotent. Returns True if active, False otherwise.

    Args:
        project: W&B project name. Defaults to env BUDDY_WEAVE_PROJECT or "buddy-agents".
        enabled_override: If set, bypasses env gate (useful for tests).
    """
    global _weave_module, _initialized, _enabled

    if _initialized:
        return is_enabled()

    if enabled_override is not None:
        _enabled = bool(enabled_override)
    else:
        _enabled = _env_truthy(os.getenv("BUDDY_WEAVE_ENABLED"))

    if not _enabled:
        _initialized = True  # mark initialized so we don't re-check every call
        logger.debug("Weave tracing disabled (set BUDDY_WEAVE_ENABLED=1 to enable)")
        return False

    try:
        import atexit
        import weave  # type: ignore

        proj = project or os.getenv("BUDDY_WEAVE_PROJECT", "buddy-agents")
        client = weave.init(proj)
        _weave_module = weave
        _initialized = True

        # Ensure queued traces flush before Python exits (otherwise the batch
        # processor loses up to N calls on shutdown).
        def _flush_on_exit() -> None:
            try:
                client.flush()
            except Exception as exc:  # noqa: BLE001
                logger.debug("weave flush on exit failed: %s", exc)

        atexit.register(_flush_on_exit)
        logger.info("Weave tracing ACTIVE — project=%s", proj)
        return True
    except ImportError:
        logger.warning(
            "BUDDY_WEAVE_ENABLED=1 but `weave` package not installed. "
            "Run: pip install weave"
        )
        _enabled = False
        _initialized = True
        return False
    except Exception as exc:  # noqa: BLE001 — observability must never crash caller
        logger.error("Weave init failed: %s — disabling", exc, exc_info=True)
        _enabled = False
        _initialized = True
        return False


def op(fn: F) -> F:
    """Decorate a callable with `weave.op` if weave is active, else return fn.

    Safe to stack on sync OR async functions. Evaluated lazily on first call so
    modules can import this decorator before init_weave() runs.
    """
    _wrapped: Optional[Callable[..., Any]] = None

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal _wrapped
        if not is_enabled():
            return fn(*args, **kwargs)
        if _wrapped is None:
            try:
                _wrapped = _weave_module.op()(fn)  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                logger.warning("weave.op wrap failed for %s: %s", fn.__qualname__, exc)
                _wrapped = fn  # fall back permanently
        return _wrapped(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


@contextmanager
def thread(thread_id: str) -> Iterator[None]:
    """Group nested ops under a thread id (e.g. trade_id, cycle_id).

    No-op when weave is disabled.
    """
    if not is_enabled():
        yield
        return
    try:
        ctx = _weave_module.thread(thread_id=thread_id)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        logger.debug("weave.thread(%s) failed: %s", thread_id, exc)
        yield
        return
    with ctx:
        yield


@contextmanager
def attributes(attrs: dict) -> Iterator[None]:
    """Attach key=value attributes to all ops called inside this block.

    In the W&B UI these become filterable columns (pair, direction, regime,
    profile, cycle_id, ...). No-op when weave is disabled.
    """
    if not is_enabled() or not attrs:
        yield
        return
    try:
        ctx = _weave_module.attributes(attrs)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        logger.debug("weave.attributes failed: %s", exc)
        yield
        return
    with ctx:
        yield


def patch_instance_methods(
    instance: Any,
    prefix: str = "_evaluate_",
    exclude: Optional[set[str]] = None,
) -> int:
    """Wrap every bound method on `instance` whose name starts with `prefix`.

    Used to trace all 12 `_evaluate_*` agent methods on ScannerAgentTeam without
    editing each method's source. Returns count of methods patched.
    """
    if not is_enabled():
        return 0

    exclude = exclude or set()
    patched = 0
    for name in dir(instance):
        if not name.startswith(prefix) or name in exclude:
            continue
        method = getattr(instance, name, None)
        if not callable(method):
            continue
        try:
            wrapped = _weave_module.op()(method)  # type: ignore[union-attr]
            setattr(instance, name, wrapped)
            patched += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not patch %s: %s", name, exc)

    if patched:
        logger.info(
            "Weave: patched %d methods on %s (prefix=%r)",
            patched, type(instance).__name__, prefix,
        )
    return patched
