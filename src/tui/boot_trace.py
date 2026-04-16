"""Boot trace — append-only timestamped log so we can see where the TUI
hangs AFTER Textual takes over the screen (at which point stdout is
wiped and the staged banner in __main__.py is no longer visible).

Usage:
    from src.tui.boot_trace import trace
    trace("on_mount: starting")

Tail in a second terminal:
    tail -f logs/buddy_boot_trace.log

Rotated on each `./buddy` run so logs don't accumulate noise across
sessions. Thread-safe (open+write+close per line — slow, but boot-time
only, and we need crash-durability because each line may be the last
one before the hang.)
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_TRACE_PATH = Path(
    os.environ.get(
        "BUDDY_BOOT_TRACE_LOG",
        os.path.join(os.getcwd(), "logs", "buddy_boot_trace.log"),
    )
)
_T0 = time.monotonic()
_LOCK = threading.Lock()
_INITIALIZED = False


def _init_once() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True
    try:
        _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Truncate on first call this run — one trace per session.
        with open(_TRACE_PATH, "w") as f:
            ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            f.write(f"=== BUDDY boot trace — {ts} — pid={os.getpid()} ===\n")
    except OSError:
        pass


def trace(msg: str) -> None:
    """Append a timestamped checkpoint. Crash-durable (open/write/close)."""
    _init_once()
    elapsed = time.monotonic() - _T0
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    thread = threading.current_thread().name
    line = f"[{ts}] +{elapsed:6.2f}s [{thread:<15}] {msg}\n"
    with _LOCK:
        try:
            with open(_TRACE_PATH, "a") as f:
                f.write(line)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
        except OSError:
            pass
