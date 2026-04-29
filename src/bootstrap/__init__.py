"""Buddy environment bootstrap — single source of truth for runtime
environment variables (macOS thread caps, BUDDY_* toggles, .env.local
sourcing).

Imported by main.py and src/tui/app.py at startup so every entry point
sees the same defaults without duplicating the env block. The bash
companion is scripts/init.sh, which is sourced by ./buddy and produces
the same environment for child processes.
"""
from __future__ import annotations

from .env import ensure_runtime_env, dump_shell_exports

__all__ = ["ensure_runtime_env", "dump_shell_exports"]
