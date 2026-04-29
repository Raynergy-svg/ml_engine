"""Single source of truth for Buddy's runtime environment.

Mythos audit 2026-04-28 (with DevOps Automator subagent) — env-var
setup was duplicated across three sites: ./buddy launcher, main.py
top-of-file block, and src/tui/embedded_scanner.py:166-174. Operators
also had to remember to manually export class-C toggles
(BUDDY_META_MANAGER_ENABLED, BUDDY_META_USE_LLM, etc.) on every
session.

This module is the deterministic Python source of truth. The shell
companion (scripts/init.sh) reuses the same defaults via
``dump_shell_exports`` so launching via ./buddy and launching via
``python -m src.tui`` see identical environments.

Behavior:
  * ``ensure_runtime_env()`` is idempotent — calling it twice is a
    no-op. Uses ``os.environ.setdefault`` so operator-supplied values
    always win.
  * ``.env.local`` and ``.env.local.toggles`` are sourced if present
    (parsed in pure Python — no shell dependency).
  * macOS thread caps + Metal guards apply only on Darwin.
  * ``BUDDY_HOT_RELOAD`` defaults to "0" so Phase 1 ships disabled.
    Operators flip to "1" once Phase 2 lands and is verified.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# Class-A: macOS thread + Metal caps. Identical to main.py:29-54 and
# the duplicate block in embedded_scanner.py. Lifted here so both
# entry points share one canonical definition.
_MACOS_DEFAULTS: Dict[str, str] = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "MKL_THREADING_LAYER": "GNU",
    "KMP_AFFINITY": "disabled",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    "ML_ENGINE_DISABLE_METAL": "1",
    "TF_CPP_MIN_LOG_LEVEL": "3",
}

# Class-B: TUI hygiene + log paths. Cross-platform.
_TUI_DEFAULTS: Dict[str, str] = {
    "BUDDY_TUI_STDERR_LOG": "logs/buddy_tui.stderr.log",
}

# Class-C: behavior toggles operators currently must remember to
# export by hand. Defaults documented in .env.local.toggles.example.
# These are setdefault — .env.local.toggles overrides if present, and
# explicit operator exports always win.
_BEHAVIOR_DEFAULTS: Dict[str, str] = {
    # Hot-reload watcher — disabled until Phase 2 ships and soaks
    "BUDDY_HOT_RELOAD": "0",
    # Meta-cybernetic pipeline routing — see src/scanner/automation/meta_manager.py
    # Default: disabled so operators opt-in deliberately. Setting "1" in
    # .env.local.toggles enables route_incident → meta-pipeline.
    "BUDDY_META_MANAGER_ENABLED": "0",
    # No-LLM policy on cycle_autonomy fallback — see commit 3124a5c.
    # Default: "0" honors the "removed LLM, committed to meta-layers" policy.
    "BUDDY_META_USE_LLM": "0",
}


def _parse_dotenv(path: Path) -> Dict[str, str]:
    """Parse a simple .env file. Pure-python — no shell dependency.

    Supports KEY=value, KEY="value with spaces", KEY='value', comments
    (# at line start or after whitespace). Skips invalid lines. Never
    raises on a malformed file.
    """
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return out

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        # Strip optional surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        # Strip inline trailing comment (only when value isn't quoted)
        elif "#" in value:
            value = value.split("#", 1)[0].rstrip()
        out[key] = value
    return out


def _apply(defaults: Dict[str, str]) -> None:
    """``setdefault`` every key — operator-supplied values always win."""
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def ensure_runtime_env(
    project_root: Optional[Path] = None,
    *,
    enforce_session_id: bool = True,
) -> None:
    """Idempotently set Buddy's runtime environment defaults.

    Order of precedence (lower wins, higher overrides):
      1. Built-in defaults (this module)
      2. .env.local (operator's secrets — OANDA_*, etc.)
      3. .env.local.toggles (operator's BUDDY_* behavior toggles)
      4. Existing os.environ values (explicit shell exports always win)

    Safe to call from any entry point; subsequent calls are no-ops.
    """
    root = Path(project_root) if project_root else PROJECT_ROOT

    # 4 (highest): existing os.environ — captured implicitly by setdefault.
    # 3: dotenv files — apply BEFORE built-in defaults so dotenv wins over
    #    built-ins, but explicit exports still win over dotenv.
    for fname in (".env.local", ".env.local.toggles"):
        for k, v in _parse_dotenv(root / fname).items():
            os.environ.setdefault(k, v)

    # 1+2: built-in defaults
    if platform.system() == "Darwin":
        _apply(_MACOS_DEFAULTS)
    _apply(_TUI_DEFAULTS)
    _apply(_BEHAVIOR_DEFAULTS)

    # Per-process session id — useful for joining reload journal,
    # heartbeat, and trade journal. New value per process unless the
    # operator pinned one in .env.local.toggles.
    if enforce_session_id and not os.environ.get("BUDDY_SESSION_ID"):
        from datetime import datetime, timezone
        os.environ["BUDDY_SESSION_ID"] = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )


def dump_shell_exports(stream=sys.stdout) -> None:
    """Print ``export KEY=value`` lines for every default this module
    would set on the current platform. Used by scripts/init.sh to
    keep the shell launch path in lockstep with the Python path:

        eval "$(python -c 'from src.bootstrap.env import dump_shell_exports; dump_shell_exports()')"

    Operator-supplied values are NOT emitted (they're already in the
    shell environment); only the missing defaults are filled in.
    """
    sources: Iterable[Dict[str, str]]
    if platform.system() == "Darwin":
        sources = (_MACOS_DEFAULTS, _TUI_DEFAULTS, _BEHAVIOR_DEFAULTS)
    else:
        sources = (_TUI_DEFAULTS, _BEHAVIOR_DEFAULTS)

    for src in sources:
        for k, v in src.items():
            if k in os.environ:
                continue  # already set — let operator value pass through
            # Single-quote-escape for shell safety
            escaped = v.replace("'", "'\"'\"'")
            print(f"export {k}='{escaped}'", file=stream)
