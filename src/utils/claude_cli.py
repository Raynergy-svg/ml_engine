"""Resolve the `claude` CLI binary robustly, independent of the caller's PATH.

Daemon processes (launchd plists, uvicorn workers spawned by launchd, etc.) run
with a minimal PATH that typically excludes user-installed tool locations like
`~/.local/bin`. `shutil.which("claude")` then fails even though the binary is
installed, producing a false "claude CLI not found on PATH" in AXIOM/reflection
callers. This module augments the search with common install locations before
giving up, and never raises -- callers get None and must degrade honestly.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Mapping, Optional

CLAUDE_CLI_PATH_ENV = "CLAUDE_CLI_PATH"

_FIXED_CANDIDATE_DIRS = (
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
)


def _candidate_dirs(home: Path) -> List[str]:
    """Common install locations for the `claude` CLI, filtered to those that exist."""
    candidates = [
        home / ".local" / "bin",
        home / ".claude" / "bin",
        home / ".npm-global" / "bin",
        *_FIXED_CANDIDATE_DIRS,
    ]
    nvm_root = home / ".nvm" / "versions" / "node"
    if nvm_root.is_dir():
        candidates.extend(sorted(nvm_root.glob("*/bin"), reverse=True))
    return [str(p) for p in candidates if p.is_dir()]


def resolve_claude_cli(
    binary_name: str = "claude",
    *,
    env: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
) -> Optional[str]:
    """Return an absolute path to the `claude` CLI, or None if it truly isn't installed.

    Resolution order:
      1. `CLAUDE_CLI_PATH` env var, if set and points at an executable file.
      2. `shutil.which` on the caller's current PATH (the common case).
      3. `shutil.which` on PATH augmented with common install dirs (~/.local/bin,
         ~/.claude/bin, ~/.npm-global/bin, /opt/homebrew/bin, /usr/local/bin,
         nvm node bin dirs) -- covers daemons launched with a minimal PATH.
    """
    env = dict(env) if env is not None else dict(os.environ)
    home = home if home is not None else Path.home()

    override = env.get(CLAUDE_CLI_PATH_ENV)
    if override:
        candidate = Path(override)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    current_path = env.get("PATH", "")
    found = shutil.which(binary_name, path=current_path or None)
    if found:
        return found

    augmented = os.pathsep.join(p for p in [current_path, *_candidate_dirs(home)] if p)
    return shutil.which(binary_name, path=augmented or None)


def augmented_path_env(env: Optional[Mapping[str, str]] = None, home: Optional[Path] = None) -> str:
    """PATH string for a subprocess env, augmented with the same install dirs.

    Use when spawning the resolved `claude` binary so that if it shells out to
    node/npm/git internally, those are reachable even under a minimal daemon PATH.
    """
    env = dict(env) if env is not None else dict(os.environ)
    home = home if home is not None else Path.home()
    current_path = env.get("PATH", "")
    return os.pathsep.join(p for p in [current_path, *_candidate_dirs(home)] if p)
