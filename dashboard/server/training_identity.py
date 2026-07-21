"""Exact source-identity preflight for governed local training.

The dataset descriptor records where training data came from.  It is not an
identity for the Python bytes executing a dashboard-requested run.  This
module resolves that separate runtime identity and refuses to name a Git
commit when any source in the bounded training path differs from HEAD.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


TRAINING_CODE_PATHS = (
    "dashboard/server/axiom_evidence_control.py",
    "dashboard/server/training_api.py",
    "dashboard/server/training_identity.py",
    "dashboard/server/training_jobs.py",
    "src/evidence",
    "src/training/labels",
    "src/training/risk_target_features.py",
    "src/training/trainers",
)


def require_runtime_git_commit(repo_root: str | Path) -> str:
    """Return HEAD only when it exactly identifies all governed training code."""
    root = Path(repo_root).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *TRAINING_CODE_PATHS,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot resolve training source identity: {exc}") from exc
    if len(commit) not in range(40, 65) or any(char not in "0123456789abcdef" for char in commit):
        raise RuntimeError("training source HEAD is not a canonical Git commit")
    if status:
        changed = ", ".join(line[3:] for line in status.splitlines()[:8])
        raise RuntimeError(f"training source differs from HEAD: {changed}")
    return commit


def read_runtime_code_identity(repo_root: str | Path) -> dict[str, Any]:
    """Return a client-safe readiness projection without weakening fail-closed use."""
    try:
        commit = require_runtime_git_commit(repo_root)
    except RuntimeError as exc:
        return {
            "available": False,
            "clean": False,
            "git_commit": None,
            "error": str(exc),
        }
    return {
        "available": True,
        "clean": True,
        "git_commit": commit,
        "error": None,
    }


__all__ = ["TRAINING_CODE_PATHS", "read_runtime_code_identity", "require_runtime_git_commit"]
