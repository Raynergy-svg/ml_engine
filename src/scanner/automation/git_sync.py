"""Git synchronization helpers for autonomous self-improvement.

Responsibilities:
  1. Commit *whitelisted* self-improvement artifacts written by Claude's
     trade reflection (learnings.md, rules/, config_adjustments.json,
     proposed_weights.json, state.json). Refuse to commit anything outside
     the whitelist — prevents accidental source-code commits.
  2. Push to the current branch with retry + exponential backoff
     (2s, 4s, 8s, 16s) per the project's network-robustness contract.
  3. Pull latest at cycle start so multiple Buddy instances stay in sync.

Design rules (CLAUDE.md compliance):
  - Explicit timeouts on every subprocess call (60s default)
  - Specific exceptions caught (CalledProcessError, TimeoutExpired)
  - Errors logged with context, never swallowed
  - No --force-push, no --no-verify, no hook bypass
  - Refuses to commit if working tree has changes to non-whitelisted paths
"""
from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import structlog

logger = structlog.get_logger(__name__)

# Paths Claude is permitted to commit. Anything else in the diff aborts the commit.
WHITELIST_PATHS: Tuple[str, ...] = (
    ".claude/learnings.md",
    ".claude/learnings-archive.md",
    ".claude/rules/",          # directory prefix — all rule files allowed
    ".claude/config_adjustments.json",
    ".claude/proposed_weights.json",
    ".claude/proposed_weights_archive/",
    ".claude/state.json",
    ".claude/brain/",
    "logs/reflection_log.jsonl",
)

DEFAULT_TIMEOUT = 60
RETRY_DELAYS = (2, 4, 8, 16)  # seconds — CLAUDE.md contract
COMMIT_PREFIX = "[buddy-reflect]"


def _run_git(
    args: Sequence[str],
    cwd: Optional[Path] = None,
    timeout: int = DEFAULT_TIMEOUT,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a git subprocess with explicit timeout. Raises CalledProcessError on non-zero
    if check=True. Never uses shell=True."""
    cmd = ["git", *args]
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def _is_whitelisted(path: str) -> bool:
    """True if the given repo-relative path is safe to commit."""
    norm = path.replace("\\", "/")
    # Strip a single leading "./" if present (not every leading "." char)
    if norm.startswith("./"):
        norm = norm[2:]
    for allowed in WHITELIST_PATHS:
        if allowed.endswith("/"):
            if norm.startswith(allowed):
                return True
        else:
            if norm == allowed:
                return True
    return False


def get_current_branch(cwd: Optional[Path] = None) -> Optional[str]:
    """Return the current branch name, or None on error."""
    try:
        proc = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, timeout=10)
        return proc.stdout.strip() or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("git_sync.branch_lookup_failed", error=str(e))
        return None


def has_disallowed_changes(cwd: Optional[Path] = None) -> List[str]:
    """List any modified/untracked files OUTSIDE the whitelist.

    Returns an empty list if the working tree is clean or only has whitelisted
    changes. A non-empty list means we must abort commit to protect src/.
    """
    try:
        proc = _run_git(
            ["status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=cwd,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("git_sync.status_failed", error=str(e))
        return ["<git-status-failed>"]

    disallowed: List[str] = []
    for line in proc.stdout.splitlines():
        # Porcelain format: "XY path" where XY is 2-char status
        if len(line) < 4:
            continue
        path = line[3:].strip()
        # Handle rename syntax "old -> new"
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        # Strip quoted paths
        path = path.strip('"')
        if not _is_whitelisted(path):
            disallowed.append(path)
    return disallowed


def commit_and_push_self_improvements(
    artifacts: Iterable[str],
    trade_id: str,
    hypothesis: str,
    mode: str = "lightweight",
    cwd: Optional[Path] = None,
    retries: int = len(RETRY_DELAYS),
    dry_run: bool = False,
) -> bool:
    """Stage + commit + push the given artifact paths.

    Aborts (returns False) if:
      - The working tree has non-whitelisted dirty files (protects src/)
      - Any of the supplied artifacts is not whitelisted
      - Git add fails
      - Commit returns nonzero for any reason except "nothing to commit"
      - Push fails after retries exhausted

    Never raises — returns bool, logs all failures.
    """
    artifacts = [str(a) for a in artifacts if a]
    if not artifacts:
        logger.info("git_sync.no_artifacts", trade_id=trade_id)
        return True  # Nothing to do is not a failure

    # Refuse to commit any non-whitelisted artifact. Defense-in-depth: the skill
    # prompt restricts writes, but the LLM might hallucinate a path.
    rejected = [a for a in artifacts if not _is_whitelisted(a)]
    if rejected:
        logger.error(
            "git_sync.rejected_artifacts",
            trade_id=trade_id,
            rejected=rejected,
        )
        return False

    # Detect any DIRTY state outside the whitelist.
    # If Buddy itself has unstaged src/ changes, we must NOT accidentally commit them.
    disallowed = has_disallowed_changes(cwd=cwd)
    if disallowed:
        logger.warning(
            "git_sync.dirty_tree_skipped",
            trade_id=trade_id,
            disallowed=disallowed[:10],
        )
        return False

    if dry_run:
        logger.info("git_sync.dry_run", trade_id=trade_id, artifacts=artifacts)
        return True

    # Stage each artifact explicitly — never `git add -A` (CLAUDE.md rule)
    try:
        for path in artifacts:
            if not Path(path).exists():
                logger.warning("git_sync.artifact_missing", path=path)
                continue
            _run_git(["add", "--", path], cwd=cwd, timeout=10)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("git_sync.add_failed", trade_id=trade_id, error=str(e))
        return False

    # Nothing to commit? Return True (idempotent).
    try:
        diff = _run_git(
            ["diff", "--cached", "--name-only"], cwd=cwd, timeout=10, check=False
        )
        if not diff.stdout.strip():
            logger.info("git_sync.nothing_staged", trade_id=trade_id)
            return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass  # Proceed to commit; it will surface any real error

    # Compose commit message
    short_hyp = (hypothesis or "").replace("\n", " ").strip()[:120]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    msg = (
        f"{COMMIT_PREFIX} {mode} reflection on trade {trade_id}\n"
        f"\n"
        f"hypothesis: {short_hyp}\n"
        f"mode: {mode}\n"
        f"timestamp: {ts}\n"
    )

    try:
        _run_git(["commit", "-m", msg], cwd=cwd, timeout=30)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "").strip()
        # "nothing to commit, working tree clean" is an acceptable race
        if "nothing to commit" in err:
            return True
        logger.error("git_sync.commit_failed", trade_id=trade_id, stderr=err[:500])
        return False
    except subprocess.TimeoutExpired as e:
        logger.error("git_sync.commit_timeout", trade_id=trade_id, error=str(e))
        return False

    # Push with retry + exponential backoff (CLAUDE.md network contract)
    branch = get_current_branch(cwd=cwd)
    if not branch:
        logger.error("git_sync.no_branch", trade_id=trade_id)
        return False

    for attempt in range(retries + 1):
        try:
            _run_git(["push", "-u", "origin", branch], cwd=cwd, timeout=DEFAULT_TIMEOUT)
            logger.info(
                "git_sync.pushed",
                trade_id=trade_id,
                branch=branch,
                attempts=attempt + 1,
            )
            return True
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            # Non-fast-forward: pull --rebase, retry ONCE
            if ("non-fast-forward" in stderr or "rejected" in stderr) and attempt == 0:
                logger.warning("git_sync.push_non_ff_rebasing", trade_id=trade_id)
                if not pull_latest(cwd=cwd, rebase=True):
                    logger.error(
                        "git_sync.rebase_pull_failed", trade_id=trade_id, stderr=stderr[:500]
                    )
                    return False
                continue  # Retry push after rebase
            # Other errors — retry with backoff if attempts remain
            if attempt < retries:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                logger.warning(
                    "git_sync.push_retry",
                    trade_id=trade_id,
                    attempt=attempt + 1,
                    delay=delay,
                    stderr=stderr[:200],
                )
                time.sleep(delay)
                continue
            logger.error(
                "git_sync.push_failed_final", trade_id=trade_id, stderr=stderr[:500]
            )
            return False
        except subprocess.TimeoutExpired as e:
            if attempt < retries:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                logger.warning(
                    "git_sync.push_timeout_retry",
                    trade_id=trade_id,
                    attempt=attempt + 1,
                    delay=delay,
                )
                time.sleep(delay)
                continue
            logger.error("git_sync.push_timeout_final", trade_id=trade_id, error=str(e))
            return False

    return False


def pull_latest(
    cwd: Optional[Path] = None,
    rebase: bool = False,
    retries: int = len(RETRY_DELAYS),
) -> bool:
    """Pull the current branch from origin with retry + backoff.

    Skips (returns True) if the working tree is dirty with non-whitelisted changes —
    prevents silent clobbering of in-progress src/ edits.
    """
    branch = get_current_branch(cwd=cwd)
    if not branch:
        return False

    disallowed = has_disallowed_changes(cwd=cwd)
    if disallowed:
        logger.warning(
            "git_sync.pull_skipped_dirty", branch=branch, disallowed=disallowed[:10]
        )
        return True  # Skip, not a hard error

    args = ["pull", "--rebase" if rebase else "--no-rebase", "origin", branch]

    for attempt in range(retries + 1):
        try:
            _run_git(args, cwd=cwd, timeout=DEFAULT_TIMEOUT)
            logger.info("git_sync.pulled", branch=branch, attempts=attempt + 1)
            return True
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            if attempt < retries:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                logger.warning(
                    "git_sync.pull_retry",
                    branch=branch,
                    attempt=attempt + 1,
                    delay=delay,
                    stderr=stderr[:200],
                )
                time.sleep(delay)
                continue
            logger.error("git_sync.pull_failed_final", branch=branch, stderr=stderr[:500])
            return False
        except subprocess.TimeoutExpired:
            if attempt < retries:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                time.sleep(delay)
                continue
            logger.error("git_sync.pull_timeout_final", branch=branch)
            return False

    return False


__all__ = [
    "WHITELIST_PATHS",
    "commit_and_push_self_improvements",
    "pull_latest",
    "get_current_branch",
    "has_disallowed_changes",
]
