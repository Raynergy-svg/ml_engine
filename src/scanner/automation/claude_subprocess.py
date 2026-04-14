"""Claude CLI subprocess invocation helper for autonomous self-improvement.

Used by ClaudeReflectionHandler to spawn headless Claude on trade close
and capture structured output (promise tags + YAML result blocks).

Design rules (CLAUDE.md compliance):
  - Explicit timeouts on every subprocess call (never block forever)
  - Catch specific exceptions (TimeoutExpired, OSError, FileNotFoundError)
  - Log retry attempts with context
  - Never swallow errors silently — raise or return structured error
  - Single-flight lock prevents parallel reflection spawns
  - Daily budget cap prevents cost runaway

Reuses the battle-tested invocation pattern from scripts/ralph.sh:244 and
src/scanner/automation/prd_agent_chain.py:445 — `claude --print
--dangerously-skip-permissions` with stdin prompt delivery.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

# Default Claude CLI invocation — matches existing patterns in the repo
CLAUDE_CLI = "claude"
DEFAULT_FLAGS = ["--dangerously-skip-permissions", "--print"]

# Single-flight lock prevents two reflections from running in parallel.
# Burst trade closes (e.g., 3 positions hit SL simultaneously) would otherwise
# spawn three parallel Claudes, wasting tokens and racing on file writes.
LOCK_PATH = Path(".claude/.reflection.lock")

# Daily budget state — tracks spend to enforce cost ceiling
BUDGET_PATH = Path(".claude/reflection_budget.json")

# Observability trail — every reflection logged here for audit + cost tracking
REFLECTION_LOG = Path("logs/reflection_log.jsonl")

# Regex to extract structured output block from Claude stdout
RESULT_BLOCK_RE = re.compile(
    r"<reflection-result>(.*?)</reflection-result>",
    re.DOTALL | re.IGNORECASE,
)
PROMISE_RE = re.compile(r"<promise>REFLECTION_COMPLETE</promise>", re.IGNORECASE)


@dataclass
class ReflectionResult:
    """Structured outcome of a Claude reflection subprocess call."""
    success: bool
    trade_id: str
    mode: str  # "lightweight" or "deep"
    duration_seconds: float
    stdout_len: int
    promise_found: bool = False
    result_block: Optional[str] = None
    artifacts_written: List[str] = field(default_factory=list)
    hypothesis: Optional[str] = None
    cost_usd: float = 0.0
    error: Optional[str] = None
    returncode: int = -1


class ReflectionBudget:
    """Daily token-budget tracker for Claude reflection spawns.

    Persists to .claude/reflection_budget.json with the schema:
      {"date": "YYYY-MM-DD", "spent_usd": float, "spawn_count": int, "deep_count": int}

    Resets automatically on date rollover.
    """
    def __init__(self, path: Path = BUDGET_PATH, daily_cap_usd: float = 5.0):
        self.path = path
        self.daily_cap_usd = daily_cap_usd

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def load(self) -> Dict[str, Any]:
        """Load today's budget state. Resets on date rollover. Never crashes."""
        today = self._today()
        default = {"date": today, "spent_usd": 0.0, "spawn_count": 0, "deep_count": 0}
        try:
            if not self.path.exists():
                return default
            with open(self.path, "r") as f:
                data = json.load(f)
            # Date rollover — reset
            if data.get("date") != today:
                return default
            # Defensive: ensure required keys present
            for k, v in default.items():
                data.setdefault(k, v)
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("reflection_budget.load_failed", error=str(e), path=str(self.path))
            return default

    def save(self, state: Dict[str, Any]) -> None:
        """Atomic write with tempfile + rename (CLAUDE.md JSON-safety gate)."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", dir=str(self.path.parent), prefix=".budget_", delete=False
            ) as tmp:
                json.dump(state, tmp, indent=2, sort_keys=True)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = tmp.name
            os.rename(tmp_path, self.path)
        except OSError as e:
            logger.warning("reflection_budget.save_failed", error=str(e))

    def allows(self, mode: str) -> bool:
        """Return True if a new reflection of given mode is within budget."""
        state = self.load()
        if state["spent_usd"] >= self.daily_cap_usd:
            logger.warning(
                "reflection_budget.exhausted",
                spent_usd=state["spent_usd"],
                cap_usd=self.daily_cap_usd,
            )
            return False
        # Deep-dives have a stricter sub-cap: max 20/day
        if mode == "deep" and state["deep_count"] >= 20:
            logger.warning(
                "reflection_budget.deep_cap_reached", deep_count=state["deep_count"]
            )
            return False
        return True

    def record(self, cost_usd: float, mode: str) -> None:
        state = self.load()
        state["spent_usd"] = float(state["spent_usd"]) + float(cost_usd)
        state["spawn_count"] = int(state["spawn_count"]) + 1
        if mode == "deep":
            state["deep_count"] = int(state["deep_count"]) + 1
        self.save(state)


class SingleFlightLock:
    """Non-blocking pidfile lock. If another reflection is in flight, skip."""
    def __init__(self, path: Path = LOCK_PATH, stale_after_seconds: int = 600):
        self.path = path
        self.stale_after_seconds = stale_after_seconds
        self._acquired = False

    def acquire(self) -> bool:
        """Return True if lock acquired (caller proceeds), False if already held."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                # Check staleness — if lock file is older than threshold, assume dead pid
                age = time.time() - self.path.stat().st_mtime
                if age > self.stale_after_seconds:
                    logger.warning(
                        "reflection_lock.stale_cleared",
                        age_seconds=age,
                        path=str(self.path),
                    )
                    self.path.unlink(missing_ok=True)
                else:
                    return False
            # O_EXCL ensures atomic create — avoids TOCTOU race
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self._acquired = True
            return True
        except FileExistsError:
            return False
        except OSError as e:
            logger.warning("reflection_lock.acquire_failed", error=str(e))
            return False

    def release(self) -> None:
        if self._acquired:
            try:
                self.path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("reflection_lock.release_failed", error=str(e))
            self._acquired = False

    def __enter__(self):
        self._got = self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    @property
    def acquired(self) -> bool:
        return self._acquired


def _parse_result_block(stdout: str) -> Dict[str, Any]:
    """Extract the <reflection-result>...</reflection-result> YAML-ish block.

    Returns {} if missing or malformed. Never raises — caller handles empty dict.
    We parse it as a loose key: value mapping (not strict YAML) to avoid adding
    a yaml dep; Claude's output follows a simple flat structure.
    """
    match = RESULT_BLOCK_RE.search(stdout or "")
    if not match:
        return {}

    block_text = match.group(1).strip()
    result: Dict[str, Any] = {}

    # Try strict JSON first (Claude may emit JSON object)
    try:
        if block_text.lstrip().startswith("{"):
            return json.loads(block_text)
    except json.JSONDecodeError:
        pass

    # Fallback: line-by-line "key: value" + "  - item" lists
    current_list_key: Optional[str] = None
    for line in block_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # List item under previous key
        if stripped.startswith("- ") and current_list_key:
            result.setdefault(current_list_key, []).append(stripped[2:].strip())
            continue
        # key: value
        if ":" in stripped:
            k, _, v = stripped.partition(":")
            key = k.strip()
            val = v.strip()
            if not val:
                # value continues as list on next lines
                current_list_key = key
                result[key] = []
            else:
                current_list_key = None
                # Try numeric coerce
                try:
                    if "." in val:
                        result[key] = float(val)
                    else:
                        result[key] = int(val)
                except ValueError:
                    result[key] = val.strip("\"'")
    return result


def _append_reflection_log(result: ReflectionResult, prompt_preview: str) -> None:
    """Append one JSONL entry to logs/reflection_log.jsonl (observability)."""
    try:
        REFLECTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trade_id": result.trade_id,
            "mode": result.mode,
            "success": result.success,
            "duration_seconds": round(result.duration_seconds, 2),
            "stdout_len": result.stdout_len,
            "returncode": result.returncode,
            "promise_found": result.promise_found,
            "artifacts_written": result.artifacts_written,
            "hypothesis": result.hypothesis,
            "cost_usd": result.cost_usd,
            "error": result.error,
            "prompt_preview": prompt_preview[:500],
        }
        with open(REFLECTION_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        logger.warning("reflection_log.append_failed", error=str(e))


def invoke_claude_reflection(
    prompt: str,
    trade_id: str,
    mode: str = "lightweight",
    timeout_seconds: int = 60,
    cwd: Optional[Path] = None,
) -> ReflectionResult:
    """Spawn headless Claude CLI with the reflection prompt.

    Args:
        prompt: Full prompt text delivered via stdin.
        trade_id: Trade identifier for logging/observability.
        mode: "lightweight" (fast, cheap, <60s) or "deep" (full analysis, <300s).
        timeout_seconds: Hard kill timeout for the subprocess.
        cwd: Working directory (defaults to project root if None).

    Returns:
        ReflectionResult with success status, parsed output, duration.
        Never raises — all errors captured in result.error.
    """
    start = time.time()
    result = ReflectionResult(
        success=False,
        trade_id=trade_id,
        mode=mode,
        duration_seconds=0.0,
        stdout_len=0,
    )

    # Deliver prompt via tempfile stdin — same pattern as scripts/ralph.sh:244.
    # Shell heredoc avoids argument-length limits and shell-escaping bugs.
    prompt_file: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", prefix="reflect_", delete=False, dir="/tmp"
        ) as tf:
            tf.write(prompt)
            prompt_file = tf.name

        logger.info(
            "claude_reflection.spawn",
            trade_id=trade_id,
            mode=mode,
            prompt_bytes=len(prompt),
            timeout=timeout_seconds,
        )

        with open(prompt_file, "r") as stdin_f:
            proc = subprocess.run(
                [CLAUDE_CLI, *DEFAULT_FLAGS],
                stdin=stdin_f,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(cwd) if cwd else None,
                # Inherit env so Claude can find its config / auth
                env=os.environ.copy(),
            )
        result.returncode = proc.returncode
        stdout = proc.stdout or ""
        result.stdout_len = len(stdout)

        # Parse structured output
        result.promise_found = bool(PROMISE_RE.search(stdout))
        block = _parse_result_block(stdout)
        if block:
            result.result_block = json.dumps(block)
            artifacts = block.get("artifacts_written") or []
            if isinstance(artifacts, list):
                result.artifacts_written = [str(a) for a in artifacts]
            result.hypothesis = block.get("hypothesis")
            cost = block.get("cost_usd", 0.0)
            try:
                result.cost_usd = float(cost) if cost is not None else 0.0
            except (TypeError, ValueError):
                result.cost_usd = 0.0

        # Success requires both: non-zero artifacts OR promise tag, AND returncode 0
        result.success = (
            proc.returncode == 0
            and (result.promise_found or bool(result.artifacts_written))
        )
        if not result.success and proc.returncode != 0:
            result.error = f"claude exited {proc.returncode}: {(proc.stderr or '')[:300]}"

    except subprocess.TimeoutExpired:
        result.error = f"claude timed out after {timeout_seconds}s"
        logger.warning("claude_reflection.timeout", trade_id=trade_id, timeout=timeout_seconds)
    except FileNotFoundError:
        result.error = "claude CLI not found on PATH"
        logger.error("claude_reflection.cli_missing")
    except OSError as e:
        result.error = f"subprocess OSError: {e}"
        logger.error("claude_reflection.os_error", error=str(e))
    finally:
        result.duration_seconds = time.time() - start
        if prompt_file:
            try:
                os.unlink(prompt_file)
            except OSError:
                pass

    _append_reflection_log(result, prompt)
    return result


__all__ = [
    "ReflectionResult",
    "ReflectionBudget",
    "SingleFlightLock",
    "invoke_claude_reflection",
    "REFLECTION_LOG",
    "LOCK_PATH",
    "BUDGET_PATH",
]
