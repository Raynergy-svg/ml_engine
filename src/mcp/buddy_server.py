"""Buddy MCP server — 15 tools for Claude orchestrator observability.

Exposes the feedback loop, diagnostics, self-heal, OANDA state, and
agent/gate health via the Model Context Protocol (stdio transport).

Uses the official ``mcp`` Python SDK (FastMCP high-level API).
Per MCP spec 2025-06-18: typed Pydantic return models on every tool,
no JSON-RPC batching. OANDA credentials from env only.

Launch:
    python src/mcp/buddy_server.py          # stdio (default)
    # or via mcp.json mcpServers entry
"""

from __future__ import annotations

import json
import sys
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Self-load .env.local so OANDA_API_TOKEN / OANDA_ACCOUNT_ID / etc are
# available regardless of how the server is launched.
#
# Audit 2026-04-28: mcp.json had `"OANDA_API_URL": "${OANDA_API_URL}"`
# which Claude Code's MCP config doesn't expand — the LITERAL string
# `${OANDA_API_URL}` was being passed as the env value, producing
# "Invalid URL '${OANDA_API_URL}/v3/...'" errors when the server
# tried to talk to OANDA. We now read .env.local directly so spawn-
# time env injection isn't required.
# ------------------------------------------------------------------ #
def _load_env_local() -> None:
    """Best-effort .env.local loader; falls back silently on any failure."""
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env.local"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            # Strip optional `export ` prefix.
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Don't clobber values already present in the process env —
            # operator-set values take precedence over .env.local.
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as e:  # noqa: BLE001 — non-fatal, log + continue
        logger.warning("buddy_mcp: .env.local load failed: %r", e)


_load_env_local()

# Ensure project root is in Python path so src.* imports resolve regardless
# of how the server is launched (direct, mcp.json, or subprocess).
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Sanitize known-broken values: if mcp.json injected a literal
# "${VAR}" placeholder (because Claude Code's MCP config doesn't
# expand placeholders), treat it as unset so getenv() falls through
# to the .env.local-loaded value or hardcoded default.
for _key in ("OANDA_API_TOKEN", "OANDA_ACCOUNT_ID", "OANDA_API_URL"):
    _val = os.environ.get(_key)
    if isinstance(_val, str) and _val.startswith("${") and _val.endswith("}"):
        logger.warning(
            "buddy_mcp: env %s contains literal placeholder %r — clearing so "
            ".env.local / defaults can apply", _key, _val,
        )
        os.environ.pop(_key, None)

# ------------------------------------------------------------------ #
# Pydantic return models (structured tool outputs per 2025-06-18 spec)
# ------------------------------------------------------------------ #


class LogRecords(BaseModel):
    """Tail of a JSONL log file."""
    count: int = 0
    records: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class DiagResult(BaseModel):
    """Output of PostTradeDiagnostics.run()."""
    status: str = "UNKNOWN"
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class HealResult(BaseModel):
    """Output of SelfHeal.apply()."""
    status: str = "UNKNOWN"
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    files_modified: List[str] = Field(default_factory=list)
    degraded_mode: bool = False
    error: Optional[str] = None


class HealWithDiag(BaseModel):
    """Combined diagnostics + self-heal result."""
    diagnostics: DiagResult
    heal: HealResult


class OperationResult(BaseModel):
    """Generic operation outcome."""
    status: str
    path: Optional[str] = None
    error: Optional[str] = None


class HealthCheckResult(BaseModel):
    """Aggregate health snapshot."""
    diagnostics: DiagResult
    last_feedback: Optional[Dict[str, Any]] = None
    last_decision: Optional[Dict[str, Any]] = None
    agent_weights_count: int = 0
    rl_model_age_hours: Optional[float] = None
    overall: str = "UNKNOWN"


# ------------------------------------------------------------------ #
# Path constants
# ------------------------------------------------------------------ #
FEEDBACK_LOG = Path("logs/trade_feedback.jsonl")
DECISION_LOG = Path("logs/system_decisions.jsonl")
AGENT_WEIGHTS_PATH = Path("trained_data/models/agent_weights.json")
RL_MODEL_PATH = Path("trained_data/models/rl_position_sizer.zip")
LEARNINGS_PATH = Path(".claude/learnings.md")
RULES_DIR = Path(".claude/rules")
OANDA_DEFAULT_URL = "https://api-fxpractice.oanda.com"

# ------------------------------------------------------------------ #
# FastMCP server
# ------------------------------------------------------------------ #
mcp = FastMCP("buddy")


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _tail_jsonl(path: Path, n: int = 50) -> LogRecords:
    """Read the last N lines of a JSONL file, parse each as JSON."""
    if not path.exists():
        return LogRecords(count=0, records=[])
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        tail = lines[-n:] if len(lines) > n else lines
        records: List[Dict[str, Any]] = []
        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"_raw": line})
        return LogRecords(count=len(records), records=records)
    except Exception as exc:
        return LogRecords(error=str(exc))


def _oanda_headers() -> Optional[Dict[str, str]]:
    """Build OANDA auth headers from env. Returns None if creds missing."""
    token = os.getenv("OANDA_API_TOKEN", "")
    if not token:
        return None
    return {
        "Authorization": "Bearer {0}".format(token),
        "Content-Type": "application/json",
    }


def _oanda_base_url() -> str:
    return os.getenv("OANDA_API_URL", OANDA_DEFAULT_URL)


def _oanda_account_id() -> str:
    return os.getenv("OANDA_ACCOUNT_ID", "")


# ------------------------------------------------------------------ #
# Auth-header URL allowlist (CWE-522 defense — audit 2026-04-28).
#
# Semgrep "Authentication Credential Pass-Through" rule correctly
# observed that requests.get(url, headers=auth_headers) is unsafe when
# `url` is influenced by user-controlled input. In our case `url` comes
# from OANDA_API_URL env which the operator *should* control, but the
# `${OANDA_API_URL}` literal-placeholder bug we just found in mcp.json
# proves env-var assumptions can fail. If the env ever points at an
# attacker domain (config tampering, MCP-config injection, env-var
# clobber by another tool), the Bearer token leaks to that domain.
#
# Defense: refuse to attach auth headers unless the resolved URL hostname
# is on this hardcoded allowlist. Failure mode is a clean error response,
# not credential leakage.
# ------------------------------------------------------------------ #
_TRUSTED_OANDA_HOSTS: frozenset = frozenset({
    "api-fxpractice.oanda.com",
    "api-fxtrade.oanda.com",
    "stream-fxpractice.oanda.com",
    "stream-fxtrade.oanda.com",
})


def _assert_trusted_oanda_url(url: str) -> Optional[Dict[str, Any]]:
    """Guard for any HTTP call that will carry OANDA auth headers.

    Returns None when `url` parses to a trusted OANDA host (caller may
    proceed). Returns a structured error dict otherwise — caller should
    return that dict directly without ever sending the request.

    Refuses anything except https + a known OANDA host. Also rejects
    URLs containing the literal `${...}` placeholder pattern (defense
    against the exact mcp.json bug that prompted this guard).
    """
    if not isinstance(url, str) or not url:
        return {"error": "blocked_url", "detail": "empty url"}
    if "${" in url:
        return {
            "error": "blocked_url",
            "detail": "url contains literal env placeholder — env var injection failed",
            "url": url,
        }
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
    except Exception as e:  # noqa: BLE001
        return {"error": "blocked_url", "detail": "url parse failed: {0}".format(e)}
    if parsed.scheme != "https":
        return {
            "error": "blocked_url",
            "detail": "scheme must be https; got {0!r}".format(parsed.scheme),
        }
    host = (parsed.hostname or "").lower()
    if host not in _TRUSTED_OANDA_HOSTS:
        return {
            "error": "blocked_url",
            "detail": (
                "host {0!r} is not an allowlisted OANDA host — refusing to "
                "send auth headers. Allowlist: {1}"
            ).format(host, sorted(_TRUSTED_OANDA_HOSTS)),
        }
    return None


# ------------------------------------------------------------------ #
# Tool 1: get_system_status
# ------------------------------------------------------------------ #
@mcp.tool()
def get_system_status() -> Dict[str, Any]:
    """Get lightweight system status without instantiating Orchestrator.

    Reads .claude/state.json and checks module file presence for
    dashboards and health probes. Avoids heavy W&B/PRD init.
    """
    try:
        state_path = Path(".claude/state.json")
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
    except Exception:
        state = {}

    return {
        "system": {
            "goal": state.get("goal", "unknown"),
            "mode": state.get("mode", "unknown"),
            "halted": state.get("halted", False),
            "config_dirty": state.get("config_dirty", False),
            "last_updated": state.get("last_updated", ""),
        },
        "portfolio": state.get("portfolio_snapshot", {}),
        "modules": {
            "agent_weights": AGENT_WEIGHTS_PATH.exists(),
            "rl_model": RL_MODEL_PATH.exists(),
            "feedback_log": FEEDBACK_LOG.exists(),
            "decision_log": DECISION_LOG.exists(),
            "learnings": LEARNINGS_PATH.exists(),
            "journal": Path("trained_data/trade_journal_rl.json").exists(),
            "state_engine": Path(".claude/state.json").exists(),
        },
        "cycles": {
            "completed": state.get("done", []),
            "next": state.get("next", ""),
        },
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server": "buddy_mcp",
        },
    }


# ------------------------------------------------------------------ #
# Tool 2: get_trade_feedback_log
# ------------------------------------------------------------------ #
@mcp.tool()
def get_trade_feedback_log(n: int = 50) -> LogRecords:
    """Read the last N records from the trade feedback log.

    Each record contains shaped_reward, pnl_pips, slippage, MAE,
    regime, and the source that triggered the feedback loop.
    """
    return _tail_jsonl(FEEDBACK_LOG, n)


# ------------------------------------------------------------------ #
# Tool 3: get_system_decisions
# ------------------------------------------------------------------ #
@mcp.tool()
def get_system_decisions(n: int = 50) -> LogRecords:
    """Read the last N records from the system decisions log.

    Each record shows a self-heal action: trigger, issue,
    action_taken, files_modified, and degraded_mode status.
    """
    return _tail_jsonl(DECISION_LOG, n)


# ------------------------------------------------------------------ #
# Tool 4: get_agent_weights
# ------------------------------------------------------------------ #
@mcp.tool()
def get_agent_weights() -> Dict[str, Any]:
    """Read the current agent weights from agent_weights.json.

    Returns the full dict: {regime_bucket: {agent_name: float}, _meta: {...}}.
    """
    if not AGENT_WEIGHTS_PATH.exists():
        return {"error": "agent_weights.json not found"}
    try:
        with AGENT_WEIGHTS_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return {"error": str(exc)}


# ------------------------------------------------------------------ #
# Tool 5: get_gate_health
# ------------------------------------------------------------------ #
@mcp.tool()
def get_gate_health() -> Dict[str, Any]:
    """Run diagnostics and extract gate firing rate issues.

    Returns only the gate-related subset of the diagnostics report
    plus current threshold values.
    """
    try:
        from src.scanner.feedback.diagnostics import PostTradeDiagnostics
        diag = PostTradeDiagnostics()
        result = diag.run()
        gate_issues = [
            i for i in result.get("issues", [])
            if isinstance(i, dict) and i.get("check") == "gate_firing_rate"
        ]
        return {
            "status": result.get("status", "UNKNOWN"),
            "gate_issues": gate_issues,
            "thresholds": {
                "gate_broken_off_lt": diag._thresholds.get("gate_broken_off_lt"),
                "gate_broken_on_gt": diag._thresholds.get("gate_broken_on_gt"),
                "gate_window": diag._thresholds.get("gate_window"),
            },
        }
    except Exception as exc:
        logger.warning("get_gate_health failed: %r", exc)
        return {"error": str(exc)}


# ------------------------------------------------------------------ #
# Tool 6: get_learnings
# ------------------------------------------------------------------ #
@mcp.tool()
def get_learnings() -> Dict[str, Any]:
    """Read .claude/learnings.md content.

    If the file exceeds 100 KB, returns only the tail (last entries).
    """
    if not LEARNINGS_PATH.exists():
        return {"content": "", "line_count": 0, "error": "file not found"}
    try:
        text = LEARNINGS_PATH.read_text(encoding="utf-8")
        lines = text.splitlines()
        if len(text) > 100_000:
            # Return last ~100 entries
            tail_lines = lines[-100:]
            return {
                "content": "\n".join(tail_lines),
                "line_count": len(lines),
                "truncated": True,
            }
        return {"content": text, "line_count": len(lines)}
    except Exception as exc:
        return {"content": "", "line_count": 0, "error": str(exc)}


# ------------------------------------------------------------------ #
# Tool 7: trigger_diagnostics
# ------------------------------------------------------------------ #
@mcp.tool()
def trigger_diagnostics() -> DiagResult:
    """Run the full PostTradeDiagnostics suite.

    Checks direction accuracy, gate firing rates, agent weight
    boundaries, drawdown streaks, and RL model staleness.
    """
    try:
        from src.scanner.feedback.diagnostics import PostTradeDiagnostics
        result = PostTradeDiagnostics().run()
        return DiagResult(**result)
    except Exception as exc:
        logger.warning("trigger_diagnostics failed: %r", exc)
        return DiagResult(status="ERROR", error=str(exc))


# ------------------------------------------------------------------ #
# Tool 8: trigger_self_heal
# ------------------------------------------------------------------ #
@mcp.tool()
def trigger_self_heal(
    diag_result: Optional[Dict[str, Any]] = None,
) -> HealWithDiag:
    """Run self-heal on a diagnostics result.

    If diag_result is not provided, runs diagnostics first and then
    applies self-heal on the result. Returns both the diagnostics
    and the heal outcome.
    """
    try:
        from src.scanner.feedback.diagnostics import PostTradeDiagnostics
        from src.scanner.feedback.self_heal import SelfHeal

        if diag_result is None:
            diag_result = PostTradeDiagnostics().run()

        heal_result = SelfHeal().apply(diag_result)
        return HealWithDiag(
            diagnostics=DiagResult(**diag_result),
            heal=HealResult(**heal_result),
        )
    except Exception as exc:
        logger.warning("trigger_self_heal failed: %r", exc)
        return HealWithDiag(
            diagnostics=DiagResult(status="ERROR", error=str(exc)),
            heal=HealResult(status="error", error=str(exc)),
        )


# ------------------------------------------------------------------ #
# Tool 9: trigger_rl_update
# ------------------------------------------------------------------ #
@mcp.tool()
def trigger_rl_update(trade_id: str) -> Dict[str, Any]:
    """Run the post-trade feedback loop for a specific trade.

    Fires logging, diagnostics, and self-heal for the given OANDA
    trade ID. Does NOT update agent weights (those stay on the
    batch sync path).
    """
    try:
        from src.scanner.feedback.post_trade_loop import PostTradeLoop
        return PostTradeLoop().run({
            "trade_id": trade_id,
            "source": "mcp_trigger",
        })
    except Exception as exc:
        logger.warning("trigger_rl_update failed: %r", exc)
        return {"status": "error", "trade_id": trade_id, "error": str(exc)}


# ------------------------------------------------------------------ #
# Tool 10: trigger_gate_retrain
# ------------------------------------------------------------------ #
@mcp.tool()
def trigger_gate_retrain() -> Dict[str, Any]:
    """Trigger gate model retraining via OnlineRetrainer.

    Respects the 60-minute cooldown. Returns the retrainer's result
    dict or an error if the retrainer is unavailable.
    """
    try:
        from online_retrainer import OnlineRetrainer
        retrainer = OnlineRetrainer()
        result = retrainer.trigger_retrain(
            X_replay=None,
            y_replay=None,
            reason="mcp_trigger",
            force=False,
        )
        return result if isinstance(result, dict) else {"status": str(result)}
    except ImportError:
        return {"error": "online_retrainer not available"}
    except Exception as exc:
        logger.warning("trigger_gate_retrain failed: %r", exc)
        return {"error": str(exc)}


# ------------------------------------------------------------------ #
# Tool 11: write_learning
# ------------------------------------------------------------------ #
@mcp.tool()
def write_learning(content: str) -> OperationResult:
    """Append a dated learning entry to .claude/learnings.md.

    Content must be non-empty and under 5000 characters.
    """
    if not content or not content.strip():
        return OperationResult(status="rejected", error="content is empty")
    if len(content) > 5000:
        return OperationResult(
            status="rejected",
            error="content exceeds 5000 chars ({0})".format(len(content)),
        )
    try:
        LEARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        entry = "- [{0}] **MCP_LEARNING**: {1}\n".format(
            date_str, content.strip(),
        )
        with LEARNINGS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(entry)
        return OperationResult(status="appended", path=str(LEARNINGS_PATH))
    except Exception as exc:
        return OperationResult(status="error", error=str(exc))


# ------------------------------------------------------------------ #
# Tool 12: write_rule
# ------------------------------------------------------------------ #
@mcp.tool()
def write_rule(filename: str, content: str) -> OperationResult:
    """Write a new rule file under .claude/rules/.

    Filename must match [a-z0-9_]+.md. Cannot overwrite trading.md
    or improvement.md (core rules). Any other filename is allowed.
    """
    if not re.match(r"^[a-z0-9_]+\.md$", filename):
        return OperationResult(
            status="rejected",
            error="filename must match ^[a-z0-9_]+\\.md$ — got: {0}".format(
                filename,
            ),
        )
    if filename in ("trading.md", "improvement.md"):
        return OperationResult(
            status="rejected",
            error="cannot overwrite core rule file: {0}".format(filename),
        )
    if not content or not content.strip():
        return OperationResult(status="rejected", error="content is empty")
    try:
        target = RULES_DIR / filename
        RULES_DIR.mkdir(parents=True, exist_ok=True)
        target.write_text(content.strip() + "\n", encoding="utf-8")
        return OperationResult(status="written", path=str(target))
    except Exception as exc:
        return OperationResult(status="error", error=str(exc))


# ------------------------------------------------------------------ #
# Tool 13: get_open_positions
# ------------------------------------------------------------------ #
@mcp.tool()
def get_open_positions() -> Dict[str, Any]:
    """Get open positions from OANDA.

    Reads OANDA_API_TOKEN and OANDA_ACCOUNT_ID from environment.
    Returns the raw OANDA API response or an error dict.
    """
    import requests

    headers = _oanda_headers()
    if headers is None:
        return {"error": "credentials_missing", "detail": "OANDA_API_TOKEN not set"}
    acct = _oanda_account_id()
    if not acct:
        return {"error": "credentials_missing", "detail": "OANDA_ACCOUNT_ID not set"}

    url = "{0}/v3/accounts/{1}/openTrades".format(_oanda_base_url(), acct)
    blocked = _assert_trusted_oanda_url(url)
    if blocked is not None:
        return blocked
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return {"error": "oanda_error", "status": resp.status_code, "body": resp.text}
    except requests.Timeout:
        return {"error": "timeout", "url": url}
    except requests.ConnectionError as exc:
        return {"error": "connection_error", "detail": str(exc)}
    except Exception as exc:
        return {"error": str(exc)}


# ------------------------------------------------------------------ #
# Tool 14: get_closed_trades
# ------------------------------------------------------------------ #
@mcp.tool()
def get_closed_trades(count: int = 50) -> Dict[str, Any]:
    """Get recently closed trades from OANDA.

    Args:
        count: Number of closed trades to retrieve (default 50, max 500).
    """
    import requests

    count = max(1, min(count, 500))
    headers = _oanda_headers()
    if headers is None:
        return {"error": "credentials_missing", "detail": "OANDA_API_TOKEN not set"}
    acct = _oanda_account_id()
    if not acct:
        return {"error": "credentials_missing", "detail": "OANDA_ACCOUNT_ID not set"}

    url = "{0}/v3/accounts/{1}/trades?state=CLOSED&count={2}".format(
        _oanda_base_url(), acct, count,
    )
    blocked = _assert_trusted_oanda_url(url)
    if blocked is not None:
        return blocked
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return {"error": "oanda_error", "status": resp.status_code, "body": resp.text}
    except requests.Timeout:
        return {"error": "timeout", "url": url}
    except requests.ConnectionError as exc:
        return {"error": "connection_error", "detail": str(exc)}
    except Exception as exc:
        return {"error": str(exc)}


# ------------------------------------------------------------------ #
# Tool 15: health_check
# ------------------------------------------------------------------ #
@mcp.tool()
def health_check() -> HealthCheckResult:
    """Aggregate health snapshot of the entire system.

    Combines diagnostics, last feedback record, last decision record,
    agent weight count, and RL model age into a single overview.
    """
    # Diagnostics
    diag = trigger_diagnostics()

    # Last feedback record
    fb = _tail_jsonl(FEEDBACK_LOG, 1)
    last_fb = fb.records[0] if fb.records else None

    # Last decision record
    dec = _tail_jsonl(DECISION_LOG, 1)
    last_dec = dec.records[0] if dec.records else None

    # Agent weights count
    weights_count = 0
    try:
        if AGENT_WEIGHTS_PATH.exists():
            with AGENT_WEIGHTS_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                for bucket_name, bucket in data.items():
                    if not bucket_name.startswith("_") and isinstance(bucket, dict):
                        weights_count += len(bucket)
    except Exception:
        pass

    # RL model age
    rl_age: Optional[float] = None
    try:
        if RL_MODEL_PATH.exists():
            mtime = os.path.getmtime(RL_MODEL_PATH)
            age_s = datetime.now(timezone.utc).timestamp() - mtime
            rl_age = round(max(0.0, age_s) / 3600.0, 1)
    except Exception:
        pass

    return HealthCheckResult(
        diagnostics=diag,
        last_feedback=last_fb,
        last_decision=last_dec,
        agent_weights_count=weights_count,
        rl_model_age_hours=rl_age,
        overall=diag.status,
    )


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
def main() -> None:
    """Start the Buddy MCP server on stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
