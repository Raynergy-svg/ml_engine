"""Operator-paste-ready TUI state snapshot.

Mythos audit 2026-05-04 — operator request: "add the ability to copy the
logs from what i see [logs from all boxes] to show you exactly what to
fix and whats going on". The current debugging loop is:

    1. Operator describes a symptom in natural language.
    2. Claude asks for specific log/state evidence.
    3. Operator runs a sequence of grep/cat/jq commands.
    4. Claude diagnoses.

This module collapses (1)-(3) into one F2-equivalent hotkey:

    operator presses 'c' → snapshot written to .claude/snapshots/<ts>.md
                        → ALSO piped to pbcopy when on macOS
                        → brain feed announces "📋 Snapshot at <path>"
    operator pastes the markdown into chat → Claude has every relevant
    surface in one block, ready to diagnose.

The snapshot is a markdown document covering every canonical
verification surface from CLAUDE.md "Honesty & verification" section,
plus a tail of buddy_debug.log so Claude can see what the runtime is
actually emitting RIGHT NOW.

Pure stdlib + safe_json_read. No new dependencies. Failures in any
section are reported inline ("ERR: …") instead of crashing the
snapshot — partial state is better than no snapshot.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Default tail counts — tunable via build_snapshot kwargs.
DEFAULT_DEBUG_LOG_LINES = 50
DEFAULT_BRAIN_LINES = 40
DEFAULT_REFLECTION_LINES = 5
DEFAULT_VIRTUAL_TRADES_LINES = 10
DEFAULT_TRADES_RECENT = 5

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_read_lines(path: Path, n: int) -> list[str]:
    """Tail the last n lines of a file. Returns ['ERR: …'] on any error."""
    try:
        if not path.exists():
            return [f"(file not present: {path})"]
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [ln.rstrip("\n") for ln in lines[-n:]]
    except OSError as e:
        return [f"ERR reading {path}: {e!r}"]


def _safe_json_read(path: Path) -> Optional[Any]:
    """Best-effort JSON read — returns None on any failure."""
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("snapshot.json_read_failed path=%s err=%r", path, e)
        return None


def _section(title: str, lines: Iterable[str]) -> str:
    """Build one markdown section."""
    body = "\n".join(lines) if lines else "(empty)"
    return f"### {title}\n\n```\n{body}\n```\n"


def _heartbeat_section() -> str:
    blob = _safe_json_read(PROJECT_ROOT / ".claude" / "heartbeat.json")
    if not blob:
        return _section("Heartbeat", ["(missing)"])
    lines = [
        f"pid             {blob.get('pid')}",
        f"mode            {blob.get('mode')}",
        f"scanner_alive   {blob.get('scanner_alive')}",
        f"cycle_count     {blob.get('cycle_count')}",
        f"ts_iso          {blob.get('ts_iso')}",
        f"last_error_ts   {blob.get('last_error_ts')}",
    ]
    return _section("Heartbeat", lines)


def _state_section() -> str:
    blob = _safe_json_read(PROJECT_ROOT / ".claude" / "state.json")
    if not blob:
        return _section("State (.claude/state.json)", ["(missing)"])
    sr = blob.get("safe_restart") or {}
    lines = [
        f"halted              {blob.get('halted')}",
        f"mode                {blob.get('mode')}",
        f"scan_cycle_count    {blob.get('scan_cycle_count')}",
        f"last_scan_at        {blob.get('last_scan_at')}",
        f"last_actor          {blob.get('last_actor')}",
        f"last_updated        {blob.get('last_updated')}",
        f"safe_restart.at     {sr.get('at')}",
        f"safe_restart.reason {sr.get('reason')}",
    ]
    return _section("State (.claude/state.json)", lines)


def _alerts_section() -> str:
    blob = _safe_json_read(PROJECT_ROOT / ".claude" / "alert_state.json")
    if not blob:
        return _section("AlertManager (.claude/alert_state.json)", ["(missing)"])
    active = blob.get("active_alerts") or {}
    last_fired = blob.get("last_fired") or {}
    lines = [f"active alerts:    {len(active) if hasattr(active, '__len__') else '?'}"]
    if isinstance(active, dict):
        for k, v in list(active.items())[:8]:
            lines.append(f"  {k}: {v}")
    elif isinstance(active, list):
        for entry in active[:8]:
            if isinstance(entry, dict):
                lines.append(
                    f"  {entry.get('alert_type', entry.get('type', '?'))}: "
                    f"value={entry.get('value', '?')} "
                    f"sev={entry.get('severity', '?')}"
                )
            else:
                lines.append(f"  {entry}")
    if isinstance(last_fired, dict):
        lines.append(f"last_fired keys:  {list(last_fired.keys())[:6]}")
    else:
        lines.append(f"last_fired:       {str(last_fired)[:120]}")
    return _section("AlertManager (.claude/alert_state.json)", lines)


def _meta_pipeline_section() -> str:
    """Recent ledger entries + in-flight package distribution."""
    out: list[str] = []
    ledger = PROJECT_ROOT / ".claude" / "meta" / "changes.jsonl"
    if ledger.exists():
        try:
            with ledger.open("r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            out.append(f"ledger entries: {len(rows)}")
            for r in rows[-5:]:
                out.append(
                    f"  ts={str(r.get('ts', ''))[:19]} "
                    f"cid={(r.get('change_id', '') or '')[:12]} "
                    f"stage={r.get('stage')} event={r.get('event')} "
                    f"kind={r.get('kind')}"
                )
        except (OSError, json.JSONDecodeError) as e:
            out.append(f"ERR reading ledger: {e!r}")
    else:
        out.append("(ledger missing)")

    changes_dir = PROJECT_ROOT / ".claude" / "meta" / "changes"
    if changes_dir.is_dir():
        from collections import Counter
        stages: Counter = Counter()
        for p in changes_dir.glob("*.json"):
            blob = _safe_json_read(p)
            if isinstance(blob, dict):
                stages[blob.get("stage", "?")] += 1
        out.append(f"in-flight stages: {dict(stages)}")
    return _section("Meta-pipeline (.claude/meta/)", out)


def _config_adjustments_section() -> str:
    blob = _safe_json_read(PROJECT_ROOT / ".claude" / "config_adjustments.json")
    if not blob:
        return _section("Config adjustments (.claude/config_adjustments.json)",
                        ["(missing)"])
    pending = blob.get("pending") or {}
    history = blob.get("history") or []
    last_applied = blob.get("last_applied") or {}
    lines = [
        f"pending count:  {len(pending)}",
        f"pending keys:   {list(pending.keys())[:6]}",
        f"history count:  {len(history)}",
        f"last_applied:   {dict(list(last_applied.items())[:6])}",
    ]
    if history:
        last = history[-1]
        lines.append(
            f"last history:   key={last.get('key')} "
            f"new={last.get('new_value')} ts={str(last.get('timestamp', ''))[:19]}"
        )
    return _section("Config adjustments", lines)


def _trades_section() -> str:
    blob = _safe_json_read(PROJECT_ROOT / "trained_data" / "trade_journal_rl.json")
    out: list[str] = []
    if not blob:
        out.append("(trade_journal_rl.json missing)")
        return _section("Trade journal", out)

    trades = blob if isinstance(blob, list) else (blob.get("trades") or [])
    out.append(f"total trades: {len(trades)}")

    # Trade journal shape: each entry has top-level {trade_id, pair, ...} plus
    # an "outcome" sub-dict carrying the actual close fields. Drill into
    # outcome for pnl/exit_reason/regime so the summary is one tight line.
    for t in trades[-DEFAULT_TRADES_RECENT:]:
        outcome = t.get("outcome") if isinstance(t.get("outcome"), dict) else {}
        tid = t.get("trade_id", "?")
        pair = t.get("pair") or t.get("instrument", "?")
        pnl_pips = outcome.get("pnl_pips", t.get("pnl_pips", "?"))
        won = outcome.get("trade_won", t.get("trade_won", "?"))
        exit_reason = outcome.get("exit_reason", t.get("exit_reason", "?"))
        close_at = (
            outcome.get("close_time") or outcome.get("closed_at")
            or t.get("close_time") or t.get("closed_at") or ""
        )[:19]
        regime_in = outcome.get("regime_at_entry", "?")
        out.append(
            f"  #{tid} {pair} pnl_pips={pnl_pips} won={won} "
            f"exit={exit_reason} regime_in={regime_in} closed={close_at}"
        )
    return _section("Trade journal", out)


def _virtual_trades_section() -> str:
    """Recent gate-rejected scans — surfaces WHY no trades are firing."""
    path = PROJECT_ROOT / "trained_data" / "virtual_trades.jsonl"
    lines = _safe_read_lines(path, DEFAULT_VIRTUAL_TRADES_LINES)
    parsed: list[str] = []
    for raw in lines:
        try:
            r = json.loads(raw)
            parsed.append(
                f"{str(r.get('ts', ''))[:19]} {r.get('pair', '?')} "
                f"conf={r.get('raw_confidence', r.get('confidence', '?'))} "
                f"fails={r.get('gate_failures', '?')}"
            )
        except (json.JSONDecodeError, AttributeError):
            parsed.append(raw[:160])
    return _section("Virtual trades (gate-rejected scans)", parsed)


def _reflection_log_section(*, max_lines: int = 30) -> str:
    """Combined system-activity tail — every "Buddy thinking" stream.

    Mythos audit 2026-05-04 — operator complaint: "the copy isnt working..
    those are old logs". The original reflection-log panel was fed
    exclusively by Claude reflection (claude_subprocess.py), so under the
    no-LLM chokepoint (e1c5ebf) it stays frozen on its last pre-fix
    entries. Real current "system thinking" now lives in three files:

      1. `.claude/meta/changes.jsonl`        — meta-pipeline stage
         transitions (intake/scorecard/promote/reject — the actual live
         autonomy loop). PRIMARY.
      2. `logs/autonomous_trainer.jsonl`     — retrain events post-split
         (b0b18e5).
      3. `logs/reflection_log.jsonl`         — Claude attempts (should stay
         empty under the policy — nonzero growth here is a real bug).

    All three merged + sorted by timestamp so the per-box `c` snapshot
    on #reflection-log shows live activity instead of a frozen Claude
    history.
    """
    events: list[tuple[str, str]] = []  # (ts, formatted)

    # Meta-pipeline stage transitions
    meta_lines = _safe_read_lines(
        PROJECT_ROOT / ".claude" / "meta" / "changes.jsonl", max_lines,
    )
    for raw in meta_lines:
        try:
            r = json.loads(raw)
            ts = str(r.get("updated_at") or r.get("ts", ""))
            events.append((
                ts,
                f"[meta]  {ts[:19]} cid={(r.get('change_id', '') or '')[:12]} "
                f"stage={r.get('stage', '?')} event={r.get('event', '?')} "
                f"kind={r.get('kind', '?')}",
            ))
        except (json.JSONDecodeError, AttributeError):
            events.append(("", f"[meta]  {raw[:200]}"))

    # Autonomous trainer events
    autotrain = _safe_read_lines(
        PROJECT_ROOT / "logs" / "autonomous_trainer.jsonl", max_lines,
    )
    for raw in autotrain:
        try:
            r = json.loads(raw)
            ts = str(r.get("ts", ""))
            events.append((
                ts,
                f"[train] {ts[:19]} {r.get('trade_id', '?')} "
                f"success={r.get('success', '?')} "
                f"hyp={(r.get('hypothesis', '') or '')[:70]!r}",
            ))
        except (json.JSONDecodeError, AttributeError):
            events.append(("", f"[train] {raw[:200]}"))

    # Claude reflection (under policy: empty)
    refl = _safe_read_lines(
        PROJECT_ROOT / "logs" / "reflection_log.jsonl", max_lines,
    )
    for raw in refl:
        try:
            r = json.loads(raw)
            ts = str(r.get("ts", ""))
            mode = r.get("mode", "?")
            err = (r.get("error") or "").strip()[:60]
            dur = r.get("duration_seconds", "?")
            events.append((
                ts,
                f"[claude] {ts[:19]} mode={mode} dur={dur}s err={err!r}",
            ))
        except (json.JSONDecodeError, AttributeError):
            events.append(("", f"[claude] {raw[:200]}"))

    # Sort by timestamp, take the most recent max_lines
    events.sort(key=lambda kv: kv[0])
    tail = [line for _, line in events[-max_lines:]]
    if not tail:
        tail = ["(no system activity events found in any source)"]
    return _section(
        f"System activity (last {max_lines} across meta + autotrain + claude)",
        tail,
    )


def _autotrain_log_section() -> str:
    """Tail of autonomous_trainer.jsonl — retrain events.

    Operator-relevant retrain observability: schedule fires, freshness
    triggers, completion status. Distinct from Claude reflection.
    """
    lines = _safe_read_lines(
        PROJECT_ROOT / "logs" / "autonomous_trainer.jsonl",
        DEFAULT_REFLECTION_LINES,
    )
    parsed: list[str] = []
    for raw in lines:
        try:
            r = json.loads(raw)
            parsed.append(
                f"{str(r.get('ts', ''))[:19]} "
                f"trade_id={r.get('trade_id', '?')} "
                f"success={r.get('success', '?')} "
                f"hypothesis={(r.get('hypothesis', '') or '')[:80]!r}"
            )
        except (json.JSONDecodeError, AttributeError):
            parsed.append(raw[:200])
    return _section("Autonomous trainer log tail", parsed)


def _brain_feed_section() -> str:
    path = PROJECT_ROOT / ".claude" / "brain" / "feed.jsonl"
    lines = _safe_read_lines(path, DEFAULT_BRAIN_LINES)
    parsed: list[str] = []
    for raw in lines:
        try:
            r = json.loads(raw)
            parsed.append(f"{str(r.get('ts', ''))[:19]}  {r.get('msg', '')[:140]}")
        except (json.JSONDecodeError, AttributeError):
            parsed.append(raw[:200])
    return _section("Brain feed (last %d lines)" % DEFAULT_BRAIN_LINES, parsed)


def _debug_log_section() -> str:
    """Tail of logs/buddy_debug.log — every logger.* line."""
    path = PROJECT_ROOT / "logs" / "buddy_debug.log"
    lines = _safe_read_lines(path, DEFAULT_DEBUG_LOG_LINES)
    return _section("Debug log (last %d lines)" % DEFAULT_DEBUG_LOG_LINES, lines)


def _model_health_section() -> str:
    """Model files + freshness summary."""
    out: list[str] = []
    joint = PROJECT_ROOT / "trained_data" / "models" / "joint"
    if joint.is_dir():
        files = sorted(p.name for p in joint.iterdir() if p.is_file())
        out.append(f"joint/ files: {len(files)}")
        for n in files[:12]:
            out.append(f"  {n}")
    meta = _safe_json_read(joint / "joint_training_meta.json")
    if isinstance(meta, dict):
        out.append(f"joint trained_at: {meta.get('trained_at')}")
        out.append(f"joint pairs:      {meta.get('selected_pairs') or meta.get('pairs')}")
    ens = _safe_json_read(PROJECT_ROOT / "trained_data" / "models" / "modular_ensemble.meta.json")
    if isinstance(ens, dict):
        out.append(f"ensemble trained_at:    {ens.get('trained_at')}")
        out.append(f"ensemble n_master_pairs:{ens.get('n_master_pairs')}")
        out.append(f"holdout_pending:        {ens.get('holdout_pending')}")
        out.append(f"last_holdout_failure_at:{ens.get('last_holdout_failure_at')}")
    return _section("Model health", out)


def _agents_section() -> str:
    """Agent weights + live regime."""
    out: list[str] = []
    weights = _safe_json_read(
        PROJECT_ROOT / "trained_data" / "models" / "agent_weights.json"
    )
    if isinstance(weights, dict):
        out.append(f"regime keys: {[k for k in weights.keys() if not k.startswith('_')]}")
        for regime in ("LOW", "NORMAL", "HIGH", "EXTREME"):
            sub = weights.get(regime)
            if isinstance(sub, dict):
                top = sorted(sub.items(), key=lambda kv: kv[1], reverse=True)[:5]
                out.append(
                    f"  {regime}: " + ", ".join(f"{k}={v:.2f}" for k, v in top)
                )
    try:
        from src.scanner.automation.meta_manager import _LiveRegimeProbe
        out.append(f"live regime: {_LiveRegimeProbe().current()!r}")
    except Exception as e:
        out.append(f"live regime probe failed: {e!r}")
    return _section("Agents", out)


def _no_llm_policy_section() -> str:
    out: list[str] = []
    out.append(f"BUDDY_META_USE_LLM env: {os.environ.get('BUDDY_META_USE_LLM', 'UNSET')!r}")
    out.append(
        f"BUDDY_META_MANAGER_ENABLED env: "
        f"{os.environ.get('BUDDY_META_MANAGER_ENABLED', 'UNSET')!r}"
    )
    try:
        from src.scanner.config import ScannerConfig
        c = ScannerConfig()
        c.apply_profile("smart")
        out.append(f"smart.meta_manager_use_llm: {c.meta_manager_use_llm}")
        out.append(f"smart.enable_meta_manager:  {c.enable_meta_manager}")
        out.append(f"smart.min_confidence:       {c.min_confidence}")
        out.append(f"smart.weighted_vote_threshold: {c.weighted_vote_threshold}")
    except Exception as e:
        out.append(f"ScannerConfig load failed: {e!r}")
    return _section("Policy state", out)


def build_snapshot() -> str:
    """Build the full markdown snapshot."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    parts: list[str] = [
        f"# Buddy TUI Snapshot — {ts}\n",
        f"_PID {os.getpid()} · cwd `{PROJECT_ROOT}`_\n",
        "## Runtime state\n",
        _heartbeat_section(),
        _state_section(),
        _no_llm_policy_section(),
        _alerts_section(),
        "\n## Autonomous loop\n",
        _meta_pipeline_section(),
        _config_adjustments_section(),
        "\n## Models & agents\n",
        _model_health_section(),
        _agents_section(),
        "\n## Trading evidence\n",
        _trades_section(),
        _virtual_trades_section(),
        "\n## Logs\n",
        _brain_feed_section(),
        _reflection_log_section(),
        _autotrain_log_section(),
        _debug_log_section(),
    ]
    return "\n".join(parts)


def write_snapshot(*, snapshots_dir: Optional[Path] = None) -> Path:
    """Write the snapshot to disk. Returns the path."""
    snapshots_dir = snapshots_dir or (PROJECT_ROOT / ".claude" / "snapshots")
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = snapshots_dir / f"tui_{ts}.md"
    path.write_text(build_snapshot(), encoding="utf-8")
    return path


def copy_to_clipboard(content: str) -> Optional[str]:
    """Best-effort clipboard copy. Returns the tool used, or None on failure."""
    for tool, args in (
        ("pbcopy", ["pbcopy"]),
        ("xclip", ["xclip", "-selection", "clipboard"]),
        ("xsel", ["xsel", "--clipboard", "--input"]),
        ("wl-copy", ["wl-copy"]),
    ):
        if shutil.which(tool):
            try:
                proc = subprocess.run(
                    args, input=content.encode("utf-8"),
                    timeout=5, check=False,
                )
                if proc.returncode == 0:
                    return tool
            except (OSError, subprocess.TimeoutExpired) as e:
                logger.debug("snapshot.clipboard_failed tool=%s err=%r", tool, e)
    return None


def write_and_copy(
    *, snapshots_dir: Optional[Path] = None,
) -> tuple[Path, Optional[str]]:
    """Write snapshot to disk + try clipboard. Returns (path, clipboard_tool_or_None)."""
    path = write_snapshot(snapshots_dir=snapshots_dir)
    tool = copy_to_clipboard(path.read_text(encoding="utf-8"))
    return path, tool
