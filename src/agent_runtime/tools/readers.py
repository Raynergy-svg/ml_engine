"""AXIOM Agent Runtime — thin read/safe-operational wrappers.

Every function here calls an EXISTING function or reads an EXISTING file; none
reimplements bot logic. Every function is read-only: none writes trading
state, none places an order, none arms or unhalts anything (enforced
structurally + tested in tests/test_agent_runtime_tools_2026_07_06.py via a
source scan for forbidden calls).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
CLAUDE_DIR = REPO_ROOT / ".claude"
LOOP_DIR = CLAUDE_DIR / "loop"
LESSONS_PATH = CLAUDE_DIR / "LESSONS.md"
NOTES_PATH = CLAUDE_DIR / "NOTES.md"
TRADE_JOURNAL_PATH = REPO_ROOT / "trained_data" / "trade_journal_rl.json"
AGENT_WEIGHTS_PATH = REPO_ROOT / "trained_data" / "models" / "agent_weights.json"

_running_status_mod = None


def _load_running_status_module():
    """Import the bot's live-lane oracle (.claude/loop/running_status.py) by
    path — mirrors dashboard.server.data_sources._load_running_status exactly
    (the module lives under a dot-prefixed directory, not an importable
    package)."""
    global _running_status_mod
    if _running_status_mod is not None:
        return _running_status_mod
    spec = importlib.util.spec_from_file_location(
        "agent_runtime_running_status", LOOP_DIR / "running_status.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _running_status_mod = mod
    return mod


def health_check() -> Dict[str, Any]:
    """LIVE-lane running:YES/NO oracle (L-017 running_status.live_lane_running)."""
    mod = _load_running_status_module()
    return mod.live_lane_running()


def gate_health() -> Dict[str, Any]:
    """Re-run verify_gate.py's Hard-NO + integrity checks and return the verdict.

    Runs the REAL script as a subprocess (the same mechanism .claude/loop/
    loop_gate.py uses) but always redirects --out to a private temp file —
    this diagnostic read must never overwrite the production
    .claude/loop/verdict.json that loop_gate's anti-tamper check depends on.
    """
    vg = LOOP_DIR / "verify_gate.py"
    fd, tmp_out = tempfile.mkstemp(prefix="agent_runtime_verdict_", suffix=".json")
    import os

    os.close(fd)
    try:
        proc = subprocess.run(
            [sys.executable, str(vg), "--repo", str(REPO_ROOT), "--quiet", "--out", tmp_out],
            capture_output=True, text=True, timeout=90,
        )
        try:
            verdict = json.loads(Path(tmp_out).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            verdict = {"gate": "FAIL", "hard_no_ok": False, "checks": []}
        verdict["returncode"] = proc.returncode
        return verdict
    finally:
        Path(tmp_out).unlink(missing_ok=True)


def tier7_status(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Read-only Tier 7 / self-heal supervisor snapshot (build_tier7_state)."""
    from src.scanner.automation.tier7_state import build_tier7_state

    return build_tier7_state(project_root or REPO_ROOT)


def lane_halt_status(state_path: Optional[Path] = None) -> Dict[str, Any]:
    """Per-lane + global halt state (StateEngine reads only — no set_halted call)."""
    from src.scanner.automation.state_engine import StateEngine

    eng = StateEngine(state_path=state_path)
    return {"global_halted": eng.get_halted(), "lanes": eng.get_lane_status()}


def read_account() -> Dict[str, Any]:
    """OANDA practice account snapshot (read-only; dashboard's own accessor)."""
    from dashboard.server.data_sources import read_account as _read_account

    return _read_account()


def oanda_account_state() -> Dict[str, Any]:
    return read_account()


def trade_feedback(limit: int = 50) -> Dict[str, Any]:
    """Tail of closed-trade journal entries (raw, read-only)."""
    try:
        entries = json.loads(TRADE_JOURNAL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        entries = []
    if not isinstance(entries, list):
        entries = []
    tail = entries[-int(limit):] if limit else entries
    return {"count": len(tail), "total_entries": len(entries), "entries": tail}


def agent_weights() -> Dict[str, Any]:
    """Learned agent-weight overrides by autonomy level (read-only)."""
    try:
        weights = json.loads(AGENT_WEIGHTS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        weights = {}
    if not isinstance(weights, dict):
        weights = {}
    return {"weights": weights}


def learnings(max_chars: int = 20000) -> Dict[str, Any]:
    """Raw LESSONS.md + NOTES.md text (read-only), each truncated for safety."""
    def _read(path: Path) -> Dict[str, Any]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return {"text": "", "truncated": False, "available": False}
        truncated = len(text) > max_chars
        return {"text": text[:max_chars], "truncated": truncated, "available": True}

    lessons = _read(LESSONS_PATH)
    notes = _read(NOTES_PATH)
    return {
        "lessons": lessons["text"], "lessons_truncated": lessons["truncated"],
        "notes": notes["text"], "notes_truncated": notes["truncated"],
    }


def read_crypto_momentum() -> Dict[str, Any]:
    from dashboard.server.data_sources import read_crypto_momentum as _read

    return _read()


def read_track_b() -> Dict[str, Any]:
    from dashboard.server.data_sources import read_track_b as _read

    return _read()


def read_equity_sleeve() -> Dict[str, Any]:
    from dashboard.server.data_sources import read_equity_sleeve as _read

    return _read()


_SHADOW_LANE_READERS = {
    "crypto_momentum": lambda: read_crypto_momentum(),
    "track_b": lambda: read_track_b(),
    "harvester": lambda: read_equity_sleeve(),
}


def shadow_ledger(lane: str) -> Dict[str, Any]:
    """Dispatch to the named shadow-lane ledger reader.

    Fail-closed on an unrecognized lane name — never silently returns an
    arbitrary/default ledger for a typo'd lane.
    """
    reader = _SHADOW_LANE_READERS.get(lane)
    if reader is None:
        raise ValueError(f"unknown shadow lane {lane!r}; allowed: {sorted(_SHADOW_LANE_READERS)}")
    return reader()


def list_shadow_lanes() -> List[str]:
    return sorted(_SHADOW_LANE_READERS)
