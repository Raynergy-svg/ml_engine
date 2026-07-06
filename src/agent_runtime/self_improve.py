"""AXIOM Agent Runtime — bounded self-improvement edit tier (SELF_IMPROVE).

A fourth action tier, narrower than OPERATIONAL and structurally incapable of
touching anything ESCALATION already guards. Where OPERATIONAL/DEESCALATION
actions are pre-written functions with no path parameter at all, SELF_IMPROVE
actions are pre-written functions that mutate a SPECIFIC on-disk target
(hardcoded in this module, never accepted as a caller-supplied parameter) via
an apply -> test -> verify -> commit-or-revert lifecycle. Nothing here lets an
arbitrary path be edited: the menu of possible edits is fixed at import time.

Two independent layers keep this bounded:

  1. Hardcoded targets. Every SELF_IMPROVE action's file path is a Python
     constant in its own function body (``_AGENT_WEIGHTS_PATH``,
     ``_LESSONS_PATH``). The proposal's ``params`` dict can only ever supply
     CONTENT parameters (which keys to purge, which lesson id) -- there is no
     ``path``/``target`` kwarg on any execute function, so a malicious or
     malformed proposal cannot redirect an action at a different file. Passing
     an unexpected kwarg raises ``TypeError`` (audited as an error, never
     committed) rather than being silently accepted.
  2. `_check_target_paths` -- a second, independent check run before any write,
     against a STRUCTURAL_DENYLIST that is refused unconditionally (checked
     first, regardless of allowlist membership) and an ALLOWLIST that is the
     only door in (fail-closed default-deny for anything not explicitly
     listed). This exists as defense in depth even though (1) already makes
     the attack surface empty -- if a future action is added carelessly, this
     is the backstop that catches it.

Lifecycle (`_run_gated_edit`): snapshot bytes (NOT `git stash` -- L-024: this
repo runs live daemons that write tracked files, so a stash can conflict and
strand work; a plain in-memory byte snapshot + direct write-back has no such
failure mode) -> apply the one registered operation -> run its test command
-> run the universal safety gate (`.claude/loop/verify_gate.py` +
`.claude/tools/risk_monitor.sh`) -> on ANY failure, restore the snapshot and
raise `PolicyDenied` (no commit, audited as `denied_by_action`) -> on success,
a SCOPED `git commit -- <path>` (never `git add -A`) so a concurrent daemon
writing an unrelated tracked file cannot be swept into this commit.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.agent_runtime.policy import ActionSpec, ActionTier, PolicyDenied

# ---------------------------------------------------------------------------
# Structural boundary -- checked before any write, unconditionally.
# ---------------------------------------------------------------------------

# Refused ALWAYS, regardless of action name or allowlist membership. Anything
# matching (exact path or prefix) stays ESCALATION / proposal-only -- routed
# to the operator, never auto-edited. Grounded in the actual files read this
# session, not guessed.
STRUCTURAL_DENYLIST: Tuple[str, ...] = (
    "src/scanner/execution.py",
    "src/scanner/automation/state_engine.py",
    "src/scanner/gates.py",
    "src/scanner/config.py",
    "src/scanner/agents/",  # agent voting/weighting code feeds trade consensus
    "src/equity/decision_gate.py",
    "src/equity/live_gate.py",
    "src/equity/track_b_live_gate.py",
    "src/crypto/crypto_live_gate.py",
    "src/brokers/",  # order placement / arm_equity_gate
    "src/risk/",
    "dashboard/server/control_safety.py",
    "dashboard/server/control",  # covers dashboard/server/control*.py
    "control_safety.py",
    "src/agent_runtime/",  # this package's own guardrail/dispatch code, incl. this file
    ".claude/tools/risk_monitor.sh",
    ".claude/tools/stop_gate.sh",
    ".claude/loop/verify_gate.py",
    ".claude/loop/loop_gate.py",
    ".claude/state.json",
    "trained_data/axiom/loop_autonomy.json",  # the agent cannot edit its own permission flag
)

# The ONLY other way in. A path that is neither denylisted (refused above,
# unconditionally) nor present here is refused too -- fail-closed default-deny,
# mirroring PolicyEngine's own "no default tier" contract.
ALLOWLIST: Tuple[str, ...] = (
    "trained_data/models/agent_weights.json",
    ".claude/LESSONS.md",
    ".claude/NOTES.md",
)


def _normalize(raw: str) -> str:
    return str(raw).replace("\\", "/").lstrip("/")


def _check_target_paths(paths: Iterable[str]) -> None:
    for raw in paths:
        norm = _normalize(raw)
        for bad in STRUCTURAL_DENYLIST:
            if norm == bad or norm.startswith(bad):
                raise PolicyDenied(
                    f"self-improve refused: {raw!r} matches structural denylist entry "
                    f"{bad!r} -- this stays ESCALATION/proposal-only, routed to the operator"
                )
        if norm not in ALLOWLIST:
            raise PolicyDenied(
                f"self-improve refused: {raw!r} is not on the explicit allowlist "
                f"{ALLOWLIST} -- fail-closed default-deny"
            )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _atomic_write_json(path: Path, data: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Gated-edit lifecycle
# ---------------------------------------------------------------------------


def _run_checked(cmd: Sequence[str], *, cwd: Path, timeout: int) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            list(cmd), cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s: {' '.join(cmd)}"
    except OSError as exc:
        return False, f"failed to run {' '.join(cmd)}: {exc!r}"
    if result.returncode != 0:
        detail = (result.stdout or "")[-600:] + (result.stderr or "")[-600:]
        return False, f"exit {result.returncode}: {' '.join(cmd)}\n{detail}"
    return True, ""


def _git_commit(repo_root: Path, rel_paths: Sequence[str], action_name: str) -> Dict[str, Any]:
    for rel in rel_paths:
        subprocess.run(["git", "add", "--", rel], cwd=str(repo_root), capture_output=True, text=True, timeout=30)

    msg = (
        f"self-improve({action_name}): autonomous gated edit via AXIOM ResidentLoop\n\n"
        "Applied by src/agent_runtime/self_improve.py after apply+test+verify_gate+"
        "risk_monitor all passed. Reverts (no commit) on any failure -- see "
        ".claude/rules/improvement.md self-improvement gate.\n\n"
        "Co-Authored-By: AXIOM ResidentLoop <noreply@anthropic.com>"
    )
    last_error = ""
    for attempt in range(3):
        result = subprocess.run(
            ["git", "commit", "-m", msg, "--", *rel_paths],
            cwd=str(repo_root), capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            return {"committed": True, "sha": sha, "reason": ""}
        last_error = (result.stdout or "") + (result.stderr or "")
        if "index.lock" in last_error and attempt < 2:
            time.sleep(1.5 * (attempt + 1))
            continue
        break
    return {"committed": False, "sha": None, "reason": last_error[-500:]}


def _run_gated_edit(
    *,
    action_name: str,
    target_paths: Sequence[str],
    apply_fn,
    test_cmd: Sequence[str],
) -> Dict[str, Any]:
    _check_target_paths(target_paths)  # structural, unconditional, before any write

    repo_root = _repo_root()
    abs_paths = [repo_root / p for p in target_paths]
    # In-memory byte snapshot for revert -- NEVER `git stash` here (L-024: this
    # repo runs live launchd daemons that write tracked files; a stash can
    # conflict and strand uncommitted work if a daemon writes mid-stash).
    snapshots: Dict[Path, Optional[bytes]] = {
        p: (p.read_bytes() if p.exists() else None) for p in abs_paths
    }

    def _revert(reason: str) -> None:
        for p, original in snapshots.items():
            if original is None:
                if p.exists():
                    p.unlink()
            else:
                p.write_bytes(original)
        raise PolicyDenied(f"self-improve edit REVERTED (no commit) -- {reason}")

    try:
        apply_detail = apply_fn()
    except PolicyDenied:
        _revert("apply_fn refused its own safety invariant")
        raise  # unreachable; _revert always raises
    except Exception as exc:  # noqa: BLE001 -- any apply failure must revert, not half-apply
        _revert(f"apply_fn raised {exc!r}")
        raise

    ok, detail = _run_checked(list(test_cmd), cwd=repo_root, timeout=180)
    if not ok:
        _revert(f"test command failed -- {detail}")

    ok, detail = _run_checked(
        [sys.executable, str(repo_root / ".claude/loop/verify_gate.py"), "--quiet"],
        cwd=repo_root, timeout=90,
    )
    if not ok:
        _revert(f"verify_gate.py FAIL -- {detail}")

    ok, detail = _run_checked(
        ["bash", str(repo_root / ".claude/tools/risk_monitor.sh")], cwd=repo_root, timeout=60,
    )
    if not ok:
        _revert(f"risk_monitor.sh ALARM -- {detail}")

    commit = _git_commit(repo_root, list(target_paths), action_name)
    if not commit["committed"]:
        _revert(f"git commit failed -- {commit['reason']}")

    return {
        "committed": True,
        "commit_sha": commit["sha"],
        "reverted": False,
        "tests_passed": True,
        "verify_gate_passed": True,
        "risk_monitor_green": True,
        "apply_detail": apply_detail,
    }


# ---------------------------------------------------------------------------
# Action: purge orphan agent-weight keys (data hygiene)
# ---------------------------------------------------------------------------

_AGENT_WEIGHTS_PATH = "trained_data/models/agent_weights.json"
_CANONICAL_AGENT_NAMES = frozenset({
    "trend", "mean_reversion", "volatility", "risk_sentinel", "uncertainty",
    "execution_quality", "momentum", "news_risk", "multi_timeframe",
    "pair_performance", "session_timing", "support_resistance",
    "order_flow", "trader_readiness", "devil_advocate",
})
_WEIGHT_REGIMES = ("EXTREME", "HIGH", "LOW", "NORMAL", "_global")


def _apply_purge_agent_weight_keys(keys: List[str]) -> Dict[str, Any]:
    path = _repo_root() / _AGENT_WEIGHTS_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    removed: Dict[str, List[str]] = {}

    for regime in _WEIGHT_REGIMES:
        regime_dict = data.get(regime)
        if not isinstance(regime_dict, dict):
            continue
        for key in keys:
            if key in regime_dict:
                removed.setdefault(regime, []).append(key)
                del regime_dict[key]
        leftover_unknown = set(regime_dict) - _CANONICAL_AGENT_NAMES
        if leftover_unknown:
            raise PolicyDenied(
                f"purge refused: {regime} still has unknown keys {sorted(leftover_unknown)} "
                "after purge -- safety invariant violated"
            )
        missing = _CANONICAL_AGENT_NAMES - set(regime_dict)
        if missing:
            raise PolicyDenied(
                f"purge refused: {regime} missing canonical agents {sorted(missing)} "
                "after purge -- safety invariant violated"
            )

    meta = data.get("_meta")
    if isinstance(meta, dict):
        for key in keys:
            for meta_key in (f"weight_history_{key}", f"weight_variance_{key}"):
                if meta_key in meta:
                    removed.setdefault("_meta", []).append(meta_key)
                    del meta[meta_key]

    _atomic_write_json(path, data)
    return {"purged": removed}


def _self_improve_purge_agent_weight_keys(keys: Optional[List[str]] = None) -> Dict[str, Any]:
    keys = list(keys or ["invalid_agent_xyz"])
    return _run_gated_edit(
        action_name="self_improve_purge_agent_weight_keys",
        target_paths=[_AGENT_WEIGHTS_PATH],
        apply_fn=lambda: _apply_purge_agent_weight_keys(keys),
        test_cmd=[sys.executable, "-m", "pytest", "tests/test_agent_team_core.py::TestWeightValidation", "-q"],
    )


# ---------------------------------------------------------------------------
# Action: add a recall-trigger index row for an EXISTING lesson (doctrine)
# ---------------------------------------------------------------------------

_LESSONS_PATH = ".claude/LESSONS.md"
_LESSON_ID_RE = re.compile(r"^L-\d{3}$")


def _apply_add_lessons_recall_row(lesson_id: str) -> Dict[str, Any]:
    if not _LESSON_ID_RE.match(lesson_id):
        raise PolicyDenied(f"invalid lesson_id {lesson_id!r} -- expected form 'L-NNN'")

    path = _repo_root() / _LESSONS_PATH
    text = path.read_text(encoding="utf-8")

    header_match = re.search(rf"^##\s+{re.escape(lesson_id)}\s+—\s+(.+?)\s+\[", text, re.M)
    if not header_match:
        raise PolicyDenied(
            f"{lesson_id} has no entry in LESSONS.md -- refusing to add a recall row "
            "for a lesson that doesn't exist"
        )
    short_name = header_match.group(1).strip()

    next_entry = text.find("\n## L-", header_match.start() + 1)
    entry_text = text[header_match.start():] if next_entry < 0 else text[header_match.start():next_entry]
    trigger_match = re.search(r"- Trigger:\s*(.+?)\n- Root cause", entry_text, re.S)
    if not trigger_match:
        raise PolicyDenied(
            f"{lesson_id} entry has no parsable 'Trigger:' line -- refusing to fabricate recall text"
        )
    trigger_text = " ".join(trigger_match.group(1).split())
    if len(trigger_text) > 160:
        trigger_text = trigger_text[:157] + "..."

    idx_start = text.lower().find("recall-trigger")
    if idx_start < 0:
        raise PolicyDenied("recall-trigger index section not found in LESSONS.md")
    idx_end = text.find("\n---", idx_start)
    if idx_end < 0:
        raise PolicyDenied("recall-trigger index section has no closing '---'")
    region = text[idx_start:idx_end]

    if re.search(rf"\*\*{re.escape(lesson_id)}\*\*", region):
        return {"already_present": True, "lesson_id": lesson_id}

    row_matches = list(re.finditer(r"^\|.*\|$", region, re.M))
    if not row_matches:
        raise PolicyDenied("no existing table rows found in recall-trigger index -- refusing free-form insert")
    last_row_end = idx_start + row_matches[-1].end()

    new_row_line = f"| {trigger_text} | **{lesson_id}** {short_name} |"
    new_text = text[:last_row_end] + "\n" + new_row_line + text[last_row_end:]
    path.write_text(new_text, encoding="utf-8")
    return {"lesson_id": lesson_id, "row_added": new_row_line}


def _self_improve_add_lessons_recall_row(lesson_id: Optional[str] = None) -> Dict[str, Any]:
    if not lesson_id:
        raise PolicyDenied("lesson_id is required")
    return _run_gated_edit(
        action_name="self_improve_add_lessons_recall_row",
        target_paths=[_LESSONS_PATH],
        apply_fn=lambda: _apply_add_lessons_recall_row(lesson_id),
        test_cmd=[sys.executable, str(_repo_root() / ".claude/loop/verify_gate.py"), "--quiet"],
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SELF_IMPROVE_ACTIONS: Dict[str, ActionSpec] = {
    "self_improve_purge_agent_weight_keys": ActionSpec(
        name="self_improve_purge_agent_weight_keys",
        tier=ActionTier.SELF_IMPROVE,
        description=(
            "Remove orphan/unknown agent keys from trained_data/models/agent_weights.json "
            "(data hygiene, hardcoded target path). Refuses if the canonical 15-agent schema "
            "would be violated after removal. Test+verify gated; commits on pass, reverts on fail."
        ),
        execute=_self_improve_purge_agent_weight_keys,
        params_hint='EXACTLY one optional kwarg: "keys" (list[str], defaults to ["invalid_agent_xyz"] '
        'if omitted). No other kwarg is accepted -- e.g. "regimes"/"reason"/"tiers" will be REJECTED.',
    ),
    "self_improve_add_lessons_recall_row": ActionSpec(
        name="self_improve_add_lessons_recall_row",
        tier=ActionTier.SELF_IMPROVE,
        description=(
            "Add a recall-trigger index row for an EXISTING .claude/LESSONS.md entry "
            "(hardcoded target path), deriving the trigger text from that lesson's own "
            "'Trigger:' line -- never fabricated. Test+verify gated; commits on pass, "
            "reverts on fail."
        ),
        execute=_self_improve_add_lessons_recall_row,
        params_hint='EXACTLY one required kwarg: "lesson_id" (str, form "L-NNN", e.g. "L-024"). '
        "No other kwarg is accepted.",
    ),
}
