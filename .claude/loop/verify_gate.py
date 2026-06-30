#!/usr/bin/env python3
"""verify_gate.py — the DETERMINISTIC half of the verifier (LOOP.md / verifier.md).

A separate *agent* (Code Reviewer) judges semantic/subjective claims. This script judges the
machine-checkable ones by RE-DERIVING them from disk — so it cannot be fooled by a worker's (or a
subagent's) narrative. It mirrors the repo's own pattern: a deterministic Constitution check paired
with a qualitative auditor, where disagreement is a hard stop.

It is defense-in-depth ON TOP OF the code guardrails (execution.py halt, HARD_MAX_GAP quarantine) —
never a replacement. It NEVER mutates the tree; it only reads and writes its own verdict artifact.

Usage:
    python3 verify_gate.py [--repo PATH] [--out PATH] [--quiet]
Exit: 0 = PASS (all hard-NO checks hold + context system intact)
      2 = FAIL (>=1 violation) — a Hard-NO failure is always a HALT-class FAIL.
Stdlib only. Repo override makes it unit-testable against fixtures.
"""
from __future__ import annotations
import argparse
import ast
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REPO = Path("/Users/buddy/Documents/ml_engine")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _integrity import gate_drift, audit_lessons  # noqa: E402  (gate-hash + lesson-content checks)


def _read(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except Exception:
        return ""


def _git(repo: Path, *args: str) -> str:
    if not (repo / ".git").exists():
        return ""
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:
        return ""


def _eval_num(node: ast.AST):
    """Evaluate a numeric AST expression over literals only; None if not a pure-literal number."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval_num(node.operand)
        return None if v is None else (v if isinstance(node.op, ast.UAdd) else -v)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        l, r = _eval_num(node.left), _eval_num(node.right)
        if l is None or r is None:
            return None
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        return l / r if r else None
    return None


def _ship_gate_violation(repo: Path):
    """AST (not regex): catch HARD_MAX_GAP > 0.10 incl .15 / 1e-1 / 0.10+0.05; fail closed on a
    non-literal value (cannot prove <=0.10). Returns a detail string on violation, else None."""
    for sub in ("src", "scripts"):
        base = repo / sub
        if not base.exists():
            continue
        for py in base.rglob("*.py"):
            if "_quarantine" in str(py):
                continue
            txt = _read(py)
            if "HARD_MAX_GAP" not in txt:
                continue
            try:
                tree = ast.parse(txt)
            except Exception:
                return f"{py.relative_to(repo)}: unparseable file assigns HARD_MAX_GAP (cannot verify)"
            rel = py.relative_to(repo)
            for node in ast.walk(tree):
                # Gather (target_nodes, value, kind) for every assignment-like form. Tuple targets
                # are walked for Names so `HARD_MAX_GAP, b = ...` is seen; AugAssign is included so
                # `HARD_MAX_GAP += x` can't escalate invisibly. (Closes verifier round-2 finding #1.)
                if isinstance(node, ast.Assign):
                    targets, val, kind = node.targets, node.value, "assign"
                elif isinstance(node, ast.AnnAssign) and node.target is not None:
                    targets, val, kind = [node.target], node.value, "ann"
                elif isinstance(node, ast.AugAssign):
                    targets, val, kind = [node.target], node.value, "aug"
                elif isinstance(node, ast.NamedExpr):  # walrus (HARD_MAX_GAP := 0.2)
                    targets, val, kind = [node.target], node.value, "assign"
                else:
                    continue
                names = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
                if "HARD_MAX_GAP" not in names:
                    continue
                # Fail-closed for anything we can't reduce to a single literal value.
                if kind == "aug":
                    return f"{rel}: HARD_MAX_GAP augmented-assignment (+=/-=) — cannot verify <=0.10 (fail-closed)"
                simple_target = all(isinstance(t, ast.Name) for t in targets)
                if not simple_target or val is None:
                    return f"{rel}: HARD_MAX_GAP non-simple target/value — cannot verify <=0.10 (fail-closed)"
                num = _eval_num(val)
                if num is None:
                    return f"{rel}: HARD_MAX_GAP is non-literal (cannot verify <=0.10) — fail-closed"
                if num > 0.10 + 1e-9:
                    return f"{rel}: HARD_MAX_GAP={num} > 0.10"
    return None


def _halt_guard_violation(repo: Path):
    """AST: require a LIVE `if ...get_halted(): ... return ...` guard in execution.py, so a
    commented-out/dead occurrence cannot satisfy the check (comments are absent from the AST)."""
    txt = _read(repo / "src/scanner/execution.py")
    if not txt:
        return "execution.py missing/empty"
    try:
        tree = ast.parse(txt)
    except Exception:
        return "execution.py unparseable (cannot verify halt guard)"
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        calls = [n for n in ast.walk(node.test) if isinstance(n, ast.Call)]
        attrs = [c.func.attr for c in calls if isinstance(c.func, ast.Attribute)]
        # Return must be in the THEN-body (node.body), not the else-branch — an inverted guard
        # `if get_halted(): proceed() else: return` must NOT pass. (Verifier round-2 finding #2.)
        body_returns = any(isinstance(n, ast.Return) for b in node.body for n in ast.walk(b))
        if "get_halted" in attrs and body_returns:
            return None
    return "no live `if ...get_halted(): return ...` guard in execution.py (commented/dead/inverted?)"


def _lessons_without_triggers(repo: Path):
    """Every `## L-NNN` lesson must be referenced in the recall-trigger index, else it won't fire at
    planning time. Returns the list of orphan lesson ids (empty/None = ok)."""
    txt = _read(repo / ".claude/LESSONS.md")
    if not txt:
        return None
    ids = re.findall(r'^##\s+(L-\d+)\b', txt, re.M)
    if not ids:
        return None
    start = txt.lower().find("recall-trigger")
    region = txt[start: txt.find("\n---", start)] if start >= 0 else ""
    missing = [lid for lid in ids if lid not in region]
    return missing or None


def check_hard_nos(repo: Path) -> list[dict]:
    """The immutable Hard NOs, each re-derived from disk. ok=False => HALT-class FAIL.

    Known evasion surface (documented so no future reader over-trusts this — it is a tripwire,
    NOT the primary guard; the hard guards live in execution.py + the HARD_MAX_GAP quarantine):
      - regexes are literal/positional: `= .15`, `= 1e-1`, computed/`float()`/env-var values, and
        the gap/env read from config can slip the pattern.
      - the halt-guard check is now AST-verified (live `if get_halted(): return` in the THEN-body).
      - ship-gate is AST-verified over Assign/AnnAssign/AugAssign/NamedExpr targets, fail-closed on
        anything not reducible to a single literal ≤0.10.
      - STILL out of scope (a STATIC tripwire cannot see these — the code-level guard in
        execution.py + the HARD_MAX_GAP quarantine remain the primary rail): fully dynamic mutation
        such as `setattr(mod, "HARD_MAX_GAP", x)` or `globals()["HARD_MAX_GAP"] = x`. Documented, not
        chased — promote to a runtime assertion if this layer ever becomes a primary guard.
    """
    checks: list[dict] = []

    # HARD NO 1 — oanda_environment defaults to "practice" (L-003).
    cfg = _read(repo / "src/scanner/config.py")
    ok = bool(re.search(r'oanda_environment:\s*str\s*=\s*"practice"', cfg))
    checks.append({"name": "oanda_env_practice", "hard_no": True, "ok": ok,
                   "detail": "config.py default is \"practice\"" if ok
                   else "config.py oanda_environment default is NOT \"practice\""})

    # Live-flip: content-scan src+scripts for any oanda_environment="live" ASSIGNMENT (catches
    # committed/staged/unstaged, default OR profile-override) — but NOT the mere presence of the
    # live-URL constant, which legitimately exists for when env IS live (broker_transport.py).
    # Plus a diff scan (unstaged+staged) for newly-ADDED env flips / endpoints. Closes the prior
    # gap where only unstaged `git diff` was inspected (verifier finding #2, 2026-06-23).
    live_hit = None
    for sub in ("src", "scripts"):
        base = repo / sub
        if not base.exists():
            continue
        for py in base.rglob("*.py"):
            if "_quarantine" in str(py):
                continue
            if re.search(r'oanda_environment\s*=\s*"live"', _read(py)):
                live_hit = f"{py.relative_to(repo)} assigns oanda_environment=\"live\""
                break
        if live_hit:
            break
    if live_hit is None:
        # diff-scan SCOPED to src/scripts (a real env flip / live endpoint lives there) and to the
        # assignment/annotation form. Docs/tests/comments under .claude/ that merely MENTION
        # oanda_environment="live" or api-fxtrade (e.g. to explain or test the guard) must NOT
        # false-trip this — that wart caused verify_gate to FAIL on its own gate-teach diff (2026-06-24).
        diffs = (_git(repo, "diff", "--", "src", "scripts") + "\n"
                 + _git(repo, "diff", "--cached", "--", "src", "scripts"))
        if re.search(r'^\+.*oanda_environment\s*[:=].*"live"', diffs, re.M) or \
           re.search(r'^\+.*api-fxtrade\.oanda\.com', diffs, re.M):
            live_hit = "src/scripts diff adds a live env flip or live endpoint"
    checks.append({"name": "no_live_flip", "hard_no": True, "ok": live_hit is None,
                   "detail": "no live env assignment in src+scripts (staged/unstaged clean)"
                   if live_hit is None else live_hit})

    # HARD NO 2 — ship gate stays <= 0.10 (L-002). AST-evaluated, not regex: catches .15, 1e-1,
    # 0.10+0.05, and fails closed on a non-literal value. Closes verifier finding #3 (2026-06-23).
    gap_v = _ship_gate_violation(repo)
    checks.append({"name": "ship_gate_le_0.10", "hard_no": True, "ok": gap_v is None,
                   "detail": "HARD_MAX_GAP <= 0.10 (AST-verified)" if gap_v is None else gap_v})

    # HARD NO 3 — halt enforcement is LIVE code (L-004): a real `if ...get_halted(): ... return`
    # guard, AST-verified so a commented-out/dead occurrence can't satisfy it. Closes finding #3.
    halt_v = _halt_guard_violation(repo)
    checks.append({"name": "halt_guard_live", "hard_no": True, "ok": halt_v is None,
                   "detail": "live get_halted() guard with return (AST-verified)" if halt_v is None else halt_v})

    # Runtime state readable. mode=live is EXECUTION mode (place orders), NOT real money — real money
    # is oanda_environment=live (checked hard above). On a PRACTICE env, mode=live = paper trading
    # (broker is api-fxpractice-only), allowed for operator-directed un-halt. So fail here only if
    # mode=live AND env is not practice (real-money execution). (operator-directed 2026-06-24.)
    sp = repo / ".claude/state.json"
    mode = halted = None
    try:
        st = json.loads(sp.read_text())
        mode, halted = st.get("mode"), st.get("halted")
        env_practice = bool(re.search(r'oanda_environment:\s*str\s*=\s*"practice"',
                                      _read(repo / "src/scanner/config.py")))
        state_ok = (mode != "live") or env_practice
        det = (f"state readable (halted={halted}, mode={mode}, env_practice={env_practice})"
               if state_ok else "state.mode=live AND env NOT practice == real-money execution")
    except Exception as e:
        state_ok = False
        det = f"state.json unreadable/unparseable == unsafe ({e})"
    checks.append({"name": "state_readable_not_live", "hard_no": True, "ok": state_ok, "detail": det})
    return checks


def check_integrity(repo: Path) -> list[dict]:
    """Structural: the context system + enforcement tooling exist on disk."""
    required = [
        "CLAUDE.md", ".claude/INTENT.md", ".claude/NOTES.md", ".claude/LESSONS.md",
        ".claude/LOOP.md", ".claude/verifier.md",
        ".claude/commands/evolve.md", ".claude/commands/verify-task.md",
        ".claude/tools/risk_monitor.sh", ".claude/tools/session_context_boot.sh",
        ".claude/tools/stop_gate.sh", ".claude/loop/verify_gate.py", ".claude/loop/loop_gate.py",
        ".claude/loop/record_cycle.py", ".claude/loop/questions.json",
        ".claude/loop/_integrity.py", ".claude/loop/gen_manifest.py", ".claude/loop/gate_manifest.json",
        ".claude/loop/record_verdict.py",
    ]
    out = []
    for rel in required:
        p = repo / rel
        ok = p.exists() and p.stat().st_size > 0
        out.append({"name": f"exists:{rel}", "hard_no": False, "ok": ok,
                    "detail": "present" if ok else "MISSING or empty"})

    # Enforced-wiring: the Stop hook must actually be REGISTERED, else the tripwire is dead code
    # (verifier finding #1, 2026-06-23). A present-but-unwired stop_gate.sh is a silent safety gap.
    # Stop hook must register the CANONICAL repo stop_gate.sh, not just any file named stop_gate.sh
    # (closes the basename-only hole — a stub `/tmp/evil/stop_gate.sh` no longer satisfies this).
    reg_ok, reg_detail = False, "settings.json missing -> Stop tripwire NOT wired"
    canonical = (repo / ".claude/tools/stop_gate.sh").resolve()

    def _resolve(cmd: str):
        cmd = cmd.replace("${CLAUDE_PROJECT_DIR}", str(repo)).replace("$CLAUDE_PROJECT_DIR", str(repo))
        try:
            return Path(cmd.strip()).resolve()
        except Exception:
            return None
    try:
        s = json.loads((repo / ".claude/settings.json").read_text())
        cmds = [h.get("command", "")
                for grp in s.get("hooks", {}).get("Stop", []) for h in grp.get("hooks", [])]
        reg_ok = any(_resolve(c) == canonical for c in cmds if c)
        reg_detail = "canonical stop_gate.sh registered as Stop hook" if reg_ok \
            else "no Stop hook resolves to the repo's stop_gate.sh -> tripwire NOT wired (or stubbed)"
    except Exception as e:
        reg_detail = f"settings.json unreadable ({e}) -> cannot confirm Stop tripwire wiring"
    out.append({"name": "stop_gate_registered", "hard_no": False, "ok": reg_ok, "detail": reg_detail})

    # Gate-script content integrity: every enforcement script must match the committed hash manifest.
    drift, man_err = gate_drift(repo)
    gate_ok = (not drift) and (man_err is None)
    gate_detail = "all gate scripts match manifest" if gate_ok else (man_err or f"gate-script drift: {drift}")
    out.append({"name": "gate_scripts_unmodified", "hard_no": False, "ok": gate_ok, "detail": gate_detail})

    # Memory enforcement: every L-NNN lesson must have a recall-trigger row, or it won't fire at
    # planning time (the whole point of the trigger index). Structurally enforces "lessons fire".
    miss = _lessons_without_triggers(repo)
    out.append({"name": "lessons_have_triggers", "hard_no": False, "ok": not miss,
                "detail": "every lesson has a recall trigger" if not miss
                else f"lessons missing recall triggers: {miss}"})

    # Lesson CONTENT integrity (red-team #3): every ## L-NNN must be well-formed (all five fields,
    # non-trivial body, unique title). A shallow/empty/duplicate lesson added to bump the counter
    # fails the gate here.
    _, lesson_problems = audit_lessons(_read(repo / ".claude/LESSONS.md"))
    out.append({"name": "lessons_well_formed", "hard_no": False, "ok": not lesson_problems,
                "detail": "all lessons well-formed" if not lesson_problems
                else f"malformed/shallow/duplicate lessons: {lesson_problems}"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    repo = Path(a.repo)

    hard = check_hard_nos(repo)
    integ = check_integrity(repo)
    checks = hard + integ
    hard_no_ok = all(c["ok"] for c in hard)
    gate = "PASS" if all(c["ok"] for c in checks) else "FAIL"

    verdict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "hard_no_ok": hard_no_ok,
        "gate": gate,
        "checks": checks,
    }
    out_path = Path(a.out) if a.out else (repo / ".claude/loop/verdict.json")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(verdict, indent=2, sort_keys=True))
        tmp.replace(out_path)  # atomic
    except Exception as e:
        print(f"verify_gate: WARN could not write verdict: {e}", file=sys.stderr)

    if not a.quiet:
        print(f"VERIFY GATE: {gate}  (hard_no_ok={hard_no_ok})")
        for c in checks:
            if not c["ok"]:
                tag = "HARD-NO" if c["hard_no"] else "integrity"
                print(f"  ✗ [{tag}] {c['name']}: {c['detail']}")
        if gate == "PASS":
            print(f"  all {len(checks)} checks confirmed from disk ({len(hard)} hard-NO + {len(integ)} integrity)")
    return 0 if gate == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
