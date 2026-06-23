#!/usr/bin/env python3
"""test_loop_enforcement.py — real-disk, no-mock tests for the enforcement layer.

Builds synthetic repo fixtures in a tmp dir and runs the actual gate scripts/binaries against them
as subprocesses (the same way the harness will). No unittest.mock, no patching — real files, real
exit codes (repo No-Mock rule). Run directly:  python3 test_loop_enforcement.py

Covers:
  risk_monitor.sh  : GREEN on good fixture; ALARM on live-flip / raised-gap / missing-halt-guard /
                     unreadable-state (fail-closed).
  verify_gate.py   : PASS on good fixture; FAIL (hard_no_ok=False) on each violation.
  loop_gate.py     : CONTINUE / STOP-DONE / STOP-CHURN(anti-stall) / HALT from disk state.
  stop_gate.sh     : exit 0 on GREEN; exit 2 on ALARM; exit 0 under stop_hook_active (loop guard).
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
LOOP_DIR = HERE.parent.parent              # .claude/loop
CLAUDE_DIR = LOOP_DIR.parent               # .claude
REAL_REPO = CLAUDE_DIR.parent              # repo root
TOOLS = CLAUDE_DIR / "tools"

VERIFY = LOOP_DIR / "verify_gate.py"
LOOPG = LOOP_DIR / "loop_gate.py"
RISK = TOOLS / "risk_monitor.sh"
STOP = TOOLS / "stop_gate.sh"

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}  {detail}")


# ---- fixture builders ---------------------------------------------------------------------------

GOOD_CONFIG = 'class C:\n    oanda_environment: str = "practice"  # immutable\n'
GOOD_EXEC = (
    "def execute_trade(self):\n"
    "    if StateEngine().get_halted():\n"
    "        logger.warning('execute_trade BLOCKED — state.halted=True')\n"
    "        return None\n"
)
GOOD_SCRIPT = "HARD_MAX_GAP = 0.10\n"
CONTEXT_FILES = [
    "CLAUDE.md", ".claude/INTENT.md", ".claude/NOTES.md", ".claude/LESSONS.md",
    ".claude/LOOP.md", ".claude/verifier.md",
    ".claude/commands/evolve.md", ".claude/commands/verify-task.md",
]


def build_repo(root: Path, *, env="practice", gap="0.10", halt=True, state=True,
               register_stop=True, extra_files: dict | None = None) -> Path:
    """Create a minimal synthetic repo fixture; copy the REAL gate scripts in so paths resolve."""
    (root / "src/scanner").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / ".claude/commands").mkdir(parents=True, exist_ok=True)
    (root / ".claude/tools").mkdir(parents=True, exist_ok=True)
    (root / ".claude/loop").mkdir(parents=True, exist_ok=True)

    (root / "src/scanner/config.py").write_text(GOOD_CONFIG.replace("practice", env))
    ex = GOOD_EXEC if halt else "def execute_trade(self):\n    return None  # guard removed\n"
    (root / "src/scanner/execution.py").write_text(ex)
    (root / "scripts/train.py").write_text(f"HARD_MAX_GAP = {gap}\n")
    if state:
        (root / ".claude/state.json").write_text(json.dumps({"halted": True, "mode": "dry_run"}))
    for rel in CONTEXT_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x\n")
    # copy the real enforcement scripts so structural + path checks pass
    for src in (RISK, STOP, TOOLS / "session_context_boot.sh"):
        if src.exists():
            (root / ".claude/tools" / src.name).write_text(src.read_text())
            os.chmod(root / ".claude/tools" / src.name, 0o755)
    for src in (VERIFY, LOOPG):
        (root / ".claude/loop" / src.name).write_text(src.read_text())
    if register_stop:
        (root / ".claude/settings.json").write_text(json.dumps({"hooks": {"Stop": [
            {"hooks": [{"type": "command", "command": str(root / ".claude/tools/stop_gate.sh")}]}]}}))
    for rel, body in (extra_files or {}).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(body)
    return root


def run(cmd: list[str], *, env_extra: dict | None = None, stdin: str = "") -> subprocess.CompletedProcess:
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run(cmd, capture_output=True, text=True, env=env, input=stdin, timeout=60)


# ---- tests --------------------------------------------------------------------------------------

def test_risk_monitor(tmp: Path):
    print("\n[risk_monitor.sh]")
    good = build_repo(tmp / "rm_good")
    r = run(["bash", str(RISK)], env_extra={"RISK_MONITOR_REPO": str(good)})
    check("GREEN exit 0 on good fixture", r.returncode == 0, f"rc={r.returncode} {r.stdout}")
    check("says GREEN", "GREEN" in r.stdout)

    for name, kw in [("live-flip", dict(env="live")), ("raised-gap", dict(gap="0.25")),
                     ("missing-halt-guard", dict(halt=False)), ("unreadable-state", dict(state=False))]:
        f = build_repo(tmp / f"rm_{name}", **kw)
        r = run(["bash", str(RISK)], env_extra={"RISK_MONITOR_REPO": str(f)})
        check(f"ALARM exit 2 on {name}", r.returncode == 2, f"rc={r.returncode} out={r.stdout}")

    # widened coverage: a non-default oanda_environment="live" assignment (Hard NO 4 / no real money)
    la = build_repo(tmp / "rm_live_assign",
                    extra_files={"src/scanner/profiles.py": 'oanda_environment = "live"\n'})
    r = run(["bash", str(RISK)], env_extra={"RISK_MONITOR_REPO": str(la)})
    check("ALARM exit 2 on live env assignment (Hard NO 4)", r.returncode == 2, f"rc={r.returncode} {r.stdout}")


def test_verify_gate(tmp: Path):
    print("\n[verify_gate.py]")
    good = build_repo(tmp / "vg_good")
    r = run([sys.executable, str(VERIFY), "--repo", str(good), "--out", str(tmp / "vg_good/v.json")])
    check("PASS exit 0 on good fixture", r.returncode == 0, f"rc={r.returncode} {r.stdout}")
    v = json.loads((tmp / "vg_good/v.json").read_text())
    check("verdict.gate==PASS", v["gate"] == "PASS")
    check("verdict.hard_no_ok==True", v["hard_no_ok"] is True)

    for name, kw in [("live", dict(env="live")), ("gap", dict(gap="0.30")), ("halt", dict(halt=False)),
                     ("state", dict(state=False))]:
        f = build_repo(tmp / f"vg_{name}", **kw)
        outp = tmp / f"vg_{name}/v.json"
        r = run([sys.executable, str(VERIFY), "--repo", str(f), "--out", str(outp)])
        v = json.loads(outp.read_text())
        check(f"FAIL exit 2 on {name}", r.returncode == 2, f"rc={r.returncode}")
        check(f"hard_no_ok False on {name}", v["hard_no_ok"] is False)

    # finding #2: a non-default oanda_environment="live" assignment elsewhere in src must FAIL
    prof = build_repo(tmp / "vg_profile_live",
                      extra_files={"src/scanner/profiles.py": 'oanda_environment = "live"\n'})
    outp = tmp / "vg_profile_live/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(prof), "--out", str(outp)])
    v = json.loads(outp.read_text())
    check("FAIL on profile-override live assignment (finding #2)", r.returncode == 2 and v["hard_no_ok"] is False, v)

    # finding #1: a present-but-UNREGISTERED stop_gate must FAIL the gate (integrity), hard_no still ok
    unwired = build_repo(tmp / "vg_unwired", register_stop=False)
    outp = tmp / "vg_unwired/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(unwired), "--out", str(outp)])
    v = json.loads(outp.read_text())
    check("FAIL when Stop hook NOT registered (finding #1)", r.returncode == 2 and v["gate"] == "FAIL", v)
    check("  hard_no still ok on unwired (it's an integrity gap)", v["hard_no_ok"] is True, v)

    # finding #3: AST ship-gate check resists regex evasion (.15 / 1e-1 / expr / non-literal)
    for label, gapval, should_fail in [
            ("dot15", ".15", True), ("sci_ok", "1e-1", False), ("sci_bad", "2e-1", True),
            ("expr", "0.10 + 0.05", True), ("nonliteral", "SOME_CONST", True)]:
        f = build_repo(tmp / f"vg_gap_{label}", gap=gapval)
        outp = tmp / f"vg_gap_{label}/v.json"
        r = run([sys.executable, str(VERIFY), "--repo", str(f), "--out", str(outp)])
        v = json.loads(outp.read_text())
        if should_fail:
            check(f"AST gap FAIL on '{gapval}'", r.returncode == 2 and v["hard_no_ok"] is False, v)
        else:
            check(f"AST gap PASS on '{gapval}' (<=0.10)", r.returncode == 0 and v["hard_no_ok"] is True, v)

    # round-2 finding #1: AugAssign + tuple-unpack targets must NOT slip a >0.10 gap through
    aug = build_repo(tmp / "vg_gap_aug",
                     extra_files={"scripts/train.py": "HARD_MAX_GAP = 0.05\nHARD_MAX_GAP += 0.10\n"})
    outp = tmp / "vg_gap_aug/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(aug), "--out", str(outp)])
    v = json.loads(outp.read_text())
    check("AST gap FAIL on augmented assignment (+=)", r.returncode == 2 and v["hard_no_ok"] is False, v)

    tup = build_repo(tmp / "vg_gap_tuple",
                     extra_files={"scripts/train.py": "HARD_MAX_GAP, b = 0.2, 1\n"})
    outp = tmp / "vg_gap_tuple/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(tup), "--out", str(outp)])
    v = json.loads(outp.read_text())
    check("AST gap FAIL on tuple-unpack target", r.returncode == 2 and v["hard_no_ok"] is False, v)

    walrus = build_repo(tmp / "vg_gap_walrus",
                        extra_files={"scripts/train.py": "x = (HARD_MAX_GAP := 0.2)\n"})
    outp = tmp / "vg_gap_walrus/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(walrus), "--out", str(outp)])
    v = json.loads(outp.read_text())
    check("AST gap FAIL on walrus assignment (:=)", r.returncode == 2 and v["hard_no_ok"] is False, v)

    # round-2 finding #2: an inverted guard (return in the ELSE branch) must NOT pass
    inv = build_repo(tmp / "vg_halt_inverted", extra_files={
        "src/scanner/execution.py":
        "def execute_trade(self):\n    if StateEngine().get_halted():\n        proceed()\n    else:\n        return None\n"})
    outp = tmp / "vg_halt_inverted/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(inv), "--out", str(outp)])
    v = json.loads(outp.read_text())
    check("AST halt-guard FAIL on inverted else-branch guard", r.returncode == 2 and v["hard_no_ok"] is False, v)

    # finding #3: a commented-out / dead halt guard must NOT satisfy the check (AST ignores comments)
    commented = build_repo(tmp / "vg_halt_comment", extra_files={
        "src/scanner/execution.py":
        "def execute_trade(self):\n    # if StateEngine().get_halted():  execute_trade BLOCKED\n"
        "    #     return None\n    return None\n"})
    outp = tmp / "vg_halt_comment/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(commented), "--out", str(outp)])
    v = json.loads(outp.read_text())
    check("AST halt-guard FAIL on commented/dead guard", r.returncode == 2 and v["hard_no_ok"] is False, v)

    # memory enforcement: a lesson with no recall-trigger row fails the gate (integrity, not Hard-NO)
    orphan = build_repo(tmp / "vg_orphan_lesson", extra_files={
        ".claude/LESSONS.md": "# Lessons\n## L-099 — orphan\nbody, no recall-trigger index\n"})
    outp = tmp / "vg_orphan_lesson/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(orphan), "--out", str(outp)])
    v = json.loads(outp.read_text())
    check("orphan lesson (no trigger) FAILs gate", r.returncode == 2 and v["gate"] == "FAIL", v)
    check("  hard_no still ok (memory integrity gap)", v["hard_no_ok"] is True, v)


def test_loop_gate(tmp: Path):
    print("\n[loop_gate.py]  (risk injected so the stop-logic is isolated)")
    repo = build_repo(tmp / "lg")

    def decide(state: dict, risk="green") -> dict:
        sp = tmp / "lg_state.json"
        sp.write_text(json.dumps(state))
        r = run([sys.executable, str(LOOPG), "--repo", str(repo), "--state", str(sp),
                 "--risk-status", risk])
        return json.loads(r.stdout), r.returncode

    # CONTINUE: progress being made
    out, rc = decide({"open_questions": 2, "cycles": [
        {"n": 1, "new_lessons": 0, "new_verified_facts": 3, "open_questions_after": 2, "verdict": "FAIL"}]})
    check("CONTINUE when new info found", out["decision"] == "CONTINUE", out)

    # STOP-DONE: PASS + no new info + zero open questions
    out, rc = decide({"open_questions": 0, "cycles": [
        {"n": 1, "new_lessons": 1, "new_verified_facts": 2, "open_questions_after": 1, "verdict": "FAIL"},
        {"n": 2, "new_lessons": 0, "new_verified_facts": 0, "open_questions_after": 0, "verdict": "PASS"}]})
    check("STOP-DONE on verified completion", out["decision"] == "STOP-DONE", out)

    # STOP-CHURN: anti-stall — 2 empty cycles, open questions not decreasing
    out, rc = decide({"open_questions": 3, "cycles": [
        {"n": 1, "new_lessons": 0, "new_verified_facts": 0, "open_questions_after": 3, "verdict": "FAIL"},
        {"n": 2, "new_lessons": 0, "new_verified_facts": 0, "open_questions_after": 3, "verdict": "FAIL"}]})
    check("STOP-CHURN on synthetic stall", out["decision"] == "STOP-CHURN", out)
    check("STOP-CHURN reason mentions stall", "stall" in out["reason"].lower())

    # churn must NOT trigger if open questions ARE decreasing (real progress despite no lesson)
    out, rc = decide({"open_questions": 1, "cycles": [
        {"n": 1, "new_lessons": 0, "new_verified_facts": 0, "open_questions_after": 3, "verdict": "FAIL"},
        {"n": 2, "new_lessons": 0, "new_verified_facts": 0, "open_questions_after": 1, "verdict": "FAIL"}]})
    check("NOT churn when open questions shrink", out["decision"] == "CONTINUE", out)

    # fresh anti-stall cases: 3-cycle deep stall, and open-questions GROWING while empty (regression)
    out, rc = decide({"open_questions": 4, "cycles": [
        {"n": 1, "new_lessons": 0, "new_verified_facts": 0, "open_questions_after": 4, "verdict": "FAIL"},
        {"n": 2, "new_lessons": 0, "new_verified_facts": 0, "open_questions_after": 4, "verdict": "FAIL"},
        {"n": 3, "new_lessons": 0, "new_verified_facts": 0, "open_questions_after": 4, "verdict": "FAIL"}]})
    check("STOP-CHURN on 3-cycle deep stall", out["decision"] == "STOP-CHURN", out)

    out, rc = decide({"open_questions": 5, "cycles": [
        {"n": 1, "new_lessons": 0, "new_verified_facts": 0, "open_questions_after": 3, "verdict": "FAIL"},
        {"n": 2, "new_lessons": 0, "new_verified_facts": 0, "open_questions_after": 5, "verdict": "FAIL"}]})
    check("STOP-CHURN when open questions GROW while empty", out["decision"] == "STOP-CHURN", out)

    # masked stall: one trivial self-reported fact per cycle must NOT dodge churn (new contract)
    out, rc = decide({"open_questions": 3, "cycles": [
        {"n": 1, "new_lessons": 0, "new_verified_facts": 0, "open_questions_after": 3, "verdict": "FAIL"},
        {"n": 2, "new_lessons": 0, "new_verified_facts": 1, "open_questions_after": 3, "verdict": "FAIL"},
        {"n": 3, "new_lessons": 0, "new_verified_facts": 0, "open_questions_after": 3, "verdict": "FAIL"}]})
    check("STOP-CHURN on masked stall (1 trivial fact/cycle, flat open qs)", out["decision"] == "STOP-CHURN", out)

    # learning a lesson IS progress -> escapes churn even with flat open questions
    out, rc = decide({"open_questions": 3, "cycles": [
        {"n": 1, "new_lessons": 1, "new_verified_facts": 0, "open_questions_after": 3, "verdict": "FAIL"},
        {"n": 2, "new_lessons": 0, "new_verified_facts": 0, "open_questions_after": 3, "verdict": "FAIL"}]})
    check("CONTINUE when a lesson was learned (progress despite flat open qs)", out["decision"] == "CONTINUE", out)

    # absolute backstop: a fabricated lesson every 3rd cycle can't keep the loop alive forever
    periodic = [{"n": i + 1, "new_lessons": 1 if i % 3 == 0 else 0, "new_verified_facts": 0,
                 "open_questions_after": 4, "verdict": "FAIL"} for i in range(7)]
    out, rc = decide({"open_questions": 4, "cycles": periodic})
    check("STOP-CHURN backstop: no question closed in 6+ cycles despite periodic lessons",
          out["decision"] == "STOP-CHURN", out)

    # backstop stays quiet when open questions net-decrease over the long window
    progressing = [{"n": i + 1, "new_lessons": 0, "new_verified_facts": 0,
                    "open_questions_after": 6 - i, "verdict": "FAIL"} for i in range(6)]
    out, rc = decide({"open_questions": 1, "cycles": progressing})
    check("backstop quiet when open questions net-decrease over long window",
          out["decision"] == "CONTINUE", out)

    # HALT: risk alarm overrides everything
    out, rc = decide({"open_questions": 0, "cycles": [
        {"n": 1, "new_lessons": 0, "new_verified_facts": 0, "open_questions_after": 0, "verdict": "PASS"}]},
        risk="alarm")
    check("HALT-SAFETY on risk alarm (exit 2)", out["decision"] == "HALT-SAFETY" and rc == 2, (out, rc))

    # STOP-BLOCKED: irreversible fork
    out, rc = decide({"open_questions": 1, "blocked_on_irreversible": True, "cycles": [
        {"n": 1, "new_lessons": 0, "new_verified_facts": 1, "open_questions_after": 1, "verdict": "FAIL"}]})
    check("STOP-BLOCKED on irreversible fork", out["decision"] == "STOP-BLOCKED", out)

    # fail-closed: unreadable loop state -> HALT
    r = run([sys.executable, str(LOOPG), "--repo", str(repo), "--state", str(tmp / "nope.json"),
             "--risk-status", "green"])
    # missing file -> treated as empty state -> CONTINUE (no cycles). Corrupt file -> HALT:
    bad = tmp / "corrupt.json"; bad.write_text("{not json")
    r = run([sys.executable, str(LOOPG), "--repo", str(repo), "--state", str(bad), "--risk-status", "green"])
    check("HALT on corrupt loop state (fail-closed, exit 2)", r.returncode == 2, r.stdout)


def test_stop_gate(tmp: Path):
    print("\n[stop_gate.sh]")
    good = build_repo(tmp / "sg_good")
    r = run(["bash", str(STOP)], env_extra={"RISK_MONITOR_REPO": str(good)}, stdin="{}")
    check("exit 0 on GREEN repo", r.returncode == 0, f"rc={r.returncode} {r.stderr}")

    bad = build_repo(tmp / "sg_bad", env="live")
    r = run(["bash", str(STOP)], env_extra={"RISK_MONITOR_REPO": str(bad)}, stdin="{}")
    check("BLOCK exit 2 on ALARM repo", r.returncode == 2, f"rc={r.returncode}")
    check("block message surfaces HALT-SAFETY", "HALT-SAFETY" in r.stderr, r.stderr)

    # loop guard: even on ALARM, stop_hook_active=true must not block (no trap)
    r = run(["bash", str(STOP)], env_extra={"RISK_MONITOR_REPO": str(bad)},
            stdin=json.dumps({"stop_hook_active": True}))
    check("loop guard: exit 0 when stop_hook_active (no trap)", r.returncode == 0, f"rc={r.returncode}")


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_risk_monitor(tmp)
        test_verify_gate(tmp)
        test_loop_gate(tmp)
        test_stop_gate(tmp)
    print(f"\n==== {_passed} passed, {_failed} failed ====")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
