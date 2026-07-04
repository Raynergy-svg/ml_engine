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
RECORD = LOOP_DIR / "record_cycle.py"
RECORDV = LOOP_DIR / "record_verdict.py"
INTEG = LOOP_DIR / "_integrity.py"
GENMAN = LOOP_DIR / "gen_manifest.py"
MANIFEST = LOOP_DIR / "gate_manifest.json"
RISK = TOOLS / "risk_monitor.sh"
STOP = TOOLS / "stop_gate.sh"
MANAGED_DIR = LOOP_DIR / "managed"

sys.path.insert(0, str(MANAGED_DIR))
import ml_engine_gate_wrapper as _wrapper  # noqa: E402  (root-owned managed Stop-hook gate)
import verify_managed_anchor as _anchor    # noqa: E402  (from-disk anchor verifier)

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
               register_stop=True, open_qs=0, mode="dry_run", extra_files: dict | None = None) -> Path:
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
        (root / ".claude/state.json").write_text(json.dumps({"halted": not (mode == "live"), "mode": mode}))
    for rel in CONTEXT_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x\n")
    # copy the real enforcement scripts so structural + path checks pass
    for src in (RISK, STOP, TOOLS / "session_context_boot.sh"):
        if src.exists():
            (root / ".claude/tools" / src.name).write_text(src.read_text())
            os.chmod(root / ".claude/tools" / src.name, 0o755)
    for src in (VERIFY, LOOPG, RECORD, RECORDV, INTEG, GENMAN):
        (root / ".claude/loop" / src.name).write_text(src.read_text())
    (root / ".claude/loop/tests").mkdir(parents=True, exist_ok=True)
    (root / ".claude/loop/tests/test_loop_enforcement.py").write_text(HERE.read_text())  # pinned suite
    if MANIFEST.exists():
        (root / ".claude/loop/gate_manifest.json").write_text(MANIFEST.read_text())
    (root / ".claude/loop/questions.json").write_text(json.dumps(
        {"questions": [{"id": f"q{i}", "status": "open"} for i in range(open_qs)]}))
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


def wf_lesson(lid: str, title: str) -> str:
    """A well-formed lesson (all five fields, non-trivial body) for fixtures."""
    return (f"## {lid} — {title}\n"
            f"- Trigger: when {title} occurs, described with enough detail to clear the length floor here.\n"
            f"- Root cause: the underlying mechanism explained at sufficient length for the content audit.\n"
            f"- Rule: do the deterministic thing and fail closed on ambiguity, always, no exceptions.\n"
            f"- Scope: the whole self-improver loop and its tooling.\n"
            f"- Source: 2026-06-23 test fixture, well-formed by construction.\n\n")


# ---- tests --------------------------------------------------------------------------------------

def test_risk_monitor(tmp_path: Path):
    tmp = tmp_path
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

    # operator-directed: mode=live on PRACTICE env = paper execution -> GREEN (not real money)
    mlp = build_repo(tmp / "rm_mode_live_practice", mode="live")
    r = run(["bash", str(RISK)], env_extra={"RISK_MONITOR_REPO": str(mlp)})
    check("GREEN exit 0 on mode=live + env=practice (paper execution allowed)", r.returncode == 0, f"rc={r.returncode} {r.stdout}")
    # but mode=live on a LIVE env (real money) stays a HARD alarm
    mll = build_repo(tmp / "rm_mode_live_liveenv", env="live", mode="live")
    r = run(["bash", str(RISK)], env_extra={"RISK_MONITOR_REPO": str(mll)})
    check("ALARM exit 2 on mode=live + env=live (real-money execution stays hard)", r.returncode == 2, f"rc={r.returncode} {r.stdout}")

    # per-lane halt visibility (2026-07-02 hardening): status line must never be blind to halted_lanes
    lanes_ok = build_repo(tmp / "rm_lanes_ok", mode="live", extra_files={".claude/state.json": json.dumps(
        {"halted": False, "mode": "live",
         "halted_lanes": {"oanda_fx": True, "equity": False, "brain": False, "crypto_momentum": True, "track_b": True}})})
    r = run(["bash", str(RISK)], env_extra={"RISK_MONITOR_REPO": str(lanes_ok)})
    check("GREEN exit 0 with well-formed halted_lanes", r.returncode == 0, f"rc={r.returncode} {r.stdout}")
    check("status line reports oanda_fx_halted=True", "oanda_fx_halted=True" in r.stdout, r.stdout)
    check("status line reports equity_halted=False", "equity_halted=False" in r.stdout, r.stdout)
    check("status line reports brain_halted=False", "brain_halted=False" in r.stdout, r.stdout)
    check("status line reports crypto_momentum_halted=True", "crypto_momentum_halted=True" in r.stdout, r.stdout)
    check("status line reports track_b_halted=True", "track_b_halted=True" in r.stdout, r.stdout)

    # halted_lanes entirely absent = valid legacy/global-only state (StateEngine.get_halted() defers
    # to global) -> must NOT alarm on its own, but must show "?" rather than a cheerful "False"
    lanes_absent = build_repo(tmp / "rm_lanes_absent", mode="live")  # default fixture has no halted_lanes key
    r = run(["bash", str(RISK)], env_extra={"RISK_MONITOR_REPO": str(lanes_absent)})
    check("GREEN exit 0 when halted_lanes entirely absent (legacy state, not an anomaly)",
          r.returncode == 0, f"rc={r.returncode} {r.stdout}")
    check("absent halted_lanes shown as '?' not silently 'False'", "oanda_fx_halted=?" in r.stdout, r.stdout)

    # halted_lanes PRESENT but missing a known lane's entry = corruption, not legacy absence -> ALARM
    lanes_incomplete = build_repo(tmp / "rm_lanes_incomplete", mode="live", extra_files={".claude/state.json": json.dumps(
        {"halted": False, "mode": "live", "halted_lanes": {"oanda_fx": True, "equity": False}})})  # brain missing
    r = run(["bash", str(RISK)], env_extra={"RISK_MONITOR_REPO": str(lanes_incomplete)})
    check("ALARM exit 2 when halted_lanes present but missing a known lane entry",
          r.returncode == 2, f"rc={r.returncode} {r.stdout}")
    check("alarm names the incomplete lane 'brain'", "brain" in r.stdout, r.stdout)

    # global halted=True must force every lane to read True regardless of halted_lanes contents
    lanes_global = build_repo(tmp / "rm_lanes_global_halt", extra_files={".claude/state.json": json.dumps(
        {"halted": True, "mode": "dry_run",
         "halted_lanes": {"oanda_fx": False, "equity": False, "brain": False, "crypto_momentum": False, "track_b": False}})})
    r = run(["bash", str(RISK)], env_extra={"RISK_MONITOR_REPO": str(lanes_global)})
    check("GREEN exit 0 with global halted=True", r.returncode == 0, f"rc={r.returncode} {r.stdout}")
    check("global halt forces all lanes True regardless of halted_lanes",
          all(f"{lane}_halted=True" in r.stdout for lane in ("oanda_fx", "equity", "brain", "crypto_momentum", "track_b")), r.stdout)


def test_verify_gate(tmp_path: Path):
    tmp = tmp_path
    print("\n[verify_gate.py]")
    good = build_repo(tmp / "vg_good")
    r = run([sys.executable, str(VERIFY), "--repo", str(good), "--out", str(tmp / "vg_good/v.json")])
    check("PASS exit 0 on good fixture", r.returncode == 0, f"rc={r.returncode} {r.stdout}")
    v = json.loads((tmp / "vg_good/v.json").read_text())
    check("verdict.gate==PASS", v["gate"] == "PASS")
    check("verdict.hard_no_ok==True", v["hard_no_ok"] is True)

    # operator-directed: mode=live on PRACTICE env -> state check OK (paper execution, not real money)
    mlp = build_repo(tmp / "vg_mode_live_practice", mode="live")
    r = run([sys.executable, str(VERIFY), "--repo", str(mlp), "--out", str(tmp / "vg_mlp.json")])
    v = json.loads((tmp / "vg_mlp.json").read_text())
    sc = next((c for c in v["checks"] if c["name"] == "state_readable_not_live"), {})
    check("mode=live + env=practice -> state check OK + gate PASS", r.returncode == 0 and sc.get("ok") is True, (sc, v["gate"]))

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

    # 2026-07-03: form (b) — the fail-closed assign-then-branch guard now live in execution.py
    # (`_halted = ...get_halted_strict()` in a try; except sets True; `if _halted: return`) must PASS
    formb = build_repo(tmp / "vg_halt_formb", extra_files={
        "src/scanner/execution.py":
        "def execute_trade(self):\n"
        "    try:\n"
        "        _halted = StateEngine(lane='oanda_fx').get_halted_strict()\n"
        "    except Exception:\n"
        "        _halted = True\n"
        "    if _halted:\n"
        "        logger.warning('execute_trade BLOCKED — state.halted=True')\n"
        "        return None\n"})
    outp = tmp / "vg_halt_formb/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(formb), "--out", str(outp)])
    v = json.loads(outp.read_text())
    check("AST halt-guard PASS on assign-then-branch get_halted_strict form",
          r.returncode == 0 and v["hard_no_ok"] is True, v)

    # inverted form (b): Return only in the ELSE branch must NOT pass (same rule as form (a))
    formb_inv = build_repo(tmp / "vg_halt_formb_inverted", extra_files={
        "src/scanner/execution.py":
        "def execute_trade(self):\n"
        "    _halted = StateEngine(lane='oanda_fx').get_halted_strict()\n"
        "    if _halted:\n"
        "        proceed()\n"
        "    else:\n"
        "        return None\n"})
    outp = tmp / "vg_halt_formb_inverted/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(formb_inv), "--out", str(outp)])
    v = json.loads(outp.read_text())
    check("AST halt-guard FAIL on inverted assign-then-branch guard",
          r.returncode == 2 and v["hard_no_ok"] is False, v)

    # memory enforcement: a lesson with no recall-trigger row fails the gate (integrity, not Hard-NO)
    orphan = build_repo(tmp / "vg_orphan_lesson", extra_files={
        ".claude/LESSONS.md": "# Lessons\n## L-099 — orphan\nbody, no recall-trigger index\n"})
    outp = tmp / "vg_orphan_lesson/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(orphan), "--out", str(outp)])
    v = json.loads(outp.read_text())
    check("orphan lesson (no trigger) FAILs gate", r.returncode == 2 and v["gate"] == "FAIL", v)
    check("  hard_no still ok (memory integrity gap)", v["hard_no_ok"] is True, v)

    # red-team #2: a neutered gate script (content drift vs manifest) must FAIL the gate
    tampered = build_repo(tmp / "vg_gate_tamper")
    rm = tampered / ".claude/tools/risk_monitor.sh"
    rm.write_text(rm.read_text() + "\n# neutered\n")  # change content -> hash drift
    outp = tmp / "vg_gate_tamper/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(tampered), "--out", str(outp)])
    v = json.loads(outp.read_text())
    drift_check = next((c for c in v["checks"] if c["name"] == "gate_scripts_unmodified"), {})
    check("neutered gate script FAILs gate (hash drift)", r.returncode == 2 and v["gate"] == "FAIL", v)
    check("  gate_scripts_unmodified flags the drift", drift_check.get("ok") is False, drift_check)

    # verifier-prescribed: manifest ENTRY-DROP (delete a pin + neuter the now-unlisted script) FAILs
    dropped = build_repo(tmp / "vg_entry_drop")
    man_path = dropped / ".claude/loop/gate_manifest.json"
    man = json.loads(man_path.read_text())
    man.pop(".claude/tools/risk_monitor.sh", None)  # drop the pin
    man_path.write_text(json.dumps(man, indent=2, sort_keys=True))
    rm2 = dropped / ".claude/tools/risk_monitor.sh"
    rm2.write_text("#!/usr/bin/env bash\necho GREEN; exit 0\n")  # neuter the now-unlisted script
    outp = tmp / "vg_entry_drop/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(dropped), "--out", str(outp)])
    v = json.loads(outp.read_text())
    dc = next((c for c in v["checks"] if c["name"] == "gate_scripts_unmodified"), {})
    check("manifest entry-drop FAILs gate (coverage check, fail-closed)", r.returncode == 2 and dc.get("ok") is False, dc)

    # red-team #3: a shallow/empty lesson (bumps the counter without content) FAILs the gate
    shallow = build_repo(tmp / "vg_shallow_lesson", extra_files={
        ".claude/LESSONS.md": "Recall-trigger\n| x | L-099 |\n---\n" + wf_lesson("L-001", "real one") +
        "## L-099 — filler\nno fields here\n"})
    outp = tmp / "vg_shallow_lesson/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(shallow), "--out", str(outp)])
    v = json.loads(outp.read_text())
    wf = next((c for c in v["checks"] if c["name"] == "lessons_well_formed"), {})
    check("shallow lesson FAILs gate (content audit)", r.returncode == 2 and wf.get("ok") is False, wf)

    # red-team #4b: a Stop hook pointing at a STUB stop_gate.sh (wrong path) must FAIL registration
    stub = build_repo(tmp / "vg_stub_hook", register_stop=False)
    (stub / ".claude/settings.json").write_text(json.dumps({"hooks": {"Stop": [
        {"hooks": [{"type": "command", "command": "/tmp/evil/stop_gate.sh"}]}]}}))
    outp = tmp / "vg_stub_hook/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(stub), "--out", str(outp)])
    v = json.loads(outp.read_text())
    reg_check = next((c for c in v["checks"] if c["name"] == "stop_gate_registered"), {})
    check("stub-path Stop hook FAILs registration (basename hole closed)", reg_check.get("ok") is False, reg_check)

    # verifier-3 finding: the test suite is the recompute trust anchor -> editing it is gate-drift
    suite_tamper = build_repo(tmp / "vg_suite_tamper")
    st = suite_tamper / ".claude/loop/tests/test_loop_enforcement.py"
    st.write_text(st.read_text() + "\n# tampered to fake a pass count\n")
    outp = tmp / "vg_suite_tamper/v.json"
    r = run([sys.executable, str(VERIFY), "--repo", str(suite_tamper), "--out", str(outp)])
    v = json.loads(outp.read_text())
    sc = next((c for c in v["checks"] if c["name"] == "gate_scripts_unmodified"), {})
    check("tampered test-suite FAILs gate (suite hash-pinned)", r.returncode == 2 and sc.get("ok") is False, sc)


def test_loop_gate(tmp_path: Path):
    tmp = tmp_path
    print("\n[loop_gate.py]  (objective signals: open_questions from questions.json; facts/lessons from deltas)")
    repo = build_repo(tmp / "lg")
    qpath = tmp / "lg_questions.json"
    spath = tmp / "lg_state.json"

    def cyc(n, tp, vp=True, lc=0, oqa=0):
        return {"n": n, "tests_passed": tp, "verify_gate_pass": vp, "lessons_count": lc, "open_questions_after": oqa}

    def decide(cycles, open_qs, *, verify="pass", risk="green", blocked=False, lessons=None,
               tests=None, attest=False, verdict="fresh"):
        qpath.write_text(json.dumps({"questions": [{"id": f"q{i}", "status": "open"} for i in range(open_qs)]}))
        st = {"cycles": cycles}
        if blocked:
            st["blocked_on_irreversible"] = True
        if attest:
            st["no_work_needed_attested"] = True
        spath.write_text(json.dumps(st))
        # inject live lesson + test counts == last cycle's recorded values so logic tests don't trip
        # the tamper checks (override via lessons=/tests= to exercise tamper itself); verdict defaults
        # fresh so done-path tests pass (override verdict="stale" to exercise the agent-verdict gate)
        lc = lessons if lessons is not None else (int(cycles[-1].get("lessons_count", 0)) if cycles else 0)
        tc = tests if tests is not None else (int(cycles[-1].get("tests_passed", 0)) if cycles else 0)
        r = run([sys.executable, str(LOOPG), "--repo", str(repo), "--state", str(spath),
                 "--questions", str(qpath), "--verify-status", verify, "--risk-status", risk,
                 "--lessons-count", str(lc), "--tests-count", str(tc), "--verdict-status", verdict])
        return json.loads(r.stdout), r.returncode

    # CONTINUE: single cycle, an open question remains
    out, rc = decide([cyc(1, 10, lc=2, oqa=1)], open_qs=1)
    check("CONTINUE single cycle with an open question", out["decision"] == "CONTINUE", out)

    # STOP-DONE: LIVE verify PASS + zero open (from list) + no new facts/lessons this cycle
    out, rc = decide([cyc(1, 10, lc=2, oqa=1), cyc(2, 10, lc=2, oqa=0)], open_qs=0)
    check("STOP-DONE: live PASS, 0 open (list), no new info", out["decision"] == "STOP-DONE", out)

    # masked stall: tests_passed climbs every cycle but open qs flat & no lesson -> CHURN (facts don't save)
    out, rc = decide([cyc(1, 10, lc=2, oqa=3), cyc(2, 11, lc=2, oqa=3), cyc(3, 12, lc=2, oqa=3)], open_qs=3)
    check("STOP-CHURN on masked stall (facts climb, flat open qs, no lesson)", out["decision"] == "STOP-CHURN", out)
    check("STOP-CHURN reason mentions stall", "stall" in out["reason"].lower())

    # NOT churn when a question closes (open qs decrease)
    out, rc = decide([cyc(1, 10, lc=2, oqa=3), cyc(2, 10, lc=2, oqa=1)], open_qs=1)
    check("NOT churn when a question closes", out["decision"] == "CONTINUE", out)

    # CONTINUE when a lesson is learned (lessons_count delta), flat open qs
    out, rc = decide([cyc(1, 10, lc=2, oqa=3), cyc(2, 10, lc=3, oqa=3)], open_qs=3)
    check("CONTINUE when a lesson learned (lessons_count delta)", out["decision"] == "CONTINUE", out)

    # 3-cycle deep stall -> CHURN
    out, rc = decide([cyc(1, 10, lc=2, oqa=4), cyc(2, 10, lc=2, oqa=4), cyc(3, 10, lc=2, oqa=4)], open_qs=4)
    check("STOP-CHURN on 3-cycle deep stall", out["decision"] == "STOP-CHURN", out)

    # open questions GROW while empty -> CHURN
    out, rc = decide([cyc(1, 10, lc=2, oqa=3), cyc(2, 10, lc=2, oqa=5)], open_qs=5)
    check("STOP-CHURN when open questions grow", out["decision"] == "STOP-CHURN", out)

    # backstop: a lesson every 3rd cycle can't keep it alive if no question net-closes over 6 cycles
    periodic = [cyc(i + 1, 10, lc=2 + (i // 3), oqa=4) for i in range(7)]
    out, rc = decide(periodic, open_qs=4)
    check("STOP-CHURN backstop: no question closed in 6+ cycles despite periodic lessons",
          out["decision"] == "STOP-CHURN", out)

    # backstop quiet when open questions net-decrease over the long window
    progressing = [cyc(i + 1, 10, lc=2, oqa=6 - i) for i in range(6)]
    out, rc = decide(progressing, open_qs=1)
    check("backstop quiet when open questions net-decrease", out["decision"] == "CONTINUE", out)

    # ANTI-TAMPER: hand-edited verdict (state says PASS, live verify FAIL) -> HALT
    out, rc = decide([cyc(1, 10, vp=True, lc=2, oqa=0)], open_qs=0, verify="fail")
    check("HALT on tampered verdict (recorded PASS != live FAIL)", out["decision"] == "HALT-SAFETY" and rc == 2, out)
    check("  tamper reason names mismatch", ("tamper" in out["reason"].lower() or "stale" in out["reason"].lower()), out)

    # ANTI-TAMPER: state claims 0 open but questions.json has 2 open -> HALT
    out, rc = decide([cyc(1, 10, vp=True, lc=2, oqa=0)], open_qs=2)
    check("HALT on tampered open-count (recorded 0 != live 2)", out["decision"] == "HALT-SAFETY" and rc == 2, out)

    # ANTI-TAMPER: state claims lessons_count=5 but LESSONS.md has 2 well-formed -> HALT
    out, rc = decide([cyc(1, 10, vp=True, lc=5, oqa=0)], open_qs=0, lessons=2)
    check("HALT on tampered lessons_count (recorded 5 != live 2)", out["decision"] == "HALT-SAFETY" and rc == 2, out)

    # ANTI-TAMPER: state claims tests_passed=10 but the pinned suite has 999 -> HALT (closes #3a/#1d)
    out, rc = decide([cyc(1, 10, vp=True, lc=2, oqa=0)], open_qs=0, tests=999, attest=True)
    check("HALT on tampered tests_passed (recorded 10 != live 999)", out["decision"] == "HALT-SAFETY" and rc == 2, out)

    # ANTI-LAZINESS (#5): done-conditions met but ZERO observable work across the loop -> NOT done
    out, rc = decide([cyc(1, 10, vp=True, lc=2, oqa=0)], open_qs=0)
    check("no STOP-DONE on a zero-work loop (anti-laziness)", out["decision"] == "CONTINUE", out)
    check("  reason names the missing observable work", "observable work" in out["reason"].lower(), out)
    # ...unless a genuine no-op is explicitly attested
    out, rc = decide([cyc(1, 10, vp=True, lc=2, oqa=0)], open_qs=0, attest=True)
    check("STOP-DONE allowed with explicit no_work_needed_attested", out["decision"] == "STOP-DONE", out)
    # observable work (a question closed) reaches STOP-DONE without attestation
    out, rc = decide([cyc(1, 10, lc=2, oqa=1), cyc(2, 10, lc=2, oqa=0)], open_qs=0)
    check("STOP-DONE when a question was closed (observable work)", out["decision"] == "STOP-DONE", out)

    # Front #1a (lazy dimension): STOP-DONE requires a fresh PASS agent-verdict; skip it -> not done
    out, rc = decide([cyc(1, 10, lc=2, oqa=1), cyc(2, 10, lc=2, oqa=0)], open_qs=0, verdict="stale")
    check("no STOP-DONE without a fresh agent-verdict (lazy self-grade closed)", out["decision"] == "CONTINUE", out)
    check("  reason names the missing agent-verdict", "agent-verdict" in out["reason"].lower(), out)

    # HALT: risk alarm (latest cycle consistent so tamper passes first)
    out, rc = decide([cyc(1, 10, vp=True, lc=2, oqa=0)], open_qs=0, risk="alarm")
    check("HALT-SAFETY on risk alarm", out["decision"] == "HALT-SAFETY" and rc == 2, out)

    # STOP-BLOCKED: irreversible fork
    out, rc = decide([cyc(1, 10, vp=True, lc=2, oqa=1)], open_qs=1, blocked=True)
    check("STOP-BLOCKED on irreversible fork", out["decision"] == "STOP-BLOCKED", out)

    # fail-closed: corrupt state -> HALT
    bad = tmp / "corrupt.json"
    bad.write_text("{not json")
    r = run([sys.executable, str(LOOPG), "--repo", str(repo), "--state", str(bad),
             "--questions", str(qpath), "--verify-status", "pass", "--risk-status", "green"])
    check("HALT on corrupt loop state (fail-closed)", r.returncode == 2, r.stdout)

    # fail-closed: missing questions.json -> HALT
    r = run([sys.executable, str(LOOPG), "--repo", str(repo), "--state", str(spath),
             "--questions", str(tmp / "no_questions.json"), "--verify-status", "pass", "--risk-status", "green"])
    check("HALT on missing questions.json (fail-closed)", r.returncode == 2, r.stdout)

    # red-team #2: loop_gate independently HALTs on gate-script drift (neutered loop_gate/risk_monitor)
    drepo = build_repo(tmp / "lg_gate_tamper")
    rm = drepo / ".claude/tools/risk_monitor.sh"
    rm.write_text(rm.read_text() + "\n# neutered\n")
    qp = drepo / ".claude/loop/questions.json"
    sp = drepo / ".claude/loop/state.json"
    sp.write_text(json.dumps({"cycles": [{"n": 1, "tests_passed": 1, "verify_gate_pass": True,
                                          "lessons_count": 0, "open_questions_after": 0}]}))
    r = run([sys.executable, str(LOOPG), "--repo", str(drepo), "--state", str(sp),
             "--questions", str(qp), "--verify-status", "pass", "--risk-status", "green"])
    check("loop_gate HALTs on gate-script drift (independent of verify_gate)", r.returncode == 2, r.stdout)

    # tests_passed recompute ACTUALLY runs the suite (proven via a fake --test-cmd), clean env so the
    # recursion guard doesn't skip it. 2 cycles with a closed question => work observed.
    env_clean = {k: v for k, v in os.environ.items() if k != "LOOP_GATE_IN_RECOMPUTE"}
    qp2 = tmp / "rc_q.json"
    qp2.write_text(json.dumps({"questions": []}))
    sp2 = tmp / "rc_s.json"

    def recompute(last_tp):
        sp2.write_text(json.dumps({"cycles": [
            {"n": 1, "tests_passed": 5, "verify_gate_pass": True, "lessons_count": 0, "open_questions_after": 1},
            {"n": 2, "tests_passed": last_tp, "verify_gate_pass": True, "lessons_count": 0, "open_questions_after": 0}]}))
        return subprocess.run([sys.executable, str(LOOPG), "--repo", str(repo), "--state", str(sp2),
            "--questions", str(qp2), "--lessons-count", "0", "--verify-status", "pass", "--risk-status", "green",
            "--test-cmd", 'echo "5 passed, 0 failed"'], capture_output=True, text=True, env=env_clean, timeout=60)

    r = recompute(5)
    check("tests recompute runs the suite & matches -> no tamper-HALT", json.loads(r.stdout)["decision"] != "HALT-SAFETY", r.stdout)
    r = recompute(6)
    check("tests recompute catches a faked tests_passed -> HALT", r.returncode == 2, r.stdout)


def test_record_cycle(tmp_path: Path):
    tmp = tmp_path
    print("\n[record_cycle.py]  (objective measurement -> state.json -> loop_gate)")
    repo = build_repo(tmp / "rc", open_qs=0)
    (repo / ".claude/LESSONS.md").write_text("# Lessons\n" + wf_lesson("L-001", "alpha") + wf_lesson("L-002", "beta"))
    spath = repo / ".claude/loop/state.json"
    r = run([sys.executable, str(RECORD), "--repo", str(repo), "--summary", "test cycle",
             "--test-cmd", 'echo "7 passed, 0 failed"', "--verify-status", "pass", "--state", str(spath)])
    rec = json.loads(r.stdout)
    check("record_cycle measured tests_passed=7 (parsed from run output)", rec["tests_passed"] == 7, rec)
    check("record_cycle measured verify_gate_pass=True", rec["verify_gate_pass"] is True, rec)
    check("record_cycle measured lessons_count=2 (well-formed) from LESSONS.md", rec["lessons_count"] == 2, rec)
    check("record_cycle measured open_questions_after=0 from questions.json", rec["open_questions_after"] == 0, rec)
    # the recorded (objective) cycle drives loop_gate; one zero-delta cycle correctly does NOT declare
    # done (anti-laziness) — full pipeline, no self-report. (--tests-count/--lessons-count match the
    # measured values so the tamper checks pass without re-running the suite.)
    out = json.loads(run([sys.executable, str(LOOPG), "--repo", str(repo), "--state", str(spath),
          "--questions", str(repo / ".claude/loop/questions.json"), "--verify-status", "pass",
          "--risk-status", "green", "--tests-count", str(rec["tests_passed"]),
          "--lessons-count", str(rec["lessons_count"])]).stdout)
    check("recorded zero-delta cycle -> loop_gate CONTINUE (anti-laziness, objective pipeline)",
          out["decision"] == "CONTINUE", out)


def test_stop_gate(tmp_path: Path):
    tmp = tmp_path
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


def test_record_verdict(tmp_path: Path):
    tmp = tmp_path
    print("\n[record_verdict.py + loop_gate fresh-verdict gate]")
    repo = build_repo(tmp / "rv", open_qs=0)
    qp = repo / ".claude/loop/questions.json"
    sp = repo / ".claude/loop/state.json"
    vp = repo / ".claude/loop/agent_verdict.json"
    two = [{"n": 1, "tests_passed": 5, "verify_gate_pass": True, "lessons_count": 0, "open_questions_after": 1},
           {"n": 2, "tests_passed": 5, "verify_gate_pass": True, "lessons_count": 0, "open_questions_after": 0}]
    sp.write_text(json.dumps({"cycles": two}))

    def lg():
        return json.loads(run([sys.executable, str(LOOPG), "--repo", str(repo), "--state", str(sp),
            "--questions", str(qp), "--lessons-count", "0", "--tests-count", "5",
            "--verify-status", "pass", "--risk-status", "green"]).stdout)

    check("no agent_verdict file -> CONTINUE (must run the separate verifier)", lg()["decision"] == "CONTINUE")
    run([sys.executable, str(RECORDV), "--repo", str(repo), "--gate", "PASS", "--state", str(sp), "--out", str(vp)])
    check("fresh PASS agent-verdict bound to state -> STOP-DONE", lg()["decision"] == "STOP-DONE")
    # change state after verifying -> verdict bound is now stale -> must re-verify
    sp.write_text(json.dumps({"cycles": two + [
        {"n": 3, "tests_passed": 6, "verify_gate_pass": True, "lessons_count": 0, "open_questions_after": 0}]}))
    out = json.loads(run([sys.executable, str(LOOPG), "--repo", str(repo), "--state", str(sp),
        "--questions", str(qp), "--lessons-count", "0", "--tests-count", "6",
        "--verify-status", "pass", "--risk-status", "green"]).stdout)
    check("stale verdict after a state change -> CONTINUE (re-verify required)", out["decision"] == "CONTINUE", out)


def test_managed_wrapper(tmp_path: Path):
    tmp = tmp_path
    print("\n[managed wrapper: ml_engine_gate_wrapper.run_gate (root-owned trust anchor)]")
    good = build_repo(tmp / "mw_good")
    code, msg = _wrapper.run_gate(good, {"cwd": str(good)})
    check("allow (0) on intact in-scope repo", code == 0, (code, msg))
    code, _ = _wrapper.run_gate(good, {"stop_hook_active": True, "cwd": str(good)})
    check("loop guard: allow (0) on stop_hook_active", code == 0)
    code, _ = _wrapper.run_gate(good, {"cwd": "/tmp/some/other/project"})
    check("out-of-scope -> allow (0) no-op (won't disrupt other projects)", code == 0)

    drifted = build_repo(tmp / "mw_drift")
    rm = drifted / ".claude/tools/risk_monitor.sh"
    rm.write_text(rm.read_text() + "\n# neutered\n")
    code, msg = _wrapper.run_gate(drifted, {"cwd": str(drifted)})
    check("BLOCK (2) on neutered gate script (wrapper self-derives the drift)", code == 2 and "drift" in msg, (code, msg))

    dropped = build_repo(tmp / "mw_drop")
    mp = dropped / ".claude/loop/gate_manifest.json"
    m = json.loads(mp.read_text())
    m.pop(".claude/tools/risk_monitor.sh", None)
    mp.write_text(json.dumps(m))
    (dropped / ".claude/tools/risk_monitor.sh").write_text("#!/usr/bin/env bash\necho GREEN; exit 0\n")
    code, msg = _wrapper.run_gate(dropped, {"cwd": str(dropped)})
    check("BLOCK (2) on manifest entry-drop (baked coverage list)", code == 2 and "unlisted" in msg, (code, msg))

    alarmed = build_repo(tmp / "mw_alarm", env="live")
    code, msg = _wrapper.run_gate(alarmed, {"cwd": str(alarmed)})
    check("BLOCK (2) on risk-monitor ALARM (live env, scripts intact)", code == 2 and "ALARM" in msg, (code, msg))

    nomani = build_repo(tmp / "mw_nomani")
    (nomani / ".claude/loop/gate_manifest.json").unlink()
    code, msg = _wrapper.run_gate(nomani, {"cwd": str(nomani)})
    check("BLOCK (2) on missing manifest (fail-closed)", code == 2)


def test_managed_anchor(tmp_path: Path):
    tmp = tmp_path
    print("\n[managed anchor verifier: verify_managed_anchor.audit]")

    def write_anchor(name, *, wire=True, disable=False, readonly=True):
        dp = tmp / name
        dp.mkdir(parents=True, exist_ok=True)
        cmd = "python3 ml_engine_gate_wrapper.py" if wire else "echo hi"
        cfg = {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": cmd}]}]}}
        if disable is not None:
            cfg["disableAllHooks"] = disable
        ms = dp / "managed-settings.json"
        ms.write_text(json.dumps(cfg))
        wr = dp / "ml_engine_gate_wrapper.py"
        wr.write_text("x")
        if readonly:
            os.chmod(ms, 0o444)
            os.chmod(wr, 0o444)
        return dp

    ok, probs, _ = _anchor.audit(tmp / "ma_empty")
    check("NOT ok when nothing installed", not ok)

    ok, probs, _ = _anchor.audit(write_anchor("ma_writable", readonly=False))
    check("NOT ok when files are user-writable (not un-tamperable)",
          not ok and any("WRITABLE" in p for p in probs), probs)

    ok, probs, _ = _anchor.audit(write_anchor("ma_nodisable", disable=None))
    check("NOT ok without disableAllHooks:false (local-disable bypass open)",
          not ok and any("disableAllHooks" in p for p in probs), probs)

    ok, probs, _ = _anchor.audit(write_anchor("ma_disabletrue", disable=True))
    check("NOT ok with disableAllHooks:true", not ok and any("disableAllHooks" in p for p in probs), probs)

    ok, probs, _ = _anchor.audit(write_anchor("ma_wrongwire", wire=False))
    check("NOT ok when Stop hook doesn't wire the wrapper",
          not ok and any("wire the wrapper" in p for p in probs), probs)

    ok, probs, _ = _anchor.audit(write_anchor("ma_good"))
    check("OK: wired + not-user-writable + disableAllHooks:false", ok, probs)


def test_no_live_flip_scope(tmp_path: Path):
    tmp = tmp_path
    print("\n[verify_gate no_live_flip scope: docs/tests mentioning live strings don't false-trip]")
    repo = build_repo(tmp / "nlf")
    genv = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for args in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-qm", "base"]):
        subprocess.run(args, cwd=repo, env=genv, capture_output=True)
    # a doc that MENTIONS the live strings (like our own gate-teach docs) must NOT trip no_live_flip
    (repo / ".claude/NOTES.md").write_text('doc mentions oanda_environment = "live" and api-fxtrade.oanda.com\n')
    r = run([sys.executable, str(VERIFY), "--repo", str(repo), "--out", str(tmp / "nlf.json")])
    v = json.loads((tmp / "nlf.json").read_text())
    nf = next((c for c in v["checks"] if c["name"] == "no_live_flip"), {})
    check("doc mention of live strings does NOT trip no_live_flip (scoped to src/scripts)", nf.get("ok") is True, nf)
    # a REAL src env flip in the diff still fails hard
    (repo / "src/scanner/config.py").write_text('class C:\n    oanda_environment: str = "live"\n')
    r = run([sys.executable, str(VERIFY), "--repo", str(repo), "--out", str(tmp / "nlf2.json")])
    v = json.loads((tmp / "nlf2.json").read_text())
    check("real src env flip still caught (hard_no_ok False)", v["hard_no_ok"] is False, v)


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_risk_monitor(tmp)
        test_verify_gate(tmp)
        test_no_live_flip_scope(tmp)
        test_loop_gate(tmp)
        test_record_cycle(tmp)
        test_record_verdict(tmp)
        test_managed_wrapper(tmp)
        test_managed_anchor(tmp)
        test_stop_gate(tmp)
    print(f"\n==== {_passed} passed, {_failed} failed ====")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
