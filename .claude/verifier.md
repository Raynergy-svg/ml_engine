# VERIFIER — independent verification contract (separate agent / model)

The worker cannot grade its own homework. This contract defines a **separate verifier** whose only
job is to re-derive every load-bearing claim **from disk, this turn**, adversarially, and return a
structured PASS/FAIL. It is the enforcement teeth behind INTENT's Definition of Done and the VERIFY
step of the loop. Invoke it via `/verify-task`. It exists because the f070d39 incident proved that a
single agent's self-report is not evidence (see `.claude/rules/honesty.md`).

## Two halves — deterministic gate + agent judgment (mirrors the repo's policy-auditor pattern)

The verifier has two halves, and **both must pass**:

1. **Deterministic gate — `.claude/loop/verify_gate.py`.** Code that RE-DERIVES the machine-checkable
   claims and every Hard NO from disk (oanda_environment=="practice", HARD_MAX_GAP≤0.10 across
   src+scripts, halt guard present, state readable/not-live, context files intact). It writes
   `.claude/loop/verdict.json` and exits 0 (PASS) / 2 (FAIL). **It cannot be fooled by a narrative** —
   a lying worker or subagent can't make grep see "practice" where the file says "live". This is the
   enforced floor. Run it; a FAIL is non-negotiable.
2. **Agent judgment — a separate Code Reviewer sub-agent** (below) for the semantic/subjective claims
   the script can't check ("is this actually wired?", "does the fix address the root cause?").

Like the repo's own `policy-auditor` + deterministic Constitution check: **disagreement is a hard
stop.** If either half FAILs, the gate FAILs. This is defense-in-depth on top of the code guardrails
(execution.py halt, HARD_MAX_GAP quarantine) — never a replacement.

## Why a *separate* agent/model

- **Fresh context = no narrative bias.** The verifier did not write the code, so it has no sunk-cost
  belief that the change works. It starts from "prove it" not "confirm it."
- **Different model tier is allowed and encouraged.** Run the verifier on a different model/instance
  than the worker when possible. Independence of *judgment*, not just of *prompt*, is the point.
- **Specialist type:** dispatch as a **Code Reviewer** sub-agent (per project rule: Review → Code
  Reviewer). For test claims, an **API Tester**; for data claims, a **Data Engineer**.

## Input — the claim manifest

The worker hands the verifier a list of load-bearing claims, each with the source it *says* backs it:

```
CLAIM: <one sentence the task's correctness depends on>
SOURCE: <file:line / grep / log path / test name / number the worker cites>
```

Plus the diff/working-tree under review and the relevant Hard NOs.

## Method — adversarial, disk-first

For each claim the verifier MUST:
1. **Open the cited source itself** (Read/Grep/run the test/read the log). Memory of a prior tool
   call is not verification.
2. **Try to refute it.** Default to skepticism. Look for the parallel code path that already does
   the thing, the gate that isn't actually wired, the counter that lies, the test that passes while
   mocking the boundary.
3. **Integration-grep before "wired".** A class working in isolation ≠ a reachable call site. Grep
   for the call site in the production entry point.
4. **Distinguish shipped-to-disk from running-in-process**, and **fix-is-correct from fix-is-causal**.

## Output — structured verdict

```
VERIFIER VERDICT — <task>, <date>, model=<which>
Per-claim:
  [CONFIRMED]    <claim> — re-derived from <source verifier opened itself>
  [REFUTED]      <claim> — <what disk actually shows> ◄ blocks PASS
  [UNVERIFIABLE] <claim> — <why; queued as explicit next action> ◄ blocks PASS if load-bearing
Hard-NO check:
  [OK] practice / halt / ship-gate / no-real-money all intact   (or)  [VIOLATION] <which + evidence>
GATE: PASS   (only if every load-bearing claim CONFIRMED and zero Hard-NO violations)
      FAIL   (one or more REFUTED / UNVERIFIABLE load-bearing claims, or a Hard-NO violation)
```

## Gate semantics

- **PASS is required for STOP-DONE** (LOOP.md). No PASS → the task is not done, full stop.
- **A Hard-NO VIOLATION is an automatic FAIL and a HALT-SAFETY trigger** — it overrides everything,
  even if every functional claim is CONFIRMED.
- **UNVERIFIABLE ≠ pass.** An honest "couldn't confirm" keeps the gate FAIL and becomes a named next
  action — never silently dropped.
- The verifier reports findings; it does **not** edit code. Fixes go back to the worker, then
  re-verify. This is the build→verify→fix loop that runs until PASS or a stopping condition fires.
