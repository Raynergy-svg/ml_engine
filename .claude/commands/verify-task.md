---
description: Independent verification — dispatch a SEPARATE Code Reviewer agent that re-derives every load-bearing claim from disk and returns a PASS/FAIL gate. Required for STOP-DONE. Distinct from the built-in /verify skill (which runs the app).
---

# /verify-task — independent claim verification (separate agent/model)

Run this before declaring any non-trivial task done. It enforces INTENT's Definition of Done and the
VERIFY step of `.claude/LOOP.md` by handing the work to a **separate verifier** that did not write
it. Contract: `.claude/verifier.md`.

## What you do

1. **Assemble the claim manifest** — list every claim the task's correctness depends on, each with
   the source the worker cites (file:line / grep / log / test / number). Be honest and complete;
   omitting a shaky claim defeats the purpose.

2. **Dispatch a separate verifier agent** — `subagent_type: "Code Reviewer"` (use **API Tester** for
   test-coverage claims, **Data Engineer** for data/pipeline claims). Run it on a different
   model/context than the worker when possible — independence of judgment is the goal. Brief it with:
   - the claim manifest,
   - the diff / working tree under review,
   - the Hard NOs (CLAUDE.md) and relevant LESSONS,
   - the explicit instruction: *re-derive each claim from disk yourself, try to refute it, do not
     trust the worker's narrative; return the structured verdict from `.claude/verifier.md`.*

3. **Run the parallel risk monitor** alongside: `.claude/tools/risk_monitor.sh`. A non-zero exit is
   a HALT-SAFETY condition that overrides any functional PASS.

4. **Apply the gate:**
   - **GATE: PASS** + risk monitor GREEN → STOP-DONE is satisfiable; report with evidence sources.
   - **GATE: FAIL** → fixes go back to the worker; re-run `/verify-task`. Build→verify→fix until PASS
     or a stopping condition (LOOP.md) fires.
   - **Hard-NO VIOLATION** → stop immediately, surface to operator, do not fix forward.

5. **Record the verdict in NOTES.md** (state only) so the loop's progress survives compaction.

## Guardrails

- The verifier reports; it does not edit. Keep worker and verifier separate — no self-grading.
- `UNVERIFIABLE` is not a pass. It keeps the gate FAIL and becomes a named next action.
- This command never relaxes a Hard NO and never marks done without an actual PASS.
