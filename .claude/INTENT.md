# INTENT — the operator's voice, standing decisions, definition of done

This file is the *why*. CLAUDE.md is the *what*, NOTES.md is the *now*, LESSONS.md is the
*never again*. Read INTENT before you plan anything non-trivial.

---

## Who I am, how I talk

- I'm the operator (Buddy). One person, one practice account, real intent to find a real edge.
- I write **terse, plain English**, sometimes voice-transcribed. If a word looks garbled or a
  sentence doesn't parse, **ask me — don't build on a guess.** A wrong guess on a trading system
  costs more than a clarifying question.
- I want a **partner who drives**, not a survey-taker. Decide reversible things yourself, tell me
  what you did and why in 1–2 lines. Save questions for the irreversible forks.

## End goal (the only thing that matters)

Use ML to find profitable trades on a **practice** account, or honestly prove the edge isn't there
yet at this data scale. Every task serves that outcome. If a task doesn't move toward a real,
verified number — or toward honestly closing a question — **question the task before doing it.**

## Standing decisions (don't re-litigate these)

1. **Production discipline over speed.** Anything touching trade execution or environment is
   **fail-closed** — when in doubt, refuse and surface, never proceed-and-hope.
2. **state.json is the source of truth for runtime state.** Files are memory; context is not. If
   you want to know if the bot is halted, read `.claude/state.json` *this turn* — don't trust your
   memory of it.
3. **Halt > break.** Staying halted costs opportunity. Unhalting a broken system costs real
   (practice) money and, worse, a false signal that the system works. Favor staying halted until
   validation is unambiguous.
4. **The edge is the ML signal, not the heuristics.** Don't pile on gates/agents to manufacture
   confidence. Empirically, price-only intraday caps at ~52% (closed many ways — see LESSONS and
   the verdict docs). Don't re-run news / foundation-model / more-data experiments without a
   *materially different* setup.
5. **Evidence over assertion.** "Should work" is not a status. Cite the file/line/log/number you
   read *this turn*. Operator pushback means re-verify from scratch, not re-explain.
6. **Reversible → act. Irreversible → ask.** Code patch, doc, experiment re-run = ship and learn.
   Unhalting, flipping dry_run→live, force-push, deleting data you didn't create = stop and confirm.
7. **Enforcement over instruction.** A rule that matters gets a deterministic gate (a hook or
   disk-reading code), not just a sentence in a doc. Advisory is for the non-load-bearing. (L-005.)

## Standing roadmap — the north star (drives every /evolve cycle)

This is not a one-off task; it is the persistent direction of the system. Every `/evolve` cycle
should ask "did this move us along the roadmap?" and the loop should keep engineering toward it:

> **Focus on strengthening the verifier (separate model/agent), tightening state/skill for better
> memory/lessons, objective stopping conditions, and parallel risk monitoring. This architecture
> turns it into a true self-improver rather than a stalled high-tier prompt responder.**

Concretely, the four standing fronts (raise the bar each cycle; never regress):
- **Verifier** — push separation harder: separate model/agent, adversarial, re-derives from disk,
  can fail the run. Make "self-grade" structurally impossible for load-bearing claims (deterministic
  half `verify_gate.py` + agent half). Close residual evasion surfaces, don't just document them.
- **Memory / state / skill** — lessons must fire at planning time (recall-trigger index, structurally
  enforced) and the loop must measurably write back what it learns each cycle (`state.json`
  new_lessons/new_verified_facts), growing LESSONS/INTENT via `/evolve` — never silent.
- **Stopping conditions** — disk-measurable and un-gameable (`loop_gate.py`); prove anti-stall on
  fresh synthetic cases each time the logic changes.
- **Risk monitoring** — parallel, fail-closed, deterministic (`stop_gate.sh` + `risk_monitor.sh`);
  widen coverage toward all four Hard NOs.

Engineering bar for work on these fronts: every claim backed by a test or a from-disk check; run
build→verify→fix until the SEPARATE verifier returns PASS with zero Hard-NO violations and no new
residual gaps. Iterate — don't stop at the first plausible draft.

## Definition of Done (a task is DONE only if ALL hold)

A change is done when:

1. **Trade-path contract honored** — it respects the fail-closed trade path: `execute_trade`
   (`src/scanner/execution.py:2050`) and the mid-cycle halt re-check (`:2081–2098`) still block
   when `state.halted=True`. You did not add a path that can fire a trade while halted.
2. **No Hard NO broken** — `oanda_environment` is still `"practice"`, halt still respected, nothing
   promoted to champion without passing the ship gate, no real-money path. (Full list: CLAUDE.md.)
3. **Scoped** — you changed what the task needed and adjacent cheap/safe gaps, nothing more. No
   "while I'm here" refactors, no profile/threshold/gate changes without my explicit decision.
4. **State reflects reality** — `.claude/state.json` (if runtime changed) and `.claude/NOTES.md`
   describe what is actually true now, not what you intended. Stale state is a defect.
5. **Evidence shown** — every status claim names its source (file:line, grep, log line, test
   output, a real number). Skipped verifications are named explicitly, not silently dropped.
6. **Independently verified** — both halves of the verifier PASS: the deterministic gate
   `python3 .claude/loop/verify_gate.py` exits 0 (re-derives the Hard NOs from disk) **and** the
   separate agent `/verify-task` returns PASS. The Stop hook (`stop_gate.sh`) already enforces
   `risk_monitor.sh` GREEN at turn-end. Self-attestation is not sufficient; f070d39 is why.

Done vs. keep-going is not a feeling — it's the **objective stopping conditions** in `.claude/LOOP.md`
(HALT-SAFETY / STOP-BLOCKED / STOP-DONE / STOP-CHURN / CONTINUE). STOP-DONE requires items 1–6 above.

If any of these fails, the task is not done — say so plainly. A half-done task reported honestly is
worth more to me than a "complete" that papers over a gap.

## What I will reliably ask you to do

- Read INTENT + NOTES + LESSONS before planning multi-step work.
- Plan first on anything non-trivial; chain the plan to a real result; report at checkpoints.
- Run `/evolve` after a real task so the system gets sharper. Never let a hard-won lesson evaporate.
