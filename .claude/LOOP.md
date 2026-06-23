# LOOP — the self-improver cycle (anti-stall by design)

This is the engine that makes the context system *improve* instead of just respond. It defines the
cycle, the **objective stopping conditions** (so the loop never stalls and never churns), the
**separation of roles** (worker ≠ verifier ≠ risk monitor — no agent grades its own homework), and
how each turn feeds the next. Read this when starting any multi-step or autonomous task.

Scope guard: this is a **meta / dev loop** (like Ralph), never the trading runtime. Claude is never
in the per-scan / per-trade hot path. Nothing here runs inside `src/scanner` execution.

---

## The cycle

```
        ┌──────────────────────────────────────────────────────────────┐
        │  ① ORIENT   read INTENT → NOTES → LESSONS(triggers) → state.json │
        │  ② PLAN     find the load-bearing question; cheapest experiment │
        │  ③ ACT      make the change / run the experiment → real result  │
        │  ④ VERIFY   INDEPENDENT agent re-derives claims from disk  ◄── separate model/agent
        │  ⑤ EVOLVE   /evolve: sort learnings → INTENT/LESSONS/skill/NOTES│
        │  ⑥ DECIDE   evaluate STOPPING CONDITIONS → CONTINUE/STOP/HALT   │
        └──────────────────────────────────────────────────────────────┘
                    ▲                                          │
                    └───────────────── CONTINUE ───────────────┘
   ⟂ PARALLEL: risk_monitor.sh runs concurrently the whole time; ALARM → jump to HALT.
```

Three roles, never collapsed into one:
- **Worker** — plans and acts (the main agent / a domain specialist sub-agent).
- **Verifier** — a *separate* agent (Code Reviewer type, ideally a different model/context) that
  re-derives every load-bearing claim from disk. See `.claude/verifier.md`. Invoke via `/verify-task`.
- **Risk monitor** — `.claude/tools/risk_monitor.sh`, independent, runs in parallel, fail-closed.

## ② Bias to action (the anti-stall rule)

Most decisions hinge on one unknown — the **load-bearing question**. Identify it and run the
*cheapest experiment that answers it*. A 3-line patch + smoke test that yields a real number beats
three analysis docs. Investigation that does not move toward answering the load-bearing question is
procrastination, and the loop treats it as churn (see anti-stall detector below). When the question
is unanswered and a cheap experiment exists, **ACT — do not produce more prose.**

## Enforced vs advisory (what the harness guarantees vs what depends on compliance)

Some of this loop is now **enforced deterministically** (runs regardless of whether the model
remembers), the rest is **advisory** (auto-loaded instruction the model follows). Be honest about
which is which:

- **ENFORCED** — `risk_monitor.sh` runs at every turn-end via the **Stop hook** (`.claude/tools/stop_gate.sh`),
  fail-closed, blocks on ALARM (loop-guarded). `verify_gate.py` re-derives the Hard NOs from disk and
  cannot be narrated around. `loop_gate.py` computes the stopping decision from the disk run-state.
  All three are covered by `.claude/loop/tests/test_loop_enforcement.py` (29 real-disk, no-mock tests).
- **ADVISORY** — reading INTENT/LESSONS, recalling triggers, dispatching the *agent* verifier and
  domain specialists, running `/evolve`. Auto-loaded and prompted, but model-compliance-gated.
- **Already enforced in CODE (independent of this loop)** — no trade while `halted` (execution.py),
  >10% model quarantined (HARD_MAX_GAP). This loop is defense-in-depth, not the primary guard.

## ⑥ Objective stopping conditions (measurable — not vibes)

These are computed from disk by **`.claude/loop/loop_gate.py`** reading `.claude/loop/state.json`
(one entry per cycle: `new_lessons`, `new_verified_facts`, `open_questions_after`, `verdict`). The
decision is mechanical, not a judgment call. Evaluate in order; first match wins.

| Signal | Objective test | Action |
|---|---|---|
| **HALT-SAFETY** | `risk_monitor.sh` exits non-zero (ALARM), OR any Hard NO is at risk | **STOP immediately.** Surface the alarm. Never "fix forward" past a safety tripwire. |
| **STOP-BLOCKED** | The only remaining step is an irreversible/destructive fork (unhalt, dry_run→live, force-push, delete data you didn't create, spend real money) | **STOP.** Present the decision to the operator with the cost-of-wrong. Do not self-authorize. |
| **STOP-DONE** | Definition of Done (INTENT) all-green **AND** `/verify-task` returned PASS (all load-bearing claims CONFIRMED) **AND** this cycle produced **no new lesson and no new verified information** | **STOP.** Report with evidence sources. Task complete. |
| **STOP-CHURN** (anti-stall) | Over the **last ≤3 cycles**, open load-bearing questions did **not** net-decrease **and** no lesson was learned. Progress = close a question OR learn a lesson; self-reported "facts" that do neither don't count (closes the masked-stall evasion). | **STOP and escalate.** You are looping without progress — hand the blocker to the operator rather than burning cycles. |
| **CONTINUE** | A load-bearing question remains, a cheap experiment can answer it, risk monitor is GREEN, and the last cycle *did* yield new verified info/lessons | Run another cycle. |

Two failure modes this kills:
- **Premature stop** — "looks done" without an independent verifier PASS does not satisfy STOP-DONE.
- **Infinite churn / "stalled high-tier prompt responder"** — STOP-CHURN forces escalation after 2
  empty cycles; "more analysis" with no new verified information is not progress.

## How a turn feeds the next (compounding, not amnesia)

- Every cycle writes its delta to **NOTES.md** (state) so a compaction or fresh session resumes mid-loop.
- Every *generalizable* failure becomes a **LESSON** with a recall-trigger, so the next ORIENT step
  retrieves it automatically instead of re-learning it. Memory that isn't retrieved at planning time
  is dead weight — LESSONS triggers are how recall happens.
- Every reusable *procedure* becomes a **skill**, so it's executed, not re-derived.
- `/evolve` is the only sanctioned writer of doctrine, and it always STOPS for approval.

## Engaging the loop

- Solo / simple task: you may play worker + run `/verify-task` (separate agent) + `risk_monitor.sh`.
- Real / risky / multi-step task: dispatch a domain-specialist **worker** sub-agent, a separate
  **Code Reviewer verifier**, and run `risk_monitor.sh` in the background. Parallelize independent
  workers. Iterate ②–⑥ until a stopping condition fires. **Engineer until the goal is proper and
  complete — but stop the instant a stopping condition says so.**
