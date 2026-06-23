---
description: Close the learning loop — reflect on the finished task, sort what was learned into INTENT / LESSONS / a skill / NOTES, and present a diff-style proposal for approval. Never silently rewrites doctrine.
---

# /evolve — make the context system sharper after a real task

You just finished a real task. Before the context evaporates, harvest it. This command turns one
task's worth of hindsight into durable, reviewable updates to the five-file system. It is the
mechanism by which intent → execution gets *more reliable over time*.

## Hard constraints (these bind /evolve itself)

- **You propose; the operator disposes.** Present everything as a diff-style proposal, then **STOP
  and wait for approval.** The only file you may write without approval is `.claude/NOTES.md`
  (state only — never doctrine).
- **Never propose loosening a Hard NO.** `oanda_environment` stays `"practice"`; halt stays
  respected; nothing promotes without the ship gate; no real-money path. If the task "revealed" a
  reason to relax one of these, that is a finding to flag, not an edit to propose. (See L-003.)
- **Never silently rewrite doctrine.** Every change to CLAUDE.md / INTENT / LESSONS is shown as a
  before→after diff with a one-line rationale.
- **No invented lessons.** Only record what the *actual task* demonstrated. One observation is
  enough for a LESSON only on catastrophic/irreversible evidence; otherwise note it and watch for
  the pattern (the repo's bar is 3+ observations to promote a rule).

**Precondition:** `/evolve` runs at the EVOLVE step of `.claude/LOOP.md`, after the task reached a
stopping condition. If the task claimed STOP-DONE, it should already have a `/verify-task` PASS +
risk-monitor GREEN; if it stopped on STOP-CHURN/STOP-BLOCKED/HALT-SAFETY, capture *why it stalled or
halted* as a candidate LESSON — those are the highest-value learnings.

## Step 1 — Reflect (write this out, briefly)

Answer, grounded in what actually happened this task:
1. What was the task, and what is the *verified* outcome? (real number / file:line / test / log —
   not "should work")
2. What surprised me — a wrong assumption, a trap I hit, a thing that worked better than expected?
3. Did anything touch the trade path, environment, halt state, or ship gate? If so, did it stay
   fail-closed? Prove it.
4. What will the *next* session wish it had known before starting?

## Step 2 — Sort each learning into exactly one bucket

For every distinct thing learned, classify it:

| Bucket | Goes to | When |
|---|---|---|
| **New standing decision** (a *why*, a durable preference, a definition-of-done change) | `INTENT.md` | The operator made or confirmed a choice that should govern future work |
| **New failure mode** (a trap with a mechanism and an imperative) | `LESSONS.md` (next `L-NNN`) | Something broke or nearly broke and there's a rule that prevents recurrence |
| **New reusable pattern** (a repeatable *how* — a workflow, a recipe) | a skill under `.claude/skills/` (or `~/.claude`) | The same procedure will be run again and is worth codifying |
| **State only** (what's true now, in-flight work, a resolved assumption) | `NOTES.md` | It's a fact about the current moment, not doctrine — **you may write this one directly** |

If something fits no bucket, it's probably not worth persisting — say so and drop it.

## Step 3 — Present the proposal (diff-style), then STOP

Format:

```
## /evolve proposal — <task name>, <date>

### Reflection
<the 4 answers from Step 1, tight>

### Proposed changes  (NOTES.md already applied; everything else needs your OK)

[NOTES.md]  ✅ applied
  + <line added>
  - <line removed/pruned>

[LESSONS.md]  ⏳ needs approval
  + ## L-005 — <name>  [ACTIVE]
  +   Trigger: ...
  +   Root cause: ...
  +   Rule: ...
  +   Scope: ...
  +   Source: ...

[INTENT.md]  ⏳ needs approval
  before: <quoted line>
  after:  <quoted line>
  why:    <one line>

[skill: <name>]  ⏳ needs approval
  new file .claude/skills/<name>/SKILL.md — <one-line purpose>

### Hard-NO check
  ✅ No proposed change relaxes oanda_environment / halt / ship-gate / real-money.
  (or) ⚠️ FLAG: the task surfaced <X> near a Hard NO — reporting, not editing.
```

Then **stop.** Do not apply the ⏳ items until the operator says go. On approval, make exactly the
approved edits — no extras, no scope creep — and confirm what landed with the new file:line.

## Step 4 — On approval

- Append LESSONS with the next sequential `L-NNN` (never reuse a number; supersede, don't delete).
- Apply INTENT/CLAUDE.md edits exactly as shown in the diff.
- Create the skill if approved.
- Update NOTES to reflect that the evolution landed; prune anything now stale.
- Report: "Evolved: <n> lessons, <n> intent edits, <n> skills. Sources: <file:line list>."
