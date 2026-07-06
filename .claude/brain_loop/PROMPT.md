# Sonnet brain loop — how to run it

Design: `docs/superpowers/specs/2026-07-01-sonnet-brain-loop-design.md`.
Code: `src/brain_loop/` (hypothesis_registry, gate_runner, promotion, monitor, derisk, cycle).
Entrypoint: `scripts/run_brain_loop.py`.

**Not scheduled/cron'd by the task that built this.** This file documents how an operator
(or a scheduled Claude Code session) would start it — starting it is a separate decision.

## 1. Deterministic-only mode (no LLM, sanity-check the machinery)

```bash
python3 scripts/run_brain_loop.py --once
```

Reads `.claude/state.json`. If `halted=true` (current state), refuses immediately, mutates
nothing, exits 0. If not halted, evaluates rails (drawdown via the real ship-gate/decision-gate)
and reports. Safe to run against the live repo at any time — read-only unless a rail breach
fires, in which case it calls the same de-escalation path (`set_halted(True)`) that already
exists as an autonomous circuit breaker elsewhere in this codebase.

To register + gate a hypothesis this cycle:

```bash
python3 scripts/run_brain_loop.py --once --propose path/to/hypothesis.json
```

`hypothesis.json` shape:

```json
{
  "hypothesis_id": "h-2026-07-02-001",
  "name": "short slug",
  "mechanism": "one sentence: what edge/mechanism this tests",
  "novelty_justification": "why this is NOT a re-run of a closed front — cite what's materially different vs L-020/021/022",
  "params": {"...": "frozen before results — cannot be edited after this cycle runs"},
  "harness_cmd": ["python3", "scripts/backtest_harness.py", "--instrument", "..."]
}
```

## 2. "Sonnet is the brain" — the actual intended mode

Not an API call embedded in this script. A **scheduled Claude Code session**, same pattern as
`ralph.sh` / `/evolve`, that each cycle:

1. Reads `.claude/INTENT.md` → `.claude/NOTES.md` → `.claude/LESSONS.md` (recall-trigger index,
   especially L-020/021/022 before proposing anything — don't re-run a closed front without a
   materially new input).
2. Decides whether to propose a new hypothesis this cycle, and if so writes the
   `hypothesis.json` above (this is where Sonnet's judgment lives — novelty, mechanism,
   whether a prior gate failure means "try a variant" or "this is exhausted, stop").
3. Runs `python3 scripts/run_brain_loop.py --once --propose hypothesis.json` and reads the
   JSON summary back.
4. If `hypothesis_result.decision.decision == "continue"`, a shadow promotion was
   auto-written to `trained_data/brain_loop/promotion_requests/`. Sonnet may summarize this to
   NOTES.md. It never touches the ARM/`start_loop` dashboard flow — that stays a human action.
5. If `halted: true` came back, STOP — do not retry, do not propose working around it. Surface
   to the operator per `.claude/LOOP.md`'s HALT-SAFETY stopping condition.
6. After a cycle that produced a new lesson (a hypothesis exhausted, a rail tripped, a novel
   gate failure mode), run `/evolve` — same as any other task.

A minimal driver for this (not built by this task, described for when the operator wants it):

```bash
claude -p "$(cat .claude/brain_loop/PROMPT.md)" --dangerously-skip-permissions=false
```
scheduled via cron / `mcp__scheduled-tasks` at whatever cadence the operator picks (daily is a
reasonable start given backtests are not free).

## 3. What this loop can NEVER do (structurally, not by convention)

Verified by `tests/test_brain_loop_capability_absence.py` (grep-based, scoped to actual call
forms and import statements per L-015 — not just word mentions in docs):

- No import of `src.scanner.execution`, `src.brokers`, or any OANDA/broker client. It cannot
  place a trade.
- No call to `flatten(`, `set_gross_leverage(`, `start_loop(`, `unhalt(`, `arm(`, `disarm(`,
  or `set_halted(False`. It cannot self-arm, cannot flatten, cannot change leverage, cannot
  unhalt, cannot bypass the ARM checkpoint (`dashboard/server/control_safety.py:44`,
  `ARMED_REQUIRED_ACTIONS`).
- No autonomous call to `propose_promotion(..., target="live")` outside `promotion.py` itself
  (which only ever writes such a request as `PENDING_OPERATOR` — it never approves it).
- `derisk.halt()` has no `value` parameter at all — it is structurally incapable of unhalting.

## 4. Promoting a hypothesis to live (human step, outside this package)

1. Check `trained_data/brain_loop/promotion_requests/*-live.json` for `status:
   "PENDING_OPERATOR"` entries (there will be none from this package alone — a human or a
   future explicit step writes the `target="live"` request after reviewing a shadow run).
2. Review the attached `decision` (ship-gate PASS reasons) and the hypothesis in
   `trained_data/brain_loop/hypotheses.jsonl` (verify the hash chain with
   `hypothesis_registry.verify_chain`).
3. ARM via the existing AXIOM dashboard flow (`POST /api/control/arm`, 15-minute TTL,
   actor-attributed), then `start_loop` for the relevant strategy. Both are ARM-gated by
   `control_safety.enforce()` — this package never calls either.
