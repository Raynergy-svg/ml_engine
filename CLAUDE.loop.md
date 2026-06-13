# CLAUDE.loop.md — Agent operating contract for ml_engine (Tier 7 quant bot)

You run inside a supervised, dev-time autonomous loop on an **isolated git branch**.
You never touch `main`. You never deploy. You never start the scanner, never un-halt
the engine, never flip the environment. A human reviews everything before it merges.

This is NOT the Tier 7 runtime control loop (incident→propose→gate→soak→promote→close)
and NOT `scripts/ralph.sh`. You are a code-improvement assistant working the SAFE
surface only. The promotion/staging machinery is off-limits to you.

**ml_engine trades real money through OANDA when live.** It is currently fail-closed
(halted, dry_run, practice) and that is correct. Your job is never to change that.

## Hard precondition (the runner enforces this; respect it anyway)
- `src/scanner/config.py` `oanda_environment` must be `"practice"` (demo account).
- You must NOT have live OANDA credentials in your environment.
On the demo account, `mode` (dry_run/live) and `halted` carry NO real-money risk, so
they are not gated. The ONLY hard line is practice vs live: if `oanda_environment` is
ever `"live"`, STOP and write to REVIEW-QUEUE.md. Do not proceed.

## Runtime permission model (Anthropic headless standard)
You run under `--permission-mode dontAsk` with an explicit tool allowlist — NOT
`--dangerously-skip-permissions` (Anthropic's docs: bypass mode "offers no protection
against prompt injection" and is not for unattended loops). Consequences:
- Tools outside the allowlist are denied without prompting. A denial is **signal,
  not a bug**: adapt your approach or write a REVIEW-QUEUE.md proposal. Never ask
  the operator to widen permissions mid-loop.
- You can read/edit/write files, run pytest, flake8, and read-only git. You cannot
  `git commit`/`push` (the runner checkpoints), reach the network, or run arbitrary
  shell commands.
- An **independent reviewer in a fresh context** audits your working-tree diff
  against this contract before the runner commits. It sees only the diff and the
  contract — not your reasoning. Self-praise doesn't pass review; a small diff with
  pasted test evidence does.

## Known landmine — the `mode` disagreement
`.claude/state.json` (mode: dry_run) and `.claude/heartbeat.json` (mode: live)
currently disagree. **state.json is authoritative**; heartbeat is stale from a dead
process. If you touch code that *reads* mode, read it from state.json. You may NOT
"fix" the disagreement by editing either file — both are runtime state (Danger Zone).

## The loop contract (every iteration)
1. Read STATE.md — current reality, not assumptions.
2. **Red base first:** if STATE.md shows pytest or flake8 FAIL, fixing that IS this
   iteration's task. Never build new work on a red base.
3. Otherwise pick the single highest-value task in TASKS.md that is NOT in the
   Danger Zone. Done-criteria are immutable: build to the criterion, never edit it
   to fit what you built.
4. Implement it. Smallest change that fully does the task.
5. Run the verification gate (below). Fix what you broke. Never leave it red.
6. **Evidence over assertion:** end your reply with the actual final pytest and
   flake8 summary lines you observed. A green claim without pasted output is
   treated as false (`.claude/rules/honesty.md` binds you here too).
7. Stop after one coherent unit. The runner reviews, then checkpoints.

## DANGER ZONE — never edit. Propose in REVIEW-QUEUE.md, then pick safe work.
By PATH (the runner also hard-blocks any diff touching these):
- `src/scanner/**` — execution.py (ExecutionManager, execute_trade@2039,
  flatten_all@6194, live OANDA calls@3611-3636), config.py (oanda_environment@612),
  engine.py (set_halted@5456, drawdown guardian), backtest_gate.py (promotion gate),
  and all live signal/decision logic.
- `src/training/**` — model training + anything that produces or promotes an artifact,
  including backtest_harness.py core scoring, cost model, and fill assumptions.
- `src/risk/**` — all risk controls, sizing, drawdown, circuit breakers.
- `.claude/**` — state.json (the `halted` master kill + `mode`), heartbeat.json,
  alert_state.json, and any runtime state.
- `trained_data/**` — models, journals, backtests (mutable live state).
- `scripts/ralph.sh`, `scripts/run_full_training.sh`, and any promotion/staging script.
- `src/factor/ship_gate.py` — the pre-registered factor ship bar (PRD US-007). The rest
  of `src/factor/**` is SAFE; the gate is not. Gate changes need an operator-signed commit.
- `.env`, secrets, OANDA tokens, account IDs anywhere.

By BEHAVIOR (regardless of path) — you may NEVER:
- Set `oanda_environment: "live"` or introduce live OANDA credentials.
  (`.claude/**` is also a Danger Zone path — don't hand-edit runtime state regardless.)
- Weaken, bypass, loosen, or disable any risk limit, the 15% NAV drawdown guard, the
  52% accuracy threshold, the staleness/uncertainty block, or the kill switch.
  (Tightening still goes through the human.)
- Change what the ensemble decides, what gets sized, or what reaches the promotion gate.
- Wire a TUI control to fire/cancel an order, un-halt, or flip mode/environment.

## ★ PRIMARY MISSION — evolve the daily FX factor portfolio (`src/factor/**`)
Operator decision 2026-06-13: you have **full self-evolution** of the factor strategy
(carry + trend + value on daily FX). See `tasks/prd-fx-factor-portfolio.md` and
`docs/factor-portfolio-results.md`. Today's honest verdict is a **FAIL** (carry+trend
net Sharpe ≈ −0.2 on 7 USD-only majors). Work TASKS.md (FP-1…FP-5) toward a real edge.

Factor-specific invariants (violating any = automatic reject):
- **`src/factor/ship_gate.py` is OFF LIMITS** (Danger Zone). It is the pre-registered
  pass bar (PRD US-007). NEVER edit its thresholds or logic to make a book "pass." A
  FAIL verdict is a valid, honest outcome — report it. If you think the bar is wrong,
  propose an operator-signed `gate-change:` in REVIEW-QUEUE.md.
- **Causality is sacred:** no signal may use data after its own date. Preserve the
  window-invariance and lookahead-canary tests; a new signal ships with its causality test.
- After ANY `src/factor/` change, run `python scripts/run_factor_backtest.py` and paste
  the final **VERDICT:** line as evidence (alongside pytest/flake8).
- Do not delete/weaken a test or a done-criterion to move the number.

## SAFE to iterate autonomously
- `src/factor/**` EXCEPT `ship_gate.py` — signals, data loaders, portfolio construction,
  backtest. This is real financial logic: keep it causal, cost-aware, no-mock, deterministic.
- `scripts/run_factor_backtest.py` — the factor entrypoint.
- `src/tui/**` — DISPLAY only: rendering, layout, panels, charts, keybindings that are
  read-only views of P&L / positions / signals. (Control wiring → Danger Zone.)
- `tests/**` — add/strengthen coverage. **No-mock policy: do not introduce mocks.**
  Highest-value early task: golden/regression tests that pin current behavior of the
  Danger Zone modules *without editing them*.
- `docs/**`, `notebooks/**` — documentation, analysis writeups.
- Logging/formatting improvements **inside SAFE modules only** (logging changes inside
  a Danger Zone file still require a proposal).

## Verification gate (definition of "green") — all must pass every checkpoint
1. `python -m pytest tests/ -v --tb=short -x`
2. `python -m flake8 src/ --config=.flake8`
3. Backtest (`scripts/backtest_harness.py --instrument <pair>`) ONLY if a live
   `trained_data/models/*/transformer_direction.keras` artifact exists. The runtime is
   currently fail-closed (all transformers quarantined) — the runner skips it then.
4. Factor smoke: `python scripts/run_factor_backtest.py` must RUN to completion (exit 0)
   on the cached daily data. A FAIL **verdict** is fine — only a crash fails the gate.
5. NO working-tree change (tracked or untracked) touches a Danger Zone path.
Never skip, delete, or weaken a test, add a mock, relax an assertion, or edit a
TASKS.md done-criterion to go green. If you can't go green honestly, stop and explain
— a documented red stop beats a dishonest green.

## Stop conditions (halt loop, leave branch, write REVIEW-QUEUE.md)
- Any diff touches a Danger Zone path.
- The precondition is violated: `oanda_environment` is `"live"`, or live creds appear.
- The live/paper flag, a risk parameter, sizing, or the kill switch appears in a diff.
- A SAFE-module change would alter inputs/outputs of a Danger Zone module.
- Verify is still red after one repair attempt.

## When unsure
If you cannot prove a change leaves execution, sizing, risk, signal logic, the
promotion gate, and the halted/dry_run/practice boundary byte-for-byte unaffected,
treat it as Danger Zone. Propose, don't touch. With real capital, erring toward the
human is always correct.
