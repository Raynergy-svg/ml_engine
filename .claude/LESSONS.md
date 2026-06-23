# LESSONS — permanent, hard-won failure modes

Append-only. **Read before planning. Never delete** — if a lesson is wrong or outdated, mark it
`[SUPERSEDED by L-NNN, date]` and write the replacement; keep the original so we remember the trap.
Format is fixed and tight:

```
## L-NNN — <short name>   [ACTIVE | SUPERSEDED ...]
- Trigger:   the observable symptom that should make you stop and recall this
- Root cause: the actual mechanism, not the surface
- Rule:      the imperative — what to always/never do now
- Scope:     where it applies
- Source:    file / commit / incident that proves it
```

Add via `/evolve` (operator-approved). One lesson = one mechanism.

## Recall-trigger index (scan this at the ORIENT step of every plan)

A lesson you don't retrieve is dead weight. Before planning, scan this table — if your task touches
a trigger, open that lesson **before** you act. This is the memory-retrieval hook the loop depends on.

| If your task touches… | Recall |
|---|---|
| feature engineering, model accuracy, "this val acc looks good", window/lookback math | **L-001** anchored-feature artifact |
| training, promotion, shipping a model, train/val gap, quarantine, champion swap | **L-002** ship gate (10% gap) |
| OANDA env, going live, broker config, account URL, "deploy", real money | **L-003** practice forever |
| execution path, order submission, halt, a new trade entry point, auto-halt | **L-004** mid-cycle halt re-check |
| any status claim, "wired", "fixed", "verified", a subagent's diagnosis | **honesty.md** + `.claude/verifier.md` (verify from disk; separate agent) |
| "this rule/gate is in place", a context file that only *instructs* behavior | **L-005** advisory ≠ enforced — back it with a hook or disk-reading code, or it's compliance-gated |
| shipping a "hardened" gate, swapping one matcher for another (regex→AST), "this closes the evasion surface" | **L-006** a stronger matcher ≠ un-gameable — fail-closed + re-verify with the separate agent |
| a loop/automation decision driven by a self-reported number (open_questions, "facts done", "verified") | **L-007** derive signals from observable artifacts + re-check latest vs live (tamper→HALT) |

---

## L-001 — OBV / anchored-feature accuracy artifact   [ACTIVE]
- Trigger: a direction model reports "healthy" val accuracy (the old ~56–70% M15 numbers) that
  collapses to ~52% the moment feature computation is fixed; or any feature whose value depends on
  where the analysis window *starts*.
- Root cause: `obv`, `cum_returns`, `atr_log`, `vol_log` were computed anchored to the window start,
  so they leaked the candle's position-within-window into the features. The model learned the
  anchor, not the market. The inflated accuracy was the leak, not an edge.
- Rule: the feature pipeline MUST be window-invariant. `FEATURE_PIPELINE_VERSION` (v2) gates this;
  the canary `tests/test_feature_window_invariance.py` must pass; v1 artifacts refuse to load. Never
  re-introduce an anchored/cumulative-from-window-start feature without the invariance test.
- Scope: all direction-model training and inference.
- Source: commit dad8624 (2026-06-10 verdict); CLAUDE.md "Empirical ceiling"; pipeline v2 contract.

## L-002 — why every transformer failed the ship gate   [ACTIVE]
- Trigger: a freshly trained direction model looks shippable (decent val acc) but `train_acc -
  val_acc` exceeds 10%. Or: you're tempted to "just ship this one, it's close."
- Root cause: price-only M15/H1 has **no shippable directional edge** at this data scale (confirmed
  4+ independent ways). A >10% train/val gap is the model memorizing noise. Earlier high numbers
  were the L-001 anchor artifact. So the gate quarantines *everything*, correctly — there is
  nothing real to ship.
- Rule: `HARD_MAX_GAP = 0.10`. Any model with gap > 0.10 is moved to `_quarantine/`; per-pair gate
  routing cannot select from `_quarantine/`. **Nothing promotes to champion without passing the
  gate.** New architectures go shadow / champion-challenger — never a live swap, never while halted.
  Never raise `HARD_MAX_GAP` without explicit operator approval. The fail-closed empty-champion
  state (`direction=None` abstention) is correct behavior, not a bug to "fix" with a fallback.
- Scope: all model promotion / ship decisions.
- Source: `.claude/rules/improvement.md` "Hard Ship Gate — 10% Train/Val Gap" (2026-05-13 directive);
  commit 1a05e75; multiple verdict docs in `docs/`.

## L-003 — oanda_environment stays "practice", forever   [ACTIVE]
- Trigger: any thought, suggestion, config, or code path that would set `oanda_environment` to
  "live", point the broker at `api-fxtrade.oanda.com`, or otherwise touch real money.
- Root cause: there is no validated edge (L-002) and the system is halted by design. Going live
  would risk real money on a coin flip AND, worse, would create a false "it works" signal. The
  whole premise of this account is practice-only.
- Rule: `oanda_environment` must remain `"practice"` (`src/scanner/config.py:738`; broker factory
  `src/brokers/factory.py:68`; base URL defaults to `api-fxpractice.oanda.com`,
  `src/scanner/execution.py:801`). Never change it, never propose changing it, never scaffold a path
  that could flip it. The `/evolve` loop can NEVER relax this. Live is never an autonomous action.
- Scope: forever / environment / every workstream including the equity harvester.
- Source: operator Hard NO (2026-06-23); config default verified on disk 2026-06-23.

## L-004 — a halt set mid-cycle must still block the trade   [ACTIVE]
- Trigger: a trade fires (or could fire) while `state.halted=True` — especially when the halt is set
  *during* a scan cycle (auto_halt_loss_streak / AlertManager), after the cycle's start-of-loop
  halt check already passed.
- Root cause: the TUI runtime path (`embedded_scanner._execute_trades` → `Scanner.execute_trades` →
  `ExecutionManager.execute_trades` → `execute_trade`) skips `submit_trade`, the other halt-aware
  entry. The start-of-cycle early-return only catches halts set *before* the cycle. Trade 1306 opened
  while halted because of exactly this gap.
- Rule: keep the mid-cycle halt re-check at `src/scanner/execution.py:2081–2098`
  (`StateEngine().get_halted()` → return failed ExecutionResult). It is read-only and must never
  raise out of the guard. Any new execution entry point must perform the same re-check. "Halted
  means halted" applies at every layer, not just the loop top.
- Scope: every code path that can reach order submission.
- Source: trade 1306 (2026-05-12T20:39 UTC); guard comment in `execution.py:2081–2098`.

## L-005 — advisory context ≠ enforced   [ACTIVE]
- Trigger: "this rule/gate is in place" backed only by a CLAUDE.md/skill instruction; a context file
  that tells the model to do X but nothing fails if it doesn't.
- Root cause: auto-loaded instructions raise the PROBABILITY of correct behavior; they do not
  GUARANTEE it. A safety rule that depends on model compliance is a false floor.
- Rule: back any load-bearing rule with a deterministic gate — a hook (Stop/PreToolUse) or
  disk-reading code (verify_gate.py) — or a code guard. Reserve "advisory" for things that shouldn't
  fire on every trivial turn. Name which layer each rule is on.
- Scope: the whole context/loop system; any future "we added a rule for X" claim.
- Source: 2026-06-23 operator diagnosis ("self-engaging vs self-enforcing") + the Stop-hook /
  verify_gate build that closed it.

## L-006 — a stronger matcher is not an un-gameable one   [ACTIVE]
- Trigger: shipping a "hardened" gate — especially swapping one matcher for another (regex → AST) —
  and claiming "this closes the evasion surface."
- Root cause: any matcher that ENUMERATES cases leaks; the same evasion class reappears one layer up.
  The regex ship-gate missed `.15`; its AST replacement then missed `+= 0.10` and tuple-unpack
  targets (values > 0.10 passed as PASS). You cannot enumerate your way to safe.
- Rule: for load-bearing gates, FAIL-CLOSED on any form you can't reduce to a provably-safe value
  (never pass-by-omission), AND re-verify your own gate with the SEPARATE verifier agent — your
  hardening is itself a load-bearing claim, not a self-evident fact.
- Scope: every deterministic gate / tripwire in the loop (verify_gate.py, risk_monitor.sh, loop_gate.py).
- Source: 2026-06-23 round-2 verifier GATE FAIL — the AST AugAssign/tuple miss surfaced while
  "closing" the regex gap; round-3 PASS only after fail-closing the unreducible forms.

## L-007 — stopping signals must be derived from reality, not self-reported   [ACTIVE]
- Trigger: a loop/automation gate decides stop/continue from a free-form number a worker types into
  a state file (open_questions, "facts done", "verified") — anything self-reported gating the loop.
- Root cause: a self-reported signal can be set to whatever keeps the loop alive or declares "done";
  the gate is only as honest as the worker. A liar in state.json defeats the whole stopping logic.
- Rule: DERIVE the signals from observable artifacts — count a tracked list (questions.json), diff
  real test counts + verifier verdict, diff LESSONS.md lesson count — and RE-CHECK the latest
  recorded cycle against LIVE reality at decide time (a fresh verify_gate run + live list count);
  mismatch = tamper → HALT (fail-closed). Keep an absolute backstop. Document the residual limit
  honestly (a worker can still falsify the recorded test/lesson counts; human review backstops that).
- Scope: loop_gate.py / record_cycle.py and any future automation signal that gates behavior.
- Source: 2026-06-23 objective-stopping-inputs increment — 59 no-mock tests incl. anti-tamper
  (recorded PASS ≠ live FAIL → HALT) and the full record→decide pipeline; verified by separate agent.
