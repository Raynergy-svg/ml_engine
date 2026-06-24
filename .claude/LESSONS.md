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
| building a gate/checker that trusts its own script wasn't neutered (could be edited to always-pass) | **L-008** hash-pin every gate in a committed manifest; checkers cross-verify; fail closed on drift |
| a content/quality gate that checks STRUCTURE (fields, length) and is treated as verifying MEANING | **L-009** structural ≠ semantic — name the floor; human review backstops meaning |
| a "done"/PASS that needs no observable change, or a recorded metric trusted without recompute | **L-010** gate done on an artifact delta + recompute metrics from their real source (fail closed) |
| a verification/judge step that needs an agent or human dispatched & honest, used to gate "done" | **L-011** enforce the deterministic half live; close the lazy-skip with a bound verdict; document the lie floor |
| a security gate wired in worker-writable config (local settings / in-repo hook) — worker can disable it | **L-012** pin it in root-owned managed settings (highest precedence); self-contained wrapper; verify not-user-writable |
| relying on a managed hook as un-disablable without closing the global-disable + interpreter-shadow paths | **L-013** also pin disableAllHooks:false (managed) + absolute interpreters; verifier must require both |

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

## L-008 — a checker that can be silently neutered is not a gate   [ACTIVE]
- Trigger: building a gate/checker that decides PASS/done while trusting that the gate script itself
  wasn't modified — a verify/risk script that a worker could edit to `return 0` / `echo GREEN`.
- Root cause: a gate that doesn't verify its OWN integrity (and its peers') evaporates with a
  one-line edit, invisibly to the gate. Existence/size checks don't catch a neutered-but-nonempty
  script.
- Rule: hash-pin every enforcement script in a committed manifest (`gate_manifest.json`); have EACH
  checker verify the FULL manifest (cross-protection — neutering one doesn't disable the others) and
  fail closed on drift / unlisted / missing / unreadable-manifest. Cross-check the manifest covers the
  canonical script set so entry-drop fails closed too. The irreducible floor (editing a script AND
  the manifest together) must at least be git-visible; document it.
- Scope: every gate / tripwire script in the loop (verify_gate, loop_gate, risk_monitor, stop_gate, …).
- Source: 2026-06-23 red-team finding #2 + Increment 1 (gate_manifest + _integrity cross-check, 64
  tests); separate verifier PASS and it prescribed the entry-drop coverage fix.

## L-009 — a structural check is not a semantic one   [ACTIVE]
- Trigger: a content/quality gate that enforces STRUCTURE (required fields present, minimum length,
  uniqueness) and is then treated/described as if it verifies MEANING or quality.
- Root cause: a static check can confirm a lesson HAS the five fields and clears a length floor, but
  it cannot judge whether the lesson teaches anything — a 377-char lorem-ipsum block carrying the
  five field labels passes `audit_lessons` (the Increment-2 verifier proved this).
- Rule: structural gates legitimately close the "empty counter-bump" cheat (fail-closed on
  shallow/empty/dup), but do NOT claim a structural check verifies quality. Name the semantic floor
  explicitly and rely on human review (operator-gated `/evolve`) for meaning. Word comments/docs as
  "structure, not meaning".
- Scope: the lesson audit and any content/quality gate in the loop.
- Source: 2026-06-23 Increment 2 separate-verifier finding — crafted a structurally-valid but vacuous
  lesson that passed the audit; floor documented in `_integrity.py` + NOTES.

## L-010 — gate "done" on observable work, and recompute metrics from their source   [ACTIVE]
- Trigger: a "done"/PASS that requires no observable change (a gate that treats "nothing happened" as
  success), or a self-reported metric (test count, lesson count) trusted without re-deriving it.
- Root cause: laziness and falsification both produce a clean-looking record with no real work — a
  gate that neither requires an observable artifact delta nor recomputes its metrics from the real
  source can be satisfied by doing nothing or by typing a number into the state file.
- Rule: gate "done" on an OBSERVABLE artifact delta (a real test added, a question closed, a lesson
  learned) OR an explicit reviewable no-op attestation; and RECOMPUTE self-reported metrics from
  their real source at decide time (re-run the pinned suite, re-count the tracked list, re-audit
  LESSONS.md) rather than trusting the record — fail closed on mismatch.
- Scope: the loop's stopping conditions and any automation "done"/success signal.
- Source: 2026-06-23 red-team Fronts #1d/#3a/#5 + Increment 3 (tests_passed recompute via the pinned
  suite, anti-laziness work-delta gate; 73 no-mock tests); separate verifier.

## L-011 — you can't deterministically prove an agent was dispatched; enforce the half you can   [ACTIVE]
- Trigger: a verification/judge step that depends on a separate agent (or human) actually being
  dispatched and reasoning honestly — used as a gate for "done" (e.g. the separate-agent verifier).
- Root cause: no static gate, from inside the same environment the worker controls, can prove an LLM
  agent was dispatched or that it judged honestly. The verdict's gate value is a CLAIM. So the agent
  half of verification is intrinsically advisory; only its lazy-skip is closable.
- Rule: enforce the DETERMINISTIC half as the real floor (re-run disk-reading checks live, hash-pin
  the gate scripts + recompute source). For the AGENT half, close the LAZY dimension — require a
  fresh verdict artifact BOUND to the current state (so skipping the verifier blocks "done", and any
  post-verify change forces re-verification) — and DOCUMENT the irreducible lie-dimension (a
  fabricated PASS) as resting on human review. Never claim the agent dispatch itself is enforced.
- Scope: any verification / LLM-judge / human-review step the loop uses to gate behavior.
- Source: 2026-06-23 red-team Front #1a + Increment 4 (record_verdict.py + loop_gate fresh-verdict
  gate, 77 no-mock tests); irreducible floor documented in NOTES.

## L-012 — pin enforcement above the worker: root-owned managed settings   [ACTIVE]
- Trigger: a security gate whose enforcement lives in a file the worker can write — local
  `.claude/settings.json`, an in-repo hook registration — so the worker can simply disable or
  re-point it. The in-repo gates can make tampering git-visible but cannot PREVENT the disable.
- Root cause: a gate wired in worker-writable config is only advisory against a worker who edits that
  config; the root of trust is inside the worker-controlled environment.
- Rule: pin the enforcement (the Stop hook) in ENTERPRISE MANAGED SETTINGS installed to a root-owned
  system path OUTSIDE the project tree — managed settings are highest precedence and "can't be
  overridden by anything" (code.claude.com/docs/en/settings), and Claude Code's write access is
  confined to the project, so the worker can't touch them without admin. Make the hook a
  self-contained root-owned wrapper that re-derives the gate hashes ITSELF (so a neutered in-repo
  checker is caught) and is scoped to the repo. Ship an install script + a from-disk verifier that
  confirms the managed file is present AND not user-writable. Honest framing: this turns "worker can
  disable the gate" into "worker can't without admin, which the operator controls" — state it plainly,
  don't claim impossibility.
- Scope: any security/enforcement hook that must survive the worker editing local config.
- Source: 2026-06-23 managed-settings trust anchor — docs verified (paths incl. the corrected Windows
  `C:\Program Files\ClaudeCode\`, precedence, `allowManagedHooksOnly`, experimental `type:"agent"`);
  root-owned wrapper + installer + verifier; 90 no-mock tests; separate verifier red-teamed it.

## L-013 — a managed hook isn't un-disablable until you close global-disable + interpreter-shadow   [ACTIVE]
- Trigger: relying on a managed/system-level hook for enforcement and assuming the worker can't
  disable it — having only confirmed the worker can't edit the root-owned hook file directly.
- Root cause: a root-owned managed hook can still be neutralized by lower-scope mechanisms the
  managed file doesn't explicitly pin: (a) `disableAllHooks: true` set in LOCAL settings disables ALL
  hooks (incl. managed) unless managed pins it `false` (managed precedence then can't be overridden);
  (b) a PATH/env-shadowed interpreter — a hook command using bare `python3`/`bash` lets the worker
  substitute a no-op binary via local `env`/PATH.
- Rule: when pinning a managed enforcement hook, ALSO set `"disableAllHooks": false` at managed scope,
  invoke ABSOLUTE interpreters (`/usr/bin/python3`, `/bin/bash`) in the hook command and the hook
  script, and make the from-disk verifier REQUIRE both before reporting the anchor secure. Don't claim
  "can't be disabled" until the global-disable and interpreter-shadow paths are both closed.
- Scope: any managed/system-level enforcement hook.
- Source: 2026-06-24 adversarial re-check of the LIVE anchor — direct edit blocked (root-owned), but
  local `disableAllHooks` and interpreter-PATH-shadow were open; closed via managed `disableAllHooks:
  false` + absolute interpreters; verifier now requires them; 92 no-mock tests; docs re-verified.
