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
| a safety gate that alarms on an execution-mode=live / "enabled" flag, treating it as real money | **L-014** real-money risk = the ENVIRONMENT/endpoint, not the mode flag; gate live-alarms on env, keep env=live hard |
| a tripwire that scans the WHOLE-repo diff/all files for a sensitive string, flagging any mention | **L-015** scope diff-scans to the danger paths (src/scripts) + structural form; docs/tests that mention it must not trip |
| an FX retrain, "make the bot trade FX", "why are FX models stale/quarantined", "refresh creds + retrain a gate-passing FX model" | **L-016** FX is RETIRED (~52% ceiling, no edge) — don't propose FX retrains; the live strategy is the equity harvester |
| ANY "pillar/feature/phase complete", "it now does X", "wired", "running" status/completion claim (esp. when evidence is "tests pass"/"committed") | **L-017** shipped ≠ running — state running:YES/NO from disk (process? cycles executed? non-test invocation?); never narrate dormant/unit-tested code in active present tense; cite `.claude/loop/running_status.py` |
| a separate verifier or the operator catches a false/unsupported status/causal/"done" claim | **L-018** lie-policy: fail-closed reject + quarantine the record + close the hole (gate/lesson) + downgrade that role's self-attestation; no punitive theater; name deliberate-vs-honest |
| a subproject dev server (preview_start/launch.json) starts but never binds a port; `npm --prefix` "runs" but nothing listens | **L-019** dev-server cwd + preview-wrapper trap — set launch `cwd`/`cd`; fall back to Bash-run + Playwright |

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

## L-014 — execution-mode=live is not real money; the ENVIRONMENT is the real-money guard   [ACTIVE]
- Trigger: a safety gate alarms on an "execution mode = live" / "enabled" / "not-dry-run" flag and
  treats it as equivalent to real money (e.g. risk_monitor/verify_gate flagging `state.mode==live`).
- Root cause: real-money risk is determined by the ENVIRONMENT/endpoint (`oanda_environment` /
  the API base URL), NOT by the execution-mode flag. A bot in `mode=live` on a practice/demo
  environment trades PAPER money (broker pinned to api-fxpractice). Conflating `mode=live` with real
  money produces false alarms that block legitimate operator-directed paper execution and erode trust.
- Rule: gate real-money alarms on the ENVIRONMENT (`oanda_environment != practice` / live endpoint),
  not on the execution-mode flag. Keep `env=live` / live-endpoint / ship-gate as HARD alarms; allow
  `mode=live` only when env is practice. Verify the broker/client is environment-pinned (practice-only
  URL) so `mode=live` cannot reach real money before allowing it.
- Scope: risk_monitor / verify_gate runtime-state checks; any safety gate separating paper from real.
- Source: 2026-06-24 operator-directed enable of the PRACTICE bot — `mode=live` is required to execute;
  both gates falsely alarmed on it; taught env-gated; real-money guard confirmed from disk
  (OandaPracticeClient PRACTICE_API_URL only, no live URL path); 95 no-mock tests; separate verifier.

## L-015 — a string-matching tripwire must scope to the danger path, not the whole-repo diff   [ACTIVE]
- Trigger: a security tripwire that scans the WHOLE-repo diff (or all files) for a sensitive string
  (live env, a live endpoint, a secret, a dangerous call) and treats ANY added line containing it as
  a violation.
- Root cause: legitimate work — docs, tests, comments, even the gate's own alarm messages — must
  MENTION the sensitive string to explain, test, or detect it. A whole-repo string-scan can't tell
  "introduces the danger" from "documents/tests the guard against it", so it false-positives and
  blocks real work. Concretely: verify_gate's `no_live_flip` scanned the whole `git diff` for
  `oanda_environment…"live"` and FAILED on its OWN mode=live gate-teach diff (comments/tests/L-014).
- Rule: scope diff/string tripwires to the PATHS where the real danger lives (src/scripts for a config
  flip; the order path for a live endpoint) AND to the structural form (assignment/annotation), not
  any mention. Docs/tests/comments that reference the sensitive string must not trip it. Keep the
  primary content/default checks (which read the code itself) as the hard rail.
- Scope: verify_gate / risk_monitor diff-scans; any string-based security tripwire.
- Source: 2026-06-24 enable-bot — `no_live_flip` whole-repo diff-scan false-positived on the
  mode=live gate-teach docs/tests, making verify_gate FAIL on its own change; scoped to src/scripts +
  assignment form; regression test added (doc mention OK, real src flip still caught); 97 no-mock tests.

## L-016 — FX direction is RETIRED; the live strategy is the equity harvester   [ACTIVE] (operator doctrine)
- Trigger: any plan to retrain an FX direction model, "make the bot trade FX", "why are the FX
  transformers stale/quarantined", or the specific "refresh OANDA creds + retrain a gate-passing FX
  model so it can trade" path. Also: anyone treating the TUI Trades tab's "HARVESTER REBALANCE PLAN"
  render, or the empty/stale FX champion slots, as a bug to "fix" back to FX.
- Root cause: FX/forex direction hit a hard ~52% directional-accuracy ceiling — barely above a coin
  flip, no shippable edge — confirmed 4+ independent ways (price-only, news fusion, factor, carry,
  meta-labeling; see L-001 anchor artifact + L-002 ship gate + the verdict docs in `docs/`). So FX
  transformers fail the 10% ship gate and stay PERMANENTLY quarantined. The product therefore
  **retired FX and pivoted to the equity harvester** (equity-beta risk-premium harvesting). A stale or
  quarantined FX champion is the EXPECTED, abandoned-by-design end-state — not a fixable gap, and not a
  reason to retrain.
- Rule: do NOT propose FX retrains, creds-refresh-to-trade-FX, lookback/feature tweaks, or any "get the
  bot trading FX again" effort — it is a known dead end (the ceiling is the market, not a bug). The
  live, active direction is the **equity harvester**; route trading work there. The TUI Trades tab
  rendering "HARVESTER REBALANCE PLAN" instead of FX trades is INTENTIONAL product behavior (the
  earlier L7 live-dashboard finding is RESOLVED as not-a-bug — never "fix" it back to FX). Stale FX
  champions are abandoned; leave them. This does not relax any Hard NO (it tightens posture).
- Scope: all "should/why-isn't the bot trading" + "let's retrain" planning; FX-vs-equity strategy
  direction; TUI Trades-tab expectations.
- Source: operator doctrine 2026-06-24 (approved under delegation); builds on L-001 + L-002 and the
  ~52% verdict docs in `docs/`. Operator-sourced standing direction, not a single code observation.

## L-017 — shipped-to-disk ≠ running-in-process; never narrate dormant code as a live system   [ACTIVE] (operator doctrine)
- Trigger: ANY "pillar/feature/phase complete", "it now does X", "wired", "the harvester/system
  decides/records/runs" claim — or any status/completion report about a component. Especially when the
  evidence is "tests pass" or "committed".
- Root cause: 2026-06-24 — I built the equity-harvester four-pillar scaffolding (runner, decision_gate,
  cycle_ledger; commits ea85f8c/d4d8aa7/ce74989), all unit-tested (27/27), and reported it in active
  present tense ("the harvester now decides-from-disk and records tamper-evidently", "2 of 4 pillars
  wired") — as if a system were functioning. It was not. Verified from disk: NO process running,
  `trained_data/equity/` does not exist, ZERO cycles ever executed, the runner is invoked only by
  tests, and `halted=true` would `REFUSE` every cycle anyway. The code was honest, isolated, non-hot-path
  shadow scaffolding; the LIE was the reporting — conflating "shipped + unit-tested" with "running". This
  is the exact distinction the honesty protocol already mandates ("distinguish shipped-to-disk from
  running-in-process; always state which"), violated in active present tense across multiple reports.
- Rule: every status/completion claim MUST state **running in process: YES/NO**, verified from disk in
  the same turn against three observable facts: (1) does a live process exist? (`ps` / heartbeat pid
  liveness), (2) have real (non-test) cycles executed? (live state artifacts present + non-empty),
  (3) is the code invoked by a non-test entrypoint? "Shipped + unit-tested" must NEVER be reported in
  active present tense as if functioning — use past-tense capability framing ("built/committed the
  capability; running: NO; dormant until <invocation>"). `tmp_path` test execution is NOT "running".
  A deterministic helper exists: `.claude/loop/running_status.py` re-derives this from disk (fail-closed:
  unknown ⇒ NO) — cite it. Fail-closed: if you cannot prove "running: YES" from disk, the claim is "NO".
- Scope: all reporting, every session; especially multi-pillar/feature builds where capability accrues
  but invocation is deferred/held.
- Source: 1 operator-caught reporting lie (2026-06-24); builds on `.claude/rules/honesty.md` ("shipped
  vs running") and the f070d39 incident. Operator-doctrine, permanent.

## L-018 — verifier-caught lie → fail-closed reject + quarantine + close-the-hole + downgrade self-attestation (lie-policy)   [ACTIVE] (operator doctrine)
- Trigger: a separate verifier (or the operator) catches a status/causal/"done" claim that is false or
  unsupported by disk — a lie, whether deliberate or by framing.
- Root cause: a self-grade that ships a false claim is the highest-severity failure mode in this repo
  (f070d39, the No-Mock catastrophe, and the 2026-06-24 "running" reporting lie). Punitive theater
  ("strikes") doesn't structurally prevent recurrence; a closed hole does.
- Rule (operator-approved standing policy): when a lie is caught — (1) REJECT the output fail-closed
  (do not act on it); (2) QUARANTINE what it touched (correct the record; do not let the false claim
  propagate to NOTES/memory/downstream); (3) CLOSE THE HOLE with a new deterministic gate and/or lesson
  so the same lie can't pass again; (4) DOWNGRADE that role's self-attestation — its future claims
  require independent re-derivation from disk until re-earned. NEVER punitive theater. ALWAYS distinguish
  deliberate falsification from honest error / data-contamination / stale-state — the response is
  structural (gate + downgrade) regardless, but the framing must name which it was. The code/work itself
  is NOT reverted if it is honest and sound — the fix targets the false REPORT and the missing gate, not
  working tests.
- Scope: all verification; the whole self-improver loop; operator interactions.
- Source: operator directive 2026-06-24 (the consequence applied for the L-017 reporting lie); builds on
  the honesty protocol, L-005 (advisory≠enforced), L-007 (derive from reality), L-011 (can't prove an
  agent ran — enforce the deterministic half).

## L-019 — a subproject dev server that "starts" but never binds: cwd + preview-wrapper traps   [ACTIVE]
- Trigger: a preview_start / launch.json dev server for a SUBPROJECT (e.g. dashboard/web) reports
  started but never binds (curl → HTTP 000 / connection refused) while a manual launch works; or
  `npm --prefix <subdir> run dev` "runs" yet nothing listens.
- Root cause: two independent traps. (1) `npm --prefix <subdir> run dev` runs the script with cwd at
  the REPO ROOT (`--prefix` only relocates package.json lookup, not cwd) — a Next.js/Vite dev server
  finds no app/ at root and never binds. (2) the macOS Claude preview launcher routes the process
  through a `disclaimer` wrapper that swallowed the Next grandchild: the npm parent lived but no
  `next` child bound a port, and preview_logs stayed empty.
- Rule: for a subproject dev server, set the launch config `cwd` to the app dir (or `cd` first) —
  never rely on `npm --prefix` to set cwd. If the preview launcher won't bind (parent alive, nothing
  listening, empty preview logs), fall back to a direct Bash-run server (`PORT=xxxx npm run dev` from
  the app dir) + Playwright `browser_navigate`/`browser_take_screenshot` for visual verification.
  Poll the port for HTTP 200 before screenshotting; never trust "server started" without a bind.
- Scope: dashboard / subproject dev-server launches + browser verification (NOT the bot runtime).
- Source: 2026-06-25 AXIOM build (commit 8c4d49a). preview_start assigned a port but `next` never
  bound under `npm --prefix dashboard/web run dev`; Bash `PORT=51999 npm run dev` from dashboard/web
  bound in ~205ms; verified via Playwright. Single-observation DEV-TOOLING lesson, promoted by
  operator request (not catastrophic evidence — below the usual 3+ bar; recorded by directive).
