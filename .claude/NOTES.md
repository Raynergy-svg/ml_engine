# NOTES — live working memory (survives compaction)

> This is the **only** file I (Claude) may update without operator approval. It holds *state*, not
> doctrine. New decisions go to INTENT, new failure modes go to LESSONS, new patterns go to a skill
> — all via `/evolve`, with operator approval. Keep this file short and true; prune what's stale.

Last touched: 2026-06-23 by Claude (context-system build + self-improver loop hardening).

## Operating mode — delegated authority (2026-06-23, survives fresh sessions)

The operator delegated standing approval authority. From here on:
- **git commit + push are AUTO-APPROVED.** Commit in scoped, well-messaged commits; push when the
  work is verified green. No per-commit approval needed.
- **The orchestrator may approve `/evolve` proposals and similar on the operator's behalf** when they
  are (a) appropriate, (b) improve the bot, and (c) **backed by test/from-disk proof**. Evidence is
  mandatory — no proof, no approval.
- **IMMUTABLE ESCALATIONS — NEVER auto-approved by orchestrator or Claude; ALWAYS escalate to the
  human:** anything that relaxes a Hard NO, touches the per-trade hot path
  (`src/scanner/execution.py`, `scripts/`, `main.py`), changes `oanda_environment` off `"practice"`,
  un-halts trading, or moves real money. Surface these; route to the operator. The `/evolve` loop may
  never propose loosening a Hard NO.

## Schema (keep this file in these sections; prune anything stale every time you touch it)
1. Current runtime state  2. In-flight work  3. Blockers  4. Judgment calls (for veto)
5. Assumptions resolved  6. Active loop status (if mid-loop: cycle #, last verifier verdict, open
load-bearing question). Memory tightening rule: NOTES holds *state only*; the moment a note becomes a
durable decision/failure/pattern, move it to INTENT/LESSONS/skill via `/evolve` and delete it here.

---

## Current runtime state (verify against disk before acting — this is a snapshot)

Source: `.claude/state.json` read 2026-06-24.

- **`halted: false` — OPERATOR-DIRECTED ENABLE (2026-06-24).** The human operator (owns the halt)
  directed "enable bot" on the PRACTICE/demo account.
- **`mode: "live"`, `status: "running"`.** mode=live is EXECUTION mode (place orders), NOT real money.
  Orders go to the PRACTICE/paper account: the order client is `OandaPracticeClient`, hard-pinned to
  `PRACTICE_API_URL = api-fxpractice.oanda.com/v3` (`src/utils/oanda_practice.py:117`) and it IGNORES
  `oanda_environment` entirely — there is NO live-URL path in the order client. Verified by separate
  agent: "Can it place a real-money order? NO" (HIGH confidence). This is why mode=live is acceptable.
- `oanda_environment: "practice"` (`src/scanner/config.py:738`) — **immutable Hard NO, untouched.**
- Gates taught (committed): risk_monitor + verify_gate alarm on `mode=live` ONLY when env≠practice;
  **env=live / real-money / ship-gate stay HARD** (env=live+mode=live → double hard alarm). L-014.
- NAV $102,183 · `open_trades: 0` · zero live transformer artifacts (all quarantined) → bot abstains
  (no champion direction model) so enabling unleashes no flood of trades; ship gate intact.
- Known residual (non-blocking, pre-existing): static env-tripwire matches `oanda_environment = "live"`
  assignment + git-diff `api-fxtrade`; a future *dict-form* profile override `"oanda_environment":
  "live"` wouldn't be caught by the tripwire — but the practice-pinned order client is the primary
  rail. Add the dict pattern IF env is ever wired into a profile dict / the client honors env.
- Branch: `ralph/equity-harvester-bot` (an equity-beta harvester workstream is in flight)

## In-flight work (from session memory, not re-verified this turn — confirm before relying)

- **Equity-harvester-bot**: 22-story PRD; TUI wired to equity (commits 3a58c6c, eb04687). An
  independent code review flagged **4 CRITICAL + 7 HIGH execution defects** (C1 crash→double-submit,
  C2 fill-detection reads aggregate position as per-order fill, C3 books PENDING as FILLED, C4 can't
  detect a MISSING corp-action split). These are the real blocker to paper/shadow trading. `[unverified — from memory recent.md, re-grep before acting]`
- **FX direction lever**: empirically closed at ~52% intraday / ~50–54% daily across every tested
  approach (price-only, news fusion, factor, carry, meta-labeling). Stay halted = the correct
  outcome, not a failure. See verdict docs in `docs/`.

## Blockers

- None for *this* task (context-system build). For trading: system is halted by design; unhalting
  is blocked on unambiguous validation that does not currently exist.

## Judgment calls I made on this build (veto any of these and I'll revise)

1. **CLAUDE.md reconciled, not overwritten.** A rich 14 KB CLAUDE.md already existed. I *prepended*
   a lean "Self-Evolving Context System" block at the top (one-liner, Tier 6/7, Hard NOs w/
   citations, pointers, working rules) and **left all existing FX doctrine below it intact.** Net
   effect: leaner *entry point*, nothing lost. If you'd rather I trim the old body, say so.
2. **`oanda_environment` line is 738, not ~612.** Your brief said ~612; line 612 is the docstring.
   The actual field default is `src/scanner/config.py:738`. I cited 738 everywhere. Reality won.
3. **Session-spawns-agents instruction** ("add so session spawns agents to prompt main AI after
   engineering it") — interpreted as: a SessionStart hook that, on every session, injects a boot
   prompt telling the main AI to read INTENT/NOTES/LESSONS first, surfaces the halted+practice
   state, and recommends dispatching domain specialist sub-agents per the working rules. Implemented
   as `.claude/tools/session_context_boot.sh`, registered as a **second, additive** SessionStart
   hook (the existing tmux hook is untouched). Hooks emit context — they can't literally spawn
   Claude agents — so "spawn agents" is realized by *instructing the main AI to dispatch them*. If
   you meant something more literal (e.g. background `Agent`/cron jobs), tell me and I'll rebuild.
4. **`.claude/commands/evolve.md`** created (the `commands/` dir didn't exist). Invoke with `/evolve`.
5. **Self-improver loop added (2026-06-23 steering).** Per operator steering, folded in: `LOOP.md`
   (cycle + 5 objective stopping conditions incl. anti-stall STOP-CHURN), `verifier.md` + `/verify-task`
   (independent verifier = separate Code Reviewer agent/model, re-derives claims from disk),
   `tools/risk_monitor.sh` (parallel fail-closed safety tripwire — runs GREEN against live repo, exit 0),
   LESSONS recall-trigger index, NOTES schema+pruning rule, DoD item 6 (verify-task PASS + monitor GREEN).
   **Judgment call:** I authored these coherently myself (they cross-reference heavily), then dispatched
   a *separate* Code Reviewer agent as the independent verifier of the whole system — that build→verify
   loop IS the "let an agent create a loop... until proper and complete" instruction, instantiated. The
   loop is a META/dev loop (like Ralph), deliberately NOT runtime code — Claude stays out of the hot path.
   If you wanted the loop wired as a live background daemon instead, say so and I'll build that variant.
6. **Enforcement layer added (2026-06-23 mandate).** Converted the inert/advisory pieces into
   deterministic gates: a `Stop` hook (`stop_gate.sh`) enforcing the risk monitor every turn-end;
   `verify_gate.py` (deterministic half of the verifier — reads disk, immune to a lying agent);
   `loop_gate.py` (stopping conditions from disk); 32 no-mock tests. **Judgment calls for veto:**
   (a) the Stop hook runs at *every* turn-end and **blocks once** on ALARM — I chose blocking (your
   fail-closed ethos) with a loop-guard so it can't trap; if turn-end checks feel heavy, I can scope
   it to risky tool calls instead. (b) The whole layer is **untracked in git** — a `git clean` would
   wipe it and leave `settings.json` pointing at a missing hook. I did NOT commit (your call); the
   verifier flagged this as the #1 residual risk. Say "commit it" and I will. (c) `L-005` body is
   proposed via `/evolve` below and awaits your approval — the LESSONS trigger-row referencing it is
   a forward-ref until you approve.

## Active loop status (2026-06-23 — standing-roadmap cycle, converged)

- `.claude/loop/state.json` has 6 cycles; `loop_gate.py` reads disk → **STOP-DONE** (risk GREEN,
  last cycle no new info, 0 open questions). Took **3 independent verify rounds** to converge.
- **Roadmap hardening landed (all 4 fronts):** verifier ship-gate + halt-guard are now **AST-verified**
  (immune to `.15`/`1e-1`/`+=`/tuple/walrus/expr/non-literal evasion, and dead/commented/inverted
  guards); `lessons_have_triggers` integrity makes memory provably fire at planning time;
  `risk_monitor.sh` widened to the 4th Hard NO (no live env assignment). **47 no-mock tests, all green.**
- **Independent verifier earned its keep:** round 2 returned **GATE FAIL** and found a real gap I
  introduced (AST missed `HARD_MAX_GAP += 0.10` and tuple-unpack → >0.10 passed as PASS) + a latent
  inverted-else halt guard. Fixed both + preempted walrus; round 3 → **GATE PASS, HIGH confidence, no
  residual gap** after the agent invented its own evasions and all were caught. Self-grade would have
  shipped the gap.
- Real-repo `verify_gate.py` PASS (20 checks); hot path (`execution.py`/`scripts/`/`main.py`) untouched.
- Prior-cycle status (enforcement layer build) is folded into LESSONS L-005 + the cycle history.

## Roadmap progress (north star — `INTENT.md` standing roadmap)

- **Committed + pushed:** enforcement layer at `1062064` → `origin/ralph/equity-harvester-bot`
  (settings.json is gitignored/machine-local → `.claude/loop/INSTALL.md` documents the wiring).
- **Verifier front:** AST-verified, fail-closed, 3 independent rounds → PASS. L-006 captures the lesson.
- **Stopping-conditions front:** STOP-CHURN now un-gameable to the self-report limit — rolling-window
  (progress = close a question OR learn a lesson) + 6-cycle absolute backstop. A separate verifier
  brute-forced 200k sequences and found no forever-CONTINUE stall. **51 no-mock tests, all green.**
- **Objective stopping inputs — DONE (2026-06-23).** Stopping signals now derive from observable
  reality, not self-report: `open_questions` counted from tracked `.claude/loop/questions.json`;
  `new_verified_facts`/`new_lessons` are per-cycle deltas of real test count + verify verdict +
  LESSONS.md count, recorded by `.claude/loop/record_cycle.py` (runs the suite); loop_gate re-checks
  the latest cycle's verdict + open-count against LIVE reality → tamper = HALT. **59 no-mock tests.**
  Separate verifier acted as a malicious worker (faked verdict/open-count, wiped history, abused
  blocked flag, malformed JSON) — every attempt failed closed; **no false STOP-DONE possible.**
- **Remaining honest limit (documented in loop_gate.py + L-007):** a worker could still falsify the
  recorded `tests_passed`/`lessons_count` integers themselves — but the live verify_gate re-run is the
  true integrity floor, and human review backstops the counts. This is the irreducible self-report
  edge; closing it further would need signing/attestation of record_cycle output (low priority).
- Operating in delegated-authority mode (commit/push + /evolve auto-approved with proof; immutable
  escalations route to human). See the "Operating mode" section above.

## Irreducible floors — the 2026-06-23 red-team program's residual trust assumptions

A red-team (separate agent, lying + lazy) attacked all four fronts; 4 hardening increments closed
every CLOSABLE cheat with deterministic, fail-closed, tested gates (commits 88b8b83 / c0d8306 /
d5992a8 / + Increment 4). What remains is genuinely irreducible — documented here per the operator's
"don't fake closure" mandate. Each names the residual trust assumption and what closing it would cost.

1. **Gate-script integrity (L-008).** All enforcement scripts + the test suite are SHA-256 hash-pinned
   (`gate_manifest.json`), cross-checked by two independent checkers, coverage-checked vs entry-drop.
   FLOOR: editing a gate script AND regenerating the manifest in one change evades hashing — but the
   diff is git-visible, and a root checker can't hash itself. Closing cost: sign the manifest with a
   key the worker can't access (CI-side), checked out-of-band.
2. **Lesson meaning (L-009).** Lessons are structurally audited (5 fields, length, uniqueness);
   empty/shallow/dup fail closed. FLOOR: a structurally-valid but vacuous lesson passes — a static
   check can't judge meaning. Closing cost: an LLM-judge (itself subject to #5) or human review (current).
3. **Recorded metrics (L-010).** open_questions, verify verdict, lessons_count, and tests_passed are
   ALL re-derived live at decide time and tamper-checked → a hand-edited count HALTs. FLOOR: none for
   these — re-derived from source (tests_passed's source, the suite, is hash-pinned per #1).
4. **Observable work (L-010).** STOP-DONE requires an observable artifact delta OR an explicit
   no_work_needed_attested. FLOOR: that attestation is a human-reviewed self-claim (can't prove intent
   from inside). Closing cost: same as #5.
5. **Separate-agent verification (L-011).** STOP-DONE requires a fresh PASS agent-verdict BOUND to the
   current state — skipping the verifier blocks done; any post-verify change forces re-verify. FLOOR:
   a worker can write `record_verdict.py --gate PASS` WITHOUT dispatching a real agent (the lie). No
   static gate inside the worker-controlled environment can prove an LLM was dispatched or judged
   honestly. The DETERMINISTIC half (verify_gate: live + hash-pinned + cross-protected) is the enforced
   verification floor. Closing cost: an out-of-band trust anchor the worker can't forge — signed
   attestation from a trusted execution environment that actually dispatched the verifier, or
   append-only out-of-band dispatch logging the gate checks instead of a self-written JSON.

Bottom line: every machine-checkable claim and every Hard NO is enforced + fail-closed. What's left
reduces to "a worker with full repo access can lie in a git-visible way (backstopped by human review)
or fabricate an agent dispatch." Closing those requires moving the root of trust OUTSIDE the
worker-controlled environment (CI signing / TEE / out-of-band logging) — operator's call on cost.

### Managed-settings trust anchor (2026-06-23, docs-verified) — moves the root of trust outside the repo
Prepared (operator runs one privileged install step — `.claude/loop/managed/INSTALL.md`): enterprise
managed-settings.json pins the Stop hook to a ROOT-OWNED self-contained wrapper at the system managed
dir (macOS `/Library/Application Support/ClaudeCode/`, Linux `/etc/claude-code/`, Windows
`C:\Program Files\ClaudeCode\` — NOT the deprecated ProgramData). Managed settings are highest
precedence ("can't be overridden"); Claude Code write access is confined to the project tree. The
wrapper re-derives the gate hashes ITSELF (catches a neutered in-repo checker). 90 no-mock tests;
separate verifier red-teamed it (PASS, no new gap; fetched the docs live). Effect on the floors:
- **Floor #1 (gate-tamper) — DISABLE + NEUTER dimensions: CLOSED to "requires admin."** A worker can't
  remove/re-point the Stop hook (managed precedence) nor neuter a checker (wrapper self-hashes).
- **Floor #5 (hook-neuter): CLOSED to "requires admin."** The tripwire can't be disabled by the worker.
- STILL OPEN (documented): edit-script-AND-manifest-together (git-visible; close by baking the manifest
  hash into the root-owned wrapper, re-install per gate change); the verification LIE dimension (L-011,
  advisory agent half); and the cwd/harness-integrity assumption (same as the in-repo stop_gate).
The honest win: this turns "worker can disable the gate" into "worker can't without admin, which the
operator controls" — until the one privileged install step runs, the anchor is advisory and the
in-repo gates + code-level guards remain primary.

**INSTALLED 2026-06-24, then ADVERSARIALLY RE-CHECKED — found 2 bypasses, hardened (RE-INSTALL needed).**
Operator ran the install. Verified from disk: the two files are `root:wheel`, NOT writable by the
worker — a worker CANNOT directly edit/remove/re-point the managed hook (proven: every write returned
Permission denied). BUT the adversarial re-check (operator-requested) found two ways a worker could
still neutralize it via LOCAL settings (which the worker CAN write):
  (1) `disableAllHooks: true` in local settings disables ALL hooks incl. managed — unless managed pins
      `disableAllHooks: false` (precedence). NOW fixed in the in-repo managed-settings.json.
  (2) interpreter PATH-shadow of bare `python3`/`bash`. NOW fixed: absolute `/usr/bin/python3` + `/bin/bash`.
`verify_managed_anchor.py` now REQUIRES `disableAllHooks:false` and currently reports **NOT-ACTIVE,
exit 1** against the installed OLD file — honest: **the operator must RE-INSTALL** the hardened file
for the closure to take effect. **I correct my prior "CLOSED in effect" claim:** until re-install, the
direct-edit vector is closed but the disableAllHooks/PATH vectors are open. Separate verifier red-teamed
the hardened anchor → PASS, no new bypass. Runtime "managed false wins" is docs-precedence-confirmed
(verbatim "can't be overridden"), not empirically reproduced (needs a live session). LOW residual noted:
risk_monitor.sh's own unqualified git/grep (not a current bypass; documented in INSTALL.md).

- 3 cycles recorded in `.claude/loop/state.json`. `loop_gate.py` computes **STOP-DONE** from disk
  (risk GREEN, last cycle no new info, 0 open questions). Build→verify→fix ran twice.
- **What is now ENFORCED (deterministic, not compliance):** Stop hook `stop_gate.sh` runs
  `risk_monitor.sh` at every turn-end, fail-closed, blocks on ALARM, loop-guarded (proven: real-repo
  exit 0; synthetic live-env → exit 2 HALT-SAFETY; `stop_hook_active` → exit 0 no-trap).
  `verify_gate.py` re-derives the Hard NOs from disk (19 checks, immune to narrative), now also asserts
  the Stop hook is REGISTERED and catches non-default live assignments. `loop_gate.py` computes the
  stopping decision. All covered by `.claude/loop/tests/test_loop_enforcement.py` — **32/32 no-mock
  tests pass** (`python3 .claude/loop/tests/test_loop_enforcement.py`).
- **Independent verifier (separate Code Reviewer agent): GATE PASS, 8/8 claims, zero Hard-NO.** It
  found 3 real hardening gaps; I fixed the 2 high-value ones (Stop-hook-registration check + stronger
  live-flip detection) and documented the 3rd (regex evasion surface). Hot path untouched
  (`execution.py`/`scripts/`/`main.py` unmodified — verified via `git status`).
- Earlier context-system cycle verdict (PASS, risk_monitor scope fix) is folded into L-001..L-004.
- Watch-item: equity-harvester C1–C4 defects below are memory-sourced `[unverified]` — re-grep first.

## Assumptions resolved on my own this turn

- Tier 6 = meta-learning ensemble (MetaLearner + Bayesian adapter + ensemble weighter, shadow);
  Tier 7 = autonomous control loop (incident→propose→gate→soak→promote→close), Claude never in the
  hot path. Source: CLAUDE.md + `docs/tier7-architecture.md`.
