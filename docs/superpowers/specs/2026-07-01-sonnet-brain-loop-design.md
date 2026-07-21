# Sonnet Brain Loop — Design

Date: 2026-07-01
Status: approved (operator directive; judgment calls flagged inline, silence = proceed per
INTENT.md rule 6 "reversible → act")

## 1. Problem

Today's session ran the LLM-as-signal-generator experiment (Track B, "agentic research-alpha",
commit `8aa9417`) end to end and got **NO EDGE** — the LLM scoring 10-K filings for a directional
tilt failed both pre-registered controls (post-cutoff sign-flip, placebo near-real). This is
consistent with the whole campaign (L-020/021/022): there is no accessible free-data return-alpha,
and an LLM reading text does not create one out of thin air.

That closes the door on "LLM picks the trade." It does not close the door on "LLM manages the
*process* that finds, tests, and retires trading strategies." The operator's ask: build that
process — the **Sonnet brain loop** — as an orchestration layer around the execution/validation
machinery that already exists in this repo, with trade placement and dangerous control actions
kept exactly where they are today: deterministic code + human checkpoint.

## 2. Non-goals (explicit, per operator's hard constraints)

- The brain does not place trades. No import of `src/scanner/execution`, no broker client
  instantiation, no OANDA credential access from `src/brain_loop/`.
- The brain does not pick directional signals. It orchestrates the *validation* of strategies
  proposed as hypotheses; the strategies themselves are the existing, gated approaches (equity
  harvester, trend sleeve) or future ones vetted through the same harness — never an LLM-generated
  price/direction call substituting for the model ensemble.
- The brain never arms itself and never bypasses the human checkpoint for `flatten`,
  `set_gross_leverage`, `start_loop`, or `unhalt` (verified: `dashboard/server/control_safety.py:44`,
  `ARMED_REQUIRED_ACTIONS = frozenset({"flatten", "set_gross_leverage", "start_loop", "unhalt"})`).
  Promotion of a NEW strategy to live still requires the operator's ARM via the existing dashboard
  flow.
- This task does not unhalt, does not start the loop live, and does not wire a running
  cron/scheduler against production state. It leaves the machinery ready to run.

## 3. What already exists (verified from disk 2026-07-01) — reuse, don't rebuild

| Piece | File:line | Role |
|---|---|---|
| ARM lockdown + actor attribution | `dashboard/server/control_safety.py:44,188,232,297`; `dashboard/server/control.py:69,297-362` | `is_armed()` re-read from disk every call, fail-closed to DISARMED, 900s TTL. `enforce(action, params)` blocks `ARMED_REQUIRED_ACTIONS` unless armed. `audit(entry, *, actor)` — actor is a required kwarg, appended to `trained_data/axiom/control_audit.jsonl`. |
| Halt (autonomous-safe) vs unhalt (human/ARM-gated) | `src/scanner/automation/state_engine.py:206-213`; autonomous `set_halted(True)` callers at `src/scanner/execution.py:6307`, `src/scanner/engine.py:5626`; only `set_halted(False)` callers are `src/tui/app.py:2724` (human TUI) and `dashboard/server/control.py:239` (ARM+`assert_unhalt_eligible()`-gated dashboard endpoint, `control_safety.py:97`) | Halting is already precedented as an autonomous circuit-breaker action. Unhalting is not — it is exactly one human path plus one ARM-gated path. |
| Risk gate 6-rule system | `src/equity/trend_risk_gates.py` (see function index below) | Pure functions: `risk_normalized_units`, `leverage_cap_scale`, `clamp_risk_pct`, `bucket_cap_gate`, `bias_detector`, `evaluate_winner_stop`, `pyramid_gate`, `one_position_gate`. All only *reduce* risk; composition order documented at `trend_risk_gates.py:16-23`. |
| Ship gate | `trained_data/backtests/SHIP_GATE.json` (schema: `gate_pass`, `net_sharpe`, `max_dd`, `universe_hash`) | Canonical promotion criterion artifact. |
| Decision gate (halt/DD/ship-gate/freshness precedence) | `src/equity/decision_gate.py:154-195`, `decide_cycle()` → `REFUSE > HALT > NO_ACT > ABSTAIN > CONTINUE` | Fail-closed on unreadable state; only `CONTINUE` permits planning. |
| Cycle ledger (tamper-evident) | `src/equity/cycle_ledger.py` | Hash-chained JSONL pattern — each record commits to the previous record's hash. |
| Tier 7 self-heal/monitor | `scripts/run_tier7_loop.py:58-141` | Bounded supervisor; autonomy L3 default / L5 raised-by-operator; zero unhalt/execution/env-flip/promotion capability (verified no such imports). Heartbeat → `.claude/tier7_state.json`. |
| `/evolve` + LESSONS.md | `.claude/commands/evolve.md`, `.claude/LESSONS.md` | Operator-approved-only doctrine writer; brain loop proposes, never silently writes. |

## 4. Architecture

**"Sonnet is the brain" = a scheduled/manual Claude Code session, not an API call embedded in
runtime code.** Consistent with CLAUDE.md ("Claude is never in the hot path — runtime is
deterministic and Claude-free") and the existing `/evolve`/`ralph.sh` pattern: a `claude -p <prompt>`
invocation (cron-able, not yet cron'd) drives a set of deterministic Python modules as its tools.
The modules work standalone with no LLM present — independently testable, independently safe.
Sonnet's job each cycle: decide *which* hypothesis to advance, judge novelty against L-022,
interpret a gate failure, draft a lesson. The modules compute every number and enforce every rail.

### 4.1 New package: `src/brain_loop/`

```
src/brain_loop/
  hypothesis_registry.py   # pre-register (frozen-before-results, hash-chained JSONL)
  gate_runner.py            # shells out to the existing backtest harness; calls decide_cycle()
  promotion.py               # writes promotion_request.json; never sets gate_pass / calls start_loop
  monitor.py                  # reads live telemetry; evaluates rails via EXISTING thresholds
  derisk.py                    # ONLY module that touches live-affecting state: halt() + tighten_risk_config()
  cycle.py                      # per-invocation orchestrator; checks get_halted() first
```

State/artifacts: `trained_data/brain_loop/hypotheses.jsonl` (hash-chained pre-registrations),
`trained_data/brain_loop/cycle_ledger.jsonl` (hash-chained cycle decisions, same pattern as
`src/equity/cycle_ledger.py`), `trained_data/brain_loop/promotion_requests/*.json`.

### 4.2 Lifecycle

```
PROPOSE (hypothesis frozen + hashed)
   │
   ▼
BACKTEST (gate_runner shells out to the real harness; brain_loop computes nothing)
   │
   ▼
GATE — decide_cycle() [halt > DD > ship-gate > freshness] + ship-gate PASS/FAIL
   │
   ├─ FAIL ──► ledger entry (NO_ACT / HALT / ABSTAIN + reason) ──► done, propose next hypothesis
   │
   ▼ PASS
PROPOSED_FOR_SHADOW  (auto — no capital, no ARM required; same as today's research runs)
   │
   ▼
PROPOSED_FOR_LIVE  ──► promotion_request.json (status=PENDING_OPERATOR) ──► STOPS HERE
                         (operator ARMs + start_loop via the existing dashboard flow — outside brain_loop)
   │
   ▼ (once live, if ever)
MONITOR (rails: DD, cost anomaly, staleness) ──► breach ──► derisk.halt() or tighten_risk_config()
                                                              (actor="brain_loop", audited)
```

### 4.3 Hard-constraint → structural enforcement map

| Constraint | Enforcement mechanism | Verified by |
|---|---|---|
| Never places a trade / picks a signal | No import of `src/scanner/execution`, no broker client, no OANDA creds in `src/brain_loop/` | Grep-based import-boundary test |
| Never self-arms / never bypasses ARM for flatten/leverage/start_loop/unhalt | Those four action names never appear in `src/brain_loop/` (not policy-excluded — structurally absent); `control_safety.enforce()` remains the fail-closed backstop underneath regardless | Grep-based capability-absence test + existing `enforce()` unit tests untouched |
| Promotion to live requires operator ARM | `promotion.py` has no function that sets `gate_pass=true` on a live artifact or calls `start_loop`; writes a proposal file only | Test: proposal write never mutates `SHIP_GATE.json`; no `start_loop` reference in module |
| De-risk/halt always allowed | Reuses the exact precedented autonomous `set_halted(True)` pattern (`execution.py:6307`, `engine.py:5626`) — not a new capability, a new caller of an existing one | Test: `derisk.halt()` calls real `StateEngine().set_halted(True)` on tmp state, verified via disk read-back |
| Practice pin | Untouched; no `oanda_environment` reference anywhere in `src/brain_loop/` | Existing `verify_gate.py` Hard-NO scan covers new files automatically |
| Frozen-before-results | `hypothesis_registry.register()` hashes params at registration; `gate_runner.run()` refuses to attach a result to a hypothesis whose hash doesn't match | Unit test: mutate params post-registration → hash mismatch → refused |

## 5. Testing

No-mock, real-disk tests per repo convention (`tmp_path`, real `StateEngine`, real
`decide_cycle`, real `clamp_risk_pct`):

- `hypothesis_registry`: hash-chain integrity, frozen-before-results refusal, corrupt-file
  fail-closed.
- `gate_runner`: calls real `decide_cycle()` against real tmp state fixtures for each precedence
  branch (REFUSE/HALT/NO_ACT/ABSTAIN/CONTINUE).
- `promotion`: proposal schema; asserts it never writes `gate_pass` or calls anything ARM-gated.
- `derisk`: `halt()` flips real `state.json` via real `StateEngine` on `tmp_path`;
  `tighten_risk_config()` respects `clamp_risk_pct` bounds and refuses out-of-band deltas.
- `cycle`: full PROPOSE→GATE→(FAIL) and PROPOSE→GATE→(PASS)→PROPOSED_FOR_SHADOW paths on synthetic
  fixtures; halted=true short-circuits to REFUSE before any hypothesis work.
- Import-boundary / capability-absence: grep tests over `src/brain_loop/**/*.py` for forbidden
  imports/action names (execution, broker clients, `flatten`, `set_gross_leverage`, `start_loop`,
  `unhalt`, arm-setting calls).
- Regression: existing `control_safety`, `decision_gate`, `trend_risk_gates`, and Tier 7 test
  suites must remain green and unmodified.

## 6. Independent verification

Dispatch a **Security Engineer** sub-agent (safety-boundary review, not general code review) with
a claim manifest: brain_loop cannot place trades; cannot self-arm or bypass ARM for the four
gated actions; halt/de-risk uses the precedented pattern; promotion never auto-sets `gate_pass` or
calls `start_loop`; practice pin untouched; no regression in existing safety test suites.

## 7. How to start it (documented, not executed, in this task)

`scripts/run_brain_loop.py --once` runs one deterministic cycle standalone (no LLM, useful for
testing/cron dry-runs). To run it as the actual "Sonnet brain": a scheduled `claude -p` invocation
(pattern identical to `ralph.sh` / `/evolve`) with a fixed prompt that reads
`.claude/brain_loop/PROMPT.md`, calls the `src/brain_loop/*` modules as tools each cycle, and never
calls anything outside that package for live-affecting actions. Not scheduled/cron'd by this task.
