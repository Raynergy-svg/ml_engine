# AXIOM loop remediation plan — 2026-07-20

Full-loop audit, training → trading. Written after an operator report that "the loop seems dead
and axiom only trades."

**Scope of verification.** Every claim below was read from disk on 2026-07-20 between 19:42 and
20:00 UTC, on branch `ralph/equity-harvester-bot`. Claims marked **[V]** were verified directly by
the session author in that window. Claims marked **[A]** come from a domain sub-agent audit and
carry the confidence the agent assigned; per `.claude/rules/honesty.md` these default to MEDIUM
unless independently re-derived. Nothing was modified while producing this document.

**Bottom line.** The loop is not dead in one place. It is dead in five independent places, and
every one of them failed silently while the system's own health signals reported green. The
safety architecture is not the problem and should not be relaxed. This is a liveness and
observability failure, not a safety-posture failure.

---

## 1. The five deaths

### Death 1 — the resident loop's reasoning brain has been failing for ~38 hours [V]

`scripts/run_agent_runtime_loop.py` (pid 905, alive) shells out to the `claude` CLI for its
DIAGNOSE phase. That subprocess exits 1 on every cycle. **[A, HIGH]** The reproduced cause is
`401 OAuth access token has expired`.

The failure is invisible because `src/agent_runtime/loop.py:357-362` catches the non-zero return
code and discards both `proc.stdout` and `proc.stderr`. The CLI prints the 401 to stdout. What
reaches the log instead is the contentless string
`claude exited 1 -- degraded cycle, no actions proposed`.

Measured from `trained_data/axiom/loop_cycles.jsonl` **[V]**:

| Metric | Value |
|---|---|
| Total cycles recorded | 2,187 |
| Degraded (`claude exited …`) | 1,446 |
| Last non-degraded cycle | 2026-07-19T05:23:46Z |

Two false-greens compound it:
- `cli_available: true` is set from `bool(cli_path)` (`loop.py:238`) — path resolution only. An
  auth failure leaves it `true`.
- `verified: true` — `_verify` (`loop.py:606-618`) checks only the practice pin and "no escalation
  executed". Both are vacuously true when zero actions ran. **VERIFY passes hardest when the loop
  does nothing.**

`trained_data/axiom/loop_autonomy.json` is `{"agent_autonomy_enabled": true}` **[V]** — the loop is
not being denied by the autonomy flag. Its brain is simply unreachable.

Known-recurrence note: project memory already records that this machine's standalone-CLI
credentials "go stale ~weekly", and that a sibling loop (`ralph.sh`) has a latent bug where it
does not abort on 401. Same failure mode, new location, still unmonitored.

### Death 2 — five launchd daemons crash-looping on a macOS TCC denial [V]

`launchctl print` for all eight ml_engine jobs shows a perfect 8/8 correlation between the
`StandardOutPath` location and job health:

| stdout path | jobs | runs | last exit |
|---|---|---|---|
| `~/Library/Logs/…` | operator, tier7, web | 1 | never exited — **alive** |
| `~/Documents/ml_engine/trained_data/axiom/…` | api, safety_monitor, tick_capture, trend, exposure_history | 162–485 | **78 EX_CONFIG** |

**Mechanism.** launchd's `xpcproxy` opens the stdout/stderr files *before* exec'ing the program.
`~/Documents` is TCC-protected; once the per-file `com.apple.macl` grant lapsed, that open was
denied and the job died before Python started. This is why there is no traceback anywhere — there
was no process to write one.

`KeepAlive` did not fail; it worked perfectly and that is what hid the outage. `runs = 485` on the
API job (no `ThrottleInterval`, so 10s retries) and `runs = 162` on the others (30s
`ThrottleInterval`) means the fleet is retrying roughly 700 times per hour, producing zero output.

**This exact bug was diagnosed and fixed here twice before** — `876c291` (operator) and `0cc5c49`
(tier7 + web). The `0cc5c49` commit message explicitly notes the remaining jobs "still point under
`~/Documents` … but carry the same latent risk — left alone here since they aren't broken." The
latent risk fired.

**Staggered decay, not one outage** **[A, HIGH]**: `safety_monitor` last successful run
2026-07-07T10:58Z (13 days; its final record was `"ok": false`), `api` last log 2026-07-15 13:14,
`trend`/`tick_capture`/`exposure_history` Jul 16 ~04:40-04:59. Three separate silent deaths.

**Data loss:** ~4 days of OANDA tick capture is permanently gone — OANDA does not backfill
streams.

### Death 3 — training closure has two independent breaks [V]

The self-heal handler `src/scanner/feedback/self_heal.py:1079` writes a marker file to
`trained_data/retrain_requests/` and returns. Its own docstring says so: *"Training itself runs off
the hot path."*

- **Break 3a — the consumer is never scheduled.** `scripts/offline_learning_cycle.py` is the
  designated consumer. Its launchd job `com.buddy.learning_loop` is absent from the installer's
  `LABELS` array (`scripts/axiom_launchd/load.sh:43`). This is *deliberate* — the comment block
  above it states the learning batch has "separate activation authority" and must be armed as an
  explicit step. That step was never taken; the plist has sat unused since Jul 7.
- **Break 3b — the consumer never trains.** In `scripts/offline_learning_cycle.py`, `markers`
  appears at exactly four places: `:287` (assignment), `:299`, `:323`, `:387` — all `len(markers)`.
  There is no call to `scripts/train_rl_suite.py:240 train_position_sizer()` anywhere in the file.

**State:** 41 markers pending (oldest `rl_sizer_20260707_071549.json`);
`trained_data/learning_loop/history.jsonl` contains **two lines in its entire lifetime** (Jul 4,
Jul 7). The Jul 4 run recorded `holdout_wins: 1, holdout_losses: 14` →
`decision: insufficient_holdout_signal`.

**No recurring model training exists on this machine** **[A, HIGH]**: no launchd job invokes a
`train_*` script; `crontab -l` has one unrelated entry. All training is manual.

These two breaks would have hidden each other. Fixing only the scheduling would drain all 41
markers, log `"markers_drained": 41`, and go quiet — while the sizer stayed 10.9 days stale. The
dashboard counts *pending markers* (`dashboard/server/data_sources.py:1400`), so a drained queue
reads as green. **The metric measures queue depth, not outcome.**

### Death 4 — the RL feedback loop never closes [A, HIGH]

`trained_data/trade_journal_rl.json` holds 247 entries, **every one with `outcome: null` and
`pnl: null`**. Zero closed records means `update_weights_from_outcome()` has never fired for any of
them. Agent weights do not learn from results.

This is the two-writer done-guard pattern already documented four times in
`.claude/rules/improvement.md` ("Live Wiring Verification Gates", final bullet).

### Death 5 — nothing can ever be armed [A, HIGH]

`LiveGate.arm()` is fully implemented (`src/equity/live_gate.py:445-568`, ~120 lines with real
preconditions) and **has zero callers**. `grep -rn "\.arm(" src scripts cli dashboard` returns only
docstrings and `is_armed()` check sites.

- No `live_gate_state.json` or `live_gate_audit.jsonl` exists anywhere under `trained_data/` —
  every gate is disarmed by absence.
- No arm script exists (`ls scripts/` has no `arm_*.py`); the TUI modal referenced in the
  docstrings governs `dry_run → live` mode, not the LiveGate.
- `arm_live_gate` and `promote_model` deliberately return HTTP 501
  (`dashboard/server/axiom_proposals.py:54`), on the stated rationale that getting
  `expected_universe_hash` subtly wrong is worse than leaving it manual.

**Consequence: even with every other break fixed, no lane could place an order.**

---

## 2. Lane status

| Lane | Evidence verdict | What blocks it |
|---|---|---|
| **equity harvester** | **`gate_pass: true`** — net Sharpe 0.906, maxDD 0.229 | halted + never armed + abstaining `stale_data:8d>7d` |
| track_b | INSUFFICIENT | cross-sectional power **met** (425/405); temporal `n_rebalances = 1` — only forward time closes it |
| crypto_momentum | fails significance | 1 forward cycle of 3 required; **no broker integration exists** |
| crypto_carry | fails cost-adjusted return | 1 cycle of 3; no broker; not wired into the agent's shadow-ledger reader |
| oanda_fx | no evidence package | halted; NOTES records the lane negative-expectancy |
| hedge | insufficient_history | `RUNTIME_ALLOWED = False`, `PAPER_ONLY = True` hardcoded (`src/hedge/hedge_scorecard.py:53-55`) |

Evidence store `trained_data/evidence/indexes/current.json`: **`"champions": {}`** — nothing has
ever been promoted. The only 8 packages are risk-policy models at QUARANTINED/REJECTED. The
`champions/` and `quarantine/` directories are empty.

The promotion state machine (`src/evidence/transition_policy.py:15-41`) has **no edge from
QUARANTINED to OPERATOR_APPROVED or CHAMPION** — the SHADOW waypoint is structurally
unskippable, and reaching CHAMPION requires four distinct actor identities and four distinct
signing keys.

---

## 3. Three corrections to the working assumptions

1. **The 7-consecutive-loss breaker never halted anything.** Alert threshold is 3
   (`src/scanner/automation/alert_manager.py:59-63`); auto-halt threshold is **8**
   (`src/scanner/config.py:673`, relaxed 5 → 8 on 2026-07-10 by operator approval). Seven losses
   fired a WARNING only. **What set the current global halt is UNKNOWN** — `.claude/state.json`
   still reads `last_actor: "operator_directed_unhalt"`, so whatever re-halted at 2026-07-15T23:04
   left no marker.
2. **The system did trade, and it is net positive.** `trained_data/axiom/trend_loop.out` is 209
   `ran=True reason=executed` vs 236 `ran=False reason=halted` — not all-halted. Lifetime realized
   P&L **+$2,419.64**, NAV $102,308.89, drawdown ~1.2% against peak.
3. **The unhalt path is self-blocked.** `.claude/loop/verdict.json` was last modified 2026-07-15
   01:47 — ~482,000s old against the `UNHALT_VERDICT_MAX_AGE_S = 86400` ceiling
   (`dashboard/server/control_safety.py:87`). An authorized operator hitting unhalt today gets
   `ControlDenied("verify_gate verdict stale")`. Re-running `/verify-task` is a prerequisite.

**Open live item:** the open `EUR_USD +200` position has `stop_loss: null, take_profit: null` —
unbracketed, caused by a `TAKE_PROFIT_ORDER_WOULD_VIOLATE_FIFO_VIOLATION_SAFEGUARD` rejection on
2026-07-15T15:55Z. Practice account, small size, but the bracket is absent.

---

## 4. What is NOT broken — do not "fix" this

The bounded-autonomy architecture is the strongest part of the system and none of the remediation
below touches it.

- **19 registered actions across 4 tiers** (`src/agent_runtime/policy.py`). Fail-closed: an action
  absent from the registry is denied, never defaulted.
- **Every "do more" verb is ESCALATION and cannot execute.** `ActionSpec.__post_init__`
  (`policy.py:126-136`) *refuses to construct* an ESCALATION spec carrying an `execute` callable.
  There is no code path from `submit()` to a state mutation for that tier. The six escalation
  actions are `unhalt_lane`, `arm_live_gate`, `increase_gross_leverage`, `promote_model`,
  `enable_new_exposure`, `change_strategy_or_code`.
- **DEESCALATION is risk-decreasing by construction** — `_deescalate_halt_lane` passes a literal
  `True` to `set_halted`; `reduce_gross_leverage` raises unless strictly lower.
- **SELF_IMPROVE denylist evaluated first and unconditionally** (`self_improve.py:64-86`). It
  cannot write execution, gates, config, brokers, risk, control-safety, its own guardrail code, or
  `loop_autonomy.json` — **the agent cannot edit its own permission flag.** Allowlist is exactly
  three paths.
- **Lifetime mutation count: 34 actions** across ~2,186 cycles — 15 weight-hygiene purges, 11 lane
  halts, 8 lessons rows. Everything else was read-only observation.

Halt topology is likewise sound: global short-circuits before any per-lane read; `oanda_fx` uses
fail-closed `get_halted_strict` at both order-adjacent points (`execution.py:2099`, `:6679`).

Two asymmetries worth noting but not urgent: `src/brain_loop/cycle.py:61` uses the fail-**open**
`get_halted` (brain places no broker orders); and the TUI unhalt path (`src/tui/app.py:2668-2769`)
lacks the ARM gate, verdict check, and drawdown rail that the dashboard path enforces — it is
materially weaker and unhalts all six lanes at once.

---

## 5. Remediation, in dependency order

### P0 — restore liveness and observability (hours; zero safety impact)

| # | Action | File | Why first |
|---|---|---|---|
| 1 | Log `stdout`/`stderr` on non-zero exit; make `cli_available` reflect auth, not path | `src/agent_runtime/loop.py:357-362`, `:238` | The 401 has been invisible for 38h. Highest value per line in this document. |
| 2 | Re-authenticate the `claude` CLI | operator action | Restores the DIAGNOSE phase |
| 3 | Repoint the 7 remaining plists' log paths to `~/Library/Logs/` | `~/Library/LaunchAgents/*.plist` **and** `scripts/axiom_launchd/*.plist` | Finishes `0cc5c49`. Must fix both or `load.sh` reinstalls the bug. Include `crypto_refresh` + `forward_daily`, which will hit this when they next fire. |
| 4 | Add `ThrottleInterval` to `com.axiom.api` | `scripts/axiom_launchd/com.axiom.api.plist` | Currently 6 failed spawns/minute indefinitely |
| 5 | Rotate `control_audit.jsonl` (151M), `loop_cycles.jsonl` (148M), `episodes.jsonl` (139M), `launchd-web.log` (93M) | `trained_data/axiom/` | **Volume at 99% — 3.3 GiB free of 228 GiB.** Independent imminent outage. Operator decision on retention; these are governance/evidence data. |
| 6 | Build a fleet watchdog: read `launchctl list`, assert every expected label has a live PID and exit 0, write a machine-readable staleness file, alert out-of-band | new | **Nothing today would ever report a dead fleet.** Must live outside the fleet it watches and must not depend on `:8888`. |
| 7 | Make `.claude/heartbeat.json` a fleet rollup, not one writer's self-report | `tier7_supervisor` writer | It currently reads healthy while 5/8 jobs crash-loop |

Note on #6/#7: `scripts/heartbeat_watchdog.sh` exists but was never installed, and targets label
`com.buddy.trader`, which is not in the fleet — it would watch nothing even if installed.

### P1 — close the learning loops (this week)

| # | Action | Why |
|---|---|---|
| 8 | Either wire `train_position_sizer()` into `offline_learning_cycle.py`, **or delete the `retrain_rl_position_sizer` action** | Do not leave an action that lies about what it does. Either close it or stop claiming it. |
| 9 | If keeping it: add `com.buddy.learning_loop` to `load.sh:43` as an explicit, separately-documented arming step | Respects the original separate-activation-authority decision |
| 10 | Stamp trade outcomes so `trade_journal_rl.json` stops being write-only; use a field dedicated to the RL consumer (e.g. `rl_weights_applied`), never a shared value-presence guard | The documented fix pattern for this exact bug class |
| 11 | Re-run `/verify-task` to refresh `.claude/loop/verdict.json` | Unhalt is unreachable until this is fresh |
| 12 | Add a liveness assertion to `_verify`: a cycle with a failed DIAGNOSE must not report `verified: true` | Closes the "verify passes hardest when nothing happens" hole |

### P2 — the actual autonomy question

| # | Action | Why |
|---|---|---|
| 13 | Build the arming path — script and/or wired UI — targeting the equity harvester | The only lane with a passing gate. Until this exists, "autonomous trading" is unreachable by construction. |
| 14 | Refresh harvester input data | It has been abstaining on `stale_data:8d>7d` regardless of halt state |
| 15 | Drain the 23 undispositioned escalation proposals | One lifetime disposition exists, by actor `test-operator`. The dashboard drain path (`axiom_proposals.py:167`) has never been used. |

---

## 6. Strategic note — what "autonomous" should mean here

Making the **FX trading lane** autonomous is the wrong target. LESSONS L-021/L-022 and
`.claude/NOTES.md` both record that FX trend is a reproducible risk premium rather than alpha, and
that the OANDA lane remains negative-expectancy. Autonomy over a negative-expectancy lane loses
money faster and more reliably.

The defensible target is **an autonomous research loop**: the cycle that turns forward shadow
observations into evidence packages and verdicts. This is what the architecture was actually
designed for, and it risks nothing while halted.

The evidence for this is concrete. track_b's own coverage note states its cross-sectional power
target (~405 filings) is **met** — 425-427 observed — and that the honest answer was a well-powered
null (+0.16 rank-IC fading to +0.09, n=291, non-significant). What remains is *temporal* power:
`n_rebalances = 1`. **That gap closes with accumulated forward time and nothing else.** A system
that reliably accrues shadow evidence while halted is genuinely autonomous, needs no gate
relaxation, and is the only version of autonomy that moves toward the end goal in `.claude/INTENT.md`.

Concretely, that reprioritizes P2 toward: keep the shadow lanes running and capturing, keep the
evidence store accumulating, and let the verdicts move on their own schedule. The arming path
(#13) matters for the harvester specifically, because it is the one lane whose evidence already
cleared.

---

## 7. Recurring theme worth promoting to a lesson

Four of the five deaths share one shape: **a health signal that measures the wrong side of the
transaction.**

- `verified: true` measures "no rule broken", not "work happened"
- `cli_available: true` measures path resolution, not auth
- the dashboard's retrain metric measures queue depth, not model freshness
- `.claude/heartbeat.json` measures one surviving writer, not the fleet
- `KeepAlive` measures process restart attempts, not process success

Each of these certified a broken state as healthy. Candidate lesson: **a liveness check must
assert that useful work occurred, not that no violation was detected** — the absence of a
violation is also what a corpse reports. Route through `/evolve` if it survives operator review.
