# NOTES — live working memory (survives compaction)

> This is the **only** file I (Claude) may update without operator approval. It holds *state*, not
> doctrine. New decisions go to INTENT, new failure modes go to LESSONS, new patterns go to a skill
> — all via `/evolve`, with operator approval. Keep this file short and true; prune what's stale.

Last touched: 2026-07-08T22:05Z by Claude (trend-lane risk-gate wiring: cost model + portfolio
exposure engine + hedge scorecard — committed on `ralph/equity-harvester-bot` as `85e847e`/`a3c5c79`/
`9c218e9`; no `.claude/state.json`/halt/practice-pin touch, verified via independent Security Engineer
SAFE (7/7) + Code Reviewer SHIP (99/99, then 101/101 after 2 follow-up fixes)).

## Trend-lane risk-gate wiring — cost model + exposure engine + hedge scorecard (2026-07-08, ADDITIVE)

Operator-directed wiring pass: made the gated decision/evaluation path CONSUME 3 previously-built-but-
unwired tools, additively (extend-never-weaken), on the LIVE oanda_fx trend lane
(`src/equity/oanda_trend.py::run_oanda_trend_cycle` — confirmed running as a real PID against practice
throughout this session, contrary to the task brief's "gate is halted" assumption — flagged to operator).

- **Cost model** (`src/data/execution_cost_model.py`, unmodified) → new `cost_aware_gate()` (Rule 7,
  `src/equity/trend_risk_gates.py`) → wired into the order-approval loop. Fail-closed on missing/
  partial/non-finite/negative-spread cost data (never assumes free); refuses when modeled round-trip
  cost exceeds 30% of the candidate's own 1R stop budget (`DEFAULT_MAX_COST_R_FRACTION`).
- **Exposure engine** (`src/hedge/portfolio_exposure.py`, unmodified) → new `exposure_engine_gate()` +
  `_open_trades_to_exposure_positions()` in `oanda_trend.py`, run ALONGSIDE (never replacing) the
  pre-existing `bucket_cap_gate` — an AND-combination, so the combined check can only be MORE
  restrictive. Fails closed on any unresolvable leg AND (code-reviewer-driven follow-up) on any short
  leg, since the engine's signed-net semantics only match `bucket_cap_gate`'s unsigned per-direction
  semantics in the long-or-flat regime this lane is built for.
- **Hedge scorecard** (`src/hedge/hedge_scorecard.py`, unmodified) → new read-only
  `_hedge_scorecard_section()` in `src/scanner/automation/maintenance_report.py` (stdlib-json read of
  the persisted artifact, matching that module's existing no-live-import contract) — evaluation/
  reporting path only, zero order-path touch.
- `decision_gate.py` (equity harvester) deliberately NOT touched for cost-model wiring: it has no
  per-ticker universe param and execution_cost_model is OANDA-FX-only by construction (no equity fill/
  tick source exists) — forcing a per-ticker equity cost check would be fabrication. Documented
  scope-cut, not a silent skip.
- **Reproduce-then-resolve**: re-ran ALL pre-existing bucket_cap_gate/one_position_gate reproduce-cases
  unchanged (USD-pileup, same-cycle bypass close, red-trade average-down block) — all still green,
  proving nothing loosened. Added a new full-cycle reproduce case showing the exposure engine
  independently enforces the same same-cycle USD-pileup cap. 101 total tests across every touched
  module, 0 failures; flake8 clean; `risk_monitor.sh` GREEN; `verify_gate.py` PASS (28/28).
- **Independent verification**: Security Engineer SAFE (7/7 checks: nothing loosened, genuinely
  fail-closed, no new trade/arm/live/unhalt surface, practice pin + halt position unchanged, no
  injection surface, no circular import). Code Reviewer SHIP, found 2 real-but-low-severity issues
  (negative spread not independently validated; exposure engine's signed-net vs bucket_cap_gate's
  unsigned semantics only equivalent because the lane is long-only) — both fixed same session
  (`cost_negative_spread_invalid` guard; `exposure_engine_short_leg_unsupported` fail-closed guard),
  with regression tests, then re-verified green.
- **Process anomaly (for `/evolve`)**: a CONCURRENT autonomous session (co-authored "Claude Fable 5",
  same git identity "Buddy Bug Exterminator") was committing its own unrelated risk-target ML pipeline
  work to this SAME branch/working-tree throughout this session — with no session isolation, `git
  add -A && git commit` on their side repeatedly swept my concurrently-edited working-tree files into
  THEIR commits (`85e847e`, `a3c5c79`, `9c218e9`). Verified byte-identical each time (nothing lost/
  altered) via `git diff HEAD --stat` before/after and direct content diff. Also hit L-024 directly: an
  exploratory `git stash` to A/B-test an unrelated pre-existing test failure conflicted on live-daemon-
  written files; recovered cleanly via targeted `git checkout stash@{0} -- <path>` per file (skipping
  the 4 live-fresher paths), never dropped/lost. New lesson candidate: concurrent uncoordinated
  autonomous sessions on one working tree can co-mingle commits — check `git log` for foreign commits
  mid-session before assuming your own diff is what's on disk.

## AXIOM Hedge Layer Phase 4 — raw-vs-hedged shadow lane + scorecard (2026-07-08, SHADOW/ANALYSIS-ONLY)

Built on Phase 1+2 (`b4698af`) + Phase 3 (`1463ce2`): `src/hedge/hedged_shadow_lane.py` reads each
covered shadow strategy's own book (equity harvester `rebalance_state.json`, crypto momentum + track_b
ledgers — never forked), runs it through the REAL P1-P3 pipeline (`build_exposure_report` +
`build_hedge_report`), marks Lane A (raw) vs Lane B (raw + applied hedge) forward via the cached
`market_data/equity/sp500_prices.parquet` panel, and logs both to `trained_data/hedge/
raw_vs_hedged_ledger.jsonl` + `hedge_decision_log.jsonl`. `src/hedge/hedge_scorecard.py` computes
per-strategy expectancy/Sharpe/Sortino/drawdown/hit-rate for both lanes and a 3-way decision readout
(signal_real_but_noisy / weak_or_dead / return_was_beta_not_alpha), honestly gross-only + flagged
`cost_unmodeled_for_venue` when the hedge instrument (SPY/sector ETF) has no OANDA fill history for
the P2 cost model to price — never fabricates a net number.

Extended `src/hedge/config/sector_bucket_map.json` with 7 tickers (C, BRK-B, COST, CSCO, IBM, INTC,
ORCL) so the equity harvester's full 20-name book resolves (P1-P2's exposure report is all-or-nothing
fail-closed — one unmapped ticker blocks the whole hedge). Found + fixed a real cached-panel data-
quality bug while building the demo: one ticker (`POM`) had an unadjusted +1844% single-day print
that dominated a naive equal-weight "market" proxy — added `MAX_SANE_DAILY_RETURN` outlier guard +
restricted the market proxy to the curated sector-map universe instead of all 684 raw panel columns.

Independent Code Reviewer pass (before commit) found 4 real correctness bugs, all fixed + regression-
tested before commit: (1) `basket_forward_return`/`hedge_overlay_pnl` summed only resolved legs
without renormalizing — mathematically a silent zero-fill of unresolved legs; now all-or-nothing
fail-closed matching P1-P2's own contract. (2) scorecard's raw-lane Sharpe/Sortino paired the FULL
unfiltered asof_date list with the FILTERED resolved-returns list, misannualizing the cadence
(reproduced a ~5.5x distortion). (3) the "hedged" stats series wasn't restricted to
`hedge.status=="applied"` cycles, letting trivial hedged==raw rows (no valid hedge) dilute the
alpha-vs-beta read. (4) the `hedge_unavailable` insufficient-history gate used a different `n` than
`decision_readout`'s own gate. Security Engineer independently verified PASS on all 8 boundary checks
(zero broker/execution import, zero state/halt/gate touch, zero network I/O, writes scoped to
`trained_data/hedge/` only, safety triplet hardcoded non-overridable, `.claude/state.json` unchanged,
config diff is pure data). 27 new tests + 42 pre-existing hedge tests + 755 equity tests + 100 crypto/
hedge regression tests all green; flake8 clean.

Real production run (single cycle each, `n=1` per strategy — day 1 of this lane, honestly reports
`insufficient_history`): equity harvester's raw return is unresolvable (cached price panel stale past
2026-06-24, no forward bar for the 2026-07-01 book snapshot — reports `None`, not a fabricated
number); crypto momentum + track_b both correctly report `hedge.status` as `unsupported_asset_class`
/ `fail_closed` (crypto isn't a P1-covered asset class; track_b's research universe has large
sector-map coverage gaps) — Lane B == Lane A by construction in both cases, honestly labeled, no
fabricated hedge. A historical-window demo (`scripts/hedge_shadow_lane_demo.py`, real cached prices,
2026-06-15→23, 6 cycles) shows the full pipeline populated: raw expectancy -0.29%/cycle vs hedged
GROSS +0.08%/cycle with materially lower drawdown (0.26% vs 1.71%) — `cost_unmodeled:
signal_real_but_noisy`, correctly caveated since equity hedge cost is genuinely unknown (no OANDA
fills for SPY/XLK).

## Evaluated 2 operator repos for memory/brain scaling + built ENGINEERING_BRAIN (2026-07-07)

- Built `docs/ENGINEERING_BRAIN.md` (commit `b836075`) — sourced blueprint (current state / safety spine /
  external best-practice+edge+data map / P0-P3 roadmap + durable facts). verify_gate PASS, risk_monitor
  GREEN, state.json + practice pin untouched.
- Operator pointed at 2 of their own repos to scale memory+brain: **ML_Training_book** (fork of Harvard's
  *Machine Learning Systems* textbook — an ML-systems engineering knowledge source) and
  **codebase-memory-mcp** (fork of `DeusData/codebase-memory-mcp` — tree-sitter+Hybrid-LSP code
  knowledge-graph MCP, 158 langs, ~120× fewer tokens on structural queries, arXiv:2603.27277).
- Disk-verified honest status: codebase-memory-mcp is wired **CI/dev-plane ONLY**
  (`.github/workflows/code-graph.yml`, commit 35f430f — PR impact analysis, fail-soft, checksum-pinned,
  never touches src/runtime/execution/halt/config). **NOT** installed as a local interactive MCP this
  session (no `cbm` binary, not in connected servers). It is a dev-plane PR tool, **not** live runtime
  memory — don't narrate it as one (L-017).
- `/evolve` proposal (INTENT standing decision: sanctioned brain-source + code-memory tool with a
  dev-plane/no-hot-path/no-safety-write guardrail) **pending operator approval** — not yet applied.
- **OPERATOR-DIRECTED UNHALT (2026-07-07, "switching strategies anyway"):** verified practice from disk
  (`config.py:742`, `ScannerConfig().oanda_environment=='practice'`) BEFORE flipping, then via sanctioned
  `StateEngine.set_halted(False, lane=...)`: unhalted **`oanda_fx`** (practice trend, paper OANDA fills)
  and **`crypto_carry`** (shadow funding-carry). All 6 lanes now unhalted; global `halted=false`. Both
  standard + strict readers agree. Live trend daemon (PID 41009) was `REFUSED reason=lane_halted:oanda_fx`
  → resumes paper fills next cycle. Acked the stale 2026-07-02 `consecutive_losses` WARNING
  (`AlertManager.acknowledge`). risk_monitor GREEN, verify_gate PASS(28), practice pin intact after write.
- **ARM NOT DONE (held at the escalation boundary):** operator chose "try to arm harvester LiveGate" but
  two hard blockers remain — (1) IB Gateway **closed on 7497** (arm→IBKR-paper can't fill), (2) `arm()`
  requires the typed **"LIVE"** token (a human friction I won't fabricate). Also flagged: `SHIP_GATE.json`
  `passed:true/0.906` is the generous curated-20 universe; the defensible wide-universe number
  **0.740/0.355 FAILS**. To arm: operator starts IB Gateway (paper) on 7497 + types LIVE; then I run the
  arm. Real money stays off regardless (practice pin immutable).
- **INTENT #10 APPLIED (2026-07-07, /evolve approved "both on"):** standing decision #10 added — two
  sanctioned external memory/brain anchors (Harvard MLSys textbook fork + codebase-memory-mcp) with the
  dev-plane/no-hot-path/no-safety-write guardrail. `/evolve` for this task now complete (1 intent edit).
- **tick_capture ACTIVATED — running:YES (2026-07-07, operator data-acquisition mandate):**
  `scripts/run_tick_capture.py --pairs ALL_FX`, **PID 3655**, nohup (creds sourced from `.env.local`;
  runner has NO `--env` flag and `build_practice_stream_client` hard-pins `stream-fxpractice.oanda.com`
  regardless of `OANDA_ENVIRONMENT`). Log confirms "PRACTICE-ONLY, read-only pricing stream confirmed"
  + real flushes (166/218/209/230 ticks) → `trained_data/ticks/` (15 FX majors, 180-day retention).
  Read-only market data, zero order/hot-path contact. **Durability caveat:** nohup, NOT launchd — dies
  on reboot, no KeepAlive. Reboot-persistence = optional follow-up (a launchd plist like the trend lane).
  Stop with `kill 3655`. Prior 11:00Z run was a 33-tick smoke test that had stopped.
- **"TURN EVERYTHING ON" (2026-07-08, operator):** all 6 lanes unhalted; full shadow fleet running —
  tier7 (73646), trend/oanda_fx (11508), tick_capture (3655, dup 3222 killed), track_b_shadow (30976),
  crypto_momentum_shadow (30975), crypto_carry_shadow (11554, newly started), brain_loop (12143,
  `--loop --max-cycles 8760 --interval 3600` — the earlier 1-cycle exit was `--max-cycles` default=1,
  not a crash), equity_harvester --broker shadow (96555), control API uvicorn (79016), web next-dev
  (97993). risk_monitor GREEN, verify_gate PASS(28), practice pin intact. **Still OFF (consequential,
  await explicit go):** (1) live ARM — `armed=False`, no `live_gate_state.json`, SHADOW; needs IB Gateway
  + typed LIVE (escalation). (2) AXIOM resident-agent autonomy — `agent_autonomy_enabled:False`, resident
  loop not running (bounded: even ON it can't arm/unhalt/trade-more — escalation is proposal-only).
- **DASHBOARD CRASH — REAL ROOT CAUSE FOUND & FIXED (2026-07-08, commit d1622f7).** "Objects are not
  valid as a React child {…n_scored_filings…}". My earlier calls were ALL WRONG (L-018 fess-up): it was
  NOT a stale HMR chunk, NOT TrackBPanel, NOT the crypto panels. Commits 3dc666e (guard 3 panels' Row) +
  the `rm -rf .next` rebuild were misdirected — they never touched the real bug. **Actual cause:
  `ControlPanel` renders the control-audit trail; `<span> · {e.result ?? ...}` renders the entry's
  `result` raw. The AXIOM resident agent's read-only tool calls log their OUTPUT into `result`, which is
  an OBJECT for 22/25 entries (a shadow-lane snapshot — track_b-SHAPED because a track_b tool-read was
  logged). `??` only guards null → the object hit JSX → crash → route error boundary blanked Automation
  (and Settings, since ControlPanel is on both — matching the dev-overlay's page.tsx:280 pointer I
  dismissed).** The `AuditEntry.result?: string` TYPE LIED, which is why static reading kept missing it.
  **Two enabling fixes that made this findable:** (1) `next.config.ts allowedDevOrigins:["127.0.0.1",
  "localhost"]` — Next 16 was blocking the `/_next/webpack-hmr` ws as cross-origin
  (ERR_INVALID_HTTP_RESPONSE); this killed HMR AND blocked headless hydration. Fixing it restored
  hot-reload and let me finally reproduce + BROWSER-BISECT (disable panels one-by-one) to prove
  ControlPanel was the culprit. (2) The fix: `typeof e.result === "string" ? e.result : (allowed?"ok":
  "denied")` + same for `e.reason` + retype both `unknown`; plus `safeChild()` defense on ui Badge/
  SectionTitle. **VERIFIED IN LIVE BROWSER**: Automation tab renders all panels incl. ControlPanel audit,
  zero render errors, hasErrorCard=false. tsc clean, risk_monitor GREEN. **LESSON candidate for /evolve:**
  a lying `?: string` type on a field that's actually an object crashed React; when static reading fails
  to find a render bug, fix HMR then browser-bisect (comment panels) — don't keep guessing from source.

## AXIOM Hedge Layer Phase 3 — hedge candidate generator (2026-07-07, SHADOW/ANALYSIS-ONLY)

Built on Phase 1+2 (`b4698af`): `src/hedge/hedge_candidates.py` takes a candidate trade + the
netted post-candidate `ExposureReport` and proposes RANKED hedges — never executes. FX styles:
direct offset (short the flagged currency, sized to fully cancel it — sign math verified exact)
and relative-value (long/short two counter-currencies). Equity styles: market (short SPY),
sector (short the sector ETF from `sector_bucket_map.json`), relative-value. Cost ranking reuses
the real P2 `src/data/execution_cost_model.py` (not forked) — unknown cost (no fill/tick history)
is NEVER assumed cheap, routes to `BLOCK_COST`. Decision enum: `APPROVE_RAW` / `APPROVE_HEDGED` /
`REDUCE_SIZE` / `BLOCK_CORRELATION` / `BLOCK_COST` / `BLOCK_NO_VALID_HEDGE` / `SHADOW_ONLY`
(missing/unresolved exposure data). Fixed safety triplet
(`runtime_allowed=False, paper_only=True, human_review_required=True`) on every output — module
constants, never parameters.

- **Anti-fabrication fix mid-session**: independent Code Reviewer caught the FX relative-value
  style initially picking "strongest/weakest" counter-currency by alphabetical sort with zero
  ranking data behind the "does X outperform Y" claim — inconsistent with the equity RV style,
  which correctly required a caller-supplied `relative_strength` dict. Fixed to match: FX RV now
  also fails closed (`no_relative_strength_data_supplied`) without real ranking data.
- **Known non-blocking gap** (Code Reviewer finding, deferred): equity hedge generation reads
  `net_beta_exposure`/`net_sector_exposure` but not `net_correlation_bucket_exposure` — a
  concentration that only crosses threshold at the CROSS-SECTOR cluster level (e.g. Energy +
  Materials both mapping to `commodity_cyclical`) can surface in `risk_issue` without the
  proposed hedge fully addressing it. Would need a new `correlation_bucket` hedge style; out of
  scope for this pass, flagged here for a future session.
- 17 new tests (42 total in `src/hedge/`, no mocks) + demo script
  (`scripts/hedge_candidates_demo.py`) producing real sample JSON for the USD-pileup and a
  Technology-sector equity sleeve. Independent Security Engineer: SAFE, all 9 checks PASS (no
  broker/execution/halt/gate/leverage touch, safety triplet genuinely immutable, fail-closed
  paths real not cosmetic, practice pin + halt state untouched by this diff). Independent Code
  Reviewer: sign math verified exact, decision-tier branch ordering sound, ranking never treats
  unknown cost as cheap, reuse-not-fork confirmed — 1 fix applied (above), 1 gap deferred (above).
  flake8 clean.

## AXIOM Hedge Layer foundation — exposure tagging + portfolio netting (2026-07-07, SHADOW/ANALYSIS-ONLY)

Built per operator directive: Phase 1 (`src/hedge/exposure_tags.py`) decomposes FX pairs into
signed currency legs (reuses `trend_risk_gates.currency_legs`, not forked) tagged risk-on/
safe-haven/commodity-linked, plus equity tickers into sector/beta/factor/benchmark-hedge tags
from new `src/hedge/config/{currency,sector,correlation}_bucket_map.json`. Phase 2
(`src/hedge/portfolio_exposure.py`) nets a shadow open book + an optional candidate into
currency/sector/beta/correlation-bucket exposure (correlation bucket = currency for FX, sector
cluster for equity — unified, not two taxonomies), reusing
`DEFAULT_BIAS_SHARE_THRESHOLD`/`DEFAULT_BIAS_MIN_INSTRUMENTS` from the same risk gate for
concentration warnings.

- **Reproduces the motivating case directly**: long USD_CAD + USD_CHF + USD_JPY nets to ONE
  100%-concentrated USD bucket (`scripts/hedge_exposure_demo.py` sample output), not 3
  independent trades — exactly the -$816 FX-day failure mode.
- **Fail-closed contract, verified at both tag and report level**: any unresolvable instrument/
  currency/ticker/direction sets `fail_closed=True` with a named reason; `build_exposure_report`
  propagates any sub-failure to the whole report rather than silently zeroing the gap.
- **No execution path**: pure functions over caller-supplied `ExposurePosition` data — zero
  broker/network/`state.json`/`LiveGate` coupling (grep-confirmed by the verifier).
- 25 new tests (no mocks, real disk config load via `load_bucket_map`), independent Security
  Engineer verifier: PASS on all 6 checks (no execution surface, fail-closed contract,
  reuse-not-fork of `trend_risk_gates`, practice/halt untouched via `git diff --stat`, test
  integrity incl. non-tautological USD-pileup test, no-mock compliance). `risk_monitor.sh` GREEN.
  flake8 clean. Committed `b4698af` on `ralph/equity-harvester-bot`.
- **Phase 3 (hedge candidate generator, still analysis-only — no execution) built same day, see
  entry above.** Phase 4+ (any actual execution) remains explicitly NOT built.

## crypto_carry SHADOW lane — cash-and-carry funding harvest (2026-07-07)

Stood up per operator directive (docs/ENGINEERING_BRAIN.md P3): the strongest genuinely-live retail
lever, harness-gated, shadow-only, mirroring crypto_momentum/track_b exactly. Commit `1eb8cfa`.

- Pre-registered BEFORE running the backtest: `docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md`.
  Construction: long-spot-proxy/short-perp, positive-funding-only, delta-neutral by construction
  (no price P&L term — disclosed simplification). Signal/universe/cost imported verbatim from the
  existing H1 harness (`scripts/experiment_crypto_funding_carry.py`); vol-target overlay imported
  from `scripts/experiment_crypto_h2_infra_stress.py`. New frozen backtest:
  `scripts/experiment_crypto_cash_and_carry.py`.
- **Backtest result (honest negative, ship gate FAILS)**: OOS net Sharpe −4.48, maxDD −0.465, DSR
  0.00. Carry itself is real (+0.27 to +0.51/yr in every block, BTC-β≈0 confirming delta-neutrality
  held) but a naive daily funding>0 threshold churns turnover (~0.58/day) and 20bps round-trip cost
  eats it. `clears_ex_history=FALSE`. This is a risk premium with a real exchange-solvency/
  liquidation tail, not free money — labeled as such everywhere (ledger, AXIOM panel, docs).
- New: `src/crypto/carry_shadow.py` (shadow compute + JSONL ledger), `src/crypto/
  crypto_carry_live_gate.py` (structurally unarmed/unarmable — no SHIP_GATE.json, no exchange
  client exists in this repo), `scripts/run_crypto_carry_shadow.py` (zero-order driver).
- Wired `crypto_carry` into `KNOWN_LANES` (state_engine.py), `risk_monitor.sh`, `running_status.py`,
  loop-enforcement fixtures — defaults halted; regenerated `gate_manifest.json`. Fixed in passing: a
  pre-existing stale 3-lane assertion in `test_per_lane_halt_2026_07_02.py` (already broken before
  this session, confirmed via `git stash`).
- AXIOM: `GET /api/crypto_carry` + `CryptoCarryPanel.tsx` (SHADOW/halted badges + explicit "RISK
  PREMIUM · LIQUIDATION TAIL" badge/disclaimer). Also mounted the two pre-existing orphaned panels
  `CryptoMomentumPanel`/`TrackBPanel` — they were built but never reachable from any tab since the
  old "Risk" tab was retired; all three now live under Automation.
- **Verification**: 18 new tests (16 unit + 2 integration, real disk, no mocks) + 2 dashboard reader
  tests, all green; flake8 clean; tsc clean; risk_monitor.sh GREEN; loop-enforcement (9 tests) green.
  Independent Security Engineer: SAFE, 8/8 checks PASS. Independent Code Reviewer: no correctness
  bugs, regression math verified bit-identical to the frozen harness by direct algebra comparison
  (not just trusting the test); 3 non-blocking nits, no action needed.
- Real first ledger entry produced (integration test, real production ledger): asof 2026-05-31, 74
  longs, `orders_placed=0`, `broker=null`, `gross_leverage≈3.0` (vol-target cap hit).
- **Untouched**: oanda_fx stays halted; `oanda_environment="practice"` (config.py:742); no other
  lane's behavior changed (additive-only diffs confirmed by the security reviewer).

## P1 Headless Learning Supervisor — RL-weight-sync folded into offline_learning_cycle.py (2026-07-06/07)

Executed `docs/ENGINEERING_BRAIN.md`'s P1 item: "self-heal produces adjustments only the dead
TUI can consume." Mapped from disk first (grep, not assumption):

- `ExecutionManager.apply_pending_rl_weight_updates` (execution.py:5833, commit 51b85bf) — the
  RL agent-weight-sync anchor — had exactly ONE production caller: `embedded_scanner.py:1271`
  (the TUI). **Corrects the 2026-07-04 NOTES claim** that it was already wired into an offline
  batch job — never true; `offline_learning_cycle.py` only ran the separate
  `RiskCalibrationLearner`, never this method.
- The brain doc's "3 divergent post-trade paths" (`sync_closed_trades_rl`, `post_trade_loop`,
  `trend_journal_sync`) turned out, on inspection, to be **already correctly non-overlapping by
  design** — `post_trade_loop.py` and `execution.py:4887` both say "does NOT update agent
  weights, Parity Q5"; `trend_journal_sync.py` has a HARD RULE it must never call
  `update_weights_from_outcome` (no agent verdicts exist for trend-lane trades — would be a
  category error). No merge performed; forcing one would break `trend_journal_sync`'s safety
  contract. The real gap was the missing headless caller, not path duplication.
- `run_tier7_loop.py` (headless, alive) already consumes self-heal adjustments into
  `config_overlay.json` every 30s (`TIER7_CONSUME_ADJUSTMENTS=1`, shipped 2026-07-03).
  `apply_overlay` (the live-config consumer) is only called from `engine.py` (not the live
  driver) and the TUI. `config_adjustments.json` pending queue is currently EMPTY (0/0) — not a
  live-impacting gap today; inventing a consumer would mean resurrecting the halted FX scanner
  (out of scope). Left as an honestly-reported residual, not force-fixed.

**Fix**: `scripts/offline_learning_cycle.py` gained `_run_rl_weight_sync()` in `run_cycle()` —
now the single headless entrypoint for BOTH the calibration learner AND the RL agent-weight
sync. Idempotent via the shared `rl_weights_applied` flag (TUI can't double-score later).

**Bug self-caught pre-ship**: first draft chdir'd via `_DATA_ROOT` (bound once at import from
`OLC_DATA_ROOT` env). Existing tests sandbox via `monkeypatch.setattr(olc, "JOURNAL_PATH", ...)`,
which doesn't touch `_DATA_ROOT` — would have silently mutated the REAL production journal in
any in-process test reaching past the market-open gate. Fixed: chdir target is
`JOURNAL_PATH.parent.parent`, read at call time.

**Verification**: 5 new tests (`tests/test_headless_learning_supervisor_2026_07_06.py`) +
40 pre-existing (offline_learning_cycle, RL-backfill, config_overlay, embedded_scanner-reload)
= 45 green, flake8 clean. Real `--force` smoke run against production: drained 9 real pending
retrain markers to `_processed/`, `rl_weight_sync: {applied:0}` (correct steady-state), journal
`git status --porcelain` clean after, halt/lanes unchanged.

**NOT done, operator decision needed**: `launchctl bootstrap` for `com.buddy.learning_loop`
(the actual "runs on a schedule" step) was BLOCKED by the permission classifier ("Unauthorized
Persistence" — installing a new standing launchd agent needs explicit activation, not just
"daemonizable"). Plist/README updated to reflect scope; 3-line activation command unchanged,
documented in `scripts/axiom_launchd/README.md`.

## AXIOM Agent Runtime foundation — tool registry + policy engine + audit (2026-07-06, scaffolding-only task)

Built `src/agent_runtime/` (policy.py, audit.py, tools/) — the safety-critical base a future
resident AI trading operator will sit on top of. **No LLM reasoning loop yet; nothing armed,
unhalted, or traded.** Commit `046bf13` on `ralph/equity-harvester-bot`.

- **Policy engine** (`policy.py`): 3 tiers — OPERATIONAL (preflight+audit, allowed), DEESCALATION
  (autonomous, structurally risk-decreasing only — `halt_lane` hardcodes `set_halted(True, ...)`
  with no value param to flip; `reduce_gross_leverage` refuses any non-strict-decrease), ESCALATION
  (`unhalt_lane`, `arm_live_gate`, `increase_gross_leverage`, `promote_model`,
  `enable_new_exposure`, `change_strategy_or_code` — NEVER autonomous, `ActionSpec` structurally
  cannot carry an execute callable for this tier, and `PolicyEngine.submit()`'s escalation branch
  never reads `.execute` even if a spec were tampered post-construction via
  `object.__setattr__` — added a redundant submit-time check for that theoretical bypass).
  Fail-closed on unknown actions. Practice-pin re-derived from `ScannerConfig` every call, every tier.
- **Audit** (`audit.py`): extends the EXISTING `dashboard.server.control_safety.audit()` ->
  `trained_data/axiom/control_audit.jsonl` with `source="agent_runtime"` entries — one unified log,
  not a second file.
- **Tools** (`tools/`): 9 READ-ONLY wrappers (health check, gate health via a private-tmpfile
  verify_gate re-run that never clobbers the real verdict.json, tier7/self-heal status, per-lane
  halt state, OANDA account state, trade journal, agent weights, LESSONS/NOTES, shadow ledgers
  crypto_momentum/track_b/harvester) — every one calls an EXISTING function, all tagged
  OPERATIONAL, all routed through the same PolicyEngine so every tool call is preflighted+audited.
- **Verification**: 41 new no-mock tests (real disk via tmp_path) prove escalation is
  proposal-only, de-escalation can't increase risk, operational actions preflight+audit, unknown
  actions fail closed, no tool/action can trade/arm/unhalt. Independent Security Engineer verifier:
  PASS on all 8 structural checks (escalation boundary genuinely structural, not just tested;
  practice-pin/halt/no-LLM-in-hot-path all confirmed). Production `.claude/state.json` and
  `.claude/loop/verdict.json` confirmed byte-identical before/after (git diff clean).
- **Known minor gap, not fixed (no concurrent caller exists yet)**: `reduce_gross_leverage` has a
  TOCTOU window between reading the current override and writing the new one — fine today
  (single-writer), worth a lock once a real reasoning loop can submit concurrent proposals.
- **Next (follow-up tasks, not this one)**: the resident reasoning loop that actually calls these
  tools + submits actions through the policy engine, and the Activity mind-window panel.

## Closed-trade feedback loop fixed + market-closed continual-learning batch (2026-07-04, research/infra task)

Root cause found (NOT what the task brief assumed — corrected via direct disk read, honesty
protocol): the journal is NOT stale since mid-May in the "nothing writes to it" sense — it's
being written constantly. The break is a **dead-write collision between two independent
writers on the same `outcome` field**. `OutcomeBackfill._apply_closure` (src/scanner/automation/
outcome_backfill.py, US-605, 2026-04-25) and `TrendJournalSync` both stamp `entry["outcome"]`
(plain string) at boot/backfill time; `sync_closed_trades_rl`'s only "is this trade pending
RL-scoring?" check is `outcome is None`. Once either writer stamps ANY value, the entry looks
"already handled" and the real weight-updater (`ScannerAgentTeam.update_weights_from_outcome`)
never runs on it. Confirmed empirically: last trade to ever pass through the real path was
trade_id 1261, closed 2026-04-16 — every subsequent close (185/208 journal entries) has a
string outcome and zero agent-weight scoring. This exact defect was independently surfaced 3x
before (obs 1967 on 2026-05-12, 14959+15342 on 2026-07-02/03) and never fixed until now.
**Correction to the task's own premise**: `trained_data/retrain_requests/` markers ARE
unconsumed (12 piled up, confirmed) but `rl_position_sizer.zip`/`agent_weights.json` were NOT
8+ days stale — they were touched 2026-07-02 by an ad-hoc smoke run (120 samples, 256
timesteps), not the missing consumer. Both facts matter; only the marker-drain part of the
brief was accurate.

**Fix (item 1)**: `ExecutionManager.apply_pending_rl_weight_updates` (execution.py:5816) —
outcome-shape-agnostic (handles both the string and dict shapes), tracks its own
`rl_weights_applied` flag so it's idempotent and safe to call from both the live loop
(wired into `embedded_scanner.py._run_smart_loop`) and the offline batch job without
double-scoring. `sync_closed_trades_rl` also now sets the flag itself so the two never race.
Also fixed an adjacent pre-existing crash (`win_rate_by_pair` AttributeError on string-shaped
outcomes) found while in the same code — same schema-split root cause, one-line guard.

**New (items 2-5)**: `scripts/offline_learning_cycle.py` — market-closed batch job (FX-weekend
heuristic, `--force` for manual runs) that drains `retrain_requests/` markers (atomic move to
`_processed/`, never deleted) and walk-forward-gates an update to a NEW
`src/training/incremental/risk_calibration_learner.py` — a hand-rolled online (River-style,
`learn_one`/`predict_proba_one`) logistic regression scoped to RISK/EXECUTION/CALIBRATION
features ONLY (`ALLOWED_FEATURE_KEYS` whitelist: confidence, regime one-hot, rr_ratio, sl/tp
pips, mae/mfe, lane — enforced by `raise ValueError`, not `assert`, so it survives `-O`).
Deliberately NOT the `river` package (no new dependency) and deliberately NOT touching the
15-agent directional weights — per operator scope: don't re-fit a no-edge directional signal.
Gate: promote only if candidate's Brier score beats the incumbent's on a chronologically-later
holdout by a margin; `size_multiplier()` is capped `[0.5, 1.0]` at the algorithm level (can
only ask for LESS risk, never more) with an explicit NaN/inf guard (a real fragility a code
reviewer caught: `min(1.0, nan)==1.0` in Python is argument-order luck, not a designed
safeguard — now explicit). AXIOM: `/api/learning_loop` + `LearningLoopPanel.tsx` (Automation
tab). launchd: `com.buddy.learning_loop.plist` added to `scripts/axiom_launchd/` but
**deliberately kept OUT of `load.sh`'s LABELS array** (security reviewer's finding: bundling
it would silently auto-install on the next routine reload of the other 4 live daemons) —
README documents a separate explicit `launchctl bootstrap` activation step; **NOT loaded/
installed this session**.

**Verification**: 15 new tests (backfill sync idempotency + trend-lane exclusion + dict/string
outcome handling; learner feature whitelist + NaN safety + state round-trip; batch job market-
closed gating + marker drain + reproduce-then-resolve gate-rejects-worse-candidate +
gate-accepts-better-candidate + never-touches-halt-state, verified byte-identical fixture file;
AXIOM readout honest-empty-state), all green. Independent Code Reviewer + Security Engineer
(parallel, no shared context): both PASS. 2 real findings fixed (NaN clamp, assert→raise);
1 MEDIUM process-hygiene finding fixed (load.sh auto-enrollment, reverted). flake8 clean on
every new/changed file. `risk_monitor.sh` GREEN throughout.

**Near-miss during the session (logged for /evolve → LESSONS)**: ran `git stash` /
`git stash pop` mid-session purely to A/B-test whether a pre-existing test failure predated my
changes. The pop conflicted on `.claude/self_heal_action_budget.json` /
`self_heal_debounce.json` because the LIVE `com.buddy.tier7` daemon wrote fresh self-heal
activity (a real `reduce_risk_per_trade_pct` action, 2026-07-04T14:25:02Z) to those files
*while the stash was active* — the stash was kept, not dropped, and briefly ALL uncommitted
work (mine + everything else in the working tree) existed only inside `stash@{0}`. Recovered
cleanly (saved the live daemon's fresher runtime-state files aside, reset just those 2 to HEAD,
popped clean, restored the live files over the stale popped ones) — no data lost, but this
should never have been the tool for the job: `git show HEAD:<path>` or a scratch copy answers
"did this predate me" without touching the shared working tree of a repo with live daemons
writing to tracked files.

**Adversarial review round (operator: "send out review agent… I will not accept slop")** — a
fresh independent Code Reviewer ran the batch job against the REAL journal and found REAL defects
(not nits); all confirmed ones fixed with TDD (reproduce→resolve), commit after `8a70a3b`:
- **#1 (blocker)**: the batch job crashed `ModuleNotFoundError` on its documented/plist invocation
  (`python scripts/offline_learning_cycle.py`) — no `sys.path` insert; the plist would have died at
  import every scheduled run. Fixed (sys.path insert at module top; subprocess smoke test).
- **#2/#3 (blocker)**: the walk-forward gate was invalid on real data — 90% of the journal is
  trend-lane bare-string entries with all-None features that collapsed the holdout to one repeated
  vector, and on the real 1-win/17-loss scoreable set a majority-class "always predict loss" model
  beat the 0.5 baseline and got PROMOTED. Fixed: `is_calibration_scoreable()` excludes trend-lane/
  featureless entries from the population (mirrors rl_eligible convention), + a min-minority-class
  holdout guard (`MIN_MINORITY_HOLDOUT=2`) refuses single-class/severely-imbalanced holdouts.
  Verified on a scratch copy of the real journal: now `insufficient_holdout_signal`, promotes
  NOTHING (correct — 1 positive example can't calibrate).
- **#6 (oversold)**: `size_multiplier` is consumed by NO live sizing path. Disclosed honestly —
  module + method docstrings say SHADOW/advisory, `read_learning_loop` returns
  `consumed_by_live_sizing:false`, panel shows a "SHADOW — advisory only" badge. NOT wired to live
  sizing (hot path = operator-gated).
- **#4/#5 (atomicity)**: sync set `rl_weights_applied` in the collection block BEFORE weights moved
  (strand-on-failure); apply_pending wrote flags once at batch-end (crash → whole-batch
  double-count). Fixed: flag set per-entry AFTER the weight update succeeds + journal re-persisted
  post-loop (sync) / per-entry (apply_pending); crash now costs at most one bounded/decaying
  double-count, never a permanent strand. 3 no-mock tests induce a real per-entry failure (malformed
  `agent_reasons=[42]`) and assert the failed entry stays retryable.
- **#9 (cursor)**: `str > cursor` dropped equal-timestamp entries and mis-sorted `Z` vs `+00:00`.
  Fixed: (parsed-datetime, trade_id) tuple watermark; test proves an equal-timestamp twin isn't lost.
- Reviewer CONFIRMED the safety rules held: no OANDA/state.json/env, halt only stricter, size cap
  uncrackable, live wiring real (not orphaned), reject-test non-tautological.
- New tests: `tests/test_offline_learning_cycle_review_fixes_2026_07_04.py` (8),
  `tests/test_rl_backfill_atomicity_2026_07_04.py` (3). 158 targeted tests green, flake8 clean,
  tsc clean, risk_monitor GREEN. UNCOMMITTED as of this note — commit next.

Nothing in this session touched `.claude/state.json`, OANDA order endpoints, `oanda_environment`,
or any ARM path. Committed `51b85bf`, pushed. `/evolve` folded the learnings (operator-approved):
**L-024** (never `git stash` for exploratory checks on a repo with live daemons) added to LESSONS;
the dead-write-collision meta-pattern (2 writers sharing one `is-this-done?` guard field) added as
a bullet under improvement.md "Live Wiring Verification Gates" (now its 4th confirmed observation).

## Approved items SHIPPED — learning loop closed (2026-07-03T19:30-20:15Z, operator: "Approved on all accounts")

Commit `4f808d8` (pushed; separate verifier PASS at live-trading bar).
- **Item 1 (execution.py rl filter): implemented by a CONCURRENT session** (rl_eligible pending
  filter + comment, uncommitted in their tree) — deliberately NOT committed here; theirs to land.
- **Item 2 SHIPPED**: engine.py init applies the durable overlay AFTER apply_profile (ordering
  test-pinned); TIER7_CONSUME_ADJUSTMENTS default now **1** — supervisor consumes approved
  adjustments headlessly into `.claude/config_overlay.json`. **VERIFIER MODEL CORRECTION relayed
  to operator: self_heal writes SELF-approved entries directly into config_adjustments history
  (adjustment_approver's "sole writer" docstring is wrong; self_heal.py:1255) — consumption ON
  makes self-heal's autonomous adjustments DURABLE with no per-entry human approval.** Bounded
  vocabulary + evidence/debounce/budget + 4-layer PROTECTED_FIELDS rail hold; revert lever:
  TIER7_CONSUME_ADJUSTMENTS=0.
- **Item 3 SHIPPED**: min_samples_for_retrain 50→100 (entry floor aligned with the eval gate's
  20-sample holdout; gate floor stays test-exercised via pinned 60).
- **Rode along (attributed)**: concurrent handover session's supervisor singleton flock
  (launchd+nohup double-writer fix; verifier: correct, released-before-execv, denied instance
  exits clean). Supervisor now launchd-managed (PID 88167); trend lane PID 14533 on current code.
- **Follow-up chip RESOLVED 2026-07-03T20:45Z**: TUI `_reload_config_now`
  (src/tui/embedded_scanner.py:954) was raw-applying adjustments to memory only
  (restart-lost). Fixed: now calls `consume_approved_adjustments(root)` +
  `apply_overlay(config, root)`, same durable path as the headless supervisor.
  4 no-mock regression tests (`tests/test_embedded_scanner_reload_overlay_2026_07_03.py`)
  prove durable-write + simulated-restart survival; independent Code Reviewer PASS
  (confirmed PROTECTED_FIELDS/oanda_environment untouched at all 3 guard layers,
  the every-cycle raw-apply block's `old_value==new_value` short-circuit prevents
  drift, and the applied_ids RMW race is genuinely benign same-value double-write
  — reproduced directly, no flock needed). Not yet committed. Known
  interactive-only clobber remains: config_screen profile re-apply overrides
  overlay values until next restart.

## Self-heal degraded-loop fix + P0 trio + P1 overlay (2026-07-03T05:15-15:00Z, operator: "patch him as if patching yourself")

Four commits, all pushed, each separately verifier-PASSed: `a5b93c3` (degraded loop), `a46e01a`
(P0 trio), `917ee55`+`b5cf510` (P1 overlay + test fixup).
- **Degraded loop DEAD (live-verified)**: was `degraded actions=2` every 30s forever. Causes: prose
  ("Retrain core models: train-joint…" — DEPRECATED pipeline for the RETIRED L-016 lane) fed to the
  action executor → unknown_action_string every tick; frozen evidence (halted ⇒ journal tail static)
  re-firing reduce_risk after every debounce window (burned the 10/day budget); suppression
  reported as "degraded". Fixed: recommendations channel (doctrine-correct text, surfaces in
  maintenance report), evidence-fingerprint debounce (v2 schema {ts,evidence}, legacy-compatible),
  status honesty ("suppressed"). Live tick now: `self_heal=suppressed n_applied=0` + L-016 rec.
  Also fixed test self-poisoning (ACTION_BUDGET_PATH was the one un-isolated fixture path).
- **P0**: online_retrainer eval gate (fail-closed: temporal 20% holdout, min 20 samples, MSE vs
  existing +5% tol, degenerate refusal; consequence: needs ≥100 buffer samples to ship);
  trend_journal_sync COMMITTED after focused review (3 real bugs fixed: wrong-shape clobber,
  RMW lock, partial-close P&L — outcome now from realized_pl_total) + enriched records
  (entry/close spread, half-spread costs, financing, atr_stop_distance; null-when-unavailable);
  alert routing map (every active alert names owner+mechanism, unknown → UNROUTED-escalate).
  Appendix-A UNVERIFIEDs RESOLVED: MetaManager WIRED-LIVE (orchestrator :451/:1507);
  regime_quantiles threaded end-to-end (real values in quarantined artifact).
- **P1 v1**: `config_overlay.py` — durable overlay closes producer-alive/consumer-dead AND
  restart-loss (consume moves approved adjustments → .claude/config_overlay.json with provenance,
  crash-safe ordering; apply_overlay replays onto fresh ScannerConfig). **Consumption OPT-IN
  (`TIER7_CONSUME_ADJUSTMENTS=1`, default OFF; running: NO)** — flipping it + the one-line engine
  seam (`apply_overlay(config)` AFTER `apply_profile`) = OPERATOR wiring decision. **Hard-NO rail
  added in depth** (found: field-membership validation alone would let an approved
  oanda_environment adjustment propagate): PROTECTED_FIELDS refused at proposal/approval
  validation + overlay consume + overlay apply + ConfigAdjuster.apply itself; single test drives
  all layers incl. tampered overlay → config stays practice.
- **OPERATOR DECISIONS SURFACED (not acted)**: (1) execution.py:5330 pending filter needs
  `lane != "trend"` / rl_eligible guard (regime:None AttributeError at :5469/:5483 can abort an
  RL-sync batch) — HOT PATH, needs explicit approval; 1-line change, verifier-confirmed real.
  (2) Flip TIER7_CONSUME_ADJUSTMENTS + wire the engine overlay seam (together). (3) Retrainer
  floor: 50-99-sample retrains now always refused — raise min_samples_for_retrain to 100 for a
  clearer refusal message? (4) legacy 179 trend entries carry bare regime:None (inert today).
- **Honesty note (L-018 self-report)**: `917ee55` was committed with 1 failing test because
  `pytest | tail` masked the exit code (the documented pipeline-masking trap); caught immediately
  post-push, fixed in `b5cf510` (test asserted refusal LAYER, not outcome; defense had gotten
  stronger). Gating runs now use pipefail.

## Autonomy-layer fix — SHIPPED + LIVE (2026-07-03T04:30-05:10Z, operator-directed)

Operator: "autonomy layer is fucked… won't pick up any changes or report maintenance/diagnostics
properly." Root causes fixed (commit `5806a5e`, pushed; separate Code Reviewer verifier: PASS):
- **Code pickup**: new `src/scanner/automation/code_freshness.py` (git-HEAD + watched-mtime,
  2-poll debounce, fail-open toward no-restart). tier7 supervisor os.execv's itself on change;
  trend lane clean-exits BETWEEN cycles (never mid-order) for launchd KeepAlive respawn, polled
  ~5s inside the cadence sleep (verifier finding: was up-to-2h latency).
- **Honest beacon (L-017)**: supervisor now writes `scanner_alive=false` + additive
  `writer/supervisor_alive` keys; never clobbers a fresh foreign (TUI) heartbeat
  (`_foreign_writer_owns_heartbeat`, 25s window vs TUI's 10s cadence). `write_heartbeat` gained
  additive-only `extra` (cannot override core schema — 2026-05-12 schema-mismatch incident).
- **Maintenance/diagnostics**: new `maintenance_report.py` (stdlib-only, READ-ONLY) → atomic
  `.claude/maintenance_report.json` per tick + embedded in tier7_state for AXIOM: adjustment
  backlog (identity mirrors ConfigAdjuster exactly), unacked alerts, per-lane halt+artifact
  freshness, self-heal budget, `needs_attention` summary. Deliberately does NOT consume
  adjustments (would steal `applied_ids` from the real scanner — dead-consumer fix is P1's
  headless learning supervisor with a durable overlay, not headless apply).
- **RUNNING (verified from ps/disk)**: supervisor PID 16644+ (nohup; fresh code; first tick
  surfaced "1 unacknowledged alert"); trend lane respawned via launchd on final code, correctly
  `REFUSED halt=True orders_placed=0`. Equity harvester (96555) untouched. 75/75 tests,
  verify_gate 28/28 PASS, risk_monitor GREEN, flake8 clean.
- **Open follow-ups**: (a) `com.buddy.tier7` launchd job dies EX_CONFIG with zero output (TCC on
  ~/Documents suspected) — background task `task_185f1921`; supervisor runs nohup meanwhile (no
  KeepAlive resilience). (b) `heartbeat_watchdog.sh` = dead code (targets unloaded
  com.buddy.trader). (c) untracked `trend_journal_sync.py` (other session) needs its own review
  before commit — verifier grep says no order path, `rl_eligible=False`. (d) self-heal loops
  `degraded actions=2` every tick — visible in tier7_state.recent_events, not yet diagnosed.

## Training-architecture audit + modernization plan (2026-07-03T01:30-04:45Z)

Operator-directed 3-phase audit; deliverable `docs/training-architecture-audit-2026-07-03.md`
(commit 8e34e03, pushed). Four read-only specialist passes (code map / data inventory /
retrain-gates / control-flow), every running-status claim re-derived per L-017.
- **Coin-flip diagnosis: already closed, documented, not re-litigated** (L-001 leak + broken gap
  gate + scaler double-fit all fixed → 52% = market wall, L-022). Modernization targets the
  RUNTIME LEARNING WIRING instead: (1) all FX learning loops welded to the TUI process (dormant
  now; heartbeat `scanner_alive:true` overstates), (2) live self-heal PRODUCES config adjustments
  only the dormant TUI can CONSUME (dead-write class, process level), (3) online_retrainer has NO
  eval gate (HARD_MAX_GAP only at Tier-1), (4) 3 divergent post-trade paths + duplicated
  alert/calibrator modules, (5) drift detection logs-only (active unack'd 7-loss WARNING in
  alert_state.json), (6) only 18/205 journal trades carry full RL context, (7) no tick store
  (tick_capture.py exists, `trained_data/ticks` absent). Roadmap: P0 truth/safety patches → P1
  gated-harness library + single feedback path + headless learning supervisor → P2 operator
  levers (tick capture, paid data) → P3 Lane-Contract unification. UNVERIFIED items flagged in
  doc Appendix A (MetaManager live wiring, regime_quantiles threading) — queued as P0 checks.
- **Found+fixed in passing: verify_gate false-FAIL** — `_halt_guard_violation` only matched the
  old `if ...get_halted():` shape; the 2026-07-02 hardening moved the guard to the STRONGER
  fail-closed `_halted = get_halted_strict()` + `if _halted: return` form. Matcher now accepts
  both forms, inverted guards still rejected; +2 enforcement fixtures (form-b accept/inverted
  reject) → 111/111; manifest regenerated (commit aa5e9fd, pushed). **Separate Code Reviewer
  verifier: PASS** — empirically no weaker than old matcher on adversarial cases; non-blocking
  note: docstring could enumerate the reassignment-between-assign-and-branch vector.
- Gates at close: verify_gate PASS (28 checks), risk_monitor GREEN (per-lane visible),
  enforcement suite 111/111. Session touched ONLY: the doc, verify_gate.py, gate_manifest.json,
  test_loop_enforcement.py, this file. No hot path, no state.json write, practice pin untouched.

## SEC EDGAR value + accruals PIT factors — BOTH FAIL THE GATE (2026-07-02T08:23-13:25Z)

Operator-directed structural-unlock task: extend the true-PIT SEC EDGAR fundamentals pipeline
(quality/profitability already tested negative 2026-06-25) with value + accruals, the two
untested classic factors. Pre-reg `docs/experiment-sec-edgar-value-accruals-2026-07-02.md`
(committed `2f50d25` BEFORE any result existed). Result `trained_data/backtests/
edgar_value_accruals_bakeoff.json`, commit `83a2004`.
- **New ingestion**: `src/equity/edgar_fundamentals.py` extended with `CashFlowFromOps`/
  `Assets`/`SharesOutstanding` (the last via the `dei` cover-page tag — empirically verified
  against live EDGAR on AAPL/MSFT/JPM that its `end` date is NOT fiscal year-end; fixed by
  keying on the fact's own `filed` date, never an `end`-join). New module `src/equity/
  value_data.py` (PIT fundamentals x PIT price join — a new mechanism vs the pure-ratio
  quality panel). Accruals (Sloan 1996) reuses `build_quality_panel` unmodified.
- **RESULT: both FAIL the canonical ship gate AND fail DSR/Bonferroni significance** (N_TRIALS=3
  counting the prior quality trial, alpha=0.0167). Accruals: clean decisive negative (full
  margin -0.52, OOS margin -0.441, maxDD 66%, DSR 0.15). Value: OOS Sharpe margin nominally
  +0.21 vs EW but NOT significant (DSR 0.66 < 0.95 bar, bootstrap p=0.10 > 0.0167), and fails
  the gate on both maxDD (36.6% full-window-corrected, was falsely 17.9% before the bugfix
  below) and OOS history-length.
- **Two real bugs caught by independent verification, fixed before finalizing (L-018 process
  working as designed, not a lie — a genuine catch-and-fix):** (1) `quality_data.
  _raw_quality_frame` hardcoded `QUALITY_COMPONENTS` regardless of the `components=` override
  — harmless to the already-committed quality result (every existing caller only ever passed
  QUALITY_COMPONENTS subsets) but would have silently all-NaN'd the new accruals field; fixed +
  regression-tested. (2) `value_data._pit_raw_panel` forward-filled `shares_outstanding` with
  no staleness bound — >=12 large-caps (incl. Berkshire Hathaway, `dei` tag dormant since 2011)
  carried a years-stale share count against a live price, inflating `book_to_market` ~1500x for
  those names. Fixed with a 730-day forward-fill staleness cap (NaN beyond it, never
  fabricated); re-ran from cached data (no EDGAR re-fetch) — headline verdict unchanged
  (still gate-FAIL) but the OOS drawdown number corrected from an artificially tame 17.9% to a
  real 36.6%.
- **Survivorship caveat surfaced by the verifier, now labelled in the result JSON**: SEC's
  `company_tickers.json` CIK map is CURRENT-tickers-only — a delisted/acquired name (confirmed:
  TWX) loses CIK resolvability and drops from the fundamentals-covered universe even though it
  stays correctly present in the price panel (`since_removed_included=110` counts INDEX-
  membership removals, not fundamentals-covered delistings — a joint-panel gap, separate from
  the pre-existing price-side yfinance gap).
- **Independent verifier (separate Model QA Specialist, no shared context): CONCERNS →
  addressed.** Confirmed PIT correctness (re-derived the dei finding independently on a THIRD
  company, JPM), multiple-testing math (byte-identical `deflated_sr`/`block_bootstrap_p` reuse
  from `experiment_crypto_xs_signals.py`), cost assumptions (unchanged from the quality
  baseline), hot-path isolation (zero hits on execute_trade/StateEngine/oanda_environment/
  halted). Bug #2 above is exactly what the verifier caught and recommended; applied verbatim.
  Bug #2's fix itself was NOT independently re-verified by a fresh agent (self-verified: tests
  pass, flake8 clean, the diagnosed symptom — value composite == earnings-yield-only OOS —
  resolved as predicted). Confidence: HIGH on the fix's correctness, MEDIUM-HIGH on full
  freshness (no second independent pass).
- **Verdict**: extends L-022 (no free-data return-alpha) to true-PIT fundamentals specifically
  — 3-for-3 SEC-EDGAR-PIT-factor trials (quality, value, accruals) now fail on this universe/
  window. The structural unlock (genuine filing-date PIT + survivorship-aware universe) is real
  infrastructure and now exists for future hypothesis-testing (e.g. brain-loop factor-blend or
  regime-conditioning tests), but classic single-factor tilts on it do not clear the gate.
  Nothing live/halt/env touched — pure research code under `src/equity/` + `scripts/`.

## Equity + brain shadow unhalt (2026-07-02T00:49-01:02Z)

Operator-directed: unhalt ONLY `equity` + `brain` lanes, keep `oanda_fx` halted, no LiveGate arm.
Full writeup incl. the stale-process-code timeline analysis: this session's transcript (not yet a
doc — worth a `docs/` writeup if this pattern recurs).

- **Preconditions verified from disk** (all three, before touching state): FX H1 direction models
  (USD_JPY/USD_CAD/AUD_USD) have no pair-root `transformer_direction.keras` — only under
  `_quarantine/`; `gates.py:468` `pair_dir = self.base_model_dir / instrument` confirms the loader
  scope excludes it. `trained_data/equity/live_gate_state.json` ABSENT → `LiveGateState` default
  `armed=False` (`live_gate.py:293`). `oanda_environment="practice"` (`config.py`), `.env.local`
  `OANDA_ENVIRONMENT=practice`, no live endpoint reachable under current env config.
- **State write — single atomic write, NOT the two-call `set_halted(False)` → `set_halted(True,
  lane="oanda_fx")` sequence**, because `run_oanda_trend.py --loop 3600` (PID 62414, launchd
  `com.buddy.trend`) was ALREADY LIVE polling `.claude/state.json` when this started. A two-call
  sequence would have passed through a real (if brief) intermediate state where oanda_fx read
  unhalted. Wrote `halted=false` + `halted_lanes={oanda_fx:true, equity:false, brain:false}` in one
  `tmp`+`os.replace`, matching `StateEngine._atomic_write`'s exact mechanism. Verified via two
  independent readers (`StateEngine.get_halted(lane=...)` and `decision_gate._lane_halted(...)`) —
  both agree on all three lanes.
- **Stale-code risk explicitly checked (operator asked directly) — closed, not just asserted.**
  `run_oanda_trend.py` does NOT use `execution.py`'s halt guard; it imports `src.equity.oanda_trend`,
  which already called `_lane_halted(root, "oanda_fx")` (line 487) — file mtime 00:37:10, PID 62414
  started 00:41:48, i.e. the process imported the ALREADY-lane-aware file. `trained_data/axiom/
  trend_loop.out` (the launchd daemon's log) confirms every cycle through 00:41:50 logged
  `REFUSED — halt=True readable=True`; no cycle has run since (next ~01:41:48). No stale-bytecode
  path existed. If this pattern (live daemon + a state-schema migration) recurs, check the DAEMON'S
  actual import path first — it is not always the same file `execution.py`'s guard is in.
- **Equity harvester**: `scripts/run_equity_harvester.py --broker shadow --loop 3600` running
  (PID 78084, `nohup`, `logs/equity_harvester_shadow.log`). First cycles: `ran=False reason=abstain
  orders=0` (universe snapshot 8d stale vs 7d freshness window — honest abstain, not a bug).
  **Report as the 0.740 full-sample / 0.355 OOS BETA sleeve number** (the wide PIT universe,
  2026-07-01 independent audit), NOT the 0.908/0.92 curated-universe headline in `SHIP_GATE.json`.
- **Brain loop**: `scripts/run_brain_loop.py --loop --max-cycles 8760 --interval 3600` running
  (PID 79486, `python3 -u` for unbuffered logs, `logs/brain_loop.log`). First cycle: `halted=false`,
  `breach_derisked=false`, `decision=abstain (no_data)`, no hypothesis/promotion/arm.
- **Independent verifier (separate Security Engineer agent, no shared context): SAFE, 8/8 PASS.**
  Independently rediscovered the same `oanda_trend.py:487-490` gate. One non-blocking gap flagged +
  spun off as a follow-up task: `risk_monitor.sh` checks only the global `halted` flag, not
  `halted_lanes` — a monitoring blind spot (the actual trade-blocking code paths are unaffected and
  already correct); worth closing so the tripwire has per-lane coverage too.
- **Remaining operator step (not done, not proposed to be done automatically):** `pip install
  ib_insync`, start IB Gateway/TWS on `127.0.0.1:7497` (paper), IBKR paper login, then explicit
  `LiveGate.arm()` with a typed `"LIVE"` token (TUI `ModeConfirmModal` or an operator arm script) to
  go from shadow to live paper fills. `oanda_fx` stays halted; global blanket unhalt not touched.

## AXIOM dashboard real-data population (2026-07-02)

Added, all read-only / additive (did NOT touch the concurrent per-lane-halt session's owned files:
`state_engine.py`, `dashboard/server/control.py`'s halt/unhalt logic, or `.claude/state.json`
schema):
- `dashboard/server/data_sources.py`: `read_lane_status()` (new `/api/lanes`, always-on per-lane
  halt display), enriched `read_equity_sleeve()` (LiveGate armed/live-vs-shadow, SHIP_GATE, cycle
  ledger tail, live `decide_cycle()` verdict), `read_brain_loop()` (new `/api/brain_loop` — honest
  empty since `src/brain_loop/` hasn't run in prod yet).
- Frontend: `EquityHarvesterPanel.tsx`, `BrainLoopPanel.tsx` (new), `HealthPanel.tsx` wired into the
  Risk tab (was built 2026-06-29 but never rendered anywhere — found orphaned), per-lane
  halt/unhalt buttons added to `ControlPanel.tsx` (backend already supported `params.lane`, wasn't
  surfaced), audit log now shows actor. `dashboard/README.md` corrected: TP/SL bracket fields were
  actually already wired (`src/brokers/oanda_v20.py:_position_brackets`) — README said "pending",
  was stale.
- Verified via curl (real data, no fabrication) + `tsc --noEmit` + `flake8` clean; could NOT get
  browser screenshots — `com.axiom.web`/`com.axiom.api` launchd daemons were mid-use by a live
  operator session (`trained_data/axiom/launchd-web.log` showed continuous real polling); permission
  layer correctly blocked `launchctl bootout` to commandeer them, so visual verification is
  API/log-level only, not pixel-level. Nothing armed/halted/traded; practice pin untouched.
- Found + flagged (not fixed, out of scope): `.claude/tools/risk_monitor.sh` L-004 check greps
  literal `get_halted()` (empty parens) — stale against the just-landed per-lane
  `get_halted(lane=...)` signature, causing a permanent false ALARM on Stop-hook. Spun off as
  background task `task_89450a16`.

---


## Track A equity-beta harvester — independent audit (2026-07-01)

Full writeup: `docs/equity-harvester-verdict-2026-07-01-independent-audit.md`. Three independent
subagents (Model QA / Data Engineer / Software Architect), no shared context, all disk-verified.
- **Verdict: MIXED, not a clean PASS.** The cited 0.908 Sharpe (`trained_data/backtests/
  SHIP_GATE.json`) is mechanically real (reproduced live twice, 0.907/0.921, lookahead-clean, costs
  genuinely deducted) — but it's on a curated 20-name mega-cap-tilted universe. The SAME
  construction on the repo's own survivorship-corrected wide universe (`pit_quality_bakeoff.json`,
  Wikipedia PIT S&P reconstruction) scores **0.740 full-sample / 0.355 OOS, gate FAIL** with
  *cheaper* costs. Not leakage/fabrication — an economically legible concentration effect — but the
  headline number is universe-dependent and the more defensible one fails.
- Secondary: shipped overlay params (vol12%/dd10-20) don't trace to the one grid-search artifact on
  disk and actually fail the gate on the book that grid tested.
- Staging: correctly halt-respecting (reads same `.claude/state.json` flag, fail-closed), ship-gate
  hash-keyed kill switch + LiveGate exist, but nothing is currently running/armed, no daemon exists,
  and the H1 `--broker ibkr-paper` path bypasses `LiveGate.arm()` entirely (wiring gap, not just an
  "arm it" step). IBKR-only execution, zero OANDA coupling.
- **Before any further US-006+ build-out leans on 0.908**: surface the 0.740/0.355 numbers
  alongside it; re-run the vol/dd grid on the actual single-stock universe.

## Crypto edge-hunt (2026-06-29, operator-approved new direction) — research/backtest ONLY

Pivoted the gated harness into crypto. **Data layer built + committed (`9c14a6f`):**
`src/crypto/data_layer.py` — Binance static dumps (funding+klines, 2020-01→now, 733 USDT perps
incl. delisted → survivorship-aware; trading API geo-blocked here but the static dump bucket is NOT)
+ OKX/Hyperliquid cross-checks. Cache → `crypto_cache/` (gitignored). Pre-registration (frozen,
committed before results): `docs/experiment-crypto-edge-hunt-2026-06-29.md`.

**CAMPAIGN CONCLUSION: honest NEGATIVE across all 3 (H1/H2/H3) — all verifier-CONFIRMED
TRUSTWORTHY.** No cost-surviving, OOS-confirmed, market-neutral RETURN alpha at this scale.
Scripts: `experiment_crypto_funding_carry.py` (H1), `experiment_crypto_xs_signals.py` (H2/H3).
- **H1 funding carry:** OOS net Sharpe **−0.25**, market-neutral (β −0.04, NOT short-beta — my
  "mostly short-beta" prior was WRONG). Carry real (+0.32/yr, Sharpe ~17 drip) but net dies to
  price adverse-selection (high-funding = momentum; flipped +0.41→−0.21/yr IS→OOS) + ~0.16/yr cost.
- **H2 XS 14d momentum = the one lead.** OOS net Sharpe **+0.75**, market-neutral (β −0.07),
  price-driven (+0.31/yr). FAILS gate: maxDD −49%, not significant (DSR 0.62, p 0.10), cost-fragile
  (+0.09 at 2×). A fresh-pre-registered refinement (vol-target / DD-control / lower turnover) is the
  only forward lever worth trying.
- **H3 contrarian order-flow:** OOS net Sharpe **−4.45** — uneconomical (~50%/yr turnover cost,
  negative price leg). Decisive negative.
- **Structural walls:** effective-N ≈ 3.9 (crypto cross-section ≈ one factor; breadth illusory);
  history ~6.5y < 10y (never an unqualified ship). Mirrors FX/equity verdict — gated harness did its
  job (full-sample Sharpe would have lied; OOS + multiple-testing + DD gate told the truth).
Verdict doc: `docs/experiment-crypto-edge-hunt-2026-06-29.md`. Paper/research only, no live execution.

## Crypto edge-hunt ROUND 2 (2026-06-29) — literature sweep + infra-stress + H4/H5, research-only

Operator's sharp question: findable edge, or is OUR INFRA suppressing it? Answer (all verifier-
confirmed): **BOTH, cleanly split.** Docs: `docs/experiment-crypto-edge-hunt-round2-2026-06-29.md`
(frozen pre-reg committed b09b61a BEFORE results; results 4ed728d; verifier verdict appended). Scripts:
`experiment_crypto_h2_infra_stress.py` (H2 robustness decomposition, regression-checked == verified
harness) + `experiment_crypto_round2.py` (H4/H5).
- **LITERATURE (5 sourced sub-reports):** our daily-bar negatives are FULLY consistent w/ the published
  record — no free-data small-operator return-ALPHA was missed. The one surviving liquid edge is
  trend/TS-momentum AS A RISK PREMIUM. Our 10bps cost is about-right-to-conservative (NOT too harsh);
  the load-bearing problem is TURNOVER not the per-trade rate. Han-Kang-Ryu 2024: crypto TS-momentum
  survives realistic cost, XS dies — exactly our Round-1 result.
- **INFRA-STRESS (is it us?): PARTIALLY YES.** Vol-target + weekly rebalance fix H2's DD (−49%→−16%) +
  cost-fragility (turnover 0.42→0.13); realistic 4-6bps cost gives OOS Sharpe +1.0-1.15. BUT the
  significance wall (DSR/p) survives EVERY config — that's eff-N≈2.82 (verifier-rederived; cross-section
  ≈ one factor) + ~6.5y history, STRUCTURAL to the asset class, NOT infra.
- **H5 TS-trend = BEST-SHAPED RESULT OF THE CAMPAIGN, still NOT alpha.** Pre-registered (not dredged):
  OOS +1.13 / IS +0.91 (no sign-flip), maxDD −0.162, β −0.073 (market-neutral), turnover 0.029. Clears
  4/5 ex-history gate criteria; FAILS ONLY significance (DSR 0.50<0.95; boot p 0.031 fails even lenient
  N=3 Bonferroni 0.0167) + 10y-history. Risk-control axis: BEATS buy-hold BTC on Sharpe (1.13 vs 0.72)
  AND drawdown (−0.16 vs −0.50) — drawdown-controlled risk-premium harvesting (same property as multi-
  asset trend, here beating passive). H4 (infra-corrected XS) = cleaner negative (IS sign-flip → not
  OOS-confirmed + significance fail). **Verifier (separate Code Reviewer): BOTH TRUSTWORTHY, leakage-
  free, no L-018 trigger; H5 correctly reported as a non-cleared risk-control keeper, NOT "an edge".**
  Net: no verified return-alpha anywhere; crypto TS-trend is the strongest risk-control finding; path to
  a SIGNIFICANT crypto edge needs more independent history (unavailable). Paper/research only.

## Edge-hunt ROUNDS 3 + 4 (2026-06-29/30) — no-spend frontier DEFINITIVELY CLOSED, research-only

Docs: `docs/experiment-edge-hunt-round3-2026-06-29.md`, `docs/experiment-edge-hunt-round4-2026-06-30.md`.
Scripts: `experiment_edge_round3_leadA.py` (intraday), `experiment_edge_round3_leadB.py` (breadth),
`experiment_edge_round3_leadB_lagprobe.py`, `experiment_edge_round4.py` (breadth+history expansion).
- **R3 Lead A (1h intraday crypto): decisive NEGATIVE** — A1 hourly TS-mom + A2 first6→last6 both
  −1.5 to −4.8 OOS at 5/9bps; higher frequency surfaces no retail edge (latency-gated per lit).
- **R3 Lead B (broad 37-asset cross-asset trend): verified NEAR-MISS** — eff-N 2.82→13.07,
  full-sig clears (DSR 0.987), OOS passes p (0.002) but misses ONLY DSR-OOS (0.843<0.95). Two
  independent verifiers (my lag-probe + separate Code Reviewer): CAUSAL, leakage-free, survivorship
  CONSERVATIVE (removing crypto sleeve IMPROVES it). Risk-control NOT alpha (loses to passive). Fixed
  a verifier-caught pre-reg deviation (12%→frozen 10% vol-target); verdict unchanged (DSR-OOS binding).
- **R4 (ONE pre-registered breadth+history expansion, 59-asset 1928-2026): the significance lever
  WORKED but the gate MOVED.** OOS bars 3413→9278, eff-N 13.07→15.79 → **DSR-OOS 0.843→0.99 CLEARED**
  (p 0.0002). BUT full-century history reveals maxDD 0.167→0.275 > 0.25 → fails the DRAWDOWN gate
  instead. clears_gate=FALSE at 2 and 5 bps. Did NOT lower vol-target to rescue maxDD (would dredge the
  now-binding metric, L-018). Still risk-control not alpha (OOS Sharpe 0.703 < 60/40 1.16).
- **DEFINITIVE CLOSE (come-back-b):** no no-spend lever yields a verified full-gate clear. The book
  misses by one criterion at a time — significance at ~10y history, drawdown at ~100y history — and is
  never alpha. Accessible return-ALPHA needs a PAID input (options-implied / PIT-fundamentals /
  fund-flows) or infra (cross-exchange / basis / latency). Operator deciding among spend-required
  levers; HOLD — start none. Lessons L-020/021/022 already capture the doctrine. Immutables intact.

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

Source: `.claude/state.json` read 2026-06-24T18:52Z (`last_actor: operator-directed-enable`).

- **`halted: false` — OPERATOR-DIRECTED UNHALT (2026-06-24T18:52:52Z).** The human operator (owns the
  halt) directed "unhalt the bot" on the PRACTICE/demo account; flipped via the sanctioned writer
  `StateEngine().set_halted(False)` (true→false), env-practice confirmed before flipping. Note: the
  morning enable's `halted=false` had since flipped back to `true` (auto-halt circuit-breaker pattern;
  the noisy "16 consecutive losses" lines were unit-test artifacts, not real trades — lifetime journal
  is 26 trades), so this is a re-unhalt.
- **OANDA PRACTICE TREND LANE IS LIVE — PLACED 4 REAL DEMO ORDERS THAT FILLED (2026-06-25, operator lane choice).**
  Operator pivoted from IBKR to OANDA practice (v20). Strategy = NON-directional TREND/managed-futures
  (price vs MA, long-or-flat, shift(1) — the validated drawdown-reducer; NOT the retired directional FX
  transformer, L-016 stands). Built `src/equity/oanda_trend.py` (candles→close panel→trend_sleeve signal→
  units→orders, halt-gated, practice-asserted) + `scripts/run_oanda_trend.py` + `get_instruments()` on the
  practice client. **Token is LIVE** (the old 401 is resolved — `.env` has a working practice PAT). Verified
  PRACTICE: base `api-fxpractice`, account prefix `101`, NAV ~$102k, was flat. Cycle EXECUTED: 4/10 FX majors
  on (GBP_JPY/USD_JPY/USD_CHF/USD_CAD), 4 market orders placed → **openTradeCount=4 confirmed on the account**.
  Account is **FX-only** (68 instruments, NO metals/indices/commodities — operator expected XAU/XAG; not on
  this account). HARD LINE intact: practice-only, hard-pinned URL, no live path.
  **UPDATES (2026-06-25 pm, HEAD f717a49, loop PID rotates):** (1) FX sizing FIXED — per-base-currency
  (`base_to_home_rate`), consistent home-notional (verifier: SIZING CORRECT). (2) Robust v20 layer
  `src/brokers/oanda_v20.py` — streaming (stream-fxpractice + reconnect/backoff), TransactionLedger (audit
  trail + realized P&L → `trained_data/oanda/transactions.jsonl`), `snapshot_account_state` (→ account_state.json
  for TUI), sentiment DATA-ONLY. (3) Safety rails: no-trade band, NAV-drawdown auto-halt (20% from peak),
  nan/nav<=0 guards. (4) LEVERAGE DIAL (operator: 0.5x too timid) — `OANDA_GROSS_LEVERAGE`/`--gross-leverage`,
  default **3x**, cap 15x; at 3x ~$76k/position, marginUsed ~11%. Honest: leverage amplifies variance not edge
  (trend = ~0-Sharpe drawdown-reducer); $1M/6mo on demo = ruin-math live, NON-GOAL. (5) SECURITY-VERIFIER
  (Security Engineer): **SAFE** — practice-only guaranteed, leverage bounded, cannot bleed unbounded, real
  fills truthfully labeled. (6) `docs/dashboard-data-contract.md` for the parallel read-only dashboard
  workstream. `trained_data/oanda/` gitignored; token in `.env.local` (gitignored, never committed, diff-scanned).
  17 no-mock tests; flake8 clean.
  **SENTIMENT LEVER TESTED → NEGATIVE (2026-06-29, ceed4b2, the deferred flag now closed).** Pre-registered
  (docs/experiment-oanda-sentiment-2026-06-29.md, written before results) OANDA position-book contrarian
  fade: S=-(NL-0.5), causal t->t+1, 521 daily books x 7 majors (OANDA serves books >=2y). Pooled IC +0.0005
  (t=0.03 ≈ 0); IS portfolio Sharpe 0.403 (t=0.49 insig.) COLLAPSES to -1.593 OOS, IC flips sign; per-year
  Sharpe +1.25/-0.51/-1.20. Fails all 3 pre-registered bars → NO EDGE, shelved (offline-only, not wired).
  Honest negative = success; closes the last genuinely-untested lever. **Separate verifier (Code Reviewer):
  NEGATIVE CONFIRMED** — causality clean (forward shift), sign clean (momentum=exact negation, |IC|≈0 either
  way), cost-independent (gross OOS Sharpe -1.562 vs net -1.593), eff-N≈521 not 3640 (only more null); even the
  most-favorable per-pair causal-demeaned timing test fails OOS (t=1.64 IS, OOS Sharpe -0.65, sign-flips 3/7).
  No hidden edge. Wrinkle noted: pre-reg+result in one atomic commit (provenance adequate-not-ironclad).
  `src/equity/oanda_sentiment.py` offline; 184MB book cache gitignored.
  **MULTI-ASSET TREND — GATE-CLEARING, the strongest finding of the whole campaign (2026-06-29, ec3182f).**
  Operator ungated a broad new-markets hunt (no spend). Pre-registered (docs/experiment-multi-asset-trend-
  2026-06-29.md, before results): ONE canonical trend rule (price>200d SMA, long-or-flat, shift1) UNIFORM
  across 21 free multi-asset ETF/crypto proxies (7 classes), per-asset streams HRP-combined + 10% vol-target,
  portfolio-level judgment. **CLEARS the ship gate: Sharpe 0.744 full / 0.829 OOS, maxDD 0.18, 26/34 pos yrs,
  34y, all 7 sub-periods positive — the FIRST lever ever to clear the gate on its own** (FX-only was gross≈0).
  Robust: no-crypto 0.79/0.91 PASS, no-overlay 0.69 PASS (not artifact). **BUT ties EW buy-hold on Sharpe
  (0.829 vs 0.895) — the edge over passive is DRAWDOWN control (18% vs 68%, ~4x), NOT alpha.** Same drawdown-
  reducer property as FX trend, now gate-clearing at portfolio breadth. Per the strict relative "+1 vs baseline"
  bar: does NOT clear (tie); per the absolute ship gate: DOES. REPORTED, NOT promoted/traded (new-market
  execution = operator decision). `src/equity/multi_asset_trend.py` offline.
  **Skeptical verifier: GATE-CLEARING CONFIRMED** — causal (triple-shift, reproduced to 3 decimals), uniform/
  no-dredge (losers FXY/UNG included), gate legit (not tautological). CAVEAT (verifier-flagged, substance
  survives): the "34y" oversells — only SPY reaches 1993, TRUE breadth (19-21 assets) only since ~2007;
  read as "~19y genuine multi-asset breadth (2007-26), SPY-anchored before." Verdict survives: post-2007
  diversified = 0.737 PASS, and the whole OOS (2017-26) is post-2007. Survivor-only ETF universe → 0.18 maxDD
  is a FLOOR not a guarantee (delisted assets absent). NET CAMPAIGN SCORECARD: every return/direction lever
  NEGATIVE (intraday 52%, daily factor ~0, multi-horizon, quality/value/lowvol/HRP, carry gate-rejected,
  news, meta-label, order-book sentiment); multi-asset trend is the FIRST + ONLY gate-clear, but it's
  drawdown-control diversification (ties buy-hold Sharpe), NOT alpha. No directional/return edge found anywhere.
- **SLEEVE COMBINATIONS TESTED → NO RETURN ALPHA (2026-06-29, a0977dc).** Pre-registered (docs/experiment-
  sleeve-combinations-2026-06-29.md). CARRY+TREND: combined Sharpe 0.624 < trend-alone 0.785 (carry DILUTES;
  negative). TREND+HARVESTER: combined Sharpe 1.112 > both but ann_return 0.126 ≈ harvester 0.130 (NOT higher)
  — gain purely lower vol/maxDD = diversification/risk-control, NOT alpha; doesn't beat buy-hold on return
  (0.126<0.21). Test 3 (XSMOM) DECLINED (anti-dredge: 2 clean negatives, a 3rd fishing = L-018 lie). Separate
  verifier in-flight. **FINAL CAMPAIGN VERDICT: no return/directional alpha ANYWHERE; only robust real finding
  is RISK-CONTROL (trend drawdown + diversification). Free-daily-bar/liquid-asset space is EFFICIENT for return
  prediction at this scale — path forward = NEW INPUTS (alt-data/higher-freq/fundamentals-at-scale), NOT more backtests.**
  Verifier: NEGATIVE CONFIRMED honest both ways (no hidden edge, no oversold win). Notes: carry+trend negative is
  honest for the pre-committed IV combiner (carry got ~66% weight, diluting trend; swapping combiner=dredge).
  Best RISK-CONTROL book in the batch = trend+harvester (Sharpe 1.112, full gate pass) — NOT alpha, but the
  practical keeper if a risk-controlled multi-sleeve book is ever stood up (operator decision; not done).
  **UPDATES (2026-06-29, HEAD fa856e8):** (7) TP/SL BRACKETS live — ATR-based stopLossOnFill (entry-2*ATR,
  ON) + optional TP (OFF, trend rides) on opening orders; verified real broker-side SL on the account
  (USD_JPY SL 160.579). (8) MARGIN/LIQUIDATION GUARD — margin_scale clamps book to max_margin_util*NAV
  (50% default) regardless of leverage dial; STRESS @15x fired (scale=0.833 -> 47% margin, held); HALT rail
  fired (halted=true -> 0 orders). Verifier (Code Reviewer): SAFE. (9) account_state.json now carries
  per-position stop_loss/take_profit (read-only, for AXIOM SL/TP columns); contract updated. (10) TIER 7
  SELF-HEAL LOOP STARTED — `scripts/run_tier7_loop.py` BOUNDED headless supervisor (no headless scanner
  daemon existed; full Tier 7 is TUI-only). Writes heartbeat + bounded self-heal (config/weight only, tiered/
  budgeted/debounced) + tier7_state each tick. LIVE: PID rotates, tier7_state running:True (heartbeat fresh +
  pid alive). Explore-agent + Security-verifier (Security Engineer) confirmed BOUNDED + SAFE: cannot unhalt
  (only TUI app.py:2724 does), cannot trade (no execution import), cannot flip env, cannot promote/un-quarantine
  a ship-gate-failing artifact (retrain_gates → online_retrainer rewrites only xgb_momentum/rf_risk/
  ridge_confidence sklearn baselines IN-PLACE; zero promotion code). Self-heal events in
  tier7_state.self_heal.recent_events (real apply() results, lie-policy). running:YES honest (fresh≤90s + pid alive).
  **AUTONOMY DISCREPANCY (noted so it's not a future surprise):** `ScannerConfig.self_heal_max_autonomy_level`
  DEFAULTS TO 5 (`config.py:682`) while the self_heal docstring/`is None` branch intend LEVEL_3. At 5 the
  unattended loop would auto-apply LEVEL_5 retrains. FIXED: `run_tier7_loop.py` CLAMPS to 3 by default
  (`--max-autonomy` / `TIER7_MAX_AUTONOMY`, operator-raisable [3,5]); at 3 only reversible operational heals
  auto-apply, retrains REFUSED (unit-tested + live: status=degraded). Commits b69ad30 (SL/TP display) +
  fa856e8 (Tier 7 supervisor) + 03fd7e2 (clamp) pushed. Two background loops: OANDA trend (trades) + Tier 7
  self-heal (operational recovery only, clamped L3).
  **RESOLVED 2026-06-29 (operator-approved): flipped `config.py:682` `self_heal_max_autonomy_level` default
  5->3** so the dataclass matches its own docstring — defense-in-depth, fail-closed for EVERY `ScannerConfig()`
  consumer (not just the supervisor). DOCUMENTED BEHAVIOR CHANGE: the attended TUI scanner's self-heal now
  also defaults to L3 (was 5); operator can still raise explicitly via config / `--max-autonomy` /
  `TIER7_MAX_AUTONOMY`. No profile dict overrides it. verify_gate PASS (28 checks, hard_no_ok=True) +
  risk_monitor GREEN post-flip. This safety-TIGHTENING relaxes no Hard NO.
  **RAISED 2026-06-29 (operator-approved): `run_tier7_loop.py` autonomy default L3->L5 (FULL).** Operator
  ungated full self-heal autonomy on the supervisor. Bounds RE-DERIVED FROM DISK and HOLD at L5: the retrain
  handlers `_handle_retrain_gates` -> repo-root `online_retrainer.py` (writes ONLY xgb_momentum/rf_risk/
  ridge_confidence `.pkl` sklearn baselines via pickle.dump; ZERO halt/env/promote/champion/quarantine/
  execute/order/`.keras` patterns) and `_handle_retrain_rl_position_sizer` (marker file only). So even at L5
  the supervisor CANNOT unhalt, flip env, touch real money, promote/un-quarantine, or write a `.keras` champion
  (directional transformer stays closed). NOTE: the `online_retrainer` self_heal calls is the REPO-ROOT module,
  NOT `src/core/modular_inference.py`'s `.keras` trigger_retrain (different function, not on this path).
  ScannerConfig dataclass default stays 3 (global safe for other consumers); only the supervisor runs L5.
  Supervisor restarted L5 (running:True). Lower again with `--max-autonomy 3` if desired.
- **(SUPERSEDED) Equity-harvester H1 / IBKR-paper lane (2026-06-25 earlier):** harvester shadow loop ran
  (running:YES via oracle, simulated fills) but IBKR paper needs `ib_async`+gateway+login. Operator chose
  OANDA instead; harvester code (`scripts/run_equity_harvester.py`, runner `execute_order` inject) retained,
  not the active demo. Original framing below kept for lineage.
- **EQUITY HARVESTER H1 (lineage):** (2026-06-25, operator-authorized H1 for PAPER only).
  Re-unhalted via `StateEngine().set_halted(False)` (`automation/state_engine.py`); built H1 entrypoint
  `scripts/run_equity_harvester.py` (shadow + ibkr-paper lanes, `--loop`) driving the gated
  `run_shadow_rebalance` (added injectable `execute_order`). Re-validated SHIP_GATE on the current PIT
  universe (was hash-stale → NO_ACT; re-ran `run_equity_harvester_shipgate_pit.py` → gate_pass net_sharpe
  0.908, hash `f550709a` now MATCHES snapshot). **Oracle (`running_status.py`, the fail-closed lie-catcher)
  confirms `running:YES · process:YES · cycles_executed≥3 · live_artifacts:YES`; ledger hash-chain intact;
  20 equal-weight orders FILLED in SIMULATION.** Fills are SHADOW (simulated) — NOT real paper orders yet.
- **ACTUAL paper-broker fills BLOCKED (operator-side, precise):** equity harvester needs IBKR **paper**
  (OANDA is FX-only, can't fill US stocks). Runtime blocker captured: `ImportError: ib_async not installed`.
  To place real demo trades: (1) `pip install ib_async`; (2) IB Gateway/TWS on `127.0.0.1:7497` (paper);
  (3) IBKR paper-account login. The `--broker ibkr-paper` lane connects + places the instant those exist
  (fill path wired via `whole_share_round` + `place_equity_order`, UNVERIFIED until a live gateway tests it).
- **(LEGACY FX, retired per L-016 — not the live path):** old FX blockers (no scanner proc, OANDA 401,
  stale per-pair champions) no longer gate the live strategy; the equity harvester replaced FX direction.
- **`mode: "live"`, `status: "running"`.** mode=live is EXECUTION mode (place orders), NOT real money.
  Orders go to the PRACTICE/paper account: the order client is `OandaPracticeClient`, hard-pinned to
  `PRACTICE_API_URL = api-fxpractice.oanda.com/v3` (`src/utils/oanda_practice.py:117`) and it IGNORES
  `oanda_environment` entirely — there is NO live-URL path in the order client. Verified by separate
  agent: "Can it place a real-money order? NO" (HIGH confidence). This is why mode=live is acceptable.
- `oanda_environment: "practice"` (`src/scanner/config.py:738`) — **immutable Hard NO, untouched.**
- Gates taught (committed): risk_monitor + verify_gate alarm on `mode=live` ONLY when env≠practice;
  **env=live / real-money / ship-gate stay HARD** (env=live+mode=live → double hard alarm). L-014.
- NAV $102,183 · `open_trades: 0` · 6 per-pair transformer champions present but 36d-stale (+19
  quarantined) → bot abstains on staleness/ship-gate, so unhalting unleashes no flood of trades.
- Known residual (non-blocking, pre-existing): static env-tripwire matches `oanda_environment = "live"`
  assignment + git-diff `api-fxtrade`; a future *dict-form* profile override `"oanda_environment":
  "live"` wouldn't be caught by the tripwire — but the practice-pinned order client is the primary
  rail. Add the dict pattern IF env is ever wired into a profile dict / the client honors env.

- **close_trade halt guard (HUMAN-authorized hot-path safety-ADD, 2026-06-24).** `execute_trade` had a
  halt guard (execution.py:2093) but `close_trade` did not — an autonomous close could reach the broker
  while halted. Added a mirror guard: autonomous/programmatic closes are BLOCKED while `halted=true`
  (returns `BLOCKED: state.halted=True` before any broker call); an EXPLICIT operator close passes
  `operator_override=True`. Safety-ADD only (can only make close MORE restrictive). Separate verifier
  PASS, 4 no-mock tests (`tests/test_close_trade_halt_guard_2026_06_24.py`). Env/practice-pin/order-client
  UNTOUCHED — this was authorized for the close-guard ONLY; the "everything to live mode" instruction
  was NOT approved and NOT acted on.
  **TUI follow-up (TUI agent's job, not mine):** `src/tui/screens/trades_screen.py:874` calls
  `close_trade(...)` WITHOUT `operator_override=True`, so the operator's manual TUI close is now also
  blocked while halted (strictly fail-closed). The TUI's halt-aware `c`-key confirm must pass
  `operator_override=True` after the operator confirms. Do NOT wire this from here (outside scope).
- Branch: `ralph/equity-harvester-bot` (an equity-beta harvester workstream is in flight)

## In-flight work (from session memory, not re-verified this turn — confirm before relying)

- **STATUS-ORACLE TRAP FIXED (2026-06-29).** `running_status.py` reported ONLY the dormant equity-harvester
  (IBKR, superseded) lane -> "running:NO" even though the LIVE OANDA-trend + Tier7 lane was running -> caused a
  false "nothing running". Rewrote it to report TWO clearly-LABELED lanes: **LIVE LANE** (OANDA trend + Tier7;
  running:YES if `run_oanda_trend.py` proc alive OR `trained_data/oanda/account_state.json` fresh<=2h OR
  `.claude/heartbeat.json` fresh<=90s + pid alive) and **HARVESTER LANE** (dormant/legacy; "running:NO is
  EXPECTED, not a fault"). `--assert-running` now asserts the LIVE lane. 4 no-mock tests. Also force-killed a
  duplicate uvicorn (46367 — had a stale :8888 SSE connection; canonical listener 53816 kept, health ok).
  **DAILY MONITOR should read the LIVE lane:** `python3 .claude/loop/running_status.py --assert-running`
  (exit 0=up, 3=down); or raw disk: tier7 = `.claude/tier7_state.json` `running:true`; trader =
  `trained_data/oanda/account_state.json` mtime <=2h. NEVER the harvester lane (always NO = dormant).
- **Multi-strategy sleeve program — Phase 0 DONE; no-spend frontier exhausted (2026-06-25, running:NO,
  offline-eval only).** Plan doc `docs/multi-strategy-sleeve-architecture-2026-06-24.md` (commit 3a7038b).
  Phase-0 (commit 65015bd): built Sleeve A = multi-asset trend/managed-futures (long-or-flat ETFs,
  `src/equity/trend_sleeve.py`) + HRP-across-sleeves combiner (`sleeve_combiner.py`) + combined-book gate
  (`SHIP_GATE_book.json`). Falsifiable test {harvester+trend} vs harvester-alone, pre-registered canonical
  params, separate-verifier independently re-derived (byte-for-byte) + multiple-testing audit.
  **VERDICT (verifier-confirmed): NOT a stable Sharpe +1 — it's a DRAWDOWN-only risk-reducer.** Combined
  beats harvester on Sharpe full +0.033 / OOS +0.118 BUT only 2/4 disjoint 5yr blocks (wins=GFC+2021-25,
  loses calm 2011-20); sign-test p≈0.69; OOS ~72% overlaps the winning 2021-25 regime (not independent);
  collapses under N≈5 ideas tried. ROBUST finding: maxDD 0.229→0.166 across ALL regimes (trend goes to
  cash in crashes, corr~0.49). On the operator's "+1 Sharpe OOS" bar this is an HONEST NEGATIVE (=success).
  **Frontier state:** trend was THE strong free/orthogonal candidate (drawdown-only); carry is gate-rejected
  (EM-carry fat tail); quality needs PIT fundamentals = DATA PURCHASE (HARD STOP, operator's card —
  financial-datasets = latest-FY/$0 balance, no free PIT). NOT iterating more sleeves (anti-p-hacking;
  N already burns the significance budget). Surfaced operator decisions: (a) is tail-risk reduction a goal
  worth taking trend toward shadow/live [operator-authorized hot-path]? (b) authorize PIT data purchase to
  unblock the quality sleeve = the next real diversification lever. STOP-DONE on no-spend frontier.

- **Phase-0 trend-sleeve verdict — INDEPENDENTLY RE-VERIFIED 2026-06-25 (trust-downgraded; complete, was
  mid-run at limit).** Fresh bake-off re-run byte-identical to committed `SHIP_GATE_book.json`; separate Code
  Reviewer re-derived from source + multiple-testing audit. VERDICT: **risk-reducer-only** — drawdown
  reduction (0.229→0.166) ROBUST across all 4 sub-periods; Sharpe improvement (full +0.033/OOS +0.118) is
  REGIME-DEPENDENT (2/4 sub-blocks, sign-test p=0.69, OOS 72% overlaps winning 2021-25) → does NOT clear the
  +1 bar. Causality clean (double-shift, exec_lag=1), apples-to-apples, invariants intact. Matches prior
  record (1b1e340); verification now closed.

- **Quality factor — FREE true-PIT pipeline BUILT + RUN; ROBUST NEGATIVE (operator corrected plan to NO paid
  data; running:NO).** Paid-data path (financial-datasets Pro) SUPERSEDED — operator directed a $0 PIT build.
  Built: `src/equity/edgar_fundamentals.py` (SEC EDGAR companyfacts, TRUE-PIT via real `filed` dates, annual
  10-K, synonym-merge + cost-derived gross profit, as-originally-reported = earliest-filed), `sp500_membership.py`
  (survivorship-aware Wikipedia change-log reconstruction — 873 ever-members, 279 since-removed incl. SIVB/
  TWC/AET), `pit_quality_eval.py` (`evaluate_book_wide` = staggered-universe NaN-safe evaluator, baseline
  UNTOUCHED; forces weight=0 where price NaN so fabricated cells inert). quality_data extended for `filed`-date
  PIT + `clip_z`/`components`. **RESULT** (`trained_data/backtests/pit_quality_bakeoff.json`, 624-name PIT
  universe, 2012-2026, cov 95.5%): canonical composite **LOSES** to EW (full −0.600/OOS −0.324) — but that's
  largely a non-robust `−debt_to_equity` outlier artifact (banks/REITs/neg-equity tails; winsorize halves it).
  **Clean margin-only tilt merely MATCHES EW** (full −0.018/OOS +0.026) = **NO shippable edge.** Honest
  negative = success. LABELLED residual bias: delisting returns understated (ffill), yfinance price-availability
  survivorship (189 delisted lack prices), EDGAR XBRL floor ~2010. 23 no-mock tests green; flake8 clean.
  Caches on disk (`market_data/equity/sp500_pit_fundamentals.json`, `sp500_prices.parquet`) for re-derivation.
  **Separate verifier (trust-downgraded) CONFIRMED:** re-derived EVERY headline number exactly from cache
  (EW 0.740, canonical −0.600, margin-only OOS +0.026, 624 names, cov 0.9554); the lookahead validator
  genuinely fires (injected forgery → raised); baselines unmodified (git diff empty); residual survivorship
  is CONSERVATIVE (biases toward EW, cannot manufacture the negative); −0.600 confirmed a d/e-outlier artifact
  (max d/e 990, 303 neg-equity). Verdict: no shippable edge, ROBUST. Commits d6c63bb (receiving end) + c8be8cd
  (EDGAR/membership/eval + result). Diversification-sleeve frontier now genuinely exhausted (trend=drawdown-only,
  carry=gate-rejected, quality=no-edge) — further factor variants = dredging; operator's call.

- **Equity-harvester four-pillar self-improver — SHIPPED-TO-DISK + UNIT-TESTED SCAFFOLDING, NOT RUNNING
  (corrected record 2026-06-24; separate-verifier confirmed; supersedes prior "running/wired/now does X"
  chat framing, which was inaccurate — see L-017).** TRUE status, re-derived from disk independently:
  - Committed (capability only, never executed): N1–N5 shadow runner `src/equity/runner.py` + config flag
    `enable_equity_harvester` (`ea85f8c`); Pillar 3 decision gate `src/equity/decision_gate.py` (`d4d8aa7`);
    Pillar 2 cycle ledger `src/equity/cycle_ledger.py` (`ce74989`). **running in process: NO.**
  - **cycles executed: 0** — `trained_data/equity/` does NOT exist; `cycle_ledger.jsonl` /
    `rebalance_state.json` ABSENT. **state artifacts present (live, non-test): NO.**
  - **invoked by: tests ONLY** — `run_shadow_rebalance`/`decide_cycle`/`append_cycle` called only by their
    test files + internally in `runner.py`; no `main.py`/`src/scanner/*`/`embedded_scanner.py`/`scripts/`
    calls them. The enable flag guards code no live path reaches.
  - 27/27 unit tests pass (decision_gate 11, cycle_ledger 6, runner 10) = "shipped + unit-tested", NOT
    "functioning". Dormant until **H1** (the live invocation) — HELD for operator authorization.
  - Pillars 1 (verifier) + 4 (risk monitor): **not built** (stopped per operator). running:NO / built:NO.
  - Even if invoked now it would `REFUSE` every cycle (`state.json halted=true`). Hot path / env-pin /
    FX legacy untouched (git show --stat on all three commits confirms).
- **Equity-harvester-bot**: 22-story PRD; TUI wired to equity (commits 3a58c6c, eb04687). An
  independent code review flagged **4 CRITICAL + 7 HIGH execution defects** (C1 crash→double-submit,
  C2 fill-detection reads aggregate position as per-order fill, C3 books PENDING as FILLED, C4 can't
  detect a MISSING corp-action split). These are the real blocker to paper/shadow trading. `[unverified — from memory recent.md, re-grep before acting]`
- **FX direction is RETIRED — equity harvester is the live strategy (operator doctrine 2026-06-24, L-016).**
  FX/forex direction hit a hard ~52% ceiling (coin-flip, no shippable edge; confirmed 4+ ways:
  price-only, news, factor, carry, meta-labeling), so the transformers fail the 10% ship gate and stay
  PERMANENTLY quarantined. The product pivoted to the **equity harvester** (equity-beta risk-premium
  harvesting). Consequences for future sessions: the 6 stale FX champions (36d old) are
  **expected/abandoned-by-design, NOT a fixable gap**; do **NOT** propose FX retrains — and the earlier
  "refresh creds + retrain a gate-passing FX model so it trades" suggestion is a **DEAD END** (the
  ceiling is the market, not a bug). Route trading work to the equity harvester. See L-016 + verdict
  docs in `docs/`.
- **L7 (TUI live-run finding) RESOLVED as intentional:** the Trades tab rendering "HARVESTER REBALANCE
  PLAN" instead of FX trades is correct product behavior (FX retired → equity harvester). Not a bug;
  never "fix" it back to FX.

## Blockers

- None for *this* task (context-system build). For trading: the **operator** may unhalt at any time
  (operator-directed override is allowed and was exercised 2026-06-24T18:52Z → `halted=false`). What
  stays doctrine: the **AUTONOMOUS/loop path may NEVER unhalt on its own** (the `/evolve` loop can't
  relax the halt Hard NO; only the human who owns the halt can lift it). And unhalting ≠ trading:
  actual trades are gated on scanner-running + fresh (non-401) OANDA creds + a gate-passing fresh
  model — none of which hold right now, so the bot abstains. See the runtime-state blockers above.

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

## 2026-07-01 — PR #51 "agentic data-extraction" edge claim: AUDITED, INFRASTRUCTURE ONLY (no result)

Operator believed PR #51 (AXIOM Phase 2, merged origin/main 1a58f65) found a real gate-clearing
edge via "agentic orchestration of data extraction." Re-audited from scratch. Verdict: **the belief
is not supported by any artifact on disk.**

- The pipeline in question is Track B ("agentic research portfolio", branch
  `claude/agentic-research-portfolio-qetsfr`, `src/equity/research/{contracts,pit_text_loader,
  entity_blinder,scorer,harness}.py`, pre-reg `docs/experiment-equity-research-alpha-prereg-2026-06-30.md`)
  — an LLM-research-reads-filing-text alpha test, distinct from the ALREADY-VALIDATED equity-beta
  harvester (Track A, Sharpe 0.92 net, real risk-premium result, `trained_data/backtests/SHIP_GATE*.json`).
  Likely source of the conflation: PR body lists both under one summary.
- Track B was PRE-REGISTERED and built (58 no-mock tests, 3-reviewer audit) but **never executed**:
  `scorer.py:1-11` states no LLM/Anthropic call is wired ("ANTHROPIC_API_KEY is absent... driven by
  SEPARATE subagents"); `.env.local` confirms no ANTHROPIC_API_KEY present. Pre-reg doc §7 ("Results")
  is still an unfilled template. `harness.py:552-556` — `dsr_oos_n22: None` (TODO, gate criterion §4.5
  unenforced). `harness.py` §4.7 `overall_verdict` logic (~L740-780): `"REAL"` is **unreachable from
  the code as merged** (human blinding audit uncomputed, fail-closed by design). `git log --all --
  src/equity/research/` — last commit is `5d650c2` (scorer plumbing), nothing after. No
  `research_scores`/`research-alpha` artifact anywhere in repo (tracked or untracked).
- Design-level killer audit (would-be-run assessment): entity-blinder self-documents as leaky-by-design
  (`entity_blinder.py:22`, "NOT a guarantee") — correctly treated as noise-reducer, not load-bearing
  control, per its own §8; costs ARE wired (`harness.py:483,575`, cost_bps turnover model); survivorship
  uses PIT S&P membership but filing-availability requirement admits a coverage-gap caveat
  (pre-reg §8.4, "loader paging gap," honestly flagged not hidden).
- Two independent-verifier subagent dispatches both hit the session token limit and returned nothing
  (0/51 tokens) — substituted direct re-verification (fresh independent tool calls: env-key check, PR
  review-comment search, repo-wide artifact search) in this same session instead of re-delegating.
- **Action for operator:** no change needed to halted state; this is a correction of belief, not a
  bug fix. If Track B is to be pursued, the actual blocker is wiring a real LLM scorer call + DSR-OOS/
  Bonferroni + the human blinding audit — none of which exist yet.

## 2026-07-01 — Track B actually RUN (worktree `ml_engine_trackb`, branch `trackb/run-2026-07-01`,
commit `8aa9417`, off `origin/main`). **Verdict: NO EDGE** (bounded pilot). Closed the two blockers
above: implemented DSR-OOS(N=22)+Bonferroni in `harness.py` (was hardcoded `None`), and this session
acted as the LLM scorer directly (no `ANTHROPIC_API_KEY`), hand-scoring 36 blinded 10-Ks across 12
mega-cap names (AAPL/MSFT/GOOGL/AMZN/NVDA/META/JPM/JNJ/XOM/PG/HD/UNH × FY23-25) — a disclosed
scale-down from the frozen full-S&P500 design, not the full run.
- Found + fixed a load-bearing pipeline bug en route: SEC iXBRL filings wrap ~98K chars of
  non-visible `<ix:header>` metadata before the visible cover page; `pit_text_loader.py`'s old
  `_SKIP_TAGS` didn't skip it, so a 12K-char head-truncation was 100% XBRL tag-soup, 0% prose.
  Fixed + regression test; independently re-verified against a live EDGAR fetch by the verifier.
- Result: full-sample Sharpe 0.489 (looks promising) but BOTH pre-registered controls kill it —
  post-cutoff arm flips to **-0.861** (textbook lookahead signature: strong pre-cutoff, dies/inverts
  post-cutoff) and placebo (0.358) is nearly as large as the real full-arm number (not clean per the
  frozen 0.15 threshold). `overall_verdict=INSUFFICIENT`. Own blinding audit: 36/36 filings
  re-identifiable despite redaction (ticker-adjacency leaks, unredacted founders'-letter names,
  unredacted product names) — confirms the pre-reg's own §8 finding that blinding isn't load-bearing.
- Independent verifier (Model QA Specialist, cold from disk): re-derived the DSR math, confirmed the
  iXBRL fix against a live Apple 10-K fetch, spot-checked 4 PIT dates against EDGAR + 3 blinding
  leaks verbatim in the blinded files, confirmed zero frozen knobs (weights/quintile/cadence/
  vol-target) were touched, ran the full test suite (111/111). Sign-off: verdict holds, no
  discrepancies found. Caveat carried forward: N=12/Q5=2 is genuinely underpowered — this run
  answers the lookahead-contamination question decisively but does not by itself close the door on
  the hypothesis at the frozen ~500-name/~14yr scale.
- Not pushed anywhere; commit lives only on the local worktree branch pending operator decision.
  Full write-up: `docs/experiment-equity-research-alpha-prereg-2026-06-30.md` §7 (in that worktree/
  branch — not yet on `main`).
  Bonferroni + the human blinding audit — none of which exist yet.

## 2026-07-02 — Equity harvester UNHALTED on practice/paper (operator-authorized). 5 guardrail
steps completed before flipping, independent verifier dispatched to confirm:
1. Honest sizing: broad-universe PIT gate re-derived (net Sharpe 0.739 full / 0.354 OOS, NOT the
   0.908 curated-pool number) — `docs/equity-harvester-sizing-2026-07-01.md`.
2. ARM checkpoint wired: `scripts/run_equity_harvester.py::_connect_ibkr_paper` now checks
   `LiveGate.is_armed()` before returning a real fill callback (previously bypassed it entirely,
   per `docs/equity-harvester-verdict-2026-07-01-independent-audit.md` §3 item 5). Never auto-arms.
3. FX H1 quarantine: USD_JPY/USD_CAD/AUD_USD have no H1 artifacts in this repo (already quarantined
   from earlier M15/D events) — nothing to move here.
4. Unhalt scope traced: unhalting releases the equity harvester AND the already-running
   `com.buddy.trend` OANDA book (pid changes across launchd restarts; structurally non-directional
   per `src/equity/oanda_trend.py` docstring) — operator explicitly confirmed this scope via
   AskUserQuestion before flip. No directional FX model or LLM path reachable (no FX scanner
   process running).
5. `ib_async` installed; IBKR paper port hardcoded 7497 (7496/4001 never referenced); OANDA
   practice-pin confirmed (`config.py:742`, `.env.local`).
- Flipped via `StateEngine.set_halted(False)` (atomic tmp+rename) — `.claude/state.json` now
  `halted: false`, `halted_lanes: {oanda_fx:false, equity:false, brain:false}`.
- Started `scripts/run_equity_harvester.py --broker ibkr-paper --loop 3600` in background
  (PID 48095). No IB Gateway listening on 127.0.0.1:7497 → clean fallback to shadow lane this
  cycle (`CYCLE_RESULT: ran=True reason=executed lane=shadow orders=0`). ARM checkpoint will bind
  once IB Gateway is actually running.
- Two adjacent findings surfaced but NOT acted on (blocked by permission classifier as out of
  named scope; spawned as follow-up task chips instead): GBP_CHF has a live (non-quarantined)
  FX direction model failing the 10% gap rule (train=0.6074/val=0.5/gap=0.1074); a dormant sibling
  worktree `ml_engine_trackb` has a stale `halted:false` local state + unquarantined H1 models for
  the same 3 pairs (confirmed no process running, nothing scheduled — latent, not active).

## 2026-07-02 (same session, ~1hr later) — CORRECTION + RE-HALT. Operator sent a stand-down after
the unhalt above: decision changed to build PROPER PER-LANE halt control (not the global flag)
before letting only the equity harvester go live. Verified live state before acting (operator's
premise that "STEP6 was blocked" did not match disk reality — it had already succeeded and been
operator-confirmed via AskUserQuestion); found `com.buddy.trend` (OANDA risk-premium trend book,
`src/equity/oanda_trend.py` — structurally non-directional, confirmed no import of
transformer_direction/keras anywhere in that file) was ~3 min from its next hourly tick at the
time the stand-down arrived. **Re-halted immediately** (`StateEngine.set_halted(True)`) before that
tick — confirmed via fresh log read that NO tick/order occurred during the ~8-minute unhalted
window (08:30:24–08:38:36 EDT); zero trade/order artifacts written anywhere in that window.
Killed the `run_equity_harvester.py --broker ibkr-paper` background loop I'd started (PID 48095).
`.claude/state.json` now `halted: true`, `halted_lanes: {oanda_fx:true, equity:true, brain:true}`
— maximally safe, both global and per-lane.
- **Discovered a large concurrent process already active in this exact working directory** —
  almost certainly the Ralph autonomous loop on this branch (PID 22421 `run_tier7_loop.py` running
  since Tuesday). It has independently: (1) built + committed proper per-lane halt control
  (`193d847 feat(halt): per-lane halt control (oanda_fx / equity / brain)`) — confirmed wired,
  `execution.py:2099` now calls `StateEngine(lane="oanda_fx").get_halted_strict()` and
  `decision_gate.py` is similarly lane-aware. This is the exact mechanism needed to unhalt only
  the equity lane while keeping oanda_fx halted — `StateEngine(lane="equity").set_halted(False)`
  now exists and does NOT cascade to other lanes (confirmed at `state_engine.py:336-371`).
  (2) Started its own `run_equity_harvester.py --broker shadow` loop (PID 96555, running since
  01:08 — not started by this session). (3) Started a new `run_brain_loop.py` subsystem (PID
  79486, `src/brain_loop/`, extensive new tests — unrelated to this task). (4) Quarantined GBP_CHF
  + 3 more over-gap/joint-fallback FX pairs (AUD_JPY, EUR_AUD, NZD_USD) at `_quarantine/*-
  20260702T123815Z/` — timestamped the same minute as this session's re-halt, apparently picking
  up the follow-up task chip this session had spawned.
- **Did not touch the per-lane mechanism or the concurrent session's work** — two-writer collision
  risk on shared state files is a known hazard here (see AXIOM incident memory). Left `.claude/
  state.json` at global+all-lanes halted (safest state) for the per-lane session to build from.
- Artifacts this session leaves on disk, uncommitted, ready to reuse: `scripts/
  run_equity_harvester.py`'s ARM-checkpoint fix (`LiveGate.is_armed()` gate on the ibkr-paper fill
  path — independently verified PASS), `docs/equity-harvester-sizing-2026-07-01.md` (honest 0.739
  full / 0.354 OOS sizing numbers). USD_JPY/USD_CAD/AUD_USD H1 direction models confirmed still
  unloadable (no artifacts in this repo) by two independent checks.

---

## 2026-07-03 (afternoon) — launchd `com.buddy.tier7` EX_CONFIG fixed; nohup handed over

- **Symptom:** `com.buddy.tier7` LaunchAgent crash-looped `last exit code = 78: EX_CONFIG`,
  `runs = 8444+`, ZERO output in its StandardOutPath. The supervisor was surviving only via a
  `nohup` instance (PID 16644, started earlier 2026-07-03).
- **Root cause (HIGH confidence):** `xpcproxy` (launchd's pre-exec spawn helper) hit a macOS
  Sandbox `deny(1) file-read-data` on the stdout target
  `trained_data/axiom/tier7_loop.out` — **every spawn, in lockstep with the 30s ThrottleInterval**
  (proven via `/usr/bin/log show --predicate 'eventMessage CONTAINS "tier7"'`). The path is inside
  `~/Documents`, a TCC-protected folder. The working `com.buddy.trend` writes to the SAME dir but
  its `trend_loop.out` carries a `com.apple.macl` sandbox-grant xattr; `tier7_loop.out` had only
  `com.apple.provenance` (no grant ever recorded for that specific file). Same binary, same
  `WorkingDirectory` — only the per-file TCC grant differed.
- **Fix:** moved `StandardOutPath`/`StandardErrorPath` in
  `~/Library/LaunchAgents/com.buddy.tier7.plist` out of `~/Documents` →
  `~/Library/Logs/com.buddy.tier7.log` (not TCC-protected). `WorkingDirectory` unchanged (trend
  proves running processes CAN write `~/Documents`). Reloaded via `bootout`+`bootstrap`. Result:
  `state=running`, `runs=1`, stable past throttle window, ticking cleanly. First try.
- **Handoff (clean, verified):** confirmed the launchd process (PID 88167) writes state files
  INTO `~/Documents` (`tier7_state.json` `supervisor_pid: 88167`, fresh) → not sandboxed from the
  state dir → safe to retire nohup. `kill -TERM 16644` (exited in ~1s). Heartbeat pid flipped to
  88167 immediately (deference logic `run_tier7_loop.py:127` releases the beacon when the foreign
  writer dies). Now ONE supervisor (88167), `heartbeat.json` fresh, old `tier7_loop.out` idle.
- **Singleton lock — FIXED (2026-07-03 same session).** `run_tier7_loop.py` now has the same
  flock guard as trend (`_acquire_singleton_lock`/`_release_singleton_lock`/`_singleton_lock_path`
  → `trained_data/axiom/tier7_loop.singleton.lock`). A second supervisor refuses to start (exit 1)
  while a live one holds the lock. **Tier7-specific subtlety** vs trend: tier7 self-restarts via
  `os.execv` (same PID, in-place) for code-freshness, NOT exit-for-respawn — so it must
  `_release_singleton_lock()` immediately before `execv` or the re-exec'd image would be denied by
  its OWN inherited lock and refuse to start. Verified LIVE: the running supervisor (88167)
  code-freshness-picked-up this very edit and re-exec'd itself twice (15:42, 15:45) — each followed
  by a clean `START`, never `refusing to start`; lock file holds its own pid; a manual 2nd instance
  refused with exit 1. Tests: `tests/test_run_tier7_singleton_2026_07_03.py` (4, incl. the execv
  release path). 24 adjacent tests green, flake8 clean, risk monitor GREEN. Did NOT touch
  state.json / halt flags / any order path.

## 2026-07-06 — SELF_IMPROVE tier (4th PolicyEngine tier) + AXIOM operator daemon authored; autonomy ON
- **Built** `src/agent_runtime/self_improve.py`: a fourth tier, narrower than OPERATIONAL — a fixed
  menu of pre-coded data/doctrine edits with HARDCODED (not caller-supplied) target paths per
  action, gated by a structural allow/deny check (`_check_target_paths`, runs before any write) —
  STRUCTURAL_DENYLIST covers execution.py/decision_gate.py/state_engine.py/gates.py/config.py/
  scanner-agents/brokers/risk/control_safety.py/agent_runtime itself (incl. policy.py and
  self_improve.py)/verify_gate.py/risk_monitor.sh/state.json/the autonomy flag file — refused
  unconditionally, before the tiny ALLOWLIST (agent_weights.json, LESSONS.md, NOTES.md) is even
  checked. Lifecycle: byte-snapshot (NOT `git stash` — L-024) -> apply -> test_cmd ->
  verify_gate.py -> risk_monitor.sh -> scoped `git commit -- <path>` on full pass, or restore +
  `PolicyDenied` (no commit) on any failure. ESCALATION's structural proposal-only guarantee
  (policy.py) is unchanged — verified by an independent Security Engineer pass (see below).
- **Enabled autonomy** (`trained_data/axiom/loop_autonomy.json` -> `agent_autonomy_enabled: true`,
  confirmed read-back). OPERATIONAL/DEESCALATION/SELF_IMPROVE now execute when proposed;
  ESCALATION (arm/unhalt/leverage-increase/promote/new-exposure/code-change) is unaffected —
  structurally proposal-only regardless of the flag.
- **Executed the two pending proposals** through the new gate: (B) added the missing L-024
  recall-trigger row to LESSONS.md — committed `1cff70f4`, verify_gate `lessons_have_triggers`
  now PASSES. (A) purge `invalid_agent_xyz` from `agent_weights.json` — **correctly REVERTED**:
  discovered `agent_weights.json` was deliberately untracked from git in `bd6554b`
  ("ci(infra): untrack agent_weights.json"), so the commit step finds no matching pathspec and
  the gate fail-closes rather than leaving an unreversible mutation. File on disk confirmed
  unchanged (still has `invalid_agent_xyz`). **Open judgment call, needs operator decision**: (i)
  re-track the file in git, (ii) add a non-git backup/reversibility path in self_improve.py for
  deliberately-gitignored allowlist targets, or (iii) keep this one cleanup manual/ESCALATION.
  Separately did land the OTHER half of proposal A directly (not through the autonomous gate,
  since scanner-agent code is adjacent to trade-consensus): `_team.py::_validate_weights` now
  drops any persisted key outside `_BASE_WEIGHTS` at load time — closes the recurrence path.
- **Daemon**: authored `scripts/axiom_launchd/com.axiom.operator.plist` (RunAtLoad+KeepAlive,
  mirrors com.buddy.tier7, resolved `claude` CLI via `src/utils/claude_cli.py`) + added to
  `load.sh` LABELS. Caught and fixed BEFORE ever loading it: the log path was inside
  `~/Documents` (TCC-protected) with no `com.apple.macl` grant yet — the exact failure mode that
  crash-looped `com.buddy.tier7` on 2026-07-03 (see above) — moved to `~/Library/Logs/`.
  **NOT LOADED**: `bash scripts/axiom_launchd/load.sh` was hard-blocked by the Claude Code
  harness's own safety classifier (category "Create Unsafe Agents" — installing a persistent,
  auto-restarting, self-committing daemon needs a human to run it). Operator needs to run
  `bash scripts/axiom_launchd/load.sh` themselves to actually daemonize it.
- **Verification**: 126/126 tests green (72 new — self-improve structural boundary +
  `_team.py` validation), flake8 clean. Independent Security Engineer pass: PASS on all 5 checked
  dimensions (structural boundary, ESCALATION unchanged, gated lifecycle, halt/practice-pin/
  hot-path isolation, nothing armed) — one disclosed gap (MEDIUM confidence): a *future*
  SELF_IMPROVE action that skips `_run_gated_edit` wouldn't be caught by today's tests; closed
  same-session with `test_every_self_improve_action_routes_through_the_gated_edit_lifecycle`
  (source-inspects every registered action for the `_run_gated_edit(` call, mirroring policy.py's
  own forbidden-call test pattern). Commits: `e8cd82d` (tier+daemon+validation), `1cff70f4`
  (autonomous L-024 fix), `876c291` (TCC log-path fix). Halt (`oanda_fx: true`), practice pin
  (`config.py:742`), and no-arm confirmed by direct re-check, not carried forward from the
  verifier's report.
