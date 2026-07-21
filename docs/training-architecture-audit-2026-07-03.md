# AXIOM Training Architecture — Audit & Modernization Plan (2026-07-03)

**Scope:** full audit of the training pipeline (Phase 1), modernization architecture (Phase 2),
and phased implementation roadmap (Phase 3). Produced by three independent domain-specialist
discovery passes (training-code map, data inventory, learning-control-flow trace) plus a
retrain-triggers/promotion-gates pass, all read-only, all citations verified from disk on
2026-07-02/03.

**Runtime snapshot at audit time** (`.claude/state.json`, read 2026-07-03T01:00Z):
global `halted: true`; all five lanes halted (`oanda_fx`, `equity`, `brain`, `crypto_momentum`,
`track_b`); `last_actor: operator-stand-down-2026-07-02-per-lane-control-pending`; NAV $102,183;
0 open trades. Running processes (verified via `ps` + heartbeat): `run_tier7_loop.py` (self-heal
supervisor, PID 29292), `run_oanda_trend.py` (launchd `com.buddy.trend`, PID 28149),
`run_equity_harvester.py --broker shadow` (PID 96555). NOT running: TUI/EmbeddedScanner,
brain_loop, crypto_momentum runner, track_b runner.

**Hard NOs honored throughout:** this document proposes nothing that unhalts, changes
`oanda_environment` off `"practice"` (`src/scanner/config.py:738`), promotes past the ship gate,
or touches real money. All Phase 2/3 items are research/infrastructure; hot-path changes are
flagged as operator escalations.

---

## 0. Executive summary

1. **The pipeline is not the reason models plateau at 52%.** The coin-flip question is answered,
   verifier-confirmed, and closed (LESSONS L-001/L-016/L-020/L-022): there *were* real defects —
   anchored-feature leakage, a broken ship-gate metric path, a double-fit scaler — and after every
   one was fixed, price-only intraday direction landed at ~52% and stayed there under news fusion,
   meta-labeling, factor overlays, and order-book sentiment. ~52% is the efficient-market wall for
   free daily/intraday bars at this data scale, not a training-methodology defect. Confidence: HIGH
   (own 22-yr walk-forward 50.3% balanced; 4+ independent negatives; 5-source literature sweep).
2. **The training *methodology* is already modern where it matters.** The gated research harness
   (pre-registration, frozen params, untouched OOS, deflated Sharpe ≥ 0.95, Bonferroni block
   bootstrap, maxDD gate, separate adversarial verifier) is 2024–2026 best practice and repeatedly
   caught in-sample optimism that would have shipped lies. It is the system's crown jewel — but it
   lives copy-pasted across experiment scripts instead of as a library.
3. **The runtime *learning wiring* is the real modernization target.** The biggest defects found are
   architectural: every FX learning loop is welded to the interactive TUI process (dormant when no
   terminal is open, while the heartbeat claims `scanner_alive: true`); the live self-heal supervisor
   *produces* config adjustments that only the dormant TUI can *consume*; three divergent post-trade
   feedback paths coexist; drift detection logs but routes to no action; and the online retrainer has
   no evaluation gate at all.
4. **The new lanes (equity harvester, brain_loop, crypto_momentum, track_b) already embody the right
   architecture** — fail-closed per-lane halt, hash-chained ledgers, pre-registration, ship gate keyed
   to a universe hash, operator-token LiveGate, shadow-first forward records. The roadmap is to
   *codify that as the Lane Contract* and migrate the legacy FX machinery onto it, not to invent a
   new architecture.

---

## Phase 1 — Audit findings

### 1.1 Training code surface (all verified with file:line by the code-map pass)

**Tiered batch pipeline** — `scripts/run_full_training.sh` (264 lines):

| Tier | What | Entry |
|---|---|---|
| 1 | Core ensemble, 8 models × 5 master pairs | `train_single_model_m1.py --instrument {pair} --model all` (line 160) |
| 2 | Transfer learning master → correlated pairs | `train_transfer_learning.py` (line 193) |
| 3 | RL suite (PPO sizer, SAC gates, PPO exits, reward model) | `train_rl_suite.py` (line 210) |
| 4+5 | XGBoost meta-labeler + Platt/isotonic calibrator | `train_meta_calibrator.py` (line 234) |
| Final | Validation report enforcing HARD_MAX_GAP | `generate_validation_report.py` (line 248) |

**Trainer zoo** (`src/training/trainers/`, all inherit `BaseTrainer`): TransformerDirection
(d_model=16, 2 heads, 1 layer; EMA + EWC warm-start + replay; `transformer_trainer.py:224,2770`),
TCN direction, TCN volatility-regime (4-class), HistGB direction, LightGBM momentum/risk, Ridge
confidence, RandomForest fallback, XGBoost meta, TransformerRegime, and the **retired**
JointMultiPairTrainer (`joint_trainer.py:43`, saves to `models/joint/`, quarantined from routing).

**Feature pipeline** — `FEATURE_PIPELINE_VERSION = "2026-06-12-v3"`
(`src/core/modular_data_loaders.py:60`); v2 made features window-invariant after the L-001 anchored
OBV artifact (OBV is now a rolling 100-bar z-score, `src/data/feature_engineering.py:105-118`);
the invariance canary `tests/test_feature_window_invariance.py` gates regressions; v1 artifacts
refuse to load. `compute_normalized_features()` (`modular_data_loaders.py:741`) emits
returns/ATR-pct/z-scores/percentile-ranks/candle-structure/BB/momentum families;
`load_direction_data()` (`:1844`) selects top-80 uncorrelated features with warm-start feature
locking (`:1913-1932`). Scaler discipline: `_assert_scaler_not_identity`
(`transformer_trainer.py:520`) detects double-fit; `_subset_scaler` (`:828`) narrows a fitted
scaler after selection instead of refitting.

**Targets:**
- Direction: binary, `future_close` vs `current_close ± threshold` at `lookahead=24` bars
  (M15 ⇒ 6-hour horizon), ambiguous moves excluded (label 0.5), magnitude-shaped sample weights
  in [0.5, 2.0] (`modular_data_loaders.py:2094-2150`). Train/val/test 70/20/10 temporal split with
  `gap=lookahead` embargo.
- Volatility regime: 4-class from ATR quantiles computed on the train fold only.
- Meta-label: XGBoost on triple-barrier outcomes of the primary model
  (`src/training/meta_labeling.py:134`).
- Confidence: Platt/isotonic from journal outcomes (`src/risk/confidence_calibration.py:100`).

**Validation:** walk-forward (`walkforward_validation.py`: n_splits=5, rolling/expanding,
purged k-fold purge_gap=24, embargo_gap=12); sequential cost-aware backtest (fill at NEXT bar open,
SL-first intra-bar, spread+slippage, `src/training/backtest_harness.py:80-150`); expectancy gate
(fail-closed: ≥20 trades, balanced acc ≥ 0.52, expectancy ≥ 0 pips after costs, class share ≤ 0.85,
`src/training/expectancy_gate.py:270-302`); the HARD ship gate (`train_single_model_m1.py:61,112-165`
— gap > 0.10 ⇒ every artifact moved to `_quarantine/`, prior known-good restored); and the newer
**gated research harness** (deflated Sharpe, block-bootstrap p, Bonferroni over trial count,
untouched OOS, maxDD ≤ 0.25, ≥ 10y history) used by every 2026-06+ experiment.

### 1.2 Data inventory (real numbers, data pass)

| Class | Granularity | Depth | Source |
|---|---|---|---|
| FX training | M15 | 65,000 candles/pair ≈ 2.6y | `models/USD_JPY/training_status.json` |
| FX daily | D | 19 pairs, 3,230 rows, 2014→2026-06-11 | `market_data/factor/*.csv` |
| Rates | monthly/daily | FRED to 1964 (monthly), 1971 (daily fixes) | `market_data/rates_*.csv`, `fred_daily/` |
| Equity | D | 3,639 × 684 tickers, 2012→2026-06-24 + true-PIT EDGAR fundamentals + survivorship-aware universe | `market_data/equity/*.parquet/json` |
| Multi-asset | D | 21 assets to 1993; 59 assets to **1927** | `market_data/multi_asset/panel*.parquet` |
| Crypto | 1h/1d + funding | 733+ USDT perps (survivorship-aware incl. delisted), 2020→**2026-05-31 (frozen)** | `crypto_cache/` (1,525 parquet, 123 MB) |
| Tick | — | **none usable**: `src/data/tick_capture.py:35` defines `trained_data/ticks` — directory does not exist; only a 5-day Dukascopy debug file (313,581 ticks, 2020) | verified by ls |

**Execution/trade logs:**
- `trade_journal_rl.json`: **205 trades** (2026-04-03 → 07-01). **Only 18 carry full agent/gate
  context** (agent votes, gate details, spread, slippage, regime); the 2026-06+ trend-lane records
  deliberately have `agents/gates/confidence = null`.
- `oanda/transactions.jsonl`: 600 records, **188 ORDER_FILLs with realized `pl` and a 7-level
  bid/ask ladder + per-level liquidity** — real microstructure at fill time, currently unused by
  any training path.
- `virtual_trades.jsonl`: 25 gate-rejected setups (sampled record had empty `features`/`agent_scores`).
- `equity/cycle_ledger.jsonl`: 37 hash-chained records; latest decision `refuse` on `global_halt` —
  the fail-closed lane behaving correctly.

**Retired FX transformer shape:** input (batch, 90, 50) — 90 M15 bars × 50 selected features —
from the quarantined artifact's `arch.json`/`meta.pkl`; last run train 0.683 / val 0.523 /
**gap 0.160 → quarantined** (`training_summary.json`). The full inference contract (scaler,
selected indices, regime quantiles, calibration) rides in the meta sidecar.

**Learning-state artifacts:** `agent_weights.json` sits essentially at `_BASE_WEIGHTS`
(4th-decimal drift only) — the RL loop has had ~18 usable observations, far below learning scale.
`SHIP_GATE.json`: equity harvester PASS net Sharpe 0.906 / maxDD 0.229 (curated 20-name universe;
the survivorship-corrected wide-universe number is **0.740 full / 0.355 OOS, gate FAIL** — per the
2026-07-01 independent audit, always report both). 76 backtest result JSONs, negatives and all.

### 1.3 The coin-flip diagnosis — answered; do not re-litigate

The question "bad features, bad methodology, or leakage?" has a documented, verifier-confirmed
answer: **all three defects existed, all three were fixed, and fixing them revealed the floor is
the market.**

| Defect found | Class | Fix | Evidence |
|---|---|---|---|
| Anchored features (obv, cum_returns, atr_log, vol_log leaked window position) — inflated "56–70%" val acc | leakage | pipeline v2 window-invariance + canary test | L-001; commit dad8624 |
| Ship-gate metric path broken (`get_metrics()` didn't exist ⇒ gap reported 0 ⇒ gate a permanent no-op); train_acc read at last epoch not best-val epoch | methodology | `_read_trainer_metrics` + best-val-epoch contract + `_quarantine_if_overshipped` | `.claude/rules/improvement.md` Hard Ship Gate; commit 1a05e75 |
| Double-fit StandardScaler (identity transform), feature-selection/scaler mismatch, regime one-hots missing at inference — 6 simultaneous train↔inference contract violations | methodology + skew | inference contract in meta sidecar, `_subset_scaler`, version refusal | improvement.md Train↔Inference Contract Gates |
| Sample-weight forwarding dropped | methodology | fixed | commit 1a05e75 |
| After ALL fixes: USD_JPY/EUR_USD/GBP_USD ≈ 52% val with >10% gap; news fusion +0.96pp with worse gap; meta-label AUC ≤ 0.53; daily factor gross ≈ 0; own-data 22y walk-forward 50.3% | **market** | none exists at this data scale | L-016/L-020/L-022 + verdict docs in `docs/` |

**Edge decay within two days** is the same phenomenon, not a separate one: a model fit to noise
has no edge to decay — apparent short-lived edge is in-sample memorization surfacing, which is
exactly what the 10% gap gate quarantines. L-020 adds the structural point: infra tuning (costs,
rebalance frequency, overlays) moves cosmetic gate failures but never significance, which is gated
by effective-N and history length.

### 1.4 Structural weaknesses (the actual modernization targets)

From the control-flow trace (all statuses re-derived from process list + disk this audit):

1. **Learning welded to the UI process.** RL weight sync
   (`execution.py:5301→_team.py:1045`), confidence recalibration (`engine.py:1072-1077, 4223`),
   Tier-6 overrides (`engine.py:436, 3588`), ConfigAdjuster consumption
   (`embedded_scanner.py:970`), AlertManager checks, and drift detection all run only inside the
   engine scan cycle — whose only live driver is the interactive TUI. TUI closed ⇒ all learning
   silently stops, while `heartbeat.json` (written by the *supervisor*) still reports
   `scanner_alive: true`. The beacon lies about what is alive.
2. **Producer-alive / consumer-dead asymmetry.** The running self-heal supervisor stages config
   adjustments (`self_heal.py:631,683`; 10 actions on 2026-07-02 per
   `self_heal_action_budget.json`) that only the dormant TUI process can apply. Pending
   adjustments and `retrain_rl_position_sizer` requests accumulate unconsumed — the $3,527
   dead-write incident class (docs/incidents.md), recurring at process level.
3. **Three divergent post-trade paths + duplicated modules.** `sync_closed_trades_rl`
   (verdict-driven RL), `post_trade_loop.py` (batch, explicitly skips RL, line 10), and
   `trend_journal_sync.py` (no verdicts, RL-skipped by design) coexist; likewise two
   AlertManager-shaped modules (`automation/alert_manager.py` vs `utils/monitoring.py:257`) and
   two confidence-calibration modules (`automation/confidence_calibrator.py` — the one the engine
   actually imports — vs `risk/confidence_calibration.py` cited in doctrine). Every duplication is
   a place a fix lands on the wrong twin.
4. **Detection without action.** `RetrainTrigger.check_drift()` fires every 20 cycles
   (`engine.py:5512-5524`) but only logs; an ACTIVE 7-consecutive-loss WARNING sits unacknowledged
   in `alert_state.json` (2026-07-02T12:58) with no routed remediation. Model decay is observed,
   not answered.
5. **No evaluation gate on the online retrainer.** `online_retrainer.py:251-395` rewrites
   `xgb_momentum/rf_risk/ridge_confidence.pkl` in place with cooldowns (60 min, ≤3/day) but zero
   gap/holdout validation — HARD_MAX_GAP is enforced only at Tier-1 batch training. A bad
   incremental retrain ships silently. (Bounded blast radius: sklearn gate models only, never the
   direction champion — but still ungated.)
6. **Two generations of doctrine stitched together.** The new lanes have the right primitives
   (fail-closed `_lane_halted`, hash-chained ledgers, pre-registration, operator-token LiveGate);
   the legacy FX loop still runs on global-halt/journal/RL machinery. Per-lane control was
   retrofitted days ago (commit 193d847) and `state.json:last_actor` still says
   "per-lane-control-pending". Every safety invariant currently must be maintained twice.
7. **The richest learning signal is starving.** Only 18/205 journal records carry the full
   agent-vote→outcome context RL learning needs; the active trend lane writes nulls by design; the
   188 real fills with 7-level book ladders feed nothing.
8. **MetaManager live wiring UNVERIFIED.** The 9-stage change pipeline
   (`automation/meta_manager.py`) and StagedDeployer soak gates (shadow ≥15 trades / canary ≥30 /
   R-floor baseline−0.5R, `staged_deployer.py:39-45,242-286`) are invoked from tests and library
   code; no instantiation was found in the running entry points this pass.

---

## Phase 2 — Modernization architecture

Design constraint from doctrine (INTENT #4, L-022): **no proposal below promises that better
features or loss functions beat the ~52% wall on free bars.** ML's legitimate remaining surfaces
here are (a) risk/regime estimation, (b) abstention/meta-decision quality, (c) alpha research
gated on *materially new inputs* (an operator spend decision), and (d) the durable meta-asset:
the gated harness itself.

### 2.1 Signal & feature layer

- **Freeze direction-prediction feature work on free bars.** The v3 pipeline is contract-sound
  (window-invariant, versioned, canary-tested). Keep it as the reference implementation; invest
  nothing further in squeezing direction accuracy from it (L-022).
- **Redirect feature engineering to risk targets** where skill demonstrably exists and the
  efficient-market wall does not apply: volatility forecasting (TCN vol-regime head already
  exists), drawdown-state estimation, and regime classification feeding sizing/overlays. These
  improve the *risk-premium lanes* (trend, harvester) — the only strategies that cleared or
  near-cleared gates — without needing to beat 52%.
- **Regime detection, properly framed:** regime features must drive *risk posture* (vol-target
  scaling, overlay activation, lane de-risking), never direction confidence inflation. The
  existing regime one-hot + quantile contract is the right shape; thread `regime_quantiles`
  end-to-end (flagged UNVERIFIED in the code-map pass — verify before relying).
- **Decay is managed structurally, not by feature cleverness:** shadow-first forward records
  (already the lane pattern), staleness rails (the >7-day component-age hard-block exists), and
  the gap gate. A "decay dashboard" per lane = rolling OOS Sharpe of the frozen rulebook vs its
  gate-time expectation, from the hash-chained ledgers.
- **New-input alpha research (operator-gated spend):** the only lever class the campaign left
  open — paid PIT fundamentals extensions, options-implied signals, and microstructure. The free
  precursor: **activate the existing `tick_capture.py` against the practice account** so a real
  tick/spread/book store accrues from now (costs nothing, creates the "materially new input"
  L-022 requires before any future intraday research is even admissible). Note the 188 existing
  ORDER_FILL ladders as seed data for a fill-quality model (slippage vs book state) — an
  execution-cost model is legitimate ML that pays regardless of alpha.

### 2.2 Training methodology

- **Codify the gated research harness as a library** (`src/research/gated_harness.py`):
  pre-registration record (hash-pinned params before results — the hypothesis_registry already
  does this for brain_loop), frozen-OOS split, deflated Sharpe with explicit trial count N,
  block-bootstrap p with Bonferroni, maxDD + history-length gates, and the survivorship/
  data-honesty label block that the EDGAR results already carry. Today this logic is re-implemented
  per `scripts/experiment_*.py`; byte-identical reuse was verified between crypto and EDGAR scripts
  — make that structural instead of copy-discipline.
- **Loss functions / reward shaping:** the current stack already uses the modern set — class-
  balanced loss, label smoothing, magnitude-shaped sample weights, EMA, EWC, SWA, triple-barrier
  meta-labels, PPO with Sharpe/drawdown-shaped reward. Adding fancier losses cannot beat a
  signal-free target (L-022) and is explicitly out of scope. The one methodological upgrade worth
  making: **CPCV (combinatorial purged cross-validation)** in the harness for experiments where
  path-dependence matters, and regime-stratified OOS reporting so a gate-pass can't hide inside
  one regime (the trend-sleeve verdict showed 2/4 sub-block dependence — make that check standard).
- **Online vs batch:** keep batch as the champion path (it's where the gate lives). Online
  updates remain bounded to sklearn gate models — but MUST gain an eval gate: holdout slice +
  refuse-to-write on degradation (same fail-closed shape as `_quarantine_if_overshipped`).
  Non-stationarity is handled by regime conditioning + staleness rails + shadow forward records,
  not by faster weight updates (faster adaptation to noise is how edge "decays in two days").
- **Live-vs-backtest reconciliation:** a standing job comparing realized fill costs
  (transactions.jsonl `pl`, ladder, slippage) against the harness cost model (spread+slippage
  pips). The cost model was judged "about right" by the literature pass — keep it honest with own
  fills as the sample grows.

### 2.3 Runtime learning architecture — the Lane Contract

Unify both generations on the pattern the new lanes already proved. **Lane Contract v1** — every
strategy lane implements:

1. `lane_id` + fail-closed halt read (`_lane_halted`; missing/corrupt entry ⇒ halted) — exists.
2. Hash-chained append-only cycle ledger — exists (`cycle_ledger.py`, `hypothesis_registry.py`).
3. Pre-registered, hash-pinned strategy params (mutation ⇒ new hypothesis) — exists for
   brain_loop/track_b; retrofit for trend + harvester params.
4. Ship gate keyed to universe/config hash, re-verified per cycle (NO_ACT on drift) — exists.
5. Shadow-first forward record; promotion to live **only** via operator-token LiveGate
   (`live_gate.py`: gate_pass + typed "LIVE" + risk ceilings + kill-switch constructable) — exists.
6. **NEW — honest liveness:** per-lane heartbeat stating exactly which loop is alive
   (`scanner_alive` must mean the scanner; supervisor liveness is a separate key). Extends
   `running_status.py`'s two-lane honesty to all lanes.
7. **NEW — decay monitor:** rolling forward-record vs gate-time expectation; breach ⇒ lane
   de-risk/halt proposal (the brain_loop `derisk.py` tighten-only pattern), never silent.

**Headless Learning Supervisor** (the fix for weaknesses 1–4): extract post-trade feedback out of
the TUI into a bounded daemon (same shape as `run_tier7_loop.py`, which proved the pattern):
- One canonical post-trade path replacing the three divergent ones: journal sync → RL weight
  update (when verdict context exists) → calibrator refit check → alert evaluation → ack/route.
- It *consumes* approved config adjustments (closing the producer/consumer asymmetry) under the
  existing approval rail (`adjustment_approver.py` bypass guard stays).
- Drift/alert events route to a bounded action (self-heal request at clamped autonomy, or a
  surfaced operator card) with an acknowledgment ledger — no more unacknowledged 7-loss warnings.
- Read-only with respect to trading: it never places orders, never unhalts (only the TUI's
  operator path and the human can), stays out of the hot path per doctrine. Claude remains
  outside the runtime loop entirely.
- Retire-by-merge the duplicate modules (one AlertManager, one calibrator), keeping the engine's
  import (`automation/confidence_calibrator.py`) as canonical and migrating doctrine references.

**Journal enrichment:** trend-lane records should carry the context that exists even without the
15-agent consensus — regime, ATR, spread at entry, realized slippage, sizing inputs — so the
forward record is trainable later. Cheap, additive, no behavior change.

### 2.4 Multi-asset readiness

Already substantially proven: the lane pattern runs FX-trend (OANDA practice), equity (shadow +
IBKR-paper gate), crypto (research + shadow lane code), and filing-text (track_b) today, each with
its own data layer (`src/equity/`, `src/crypto/data_layer.py`) and survivorship-aware universes.
Scaling = new `DataLayer` + `LaneRunner` implementing the Lane Contract; the gate math, ledgers,
LiveGate, and per-lane halt are asset-agnostic already. Futures: the IBKR broker abstraction
exists (483 tests) — a futures lane is a Lane Contract instance away once operator-authorized.
The one structural caveat to carry into every new asset class: effective-N and history length gate
significance (L-020) — breadth must be real, not cosmetic.

---

## Phase 3 — Implementation roadmap

Ordering principle: honesty/wiring fixes first (cheap, no operator decision, close active lies),
then consolidation, then operator-gated levers. Nothing below touches the hot path without an
explicit escalation; everything ships with no-mock tests + separate-verifier PASS per DoD.

### P0 — Truth and safety patches (days; no operator decision needed)
| Item | Fixes weakness | Validation checkpoint |
|---|---|---|
| Heartbeat honesty: `scanner_alive` only when the engine loop is actually alive; supervisor gets its own key; extend `running_status.py` to all lanes | #1 (the lying beacon) | from-disk check: TUI down ⇒ `scanner_alive:false` |
| Eval gate on `online_retrainer` (holdout + refuse-to-write on degradation, quarantine-shaped) | #5 | injected bad retrain refused in test |
| Alert routing + ack ledger (route the standing 7-loss WARNING class to a surfaced action) | #4 | active alert acquires ack/action record |
| Trend-lane journal enrichment (regime/ATR/spread/slippage fields) | #7 | next trend cycle writes enriched record |
| Verify `regime_quantiles` threading end-to-end; verify MetaManager live wiring (or record as dormant-by-design) | #8 + UNVERIFIED items | grep + one smoke invocation, documented either way |
| risk_monitor per-lane halt coverage (already spun off as background task) | monitoring blind spot | risk_monitor GREEN reflects lanes |

### P1 — Consolidation (1–2 weeks)
1. **Extract `gated_harness` library** from the experiment scripts; port one existing experiment
   as the regression proof (byte-identical results = pass). Add CPCV + regime-stratified OOS.
2. **Single post-trade feedback module**; migrate the three paths; delete-by-merge duplicate
   AlertManager/calibrator twins. Checkpoint: one grep-provable call path from trade close →
   weights/calibration/alerts.
3. **Headless Learning Supervisor** (bounded daemon consuming approved adjustments, running the
   canonical feedback path). Checkpoint: TUI closed for 24h ⇒ adjustments still consumed, RL sync
   still fires on closed trades, heartbeat stays honest. Escalation note: touches
   `execution.py`-adjacent feedback call sites ⇒ operator review before merge.

### P2 — Operator-decision levers (surface, don't start)
- **Activate `tick_capture.py`** against the practice account (free; builds the "materially new
  input" precondition for any future intraday research; zero hot-path contact).
- **Fill-quality/execution-cost model** from the 188+ accumulating ORDER_FILL ladders (legitimate
  ML, pays regardless of alpha).
- **Paid data** (PIT fundamentals extension / options-implied) — the only remaining alpha-lever
  class per L-022; spend decision is the operator's card.
- **Crypto cache refresh cadence** (currently frozen at 2026-05-31) if crypto lanes are to keep
  accruing honest forward records.

### P3 — Lane unification (after P1 proves out)
- Migrate legacy FX machinery onto the Lane Contract (per-lane ledger + pre-registered params for
  trend lane; retire global-only code paths); finish the joint-fallback deprecation steps 3–5
  already tracked in `.claude/rules/improvement.md`.
- Lane decay dashboards in AXIOM from the hash-chained ledgers.
- Multi-asset expansion (futures lane) if/when operator authorizes — Lane Contract instance +
  DataLayer, nothing architecturally new.

### What this roadmap deliberately does NOT contain
- FX direction retrains or "one more" free-data alpha hunt (L-016/L-022).
- Transformer scaling/architecture upgrades (signal-bound, not capacity-bound — the SOTA
  modernization review already rejected this, docs/sota-modernization-review-2026-06-15.md).
- Any relaxation of HARD_MAX_GAP, halt semantics, practice pin, or ship-gate-before-champion.
- Autonomous promotion to live anywhere — shadow→live stays operator-token-gated by construction.

---

## Appendix A — confidence ledger for load-bearing claims

| Claim | Confidence | Source read this audit |
|---|---|---|
| 52% wall is market, not pipeline | HIGH | L-001/016/020/022 + verdict docs + 4 independent negatives |
| Learning loops dormant without TUI | HIGH | ps + heartbeat + call-site greps (control-flow pass, 2026-07-03) |
| Self-heal producer / dead consumer asymmetry | HIGH | self_heal.py:631,683 + consumer call sites + action budget file |
| Online retrainer ungated | HIGH (confirmed by two passes) | online_retrainer.py:251-395 |
| Only 18/205 trades carry full RL context | HIGH | journal key-level inspection (data pass) |
| Tier-6 "not purely shadow" (engine.py:3588 consumes overrides) | MEDIUM | one consumption site traced |
| MetaManager wiring — RESOLVED 2026-07-03: WIRED-LIVE via Orchestrator dispatch (orchestrator.py:451-458, drain at :1507) invoked from the TUI scan loop; real events in .claude/meta/changes.jsonl through 2026-07-02. Shares the "welded to TUI" condition (weakness #1), not dead code. | HIGH | P0 investigation, file:line + ledger read |
| regime_quantiles threading — RESOLVED 2026-07-03: fully threaded (trainer writes transformer_trainer.py:3230; gates reads gates.py:1269; inference uses gates.py:2194-2219; quarantined USD_JPY artifact carries real q25/q50/q75). The 2026-06-10 "unthreaded" suspicion was fixed by the later train_single_model_m1.py threading commit. | HIGH | P0 investigation incl. pickle read of real artifact |

## Appendix B — discovery provenance
Four read-only passes on 2026-07-02/03: training-code map (Explore), data inventory
(Data Engineer), retrain-triggers/promotion-gates (specialist), learning-control-flow +
new-lanes trace (Software Architect). Statuses per L-017: every "running/dormant" claim above was
re-derived from `ps`/heartbeat/launchd + disk artifacts at audit time, not from memory.
