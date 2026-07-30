# Phase J2–J6 — non-FX lane audit (equity, crypto momentum, crypto carry, Track B, hedge)

**Audited:** 2026-07-30 · **Method:** read-only. Every claim was re-derived from files read or
commands run **in this session**. No roadmap status line was accepted as evidence. Integration
greps excluded the module's own package and `tests/` before anything was called "wired".

**Checkout audited:** `/home/user/ml_engine` (Linux), HEAD `c73a70d` on branch
`claude/axiom-training-audit-uuy5wp`. The last commit that touched lane *data* is `074758d`
(2026-07-21). The launchd definitions in `scripts/axiom_launchd/*.plist` target
`/Users/buddy/Documents/ml_engine` (macOS) and their output dirs (`trained_data/axiom/`,
`crypto_cache/`, `trained_data/oanda/`) are gitignored and **absent here**. Where a claim depends
on the operator's machine, it is tagged. Where a claim depends on committed artifacts, it is HIGH.

**Test evidence produced this session** (pytest 9.0.2 + pandas 3.0.5 installed to a scratchpad
`--target`; no project file modified):

| Suite | Result |
|---|---|
| 5 lane evidence slices (`test_{crypto_carry,crypto_momentum,track_b,hedge,equity_research}_evidence_slice.py`) | **79 passed** |
| 3 shadow lanes (`test_{crypto_momentum,crypto_carry,track_b}_shadow.py`, `-m "not integration"`) | **50 passed, 6 deselected** |
| 4 hedge suites (`hedged_shadow_lane`, `portfolio_exposure`, `candidates`, `exposure_history`) | **52 passed** |
| 2 hedge suites (`hedge_scorecard`, `exposure_tags`) | **33 passed** |
| 3 equity suites (`universe`, `corporate_actions`, `decision_gate`) | **81 passed** |

**295 tests pass, 0 fail.** The libraries are real and correct. That is not the problem.
The 6 deselected crypto tests are `@pytest.mark.integration` and require `crypto_cache/`,
which does not exist in this checkout — so no crypto number below could be recomputed here.

---

## 0.0 CORRECTION — the checkout changed underneath this audit

**I was wrong about two findings, and I am correcting them rather than reframing them.**

At ~15:44 UTC, `git status` revealed that a **concurrent writer** (another agent session plus a
live scanner process) had been modifying this working tree *during* the audit. Two claims I made
at ~15:23–15:35 were true when read and are **no longer true**:

| Claim as originally written | Status at 15:23–15:35 (when read) | Status at 15:44 (re-derived) |
|---|---|---|
| All 5 lanes return `lane_not_configured` | **TRUE** — `.claude/state.json` had `halted: false` and no `halted_lanes` key | **SUPERSEDED.** `state.json` was rewritten at `15:41:29Z`: `halted: true`, `config_dirty: true`, and `halted_lanes` now present with all 6 `KNOWN_LANES` set to `true`. All lanes now return `(True, True, 'global_halt')` — an **explicit, deliberate** halt. |
| `scripts/capture_track_b_new_filings.py` and `scripts/check_risk_target_p2_readiness.py` never existed | **TRUE** — `ls` failed, `git log --all -- <path>` was empty | **SUPERSEDED.** Both files now exist as untracked, authored at 15:40 and 15:42 by a parallel remediation effort (the new file's docstring cites "audit §3.2, gap G8" — a different audit document). |

Also concurrently modified: `src/crypto/carry_scorecard.py`, `src/crypto/momentum_scorecard.py`,
`src/hedge/{hedge_scorecard,hedged_shadow_lane,exposure_history}.py`, `src/equity/backtest.py`,
`src/equity/research/pit_text_loader.py` — a drawdown-convention refactor introducing
`src/research/drawdown_convention.py`. **Line numbers in `carry_scorecard.py` shifted by +2**
(`REQUIRED_TAIL_SCENARIOS` is now `:25`, `MAX_TRACKING_ERROR` `:22`, `MIN_CAPACITY_USD` `:23`).

**What I re-verified at 15:44 and confirmed UNCHANGED — every structural finding in this report:**

| Re-derived claim | Result |
|---|---|
| Tail-scenario producers for J4 | Still exactly 2 files (`carry_scorecard.py`, `crypto_carry/manifests.py`) — schema + consumer, **no producer** |
| `corporate_actions.py` callers | Still exactly 1 file, `tests/test_equity_corporate_actions.py` — **zero production callers** |
| `VALID_ASSET_CLASSES` | Still `("fx", "equity")` at `portfolio_exposure.py:52` — **crypto still unsupported** |
| Hedge slippage | `grep -rn "slippage" src/hedge/` still **empty** |
| All 5 lane ledger row counts | **Unchanged**: 1 / 1 / 1 / 3 / 141 (`git diff --stat` on all five data dirs is empty) |

**Net effect on the report:** the *forward-record famine* — the actual subject of this audit — is
untouched. What changed is the *diagnosis of the immediate halt cause* (gap X-1 below) and the
existence of two ingestion scripts (gaps X-2 / J5-1). Those three entries are rewritten in place.
Everything else stands as written.

**Methodological lesson worth recording:** this checkout is not quiescent. Any future audit here
must snapshot (`git stash create` or a worktree) before reading, or timestamp every claim.

---

## 0. THE HEADLINE — all five lanes are halted, and the halt is now explicit

Re-derived at 15:44 UTC:

```
>>> from src.equity.decision_gate import _lane_halted
equity / crypto_momentum / crypto_carry / track_b / oanda_fx / brain
  → (True, True, 'global_halt')   # all six
```

`.claude/state.json` (written `2026-07-30T15:41:29Z`) now carries `halted: true` plus an explicit
`halted_lanes` dict with all six `KNOWN_LANES` set to `true`. Every lane-scoped runner refuses on
the first rail — before any gate, ship-gate, or significance check is consulted.

Thirty minutes earlier the same call returned `'lane_not_configured'` for every lane, because
`halted_lanes` did not exist. `src/equity/decision_gate.py:120-123` treats a missing
`halted_lanes` dict as HALTED by design ("a lane must be explicitly unhalted; it is never assumed
safe to run by omission") — a correct fail-closed default that had been silently doing all the
blocking. `src/scanner/automation/state_engine.py:32` declares
`KNOWN_LANES = ("oanda_fx", "equity", "brain", "crypto_momentum", "track_b", "crypto_carry")`.

**Either way — implicit or explicit — the operative fact for this audit is unchanged: no lane has
been blocked by an evidence verdict. The 43 `global_halt` refusals in
`trained_data/equity/cycle_ledger.jsonl` are the halt rail firing, not a gate failing.** The
distinction between "the evidence says no" and "the halt rail is on" is what the rest of this
report keeps apart.

---

## 1. The (a) vs (b) split — what is honestly shadow vs what is merely unwired

The operator's directive is "everything is to be live, I am not taking shadow as an answer."
Some of these lanes cannot honestly go live. Some are only shadow because a script is missing.

### (a) Shadow because the EVIDENCE says there is no demonstrated edge — promoting these would be dishonest

| Lane | The number on disk | Source read this session |
|---|---|---|
| **Crypto momentum (J3)** | `clears_ex_history = FALSE`; significance FAILS at both N=15 and N=3 | `src/crypto/momentum_shadow.py:87` (`GATE_VERDICT`), `docs/experiment-crypto-edge-hunt-round2-2026-06-29.md` §7 |
| **Crypto carry (J4)** | `clears_ex_history = FALSE` — "turnover cost dominates a real funding premium". The single logged cycle: gross carry **+0.093%**, cost **+0.162%**, **net −0.069%/day** at 81% daily turnover | `src/crypto/carry_shadow.py:97`; `trained_data/crypto_carry/shadow_carry_ledger.jsonl` |
| **Track B (J5)** | Gate **FAIL** on every arm. Post-cutoff: net Sharpe 0.587 but `positive_years=1/2`, `history_length=2 <10`, `walk_forward=false`. DSR 0.071, bootstrap p=0.249 vs Bonferroni α=0.0023 | `trained_data/research/track_b_postcutoff_2026_07_02/harness_result_primary_2bps.json` |
| **Equity quality sleeve (J2 candidate)** | Wide survivorship-corrected: quality tilt net Sharpe **0.14** vs EW baseline **0.74** (full), **0.031 vs 0.355** (OOS). Quality **loses** to equal-weight in all 3 disjoint sub-periods | `trained_data/backtests/pit_quality_bakeoff.json` |

For these, "make it live" is not a wiring task. Forcing a promotion would require either
overriding a pre-registered gate or re-running until a favourable number appears — both are the
L-018 failure mode the repo's own pre-registration discipline exists to prevent. **The correct
answer for these four is: keep them shadow, and fix the reason they cannot accumulate evidence
(which IS a wiring problem — see (b)).**

One important nuance, stated in the lane's own ledger and preserved here: Track B's verdict is
`INSUFFICIENT`, **not** `NO EDGE`. The post-cutoff rank-IC point estimate is +0.09 (p=0.119,
n=291, 1 rebalance) with a placebo of −0.027. That is a signal that has not been *tested*, not
one that has been *refuted*. The binding constraint is forward time and rebalance count — and
the job that would generate them does not exist (§6, GAP J5-1).

### (b) Shadow ONLY because nobody finished the wiring — these are the real targets

| # | What | Evidence it is a wiring gap, not an evidence gap |
|---|---|---|
| **b1** | **All 5 lanes hard-blocked at the halt rail, never at a gate** | §0 + §0.0. Until 15:41 the block was an *implicit* `lane_not_configured` (no `halted_lanes` key); it is now an *explicit* `global_halt`. Neither is an evidence verdict. Unblocking is an operator state decision, not research. |
| **b2** | **`scripts/capture_track_b_new_filings.py` was missing for the entire life of the lane — written 2026-07-30T15:42 by a parallel effort, still untracked and unrun** | `scripts/run_forward_capture_daily.py:17` has invoked it as job 1 of 4 with `accepted_codes={0}` since the `com.buddy.forward_daily` LaunchAgent was written. `git log --all -- <path>` was **empty** at 15:35. The new file's own docstring states: *"the file never existed — so that job exited 2 every night and no filing ever reached the canonical contract on a schedule."* **Track B has ingested zero new filings since 2026-07-02**, and this was the cause. The fix now exists on disk but has never been executed or committed. |
| **b3** | **`scripts/check_risk_target_p2_readiness.py` was missing — written 2026-07-30T15:40, still untracked and unrun** | Same graph, job 4 (`scripts/run_forward_capture_daily.py:20`). |
| **b4** | **Crypto momentum + crypto carry shadow runners are on NO schedule** | `scripts/run_crypto_momentum_shadow.py` and `run_crypto_carry_shadow.py` exist and pass their tests, but appear in **no** plist and **not** in `run_forward_capture_daily.py`'s job tuple. Result: **1 forward cycle each, ever.** |
| **b5** | **The equity harvester runner is on NO schedule** | `scripts/run_equity_harvester.py` appears in no plist. `trained_data/equity/cycle_ledger.jsonl` has 141 rows across only **2 distinct as-of dates** (2026-06-24, 2026-07-01). |
| **b6** | **`src/equity/corporate_actions.py` (991 lines) has ZERO non-test callers** | `grep -rn "corporate_actions" --include=*.py .` returns exactly 2 hits, both in `tests/test_equity_corporate_actions.py`. Splits, cash dividends, delistings and cash-M&A are modelled, tested (81 passing), and **never applied to any book or backtest**. |
| **b7** | **All 5 lane evidence packages have a dashboard READER and no PRODUCER** | `src/evidence/{crypto_momentum,crypto_carry,track_b,hedge_eval,equity_research}/worker.py` are each ~300 lines, fully tested (79 passing), and referenced from exactly one place outside their own package: `dashboard/server/training_cockpit.py:72-76` — a *reader* spec. Only `risk_target` has a producer entry point (`scripts/run_risk_target_evidence_slice.py:34`). |
| **b8** | **OKX + Hyperliquid venue cross-check fetchers have zero callers** | `src/crypto/data_layer.py:318,360,377,392` define `fetch_okx_ohlcv`, `fetch_okx_funding_recent`, and the Hyperliquid meta/candle fetchers. `grep -rn "okx\|hyperliquid" src/ scripts/` outside `data_layer.py` → **empty**. "Venue and contract normalization" (J3 upgrade #3) is defined and unused; Binance is the sole consumed venue. |
| **b9** | **`src/hedge/config/sector_bucket_map.json` has 34 tickers** | This is why the Track B hedge cycle came back `fail_closed` with `unknown_ticker:A, ADI, ADP, AME, ANET, CFG, CIEN, CME`. Extending a seed JSON map is data entry, not research. |
| **b10** | **`hedged_shadow_lane` cannot price the equity book** | The one equity hedge cycle logged `insufficient_forward_price_data:20_of_20_tickers_unresolved`. `market_data/equity/sp500_prices.parquet` ends before the hedge as-of date. A price-panel refresh is a data task. |
| **b11** | **`refresh_crypto_forward_data.py` persists no execution receipt** | It `print`s a JSON summary and exits (`scripts/refresh_crypto_forward_data.py:66-70`). There is no state file, no manifest write. **There is no way to prove from disk that it ever ran** — and `discover_cached_symbols()` returns empty (exit 2) on any machine without `crypto_cache/`. |

**b1–b5 and b11 are, together, the entire reason four of these lanes have ≤3 forward records.**
None of them is a research question.

---

## 2. Forward-record reality check — real numbers from real files

| Lane | Ledger path | Rows | Distinct as-of dates | First → last as-of | Last write (`cycle_ts`) | Days since |
|---|---|---:|---:|---|---|---:|
| Equity harvester | `trained_data/equity/cycle_ledger.jsonl` | 141 | **2** | 2026-06-24 → 2026-07-01 | n/a (hash-chain only) | 29 |
| Crypto momentum | `trained_data/crypto/shadow_momentum_ledger.jsonl` | **1** | 1 | 2026-05-31 | 2026-07-02T23:18:05Z | 28 |
| Crypto carry | `trained_data/crypto_carry/shadow_carry_ledger.jsonl` | **1** | 1 | 2026-05-31 | 2026-07-07T10:38:22Z | 23 |
| Track B | `trained_data/research/track_b_shadow_ledger.jsonl` | **1** | 1 | 2026-06-24 | 2026-07-03T01:07:23Z | 27 |
| Hedge raw-vs-hedged | `trained_data/hedge/raw_vs_hedged_ledger.jsonl` | **3** | 3 | 2026-05-31 → 2026-07-01 | 2026-07-08T02:32:38Z | 22 |
| Hedge decisions | `trained_data/hedge/hedge_decision_log.jsonl` | **3** | 3 | same | same | 22 |
| Hedge exposure history | `trained_data/hedge/exposure_history.jsonl` | **file absent** | — | — | — | — |

Every one of these files was committed in a single commit (`074758d`, 2026-07-21) and has not
changed since. `git log --format=%ad -- <path>` returns exactly one commit for each.
**Confidence HIGH.**

The three hedge rows share a `cycle_ts` to the same second (`2026-07-08T02:32:38.779/.780/.781`) —
they are one batch run of `run_all()`, not three days of accumulation. `fx_trend` is in
`hedged_shadow_lane.STRATEGIES` (`:146`) but produced **no row**.

Equity's 141 rows resolve to: `decision=continue` 96, `refuse` 43, `abstain` 2. All 43 refusals
carry reason `global_halt`; the 2 abstentions carry `stale_data:8d>7d`. Total orders across all
141 cycles: **22**.

---

## 3. J2 — Equity harvester and sleeve program

### 3.1 Data on disk

| Artifact | Path | Real content (read this session) |
|---|---|---|
| PIT universe snapshot | `market_data/equity/universe_snapshot_pit.json` | 4.93 MB. `built_at` 2026-07-02T05:06:02Z, `pipeline_version` `2026-06-18-eq1`, `universe_hash` `896fc636…`. 199 monthly reconstitution dates 2010-01-04 → 2026-07-01. 77,920 `(date, ticker)` members over 3,896 daily dates, **20 names/day**, **37 unique tickers over 16 years**. |
| PIT fundamentals | `market_data/equity/sp500_pit_fundamentals.json` | 2.22 MB, **624 tickers**, `source: EDGAR-XBRL`, `pit: true`, keyed on `filed` date. |
| PIT fundamentals (ext) | `market_data/equity/sp500_pit_fundamentals_ext.json` | 3.81 MB, **629 tickers**, adds `equity`, `shares_outstanding`, CFO/Assets. |
| Price panel | `market_data/equity/sp500_prices.parquet` | Present. Could not be opened here (pandas 3.0.5 vs the file's writer); its staleness is proven indirectly by the hedge lane's `20_of_20_tickers_unresolved`. |
| Rebalance state | `trained_data/equity/rebalance_state.json` | `RB-20260701T000000+0000-896fc636`, 20 names at exactly 0.05 each, `orders: []`. |

**Universe snapshot is 28 days stale** (last reconstitution 2026-07-01; next due 2026-08-01).

### 3.2 Survivorship, corporate actions, PIT, delisting

| Requirement | Status | Evidence |
|---|---|---|
| Survivorship-aware universe | **PARTIAL — two universes exist, and the one the SHIP_GATE uses is the biased one** | `src/equity/sp500_membership.py:1` builds a genuinely survivorship-aware S&P PIT membership *including since-removed names* — but its docstring is tagged `running:NO` and it is **unreachable** from `scripts/run_equity_harvester.py` (§3.5). The deployed `universe_snapshot_pit.json` comes from `run_equity_harvester_shipgate_pit.py`, whose `CANDIDATE_POOL` is a hand-listed set of **currently-listed yfinance tickers**. That script's own docstring (`:22-25`) says: *"yfinance only serves currently-listed tickers, so names delisted before today are absent… it cannot fully remove survivorship… Treat a PASS here as 'necessary, not yet sufficient'."* |
| Corporate actions | **ISOLATED** | `src/equity/corporate_actions.py` — 991 lines, splits / cash dividends / delisting / cash-M&A, `ACTION_DELIST` at `:62`, `delisted_tickers` in the result at `:414`. **Zero non-test callers** (b6). 81 tests pass. Not applied to any backtest or book. |
| PIT fundamentals | **LIVE (as data)** | Real EDGAR-XBRL, filing-date aligned, `pit: true`. Consumed by `pit_quality_eval` (offline). |
| Delisted names retained | **NO for the harvester, YES for the quality bake-off** | `pit_quality_bakeoff.json` → `universe.since_removed_included: 108`, `ever_members_in_window: 873`, `evaluated_with_price_and_pit_fundamentals: 624`. The harvester's 37-name pool retains none. The bake-off's own `data_honesty.residual_bias_labelled` names **two** further survivorship channels honestly: price-side (yfinance lacks delisted tickers) and a fundamentals-side gap where `load_cik_map()` sources SEC's *current-tickers-only* registry, so a delisted constituent (confirmed: TWX) drops out of the fundamentals panel entirely. |
| Sector classifications | **PARTIAL** | Present in `src/hedge/config/sector_bucket_map.json` (34 tickers, self-labelled `SEED map — deliberately partial`), and in the quality bake-off's panel. No PIT sector history. |
| Transaction-cost estimates | **LIVE** | `cost_bps: 2.0` + `slippage_bps_per_pct_adv: 5.0` with `adv_supplied: true`, `execution_lag: 1` — `equity_harvester_singlestock_20260702T050603Z.json`. |
| Capacity / liquidity | **PARTIAL** | ADV20$ is the universe ranking metric and drives the slippage model. No explicit capacity ceiling. |

### 3.3 The curated-vs-wide divergence — both numbers, from disk

The roadmap says the curated and wide results "differ materially". They do. Here is every
number I could find, with its exact source:

| Construction | Universe | Net Sharpe | Max DD | Positive yrs | Gate | Source file |
|---|---|---:|---:|---:|---|---|
| Single-stock harvester, **curated** top-20-ADV | 37 unique tickers, currently-listed only, `hash 896fc636` | **0.906** | 0.229 | 13/17 | **PASS** | `trained_data/backtests/SHIP_GATE.json` |
| Same, extended through the GFC | `hash 26c92e1d` | **0.800** | 0.229 | 17/22 | PASS | `SHIP_GATE_gfc_2005_2026.json` |
| Harvester alone, **OOS** (2019-07→2026-05, 34%) | curated | 1.035 | 0.229 | 7/8 | **FAIL** (8 yrs < 10) | `SHIP_GATE_book.json` |
| **EW baseline, wide survivorship-corrected** | 873 ever-members / 624 evaluated / **108 since-removed included** | **0.740** | 0.195 | 12/15 | PASS | `pit_quality_bakeoff.json` |
| Same, **OOS** (2021-07→2026-06) | wide | **0.355** | 0.195 | 5/6 | **FAIL** | `pit_quality_bakeoff.json` |
| **Quality tilt**, wide | wide | **0.140** | 0.387 | 7/15 | **FAIL** | `pit_quality_bakeoff.json` |
| Quality tilt, wide, **OOS** | wide | **0.031** | 0.315 | 3/6 | **FAIL** | `pit_quality_bakeoff.json` |
| Combined book (harvester + trend sleeve) | curated | 0.873 full / 1.153 OOS | 0.166 | 16/21 | PASS full, FAIL OOS | `SHIP_GATE_book.json` |

**The honest read:** 0.906 (curated, 37 survivor names) vs 0.740 (wide, 873 ever-members with
108 delisted retained) is a **−0.17 Sharpe survivorship haircut on the equal-weight baseline
alone** — and that wide number still understates the true haircut, because the bake-off's own
`data_honesty` block states delisting returns are UNDERSTATED (forward-fill flattens the final
gap) and that two further survivorship channels remain. Corroborating tell: the GFC-extended
curated run reports **2008 = −15.8%** for a long-only US large-cap book against an S&P 500 that
fell ~37%. Some of that is the `dd_hard=0.20` overlay; not all of it.

**No wide-survivorship run of the harvester construction itself exists on disk.** The
comparison above is EW-baseline-vs-harvester across two different universes — the closest
available, and explicitly not like-for-like. That is itself a J2 gap (GAP J2-2).

### 3.4 The 9 required reporting fields

| # | Field | Status | Where |
|---|---|---|---|
| 1 | Curated-universe result | **LIVE** | `SHIP_GATE.json` net_sharpe 0.906 |
| 2 | Wide survivorship-corrected result | **PARTIAL** | Exists for EW baseline + quality tilt (`pit_quality_bakeoff.json`); **does not exist for the harvester construction** |
| 3 | Full-history result | **LIVE** | 2010-2026 (17y) and 2005-2026 (22y) |
| 4 | OOS result | **LIVE** | `SHIP_GATE_book.json` `out_of_sample` block, fraction 0.34 |
| 5 | Drawdown | **LIVE** | 0.229 curated / 0.195 wide-EW |
| 6 | Turnover | **LIVE** | `avg_turnover 0.0241`, `total_turnover 99.97` |
| 7 | Concentration | **PARTIAL** | Computed at *runtime* only — `src/equity/risk_agents.py:25` Herfindahl + top-K, `concentration_cap=0.30`, `concentration_top_k=3`. **Not a reported backtest field.** The book is EW-20 (5%/name) by construction. |
| 8 | Beta | **ABSENT** | `grep -n "beta" src/equity/backtest.py src/factor/ship_gate.py src/equity/harvester_strategy.py` → only the phrase "equity-beta harvester" in a docstring. No beta is computed or reported anywhere in the equity lane. |
| 9 | Marginal portfolio contribution | **LIVE** | `SHIP_GATE_book.json`: `net_sharpe_margin` +0.033 full / +0.118 OOS, plus 4 disjoint sub-periods where the combined book beats the harvester in only **2 of 4** (`margin` −0.021 and −0.064 in 2011-2015 and 2016-2020). |

**7 of 9 present in some form; beta ABSENT; wide-universe result exists only for a different construction.**

### 3.5 Reachability — what actually runs

Computed this session by walking `src.*` imports from `scripts/run_equity_harvester.py`.
**13 of 52 `src/equity/*` modules are reachable.**

| Reachable (13) | Unreachable (39, selected) |
|---|---|
| `backtest`, `cycle_ledger`, `data_loader`, `decision_gate`, `harvester_strategy`, `live_gate`, `order_lifecycle`, `rebalance`, `risk_agents`, `runner`, `ship_gate`, `strategy`, `universe` | **`corporate_actions`**, **`control_loop`** (1094 L), **`shadow_pipeline`** (792 L), **`kill_switch`** (883 L), `market_calendar` (696 L), `depth_pricing` (571 L), `executors` (911 L), `shadow_fills` (599 L), `reflection`, `multi_horizon`, `variant_eval`, `pit_quality_eval`, `sleeve_combiner`, `hrp`, `lowvol`, `quality`, `quality_data`, `trend_sleeve`, `multi_asset_trend`, `value_data`, `edgar_fundamentals`, **`sp500_membership`**, all 9 `research/*` |

`control_loop` is imported once, by `src/tui/data_provider.py:84` — **for path constants only**.
`shadow_pipeline` and `kill_switch` have no non-test importers at all.

### 3.6 Live-promotion path

`src/equity/live_gate.py` is a genuine, well-built gate: ship-gate + universe-hash binding,
typed `"LIVE"` token (`:84`), NAV-fraction ceiling, kill-switch and drawdown-guardian
constructability checks. `LiveGate.arm()` has **zero production callers** — by design; arming is
an out-of-band operator action. `scripts/run_equity_harvester.py:162-171` correctly checks
`is_armed()` before returning a real fill callback (a 2026-07-01 audit fix for a path that
previously bypassed it entirely).

**So the equity harvester's live blockers are, in order: (1) `lane_not_configured`, (2) LiveGate
not armed, (3) IB Gateway not running on 127.0.0.1. None of the three is an evidence problem —
its ship gate is `gate_pass: true`.**

---

## 4. J3 — Crypto momentum

### 4.1 The 7 frozen construction parameters — all pinned, all by import

| # | Param | Value | Pinned at | Frozen? |
|---|---|---|---|---|
| 1 | 14-day cross-sectional momentum | `LOOKBACK_D = 14` | `src/crypto/momentum_shadow.py:74` ← `_h.SIGNALS["momentum"]["lookback"]` | YES |
| 2 | Long/short quintiles | `QUINTILE = 0.20` | `:76` ← `_h.QUINTILE` | YES |
| 3 | Weekly rebalance | `REBALANCE_DAYS = 7` | `:82` ← `_round2.REBALANCE_D` | YES |
| 4 | Funding-aware P&L | `carry_pnl = (-Wprev * fr).sum(axis=1)` | `:155` and `:165`; `gross = price_pnl + carry_pnl` | YES |
| 5 | Volatility targeting | `TARGET_ANN_VOL = 0.10`, `VOL_WINDOW = 30`, `VOL_TARGET_ENABLED = True` | `:79`, `:80`, `:83` | YES |
| 6 | Maximum leverage | `MAX_LEV = 3.0` | `:81` ← `_h2i.MAX_LEV` | YES |
| 7 | Explicit costs | `COST_BPS = 10.0` | `:77` ← `_h.COST_BPS` | YES |

**Freeze discipline: STRONG.** Nothing is redefined locally — every constant is imported from
the verifier-confirmed experiment harness, with a comment naming the source doc
(`docs/experiment-crypto-edge-hunt-round2-2026-06-29.md#3-h4`, 258 lines, present). `_round2.REBALANCE_D`
is imported rather than hardcoded. `construction_manifest()` (`:92-108`) serializes all 7 into
every ledger row — verified present in the one row on disk. A regression test
(`test_momentum_shadow_regression_matches_frozen_harness`) binds the math to `backtest_flex`.

**Residual risk (LOW severity):** the freeze is by *import from a mutable experiment script*, not
by a signed manifest hash. Editing `scripts/experiment_crypto_round2.py:REBALANCE_D` would
silently retune the "frozen" lane. The regression test would catch the math change; nothing
catches a constant change that both sides agree on.

### 4.2 The 10 professional upgrades

| # | Upgrade | Status | Evidence |
|---|---|---|---|
| 1 | Automated monthly/daily refresh | **PARTIAL** | `com.buddy.crypto_refresh.plist` runs `refresh_crypto_forward_data.py` **monthly only** (`Day 2, 03:20`). **No daily refresh exists.** And the script writes no state (b11) — there is no disk-provable execution record. On this machine `discover_cached_symbols()` returns empty → `return 2`. |
| 2 | Data-freshness manifest | **ISOLATED** | `data_layer.coverage_report()` (`:436-461`) computes per-symbol funding start/end/rows + a `delisted` flag. **Zero callers** — never persisted, never surfaced. |
| 3 | Venue and contract normalization | **ISOLATED** | OKX (`:318`, `:360`) and Hyperliquid (`:377`, `:392`) fetchers exist with zero callers (b8). Binance is the only venue consumed. |
| 4 | Delisted-symbol retention | **PARTIAL — mechanism real, loss magnitude not** | `data_layer.py:10-18`: `list_binance_perp_symbols()` enumerates the static archive which retains LUNAUSDT/FTTUSDT etc. Its own caveat: *"the dumps stop emitting funding/klines when a symbol delists; we close at last observed price, which UNDERSTATES true delisting losses (no −100% liquidation gap is modeled)."* Same understatement class as the equity lane. |
| 5 | Liquidity and capacity limits | **ABSENT for momentum** | No ADV/capacity screen in `momentum_shadow.py`. (Carry has `MIN_ADV_USD = 10M`.) |
| 6 | Funding and basis stress | **ABSENT** | `grep basis_blowout` across the repo hits only `carry_scorecard.py` and `evidence/crypto_carry/manifests.py` — the *schema*, not a producer. |
| 7 | Separate forward/backtest ledgers | **LIVE** | `LEDGER_PATH_DEFAULT` is forward-only (`:89`); `forward_oos_summary` (`:357-390`) explicitly annotates *"annualized from forward shadow cycles only (post-activation), NOT the backtest"* and refuses a Sharpe at n<2. Genuinely correct. |
| 8 | Effective-N monitoring | **ABSENT for the forward lane** | `effective_n` appears in Track B's harness output (44.65) but nowhere in `momentum_shadow.py` or `momentum_scorecard.py`. |
| 9 | Cost and leverage sensitivity | **ISOLATED** | Sensitivity dimensions are declared in `src/evidence/crypto_momentum/evaluation.py:154` ("every declared adverse funding/basis/outage stress must remain above the floor") but that evaluator has no producer (b7). |
| 10 | Exchange outage scenarios | **ABSENT** | Only referenced as a *registry* outage in the evidence worker's fail-open comment (`worker.py:72,277`) — an unrelated meaning. |

**2 of 10 LIVE, 3 PARTIAL, 3 ISOLATED, 2 ABSENT.**

### 4.3 Data + forward record

Cached data lives in `crypto_cache/` — **gitignored (`.gitignore:151 *_cache/`) and absent from
this checkout**, so no crypto row count, coverage window, or symbol count could be verified here.
**Confidence LOW on crypto data volume; HIGH on everything committed.**

What is committed: 1 forward cycle, `asof_date 2026-05-31`, `universe_size 116`, 23 longs / 23
shorts, `gross_leverage 0.397`, `today_net_return +1.024%`, `cumulative_shadow_return +1.024%`,
`orders_placed 0`. The **32-day lag between `asof_date` (2026-05-31) and `cycle_ts` (2026-07-02)**
is structural: the free Binance monthly dump means the last well-populated date trails the
present by up to a month (`momentum_shadow.py:246-250` documents exactly this).

**Downstream:** feeds `hedged_shadow_lane.load_crypto_momentum_book` (`:238`) → returns
`unsupported_asset_class`. Feeds nothing else. Effectively a dead-end.

---

## 5. J4 — Crypto carry

### 5.1 The 11 required data-model fields

| # | Field | Status | Note |
|---|---|---|---|
| 1 | Spot instrument | **ABSENT** | The book keys on perp symbols only. `ShadowCycleResult.today_price_return` is hardcoded *"always 0.0 — delta-neutral by construction"* (`carry_shadow.py:187`). The spot leg is **assumed to be a perfect hedge**, not modelled. |
| 2 | Perpetual instrument | **LIVE** | Symbol keys in the book. |
| 3 | Venue | **ABSENT** | No venue field anywhere in the ledger row or `construction_manifest()`. Binance implied. |
| 4 | Funding timestamp | **PARTIAL** | Funding index is daily-resampled in `data_layer`; not carried into the ledger. |
| 5 | Basis | **ABSENT** | Not computed. Funding rate is used as the entire carry proxy. |
| 6 | Margin requirement | **ABSENT** | Schema only (`carry_scorecard.MAX_MARGIN_UTILIZATION = 0.70`, `:21`). No producer. |
| 7 | Liquidation threshold | **ABSENT** | Same — schema only. |
| 8 | Borrow / transfer cost | **ABSENT** | Not modelled. |
| 9 | Fees | **PARTIAL** | Single lumped `COST_BPS_ROUNDTRIP = 20.0` (`carry_shadow.py:89`) for both legs. Not decomposed. |
| 10 | Liquidity | **PARTIAL** | `MIN_ADV_USD = 10_000_000` eligibility screen (`carry_shadow.py:87`). |
| 11 | Counterparty-risk classification | **ABSENT** | `carry_scorecard` has `VERDICT_MISSING_COUNTERPARTY` (`:38`) as a gate. No producer emits a classification. |

> Line numbers in `carry_scorecard.py` shifted +2 at 15:42 (concurrent drawdown-convention
> refactor, §0.0). Values above re-verified at 15:44.

**2 of 11 present, 3 partial, 6 absent.**

### 5.2 The 12 required evaluations

| # | Evaluation | Status | Evidence |
|---|---|---|---|
| 1 | Gross funding capture | **LIVE** | `today_carry_return = +0.000931` in the one ledger row. |
| 2 | Net capture after both legs | **LIVE** | `today_net_return = −0.000691` after `today_cost = 0.001622`. |
| 3 | Rebalance and turnover | **LIVE** | `today_turnover = 0.811` at `REBALANCE_DAYS = 1`. |
| 4 | Spot-perp tracking error | **ABSENT** | Set to zero by construction (`carry_shadow.py:187`). `carry_scorecard.MAX_TRACKING_ERROR = 0.05` (`:22`) is a bar with nothing measuring against it. |
| 5 | Margin requirements | **ABSENT** | Schema only. |
| 6 | Basis blowout | **ABSENT** | `REQUIRED_TAIL_SCENARIOS[0]` (`:25`). No producer. |
| 7 | Liquidation stress | **ABSENT** | `REQUIRED_TAIL_SCENARIOS[1]`. No producer. |
| 8 | Venue failure | **ABSENT** | `REQUIRED_TAIL_SCENARIOS[2]`. No producer. |
| 9 | Withdrawal suspension | **ABSENT** | `REQUIRED_TAIL_SCENARIOS[3]`. No producer. |
| 10 | Stablecoin depeg | **ABSENT** | `REQUIRED_TAIL_SCENARIOS[4]`. No producer. |
| 11 | Cross-venue settlement risk | **ABSENT** | `REQUIRED_TAIL_SCENARIOS[5]`. No producer. |
| 12 | Capacity | **ABSENT (as an evaluation)** | `MIN_CAPACITY_USD = 10_000` (`:21`) is a bar; `capacity_usd` is a required *input* the scorecard validates (`:197-200`). Nothing computes it. |

### 5.3 "Is there any stress code at all, or only a Sharpe number?"

**There is a rigorous stress *contract* and zero stress *computation*.**

`src/crypto/carry_scorecard.py` is a genuinely strong gate: it refuses a verdict on Sharpe alone.
`REQUIRED_TAIL_SCENARIOS` (`:25-32`) enumerates all six tails; the aggregate requires *every* one
to have a valid record; `VERDICT_MISSING_TAIL` / `VERDICT_MISSING_COUNTERPARTY` (`:37-38`) are
first-class outcomes. The design is exactly what J4 asks for.

But `grep -rln "basis_blowout" --include=*.py .` (re-run at 15:44) returns **exactly two files**:
`src/crypto/carry_scorecard.py` (the consumer) and `src/evidence/crypto_carry/manifests.py`
(the schema). **No module in this repo produces a tail-scenario record.** The 79 passing evidence
tests exercise the gate with test-constructed inputs.

Meanwhile the *actual* backtest is candid about it —
`scripts/experiment_crypto_cash_and_carry.py:16`: *"exchange-solvency/liquidation tail not
[modeled]"* — and `carry_shadow.py:100` carries a `RISK_PREMIUM_NOTE` citing BIS WP 1087 and
stating this is *"a risk premium with a fat tail, not free money."*

**Classification: the gate is LIVE (as code, correct, tested). Every one of its 6 mandatory
inputs is ABSENT. The lane can therefore never clear its own gate, which is the correct
fail-closed behaviour.**

### 5.4 Frozen construction

Same import-only discipline as J3: `ANN`, `LOOKBACK_D=3`, `MIN_ADV_USD=10M`, `MIN_HISTORY_D=90`,
`COST_BPS_ROUNDTRIP=20`, `TARGET_ANN_VOL=0.10`, `VOL_WINDOW=30`, `MAX_LEV=3.0` all imported from
`scripts/experiment_crypto_cash_and_carry.py` (`carry_shadow.py:85-94`). Pre-reg doc
`docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md` exists (135 lines) and is cited as
*"frozen BEFORE the backtest ran"*. `REBALANCE_DAYS = 1` is the one locally-set constant (`:93`).

---

## 6. J5 — Track B

### 6.1 The 9-stage pipeline

| Stage | Status | Evidence |
|---|---|---|
| SEC filing retrieval | **PARTIAL / BROKEN for forward** | `scripts/track_b_fetch_and_blind_postcutoff.py` exists and produced the 2026-07-02 batch. The **daily** ingestion job `scripts/capture_track_b_new_filings.py` did not exist for the entire life of the lane; it was authored at 2026-07-30T15:42 by a parallel effort and is **untracked and never executed** (b2, §0.0). Zero filings have reached the lane on a schedule, ever. |
| Document hash | **ABSENT in production artifacts** | `_index.json` task records carry only `as_of, form, original_chars, scored_chars, ticker, truncated, audit`. No hash. |
| PIT cutoff | **LIVE** | `harness.py:810-838` splits `pre_cutoff` / `post_cutoff` strictly on `MODEL_CUTOFF = 2026-02-01`; `load_frozen_scores` (`track_b_shadow.py:169`) drops every `as_of < MODEL_CUTOFF` from the live book. |
| Blinding | **LIVE, with an honest residual** | `src/equity/research/entity_blinder.py` (733 L). The `audit` block on task_0000 reports `counts: {company 37, date 48, ticker 361}`, `any_leak_signal: true`, `leak_candidate_count: 399`, and states its own heuristic is *"NOT exhaustive"* and that *"the load-bearing lookahead control is the post-cutoff arm, not this blinder."* |
| Scoring | **PARTIAL — manual** | `meta.model`: *"claude-sonnet-5 (this session + dispatched Data Engineer subagents acting as scorers; **no ANTHROPIC_API_KEY configured for an automated fan-out**)"*. |
| Score validation | **PARTIAL** | Range/type validation in `scripts/track_b_collect_scores.py:_valid()` (4 fields, explicit bounds) and `ResearchScore.validate()` (raises on out-of-range — deliberately loud, `track_b_shadow.py:188-190`). Abstentions dropped, never zero-filled: `n_abstained: 81`. |
| Cross-sectional construction | **LIVE** | Q5 top-20% equal-weight long-only, `min_names_for_quintile: 5`. |
| Portfolio backtest | **LIVE** | `harness_result_primary_2bps.json` + `harness_result_stress_5bps.json` (both arms, 2bps and 5bps). |
| Placebo | **LIVE** | `placebo_rank_ic: −0.0274` (p=0.642) alongside the real +0.0915. Clean. |
| Forward shadow record | **PARTIAL** | 1 row, 27 days stale. |

### 6.2 The 12 required lineage fields — contract vs reality

`src/evidence/track_b/lineage.py:28-33` declares `REQUIRED_SCORE_LINEAGE` with all 15 fields
(`filing_accession, ticker, composite, document_hash, filing_date, as_of_date, model_cutoff,
scorer_model, scorer_version, prompt_digest, parameters, rationale, extracted_spans,
score_validation, batch_id`). The contract is complete and correct.

**But the scorer that satisfies it is `SCORER_MODEL = "track-b-frozen-lexicon"` (`:14`) — a
deterministic keyword-matching lexicon (`POSITIVE`/`NEGATIVE`/`OUTLOOK_*` tuples at `:23-26`).
It is not the scorer that produced the 420 scores on disk.**

Actual score record on disk (`scores_artifact.json`, first entry, read verbatim):

| Required field | Present in the real artifact? |
|---|---|
| Filing accession | **NO** |
| Document hash | **NO** |
| Filing date | **NO** (only `as_of`) |
| As-of date | **YES** (`as_of`) |
| Model cutoff | **file-level only** (`meta.model_cutoff`), not per score |
| Scorer model + version | **file-level only**, and free-text, unversioned |
| Prompt digest | **NO** |
| Parameters | **NO** |
| Rationale | **YES** |
| Extracted spans | **YES** (`spans_used`, 4 verbatim spans) |
| Score validation | **NO** (validation happens at load, result not persisted) |
| Batch identity | **NO** (batch files are written then the mapping is discarded) |

**4 of 12 present, 2 at file level only, 6 absent.** The lineage module that *would* carry all 12
is never run against a real filing. **Classification: lineage contract ISOLATED; production
scores PARTIAL.**

### 6.3 The professional upgrades

| Upgrade | Status | Evidence |
|---|---|---|
| Automated SEC ingestion | **ABSENT → newly PARTIAL (unverified)** | b2 — the script referenced by the daily graph never existed until 15:42 today. It now exists untracked and unrun; its own docstring confirms *"that job exited 2 every night and no filing ever reached the canonical contract on a schedule."* |
| Explicit scoring queue | **PARTIAL** | `scripts/track_b_build_score_batches.py` writes `score_batches/batch_NNN.txt` + a manifest, but `OUT_DIR` is **hardcoded** to `trained_data/research/track_b_postcutoff_2026_07_02`. One-shot, not a queue. |
| Idempotent scoring | **PARTIAL** | `load_frozen_scores` dedupes by `(ticker, as_of)` with last-write-wins (`track_b_shadow.py:154`, `:194`) — idempotent at *load*. Scoring itself has no idempotency key; re-running a batch re-scores. |
| Retry / rate-limit handling | **ABSENT** | Scoring is in-session LLM dispatch; no retry, no backoff, no rate-limit code. |
| Manual-review queue for malformed/extreme scores | **ABSENT** | Malformed → silently dropped (`collect_scores._valid()` → abstention). Out-of-range on a well-formed record → **raises and stops the pipeline** (`track_b_shadow.py:188-190`, deliberately). Neither routes anywhere for review. |
| Model-cutoff enforcement | **LIVE — real code** | `harness.py:810-838` (arm split), `track_b_shadow.py:169` (live-book filter), `contracts.py:57-58` (`ARM_PRE_CUTOFF`/`ARM_POST_CUTOFF`). Not prose. |
| Contamination tests | **LIVE — real code** | Three independent channels: the pre-cutoff arm as the contamination arm; the un-blinded demo arm (`scorer.py:147,234` — *"DEMONSTRATES lookahead contamination (a pretrained model can recall…)"*); and the placebo. |
| Rebalance-date tracking | **PARTIAL** | `rebalance_step_days: 21`; `cross_sectional_ic_spread.json` has **`n_rebalances: 1`** and `ic_std: null`, `ic_ir: null` — one rebalance cannot produce an IC information ratio. |
| Forward eval independent of the scoring worker | **PARTIAL** | `track_b_shadow.py` reads frozen artifacts and never invokes a scorer — architecturally independent. But it has ticked once. |

### 6.4 A discrepancy worth naming

`load_frozen_scores()` run live in this session returns **428 post-cutoff scores** across 54
distinct `as_of` dates (2026-02-05 → 2026-07-01). The single forward ledger row records
`n_scored_filings: 48` and a `coverage_note` claiming *"48 post-cutoff scored filings vs ~405
needed for 80% power"*. The score inventory now on disk (428) **exceeds** the stated power
requirement (405), yet the lane's own recorded self-assessment still says 48. Nobody has
reconciled this because the lane has not ticked since 2026-07-03. **Confidence HIGH on both
numbers (computed/read this session); MEDIUM on the interpretation** — the 48 may be a
per-rebalance cross-section rather than a cumulative count, which the note's phrasing does not
make clear. Either way the note is stale and is currently the lane's headline justification for
staying shadow.

---

## 7. J6 — Hedge and exposure

### 7.1 The 10 exposure taxonomy classes

`src/hedge/portfolio_exposure.py:52`: `VALID_ASSET_CLASSES = ("fx", "equity")`.

| # | Class | Status | Evidence |
|---|---|---|---|
| 1 | FX currency buckets | **LIVE** | `currency_bucket_map.json`, 9 entries (USD, JPY, CHF, EUR, GBP, AUD, NZD, CAD, XAU), risk-on/safe-haven/commodity tags. Reuses `trend_risk_gates.currency_legs` — not forked. |
| 2 | Equity market beta | **PARTIAL** | `market_beta` in `sector_bucket_map.json`, whose own `_meta` says *"approximate/illustrative, not a live data feed"*. 34 tickers. |
| 3 | Equity sectors | **PARTIAL** | 11 sectors mapped; only 34 tickers carry a sector (b9). |
| 4 | Crypto beta | **ABSENT** | Not in `VALID_ASSET_CLASSES`. |
| 5 | Stablecoin exposure | **ABSENT** | No code path. |
| 6 | Exchange/venue exposure | **ABSENT** | No code path. |
| 7 | Funding/basis exposure | **ABSENT** | No code path. |
| 8 | Carry liquidation exposure | **ABSENT** | No code path. |
| 9 | Correlation clusters | **LIVE** | `correlation_bucket_map.json`, 11 sectors → 4 macro clusters; FX uses the currency itself as its bucket. Confirmed working: the equity cycle logged *"20 position(s) net to 4 correlation bucket(s)"*. |
| 10 | Shared liquidity risk | **ABSENT** | No code path. |

**3 modelled (1 fully), 2 partial, 5 absent. The roadmap's claim that "crypto is currently
unsupported as a complete exposure class" is CONFIRMED — and understates it: 4 of the 5 absent
classes (crypto beta, stablecoin, venue, funding/basis, carry liquidation) are all crypto-side.**

The lane handles this honestly rather than fabricating: `hedged_shadow_lane.py:56-62` states
crypto *"naturally comes back `fail_closed` / `DECISION_SHADOW_ONLY` via P1-P3's own existing
fail-closed contract… rather than fabricating a crypto hedge."* Verified in the ledger: the
crypto row has `hedge.status: "unsupported_asset_class"`, `hedged == raw`.

### 7.2 Are raw-vs-hedged forward records being written? How many?

**3 rows, all written in one batch at `2026-07-08T02:32:38`. Zero of the 3 produced a usable
raw-vs-hedged differential.**

| Strategy | as-of | Decision | Hedge status | Raw net return | Hedged net return |
|---|---|---|---|---|---|
| equity_harvester | 2026-07-01 | `APPROVE_RAW` | `applied` (short SPY, $103k) | **`null`** — `insufficient_forward_price_data:20_of_20_tickers_unresolved` | **`null`** — `overlay_unresolved:1_of_1_legs_unresolved` |
| crypto_momentum | 2026-05-31 | `SHADOW_ONLY` | `unsupported_asset_class` | +0.010242 | +0.010242 (identical — no hedge) |
| track_b | 2026-06-24 | `SHADOW_ONLY` | `fail_closed` (8 unknown tickers) | +0.001264 | +0.001264 (identical) |

`fx_trend` is listed in `STRATEGIES` (`:146`) and produced no row at all.

`trained_data/hedge/hedge_scorecard_report.json` confirms: **all three** scorecards report
`hedged_net.n: 0` and `verdict: "insufficient_history"`. Two say *"raw and hedged lanes are
identical by construction; no alpha-vs-beta read is possible from this history."*

`trained_data/hedge/exposure_history.jsonl` does not exist (its scheduled writer
`com.buddy.exposure_history.plist` → `scripts/run_exposure_history_capture.py` reads
`trained_data/oanda/account_state.json`, which is gitignored and absent here — **unverifiable
from this checkout, tagged UNKNOWN**).

### 7.3 The 8 training targets

| # | Target | Status | Where computed | Value produced? |
|---|---|---|---|---|
| 1 | Hedge effectiveness | **LIVE (code)** | `hedge_candidates.py:122-123` `exposure_reduction` / `_pct` | Yes: 103000.0 / 1.0 for the equity row |
| 2 | Hedge cost | **PARTIAL** | `:164-183` `_evaluate_cost` → real `src.data.execution_cost_model` | `cost_known: false`, `cost_source: "insufficient_data"` — **honestly unknown, not faked as zero** (no OANDA fills exist for SPY) |
| 3 | Residual exposure | **PARTIAL** | Inferred from `exposure_reduction_pct`; no explicit residual field | — |
| 4 | Concentration | **LIVE (code)** | `concentration_warnings` in every exposure report | `[]` (0 warnings) |
| 5 | Hedge slippage | **ABSENT** | `grep -rn "slippage" src/hedge/` → **empty** | — |
| 6 | Drawdown reduction | **LIVE (code)** | `hedge_scorecard.py:147` `_max_drawdown`, raw vs hedged in `_three_way_decision` (`:216`) | n=0 → `null` |
| 7 | Tail-risk reduction | **PARTIAL** | `sortino_annualized` (`:186`) is a downside-deviation proxy. No CVaR / VaR / explicit tail metric | n=0 → `null` |
| 8 | Does hedging improve after-cost expectancy | **LIVE (code)** | `decision_readout` / `_three_way_decision` (`:216-254`), three-way raw vs hedged-net vs hedged-gross | `insufficient_history` |

**5 of 8 implemented, 2 partial, 1 (slippage) absent — and all 8 evaluate to `null` because
n=0.** There is no trainer or model consuming these; `grep -rln "hedge" src/training/` returns
only `risk_target_readout.py`. "Training targets" is currently a metrics vocabulary, not a
learning loop.

### 7.4 Required comparison (raw vs hedged vs combined)

Raw and hedged lanes both exist in the ledger schema. **A combined-portfolio lane does not exist
at all** in `hedged_shadow_lane.py` — the module runs per-strategy (`run_cycle_for_strategy`,
`:697`) and `run_all` (`:824`) iterates, never nets. `SHADOW_NOTIONAL_DEFAULT` is explicitly *"a
modeling constant, not a real account… there is no single shared real NAV across three
independent shadow lanes"* (`:40-45`).

---

## 8. Deliverable classification summary

| Lane | ABSENT | ISOLATED | PARTIAL | LIVE | Total |
|---|---:|---:|---:|---:|---:|
| J2 equity (data reqs + 9 reporting fields) | 1 | 1 | 6 | 8 | 16 |
| J3 crypto momentum (7 frozen + 10 upgrades) | 2 | 3 | 3 | 9 | 17 |
| J4 crypto carry (11 fields + 12 evals) | 15 | 1 | 4 | 3 | 23 |
| J5 Track B (9 stages + 12 lineage + 9 upgrades) | 10 | 1 | 12 | 7 | 30 |
| J6 hedge (10 classes + 8 targets) | 6 | 0 | 6 | 6 | 18 |
| **Total** | **34** | **6** | **31** | **33** | **104** |

Definitions used: **LIVE** = code exists, is reachable from a production entry point, and has
produced real output on disk. **PARTIAL** = wired but incomplete, or producing degenerate/`null`
output. **ISOLATED** = complete and tested but with zero non-test callers or no data producer.
**ABSENT** = no implementation.

**Every one of the 6 ISOLATED items and a large share of the 31 PARTIAL items is category (b) —
a wiring gap, not an evidence gap.**

---

## 9. GAP REGISTER

Effort: **S** ≤ 1 day · **M** 2-5 days · **L** > 1 week. Dependency order is within-lane;
the CROSS gaps block everything.

### CROSS-LANE (do these first — they gate all five lanes)

| ID | Gap | Work | Effort | Depends on |
|---|---|---|---|---|
| **X-1** | All 5 lanes blocked at the halt rail (`global_halt` as of 15:41; `lane_not_configured` before that) | **PARTLY DONE 15:41** — `halted_lanes` now exists with all 6 keys. Remaining work: make the per-lane flags reflect intent rather than a blanket `true`. The *research* lanes (crypto_momentum, crypto_carry, track_b) write no orders (`orders_placed: 0` in every ledger row, and `crypto_live_gate.py` has no broker at all) — they can be unhalted for evidence accumulation without any execution risk. Set those three `false`, keep `oanda_fx`/`equity` gated on the operator's own live decision. Use `StateEngine` so key validation runs; do not hand-edit. Verify with the `_lane_halted` one-liner in §0. | **S** | — |
| **X-2** | `run_forward_capture_daily.py` job graph — 2 of 4 jobs referenced scripts that did not exist | **PARTLY DONE 15:40/15:42** — both scripts now exist, **untracked and never executed**. Remaining work: commit them, run each once by hand and confirm exit 0, then confirm the graph returns 0. **Until the graph returns 0, nobody can distinguish a real nightly failure from the standing one.** | **S** | — |
| **X-3** | Crypto momentum, crypto carry, and equity harvester are on no schedule | Add all three to `run_forward_capture_daily.py`'s job tuple (they are already lane-halt-aware and one-shot capable), then reload launchd. Crypto lanes need `--refresh` cadence tuned to the monthly dump. | **S** | X-1 |
| **X-4** | No lane evidence package is ever produced | Write `scripts/run_{crypto_momentum,crypto_carry,track_b,hedge_eval,equity_research}_evidence_slice.py` modelled on the existing `scripts/run_risk_target_evidence_slice.py`. The workers, evaluators, manifests and slices already exist and pass 79 tests; only the driver is missing. | **M** | X-3 (needs ledger rows to package) |
| **X-5** | `refresh_crypto_forward_data.py` leaves no execution receipt | Persist a freshness manifest (symbols, rows, coverage_start/end, per-symbol `delisted` flag, run ts) via the already-written-and-unused `data_layer.coverage_report()`. Atomic write. This also closes J3 upgrade #2. | **S** | — |

### J2 — Equity

| ID | Gap | Work | Effort | Depends on |
|---|---|---|---|---|
| **J2-1** | `corporate_actions.py` (991 L, tested, ISOLATED) applied to nothing | Wire `apply_actions` into `src/equity/backtest.py` and `src/equity/rebalance.py`. Requires an action feed — start with the split/dividend data already implied by yfinance adjusted closes, then add explicit delisting/M&A records. | **M** | — |
| **J2-2** | No wide-survivorship run of the **harvester construction** exists | Re-run `ship_gate` on the `sp500_membership` survivorship-aware universe (873 ever-members, 108 since-removed) instead of the 37-name yfinance pool. Publish both numbers side by side. **Expect the 0.906 to fall.** This is the single most important honesty artifact in J2. | **M** | J2-1 (delisting returns must be real, or the wide number is still optimistic) |
| **J2-3** | Beta is not computed anywhere (reporting field 8 of 9) | Add rolling and full-sample beta vs SPY (or the EW panel proxy already used by the hedge lane) to the ship-gate report payload. | **S** | — |
| **J2-4** | Concentration is a runtime gate, never a reported backtest field | Emit Herfindahl + top-3 share into the ship-gate report from `risk_agents.concentration_check` (already computed). | **S** | — |
| **J2-5** | Universe snapshot 28 days stale; price panel too stale to price any book | Schedule the monthly universe rebuild (first business day) and a daily price-panel refresh. This is also the fix for J6-2. | **S** | X-3 |
| **J2-6** | Fundamentals-side survivorship: `load_cik_map()` uses SEC's current-tickers-only registry (confirmed drop: TWX) | Resolve CIKs from a historical filer registry or cache CIKs at first observation and never re-resolve. | **M** | — |
| **J2-7** | 39 of 52 equity modules unreachable, incl. `kill_switch` (883 L) and `control_loop` (1094 L) | Decide per module: wire, or move to `src/equity/research/` and mark `running:NO` like the rest. **`kill_switch` being unreachable while `live_gate` claims to check its constructability is the one that matters** — `live_gate` constructs it at arm time only. | **M** | — |

### J3 — Crypto momentum

| ID | Gap | Work | Effort | Depends on |
|---|---|---|---|---|
| **J3-1** | 1 forward cycle in ~2 months | Covered by X-1 + X-3. Nothing else needed — the lane is one-shot ready and tested. | **S** | X-1, X-3 |
| **J3-2** | No daily refresh (monthly only) | Add a daily plist for the current-month kline pull; funding stays monthly (that is the dump's cadence, not a bug). | **S** | X-5 |
| **J3-3** | Venue normalization ISOLATED (OKX/HL fetchers, zero callers) | Add a cross-venue reconciliation step that compares Binance funding/close against OKX+HL for the top-N symbols and writes a divergence report. Fail-closed on divergence beyond a threshold. | **M** | X-5 |
| **J3-4** | Delisting loss understated (closes at last observed price, no −100% gap) | Model a terminal delisting return using the last observed funding/price trajectory, or explicitly floor at the exchange's insurance-fund haircut. Currently the backtest is optimistic in exactly the tail that matters. | **M** | — |
| **J3-5** | Effective-N, cost/leverage sensitivity, exchange-outage scenarios all ABSENT/ISOLATED | Implement the three producers the `crypto_momentum` evaluator already demands (`evaluation.py:154`). Without them X-4 cannot produce a passing package. | **M** | X-4 |
| **J3-6** | No liquidity/capacity screen for momentum (carry has one) | Port `MIN_ADV_USD` from `carry_shadow` into the momentum eligibility mask. **This changes the frozen construction — it must be a NEW pre-registration, not an edit to the frozen one.** | **M** | pre-reg doc |

### J4 — Crypto carry

| ID | Gap | Work | Effort | Depends on |
|---|---|---|---|---|
| **J4-1** | Spot leg not modelled — `today_price_return` hardcoded 0.0 | Add the spot instrument to the data model and compute real spot-perp tracking error. **This is the load-bearing gap: the strategy's entire premise is that the spot leg hedges the perp, and that assumption is currently an axiom, not a measurement.** | **L** | — |
| **J4-2** | All 6 mandatory tail scenarios have no producer | Write a `carry_stress.py` producing `basis_blowout`, `liquidation`, `venue_failure`, `withdrawal_suspension`, `stablecoin_depeg`, `cross_venue_settlement` records in the shape `carry_scorecard._tail_record_valid` already expects. Historical episodes exist (FTX Nov-2022, UST May-2022, Binance withdrawal halts) and the delisted-symbol archive retains them. | **L** | J4-1 |
| **J4-3** | 6 of 11 data-model fields absent (venue, basis, margin, liquidation threshold, borrow cost, counterparty class) | Extend `ShadowCycleResult.construction` and the per-symbol record. Venue and counterparty class are static config; margin/liquidation need exchange tier tables. | **M** | J4-1 |
| **J4-4** | Capacity never computed | Compute per-symbol capacity from ADV × participation cap; the scorecard bar (`MIN_CAPACITY_USD`) already exists. | **S** | — |
| **J4-5** | Fees lumped into one 20bps round-trip | Decompose into spot taker + perp taker + funding settlement + transfer. At 81% daily turnover, cost is the whole result — a wrong lump is a wrong verdict. | **M** | — |

> **Honesty note on J4:** even with every gap above closed, the measured result is **−0.069% net
> per day** against **+0.093% gross funding capture** — cost dominates. Closing these gaps will
> most likely *confirm* the negative verdict with better evidence, not overturn it. That is a
> legitimate and valuable outcome; it should not be pitched as a path to promotion.

### J5 — Track B

| ID | Gap | Work | Effort | Depends on |
|---|---|---|---|---|
| **J5-1** | Zero new filings ingested since 2026-07-02 | **PARTLY DONE 15:42** — `scripts/capture_track_b_new_filings.py` now exists (untracked, unrun). It uses `sp500_membership` → `load_cik_map` → `select_pit_filing` → `pit_text_loader.load_pit_filing`, with a per-ticker accession watermark via `ControlStateStore` and a distinct exit 4 for post-fetch canonical-capture failure. Remaining work: **run it once and verify filings actually land**, then commit and schedule. Note it inherits gap **J2-6** — `load_cik_map()` uses SEC's current-tickers-only registry, so delisted issuers are unreachable by construction. | **S** (verify) | X-2 |
| **J5-2** | Scoring is manual (in-session LLM subagents) | Wire `ANTHROPIC_API_KEY` and build an automated fan-out with retry + rate-limit + backoff. The prompt and rubric are already frozen (`PRIMARY_WEIGHTS`, prereg §3.2). | **M** | J5-1 |
| **J5-3** | 6 of 12 lineage fields absent from real score artifacts | Extend the score record: `filing_accession`, `document_hash` (SHA-256 of fetched bytes), `filing_date`, `prompt_digest`, `parameters`, `score_validation` result, `batch_id`. `src/evidence/track_b/lineage.py:28` already declares the exact contract — make the production path satisfy it instead of the lexicon stub. | **M** | J5-2 |
| **J5-4** | No manual-review queue | Route `_valid()` rejections and out-of-range raises to a `review_queue.jsonl` instead of dropping / crashing. | **S** | J5-2 |
| **J5-5** | Batch scripts hardcode `track_b_postcutoff_2026_07_02` | Parameterize `OUT_DIR` in `track_b_build_score_batches.py` and `track_b_collect_scores.py`. Trivially blocks any second run. | **S** | — |
| **J5-6** | Only 1 rebalance → `ic_std`/`ic_ir` are `null` | Falls out of J5-1 + X-3. Needs ~12+ rebalances (≈9 months at 21-day steps) before an IC information ratio means anything. **No amount of engineering shortens this.** | **L** (calendar) | J5-1, X-3 |
| **J5-7** | Ledger's `n_scored_filings: 48` vs 428 loadable post-cutoff scores | Reconcile and rewrite the `coverage_note`. It is currently the lane's headline shadow justification and it does not match the artifacts. | **S** | — |

### J6 — Hedge

| ID | Gap | Work | Effort | Depends on |
|---|---|---|---|---|
| **J6-1** | 3 forward records total, 0 usable comparisons | Covered by X-3 (schedule `hedged_shadow_lane` daily — it is already job 3 of the daily graph, but the graph fails at job 1). | **S** | X-2, X-3 |
| **J6-2** | Equity book unpriceable — `20_of_20_tickers_unresolved` | Refresh `market_data/equity/sp500_prices.parquet` daily. **Without this the hedge lane can never produce a raw-vs-hedged number for the one strategy whose ship gate passes.** | **S** | J2-5 |
| **J6-3** | `sector_bucket_map.json` has 34 tickers → Track B books fail closed | Extend to the full S&P 500 with sector + beta. Data entry from an existing sector classification, not research. | **S** | — |
| **J6-4** | Crypto unsupported as an exposure class (5 of 10 taxonomy classes absent) | Add `"crypto"` to `VALID_ASSET_CLASSES` plus crypto-beta (BTC/ETH proxy), stablecoin, venue, funding/basis, and carry-liquidation buckets with their own bucket maps. This is the "real, separate data-engineering task" the module docstring names. | **L** | J4-3 (venue/counterparty fields must exist first) |
| **J6-5** | Hedge slippage not computed (target 5 of 8) | Add a slippage estimate to `HedgeProposal` from the same execution-cost model, and surface `cost_known=false` rather than 0 where no fills exist (the existing honest pattern). | **S** | — |
| **J6-6** | No combined-portfolio lane | Implement the third leg of the required comparison: net the per-strategy books to a single portfolio on a shared notional and score raw vs hedged vs combined. | **M** | J6-1, J6-3 |
| **J6-7** | Tail-risk reduction proxied by Sortino only | Add CVaR-95 / VaR-95 raw vs hedged to `hedge_scorecard._series_stats`. | **S** | J6-1 |
| **J6-8** | `fx_trend` in `STRATEGIES` produces no row | Diagnose `load_fx_trend_book` (`:287`) — it reads `trained_data/oanda/account_state.json`, which is absent here. Verify on the operator's machine. | **S** | — |
| **J6-9** | No trainer consumes the 8 "training targets" | Decide whether J6 is a monitoring layer or a learning layer. If learning: needs ≥100 hedged cycles before any fit is meaningful. If monitoring: rename the section and drop "training targets". | **S** (decision) | J6-1 |

### Recommended execution order

```
X-1 ─┬─ X-2 ─┬─ X-3 ─┬─ J3-1  (crypto momentum starts accumulating)
     │       │       ├─ J6-1 ─ J6-2 ─ J6-6  (hedge produces a real number)
     │       │       └─ J5-6
     │       └─ J5-1 ─ J5-2 ─ J5-3 ─ J5-4
     └─ X-5 ─ J3-2

parallel, no dependencies: J2-3, J2-4, J2-5, J5-5, J5-7, J6-3, J6-5, J6-8
then:  J2-1 ─ J2-2   (the honesty artifact)
later: J4-1 ─ J4-2 ─ J4-3   (only if the carry lane is worth the L-effort)
```

X-1 through X-5 are **six person-days of work that unblock four lanes**. Everything downstream
of them is calendar-bound: forward evidence cannot be manufactured, only started.

---

## 10. Confidence register

| Claim | Confidence | Basis |
|---|---|---|
| **This working tree was being written by a concurrent process during the audit** | **HIGH** | `git status` at 15:44 vs `ls`/`git log` at 15:23–15:35; file mtimes 15:40/15:41/15:42. See §0.0. |
| All 5 lanes are halted at the halt rail, never at a gate | **HIGH** | `_lane_halted` executed twice against real disk (15:35 → `lane_not_configured`; 15:44 → `global_halt`). Both are the halt rail. |
| Forward-record counts (1/1/1/3/141-rows-2-dates) | **HIGH** | Files read + `git log` per path; **re-confirmed unchanged at 15:44** (`git diff --stat` on all five data dirs empty) |
| `capture_track_b_new_filings.py` / `check_risk_target_p2_readiness.py` were absent for the lane's whole life | **HIGH** | `git log --all -- <path>` empty at 15:35; both untracked files appeared at 15:40/15:42 and the new file's own docstring corroborates the prior absence |
| Those two scripts have been *executed* | **NO** — never claimed | Untracked, no run receipt on disk. Verifying them is GAP X-2. |
| `corporate_actions.py` has zero non-test callers | **HIGH** | Repo-wide grep, 2 hits both in the test file; **re-run at 15:44, unchanged** |
| 5 lane evidence workers have no producer | **HIGH** | Per-lane grep; only `training_cockpit.py` reader specs |
| OKX/Hyperliquid fetchers unused | **HIGH** | Repo-wide grep, zero hits outside `data_layer.py` |
| No tail-scenario producer for J4 | **HIGH** | `grep -rln "basis_blowout"` → 2 files, both schema/consumer |
| Curated 0.906 vs wide-EW 0.740 survivorship haircut | **MEDIUM** | Both numbers read from disk (HIGH). The comparison is across two *different constructions* — it is the closest available, not like-for-like. GAP J2-2 exists to replace it. |
| 295 lane tests pass | **HIGH** | pytest run this session; 6 crypto integration tests deselected (need absent `crypto_cache/`) |
| Crypto data volume / coverage window | **LOW** | `crypto_cache/` gitignored and absent from this checkout — not verifiable from here |
| Whether the operator's launchd agents are loaded | **UNKNOWN** | macOS paths; not inspectable from this Linux clone. The committed ledgers' single-batch timestamps are strong *circumstantial* evidence that nothing has run since 2026-07-08. |
| `exposure_history.jsonl` / `fx_trend` book status | **UNKNOWN** | Depends on `trained_data/oanda/` which is gitignored and absent |
