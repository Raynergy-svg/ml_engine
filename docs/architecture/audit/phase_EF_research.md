# Phase E / Phase F research-methodology audit

**Scope:** `docs/architecture/AXIOM_PROFESSIONAL_TRAINING_ROADMAP.md` §8 (Phase E, line 752),
§9 (Phase F, line 802), §18 (Testing standard, line 1550).
**Method:** every claim below re-derived from files read in this session. Roadmap status text was
NOT used as evidence. Confidence tagged per claim. Read-only — no source file was modified.
**Date:** 2026-07-30. **Auditor:** Model QA Specialist (independent).

---

## 0. Executive verdict

| Deliverable family | Verdict |
|---|---|
| Phase E — 11 harness modules exist | **PARTIAL** (all 11 present + fully implemented, **all 11 ISOLATED**) |
| Phase E — exit gate ("all new research workloads call the same harness") | **FAILED** — zero workloads call it |
| Phase E — migration gate ("port one experiment, byte-identical") | **ABSENT** — not started, no test |
| Phase F — legacy RL repair | **LIVE** — genuinely fixed (roadmap's premise is now stale) |
| Phase F — equity/Track B pushdown | **PARTIAL** — primitive exists, price path unmigrated |
| Phase F — crypto partitioning | **PARTIAL** — history LIVE, cross-section ISOLATED |
| Phase F — baseline report exit gate | **PARTIAL / INVALID** — files exist, measurements don't measure the workload |
| §18 — 5 test categories for research code | 2 solid, 3 partial; **trial-inflation and survivorship adversarial tests = zero** |

**Headline correctness risk:** four independent `max_drawdown` implementations ship under two
opposite sign conventions, and `gated_harness.hard_gate` is the only consumer expecting the
negative one. Demonstrated in this session: an **86.3% drawdown silently PASSES a 25% drawdown
gate** when a positive-convention metric dict is handed to `hard_gate`. Since Phase E's stated
exit gate is to route every existing lane through exactly that function, the migration itself is
the trigger. Confidence **HIGH** (executed, §2.4).

---

## 1. Phase E deliverables — module inventory

`src/research/gated_harness/` exists. All 11 roadmap-named modules are present, plus `__init__.py`.
681 LOC total. **None are stubs** — every module contains working logic with input validation and
fail-closed error paths. Read in full this session.

| Module | LOC | State | Public surface |
|---|---:|---|---|
| `preregistration.py` | 50 | Implemented | `ResearchSpecification` (pydantic `StrictContract`), `hypothesis_digest` @ :47-50 |
| `temporal_splits.py` | 54 | Implemented | `TemporalHoldout`, `split_digest` @ :19-27, `chronological_holdout` @ :30 |
| `purging.py` | 47 | Implemented | `purge_train_indices` @ :10, `purged_kfold` @ :33 |
| `cpcv.py` | 33 | Implemented | `combinatorial_purged_splits` @ :13 |
| `backtest.py` | 55 | Implemented | `validate_returns`, `max_drawdown` @ :23, `summarize_returns` @ :29, `hard_gate` @ :43 |
| `costs.py` | 50 | Implemented | `apply_turnover_costs` @ :18, `evaluate_cost_scenarios` @ :38 |
| `significance.py` | 117 | Implemented | `deflated_sharpe_ratio` @ :19, `circular_block_bootstrap_sharpe_pvalue` @ :57, `corrected_significance` @ :91 |
| `regimes.py` | 18 | Implemented | `stratify_returns` @ :10 |
| `stress.py` | 62 | Implemented | `placebo_control` @ :14, `assert_prefix_causality` @ :29, `stress_matrix` @ :46 |
| `verifier.py` | 41 | Implemented | `IndependentReplayVerifier` @ :20, self-verification refused @ :31-32 |
| `report.py` | 133 | Implemented | `ResearchEvaluationReport`, `evaluate_research` @ :53, `report_digest` @ :48-50 |

### 1.1 Adoption — the load-bearing finding

Integration grep across `*.py *.md *.sh *.toml *.json`, excluding the package itself: the string
`gated_harness` appears in exactly **6 files**, and only **one is executable code**:

- `tests/test_gated_harness.py:12-21` — the only importer.
- `README.md:195`, `docs/ENGINEERING_BRAIN.md:375,419`,
  `docs/training-architecture-audit-2026-07-03.md:239,329`, the roadmap itself — all prose.

Per-symbol external-caller counts (grep `\bsymbol\b` outside the package) resolve to
`tests/test_gated_harness.py` in every case. Symbols with **zero callers anywhere**, including
tests: `circular_block_bootstrap_sharpe_pvalue`, `stratify_returns`, `stress_matrix`,
`evaluate_cost_scenarios` (reached only transitively from `report.py`).
`hypothesis_digest` has exactly one external reference: `tests/test_gated_harness.py:57`.

`git log -- src/research/gated_harness/` returns a single commit (`ae685fa`, evidence foundation)
and nothing since — consistent with *built once, never adopted*.

**Classification: all 11 modules ISOLATED.** Phase E's exit gate ("All new research workloads call
the same harness library rather than copying evaluation code") is **FAILED**, and its migration
gate ("port one already completed experiment and require byte-identical results") has **not been
started** — no ported experiment, no byte-identity test. Confidence **HIGH**.

### 1.2 The harness is itself a fork, not an extraction

`src/research/gated_harness/significance.py:19-117` is a near-verbatim copy of
`src/equity/research/harness.py:550-673`: identical formulas, identical dict keys, identical frozen
constants (`block=21`, `n_reps=5000`, `seed=20260630`, Euler–Mascheroni), and `corrected_significance`
is `_dsr_oos_n22` with `BONFERRONI_ALPHA` replaced by a `family_alpha/n_trials` parameter. The
source module was **not** migrated to import the extraction. Extraction that leaves the original in
place is duplication +1, not consolidation. Confidence **HIGH**.

Note a third parallel stack: `src/evidence/` (contracts, workers, `local_import.py`, envelope
digests) is a genuinely live governance layer implementing its own replay/digest/lineage discipline.
It overlaps Phase E's `verifier.py` and `report.py` in purpose. Any Phase E migration must reconcile
with `src/evidence/`, not ignore it.

---

## 2. The duplication problem — quantified

### 2.1 Inventory of independent implementations

| Statistical gate | Independent implementations | Locations |
|---|---:|---|
| Deflated Sharpe | **4 distinct** (6 sites) | `scripts/experiment_crypto_funding_carry.py:200`, `scripts/experiment_crypto_xs_signals.py:202`, `scripts/experiment_edgar_value_accruals_2026_07_02.py:111` (3 copies of formula A) · `src/equity/research/harness.py:550` (B) · `src/equity/research/harness.py:645` (B wrapper) · `src/research/gated_harness/significance.py:19` (B, forked) |
| Block bootstrap | **3 distinct** (5 sites) | same 3 scripts @ `:213` (mean-based, block=5, seed=12345) · `src/equity/research/harness.py:603` (Sharpe-based, block=21, seed=20260630) · `src/research/gated_harness/significance.py:57` (fork of the latter) |
| Bonferroni / trial budget | **2 live budgets** | `src/equity/research/contracts.py:52` `N_TRIALS=24` → α=0.00208 · scripts `N_TRIALS=3` → α=0.01667 · `gated_harness/significance.py:102` parameterised |
| Purged CV | **2 distinct** | `src/research/gated_harness/purging.py:33` · `src/training/walkforward_validation.py:365` |
| CPCV | **2 distinct** | `src/research/gated_harness/cpcv.py:13` · `src/training/walkforward_validation.py:409` (`CombinatorialPurgedCV`) |
| Cost model | **1 shared + N hardcoded** | `gated_harness/costs.py:18` (isolated) · per-script literals: `experiment_crypto_funding_carry.py:45` `COST_BPS=10.0`, `experiment_crypto_xs_signals.py:41` `10.0`, `experiment_edge_round3_leadA.py:42` `5.0`, `experiment_edge_round3_leadB.py:94` `2.0` |
| Max drawdown | **≥6 distinct** | `gated_harness/backtest.py:23` · `src/factor/backtest.py:103` · `src/equity/backtest.py:70-72` · `src/crypto/momentum_scorecard.py:117` · `src/crypto/carry_scorecard.py:60` · `src/hedge/hedge_scorecard.py:147` |
| Effective N | **2 incompatible definitions** | `src/equity/research/harness.py:510` · `scripts/experiment_edge_round3_leadB.py:71` |

`src/training/walkforward_validation.py` — `purged_kfold_split` and `CombinatorialPurgedCV` have
**zero callers** outside their own file (grep across `src/ scripts/ tests/`). They are dead code,
which lowers today's exposure but leaves a live trap for anyone who reaches for the obvious name.

### 2.2 Do the duplicates agree numerically? — Deflated Sharpe

Executed this session (numpy 2.4.6 / pandas 3.0.5, seeded generators), scripts' formula A vs
gated-harness formula B, **matched at n_trials=3**:

| Series | A (scripts) | B (harness) | Δ |
|---|---:|---:|---:|
| gaussian mild edge | 0.0014 | 0.0014 | −0.0000 |
| neg-skew fat-tail | 0.0010 | 0.0011 | +0.0001 |
| pos-skew | 0.1282 | 0.1263 | −0.0019 |
| t(3) heavy tails | 0.0137 | 0.0135 | −0.0002 |
| short n=60 | 0.0361 | 0.0361 | +0.0000 |

The two formulas differ structurally — A sets the multiple-testing benchmark
`sr0 = E[max]/√(T−1)` (Gaussian SR standard error); B sets `sr0 = E[max]·sr_std` where `sr_std` is
the PSR denominator carrying skew/kurtosis. They coincide only when skew=0 and kurtosis=3.
**Empirically the divergence is small (≤0.002 DSR) at realistic per-period Sharpes.** Confidence
**HIGH** — this one is a documentation/consistency defect, not a live verdict-flipper.

Two secondary divergences in the same pair: admissibility floor is `t<30 → NaN` in the scripts vs
`t<10 → None` in `gated_harness/significance.py:12`, so a 15-bar series gets a real number from one
and an abstention from the other; and `EULER` is 10 d.p. in the scripts vs 16 d.p. at
`significance.py:16` (numerically negligible).

### 2.3 Do the duplicates agree? — Block bootstrap and trial count

**Block bootstrap.** The event `{Sharpe ≤ 0}` and `{mean ≤ 0}` are identical (sd > 0), so the two
estimands coincide; the divergence is entirely **block length 5 vs 21** and seed. Measured Δp on the
same series: up to **0.065** (short n=60 case: 0.803 vs 0.738). At α=0.0021 (N=24) a 6.5pp shift is
not verdict-critical; at α=0.0167 (N=3) it is closer to the margin. The block=5 choice understates
autocorrelation in overlapping-label return streams and is the **more permissive** of the two.
Severity **Medium**. Confidence **HIGH** (measured).

**Trial count — this is the material one.** Two campaign budgets coexist:
`src/equity/research/contracts.py:52` maintains a running count (`N_TRIALS=24`, documented as
edge-round-4=21 → Track B=22 → FINRA +2), while the crypto and EDGAR experiment scripts hardcode
`N_TRIALS=3`. `docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md:78` explicitly declares
"N_trials=3, shared multiple-testing budget"; `docs/prereg-finra-short-volume-2026-07-04.md:112-115`
declares the cumulative 24. **The same repository runs two mutually inconsistent family-wise error
budgets.** Measured effect on the DSR ≥ 0.95 bar, same return series:

| Ann. Sharpe | N=3 (scripts) | N=15 | N=21 | N=22 | N=24 (equity) | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1.39 | 0.9381 | 0.7327 | 0.6807 | 0.6735 | 0.6598 | fail both |
| **1.74** | **0.9844** | 0.8920 | 0.8612 | 0.8567 | **0.8480** | **PASS@3 → FAIL@24** |
| **2.10** | **0.9972** | 0.9679 | 0.9554 | 0.9534 | **0.9497** | **PASS@3 → FAIL@24** |
| 2.55 | 0.9998 | 0.9955 | 0.9931 | 0.9927 | 0.9919 | pass both |

Bonferroni α likewise differs 8×: 0.01667 vs 0.00208. Any strategy landing in the Sharpe 1.7–2.1
band is declared significant by one lane's machinery and insignificant by the other's.
**Severity High — this is trial inflation, the exact failure mode the roadmap's "trial-count
accounting" capability exists to prevent.** Confidence **HIGH** (measured).

Notably, `scripts/experiment_edgar_value_accruals_2026_07_02.py:95-98` carries a comment claiming
the block is "COPIED VERBATIM … per the pre-registration's explicit reuse instruction. Not
reimplemented; do not fork the math independently." The instruction was honoured for the *formula*
and violated for the *budget* — the file declares `N_TRIALS=3` at :83 while the equity lane it
belongs to was at 22 on that date. The reuse-by-copy convention is precisely what let the constant
drift.

### 2.4 CORRECTNESS LANDMINE — drawdown sign convention

Two conventions ship simultaneously:

- **Negative:** `src/research/gated_harness/backtest.py:26` → `(equity/equity.cummax() − 1).min()`.
  Its gate at `:52` tests `metrics["max_drawdown"] >= -abs(maximum_drawdown)`.
- **Positive:** `src/factor/backtest.py:103-106` (`-dd.min()`), `src/equity/backtest.py:72`
  (`((peak-eq)/peak).max()`), `src/crypto/momentum_scorecard.py:117` (docstring: ">= 0.0").
  Their gates test the opposite direction — `src/factor/ship_gate.py:59` `max_dd <= MAX_DRAWDOWN`
  (0.25 @ `:21`), `src/crypto/momentum_scorecard.py:411` `pooled_oos_drawdown <= max_drawdown_limit`.

All of them emit the key **`max_drawdown`**. Executed this session on an 86.3%-drawdown series:

- `gated_harness.max_drawdown` = **−0.8633**; `hard_gate(maximum_drawdown=0.25)` → **False** (correct).
- factor/equity/crypto convention = **+0.8633**; same `hard_gate` call → **True**.

**An 86.3% drawdown passes a 25% drawdown gate.** No cross-feed exists *today* (each gate is fed by
its own producer), so this is latent, not active. But Phase E's exit gate is to route every lane's
metrics through `hard_gate` — **executing the roadmap as written is the trigger**. This is the same
class as the documented "$3,527 dead-write" and "No-Mock catastrophe" incidents: a silently
satisfied guard. Severity **High**. Confidence **HIGH** (executed).

### 2.5 Semantic divergence — effective N

`src/equity/research/harness.py:510-526` defines effective N as the **average count of held names on
non-empty bars** (breadth). `scripts/experiment_edge_round3_leadB.py:71-81` defines it as the
**correlation-eigenvalue participation ratio (Σλ)²/Σλ²** (independent return streams). These answer
different questions and are not comparable. `gated_harness/preregistration.py:33` accepts a
`minimum_effective_n` threshold with **no binding definition** — `report.py:98` compares whatever
float the caller passes. A shared threshold over an unfixed definition is not a gate. Severity
**Medium** (correctness-adjacent: the number is meaningless without its definition).

### 2.6 Divergence — purged CV embargo semantics

`gated_harness/purging.py:26-29` implements embargo correctly: for each contiguous test block it
blocks **training** rows in `[lo − purge, hi + purge + embargo]`, i.e. the embargo extends the
excluded *train* region after the test block.

`src/training/walkforward_validation.py:399` implements the opposite: `test_start_embargoed =
test_start + embargo_gap` — it **shrinks the test set**, and its docstring at `:383` states
"Removes samples from test that are too close to training history". That inverts López de Prado's
embargo, which exists to drop *train* labels overlapping forward into the test window. It also
means `purged_kfold_split` applies **no embargo at all** on the train side (`:400`
`train_after_test = arange(test_end + purge_gap, n)` — purge only). `CombinatorialPurgedCV` at
`:409` has **no embargo parameter whatsoever**.

Both are dead code (zero external callers), so severity today is **Low**, but as a named,
discoverable "purged CV" API it is a correctness trap. Severity **Medium** if adopted.

---

## 3. The 15 mandatory capabilities — implementation / sharing / call status

Classification key: **ABSENT** (no implementation) · **ISOLATED** (exists, only tests call it) ·
**PARTIAL** (a live analog exists but incomplete or forked) · **LIVE** (called by a current
research workload).

| # | Capability | Implementation(s) | Shared or copied? | Called by a workload? | Class |
|---|---|---|---|---|---|
| 1 | Frozen hypothesis hash | `gated_harness/preregistration.py:47-50` | shared lib | No — sole ref `tests/test_gated_harness.py:57`. **Zero prereg doc carries a hash** (regex for 40–64 hex over `docs/prereg-*.md docs/experiment-*.md` → 0 hits) | **ISOLATED** |
| 2 | Untouched OOS | `gated_harness/temporal_splits.py:30` (+`split_digest`) · live analog: four-arm pre/post-cutoff, `harness.py:827+`, `ARM_POST_CUTOFF` @ `contracts.py:57` | forked concept | Analog LIVE; harness version no | **PARTIAL** |
| 3 | Trial-count accounting | `contracts.py:52` `N_TRIALS=24` (hand-maintained constant) · scripts `N_TRIALS=3` · `gated_harness` `trial_budget` field | **two divergent budgets** | Yes, both — inconsistently. No ledger file; `docs/experiments-log.md` is a single HistGB runbook, not a trial register | **PARTIAL — divergent** |
| 4 | Deflated Sharpe | 4 distinct impls / 6 sites (§2.1) | **copy-pasted** | Yes (scripts + equity harness) | **LIVE — duplicated** |
| 5 | Block bootstrap | 3 distinct impls / 5 sites | **copy-pasted, divergent params** | Yes | **LIVE — duplicated** |
| 6 | Bonferroni / declared correction | `contracts.py:53` · scripts `0.05/3` · `significance.py:102` | copied | Yes | **LIVE — divergent** |
| 7 | CPCV | `gated_harness/cpcv.py:13` · `walkforward_validation.py:409` | 2 impls | **No research workload uses CPCV at all** | **ABSENT in practice** |
| 8 | Regime-stratified results | `gated_harness/regimes.py:10` only | shared lib | No — `stratify_returns` has zero external callers; grep for `regime` in `harness.py` and the crypto/EDGAR scripts → 0 hits | **ISOLATED** |
| 9 | Minimum effective sample size | `harness.py:510` · `experiment_edge_round3_leadB.py:71` · threshold-only field in `preregistration.py:33` | **2 incompatible definitions** | Yes (both) | **LIVE — divergent** |
| 10 | Minimum history | `gated_harness/backtest.py:51` · `ship_gate.py:19-20` (`MIN_POSITIVE_YEARS=6`, `MIN_TOTAL_YEARS=10`) | separate, different units (obs vs years) | Analog LIVE | **PARTIAL** |
| 11 | Cost scenarios | `gated_harness/costs.py:18,38` | shared lib, **isolated** | No — per-script literals instead (10.0 / 5.0 / 2.0 bps + `STRESS_MULT`) | **ISOLATED** (live path is hardcoded) |
| 12 | Drawdown gate | ≥6 impls, **2 sign conventions** (§2.4) | copy-pasted | Yes | **LIVE — CORRECTNESS RISK** |
| 13 | Placebo tests | `gated_harness/stress.py:14` (isolated) · live: `harness.py:295` `_placebo_permute`, `ARM_PLACEBO`, `tests/test_research_harness.py:158` | forked | Analog LIVE | **PARTIAL** |
| 14 | Causality tests | `gated_harness/stress.py:29` `assert_prefix_causality` (tests only) · live analogs in tests: `test_equity_backtest.py:229`, `test_factor_backtest.py:26`, `test_edgar_value_accruals_2026_07_02.py:211`, `test_equity_quality_data_2026_06_25.py:76` | forked | Only from tests — **never a report gate** | **PARTIAL** |
| 15 | Survivorship / PIT declaration | `preregistration.py:35-36` + `report.py:44-45` (string fields) | shared lib | No. Sole assertion `tests/test_gated_harness.py:167` checks the string round-trips. Live substitute: prose in prereg docs + `research_entity_blinder` | **ISOLATED** |
| 16 | Independent replay | `gated_harness/verifier.py:20` (isolated) · **genuinely live**: `src/evidence/*/local_import.py` + `evaluation_report_digests` (`contracts/models.py:267`), exercised at `tests/test_hedge_evidence_slice.py:263`, `tests/test_risk_target_evidence_slice.py:270,380` | two parallel systems | Evidence-lane replay LIVE; harness verifier no | **PARTIAL** |

**Score: 0 of 16 capabilities are LIVE *through the shared harness*.** 4 are ISOLATED, 5 PARTIAL
(live via a forked analog), 1 ABSENT in practice, 6 live-but-duplicated across 2–6 copies.

---

## 4. Pre-registration and trial tracking

Files found: `docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md`,
`docs/prereg-finra-short-volume-2026-07-04.md`, `docs/prereg-risk-target-vol-drawdown-2026-07-08.md`,
plus 10 `docs/experiment-*.md` runbooks.

- **Frozen hypothesis hashes: NONE.** A regex for any 40–64-char hex string across all
  `docs/prereg-*.md` and `docs/experiment-*.md` returns **zero matches**. Hypothesis identity is
  asserted in prose ("FROZEN", "Do NOT retune after seeing results" —
  `src/equity/research/contracts.py:38-42`), not content-addressed. The machinery to fix this exists
  and is unused (`preregistration.py:47-50`). Confidence **HIGH**.
- **Trial count: tracked, but in prose + a hand-edited constant, and forked.**
  `contracts.py:48-53` is the closest thing to a register: a comment chain
  ("edge-round-4 = 21; Track B = 22 … 2026-07-04: bumped 22 → 24") plus `N_TRIALS: int = 24`.
  `docs/prereg-finra-short-volume-2026-07-04.md:112-115` narrates the same increment. There is **no
  machine-readable trial ledger**, no test asserting monotonicity, and no mechanism preventing a new
  script from declaring its own budget — which is exactly what the crypto/EDGAR scripts did (`=3`).
- `docs/experiments-log.md` (38 lines) is **not** an experiment register — it is a single pending
  HistGB capacity-shrink runbook relocated from the rules file. There is no per-experiment log
  binding hypothesis → trial index → result.

---

## 5. Phase F — bottleneck claims, confirmed or refuted

### 5.1 Legacy RL data generation — roadmap premise is STALE; **REFUTED**

The roadmap asks to eliminate growing-prefix DataFrame copies and per-row inference. Reading the
actual code: **this work is already done and live.**

- `scripts/train_rl_suite.py:201-236` `prepare_rl_training_data` computes a `source_identity`,
  reuses an existing immutable artifact via `find_rl_matrices`, otherwise builds once and calls
  `materialize_rl_matrices`, then reloads memory-mapped via `load_rl_matrices`. Matches "materialize
  training matrices once" + "memory-mapped files" + "separate sample creation from RL fitting".
- `scripts/train_rl_suite.py:60-113` `load_ensemble_data_for_rl` reads each pair's CSV **once**,
  builds features once, `del df, df_feat; gc.collect()` per pair, concatenates at the end. No
  growing prefix.
- `scripts/train_rl_suite.py:140-148` calls
  `predict_fixed_windows(model, X, sequence_length=…, batch_size=256)` — **batched** inference.
- `src/training/performance/rl_dataset.py:46-61` `fixed_window_view` uses
  `np.lib.stride_tricks.sliding_window_view` and sets `writeable=False` — a **non-copying** view,
  explicitly documented as replacing "the legacy `[i-seq:i]` windows".
  `predict_fixed_windows` @ `:64-88` iterates in `batch_size` chunks.

Confidence **HIGH**. Recommend the roadmap text be corrected; re-litigating this is wasted effort.

### 5.2 Surviving growing-prefix + per-row inference — **CONFIRMED, unrepaired**

`src/training/backtest_harness.py:128`, inside the `while i < n - 2` loop opened at `:127`:

> `direction = signal_fn(df.iloc[: i + 1])`

This is the canonical anti-pattern the roadmap describes — a **growing-prefix DataFrame copy on
every bar** (O(n²) allocation) with **one model inference per row**, un-batched. It is not in the RL
family the roadmap names, so it was missed by the Phase F remediation; but it is the cost-aware
backtest that `scripts/backtest_harness.py` and the promotion gate depend on. Severity **Medium**
(performance, not correctness). Confidence **HIGH**.

Lower-priority prefix scans also present: `src/scanner/agents/synthetic_behaviors.py:301`
(`volatility.iloc[:i].rank(pct=True)` inside a per-bar loop), `src/training/feature_analysis.py:121`.

### 5.3 Equity / Track B full-corpus loads — **CONFIRMED**

`src/equity/track_b_shadow.py:234-246` `_load_cached_price_panel`: `pd.read_parquet(
CACHED_PRICE_PANEL_PATH)` with **no `columns=` and no `filters=`**, then selects the wanted tickers
*in pandas after the full panel is materialised* (`:242 cols = [t for t in tickers if t in
df.columns]`, `:246 return df[cols]`). This is exactly "load the full corpus into one worker",
directly contradicting Phase F's "predicate-pushdown date and ticker selection".

The primitive to fix it **exists and is unused for this path**: `src/equity/research/partitioned_store.py`
`EquityResearchStore` (`:20`) offers `partition_prices`, `read_prices` (`:34`), `write_pit_universe`
(`:52`), `read_rebalance_partition` (`:136`), built on
`src/training/performance/partition_store.py` (docstring `:1`: "Immutable Parquet partitions with
entity/date predicate pushdown"; real `dataset.to_table(filter=predicate, columns=selected)` at
`:176`). Adoption grep: **only `iter_filing_worker_batches` is used in production**
(`src/equity/research/pit_text_loader.py:433` — this satisfies "batch SEC parsing"). The
`EquityResearchStore` class itself is referenced only by `scripts/profile_phase_f_training.py:25`
and `tests/test_phase_f_performance.py:20`. Classification: primitive **ISOLATED**, price path
**unmigrated**. Confidence **HIGH**.

### 5.4 Crypto cross-section construction — **PARTIAL**

- **LIVE:** `src/crypto/research_store.py` `partition_history` / `load_history` are called from
  `src/crypto/data_layer.py:197, 247, 262, 309`. "Partition klines and funding by symbol and month"
  is done.
- **ISOLATED:** `materialize_eligibility` (`research_store.py:50`) and
  `materialize_strategy_dataset` (`:91`) — the two functions implementing "materialize eligibility
  separately", "construct only required cross-sections" and "separate momentum and carry datasets"
  — have callers **only in `tests/test_phase_f_performance.py:260,263,270,276,283`**. No crypto
  workload calls them. Confidence **HIGH**.

### 5.5 Performance baseline report — **EXISTS but does not measure the workload**

Six baselines on disk, one per family, schema v1.0.0:
`trained_data/performance_baselines/phase_f/{legacy_rl,equity_research,track_b,crypto_momentum,crypto_carry,risk_target}.json`,
all stamped `measured_at_utc` 2026-07-12T23:44:3x. **Every one of the 10 Phase-F exit-gate fields is
present as a key** (wall time, peak RSS, CPU util, GPU util, I/O volume, dataset size,
feature-computation / fit / evaluation phase times, artifact size). So the *structure* of the exit
gate is satisfied. The *content* is not:

1. **The measured workload is the replacement, at toy scale, with the expensive parts removed.**
   `scripts/profile_phase_f_training.py:63-101` `profile_legacy_rl` reads `.tail(max_rows)` (512
   rows), then at `:80` sets `predictions = np.full((len(features), 2), 0.5, dtype=np.float32)` — a
   **hardcoded constant instead of any model inference** — and at `:91-92` the `fit` phase body is
   literally `pass  # preparation baseline: fitting is intentionally separated`. Recorded results:
   `legacy_rl` wall time **0.0427 s**, `fit` **7.08e-07 s**, 512 rows. `track_b` wall **0.0047 s**,
   420 rows. These cannot support a before/after claim about a bottleneck, and they measure the
   *new* path, so there is no "before" at all.
2. **I/O volume is structurally zero.** `src/training/performance/metrics.py:63-66` prefers psutil
   process I/O counters and falls back to `resource.getrusage`; on the recording host (macOS, per
   the embedded `environment.platform`) the fallback yields `read_bytes: 0, write_bytes: 0` in all
   six files.
3. **GPU utilization is null in all six** — `"average_percent": null,
   "measurement": "backend exposes no process utilization counter"`.
4. **Peak RSS is not attributable per family.** `metrics.py:146-151` samples **process-wide** RSS;
   the six families were profiled in one process, and `track_b`, `crypto_carry` and
   `crypto_momentum` all report the identical `peak_rss_bytes: 320782336`.

So 3 of 10 exit-gate fields are non-measurements and a 4th is contaminated. **Answer to "is there
ANY performance baseline report on disk": yes, six — but none of them is a valid baseline for the
family it names.** Confidence **HIGH**.

---

## 6. §18 testing standard — coverage for research code

**Real collection counts** (`python -m pytest tests/ -q --collect-only`), run twice this session:
**9266 collected / 72 collection errors**, and **9055 collected / 91 errors** on a prior run —
collection is **nondeterministic**, itself a finding. Errors are optional-dependency import
failures, not logic failures: `textual` 41, `structlog` 14, `tensorflow` 7, `pyarrow` 3, `mlflow` 2,
`defusedxml` 2, `torch` 1, `fastapi` 1, `exchange_calendars` 1, plus `pyo3_runtime.PanicException`
on ~14 `src/evidence/` slice tests. `tests/test_gated_harness.py` collects cleanly (16 tests).

Research-relevant suites: `test_gated_harness.py` 16 · `test_research_harness.py` 22 ·
`test_phase_f_performance.py` 12 · `test_edgar_value_accruals_2026_07_02.py` 15.

| §18 category | Status | Evidence |
|---|---|---|
| **Unit** (features, targets, metrics, serialization, gates, transitions) | **EXISTS** | 65 tests across the 4 suites above; gates covered at `test_gated_harness.py:118-200` |
| **Property** (no future data affects past features; hashes change with content) | **PARTIAL** | No property-based framework at all — `hypothesis` imported in **0** test files. Invariants are hand-written examples: `assert_prefix_causality` (`test_gated_harness.py:157`), `test_equity_backtest.py:229`. Hash-changes-with-content covered in the evidence lane |
| **Regression** (frozen experiment reproduces; local vs worker agree) | **PARTIAL** | Replay-reproduces tests exist only in the **evidence** lane: `test_hedge_evidence_slice.py:263`, `test_risk_target_evidence_slice.py:270,380`; byte-identity at `test_research_entity_blinder.py:211`. **No frozen-experiment reproduction test for any gated_harness module or experiment script** — Phase E's byte-identical migration gate has zero test |
| **Integration** (snapshot→package→importer→quarantine→gate) | **EXISTS for evidence lane only** | `src/evidence/*/local_import.py` + the `*_evidence_slice` / `*_worker_no_authority` suites. **Zero integration tests for `gated_harness`** — it has no pipeline to integrate into |
| **Adversarial** | **PARTIAL — two named categories at zero** | see below |

Adversarial test-name grep (`def test_*<concept>` across `tests/`):
leakage **15**, lookahead **14**, tamper **9**, purge **9**, placebo **6**, embargo **4**,
forged **3**, contamination **2**, degenerate **2**, **trial inflation 0**, **cost
underestimation 0**, **false liveness 0**.

Two of these deserve emphasis:

- **Trial inflation: 0 tests.** The two `*trial*` hits (`test_gated_harness.py:64,134`) test the
  isolated library's own arithmetic. **Nothing asserts that the campaign trial counter is consistent
  across lanes** — which is precisely why `N_TRIALS=3` and `N_TRIALS=24` coexist undetected (§2.3).
- **Survivorship: 0 genuine tests.** All 28 `*surviv*` hits are durability (`survives restart`,
  `survives corrupt JSON`) except `test_gated_harness.py:167`, which only asserts a policy **string**
  round-trips into the report. There is no test that plants a survivorship-biased universe and
  asserts detection.

---

## 7. Consolidated classification

| Deliverable | Class |
|---|---|
| `gated_harness/` — all 11 modules, implemented | **ISOLATED** |
| Phase E exit gate (one shared library) | **ABSENT** |
| Phase E migration gate (byte-identical port) | **ABSENT** |
| Frozen hypothesis hash (machinery / practice) | **ISOLATED / ABSENT** |
| Untouched OOS | **PARTIAL** (four-arm analog LIVE) |
| Trial-count accounting | **PARTIAL — divergent** (two budgets) |
| Deflated Sharpe / block bootstrap / Bonferroni | **LIVE — duplicated 3–6×** |
| CPCV | **ABSENT in practice** (2 impls, 0 research callers) |
| Regime-stratified reporting | **ISOLATED** |
| Minimum effective N | **LIVE — 2 incompatible definitions** |
| Minimum history | **PARTIAL** |
| Cost scenarios | **ISOLATED** (live path hardcodes bps) |
| Drawdown gate | **LIVE — CORRECTNESS RISK (sign)** |
| Placebo / causality | **PARTIAL** (live in equity lane / tests only) |
| Survivorship + PIT declaration | **ISOLATED** (string field, no test) |
| Independent replay | **PARTIAL** (evidence lane LIVE; harness verifier ISOLATED) |
| Phase F — legacy RL repair | **LIVE** |
| Phase F — `backtest_harness.py:128` prefix loop | **ABSENT (unrepaired)** |
| Phase F — equity/Track B pushdown | **PARTIAL** (`iter_filing_worker_batches` LIVE; `EquityResearchStore` ISOLATED) |
| Phase F — crypto partitioning | **PARTIAL** (history LIVE; eligibility/cross-section ISOLATED) |
| Phase F — baseline reports | **PARTIAL / INVALID** |

---

## 8. GAP REGISTER

Effort: **S** ≤ 1 day · **M** ≤ 1 week · **L** > 1 week.
Type: **CORRECTNESS** (can produce a wrong verdict / silently disable a gate) vs **STRUCTURAL**
(duplication, missing coverage, performance).

### Correctness gaps — do these first

| ID | Gap | Type | Effort | Depends on | Concrete work |
|---|---|---|---|---|---|
| **G1** | Drawdown sign convention split across ≥6 impls; `hard_gate` passes an 86.3% DD as a 25% gate (§2.4) | **CORRECTNESS — High** | **S** | — | Pick one convention (recommend **positive fraction**, the majority: `factor/backtest.py:103`, `equity/backtest.py:72`, `crypto/momentum_scorecard.py:117`). Change `gated_harness/backtest.py:26` to return positive and `:52` to `<= abs(maximum_drawdown)`. Add a regression test asserting the gate **rejects** a positive 0.86 DD. Add a shared `max_drawdown` in the canonical lib and mark the 5 others for retirement under G4 |
| **G2** | Two family-wise error budgets (`N_TRIALS=3` vs `24`); flips DSR verdicts in the Sharpe 1.7–2.1 band; α differs 8× (§2.3) | **CORRECTNESS — High** | **M** | — | Create a single machine-readable trial ledger (JSONL, atomic writes per repo convention) keyed hypothesis-digest → trial index. Make `N_TRIALS` **derived** from `len(ledger)`, not hand-edited. Delete the `N_TRIALS=3` literals in `experiment_crypto_funding_carry.py:48`, `experiment_crypto_xs_signals.py:44`, `experiment_edgar_value_accruals_2026_07_02.py:83`. **Re-state, do not silently re-run**, which historical verdicts were issued at N=3 and what they become at the true budget |
| **G3** | No test would ever catch G2 — trial-inflation adversarial coverage is zero (§6) | **CORRECTNESS — High** | **S** | G2 | Add adversarial tests: (a) two lanes declaring different budgets → fail; (b) ledger non-monotonic → fail; (c) a result that passes at N=3 and fails at N=ledger → assert the ledger value binds |
| **G4** | 4 DSR / 3 bootstrap / ≥6 drawdown / 2 effective-N implementations; block=5 vs 21 shifts p by up to 6.5pp (§2.1–2.2, §2.5) | **CORRECTNESS — Medium** | **M** | G1, G2 | Make `gated_harness/significance.py` the single source; convert `src/equity/research/harness.py:550,603,645` into thin re-exports (it is already byte-equivalent — safe first move). Replace the 3 script copies with imports. Freeze **one** block size with a written justification. Freeze **one** effective-N definition and bind `minimum_effective_n` to it in `preregistration.py` |
| **G5** | `walkforward_validation.py:365,409` inverts embargo semantics and `CombinatorialPurgedCV` has none (§2.6) | **CORRECTNESS — Medium** (Low today: dead code) | **S** | — | Either delete both (zero callers today) or fix `:399` to extend the **train**-side exclusion and add `embargo_gap` to `CombinatorialPurgedCV`. Deleting is cheaper and removes the trap |
| **G6** | Survivorship is a declared string with no test; PIT likewise (§3 #15, §6) | **CORRECTNESS — Medium** | **M** | G7 | Add an adversarial test that plants a survivorship-biased universe (delisted names dropped) and asserts the gate fails. Make `survivorship_policy` a `Literal` over accepted policies, not free text |

### Structural gaps

| ID | Gap | Type | Effort | Depends on | Concrete work |
|---|---|---|---|---|---|
| **G7** | Phase E migration never started — 0 workloads call the harness; no byte-identical port (§1.1) | STRUCTURAL | **L** | G1, G2, G4 | Port **one** experiment (recommend `experiment_edgar_value_accruals_2026_07_02.py` — it already declares reuse intent at `:95-98` and is single-lane). Add a regression test asserting the ported run reproduces the archived result **byte-identically**, which is Phase E's own stated gate. Only then port the next |
| **G8** | No frozen hypothesis hashes in any prereg doc (§4) | STRUCTURAL | **S** | G7 | Emit `hypothesis_digest` at prereg time; paste it into the doc header; add a test that a doc's declared digest matches the `ResearchSpecification` in code. This is a ~1-hour change that makes "FROZEN" verifiable rather than asserted |
| **G9** | Regime stratification, cost scenarios, CPCV never used by any workload (§3 #7,#8,#11) | STRUCTURAL | **M** | G7 | Land as part of the G7 port: the ported experiment must emit regime-stratified metrics, run the declared bps ladder through `evaluate_cost_scenarios` instead of a literal, and produce a non-zero `cpcv_path_count` when `use_cpcv` |
| **G10** | Two parallel governance stacks — `gated_harness/verifier.py` vs the live `src/evidence/` replay (§1.2, §3 #16) | STRUCTURAL | **M** | G7 | Decide one. `src/evidence/` is live and tested; the harness verifier is not. Recommend the harness **produce** a report whose digest feeds `evaluation_report_digests` (`contracts/models.py:267`) rather than owning a second replay path |
| **G11** | Phase F baselines measure the replacement at toy scale with `fit` stubbed and inference hardcoded to 0.5 (§5.5) | STRUCTURAL | **M** | — | Profile the **real** workload at real scale; remove the `predictions = np.full(..., 0.5)` stub at `profile_phase_f_training.py:80` and the `pass` at `:91`; profile each family in a **subprocess** so `peak_rss` is attributable; record I/O on a host where the counter works, or mark the field explicitly unmeasured rather than 0 |
| **G12** | `backtest_harness.py:128` growing-prefix + per-row inference (§5.2) | STRUCTURAL (perf) | **M** | — | Hoist feature computation out of the `while` loop; pass precomputed arrays and an index to `signal_fn`; batch inference with the existing `predict_fixed_windows`. The primitive already exists at `rl_dataset.py:64` |
| **G13** | Track B loads the full price panel (§5.3) | STRUCTURAL (perf) | **S** | — | `track_b_shadow.py:239` → route through `EquityResearchStore.read_prices` (`partitioned_store.py:34`), or minimally pass `columns=cols` + `filters=` to `read_parquet`. Primitive already built and tested |
| **G14** | Crypto eligibility / cross-section materialization isolated (§5.4) | STRUCTURAL (perf) | **S** | — | Wire `materialize_eligibility` and `materialize_strategy_dataset` into the crypto workloads; they are already covered by `test_phase_f_performance.py:260-283` |
| **G15** | Roadmap §9 asserts an RL bottleneck that no longer exists (§5.1) | STRUCTURAL (doc) | **S** | — | Amend §9 to record the RL path as complete and re-point it at `backtest_harness.py:128`, the real surviving instance |
| **G16** | 72–91 nondeterministic collection errors; no `hypothesis`; no frozen-experiment regression test (§6) | STRUCTURAL | **M** | G7 | Pin optional deps or mark the suites `@pytest.mark.integration` so collection is deterministic; add `hypothesis` for the §18 property category; add the frozen-report regression test under G7 |

### Dependency order

```
G1 ─┐
G2 ─┼─> G3 ─> G4 ─> G7 ─> {G8, G9, G10, G16}
G5 ─┘                G6 ──┘
G11, G12, G13, G14, G15  (independent, parallelizable)
```

G1 and G5 are same-day, no-dependency fixes. G1 must land **before** any Phase E migration — the
migration is what activates the drawdown-gate bypass.

---

## 9. Method notes and confidence

- Every file citation was opened in this session. Symbol adoption was established by grep over
  `src/ scripts/ tests/`, excluding the defining file, per the repo's integration-grep rule.
- Numerical claims in §2.2, §2.3 and §2.4 were **executed** (numpy 2.4.6, pandas 3.0.5) against the
  real modules — confidence **HIGH**. Comparison script lives in the session scratchpad, not in the
  repo.
- Test counts come from two real `--collect-only` runs; both are recorded because they disagree.
- Confidence **MEDIUM** on one point: whether the crypto/EDGAR experiment scripts are still
  *current* workloads. They are on disk and referenced by live shadow lanes
  (`src/crypto/momentum_shadow.py:10`, `src/crypto/momentum_scorecard.py:10` cite their DSR
  verdicts), so their divergent `N_TRIALS=3` verdicts are load-bearing for decisions already made.
  If they are considered retired, G2's remediation shrinks to re-labelling historical results.
- No source file was modified. The only write is this document.
