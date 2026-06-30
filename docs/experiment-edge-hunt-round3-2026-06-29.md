# Edge Hunt — Round 3 Pre-Registration (Lead A: higher-frequency · Lead B: breadth-for-significance)

**Date written:** 2026-06-29 (BEFORE any Round-3 result was computed)
**Branch:** ralph/equity-harvester-bot
**Status:** PRE-REGISTERED. Hypotheses, universes, params, costs, OOS splits, and gates are
FROZEN below and committed BEFORE any result (separate commit from results). Research /
backtest only — **no live execution, no real money, no order path, no spend (free data only).**

Operator mandate (2026-06-29): pursue the two remaining no-spend, research-endorsed untested
leads toward an actual edge. Discipline (L-018): pre-register before results, untouched OOS,
survivorship-aware, realistic + stressed costs, multiple-testing correction across ALL tests,
separate-verifier re-derivation + leakage audit on anything that clears, return-vs-risk
decomposition. A dredged positive is the worst outcome — refused.

---

## 0. Why Round 3, and the two leads

Round 2 established: crypto TS-trend (H5) is the best-shaped result but fails the gate ONLY on
statistical significance, because crypto's cross-section is ~one factor (effective-N ≈ 2.82) and
history is ~6.5y. The infra-stress proved the significance wall is NOT an infrastructure artifact.
Two no-spend levers remain, each attacking a different wall:

- **LEAD A — higher frequency.** We only ever tested DAILY bars; the infra-stress showed the
  signal is frequency-sensitive. Free Binance 1h klines let us test whether real, cost-surviving,
  OOS-confirmed edge appears at a frequency we never sampled. (Literature, sweep B: intraday edge
  may exist but the strong forms are latency-gated; test the retail-shaped slice honestly.)
- **LEAD B — breadth for significance.** H5 failed ONLY on significance because effective-N ≈ 3.
  The fix is not a different cost model — it is more INDEPENDENT bets. A broad cross-asset
  TS-trend book (equities / bonds / credit / commodities / metals / FX / crypto) has genuinely
  different return drivers; does the combined breadth raise effective-N + the (much longer)
  track record enough to CLEAR the DSR/Bonferroni significance gate while keeping drawdown
  control? This is the single most plausible no-spend path to a gate-clearing result.

---

## 1. Anti-p-hacking accounting (binding)

Cumulative campaign search budget is carried forward. Round-1 crypto (3) + Round-2 infra-stress
grid (12) + Round-2 H4/H5 (2) = 17 crypto configs already examined; plus the multi-asset trend +
sleeve combos. **Frozen multiple-testing count for Round 3: `N_TRIALS = 20`** (conservative
cumulative count; harsher than Round-2's 15). DSR + Bonferroni use N=20: Bonferroni α = 0.05/20 =
**0.0025**. Round-3 adds exactly **3 new pre-registered tests** (Lead A: 2 signals; Lead B: 1
book); cost-stress re-runs are robustness, not new tests. ONE signal + ONE param set each, frozen
below — no sweep, no post-hoc rescue. Any further idea is a new pre-registration.

---

## 2. LEAD B — broad cross-asset TS-trend book (PRIMARY; highest prior)

- **Universe (FROZEN, free / yfinance daily, auto-adjusted) — broadest no-spend cross-asset set,
  chosen for GENUINE driver diversity (not redundant sectors):**
  - US equity: SPY, QQQ, IWM · Intl/EM equity: EFA, EEM, VGK, EWJ
  - Rates: TLT, IEF, SHY · Credit: LQD, HYG, EMB · Inflation: TIP
  - Energy: USO, UNG, DBC · Ags: DBA, CORN, WEAT · Metals: GLD, SLV, CPER
  - Real estate: VNQ · FX: UUP, FXE, FXY, FXB, FXA
  - Crypto: BTC-USD, ETH-USD, SOL-USD, XRP-USD, LTC-USD, BCH-USD, DOGE-USD, ADA-USD
  (~37 names; final = those with ≥250 daily bars in the cache.)
- **Construction (FROZEN — the verifier-checked canonical rule from `src/equity/multi_asset_trend.py`,
  UNCHANGED, no per-asset tuning):** `single_asset_trend_returns` = price > 200d SMA, long-or-flat,
  shift(1) causal, monthly (step=21), 2 bps/side cost → per-asset net streams → `combine_sleeves`
  (HRP) → 10% vol-target overlay (`overlay`, causal, DD circuit-breaker, max_lev 3×). ONE rule.
- **Cost:** 2 bps/side (liquid ETFs; ~1–3 bps real — Frazzini et al. median 6 bps for stocks, ETFs
  tighter). **Stress (robustness, not a new test):** re-run at 5 bps/side.
- **OOS holdout:** last **35%** of the combined track (matches the prior multi-asset test);
  touched once at verdict.
- **Survivorship caveat (honest, pre-stated):** ETF selection is current-survivor at the
  ASSET-CLASS level (standard managed-futures proxy); crypto via yfinance lists current survivors
  (dead alts like LUNA absent) → mild OPTIMISTIC bias on the crypto sleeve. Stated, not hidden.

### Lead-B gate (FROZEN) — the prior multi-asset gate PLUS the significance test it never faced
A book "**clears**" iff ALL of:
1. OOS net Sharpe ≥ 0.40.
2. Max drawdown ≤ 0.25 (full sample).
3. Majority of calendar years positive.
4. OOS-confirmed: OOS Sharpe ≥ 0.40 AND no sign-flip vs in-sample.
5. **SIGNIFICANCE (the new bar): DSR(N=20) ≥ 0.95 AND Bonferroni block-bootstrap p < 0.0025**,
   computed on the portfolio daily net returns. Effective-N of the per-asset trend-stream matrix
   reported (participation ratio of correlation eigenvalues) as the diagnostic for whether breadth
   was sufficient.
6. **Return-vs-risk decomposition (mandatory, L-021):** compute equity-beta (vs SPY) and compare
   to EW buy-hold + 60/40 baselines. Classify honestly: is it (a) ALPHA (market-neutral excess
   return) or (b) a drawdown-controlled RISK-PREMIUM (beats passive on DD, ties/loses on return)?
   A significant, drawdown-controlled trend book is a real RISK-CONTROL result — reported as such,
   NOT as "alpha" (L-021), even if it clears the numeric gate.

**Pre-stated expectation:** breadth + the long ETF track record PLAUSIBLY clears the DSR
significance bar (unlike crypto's 6.5y). Whether it clears on OOS net Sharpe ≥ 0.40 AND keeps
DD ≤ 0.25 is the open question; and it will almost certainly be a RISK-PREMIUM (equity-beta > 0),
not market-neutral alpha. If it clears significance + DD + OOS Sharpe, that is a verified
gate-clearing RISK-CONTROL book — report immediately (after verifier).

---

## 3. LEAD A — higher-frequency (intraday) crypto (2 frozen signals)

- **Data (FROZEN, free Binance 1h static-dump klines via `data_layer.fetch_binance_klines(s,"1h")`):**
  the **top-30 USDT perps by trailing-30d ADV** as of each rebalance (point-in-time, survivorship-
  aware — delisted included while alive). 1h subset (not all 700) to keep the fetch tractable;
  stated. Span 2020→2026-05. OOS = 2024-01-01→present (touched once).
- **Costs:** 5 bps/side (realistic liquid-perp taker+slippage, lit-confirmed). **Stress:** 9 bps/side.
- **Signal A1 — intraday time-series momentum (FROZEN):** per coin, position for hour `t→t+1` =
  sign of trailing **24-hour** return (known at `t−1h`), inverse-vol sized to 10% ann portfolio vol,
  rebalanced hourly... [held between]. Directional TS-trend at 1h. One lookback, no sweep.
- **Signal A2 — intraday momentum "first-vs-last" (FROZEN; the Gao et al. retail-shaped analog):**
  define UTC day; signal = sign of the **first 6h** return of the UTC day; position the **last 6h**
  of the same day in that direction (flat otherwise), cross-sectional dollar-neutral across the
  top-30, 5 bps cost. One definition, no sweep.
- **Lead-A gate (FROZEN):** the crypto gate (§ Round-1/2): OOS net Sharpe ≥ 0.40 · maxDD ≤ 0.25 ·
  OOS-confirmed · **DSR(N=20) ≥ 0.95 & Bonferroni p < 0.0025** · |BTC-β| ≤ 0.15 · history caveat.
  Realistic + stressed cost both reported.
- **Pre-stated expectation (lit sweep B):** intraday crypto momentum is likely FRAGILE OOS and/or
  cost-killed at hourly turnover; the strong intraday edges are latency-gated. Tested honestly.

---

## 4. Separate verifier (L-018) — on anything that clears

If Lead A or Lead B clears its gate, a separate Code-Reviewer/quant subagent independently
(a) re-derives the headline figures from cache + script, (b) leakage-audits (signal lag, PIT
universe, no look-ahead vol-scaling, survivorship, no OOS peek), (c) re-confirms the DSR/Bonferroni
+ effective-N, (d) checks config == this frozen doc. A verifier-caught false claim → L-018
fail-closed reject + quarantine. No gate-clear is reported to the operator as real until verified.

---

## 5. Results (appended after the fact — EMPTY at pre-registration time)

_(filled in a separate commit after the runs + verifier)_
