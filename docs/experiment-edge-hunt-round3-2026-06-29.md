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

## 5. Results (appended after the fact — separate commit from §0-4)

### LEAD B — broad cross-asset trend: NEAR-MISS (does NOT clear; breadth worked, OOS significance narrowly fails)

Run: `scripts/experiment_edge_round3_leadB.py` → `edge_round3_leadB_{2,5}bps_*.json`. Panel: 37/37
assets, 9753 daily bars (1990-2026 where available), OOS = last 35%.

**Effective-N of the trend-stream matrix = 13.07** (vs crypto's 2.82 — breadth raised it ~5×, the
hypothesis worked).

| cost | full Sharpe | full maxDD | full DSR(N20) | full p | OOS Sharpe | OOS maxDD | OOS DSR(N20) | OOS p |
|---|---|---|---|---|---|---|---|---|
| 2 bps | 0.64 | 0.195 | **0.98 ✓** | **0.0 ✓** | 0.764 | 0.187 | **0.819 ✗** | 0.0034 ✗ |
| 5 bps | — | — | — | — | ~0.74 | — | 0.807 ✗ | 0.0038 ✗ |

**Gate:** OOS Sharpe≥0.40 ✓ · maxDD≤0.25 ✓ · majority-years-positive ✓ (23/34) · OOS-confirmed ✓ ·
**FULL-sample significance ✓ (DSR 0.98, p 0.0)** · **OOS significance ✗ (DSR 0.819<0.95; p 0.0034>
0.0025 Bonferroni)** → `clears_gate = FALSE`. A genuine NEAR-MISS: breadth lifted effective-N 5× and
made full-sample significance clear decisively; the OOS-only test narrowly misses, primarily a
holdout-length/power issue (OOS = ~10y vs full ~34y), not a signal-quality collapse.

**Return-vs-risk decomposition (L-021): it is NOT alpha.** OOS Sharpe 0.764 LOSES to EW buy-hold
(1.10) and 60/40 (1.28); β-to-SPY 0.14 (low), maxDD 0.187 (vs EW −0.765). It is a low-beta,
low-drawdown RISK-CONTROL book — drawdown-controlled risk-premium harvesting — that does not beat
passive on return in this OOS. Reported as risk-control, not edge.

### LEAD A — higher-frequency (1h) intraday crypto: DECISIVE NEGATIVE

Run: `scripts/experiment_edge_round3_leadA.py --top 30` → `edge_round3_leadA_*.json`. 1h panel:
56,232 hours × 30 coins (top-30 by ADV, survivorship-aware incl. LUNA), OOS 2024+.

| signal | cost | OOS Sharpe | IS Sharpe | full maxDD | DSR | p | β | clears |
|---|---|---|---|---|---|---|---|---|
| A1 hourly TS-momentum | 5 bps | −1.54 | −1.34 | −0.66 | 0.0 | 0.99 | −0.04 | ✗ |
| A1 hourly TS-momentum | 9 bps | −2.87 | −2.68 | −0.87 | 0.0 | 1.0 | −0.04 | ✗ |
| A2 first-6h→last-6h | 5 bps | −2.76 | −0.34 | −0.77 | 0.0 | 1.0 | +0.01 | ✗ |
| A2 first-6h→last-6h | 9 bps | −4.81 | −1.78 | −0.95 | 0.0 | 1.0 | +0.01 | ✗ |

**Decisive negative on both signals at both cost levels.** Higher frequency surfaces NO retail-
accessible edge in crypto — hourly turnover cost is fatal and the intraday-momentum effect is
absent/reversed. Consistent with the literature: documented intraday edge is latency/colocation-
gated; the retail-shaped slice does not survive. No edge.

### Synthesis
NEITHER lead clears the gate. Lead A = decisive negative (frequency is not the missing lever for a
retail price-taker). Lead B = honest near-miss (breadth IS the right lever — eff-N 2.82→13.07,
full-sample significance clears — but the result is risk-control, not alpha, and the OOS-only
significance narrowly misses on holdout power).

**VERIFIER (§4) — Lead B causality leak-probe (independent recompute): CAUSAL, leakage-free.**
Forward-lag probe (re-derived independently): look-ahead (lag=0) OOS Sharpe +1.106 vs as-run
(lag=1) +1.086 vs extra-lag (lag=2/3) +1.072/+0.916 — **monotonic, gentle degradation; the
look-ahead "cheat" buys only +0.02**, the fingerprint of a real causal signal with NO material
HRP/overlay/SMA look-ahead. (The construction `single_asset_trend_returns`/`overlay` was already
verifier-confirmed causal in the prior multi-asset work; this re-confirms it on the expanded
universe.) All headline numbers reproduced on re-run. Survivorship bias (current-survivor ETFs +
yfinance crypto) is disclosed and biases the trend result mildly OPTIMISTIC — yet it STILL does not
clear, so the negative is conservative. Separate Code-Reviewer corroboration was dispatched in
parallel. **Lead B is an honest, leakage-free NEAR-MISS — does not clear, is risk-control not alpha.**

## 6. FINAL VERDICT — come-back-(b): definitively exhausted, closest-yet, named remaining lever

**No verified gate-clearing return-ALPHA edge exists in any no-spend lever.** Both Round-3 leads are
resolved:
- **Lead A (higher frequency): decisively negative.** Frequency is not the missing dimension for a
  retail price-taker — intraday crypto momentum is absent/reversed and hourly turnover cost is fatal
  (OOS Sharpe −1.5 to −4.8). Matches the literature (documented intraday edge is latency/colocation-
  gated; the retail slice does not survive).
- **Lead B (breadth for significance): the closest the campaign has ever come — a verified near-miss,
  but risk-control, not alpha.** Breadth WORKED on its own terms: effective-N 2.82→13.07, and
  full-sample significance CLEARS decisively (DSR 0.98, p≈0). It misses on exactly ONE frozen
  criterion — **OOS statistical significance: DSR-OOS 0.819 (needs ≥0.95, short by 0.13) and
  bootstrap p-OOS 0.0034 (significant at conventional p<0.01, but fails the Bonferroni-for-20-trials
  bar of 0.0025 by a hair).** That miss is a **statistical-power limit of a ~10-year OOS holdout**,
  not a fake signal (causality verified). AND — decisively — even if it cleared, the return-vs-risk
  decomposition shows it is **NOT alpha**: OOS Sharpe 0.764 loses to EW buy-hold (1.10) and 60/40
  (1.28); it is a low-beta (0.14), low-drawdown (0.19) RISK-CONTROL book (L-021).

**The one missing criterion, plainly:** the broad trend book is a real, causal, statistically-strong-
over-34-years risk premium whose 10-year holdout is *almost* significant after correcting for 20
trials. What would close that gap is NOT a better signal or cost model — it is **more independent
out-of-sample history** and/or **more genuinely-independent return drivers**, both of which are
structurally capped (cross-asset macro has ~7–10 truly independent drivers; OOS calendar length is
fixed). No no-spend lever changes that.

**The named remaining lever (requires a SPEND — operator's decision, NOT taken):**
- For **return-ALPHA** (the original goal): the only literature-endorsed accessible edges need PAID
  inputs — options-implied vol/skew (OptionMetrics/IvyDB, ~institutional), point-in-time
  fundamentals (quality/value done right), or fund-flow / short-interest feeds; OR microstructure /
  cross-exchange-basis infrastructure (multi-venue + latency + ~$20k basis capital). All cost money;
  none is free.
- For **RISK-CONTROL** (no spend): the broad cross-asset trend book is deployable TODAY as a
  drawdown-controlled allocation (β 0.14, maxDD 0.19, 34y causal) — but that is risk-premium
  harvesting, a strategic/hot-path decision, NOT the alpha the hunt sought.

Honest end-state: **accessible return-alpha needs inputs we don't have (a data/infra spend); the
best no-spend result is a verified, causal, drawdown-controlled risk-premium book that narrowly
misses the significance bar and is not alpha anyway.** Consistent with L-020/021/022 and the
published literature. Paper/research only; immutables intact.
