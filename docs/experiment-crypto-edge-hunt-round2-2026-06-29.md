# Crypto Edge Hunt — Round 2 Pre-Registration (literature-driven + infra-corrected)

**Date written:** 2026-06-29 (BEFORE H4/H5 results were computed)
**Branch:** ralph/equity-harvester-bot
**Status:** PRE-REGISTERED. Hypotheses, params, universe, costs, OOS split, and gate are
FROZEN below and committed BEFORE any result (no pre-reg+result in one commit — the
provenance wrinkle the Round-1 doc flagged is fixed here). Results are appended only in
§7, after the fact, in a SEPARATE commit. Research/backtest only — **no live execution,
no real money, no order path.**

Carries the campaign discipline (L-016/L-017/L-018): pre-registration, untouched OOS
holdout, multiple-testing correction, separate verifier, explicit **return-alpha vs
risk-control** decomposition. Honest negative = success. Dredged positive = refuse.

---

## 0. Why a Round 2, and what changed

Round 1 (`docs/experiment-crypto-edge-hunt-2026-06-29.md`) tested H1 funding carry / H2
XS-14d-momentum / H3 order-flow → all gate-FAIL; H2 was the lead (+0.75 OOS, neutral,
price-driven) but failed on DD −49%, significance (DSR 0.62, p 0.10), and cost-fragility.

Two new inputs justify Round 2:

1. **A literature sweep** (5 independent sourced sub-reports, 2026-06-29) converged on:
   our daily-bar negatives are fully consistent with the published record; the ONE
   surviving liquid small-operator edge is **time-series / trend momentum as a risk
   premium**; and our 10bps/side cost is about-right-to-conservative for liquid crypto
   perps — **the load-bearing problem is TURNOVER, not the per-trade rate**. Han, Kang &
   Ryu (2024) document *exactly our Round-1 result*: crypto **TS-momentum survives
   realistic cost; XS-momentum dies net**. Han et al. (JFQA 2025) document a crypto trend
   factor that "survives transaction costs" at **weekly** rebalance.
   Sources recorded in §8.

2. **An infra-stress decomposition of H2** (`scripts/experiment_crypto_h2_infra_stress.py`,
   `crypto_h2_infra_stress_*.json`, regression-checked to reproduce the published H2 OOS
   Sharpe 0.7497 exactly). Finding: vol-targeting + weekly rebalance bring full-sample DD
   −0.49 → −0.196 (PASS ≤0.25), keep OOS Sharpe +0.85, β −0.03 (neutral) — i.e. **3 of 4
   ex-history gate criteria pass** — BUT the **significance criterion fails for every
   config** (best DSR ≈ 0.68, p ≈ 0.078); effective-N ≈ 3.9 is the wall, not infra.

**The honest split the infra-stress establishes (the answer to "is it our infra?"):**
infra (daily rebalance / no overlay / 2× cost stress) WAS suppressing the *cosmetic* gate
failures (DD, cost-fragility); it was NOT suppressing the *substantive* one (significance
under honest multiple-testing). Round 2 formalizes both halves with frozen, verifier-
checkable tests.

**Binding caveat (unchanged from Round 1, NOT a post-hoc excuse):** `MIN_TOTAL_YEARS = 10`
CANNOT be met by crypto (~6.5y / ~1.5 cycles). The strongest attainable verdict is "clears
ex-history with a permanent insufficient-history caveat" — never an unqualified ship.

---

## 1. Anti-p-hacking accounting (READ FIRST — this is the binding constraint)

This is a literature-driven *wide* search, which is exactly where false positives breed
(L-018). The whole crypto campaign's search budget is accounted here:

- Round-1 primary tests: H1, H2, H3 = **3**.
- Round-1→2 infra-stress grid on H2: 5 cost levels + 5 rebalance freqs + 2 vol-target =
  **12 configs** examined (exploratory; OOS was viewed across the grid).
- Round-2 new tests: H4, H5 = **2**.

**Frozen multiple-testing count for Round 2: `N_TRIALS = 15`** (3 + 12, the configs whose
OOS was examined before freezing H4/H5; H4/H5 themselves are pre-registered so do not need
to be added to their own correction, but using 15 is deliberately conservative). The
Deflated-Sharpe and Bonferroni corrections below use N=15 — **harsher** than Round-1's N=3.
This makes a Round-2 "pass" strictly harder to obtain than Round-1's, by design.

**Pledge:** ONE signal + ONE param set per hypothesis, frozen below. A failing hypothesis
is recorded as a negative — params are NOT tweaked to rescue it. No post-hoc variant is run
under this doc; any further idea is a new pre-registration that increments N again.

---

## 2. Shared harness, universe, costs (IDENTICAL to Round 1 — reused, not rebuilt)

- **Harness:** the verifier-confirmed `backtest()` / `build_panels()` / metric functions
  from `scripts/experiment_crypto_xs_signals.py`. Round-2 adds only a rebalance-frequency +
  vol-target wrapper (`scripts/experiment_crypto_h2_infra_stress.py:backtest_flex`,
  regression-checked == harness at daily/no-overlay) and an H5 TS-trend builder.
- **Universe (point-in-time, survivorship-aware):** Binance USDⓈ-M `*USDT` perps; eligible
  at `t` iff ≥90d history, trailing-30d median quote-vol ≥ $10M ADV, listed at `t`. Delisted
  coins held while alive, closed at last price (understates delisting losses — stated).
- **Costs:** 10 bps per side (taker ~5 + slippage ~5 @ 1% ADV). Stress: 2× (20 bps).
  Literature (sweep C) confirms 10bps is about-right-to-conservative for liquid perps.
- **Execution lag = 1 day** (signal at `t`, held `t→t+1`). Daily price panel.
- **OOS holdout:** IS 2020-01-01→2023-12-31; **OOS 2024-01-01→present, touched once.**

---

## 3. H4 — infra-corrected cross-sectional momentum (CONFIRMATORY)

- **Economic logic / purpose:** formalize the infra-stress finding as a single frozen,
  verifier-checkable config. Does the *infra-corrected* version of the Round-1 H2 lead
  clear the gate once DD and turnover are managed the way the literature prescribes?
- **Signal:** `m_i(t)` = trailing **14-day** close-to-close return (IDENTICAL to H2). One
  value, no sweep.
- **Construction (frozen):** LONG top-quintile / SHORT bottom-quintile by `m`, dollar-
  neutral; **vol-targeted to 10% annualized** portfolio vol using a 30-day trailing realized
  vol estimate, **lagged 1 day** (no look-ahead), leverage capped at 3×; **weekly (7-day)
  rebalance** (book held between rebalances). 10 bps/side. 1-day signal lag.
- **Pre-stated expectation (from the exploratory infra-stress, recorded for honesty):**
  PASS sharpe / DD / neutrality; **FAIL significance** (DSR < 0.95, p > 0.0167) under
  N=15. If it instead PASSES significance under N=15, that is a genuine surprise to be
  scrutinized hardest by the verifier (suspect selection leakage first).

## 4. H5 — crypto time-series / trend momentum (THE NEW LITERATURE-CONVERGENT LEVER)

- **Economic logic:** time-series momentum / trend is the single liquid, small-operator-
  accessible edge the replication literature says survives (Moskowitz-Ooi-Pedersen 2012;
  AQR "Century of Evidence"); Han-Kang-Ryu 2024 + Han et al. JFQA 2025 document it
  surviving costs in crypto specifically where XS-momentum dies. It is a **risk premium**
  (compensation for crash risk), NOT direction alpha — so it is expected to carry market
  beta and to be judged on the risk-control axis as well as the alpha gate.
- **Signal (frozen):** per coin `i`, `sign(trailing 30-day return_i(t))` ∈ {−1, +1}. One
  lookback (30d ≈ the canonical monthly TS-momentum horizon), no sweep.
- **Construction (frozen):** position_i = sign_i × inverse-vol weight (1/σ_i, σ_i = 30-day
  trailing realized vol of coin i, lagged 1 day), normalized so the **portfolio** targets
  10% annualized vol (lagged realized-vol scaling, leverage cap 3×); **weekly (7-day)
  rebalance**; same eligibility/universe/costs/lag. This is directional (long-or-short by
  trend), NOT dollar-neutral by construction.
- **Two-axis judgment (pre-registered):**
  1. **Alpha gate** (the canonical crypto gate, §5) — likely FAILS on `|BTC-β|≤0.15`
     because trend is directional; that is an expected, honest classification, not a
     failure to hide.
  2. **Risk-control decomposition** — report full/OOS Sharpe, maxDD, BTC-β, and compare to
     buy-and-hold BTC and an equal-weight long-only crypto book. The question on THIS axis:
     does crypto trend deliver **drawdown-controlled risk-premium harvesting** (lower maxDD
     than buy-hold at comparable/again Sharpe), the same property multi-asset trend showed —
     or not? A "risk-control win, not alpha" is the expected best case and is reported as
     such (mirrors the multi-asset-trend verdict).

## 5. Gate (crypto variant — reused thresholds, N=15 correction)

A hypothesis "**clears (ex-history)** on the ALPHA axis" iff ALL of:
1. OOS net Sharpe ≥ 0.40 (after 1× costs).
2. Max drawdown ≤ 0.25 (full sample, after costs).
3. Causal/walk-forward == True (lag ≥1, PIT universe, no future data).
4. OOS-confirmed: OOS net Sharpe ≥ 0.40 AND no sign-flip vs IS.
5. Multiple-testing survives: **DSR (N=15) ≥ 0.95 AND Bonferroni bootstrap p < 0.05/15 =
   0.0033.** (Harsher than Round-1's 0.0167.) Effective-N re-checked by the verifier.
6. Market-neutral: |BTC-β| ≤ 0.15 AND return is harvested spread/trend, not disguised beta.
7. History ≥10y: CANNOT be met → permanent caveat.

The RISK-CONTROL axis (H5 only) is judged qualitatively per §4.2 — NOT a ship gate, an
honest classification of whether the finding is the (already-known) trend risk-premium.

---

## 6. Separate verifier (independent, per L-018)

After H4/H5 run, a separate Code-Reviewer/quant subagent independently (a) re-derives the
headline OOS Sharpe / maxDD / β / DSR / p from cached data + script, (b) audits leakage
(lag, PIT membership, no look-ahead vol scaling, no survivorship peek), (c) re-confirms the
N=15 correction and effective-N, (d) checks H4/H5 configs match THIS frozen doc (no post-hoc
tuning). A verifier-caught false claim → L-018 fail-closed reject + quarantine.

---

## 7. Results (appended after the fact — EMPTY at pre-registration time)

_(to be filled by a separate commit after H4/H5 run + verifier)_

---

## 8. Literature sources (the 5-sweep synthesis that motivated Round 2)

- Replication: Harvey-Liu-Zhu RFS 2016 (SSRN 2249314); Hou-Xue-Zhang "Replicating
  Anomalies" (SSRN 2961979); Chen-Zimmermann open-source AP (SSRN 3604626); McLean-Pontiff
  JF 2016 (SSRN 2156623).
- Frequency: Gao-Han-Li-Zhou JFE 2018 (SSRN 2440866); Lou-Polk-Skouras JFE 2019; Moskowitz-
  Ooi-Pedersen JFE 2012 (SSRN 2089463); Hasbrouck-Saar low-latency.
- Costs: Novy-Marx-Velikov (SSRN 2535173); Frazzini-Israel-Moskowitz (SSRN 2294498);
  Gârleanu-Pedersen JF 2013; Bailey-López de Prado DSR (SSRN 2460551); **Han-Kang-Ryu 2024
  crypto TS-vs-XS momentum under realistic cost (SSRN 4675565)**.
- Crypto: Liu-Tsyvinski RFS 2021 (NBER w24877); Liu-Tsyvinski-Wu JF 2022 (NBER w25882);
  **Han et al. "A Trend Factor for the Cross Section of Cryptocurrency Returns" JFQA 2025**;
  Grobys-Shahzad IJFE 2025 (momentum infinite-variance warning); Schmeling-Schrimpf-Todorov
  BIS WP 1087 (crypto carry); Makarov-Schoar JFE 2020 (cross-exchange arb trapped).
- Alt-data/niche: Muravyev-Pearson-Pollet JF 2025 (borrow fee IS the anomaly);
  Da-Engelberg-Gao FEARS; Lucca-Moench pre-FOMC (now dead).
