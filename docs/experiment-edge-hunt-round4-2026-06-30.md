# Edge Hunt — Round 4 Pre-Registration (ONE breadth+history expansion to close the binding DSR-OOS miss)

**Date written:** 2026-06-30 (BEFORE any Round-4 result was computed)
**Branch:** ralph/equity-harvester-bot
**Status:** PRE-REGISTERED. The expansion below is FROZEN and committed BEFORE any result
(separate commit). Research/backtest only — **no live execution, no real money, no spend
(free yfinance data only).**

## 0. The single binding miss, and the ONE legitimate no-spend shot

Round-3 Lead B (broad 37-asset cross-asset trend book, `docs/experiment-edge-hunt-round3-
2026-06-29.md`) is verified causal + leakage-free and:
- full-sample significant (DSR-full 0.987, p≈0),
- OOS-significant on the p-value (p-OOS 0.002 < 0.0025 Bonferroni),
- and misses the FULL gate on **exactly one sub-metric: DSR-OOS = 0.843 < 0.95.**

That miss is a statistical-POWER limit of the ~10-year (3,413-bar) holdout, not a weak/decaying
signal (OOS Sharpe 0.788 > full-sample 0.665). DSR-OOS power rises with the holdout length T
(`sr0 = E[maxSharpe]/√(T−1)`, so the haircut shrinks ∝ 1/√(T−1)) and with effective-N (more
genuinely-independent bets → higher achievable Sharpe). The two legitimate, no-spend levers:
1. **More independent history** — extend the free panel back with long-history index/futures
   price proxies (more OOS bars).
2. **More genuinely-independent sleeves** — add free, liquid, weakly-correlated markets (raise
   effective-N beyond 13).

**This is run as ONE pre-registered expansion, ONE time.** It is NOT iterated to pass.

## 1. HARD anti-p-hacking guard (the binding constraint — this is the trap)

The temptation is to add data/sleeves until DSR-OOS crosses 0.95 — that dredges the single gating
metric and is the L-018 lie. Guard:
- The EXACT universe (§2), construction (§3, UNCHANGED from Round 3), OOS rule, gate, and N are
  FROZEN here and committed BEFORE the run.
- **Exactly ONE run** (plus a 5 bps cost-stress robustness re-run, which is not a new test). No
  universe variants, no OOS-fraction tuning, no "try another combination."
- If it clears the FULL gate (incl. **DSR-OOS ≥ 0.95**) AND a separate verifier re-derives it
  leakage-free → real result, report. If it does NOT clear → **definitive close of the no-spend
  frontier.** Either outcome is accepted as final.
- Multiple-testing count incremented: **N_TRIALS = 21** (Round-3's 20 + this one expansion).
  Bonferroni α = 0.05/21 = 0.00238.

## 2. FROZEN universe (free yfinance; the Round-3 37 + 22 long-history/independent additions)

Verified 2026-06-30 that all 22 additions download with sufficient history (probe in commit msg).

- **Round-3 base (37):** SPY, QQQ, IWM, EFA, EEM, VGK, EWJ, TLT, IEF, SHY, LQD, HYG, EMB, TIP,
  USO, UNG, DBC, DBA, CORN, WEAT, GLD, SLV, CPER, VNQ, UUP, FXE, FXY, FXB, FXA, BTC-USD, ETH-USD,
  SOL-USD, XRP-USD, LTC-USD, BCH-USD, DOGE-USD, ADA-USD.
- **Long-history index proxies (extend T, add regional breadth):** ^GSPC (S&P, 1927), ^IXIC
  (Nasdaq, 1971), ^RUT (Russell 2000, 1987), ^N225 (Nikkei, 1965), ^GDAXI (DAX, 1987), ^FTSE
  (FTSE 100, 1984), ^HSI (Hang Seng, 1986), DX-Y.NYB (US Dollar Index, 1971).
- **Long-history commodity futures (extend T):** GC=F (gold), SI=F (silver), CL=F (WTI), HG=F
  (copper), NG=F (natgas), ZC=F (corn), ZW=F (wheat), ZS=F (soybeans) — all 2000+.
- **New independent sleeves (raise effective-N):** PPLT (platinum), PALL (palladium), FXF (Swiss
  franc), FXC (Canadian dollar), EWZ (Brazil), INDA (India).

**Total = 59 tickers.** Assets with <250 bars are dropped (none expected). Redundancy with the base
(e.g. ^GSPC vs SPY, GC=F vs GLD) is intentional and SAFE: it extends history, and the effective-N
(correlation-eigenvalue participation ratio) + HRP combiner both account for correlation — redundant
copies do NOT inflate effective-N. The long indices/futures provide the deep history; the new
metals/FX/EM sleeves provide independent breadth.

**Survivorship caveat (honest, unchanged):** index/ETF/futures selection is current-survivor at the
market level (standard managed-futures-proxy practice). The Round-3 verifier showed this is
CONSERVATIVE for this book (removing the survivorship-biased crypto sleeve IMPROVED the result).

## 3. Construction (FROZEN — IDENTICAL to Round 3, the only change is the universe)

`src/equity/multi_asset_trend.py`, unchanged: `single_asset_trend_returns` (price > 200d SMA,
long-or-flat, shift(1) causal, monthly step=21, 2 bps/side) → per-asset net streams →
`combine_sleeves` (HRP, causal, trailing-window weights) → 10% vol-target overlay
(`overlay`, causal, lagged realized-vol, DD circuit-breaker, max_lev 3×). **Frozen `target_vol=0.10`
passed explicitly** (the Round-3 deviation is not repeated). Cost 2 bps; stress 5 bps.

**OOS holdout:** last **35%** of the combined-portfolio return series (IDENTICAL rule to Round 3;
the OOS_start is mechanically determined by the longer panel — reported, not chosen). Touched once.

## 4. Gate (FROZEN — the full Round-3 gate, N=21)

Clears iff ALL: (1) OOS net Sharpe ≥ 0.40; (2) maxDD ≤ 0.25 (full); (3) majority years positive;
(4) OOS-confirmed (no IS→OOS sign flip); (5) **DSR-OOS(N=21) ≥ 0.95 AND Bonferroni block-bootstrap
p-OOS < 0.00238**; (6) effective-N reported. Plus the return-vs-risk decomposition (L-021): equity-β
+ vs EW buy-hold + 60/40 — classify alpha vs drawdown-controlled risk-premium.

## 5. Honest framing (pre-stated)

**Even if it FULLY clears, this is a RISK-CONTROL book, not return-alpha** (Round-3: OOS Sharpe loses
to buy-hold, β ≈ 0.13). Clearing only determines whether the no-spend cross-asset trend book becomes
a fully-gate-passing, deployable RISK-CONTROL artifact vs. a near-miss. The return-ALPHA door stays
closed without a paid input regardless. Separate verifier (L-018) re-derives + leakage-audits any
clear before it is reported as real.

## 6. Results (appended after the fact — separate commit from §0-5)

Run: `scripts/experiment_edge_round4.py` → `edge_round4_{2,5}bps_*.json`. Panel: 59/59 assets,
**26,507 daily bars, 1928-01-03 → 2026-06-30**; OOS = last 35% → **OOS_start 1995-08-14, 9,278 OOS
bars** (vs Round-3's 3,413 — the history lever delivered ~2.7× more OOS bars). Effective-N **15.79**
(vs Round-3's 13.07 — the breadth lever raised it too).

| cost | full Sharpe | full maxDD | DSR-full | OOS Sharpe | OOS maxDD | **DSR-OOS(N21)** | p-OOS |
|---|---|---|---|---|---|---|---|
| 2 bps | 0.511 | **0.275** | 1.00 | 0.703 | 0.275 | **0.99 ✓** | 0.0002 ✓ |
| 5 bps | 0.503 | 0.278 | — | 0.691 | 0.278 | **0.988 ✓** | 0.0002 ✓ |

**Gate:** OOS Sharpe≥0.40 ✓ · **maxDD≤0.25 ✗ (0.275)** · majority-years-positive ✓ (63/99) ·
OOS-confirmed ✓ · **DSR-OOS(N21)≥0.95 ✓ (0.99) & p-OOS<0.00238 ✓ (0.0002)** → `clears_gate = FALSE`.

**What happened — the lever worked, the gate moved:** the expansion did EXACTLY what it was designed
to. The Round-3 binding miss — **DSR-OOS — is now decisively CLEARED (0.843 → 0.99)**: more
independent history (9,278 vs 3,413 OOS bars) shrank the power haircut, and effective-N rose
13.07 → 15.79. The hypothesis (breadth+history raises OOS-DSR power) is VINDICATED. **But extending
the panel to 1928 surfaced the deeper drawdowns of the 1930s/1970s/2008: full-sample maxDD rose
0.167 → 0.275, now breaching the 0.25 gate.** The book does not clear — it now fails a DIFFERENT
criterion (drawdown), at both cost levels. This is the honest cost of a full-century lookback: more
history buys statistical power AND reveals deeper tails.

**Return-vs-risk (L-021): still NOT alpha.** OOS Sharpe 0.703 loses to 60/40 (1.16) and EW buy-hold
(0.917); β-to-SPY 0.18. A drawdown-controlled risk-premium book — and over the full century its
drawdown control is weaker (0.275) than the gate demands.

**Anti-p-hacking note (binding):** maxDD 0.275 misses the 0.25 cap by 0.025. Lowering the vol-target
(e.g. to 8%) would pull maxDD under 0.25 — but that is dredging the NOW-binding metric (the exact
L-018 trap), so it is NOT done. ONE pre-registered run, accepted. No verifier dispatched: the pre-reg
gates separate-verifier re-derivation on a FULL-gate CLEAR (condition a); this is a non-clear on a
deterministic maxDD, so condition (b) — definitive close — applies. The construction is the
already-twice-verifier-confirmed causal one (Round 3); only the universe changed.

## 7. FINAL VERDICT — definitive close of the no-spend frontier (come-back-(b))

The no-spend frontier is now FINAL. The single most plausible no-spend path to a gate-clear — breadth
+ history expansion of the cross-asset trend book — was pre-registered and run ONCE. It **closed the
significance miss decisively (DSR-OOS 0.843 → 0.99)** but does **not clear the full gate**: the
full-century history reveals maxDD 0.275 > 0.25. The book remains, at every scale tested,
RISK-CONTROL, not return-alpha (loses to passive on return).

**No no-spend lever produces a verified, full-gate-clearing result.** Across the whole campaign —
FX/equity/multi-asset/crypto; daily and intraday; infra-corrected; breadth- and history-expanded —
the conclusion is consistent and now exhaustive: **accessible return-ALPHA requires a paid input
(options-implied / PIT-fundamentals / fund-flows) or infrastructure (cross-exchange / basis /
latency) we do not have.** The best no-spend artifact is a real, causal, century-significant
risk-premium TREND book that misses the full gate by one criterion at a time (significance at
~10y history; drawdown at ~100y history) and is not alpha regardless. Operator's decision among the
spend-required levers. Paper/research only; immutables intact.
