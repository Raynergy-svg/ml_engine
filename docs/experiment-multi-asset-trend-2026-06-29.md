# PRE-REGISTRATION — Diversified multi-asset TREND / managed-futures (2026-06-29)

Written BEFORE fetching data or seeing any result (anti-p-hacking, L-018). Fixes the
universe, the ONE canonical rule, params, portfolio construction, metric, and win-bar
in advance. The hypothesis: trend pays through BREADTH across uncorrelated asset
classes — a diversified multi-asset trend book may have a real positive risk-adjusted
edge that the FX-majors-only test did NOT (FX-only daily factor was gross≈0).

## Universe (FIXED — exact list, free yfinance proxies, no per-asset selection later)
One liquid ETF/proxy per exposure, 7 classes, ~22 assets:
- US/Global equity: SPY, QQQ, IWM, EFA, EEM
- Rates/bonds: TLT, IEF, LQD, HYG
- Commodities: DBC, USO, UNG, DBA
- Metals: GLD, SLV
- Real estate: VNQ
- FX: UUP, FXE, FXY
- Crypto: BTC-USD, ETH-USD
Each asset ENTERS the book when it has ≥ `sma_window` history (staggered inception
handled — no IPO/listing look-ahead). Survivorship: these are persistent index/asset
proxies that all still trade (not single stocks); the set is fixed in advance, NOT
filtered to winners. (Caveat reported: this is current-survivor ETF selection at the
asset-class level — standard managed-futures proxy practice, not cherry-picked markets.)

## The ONE canonical trend rule (uniform across ALL assets — no per-asset tuning)
- Signal: `on[i,t] = price[i,t-1] > SMA(price, 200d)[i, ≤t-1]` — long-or-flat, both legs
  `shift(1)` => strictly causal. (Identical to the already-validated trend_sleeve rule.)
- Rebalance: monthly (step = 21 trading days). Execution lag ≥ 1 bar.

## Portfolio construction (FIXED)
- Per-asset single-asset trend net-return stream (long-or-flat; 2 bps/side cost on turnover).
- Combine the per-asset streams with the **HRP-across-sleeves combiner**
  (`sleeve_combiner.combine_sleeves`, 252d/21d) — diversification-aware, the operator's
  named tool. (HRP on the return matrix; inverse-variance for the degenerate small-n window.)
- **Vol-target overlay** to 10% annualized on the combined stream (causal scalar; reuse
  `backtest.overlay`, which shift(1)s). Net of cost.

## Metric + win-bar (FIXED)
- Ship gate (`src.factor.ship_gate.evaluate_gate`): net Sharpe ≥ 0.40, max DD ≤ 0.25,
  positive years ≥ 6, total years ≥ 10.
- OOS: chronological split, IS = first 65%, OOS = last 35%.
- Baselines: (a) EW buy-and-hold of the 22 assets, (b) 60/40 (SPY/TLT), (c) the FX-only
  trend result (prior negative).
- **WIN** = full-sample clears the gate AND OOS net Sharpe clears the gate (≥0.40) AND
  beats the EW buy-hold baseline by a REAL OOS margin. Judged at the PORTFOLIO level
  (diversified-basket Sharpe), NOT by picking the best individual markets.

## Anti-p-hacking (CRITICAL — wide multi-market search breeds false positives)
- ONE rule, ONE param set (above), applied uniformly. No per-asset tuning, no market
  cherry-picking, no variant sweep. Per-asset Sharpes reported for transparency only —
  the decision is the portfolio.
- Significance corrected for breadth: a portfolio of ~22 trend streams could look good by
  diversification of noise; the OOS held-out window (the search never touches it) is the
  real test. Separate verifier independently re-derives + flags multiple-testing exposure.
- Honest NEGATIVE = success. A dredged positive (tuned/cherry-picked) = the L-018 lie, refuse it.
- If a real edge appears: REPORT it, do NOT auto-promote/trade — new-market execution is a
  separate operator decision. Directional transformer stays closed; practice-only.

---

## RESULT (post-hoc, 2026-06-29) — GATE-CLEARING (the strongest finding of the search), but a TIE vs buy-hold on Sharpe

21/22 assets, 1993-2026 (34y, 9751 days). `trained_data/backtests/multi_asset_trend_result.json`.

| Strategy | Full Sharpe | Full maxDD | OOS Sharpe | OOS maxDD | gate? |
|---|---|---|---|---|---|
| **Diversified multi-asset trend** | **0.744** | **0.18** | **0.829** | 0.165 | **PASS** (full & OOS) |
| EW buy-hold (21 assets) | 0.761 | **0.678** | 0.895 | 0.678 | FAIL (DD) |
| 60/40 (SPY/TLT) | 0.948 | 0.299 | — | — | FAIL (DD) |

Robustness (labelled, not cherry-picking): **no-crypto** (19 assets) full 0.794 / OOS 0.914, gate PASS — crypto isn't driving it. **No vol-target overlay** (raw HRP combine): 0.693, maxDD 0.10, gate PASS — overlay leverage isn't manufacturing Sharpe. Positive in ALL 7 disjoint 5y sub-periods (range 0.21–1.38).

### Adjudication against the pre-registered win-bar — SPLIT
1. **Clears the ship gate (absolute):** YES — Sharpe 0.744 full / 0.829 OOS, maxDD 0.18, 26/34 positive years, OOS-confirmed. **The FIRST and ONLY lever in the entire campaign to clear the gate on its own** (FX-only trend was gross≈0; the raw baskets fail the DD cap at 68%/30%).
2. **Beats EW buy-hold by a REAL OOS Sharpe margin (relative "+1"):** NO — OOS 0.829 vs 0.895 (slightly below; no-crypto 0.914 is marginally above). It TIES buy-hold on Sharpe.

### Plain verdict
The "trend pays through breadth" hypothesis is **confirmed in the sense that matters**: breadth across uncorrelated asset classes turned the FX-only dead-end into a **robust, gate-clearing managed-futures book** (~0.7–0.9 Sharpe, 10–18% maxDD, 34y, OOS-confirmed, all sub-periods positive). **But the advantage over simply buy-and-holding the same basket is DRAWDOWN CONTROL (18% vs 68% — a 4× reduction), NOT excess Sharpe.** Same drawdown-reducer property the FX trend sleeve showed, now strong enough at portfolio scale to clear the gate. This is a genuine, defensible *risk-adjusted* result — the best of the search — but it is NOT alpha over passive holding. Per the strict relative "+1" bar it does not clear; per the absolute ship gate it does.

REPORTED, NOT promoted/traded — new-market (multi-asset ETF) execution is a separate operator decision and would itself be a different account/broker. Directional transformer stays closed; practice-only.
