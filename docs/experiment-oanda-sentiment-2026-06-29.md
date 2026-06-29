# PRE-REGISTRATION — OANDA position-book contrarian sentiment test (2026-06-29)

Written BEFORE fetching data or seeing any result (anti-p-hacking, L-018). This file
fixes the signal, params, universe, metric, and win-bar in advance. One signal, one
param set, no sweep. A winner must clear the bar AND survive an out-of-sample split.
An honest NEGATIVE is the expected base rate and counts as success.

## Hypothesis
Retail FX positioning is a contrarian signal: when the crowd is net-long an instrument,
fade it (expect underperformance), and vice versa. OANDA's position book publishes the
long/short distribution of open positions — a direct, public retail-crowding gauge.

## Data (fixed)
- Source: OANDA **practice** `/instruments/{inst}/positionBook?time=…` (20-min grid;
  each snapshot carries `price` = mid at book time). Practice-pinned; read-only.
- Cadence: ONE snapshot per day at **17:00Z** (aligned), 2024-06-29 → 2026-06-29 (~2y).
- Universe (7 majors, fixed): EUR_USD, USD_JPY, GBP_USD, USD_CHF, AUD_USD, USD_CAD, NZD_USD.
- Price series = each book's own `price` field (signal and return from the SAME snapshot
  grid → no cross-source misalignment).

## Signal (causal, param-free)
- `NL_i,t` = fraction of positions LONG = Σ longCountPercent / (Σ longCountPercent + Σ shortCountPercent), from the book at time t.
- Contrarian signal `S_i,t = -(NL_i,t - 0.5)`  (crowded-long → negative → fade).
- **Strictly causal:** S_i,t uses ONLY the book at t; the realized return is
  `r_i,t = price_i,(t+1) / price_i,t - 1` (book-time t → next book-time t+1). The signal
  is known strictly before the return window. No look-ahead, no future leakage, no
  in-sample normalization that peeks (raw level only — no rolling z that uses the future).

## Metric + win-bar (fixed)
1. PRIMARY — **Information Coefficient**: pooled Pearson corr `IC = corr(S, r)` over all
   (i,t), with a t-stat. WIN requires IC of the hypothesized (negative-fade → positive
   contrarian) sign AND |t| ≥ 3 (raised bar for pooled, cross-sectionally-correlated FX).
2. SECONDARY — **contrarian portfolio net Sharpe** (cost-aware): daily-rebalanced EW of the
   per-instrument contrarian bets `p_i,t = S_i,t` (sign-and-magnitude), gross leverage 1.0,
   cost = **1.5 bps per side** on position change (realistic major spread). Annualized net
   Sharpe vs the ship gate (`MIN_NET_SHARPE=0.40`) and vs a flat baseline (0).
3. **Out-of-sample:** split chronologically — IS = first 65%, OOS = last 35%. A win must
   show the SAME-sign IC in OOS AND OOS net Sharpe beating baseline by a real margin
   (the "+1" bar), not just full-sample. Plus disjoint-year sub-period stability.

## Anti-p-hacking (mandatory)
- ONE signal, ONE param set (above). No variant sweep; if I report a secondary horizon
  (e.g. 5-day) it is labelled exploratory, NOT counted as the pre-registered result.
- Winner confirmed on the held-out OOS window before any claim.
- Separate verifier flags multiple-testing exposure + effective-N (FX majors are
  cross-correlated → pooled N overstates independence; USD pairs especially).

## Known limitation (stated up front, not an excuse)
2 years CANNOT satisfy the ship gate's `MIN_TOTAL_YEARS=10`. So even a positive is at best
"promising, under-powered" — never a shippable edge on this window. The IC t-stat over
~3.6k pooled obs has real power to detect a weak edge; a clean NEGATIVE (IC≈0, t<2,
portfolio Sharpe<0.4 OOS) is conclusive that this lever has no usable edge at this scale.

## Decision rule
- IC right-sign with |t|≥3 AND OOS same-sign AND OOS portfolio Sharpe>0.4 with margin →
  "promising, needs ≥10y to confirm" (still not shippable). Else → NEGATIVE (no edge), shelve.

---

## RESULT (post-hoc, 2026-06-29) — NEGATIVE: no usable edge

Ran exactly as pre-registered. 521 daily books × 7 majors (n=3640 pooled obs).
Result in `trained_data/backtests/oanda_sentiment_result.json`.

| Window | Pooled IC (t) | Contrarian portfolio net Sharpe |
|---|---|---|
| Full | +0.0005 (t=0.03) | −0.111 |
| In-sample (65%) | +0.0100 (t=0.49) | +0.403 |
| Out-of-sample (35%) | **−0.0179 (t=−0.64)** | **−1.593** |
| By year | 2024 IC −0.022 / 2025 +0.024 / 2026 −0.019 | Sharpe +1.25 / −0.51 / −1.20 |

**Decision-rule check: FAILS all three.** (1) |t| never ≥3 (max |t|=0.64); the pooled
IC is ≈0. (2) IC FLIPS sign IS→OOS (+0.010 → −0.018). (3) OOS portfolio Sharpe is
−1.59, not >0.4 — the in-sample 0.40 was noise (t=0.49) that collapsed out-of-sample.
Per-year Sharpe sign-flips every year (no stability).

**Verdict: NEGATIVE — the OANDA position-book contrarian signal has no usable edge at
this scale.** An honest negative = success (it closes the last genuinely-untested lever
without self-deception). Shelved; NOT wired into anything. The data limitation (2y < 10y
gate) is moot — the signal is dead in-sample too, so more data wouldn't resurrect it.
Practice trend lane unchanged; directional transformer stays closed.
