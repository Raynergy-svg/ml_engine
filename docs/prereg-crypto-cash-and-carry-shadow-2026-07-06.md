# Pre-registration — crypto cash-and-carry SHADOW lane (2026-07-06)

Frozen BEFORE running `scripts/experiment_crypto_cash_and_carry.py` or looking at any result.
Per repo convention (anti-p-hacking pledge, `docs/experiment-crypto-edge-hunt-2026-06-29.md` §5):
one construction, one param set, no sweep. Results are appended to §5 after the fact, never
edited into the frozen sections above.

## 0. Why this is a NEW construction, not a re-run of H1

H1 (`scripts/experiment_crypto_funding_carry.py`, pre-registered
`docs/experiment-crypto-edge-hunt-2026-06-29.md` §3) tested a **cross-sectional relative-value**
funding book: short the highest-funding quintile, long the lowest-funding quintile, dollar-neutral.
It failed the ship gate (OOS net Sharpe −0.25, DSR 0.11) — not because the funding premium was
fake (carry ann return was genuinely +0.32/yr OOS, Sharpe ~17, market-neutral at BTC-β −0.04), but
because **shorting the highest-funding names is shorting euphoria/momentum**, and that price
exposure flipped sign IS→OOS and ate the carry.

The construction below is the textbook **cash-and-carry** trade the operator asked for: **long
spot (proxy: perp price) vs short perpetual**, held per-asset. This is structurally different from
H1's cross-sectional spread book — it is delta-neutral BY CONSTRUCTION (the spot leg cancels the
perp's price exposure), so there is no price P&L term to adverse-select against. It only ever
harvests **positive** funding (the side actually accessible without a stock-borrow-style facility
for a short spot leg). Per the operator's instruction ("reuse its signal def, don't re-fit"), the
FUNDING SIGNAL, universe filters, and cost/vol-target constants are imported verbatim from the H1
and H2/H4 frozen harnesses — nothing here re-derives or tunes a parameter H1 or H4 already froze.

## 1. Frozen construction

- **Universe** (imported from H1, `experiment_crypto_funding_carry.build_panels`, unchanged):
  Binance USDT perps, ≥90d history, trailing-30d median quote-volume ADV ≥ $10M, survivorship-aware
  (delisted coins remain eligible while alive).
- **Signal** (imported from H1, unchanged): `s_i(t)` = trailing 3-day mean funding rate, known at
  end of day t−1 (causal, `.shift(1)`), `LOOKBACK_D=3`.
- **Selection rule (NEW — the only new judgment call, frozen here before any result)**: on each
  day, the book holds every eligible symbol with `s_i(t) > 0` (positive expected funding — the
  side a long-spot/short-perp position actually collects). Equal-weight across selected names,
  pre-overlay gross exposure normalized to 1.0. No short leg on negative-funding names (that would
  require reverse cash-and-carry — short spot, long perp — which needs a borrow facility this
  construction does not assume; disclosed as a scope limitation, not a bug).
- **P&L (the structural difference from H1)**: assumes the spot leg tracks the perp price
  essentially exactly (delta-neutral by design), so **there is no price P&L term** — daily return
  = funding collected minus cost. This is a modeling simplification: real spot–perp basis carries
  small residual noise (typically far smaller than the funding rate itself in liquid pairs) that is
  NOT modeled here. Disclosed explicitly, not hidden — see §2.
- **Cost** (imported `COST_BPS=10.0` from H1/H2, doubled): a cash-and-carry position has TWO legs
  (spot entry/exit + perp entry/exit), so realized cost = turnover × (2 × 10bps) = 20bps per
  round-trip, applied to the day's position turnover. This is a conservative (higher) cost
  assumption than H1's single-leg 10bps, not a favorable one.
- **Vol-target overlay** (imported from H2/H4, `experiment_crypto_h2_infra_stress`, unchanged):
  `TARGET_ANN_VOL=0.10`, `VOL_WINDOW=30`, `MAX_LEV=3.0` — scales gross exposure to target 10%
  annualized vol off the trailing 30d realized vol of the unlevered return series, capped at 3x.
- **Rebalance**: daily (matches the signal's own daily cadence; no weekly overlay — unlike H4's
  momentum lane, there is no momentum-chasing turnover motivation to dampen here).
- **Universe/backtest span**: same as H1 — 2020-01 → present. **OOS = 2024-01-01+** (untouched by
  this construction's judgment call; read once at verdict time, no sweep).

## 2. Honest scope limitations (disclosed up front, not discovered later)

1. **Basis risk assumed zero.** Real cash-and-carry P&L includes small spot–perp basis noise on
   entry/exit; this construction nets the spot leg against the perp leg analytically (proxying spot
   with the perp's own price series) rather than modeling a second, independent spot price feed.
   `src/crypto/data_layer.py` does not carry an independent spot OHLCV series for the full universe
   (only OKX BTC/major spot as a cross-check) — building one is out of scope for a shadow lane.
2. **Positive-funding-only.** No reverse-carry leg (short spot / long perp on negative funding) —
   real-world short-spot execution needs a borrow facility this construction does not assume exists
   for a long tail of alts. This UNDERSTATES the strategy's full capacity, not overstates it.
3. **Exchange-solvency / liquidation tail is not in this backtest at all.** A real cash-and-carry
   position is exposed to counterparty/exchange failure (the FTX-class tail) and to margin/
   liquidation risk on the perp leg during a violent basis blowout — neither is modeled by a daily
   mark-to-market backtest. This is the single most important caveat for this lane and is repeated
   verbatim in the AXIOM panel (see `src/crypto/carry_shadow.py` and `CryptoCarryPanel.tsx`).
4. Same survivorship/delisting caveat as `data_layer.py` and H1: dumps stop emitting data on
   delist; losses from a delisting event are UNDERSTATED, not overstated.

## 3. Gate (reused verbatim from H1 — `docs/experiment-crypto-edge-hunt-2026-06-29.md` §5)

Same six criteria as H1: OOS net Sharpe ≥ 0.40, maxDD ≤ 0.25, OOS-confirmed (no sign flip vs IS),
DSR ≥ 0.95 & Bonferroni bootstrap p < 0.05/3 (N_trials=3, shared multiple-testing budget with
H1–H3), |BTC-beta| ≤ 0.15, history ≥ 10y (structurally cannot be met with a 2020-01 start — binding
caveat, never relaxed, mirrors H1/H2/H3).

## 4. What ships regardless of the gate result

This is a **SHADOW lane**, not a live-promotion request. Per the operator's brief: stand up the
lane, harness-gated, so a genuine live-forward OOS record accumulates — whether or not the backtest
above clears the gate. No `trained_data/crypto_carry/SHIP_GATE.json` will be fabricated; the
LiveGate (`src/crypto/crypto_carry_live_gate.py`) starts and stays disarmed regardless of this
backtest's outcome, and no exchange/broker client exists for crypto in this repo — arming has no
execution path to route through even if a future gate pass occurred.

## 5. Results (appended after the fact — empty at pre-registration time)

### Cash-and-carry (positive-funding-only, daily): **NEGATIVE (gate FAIL)** (2026-07-07)

Run: `scripts/experiment_crypto_cash_and_carry.py` →
`trained_data/backtests/crypto_cash_and_carry_20260707T015642Z.json`.
Universe: 703 ever-liquid USDT perps, 2020-01→2026-06.

| Block | net Sharpe | net ann ret | carry ann | cost ann | avg turnover | maxDD |
|---|---|---|---|---|---|---|
| IS (pre-2024) | **+2.62** | +0.167 | +0.511 | 0.344 | 0.471 | −0.517 |
| **OOS (2024+)** | **−4.48** | **−0.155** | +0.267 | **0.421** | 0.577 | **−0.465** |
| Full | +0.79 | +0.043 | +0.417 | 0.374 | 0.512 | −0.640 |

OOS significance: PSR(>0)=0.00, DSR=0.00 (bench ann-Sharpe 0.54), bootstrap p(mean≤0)=0.9998.
BTC-β (OOS) = +0.005 (genuinely market-neutral, as designed — the delta-neutral construction
works as intended on the price-exposure axis).

**Gate:** net Sharpe≥0.40 ✗ (−4.48) · maxDD≤0.25 ✗ (0.47 OOS / 0.64 full) · OOS-confirmed ✗ (sign
flip vs IS) · DSR≥0.95 & p<0.0167 ✗ · |BTC-β|≤0.15 ✓ (+0.005) · history≥10y ✗ (by construction).
→ `clears_ex_history = FALSE`.

**Decomposition (honest, not rescued post-hoc):** the funding premium is real and large in every
block (carry ann +0.27 to +0.51/yr) — consistent with H1's finding that crypto funding carry is a
genuine, sizeable, market-neutral premium (BTC-β ≈ 0, confirming the delta-neutral construction
does what it's supposed to on the price-exposure axis). It dies here to **turnover cost, not price
adverse-selection** (the failure mode H1 had): the binary "funding > 0" membership test flickers
daily as marginal names cross zero, driving ~0.5–0.6 average daily turnover; at the conservative
20bps round-trip (two legs), that's 34–42%/yr in cost against a 27–51%/yr carry — cost eats most or
all of the premium, and in the OOS window (where turnover was highest, 0.577) it eats more than all
of it. The IS Sharpe of +2.62 is the exact overfit trap the OOS holdout exists to catch: a
naive backtest stopped at IS would have shipped a phantom edge. **This is the honest OOS-confirmed
negative, not a bug** — verified: BTC-beta near zero confirms delta-neutrality held, carry sign is
correct in every block (always positive, as expected for a positive-funding-only book), and the
IS→OOS turnover/cost trend (not a sign flip in the carry itself) is what drove the net collapse.

**Plain verdict:** the classic cash-and-carry construction, run naively with a same-day binary
funding-sign filter, does **not** clear the ship gate — turnover cost dominates a real and
sizeable funding premium. A natural next step (a holding-period/hysteresis filter to reduce
churn on marginal names) is explicitly NOT run here — doing so post-hoc after seeing this result
would be exactly the p-hacking this pre-registration exists to prevent. It would need its own
fresh pre-registration and an increment to the shared multiple-testing budget (`N_trials`).
**This shadow lane is stood up regardless of the gate result** (operator directive, §4) — the
frozen construction above runs forward to accumulate a genuine live-OOS record, same purpose as
the `crypto_momentum` (H4) and `track_b` shadow lanes.
