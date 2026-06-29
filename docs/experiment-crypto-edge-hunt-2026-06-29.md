# Crypto-Native Edge Hunt — Pre-Registration (master)

**Date written:** 2026-06-29 (BEFORE any backtest result was computed)
**Branch:** ralph/equity-harvester-bot
**Status:** PRE-REGISTERED. Hypotheses, params, universe, costs, OOS split, and gate
are FROZEN below. Results are appended only in per-hypothesis verdict sections, after
the fact. Research/backtest only — **no live execution, no real money, no order path.**

This carries the FX/equity campaign's discipline (L-016/L-017/L-018) into crypto: the
durable asset is the gated harness — pre-registration, untouched OOS holdout,
multiple-testing correction, separate verifier, and an explicit
**return-alpha vs risk-control** decomposition. Honest negative = success. Dredged
positive = refuse.

---

## 0. Why crypto, and the binding honesty caveat up front

The campaign concluded free daily-bar **return prediction** in FX/equities is efficient;
only risk-control (trend drawdown management) survived. Crypto is tested here because it
has a genuinely crypto-native carry mechanism — **perpetual funding** — that does not
exist in FX/equities and is plausibly harvestable by a small systematic operator.

**Binding caveat (pre-registered, NOT a post-hoc excuse):** the canonical ship gate
requires `MIN_TOTAL_YEARS = 10` (`src/factor/ship_gate.py:17-21`). **Crypto perp funding
cannot meet this** — Binance USDⓈ-M funding begins 2020-01 (verified: 2019-12 → HTTP 404),
i.e. ~6.5 years for BTC and far less for the alt cross-section (~1.5 market cycles). The
history-depth criterion therefore **fails by construction for every crypto strategy**. We
do NOT relax it. The strongest attainable verdict is *"clears return / risk / OOS /
neutrality / multiple-testing, with a permanent insufficient-history caveat"* — never an
unqualified ship. (Same failure mode the EM-carry verdict hit on `history_length`.)

---

## 1. Data layer (built: `src/crypto/data_layer.py`)

| Source | Role | What | Depth | Survivorship |
|---|---|---|---|---|
| **Binance static dumps** (`data.binance.vision`) | **primary** | funding (8h) + klines (1d/1h) | 2020-01 → now | **complete** — 815 symbol dirs incl. delisted (LUNA, FTT, SRM, RAY…) |
| **OKX** public REST | CEX cross-check | OHLCV (→2020), funding (~3mo) | varies | live-listed only |
| **Hyperliquid** public REST | DEX cross-check | funding (hourly) | 2023-07 → now | live-listed only |

- **Geo note:** the Binance/Bybit *trading* APIs are geo-blocked from this environment
  ("restricted location"). The Binance *static dump* bucket is **not** blocked — that is
  the depth + survivorship backbone. OKX + Hyperliquid REST are reachable and serve as
  independent CEX/DEX cross-checks (does an edge replicate off Binance, and on a different
  funding mechanism — 1h DEX vs 8h CEX?).
- **Survivorship handling:** `list_binance_perp_symbols()` enumerates ALL archived symbols
  including delisted ones. The point-in-time universe (§3) includes a coin while it was
  alive and closes the position at last observed price when its archive ends.
  **LIMITATION (honest):** a delisting that ends the archive is modeled as a flat close,
  which **understates** true delisting losses (no liquidation/-100% gap modeled). This
  biases carry results *optimistically* on the dead-coin tail — stated, not hidden.
- **No secrets:** zero API keys; all endpoints public. Cache → `crypto_cache/`
  (matches `*_cache/` in `.gitignore`; never committed).
- **Verified 2026-06-29 (real fetch):** BTC funding 7,029 rows 2020-01-01→2026-05-31
  (~12%/yr mean); LUNAUSDT funding stops 2022-05-13 (the collapse) and auto-flags
  `delisted=True`; FTTUSDT flagged delisted; OKX BTC OHLCV 2,407 rows to today; HL BTC
  hourly funding 26,274 rows since 2023-07. Monthly dumps lag ~1 month (current = 2026-05),
  so OOS effectively runs to ~2026-05.

---

## 2. Cost model (pre-registered, applied to ALL hypotheses)

Crypto edges die to costs, so costs are modeled before results, not tuned after.

- **Taker fee:** 5 bps per side (Binance USDⓈ-M taker ≈ 4.5–5 bps; conservative).
- **Slippage:** 5 bps per 1% ADV participation (reuses `src/equity/ship_gate.py:83`
  `DEFAULT_SLIPPAGE_BPS_PER_PCT_ADV`). Pre-registered participation = **1% of trailing
  30d ADV** per name ⇒ 5 bps slippage per side. Combined entry/exit ≈ **10 bps per side**.
- **Cost charge:** `cost_t = turnover_t × (taker + slippage)` where
  `turnover_t = Σ_i |Δweight_i|`. Funding payments are NOT a "cost" — they are the
  signal/return, modeled exactly from data (§3, H1).
- **Stress robustness (reported, not a separate hypothesis):** re-run each headline at
  **2× costs** (20 bps/side). A result that only survives at 1× and dies at 2× is flagged
  fragile.

---

## 3. Pre-registered hypotheses (ONE canonical rule + ONE param set each — NO sweep)

**Shared universe (point-in-time, survivorship-aware):** at each daily rebalance `t`, a
Binance USDⓈ-M `*USDT` perp is **eligible** iff (a) ≥ **90 days** of funding+price history
as of `t`, (b) trailing-30d median `quote_volume` ≥ **$10M** ADV, (c) listed at `t` (has
data). Eligible coins that later delist remain eligible while alive. Within the eligible
set, rank by the hypothesis signal and form **top/bottom quintiles (20%)**, equal-weight
within leg, scaled **dollar-neutral** (Σlong$ = Σshort$), gross exposure = 1.0.
**Execution lag = 1 day** (signal at `t`, position held `t→t+1`; causal, matches
`src/factor/backtest.py` convention). Daily rebalance.

### H1 — Perp funding carry (LEAD candidate)
- **Economic logic:** when funding > 0, perp longs pay shorts each interval → a short
  perp *receives* funding; when funding < 0, a long receives it.
- **Signal:** `s_i(t)` = trailing **3-day** mean funding rate of coin `i` (last 9 ×8h
  intervals). One value, no sweep.
- **Rule:** SHORT top-quintile by `s` (highest funding → receive), LONG bottom-quintile
  (lowest/most-negative funding → receive), dollar-neutral, daily.
- **P&L:** `ret_t = price_pnl(dollar-neutral book, t→t+1) + funding_collected_t − cost_t`.
- **Return-vs-risk decomposition (the anti-"beta dressed as alpha" test):** report
  separately (i) GROSS carry = funding component, (ii) PRICE P&L of the neutral book,
  (iii) **BTC-beta** of strategy returns (regress on BTC daily return). If net return is
  mostly negative price-beta rather than harvested funding, it is **short-market risk
  premium, not alpha** → labeled as such.

### H2 — Cross-sectional momentum (ONE canonical rule)
- **Signal:** `m_i(t)` = trailing **14-day** close-to-close return. One value, no sweep.
- **Rule:** LONG top-quintile by `m`, SHORT bottom-quintile, dollar-neutral, daily, same
  costs & lag. Decomposition: BTC-beta + IS/OOS split.
- (Short-horizon reversion is NOT part of H2. If pursued later it is a NEW pre-registered
  test that increments the multiple-testing count `N`.)

### H3 — Order-flow imbalance (CONDITIONAL — run only if H1 or H2 is clean)
- **Signal:** `OFI_i(t)` = trailing-1d taker-buy ratio (`taker_base / volume` from klines)
  − 0.5. Free microstructure signal, orthogonal to funding/price.
- **Rule (contrarian):** LONG bottom-quintile OFI (faded selling), SHORT top-quintile OFI
  (faded buying), dollar-neutral, daily, same costs & lag. One value, no sweep.

---

## 4. OOS holdout (pre-registered, touched ONCE)

- **In-sample (IS):** 2020-01-01 → 2023-12-31 (sanity / construction; covers 2021 bull +
  2022 FTX bear).
- **Out-of-sample (OOS):** 2024-01-01 → present — **the search never touches this**; read
  once at verdict time. Per-hypothesis params are frozen in §3 with **no sweep**, so the
  IS-overfit channel is minimal by design; the holdout additionally guards the universe /
  lookback judgment calls. Verdict is judged on **OOS**, with IS shown for context.

---

## 5. Gate (pre-registered — crypto variant; reuses canonical thresholds)

A hypothesis "**clears (ex-history)**" iff ALL of:
1. **OOS net Sharpe ≥ 0.40** (after 1× costs) — `MIN_NET_SHARPE`, `src/factor/ship_gate.py:17`.
2. **Max drawdown ≤ 0.25** (full sample, after costs) — `MAX_DRAWDOWN`.
3. **Causal / walk-forward == True** (execution lag ≥ 1, point-in-time universe, no future
   funding).
4. **OOS-confirmed:** OOS net Sharpe ≥ 0.40 AND no sign-flip vs IS.
5. **Multiple-testing survives:** Deflated Sharpe Ratio (Bailey & López de Prado) with
   `N_trials = 3` remains significant (DSR ≥ 0.95) AND Bonferroni bootstrap p < 0.05/3 =
   0.0167. **Effective-N caveat:** crypto names are highly cross-correlated, so the
   effective number of independent bets ≪ nominal — the separate verifier re-checks this
   (naive t-stats overstate significance).
6. **Market-neutral return:** |BTC-beta| ≤ 0.15 AND the return is harvested carry/spread,
   not disguised market beta. Failing this ⇒ labeled **risk-premium/beta, not alpha**.
7. **History depth (`MIN_TOTAL_YEARS = 10`): CANNOT BE MET** → permanent caveat (see §0).
   Best attainable verdict = "clears ex-history with binding depth caveat."

**Anti-p-hacking pledge:** one signal + one param set per hypothesis, frozen above. A
failing hypothesis is recorded as a negative — params are NOT tweaked to rescue it. Any
post-hoc variant is a new pre-registered test that increments `N` and is reported as such.

**Multiple-testing budget:** `N = 3` primary tests (H1, H2, H3). 2× cost is robustness
(not selection), does not increment `N`.

---

## 6. Separate verifier (independent, per L-018)

After each hypothesis runs, a separate Code-Reviewer/quant subagent independently:
(a) re-derives the headline OOS Sharpe / maxDD / beta from cached data + script,
(b) audits for leakage (lag, point-in-time membership, no future funding, no survivorship
peek), (c) re-confirms the DSR/Bonferroni math and flags effective-N. Verdict recorded
here + commit `eval(crypto-<h>): … VERDICT (<CLEARS-EX-HISTORY|NEGATIVE> CONFIRMED)`.
A verifier-caught false claim triggers the L-018 fail-closed reject + quarantine.

---

## 7. Results (appended after the fact — empty at pre-registration time)

### H1 — funding carry: **NEGATIVE (gate FAIL) — independently CONFIRMED** (2026-06-29)

Run: `scripts/experiment_crypto_funding_carry.py` →
`trained_data/backtests/crypto_funding_carry_20260629T213504Z.json`.
Universe: 693 ever-liquid USDT perps (27 delisted-during-sample), 2020-01→2026-05.

| Block | net Sharpe | gross Sharpe | carry ann | price ann | cost ann | maxDD | BTC-β |
|---|---|---|---|---|---|---|---|
| IS (pre-2024) | **+1.98** | +2.78 | +0.175 | **+0.410** | 0.169 | −0.265 | −0.00 |
| **OOS (2024+)** | **−0.25** | +0.60 | +0.319 | **−0.207** | 0.158 | **−0.299** | **−0.04** |
| Full | +1.20 | +2.01 | +0.229 | +0.178 | 0.165 | **−0.381** | −0.01 |
| Stress 2× OOS | −1.09 | +0.60 | +0.319 | −0.207 | 0.316 | −0.435 | −0.04 |
| Survivor-only OOS | −0.31 | +0.54 | +0.319 | −0.218 | 0.158 | −0.305 | −0.04 |

OOS significance: PSR(>0)=0.35, DSR=0.11 (bench ann-Sharpe 0.55), bootstrap p(mean≤0)=0.63.

**Gate:** net Sharpe≥0.40 ✗ (−0.25) · maxDD≤0.25 ✗ (0.30 OOS / 0.38 full) · OOS-confirmed
✗ (sign flip vs IS) · DSR≥0.95 & p<0.0167 ✗ · **|BTC-β|≤0.15 ✓ (−0.04)** ·
history≥10y ✗ (by construction). → `clears_ex_history = FALSE`.

**Decomposition (return-vs-risk):** the carry IS real and large (+0.32/yr OOS, Sharpe ~17 —
a low-variance funding drip), and the book IS genuinely **market-neutral** (BTC-β −0.04 —
NOT short-beta, NOT "beta dressed as alpha"). It dies to (1) **price adverse-selection**:
shorting the highest-funding coins = shorting euphoria/momentum, whose payoff **flipped sign
IS→OOS** (+0.41/yr → −0.21/yr) — a non-stationary price relationship; and (2) **costs**
(~0.16/yr from 0.43 daily turnover). Dead-coin carry inflation ≈ 0 in OOS (LUNA/FTT deaths
fell in the IS window). Full-sample Sharpe (+1.20) would have LIED; the OOS holdout caught it.

**Separate-verifier (Code Reviewer, full-universe independent re-derivation):** MATCH on every
sign/magnitude (OOS net Sharpe −0.287 vs −0.247; carry-ann & β near-exact). Causality: no
lookahead (signal & ADV `.shift(1)`, P&L from prior-bar weights). Survivorship: dead coins
held while alive, NOT dropped; LUNA's −99.97% crash IS captured (only the final liquidation
gap uncounted → tail under-count is small and biases carry *optimistically*). Costs/funding
sign-correct, no double-count. **VERDICT: TRUSTWORTHY — honest negative, no bug.**

**Plain verdict:** Naive cross-sectional perp funding carry has **NO cost-surviving,
OOS-confirmed market-neutral RETURN alpha** at this scale. The funding premium is real but is
a payment for taking the losing, non-stationary side of short-term momentum, and costs finish
it. Honest negative = success. (Binding caveat stands: even had it cleared, ~6.5y / ~1.5
cycles < 10y → never an unqualified ship.)

### H2 — XS momentum: _pending run (HELD for operator go-ahead)_
### H3 — order-flow imbalance: _pending (conditional)_
