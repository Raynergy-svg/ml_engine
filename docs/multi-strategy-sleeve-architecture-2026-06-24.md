# Multi-Strategy Sleeve Architecture — design plan (2026-06-24)

**Status: PLAN ONLY. `running:NO`. Nothing here is built, run, promoted, or unhalted.**
Immutables unchanged: practice-only, ship gate is the arbiter, FX legacy untouched,
`halted=true`, "live mode" denied, no real money. Every step is tagged **[non-hot-path]**
(buildable offline now) or **[HOT-PATH — operator-authorized]** (touches the live trade
path; held for explicit operator go, like H1/H2/H3).

## Why this plan exists (the load-bearing insight)

A fund is many **weakly-correlated positive-expectancy return streams combined**, not one
strategy iterated. Buddy has exactly one shippable strategy today: the long, vol-targeted,
equal-weight equity harvester (`src/equity/harvester_strategy.py`), which passes the ship
gate and *beats* every variation we tested. Those tests are why this plan is shaped the way
it is — they were the wrong kind of "more":

- Multi-horizon vol blend (`multi_horizon.py`) — lost (−0.054 full / −0.089 OOS Sharpe).
- Cross-sectional tilts low-vol / HRP (`variant_eval.py` bake-off) — all lost; quality
  un-evaluable (no PIT fundamentals).

All of these are **correlated re-slicings of the same equal-weight book** — different
*timeframes/weightings of one return stream*. Adding correlated frames to one stream cannot
raise the stream's Sharpe much; the math says so and the gate confirmed it 4×. The real lever
is **orthogonal streams**: `Sharpe_combined ≈ Sharpe_avg / sqrt(avg_pairwise_corr)`. Two
sleeves at Sharpe 0.6 and pairwise corr 0.1 combine to ≈0.8+; the same two at corr 0.9 give
≈0.6. **The asset is uncorrelated breadth, plus a disciplined process to find it.**

---

## 1. The Sleeve abstraction (and how little new structure it needs)

A **sleeve** is one independently ship-gated return stream: `(instrument set) → (causal signal
→ long-only target weights) → (its own backtest + ship-gate) → (its own execution lane share)`.
The harvester is **sleeve #1**. The good news: the contract already exists in pieces.

**Existing anchors (reuse, don't reinvent):**
- Signal→weights contract: `src/equity/strategy.py::EquityStrategy` protocol —
  `compute_target_weights(asof) -> Dict[str, float]` (causal, long-only, bounded). A sleeve
  IS an `EquityStrategy`-shaped object over its own instrument set.
- Per-sleeve evaluation: `src/equity/variant_eval.py::evaluate_book(snapshot, prices,
  base_weights, ...)` — **this is already a generalized per-sleeve gate evaluator.** It takes
  ANY long-only weight panel, applies the overlay + cost-aware backtest, and scores
  `src/factor/ship_gate.py::evaluate_gate` (net_sharpe≥0.40, pos_yrs≥6, total_yrs≥10,
  maxDD≤0.25). `evaluate_book(EW)` reproduces `ship_gate.evaluate_harvester` to <1e-9.
- Per-sleeve return series: `src/equity/backtest.py::run_portfolio_backtest` → `BacktestResult`
  (the net P&L series each sleeve contributes to the combiner).
- Execution lane: `src/equity/rebalance.py::RebalanceScheduler` (plan→orders) →
  `executors.py`/`order_lifecycle.py` → IBKR. Already a separate, FX-isolated lane.

**New structure required (small) [non-hot-path]:**
- `SleeveSpec` / `SleeveResult` dataclasses + a `SleeveRegistry` (name, instrument-set,
  weight-fn, params, ship-gate artifact path, status ∈ {candidate, shadow, live, retired}).
- Per-sleeve ship-gate artifact: `trained_data/backtests/SHIP_GATE_<sleeve>.json` (one gate
  verdict per sleeve, same schema as `SHIP_GATE.json`). Sleeve cannot leave `candidate`
  without its own gate PASS — re-derived by the separate verifier.
- Per-sleeve instances of the Pillar-2/3 instrumentation already built:
  `decision_gate.py::decide_cycle` (abstain/halt/no-act) and `cycle_ledger.py` (tamper-evident
  per-cycle record) become per-sleeve.

**Contract a sleeve must satisfy (the gate the registry enforces):**
1. `compute_weights(asof) -> long-only weight panel` over its declared instrument set, **strictly
   causal** (data ≤ t−1; `shift(1)` discipline — the #1 lie vector, enforced by a causality test
   like the bake-off's perturbation test).
2. A registered `instrument_set` + data source with documented availability (no fabricated data).
3. A passing OOS ship-gate verdict, independently re-derived by the separate verifier.
4. A net return series for the combiner.

---

## 2. Candidate uncorrelated sleeves (honest about data gaps)

Long-biased, low-turnover, retail-feasible. Ranked by near-term viability for THIS system.

### Sleeve A — Trend / managed-futures-style, multi-asset (long-or-flat) — **BEST near-term candidate**
- **Thesis:** time-series momentum across a broad multi-asset ETF set (e.g. SPY/QQQ equity,
  TLT/IEF bonds, GLD gold, DBC/commodity, maybe VNQ REIT). Hold the asset when its own trend is
  up (e.g. price > 200d MA / positive 12-1 momentum), go to **cash** when down. Long-only/long-flat
  — no shorting, no leverage.
- **Why weakly correlated to the equal-weight equity book:** (a) spans bonds/gold/commodities,
  not just equities; (b) the long-or-flat switch means it **goes to cash in equity drawdowns** —
  so it is *negatively* correlated to the equity book exactly when that matters (2008/2020). This
  is the classic crisis-diversifier. Managed-futures' low/negative equity correlation is the most
  robust documented diversification benefit in the literature.
- **Data/instruments:** daily ETF prices — **yfinance-feasible today** (same pipeline as the
  bake-off). No fundamentals, no paid data. Low turnover (monthly). **This is buildable offline now.**
- **Honesty caveats:** ETF history is shorter than 2006 for some sleeves (commodity ETFs ~2006+);
  trend has decayed in some asset classes post-2010 (we saw this in the FX factor verdict — trend
  went negative post-QE on FX majors). Multi-asset trend is more robust than single-asset, but the
  gate (and OOS) is the arbiter, not the prior.

### Sleeve B — Defensive / quality equity — **blocked on data**
- **Thesis:** AQR QMJ long-leg: overweight profitable, low-leverage, stable-earnings firms.
- **Why weakly correlated:** quality has a different cyclicality than market-cap/equal-weight
  beta; defensive in drawdowns. BUT it is still a **long-equity** sleeve → meaningfully correlated
  to the equal-weight book (corr likely 0.7–0.9). It is a *return-quality* improver more than a
  *diversifier*. Honest expectation: modest, not a true orthogonal stream.
- **Data/instruments:** needs **point-in-time fundamentals** (ROE/leverage/earnings-stability)
  for the universe. **We do not have this** — confirmed: financial-datasets returns latest-FY
  snapshot only (and $0 balance in this env); yfinance is current-snapshot only. The
  `quality.py` tilt **mechanism** exists and is tested, but produces no result without a real PIT
  panel. **Blocked until a PIT data source is acquired** (paid Sharadar/Compustat, or a robust
  SEC-EDGAR XBRL parser — itself a project).

### Sleeve C — Carry (bond/credit/FX via ETFs) — **known-hard, tested, deprioritize**
- **Thesis:** harvest yield/carry (e.g. long higher-yielding bond/credit ETFs vs cash).
- **Why weakly correlated:** different driver (rates/credit), BUT — we **already tested EM/global
  carry** (`scripts/experiment_em_carry.py`): full-cycle net Sharpe ≈0.50 with **40% maxDD and a
  fat left tail** (carry crashes every EM crisis) → **fails the drawdown gate**. Carry's "weak
  correlation" is a lie in the left tail: it correlates to *everything* in a crisis. **Deprioritize
  unless a crash-aware overlay is designed; the gate already rejected the naive version.**

### Sleeve D — Low-vol / min-variance equity — **NOT a diversifier**
- Tested as a tilt; it's a **long-equity** stream → correlated to the equal-weight book (it cut
  drawdown but lost Sharpe). Useful as a risk-reducing *re-weighting of sleeve #1*, **not** an
  independent sleeve. Don't double-count it as breadth.

**Honest summary:** exactly **one** genuinely orthogonal, retail-feasible, data-available
candidate exists right now — **Sleeve A (multi-asset trend/managed-futures)**. B needs a data
unlock; C is gate-rejected as-is; D isn't a real diversifier. So the multi-strategy thesis is
real but **bottlenecked on (1) sleeve A actually clearing the gate and (2) the data unlock for B**.

---

## 3. Combining sleeves (the legitimate "multiple frames")

Each sleeve produces a net return series (its `BacktestResult` P&L). The **combiner** allocates a
risk budget across sleeves, correlation-aware.

- **Allocation methods (offline, escalating sophistication):**
  1. Inverse-vol across sleeves (simple, robust): weight ∝ 1/σ_sleeve, re-estimated on a trailing
     window (causal, `shift(1)`).
  2. Correlation-penalized / risk-budget: down-weight sleeves correlated to the existing book.
  3. **HRP across sleeves** — and here is the irony: **HRP failed as a cross-sectional *stock*
     weighter (bake-off) but is the *natural* sleeve-combiner.** `src/equity/hrp.py` already exists;
     point it at the **sleeve return matrix** instead of single-name returns. HRP's hierarchical
     risk-budgeting is designed exactly for combining a handful of correlated/uncorrelated *strategies*.
- **Why this is the legit multiple-frames (not the timeframe trap we just disproved):** combining
  weakly-correlated *positive-expectancy* streams raises the *combined* Sharpe via
  `Sharpe_avg / sqrt(avg_corr)`. Multi-horizon/tilt variants failed because their "frames" were
  ~1.0 correlated to the same stream — zero diversification. Sleeve A is genuinely different
  instruments + a crisis-flat switch → low/negative corr → real diversification.
- **The combined book gets its OWN gate:** the *portfolio of sleeves* must clear
  `evaluate_gate` (net Sharpe≥0.40, maxDD≤0.25, etc.) on its combined return series, OOS —
  not just each sleeve individually. A sleeve that passes alone but wrecks the combined drawdown
  does not promote. New artifact: `trained_data/backtests/SHIP_GATE_book.json`. **[non-hot-path]**
  — `evaluate_book` already gives per-sleeve series; the combiner + book-gate is new offline code.

---

## 4. How the agentic self-improver loop drives this

The loop's durable, hedge-fund-grade asset is not any sleeve — it is **running a disciplined quant
research process** that we have already demonstrated works (we killed 4 candidate ideas honestly).
The loop becomes a **sleeve factory**:

1. **Generate** a candidate sleeve (instrument set + causal signal) — operator-seeded or
   loop-proposed.
2. **Backtest causally, OOS** via `evaluate_book` / the book-gate. Strict `shift(1)`/no-lookahead.
3. **Independently re-derive** the gate verdict with the **separate Code-Reviewer verifier**
   (re-run from scratch, reproduce numbers, confirm causality, confirm data is real-or-honestly-
   labeled) — exactly the L-018 protocol we just ran 3×. **Self-attestation is not accepted.**
4. **Kill losers honestly** — a sleeve that doesn't clear **+1 over the existing book OOS** is a
   negative result, recorded as such (this is a success, per the operator's bar). Anti-lie
   discipline (L-017 `running:NO`, L-018 verifier-caught-lie policy) on every claim.
5. **Promote only +1 winners** — and promotion to *live* is **[HOT-PATH — operator-authorized]**,
   never autonomous. The loop can promote `candidate→shadow` offline; `shadow→live` is the
   operator's typed decision via `live_gate.LiveGate` (NAV≤25%, "LIVE" confirm).
6. **Monitor + retire** — live sleeves are watched for decay (rolling Sharpe vs gate); a decayed
   sleeve is retired (weight→0) by the combiner, gated and verifier-checked. `cycle_ledger.py`
   (Pillar 2) records every sleeve's cycles tamper-evidently; `decision_gate.py` (Pillar 3) abstains/
   halts per sleeve on stale data / drawdown / global halt.

This is the cybernetic loop pointed at *strategy breadth* instead of one strategy: incident→
propose→gate→soak→promote→retire, Claude never in the per-trade hot path, the separate verifier
as the trust anchor on every promotion.

---

## 5. Phased roadmap (each phase gated by the operator's "+1 tested to help" bar)

### Phase 0 — Sleeve scaffolding + first orthogonal sleeve, OFFLINE  **[non-hot-path, `running:NO`]**
- Formalize `SleeveSpec`/`SleeveResult`/`SleeveRegistry` + per-sleeve `SHIP_GATE_<sleeve>.json`
  (reuses `evaluate_book`, `ship_gate`, `backtest`). Harvester registered as sleeve #1.
- Build **Sleeve A (multi-asset trend)** as an offline sleeve; ship-gate it OOS; separate-verifier
  re-derive. **Honest kill if it doesn't clear the gate** (likely outcome for some asset sets —
  trend has decayed; the gate decides).
- Build the **combiner** (inverse-vol → HRP-across-sleeves) + the **book gate**
  (`SHIP_GATE_book.json`). Evaluate `{harvester + sleeve A}` combined, OOS. **The real test:** does
  the *combined book* beat the harvester-alone Sharpe by a real OOS margin? That is the multi-
  strategy thesis, falsifiable.
- Exit criterion: combined book clears **+1 over harvester-alone OOS**, verifier-confirmed. If not,
  honest negative — the breadth isn't there yet at this data scale, reported as such.

### Phase 1 — Data unlock (gates the rest of the breadth)  **[non-hot-path; needs external data]**
- Acquire **point-in-time fundamentals** (Sharadar/Compustat license, or a vetted SEC-EDGAR XBRL
  pipeline) → unblocks **Sleeve B (quality)** via the existing `quality.py` mechanism.
- Acquire **survivorship-free universe** history → truer drawdowns (the current 38-name universe is
  survivor-biased; 0.229 maxDD understates the 2008 tail — documented caveat). This raises the
  honesty of every gate verdict, including the harvester's.
- **This phase is a procurement/operator decision, not something the loop can self-serve.** Flag for
  operator: the multi-strategy ceiling is data-bound, not code-bound.

### Phase 2 — Live multi-sleeve book  **[HOT-PATH — OPERATOR-AUTHORIZED, each step]**
- Wire the combined book into the live equity lane. The driver already exists but is **orphaned**:
  `src/equity/control_loop.py::AutonomousLoop` (no entrypoint invokes it — the held H1). Promotion
  sequence per sleeve and for the book:
  - H1 (entrypoint that invokes the runner/loop) — **operator-authorized**, separate process, no
    `scripts/`/`execution.py`/`main.py` edit without typed go.
  - shadow → live via `live_gate.LiveGate` (typed "LIVE", NAV≤25%), `kill_switch`, `risk_agents`
    drawdown guardian — all already built; each arming is operator-typed, never autonomous.
  - `decision_gate`/`cycle_ledger` per sleeve enforce abstain/halt/tamper-evidence live.
- **Nothing in Phase 2 happens without explicit operator authorization and a verifier-PASS on the
  promotion.** practice-only, no real money, `halted=true` until the operator unhalts.

### Per-phase gate (non-negotiable)
Every phase advances only on: **separate-verifier PASS** (independent re-derivation) + **+1 OOS
margin** over the incumbent + **running-status honesty** (`running_status.py`) + **no Hard-NO
touched**. A phase that fails its bar is a recorded negative, not a reason to relax the bar.

---

## Hot-path / non-hot-path ledger (quick reference)

| Step | Bucket |
|---|---|
| SleeveSpec/Registry, per-sleeve gate artifact, `evaluate_book` reuse | **non-hot-path** |
| Sleeve A (trend) offline backtest + ship-gate | **non-hot-path** |
| Sleeve combiner (inv-vol / HRP-across-sleeves) + book gate | **non-hot-path** |
| Per-sleeve `decision_gate`/`cycle_ledger` instrumentation | **non-hot-path** |
| Loop-as-sleeve-factory (generate→gate→verifier→kill/promote-to-shadow) | **non-hot-path** |
| Acquire PIT fundamentals / survivorship-free data | **non-hot-path (external procurement)** |
| Quality sleeve B (once data exists) | **non-hot-path** |
| Invoke the live driver (`AutonomousLoop` entrypoint, H1) | **HOT-PATH — operator-authorized** |
| shadow→live arm (`LiveGate`), real orders | **HOT-PATH — operator-authorized** |
| unhalt the book | **operator-only (owns the halt)** |

## Bottom line for the operator
- The multi-strategy/sleeve architecture is a **small addition** on top of what exists
  (`evaluate_book` is already the per-sleeve gate; `hrp.py` is already the sleeve-combiner;
  `EquityStrategy` is already the signal contract; the live lane + Pillars already exist).
- The **one near-term orthogonal sleeve worth testing now is multi-asset trend (Sleeve A)** —
  price-only, retail ETFs, buildable offline. Everything else is data-bound (B) or gate-rejected (C).
- The **durable asset is the loop running a disciplined, verifier-anchored research process** that
  kills losers honestly and promotes only verifier-confirmed +1 winners — which we have already
  demonstrated 4×.
- **This is a plan. Nothing is built, run, promoted, or unhalted. running:NO.** Phase-0 is buildable
  non-hot-path on your word; Phase-2 live steps each need your explicit authorization.
