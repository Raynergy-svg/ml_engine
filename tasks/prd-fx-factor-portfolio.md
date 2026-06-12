# PRD: Daily FX Factor Portfolio (carry + trend + value)

**Status:** Draft v1 — 2026-06-12 · operator-approved direction ("daily factor pivot")
**Replaces:** the intraday ML direction stack as the system's trading strategy.
**Hard gate:** NO deployment code (even practice-live) until US-007's ship gate passes.

## 1. Introduction / Overview

The 2026-06-10 verdict (commit dad8624) closed the intraday question: price-only M15
direction has no shippable edge (~52% val, >10% gap, all majors; the prior ~70% was a
leakage artifact). This PRD pivots the system to the strategy class with the strongest
documented, decades-persistent evidence that survives retail costs: a **low-frequency
systematic FX factor portfolio** — carry, trend, and value across G10 USD majors, with
volatility-targeted sizing. It reuses the project's real assets (cost-aware backtest
harness, walk-forward validation, execution/risk plumbing, fail-closed discipline) and
freezes the layers built for a signal that doesn't exist (agent voting, Tier 7
meta-pipeline, intraday scanning).

## 2. Goals

- A walk-forward, cost-aware (spread + financing) backtest of the 3-factor portfolio
  over ≥10 years of daily data on 7 USD pairs.
- A mechanical ship gate: **net Sharpe ≥ 0.4, positive in ≥6 of 10 years, max drawdown
  ≤ 25%, zero full-sample fitting** — evaluated by code, not judgment.
- If the gate passes: a weekly-rebalance practice deployment producing a verifiable
  track record whose tracking error vs. the backtest is itself measured.
- Honest expectations, in writing (see §5) — the deliverable at $1k capital is a
  **verified track record**, not income.

## 3. User Stories

### US-001: Daily data layer (7 pairs, 10y+)
**Description:** As the backtest, I need clean daily OHLC for EUR_USD, USD_JPY,
GBP_USD, AUD_USD, NZD_USD, USD_CAD, USD_CHF so factor signals have history to work on.
**Acceptance Criteria:**
- [ ] Loader fetches/caches OANDA `D` candles per pair to `market_data/factor/{PAIR}_D.csv`
- [ ] ≥2,600 rows per pair (≥10 years); hard-fail with a named error if a pair has less
- [ ] Gap validation: no missing stretch > 5 business days, else loud warning with dates
- [ ] No-mock tests: loader round-trips a tmp_path cache; validation triggers on a
      synthetic gapped frame
- [ ] pytest + flake8 green

### US-002: Carry signal
**Description:** As the portfolio, I need each pair's carry (interest-rate differential)
so I can be long high-yielders vs. funders.
**Acceptance Criteria:**
- [ ] Policy-rate differential per pair from a free source (FRED / central-bank series),
      monthly → daily forward-fill, stored alongside the price cache
- [ ] Signal = cross-sectional rank of differential (not raw level), recomputed daily,
      using ONLY data available at that date (no lookahead)
- [ ] Spot-check test: known rate regimes (e.g., 2015 CHF ≈ negative, 2023 USD high)
      assert the expected sign
- [ ] Window-invariance test in the style of `tests/test_feature_window_invariance.py`

### US-003: Trend signal
**Description:** As the portfolio, I need time-series momentum so I ride persistent moves.
**Acceptance Criteria:**
- [ ] TSMOM composite: sign of 3m, 6m, 12m total return (carry-adjusted close), equal-weighted
- [ ] Causal: signal at date T uses prices ≤ T only; invariance test proves the same date's
      signal is identical regardless of frame start
- [ ] No-mock tests on synthetic trending / mean-reverting series assert expected signs

### US-004: Value signal (staged — starts only after US-006 produces its first number)
**Description:** As the portfolio, I need a PPP/REER-deviation value factor for
diversification against carry crashes.
**Acceptance Criteria:**
- [ ] BIS REER (monthly, free) per currency; deviation z-score vs 5y mean; long cheap, short rich
- [ ] Same causality + invariance tests as US-002/003
- [ ] Backtest re-run with and without value; both results recorded in the report

### US-005: Portfolio construction + vol targeting
**Description:** As the trader, I want factor scores combined into target weights at a
fixed risk budget so position sizes are systematic, not vibes.
**Acceptance Criteria:**
- [ ] Combined score = equal-weighted average of available factor signals per pair
- [ ] Vol targeting: 10% annualized portfolio target via 60-day realized vol scaling
- [ ] Hard guards: gross leverage ≤ 4:1, per-pair weight ≤ 30% of gross, weekly rebalance
      with a no-trade band (skip trades < 5% weight change) to cut turnover
- [ ] Pure function: (signals, prices, equity) → target units; property tests for the guards

### US-006: Multi-asset cost-aware backtest
**Description:** As the operator, I need the honest number — net-of-cost performance over
10+ years — before any other work continues.
**Acceptance Criteria:**
- [ ] Extends/reuses `src/training/backtest_harness.py` discipline: fills at next day's
      open, per-pair spread table, **OANDA financing cost model** (carry is partly eaten
      by retail financing markup — model ±1%/yr markup explicitly)
- [ ] Walk-forward: any fitted parameter (vol lookback, bands) chosen on past data only
- [ ] Outputs: equity curve, net Sharpe, per-year returns, max DD, turnover, cost drag
      — written atomically to `trained_data/backtests/factor_portfolio_{ts}.json`
- [ ] ≥8 no-mock tests incl. a zero-cost vs. with-cost sanity spread and a lookahead canary

### US-007: Mechanical ship gate
**Description:** As the operator, I want the deploy decision made by code against the
pre-registered bar so I can't fool myself.
**Acceptance Criteria:**
- [ ] Gate function reads the US-006 artifact and returns PASS/FAIL with per-criterion detail:
      net Sharpe ≥ 0.4 · positive years ≥ 6/10 · max DD ≤ 25% · walk-forward flag true
- [ ] FAIL on any criterion = the project answer is "no deployable edge at this bar";
      report says so plainly
- [ ] Gate criteria live in code as constants with this PRD referenced; changing them
      requires an operator-signed commit message ("gate-change: ...")

### US-008: Practice deployment (BLOCKED until US-007 = PASS)
**Description:** As the operator, I want the passing portfolio traded weekly on the
practice account to build the track record.
**Acceptance Criteria:**
- [ ] Weekly job computes target weights and reconciles via `ExecutionManager` (practice)
- [ ] Drawdown guardian active; kill at 15% account DD (tighter than backtest max)
- [ ] Monthly tracking report: realized vs backtest-expected return distribution
- [ ] Runs with the agent-voting/meta layers OFF

### US-009: Freeze the intraday stack
**Description:** As the maintainer, I want the no-edge machinery frozen (not deleted) so
it stops consuming sessions.
**Acceptance Criteria:**
- [ ] A `factor` config profile: intraday scanning, 15-agent voting, meta-pipeline,
      self-heal, homework all disabled
- [ ] CLAUDE.md strategy section updated to name the factor portfolio as the active
      strategy and the intraday stack as FROZEN with this PRD as the why
- [ ] Nothing deleted; Tier 7 control plane documented as dormant-pending-signal

## 4. Functional Requirements

- FR-1: All signals computed from data available at signal date (causality enforced by test).
- FR-2: All randomness seeded; backtest reruns are byte-identical.
- FR-3: Costs modeled per trade: half-spread per side + financing accrual daily.
- FR-4: Every artifact (signals, weights, backtest report) written atomically with a
        version field; the factor pipeline gets its own `FACTOR_PIPELINE_VERSION`.
- FR-5: No mocks anywhere in new tests (project rule).
- FR-6: No LLM in any runtime/decision path (project rule, unchanged).
- FR-7: US-008 code physically cannot run unless the gate artifact says PASS (fail-closed).

## 5. Expectations — written down so they can't drift (operator-acknowledged)

At $1,000 capital, 10% vol target, and the gate-level Sharpe of 0.4–0.7, the honest
annual expectation is **≈ $40–$70, with drawdowns of $100–$250 along the way**.
**Turning $1k into $5k in a year (400%) is a NON-GOAL**: it would require ~8× this risk,
where the strategy's ordinary drawdowns exceed the account — expected outcome is ruin,
not income. The deliverable at this capital level is a verified multi-year track record
on honest instrumentation; capital scales after the track record exists, not before.

## 6. Non-Goals (Out of Scope)

- Intraday anything. No M15, no transformer, no agent consensus on entries.
- The news/macro experiment (runs separately under its own 2026-06-11 kill rule; if it
  ever ships, it ships as a *separate* signal, not inside this portfolio's v1).
- Leverage above 4:1 gross, ever, in any profile.
- Real-money deployment — out of scope for this PRD entirely; requires ≥6 months of
  practice track record within tracking bounds AND a separate operator decision.
- Crypto/equities/futures (IBKR work stays shelved).
- Any new model training. This strategy fits almost nothing by design.

## 7. Technical Considerations

- Reuse: backtest-harness discipline (next-bar fills, atomic reports), walk-forward
  validation patterns, ExecutionManager, drawdown guardian, window-invariance test style.
- Data realities to verify in US-001 before anything else: OANDA D-candle depth per pair
  (need 10y; if short, fall back to a free daily source for backtest-only history) and
  OANDA's actual financing rates vs. policy-differential proxy (US-006 models the gap).
- The 60d-realized-vol targeter is deliberately simple; the TCN vol head can be evaluated
  as an upgrade LATER, against the realized-vol baseline, never as a v1 dependency.

## 8. Success Metrics

- Primary: US-007 gate verdict produced from a ≥10-year walk-forward backtest — either
  outcome is success (deploy, or a closed question with evidence).
- If deployed: 6-month practice tracking — realized return within 2σ of backtest
  expectation; max practice DD ≤ backtest max DD; zero manual interventions.

## 9. Open Questions

- OQ-1: OANDA daily history depth for all 7 pairs — verified in US-001 (blocking).
- OQ-2: Financing-markup magnitude at OANDA retail (eats carry) — measured in US-006.
- OQ-3: Add crosses (e.g., AUD_NZD) in v2 for cross-sectional breadth, or stay USD-only?
- OQ-4: Weekly vs. monthly rebalance — US-006 reports both; pick by net-of-cost Sharpe.
