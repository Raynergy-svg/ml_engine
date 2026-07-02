# SEC EDGAR PIT fundamentals — VALUE + ACCRUALS factor pre-registration (2026-07-02)

Frozen BEFORE any factor Sharpe/DSR number is computed. Committed to git before the
experiment script is run. Anti-p-hacking: this is the whole test; no post-hoc window,
universe, or lag changes after seeing results (per L-018 lie-policy discipline).

## 0. What this is NOT re-testing (scope discipline, L-022)

The SEC EDGAR true-PIT fundamentals infra (`src/equity/edgar_fundamentals.py`,
`src/equity/sp500_membership.py`, `src/equity/pit_quality_eval.py`, cache
`market_data/equity/sp500_pit_fundamentals.json` + `sp500_prices.parquet`) **already
exists** and a QUALITY/PROFITABILITY composite (`+gross_margin, +net_margin,
-debt_to_equity`) was **already pre-registered, run, and independently verified twice**
on 2026-06-25 / 2026-07-01 — result: `trained_data/backtests/pit_quality_bakeoff.json`,
**robust negative** (full-sample net Sharpe margin vs EW **-0.600**, OOS **-0.324**;
clean margin-only variant merely **ties** EW, OOS margin +0.026). This experiment does
**not** re-run that test. It extends the same infra with two factors that have not yet
been tested with this true-PIT data source: **value** and **accruals**.

## 1. Data source (operator's pick, confirmed correct)

SEC EDGAR XBRL `companyfacts` API (`data.sec.gov`) — free, no key, official filer data.
PIT correctness: every fact carries its real `filed` date; factors are aligned on
**filing date, never period-end**; "as-originally-reported" = earliest-filed observation
per period (later filings carry restated comparatives — using them is lookahead).
Universe: survivorship-aware S&P 500 reconstruction (`sp500_membership.py`, Wikipedia
change-log, 873 ever-members incl. 279 since-removed). Prices: yfinance adjusted close,
**prices only, never fundamentals** (unchanged rule from the quality build).

## 2. New ingestion (extends `edgar_fundamentals.py`, does not modify existing outputs)

New XBRL concepts, same synonym-merge + earliest-filed-wins pattern as the existing
`gross_margin`/`net_margin`/`debt_to_equity`:

- `CashFlowFromOps`: `NetCashProvidedByUsedInOperatingActivities` (+ synonyms) — flow,
  annual (10-K only, full-year span), earliest-filed.
- `TotalAssets`: `Assets` — instantaneous stock concept, FY-end, earliest-filed.
- `SharesOutstanding`: `dei:EntityCommonStockSharesOutstanding` (10-K cover page,
  primary) with `us-gaap:CommonStockSharesOutstanding` fallback — instantaneous,
  earliest-filed. This is a **new taxonomy namespace** (`dei`, not `us-gaap`) — the
  loader must read `facts.dei` in addition to `facts["us-gaap"]`.

Existing fields (`gross_margin`, `net_margin`, `debt_to_equity`, revenue/NI/liab/equity
merges) are untouched. All new fields are additive to the per-period record; a record
missing an input for a given ratio leaves that ratio NaN (never fabricated/zero-filled).

## 3. Factors (pre-registered formulas, tilt=1.0, cross-sectional z, mean-of-available)

**VALUE** — classic price-scaled value, requires a PIT fundamentals × PIT price join
(new mechanism vs the pure-ratio quality panel):
- `book_to_market(t) = StockholdersEquity_PIT / (SharesOutstanding_PIT × Price(t))`
- `earnings_yield(t) = NetIncomeLoss_PIT(annual) / (SharesOutstanding_PIT × Price(t))`
- `value_score = mean available z of {+book_to_market, +earnings_yield}`
- Causality: `StockholdersEquity`/`NetIncomeLoss`/`SharesOutstanding` held at their
  `filed` availability date (forward-filled, never backfilled); `Price(t)` is same-day
  (public real-time) — the join is causal because the fundamentals leg is always the
  PIT-lagged, filing-availability value, never the future-refiled one.

**ACCRUALS** (Sloan 1996) — pure fundamental ratio, reuses the existing generic
`build_quality_panel(..., components=[...])` mechanism unmodified:
- `accruals(t) = (NetIncomeLoss_PIT - CashFlowFromOps_PIT) / TotalAssets_PIT`
  (all annual, filing-date PIT, `filed` = max of the three inputs' filed dates)
- `accruals_score = z of {-accruals}` (LOW accruals → predicted higher subsequent
  return / higher earnings quality — the sign is negative, per the published anomaly)

## 4. Universe, window, gate — identical to the already-verified quality bakeoff

Same wide survivorship-aware universe, `2012-01-01` → present, OOS = trailing 34%
(same split convention). Gate: canonical `src.factor.ship_gate.evaluate_gate`
(`MIN_NET_SHARPE=0.40, MIN_POSITIVE_YEARS=6, MIN_TOTAL_YEARS=10, MAX_DRAWDOWN=0.25,
walk_forward=True`), applied via the existing `evaluate_book_wide` NaN-safe
staggered-universe evaluator (unmodified). Costs: `DEFAULT_COST_BPS=2.0`,
`execution_lag=1`, no per-name slippage/ADV term (wide universe lacks ADV — same
documented limitation as the quality bakeoff, not new). EW baseline is **recomputed
in this run** (coverage differs slightly with the new concept requirements) for an
apples-to-apples comparison, not reused from the old JSON.

## 5. Multiple-testing budget (Bonferroni + Deflated Sharpe Ratio)

Family = "SEC-EDGAR-true-PIT equity fundamental factor" trials. Cumulative count:
1. Quality composite (2026-06-25, already run, negative) — **trial 1**.
2. Value composite (this experiment) — **trial 2**.
3. Accruals composite (this experiment) — **trial 3**.

`N_TRIALS = 3`. Bonferroni α = 0.05/3 ≈ **0.0167**. Reuses the already-tested
`deflated_sr` / `block_bootstrap_p` implementations from
`scripts/experiment_crypto_xs_signals.py` (not reimplemented) — DSR ≥ 0.95 AND
bootstrap p < 0.0167 required for "statistically significant", applied to each
factor's OOS net-return series independently. One labelled robustness sensitivity
per factor is permitted (mirroring the quality bakeoff's winsorized/component-only
checks) and will be reported regardless of sign — no silent drop of an unfavorable
variant.

## 6. Reporting commitment

Report, per factor, unconditionally: full-sample net Sharpe vs EW margin, OOS net
Sharpe vs EW margin, maxDD, positive-years/total-years, gate PASS/FAIL per criterion,
DSR + bootstrap-p vs the N=3 Bonferroni bar, and the one robustness check. A negative
result is reported as plainly as a positive one — an honest negative closes a real
question (per L-022 doctrine) and is not a failure of the work.

## 7. Independent verification (separate agent, no shared context, before finalizing)

A separate Model QA Specialist re-derives, from disk, independent of this document's
narrative: (a) PIT correctness of the 3 new concepts — filed-date alignment, no
period-end lookahead, no restated-value use, including the `dei` namespace join and
the price-join causality for the value factor specifically; (b) survivorship handling
unchanged/correct; (c) cost/turnover assumptions; (d) DSR/Bonferroni arithmetic and
the honesty of `N_TRIALS=3`; (e) confirms zero touches to halt/execution/env/hot-path
code. Verdict required before this experiment's result is reported as final.
