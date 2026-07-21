# Pre-registration — FINRA Daily Short-Sale Volume, novel alt-data lever

Frozen 2026-07-04T06:08Z, BEFORE any signal/backtest code is run. This document is committed
before results exist; if a later commit changes any number in this file, that is dredging and
must be called out explicitly, not silently edited.

## Why this source

The standard free-data doors are closed and documented (`.claude/LESSONS.md` L-018/L-020/L-022):
daily price direction, factor-zoo/carry/trend (FX + equities), news/macro NLP fusion, PIT
SEC-fundamentals factors, and agentic SEC-filing-text sentiment (Track B, N=420, IC fades to a
well-powered null) are all closed, negative results. Two research agents dispatched this session
independently surveyed novel free/alt-data candidates (Google Trends, Wikipedia pageviews,
Reddit/StockTwits, options OI/IV, short interest, intraday free tiers, cross-asset lead-lag,
incremental on-chain) and ranked **FINRA daily short-sale volume** highest on
expected-edge × honest-accessibility: genuinely free, zero ToS risk (public regulatory data),
and — critically — it is the only candidate offering a PIT-safe *and* free history deep enough to
build a real time-series panel. It also tests a data axis none of the closed doors touch: dealer
-reported short-side order flow (positioning), not price, fundamentals, news, or filing text.

Cross-asset lead-lag and intraday-price-quality upgrades were explicitly rejected by the research
agent as relabelings of already-closed levers (factor/carry verdicts) or infra upgrades, not new
signal types. Google Trends / Reddit were rejected for fragile PIT integrity and/or ToS risk and a
saturated research literature. Options OI/IV and incremental on-chain metrics were rejected because
the *useful* parts (skew, funding rates, netflow) sit behind paid tiers — not actually free.

## Verified access facts (checked live, 2026-07-04, this session)

- **Endpoint**: `https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt` — the
  consolidated-tape (all Reg NMS markets) daily short-volume file. No API key, no auth, no rate
  limit encountered. Requires a browser-like `User-Agent` header (a bare/no-UA request 403s; this
  is CDN bot-filtering, not an access restriction — confirmed by successfully fetching with
  `User-Agent: Mozilla/5.0`).
- **Format** (verified via direct fetch of `CNMSshvol20260702.txt`): pipe-delimited,
  `Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market`. One row per (date, ticker).
- **History depth — IMPORTANT, changes the analysis plan**: this is a *rolling* CDN retention
  window, not a fixed-start archive. Binary-searched live this session: `20190102` → HTTP 200,
  `20181231` → HTTP 200, everything tested in 2018 and earlier (`20180701`, `20180401`, `20180201`,
  `20180115`, `20180108`, `20180103`, `20170103`, `20160104`, `20150102`) → HTTP 403. Usable free
  history is therefore **~2019-01-02 through the present (~7.5 years)**, not the ~15 years initially
  assumed. This will keep rolling forward — a re-run of this experiment in 2027 will have a later
  start date available.
- **Consequence for the mechanical ship gate**: `src/factor/ship_gate.py:20`
  (`MIN_TOTAL_YEARS = 10`) is **structurally unmeetable** by this free source at ~7.5 years of
  depth, regardless of signal quality. Pre-registering this NOW so a `history_length` FAIL in the
  gate report is read correctly as a **data-availability ceiling**, not a signal failure. Per the
  Track-B precedent (which also could not clear the full 5-criterion ship gate and instead reported
  against the DSR/Bonferroni significance bar), **the primary verdict for this experiment is the
  significance gate** (below), with the mechanical `evaluate_gate()` output reported alongside for
  transparency, not as the deciding bar.

## Universe and price data (reused, not rebuilt)

- Equity price panel: `market_data/equity/sp500_prices.parquet` (684 tickers, 2012-01-03 through
  2026-06-24, close+volume). Cross-section for this experiment = tickers present in that panel
  **intersected** with tickers present in each day's FINRA file. No substitution, no backfill —
  a ticker missing from either side on a given day is simply absent from that day's cross-section.
- Sample period: **2019-01-02 through 2026-06-24** (capped at the price panel's known max date;
  memory already documents this cache as ~8 days stale as of 2026-07-02, which is an existing,
  orthogonal freshness gap, not something this experiment introduces).
- No new universe construction, no new survivorship logic — reusing the already-audited price
  cache means the survivorship properties (or lack thereof) of that panel apply here unchanged;
  this experiment does not add a NEW survivorship assumption on top.

## Signal construction (frozen before results)

`ShortVolumeRatio_t = ShortVolume_t / TotalVolume_t` per ticker per trading day.

**H1 (primary)** — abnormal short-volume PRESSURE (level) predicts continuation:
`z_t = (SVR_t - mean(SVR, trailing 60 sessions)) / std(SVR, trailing 60 sessions)`, computed
**per-ticker against its own trailing history only** (no cross-sectional peeking, no future data —
inherently PIT-safe by construction). Hypothesis: stocks in the **top quintile** of `z_t` (unusually
heavy short pressure) underperform stocks in the **bottom quintile** (unusually light short
pressure) over the following week. Long bottom quintile / short top quintile, equal-weighted
within quintile.

**H2 (secondary, exploratory but still gated)** — short-term reversal on the *change* in pressure:
`Δz_t = z_t - z_{t-5}` (5-session change). Hypothesis: a rapid INCREASE in short pressure predicts a
negative return over the next 1-3 sessions (fast liquidity-driven signal, distinct mechanism from
H1's persistent-pressure story). Same quintile L/S construction, shorter forward horizon.

Both hypotheses use only trailing/contemporaneous data as of trading day `t`; the FINRA file for
day `t` is published same-evening, so the earliest tradeable action is the `t+1` open — this
matches the existing harness's `execution_lag=1` convention exactly
(`src/equity/research/harness.py:22-23,332-337`), applied without modification.

## Reused infrastructure (no hand-rolled statistics)

- `src.equity.backtest.overlay` / `run_portfolio_backtest` — causal 10%-vol-target + drawdown
  de-gross overlay, `cost_bps` per-side turnover cost, `execution_lag=1` (shift(1) on weights).
- `src.equity.research.harness._rebalance_dates(prices.index, rebalance_step_days=5)` — weekly
  cadence (5 trading-day step). This is the deliberate power fix vs. the Track B binding
  constraint (`project_track_b_scaleup_verdict_2026_07_04`: "binding constraint now = #rebalance-
  dates ... only 1"). ~7.5 years × ~52 weeks ≈ **390 independent rebalances**, vs. Track B's single
  rebalance — this is the load-bearing reason this experiment can be well-powered where Track B
  could not be, independent of whether the signal itself turns out to be real.
- `src.equity.research.harness._quintile_long_short_weights`, `_build_weight_panel`,
  `_apply_overlay` — adapter approach: the SVR z-score (sign-flipped so higher = better, i.e.
  `composite = -clip(z_t, -1, 1)`) is packaged into the existing `ResearchScore` dataclass shape
  (`fundamental_quality` field carries the signal, `accounting_red_flags`/`forward_outlook` = 0,
  weights = `{"fundamental_quality": 1.0}`) so the frozen quintile/overlay/backtest machinery is
  reused byte-for-bit identical to Track B's, not reimplemented.
- `src.equity.research.harness._deflated_sharpe_ratio`, `_circular_block_bootstrap_sharpe_pvalue`,
  `_dsr_oos_n22` — DSR (Bailey & Lopez de Prado 2014) + fixed-seed (20260630) circular block
  bootstrap (block=21, reps=5000) Sharpe p-value. Called with an explicit `n_trials` override (see
  multiple-testing budget below) rather than mutating Track B's own historical trial count.
- `src.factor.ship_gate.evaluate_gate` — reported for transparency; expected structural FAIL on
  `history_length` per the retention-window finding above.

## Multiple-testing budget (frozen BEFORE results)

`src/equity/research/contracts.py:48-50` documents a running campaign count: `N_TRIALS = 22`
(edge-round-4 was 21; Track B was 22). This experiment registers **two new trials** (H1, H2) —
cumulative `N_TRIALS = 24`, `BONFERRONI_ALPHA = 0.05 / 24 ≈ 0.002083`. `contracts.py` is updated in
the same commit as this document to `N_TRIALS = 24` with a comment pointing at this file, so the
budget is monotonically honest for whatever experiment runs next. Both H1 and H2 must independently
clear `DSR ≥ 0.95 AND bootstrap p < 0.002083` to be called significant — no picking whichever of the
two happens to clear a laxer bar after the fact.

## Cost model

`cost_bps` = the harness's existing default (`DEFAULT_COST_BPS` in `src/equity/backtest.py`) — not
re-tuned for this experiment. `MIN_NET_SHARPE = 0.40` / `MAX_DRAWDOWN = 0.25`
(`src/factor/ship_gate.py:18,21`) are reported as secondary informational bars alongside the
significance gate, unchanged from the existing frozen constants.

## What counts as a pass, what counts as a fail

- **PASS (candidate for shadow-lane flagging only — no execution changes)**: H1 or H2 clears
  `DSR ≥ 0.95 AND bootstrap p < 0.002083` AND net Sharpe ≥ 0.40 AND max drawdown ≤ 0.25 over the
  full ~7.5-year sample (walk-forward: signal at `t` never uses information after `t`, which is
  true by construction here — no separate walk-forward split is layered on top since the z-score
  and forward return already respect strict time ordering).
- **FAIL**: either hypothesis fails the significance gate, the net Sharpe/drawdown bars, or the
  point estimate is directionally opposite to the pre-registered hypothesis (that is also a FAIL,
  not a "flip the sign and call it a discovery" — a sign-flip result gets reported as a dead end,
  matching L-018 dredging discipline).
- A FAIL on `evaluate_gate()`'s `history_length` criterion ALONE (with everything else passing) is
  reported as "no deployable edge at the mechanical ship-gate bar given current free-data depth,
  but a positive/significant point estimate at the significance gate" — the honest, narrower
  finding, not inflated language.

## What this experiment will NOT do

No execution/broker code touched. No `state.json` halt changes. No OANDA calls. Practice pin is
untouched throughout (this experiment is pure equities-research code with zero broker imports,
matching the crypto/Track-B shadow-lane pattern). If either hypothesis clears the significance
gate, it is flagged as a shadow-lane CANDIDATE only — not armed, not connected, not traded.
