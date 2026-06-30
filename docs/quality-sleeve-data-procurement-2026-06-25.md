# Quality-Sleeve PIT Data — Procurement Spec & Runbook (2026-06-25)

**Status:** operator authorized the PIT fundamental-data purchase (2026-06-25).
Receiving end BUILT and tested (`src/equity/quality_data.py`, 9 tests). Quality
sleeve remains **`NOT_EVALUATED`** on disk until real Pro data is sourced — no
fabrication (L-018). running:NO throughout; this is an OFFLINE evaluation.

## What I can / can't do
- **Can't:** enter payment or click "upgrade" — that is the operator's action, and
  no real money flows from Claude. The data does not exist on disk until you
  upgrade + I fetch-and-dump it.
- **Can (done):** built the lookahead-proof panel builder + anti-lookahead
  validator + coverage gate so your dollars convert straight to a real result.

## Re-verified finding (why the purchase is needed)
- `financial-datasets` **free tier returns only `2025-06-25` onward** (confirmed
  live: `get_financial_metrics(AAPL, 2008-2010)` → *"Your plan includes data from
  2025-06-25 onwards. Upgrade to Pro for full historical coverage."*).
- The 38-name universe needs **2006-01-03 → 2026-05-29** quarterly fundamentals.
  Free coverage (1 quarter) cannot evaluate the sleeve.

## Recommended purchase (pick one)
**① financial-datasets Pro — RECOMMENDED.**
- *Why:* records carry `accession_number` + `filing_url` (the real SEC EDGAR
  filing) → **PIT-traceable**; already wired into this session's MCP (no new client,
  no API key in the codebase for the offline eval); has the fields we need
  (`gross_margin`, `net_margin`, `debt_to_equity`, …).
- *Cost:* verify current Pro tier price at financial-datasets before subscribing
  (their pricing changes; I will not assert a number I haven't confirmed).

**② Sharadar Core US Fundamentals (SF1) via Nasdaq Data Link — alternative.**
- *Why:* explicitly as-reported PIT (`datekey` = filing date), survivorship-bias-
  free, deep history. *Trade-off:* needs a separate adapter; not MCP-integrated.
- The panel builder is **vendor-agnostic** — it consumes a normalized
  `{ticker: [record,…]}` dump, so swapping vendors only changes the dump producer.

> Decision is yours (it's your money). I'm proceeding on **financial-datasets Pro**
> because it's PIT-traceable AND already integrated — say so if you'd rather Sharadar.

## Exact data spec
- **Tickers (38):** AAPL AMZN BA BAC BRK-B C CAT COST CSCO CVX DIS F GE GOOGL GS HD
  IBM INTC JNJ JPM KO MCD MMM MRK MS MSFT NKE NVDA ORCL PEP PFE PG T UNH VZ WFC WMT XOM
- **Window:** 2006-01-03 → 2026-05-29, **quarterly**.
- **Fields (minimum):** `report_period`, `gross_margin`, `net_margin`,
  `debt_to_equity` (+ `accession_number`/`filing_url` if available, for tighter PIT).

## PIT discipline (the anti-lie mechanism)
- Every record is lagged to **`report_period + 90 days`** (conservative public-
  availability date — clears even late large-cap filers). A value influences a
  weight only STRICTLY AFTER it was knowable; forward-filled until the next filing.
- `validate_panel_pit` **independently re-derives** this and **raises** on any
  cell scored before its filing was public. `panel_coverage < 60%` → refuse with
  `NOT_EVALUATED` rather than score a thin/biased panel.

## Pre-registered test (fixed BEFORE any quality number exists — anti-p-hacking)
- **Score Q** = mean of available cross-sectional z-scores of
  `{+gross_margin, +net_margin, −debt_to_equity}`, `tilt=1.0`.
- **Comparison:** quality-tilt book vs baseline EW, full-sample **and** OOS tail,
  SAME ship gate, plus the disjoint-5y sub-period stability bar the trend sleeve
  used. A win must persist across regimes — a one-regime win is dredging, not edge.

## Runbook (once Pro is live)
1. Operator upgrades the financial-datasets plan (the purchase).
2. I fetch the 38 tickers' quarterly metrics via MCP and write the dump to
   `market_data/equity/quality_metrics.json` (`{ticker: [record,…]}`).
3. `run_quality_bakeoff()` (`src/equity/quality_data.py`) →
   load → build PIT panel → validate no-lookahead → coverage gate →
   `variant_eval.run_bakeoff(quality_panel=…)` → writes the 4-arm verdict.
4. **Separate verifier** independently re-derives the result + multiple-testing
   audit before any claim of a quality edge. Negative reported honestly = success.
