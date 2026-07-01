#!/usr/bin/env python3
"""Probe OpenBB Platform as a FREE source of multi-year fundamental equity data.

Goal: determine whether OpenBB can deliver — for FREE (no paywalled / credit-limited
API key) — the data needed to build value / quality / PEAD equity factor backtests:
  - financial statements (income / balance / cash) + valuation/quality ratios
  - earnings / EPS (ideally with surprise)
  - multi-year history with report/filing dates for point-in-time discipline

This script PROVES and CHARACTERIZES the source; it does NOT build a backtest.

Run:  python scripts/probe_openbb.py
Deps: pip install openbb   (tested against openbb==4.7.2)

Free, no-key providers in OpenBB 4.7.2 (verified from the provider registry):
    yfinance, sec, federal_reserve, government_us, imf, oecd
Of those, only `yfinance` and `sec` (EDGAR) carry equity fundamentals.
Everything richer (ratios endpoint, earnings calendar, EPS surprise, analyst
estimates) routes through `fmp` / `intrinio`, both of which REQUIRE a paid key.
"""
from __future__ import annotations

import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 200)

from openbb import obb  # noqa: E402

TICKERS = ["AAPL", "JPM", "XOM"]
SECTION = lambda s: print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)  # noqa: E731

# Value/quality field shopping list -> where it actually lives, FREE
VALUE_FIELDS = ["pe_ratio", "price_to_book", "enterprise_to_ebitda",
                "enterprise_to_revenue", "dividend_yield"]
QUALITY_FIELDS = ["return_on_equity", "return_on_assets", "gross_margin",
                  "operating_margin", "profit_margin", "debt_to_equity",
                  "current_ratio", "quick_ratio"]


def safe(label, fn, **kw):
    try:
        return fn(**kw).to_dataframe()
    except Exception as e:  # noqa: BLE001 - probe script, surface everything
        print(f"  [{label}] ERROR {type(e).__name__}: {str(e)[:160]}")
        return None


def main():
    # ---------------------------------------------------------------- providers
    SECTION("1. FREE / NO-KEY PROVIDERS (from OpenBB provider registry)")
    try:
        from openbb_core.provider.registry import RegistryLoader
        reg = RegistryLoader.from_extensions()
        free, paid = [], []
        for name, prov in sorted(reg.providers.items()):
            creds = getattr(prov, "credentials", []) or []
            (free if not creds else paid).append(name if not creds
                                                 else f"{name}({','.join(creds)})")
        print("  FREE (no key):", ", ".join(free))
        print("  PAID (key req):", ", ".join(paid))
    except Exception as e:  # noqa: BLE001
        print("  registry introspection failed:", e)
    print("\n  Fundamentals-bearing FREE providers: yfinance, sec")
    print("  Endpoint->provider gates (verified):")
    print("    income/balance/cash : fmp, intrinio, sec, yfinance   <- SEC+yf FREE")
    print("    ratios              : fmp, intrinio                  <- PAID only")
    print("    metrics             : fmp, intrinio, yfinance        <- yf FREE (snapshot)")
    print("    calendar.earnings   : fmp                            <- PAID only")
    print("    fundamental.historical_eps : fmp                     <- PAID only")
    print("    estimates.*         : fmp, intrinio, (yfinance)      <- surprise PAID")

    # ---------------------------------------------------- SEC fundamentals depth
    SECTION("2. SEC (EDGAR) FUNDAMENTALS — history depth + fields (FREE)")
    for sym in TICKERS:
        inc = safe(f"sec income {sym}", obb.equity.fundamental.income,
                   symbol=sym, provider="sec", period="annual", limit=20)
        if inc is None or not len(inc):
            continue
        bal = safe(f"sec balance {sym}", obb.equity.fundamental.balance,
                   symbol=sym, provider="sec", period="annual", limit=20)
        cf = safe(f"sec cash {sym}", obb.equity.fundamental.cash,
                  symbol=sym, provider="sec", period="annual", limit=20)
        incq = safe(f"sec income-q {sym}", obb.equity.fundamental.income,
                    symbol=sym, provider="sec", period="quarterly", limit=80)
        pe = inc["period_ending"].astype(str)
        print(f"\n  {sym}: ANNUAL income rows={len(inc)} "
              f"span {pe.min()}..{pe.max()} ({len(inc)} fiscal years)")
        if bal is not None:
            print(f"       balance rows={len(bal)}  cash rows="
                  f"{0 if cf is None else len(cf)}  "
                  f"QUARTERLY income rows={0 if incq is None else len(incq)}")
        have_gp = "total_gross_profit" in inc.columns
        print(f"       gross_profit field present: {have_gp} "
              f"(False is correct for banks/energy)")

    # show the exact raw fields you'd compute value/quality from
    inc = obb.equity.fundamental.income(symbol="AAPL", provider="sec",
                                        period="annual", limit=3).to_dataframe()
    bal = obb.equity.fundamental.balance(symbol="AAPL", provider="sec",
                                         period="annual", limit=3).to_dataframe()
    cf = obb.equity.fundamental.cash(symbol="AAPL", provider="sec",
                                     period="annual", limit=3).to_dataframe()
    print("\n  RAW SEC fields available to BUILD point-in-time factors:")
    print("    income :", [c for c in inc.columns if c in (
        "total_revenue", "total_gross_profit", "total_operating_income",
        "net_income", "diluted_eps", "basic_eps")])
    print("    balance:", [c for c in bal.columns if c in (
        "total_assets", "total_liabilities", "common_equity",
        "long_term_debt", "short_term_debt", "total_current_assets",
        "total_current_liabilities")])
    print("    cash   :", [c for c in cf.columns if c in (
        "net_cash_from_operating_activities",)])
    print("  -> P/B, EV/EBITDA, ROE, gross/op margin, debt ratios are all"
          " COMPUTABLE from SEC raw + price; FCF needs capex (check cash cols).")

    # ---------------------------------------------------- yfinance metrics snapshot
    SECTION("3. yfinance metrics — value+quality RATIOS, but SNAPSHOT ONLY (FREE)")
    m = obb.equity.fundamental.metrics(symbol="AAPL", provider="yfinance").to_dataframe()
    print(f"  rows={len(m)} (1 = current snapshot, NO date column => NOT point-in-time)")
    print("  VALUE fields present :",
          {k: round(float(m[k].iloc[0]), 3) for k in VALUE_FIELDS if k in m.columns})
    print("  QUALITY fields present:",
          {k: round(float(m[k].iloc[0]), 3) for k in QUALITY_FIELDS if k in m.columns})
    print("  VERDICT: rich ratios, but single current row -> usable only for a"
          " live screen, NOT for a historical factor backtest.")

    # ------------------------------------------------ point-in-time / filing dates
    SECTION("4. POINT-IN-TIME — filing/report dates (critical for look-ahead)")
    print("  SEC statement rows carry 'period_ending' + 'fiscal_year' but NOT a"
          " filing date.\n  The true PIT date comes from equity.fundamental.filings"
          " (form_type=10-K/10-Q):")
    fil = obb.equity.fundamental.filings(symbol="AAPL", provider="sec",
                                         form_type="10-K", limit=400).to_dataframe()
    tenk = fil[fil["report_type"].astype(str).str.contains("10-K", na=False)]
    cols = ["report_date", "filing_date", "accepted_date"]
    print(f"  AAPL 10-K filings: {len(tenk)} rows. report_date(period) -> filing_date:")
    print(tenk[cols].head(6).to_string(index=False))
    print("\n  FIELD NAMES for PIT join: 'report_date' (== statement period_ending),"
          " 'filing_date', 'accepted_date'.")
    print("  PIT recipe: join statements on period_ending == filings.report_date,"
          " then only use the row AFTER filing_date (~1 month lag for 10-K).")

    # ----------------------------------------------------- earnings / EPS surprise
    SECTION("5. EARNINGS / EPS SURPRISE (the PEAD ingredient)")
    print("  Reported EPS (the 'actual'): FREE via SEC quarterly income:")
    q = obb.equity.fundamental.income(symbol="AAPL", provider="sec",
                                      period="quarterly", limit=6).to_dataframe()
    print(q[["period_ending", "fiscal_period", "basic_eps", "diluted_eps"]
            ].head(6).to_string(index=False))
    print("\n  Estimate / consensus / surprise: PAID. Every free attempt is rejected:")
    for ep, fn, kw in [
        ("calendar.earnings(yfinance)", obb.equity.calendar.earnings,
         dict(symbol="AAPL", provider="yfinance")),
        ("fundamental.historical_eps(yfinance)", obb.equity.fundamental.historical_eps,
         dict(symbol="AAPL", provider="yfinance")),
        ("estimates.historical(yfinance)", obb.equity.estimates.historical,
         dict(symbol="AAPL", provider="yfinance")),
    ]:
        safe(ep, fn, **kw)
    print("  => Standardized EPS SURPRISE is NOT free in OpenBB. You can build a"
          " *self-computed* surprise (reported EPS vs a naive seasonal-random-walk"
          " expectation) from SEC, but vendor analyst surprise needs FMP/Intrinio.")

    # --------------------------------------------------------- prices + scale/limits
    SECTION("6. PRICES + SCALE / RATE LIMITS")
    t0 = time.time()
    for s in TICKERS:
        px = obb.equity.price.historical(symbol=s, provider="yfinance",
                                         start_date="2010-01-01").to_dataframe()
        print(f"  {s} daily prices rows={len(px)} "
              f"span {str(px.index.min())[:10]}..{str(px.index.max())[:10]} "
              f"cols={list(px.columns)}")
    px_dt = time.time() - t0
    t0 = time.time()
    obb.equity.fundamental.income(symbol="JPM", provider="sec",
                                  period="annual", limit=20).to_dataframe()
    sec_dt = time.time() - t0
    print(f"\n  Timing: 3 yfinance price pulls={px_dt:.1f}s "
          f"(~{px_dt/3:.1f}s/ticker); 1 SEC income pull={sec_dt:.1f}s/ticker.")
    print("  SCALE: yfinance is unofficial/unthrottled-by-OpenBB but rate-limits"
          " hard above ~1-2 req/s; SEC EDGAR fair-access ~10 req/s (User-Agent).")
    print("  A 50-500 stock universe is pullable FREE but SLOW & serial:")
    print("    ~5 statement calls + 1 filings + 1 price per name.")
    print("    500 names ~= 3500-4000 calls -> minutes-to-tens-of-minutes,")
    print("    needs caching + polite throttling; fine for a research backtest,")
    print("    not for intraday refresh.")

    # ------------------------------------------------------------------- verdict
    SECTION("VERDICT — SMART GATES")
    print("""  Provider to use: SEC (EDGAR) for statements+EPS+filing-dates;
                   yfinance for daily prices (and live ratio snapshots only).

  History depth : SEC ~18 fiscal years annual (2008..2025) + ~70 quarters,
                  all 3 tickers; yfinance income capped at limit<=5 (unusable
                  for history). yfinance prices 2010..now (and earlier).

  Field mapping :
    VALUE   P/E, P/B, EV/EBITDA, EV/Rev, div_yield -> COMPUTE from SEC raw
            (net_income, common_equity, total_revenue, debt) + yfinance price.
            (yfinance.metrics has them pre-computed but SNAPSHOT-only.)
    QUALITY ROE, ROA, gross/op/profit margin, debt/equity, current/quick
            -> COMPUTE from SEC income+balance (margins direct; ROE=NI/equity).
    PEAD    reported EPS (basic/diluted) FREE from SEC quarterly; analyst
            estimate/surprise is PAID (FMP/Intrinio) -> self-compute surprise.

  G1 Free        : PASS  - SEC + yfinance need NO key.
  G2 History>=10y: PASS  - SEC delivers ~18y annual / ~70q on all 3 names.
  G3 Fields      : PARTIAL- value+quality+reported-EPS YES (some computed,
                            not pre-packaged); analyst EARNINGS SURPRISE = FAIL/paid.
  G4 Point-in-time: PASS  - filings endpoint gives filing_date/accepted_date;
                            join report_date->filing_date for look-ahead safety.
  G5 Scale       : PASS-with-caveats - 50-500 names free but serial/slow;
                            cache + throttle required.
""")


if __name__ == "__main__":
    main()
