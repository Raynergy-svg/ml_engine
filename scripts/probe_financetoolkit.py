#!/usr/bin/env python3
"""
Probe: Is FinanceToolkit a viable FREE source of multi-year fundamental
equity data for value/quality/PEAD factor backtests?

Tests the genuinely-free YahooFinance backend (no FMP paid key) for:
  income / balance / cashflow statements, ratios (P/E, P/B, ROE, margins, FCF),
  and earnings/EPS — with history depth and period/report dates checked.

Run:  python scripts/probe_financetoolkit.py
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

TICKERS = ["AAPL", "JPM", "XOM"]


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def safe_span(df) -> str:
    """Return 'N cols (first..last)' for a statement DataFrame (cols = periods)."""
    try:
        if df is None or getattr(df, "empty", True):
            return "EMPTY"
        cols = list(df.columns)
        return f"{len(cols)} periods ({cols[0]} .. {cols[-1]})"
    except Exception as e:  # noqa: BLE001 - probe script, surface everything
        return f"ERR: {e!r}"


def main() -> int:
    import financetoolkit
    from financetoolkit import Toolkit

    ver = getattr(financetoolkit, "__version__", "n/a (read via pip show)")
    section("ENVIRONMENT")
    print(f"financetoolkit version (attr): {ver}")
    print("Pinned in this probe: expecting 2.1.0 (pip show financetoolkit)")
    print(f"Backend under test: YahooFinance (NO FMP api_key supplied)")

    # ---- Construct with the FREE path: no api_key, enforce YahooFinance ------
    section("TOOLKIT CONSTRUCTION (free path)")
    try:
        tk = Toolkit(
            tickers=TICKERS,
            api_key="",                     # explicitly NO paid key
            enforce_source="YahooFinance",  # force the free backend
            start_date="2004-01-01",        # ask for 20+ yrs to probe depth
            progress_bar=False,
        )
        print("Constructed Toolkit(enforce_source='YahooFinance', api_key='') OK")
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: construction failed: {e!r}")
        return 1

    results: dict[str, str] = {}

    # ---- Financial statements (annual) --------------------------------------
    section("FINANCIAL STATEMENTS (annual, free YahooFinance backend)")
    statements = {}
    for name, fn in [
        ("income", "get_income_statement"),
        ("balance", "get_balance_sheet_statement"),
        ("cashflow", "get_cash_flow_statement"),
    ]:
        try:
            df = getattr(tk, fn)()
            statements[name] = df
            # multi-index (ticker, line-item) x periods, OR per-ticker
            print(f"\n[{name}] via tk.{fn}()")
            print(f"  shape={getattr(df, 'shape', None)}  span={safe_span(df)}")
            # show a few line-item names to prove field availability
            try:
                idx = df.index
                if hasattr(idx, "get_level_values"):
                    items = list(dict.fromkeys(idx.get_level_values(-1)))[:8]
                else:
                    items = list(idx)[:8]
                print(f"  sample line-items: {items}")
            except Exception as e:  # noqa: BLE001
                print(f"  (could not list line-items: {e!r})")
            results[f"stmt_{name}"] = safe_span(df)
        except Exception as e:  # noqa: BLE001
            print(f"\n[{name}] FAILED: {e!r}")
            results[f"stmt_{name}"] = f"ERR: {e!r}"

    # ---- Ratios (value + quality) -------------------------------------------
    section("RATIOS — value + quality (free backend)")
    ratio_probe = {}
    try:
        ratios = tk.ratios  # Ratios controller
        print(f"tk.ratios controller: {type(ratios).__name__}")
        ratio_methods = {
            "P/E (get_price_to_earnings_ratio)": "get_price_to_earnings_ratio",
            "P/B (get_price_to_book_ratio)": "get_price_to_book_ratio",
            "EV/EBITDA (get_ev_to_ebitda_ratio)": "get_ev_to_ebitda_ratio",
            "ROE (get_return_on_equity)": "get_return_on_equity",
            "Gross margin (get_gross_margin)": "get_gross_margin",
            "Operating margin (get_operating_margin)": "get_operating_margin",
            "Debt/Equity (get_debt_to_equity_ratio)": "get_debt_to_equity_ratio",
            "EPS (get_earnings_per_share)": "get_earnings_per_share",
            "FCF yield (get_free_cash_flow_yield)": "get_free_cash_flow_yield",
            "Valuation set (collect_valuation_ratios)":
                "collect_valuation_ratios",
            "Profitability set (collect_profitability_ratios)":
                "collect_profitability_ratios",
        }
        for label, meth in ratio_methods.items():
            try:
                if not hasattr(ratios, meth):
                    print(f"  [{label}] method MISSING: ratios.{meth}")
                    ratio_probe[label] = "MISSING METHOD"
                    continue
                df = getattr(ratios, meth)()
                span = safe_span(df)
                print(f"  [{label}] ratios.{meth}() -> {span}")
                ratio_probe[label] = span
            except Exception as e:  # noqa: BLE001
                print(f"  [{label}] ratios.{meth}() FAILED: {e!r}")
                ratio_probe[label] = f"ERR: {e!r}"
    except Exception as e:  # noqa: BLE001
        print(f"tk.ratios controller FAILED: {e!r}")
        ratio_probe["controller"] = f"ERR: {e!r}"

    # ---- Earnings calendar (the real PEAD / point-in-time endpoint) ---------
    section("EARNINGS CALENDAR — announce/report dates (PEAD PIT)")
    earn_cal_status = "UNKNOWN"
    try:
        cal = tk.get_earnings_calendar()
        print(f"  tk.get_earnings_calendar() -> shape={getattr(cal,'shape',None)}")
        if cal is not None and not getattr(cal, "empty", True):
            # show columns + a few rows to see if actual dates are present
            print(f"  columns: {list(cal.columns)[:10]}")
            try:
                print(cal.head(6).to_string()[:1200])
            except Exception:  # noqa: BLE001
                pass
            earn_cal_status = f"PRESENT shape={cal.shape}"
        else:
            earn_cal_status = "EMPTY on free backend"
        print(f"  earnings-calendar status: {earn_cal_status}")
    except Exception as e:  # noqa: BLE001
        print(f"  get_earnings_calendar() FAILED: {e!r}")
        earn_cal_status = f"ERR: {e!r}"

    # ---- Earnings / EPS (PEAD inputs) ---------------------------------------
    section("EARNINGS / EPS (PEAD inputs, free backend)")
    eps_status = "UNKNOWN"
    # EPS usually lives on the income statement; surprise/announce dates are the
    # real PEAD requirement.
    try:
        inc = statements.get("income")
        eps_found = False
        if inc is not None and not inc.empty:
            idx = inc.index
            items = (list(idx.get_level_values(-1))
                     if hasattr(idx, "get_level_values") else list(idx))
            eps_rows = [i for i in items if "EPS" in str(i).upper()
                        or "Earnings Per Share" in str(i)]
            if eps_rows:
                eps_found = True
                print(f"  EPS line-items in income statement: {eps_rows[:5]}")
        # Dedicated earnings-calendar / surprise endpoints?
        cal_methods = [m for m in dir(tk)
                       if "earning" in m.lower() or "surprise" in m.lower()]
        print(f"  Toolkit earnings-related methods: {cal_methods}")
        eps_status = ("EPS in statements: YES" if eps_found
                      else "EPS in statements: NO")
        print(f"  {eps_status}")
        print("  NOTE: announce/report DATE for PEAD = key gate-4 question")
    except Exception as e:  # noqa: BLE001
        print(f"  earnings probe FAILED: {e!r}")
        eps_status = f"ERR: {e!r}"

    # ---- Point-in-time: are period/report dates exposed? --------------------
    section("POINT-IN-TIME — period/report dates")
    pit_note = "columns ARE periods (fiscal year/quarter) — see span above"
    print(f"  Statement columns are the period labels: {pit_note}")
    inc = statements.get("income")
    if inc is not None and not inc.empty:
        print(f"  income columns (period labels): {list(inc.columns)}")
    print("  CAVEAT: Yahoo statements give FISCAL PERIOD END, not the actual")
    print("  FILING/ANNOUNCE date. True PIT discipline needs the report lag")
    print("  (typically +45-90d for annual, +30-45d for quarterly) applied")
    print("  manually, OR a dedicated earnings-date endpoint.")

    # ---- SUMMARY ------------------------------------------------------------
    section("SUMMARY (free YahooFinance backend, no paid key)")
    print("Statements:")
    for k, v in results.items():
        print(f"  {k:16s}: {v}")
    print("Ratios:")
    for k, v in ratio_probe.items():
        print(f"  {k}: {v}")
    print(f"Earnings/EPS: {eps_status}")
    print(f"Earnings calendar (PIT dates): {earn_cal_status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
