"""FREE true-point-in-time fundamentals via SEC EDGAR (offline; running:NO).

The operator corrected the data plan: NO paid data. This module sources quality
fundamentals from the SEC EDGAR XBRL ``companyfacts`` API — free, official, and
crucially it preserves the real **filing date** (`filed`) of every fact. Factors
are aligned on the FILING DATE (genuine PIT), never on fiscal period-end. That is
the whole point: a 10-K for FY ending 2014-12-31 is not public until it is filed
~weeks later, so it can only influence a weight from its `filed` date forward.

HONESTY LABELLING (mandatory, L-018):
  - Every emitted fundamental is tagged ``source="EDGAR-XBRL"`` + ``pit=True`` and
    carries its real ``filed`` date. These are true-PIT.
  - yfinance is used for PRICES ONLY elsewhere — NEVER for fundamentals. A yfinance
    snapshot ratio applied historically is lookahead and would manufacture fake
    alpha; this module does not touch it.
  - "As-originally-reported": for each fiscal period we keep the EARLIEST-filed
    observation (later filings carry restated comparatives — using them = lookahead).
  - A factor that cannot be computed PIT-free for a name/year is left NaN; the
    downstream coverage gate reports NOT_EVALUATED rather than fabricating.

KNOWN LIMITATION (labelled, not hidden): EDGAR XBRL structured data begins ~2009
(the XBRL mandate phase-in), so the PIT-clean factor history floor is ~2010. Annual
(10-K) basis: each year's factors update on the 10-K filing date and are held until
the next 10-K (standard quality-factor construction).

OFFLINE only — network fetch (EDGAR) + JSON dump to disk; nothing runs in a
process, trades, unhalts, or touches the hot path.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

# SEC requires a descriptive UA with contact info; 10 req/s ceiling.
SEC_UA = "ml_engine-research dcertan84@gmail.com"
_REQUEST_SPACING_S = 0.13          # stay comfortably under 10 req/s
_TIMEOUT = (5, 30)                  # (connect, read)
CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Concept priority lists (XBRL us-gaap tags vary by filer/era).
_REVENUE_CONCEPTS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
_GROSS_PROFIT = ("GrossProfit",)
_COST_CONCEPTS = (
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "CostOfGoodsSold",
)
_NET_INCOME = ("NetIncomeLoss",)
_LIABILITIES = ("Liabilities",)
_EQUITY = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)


def _get_json(session: requests.Session, url: str, *, retries: int = 4) -> Optional[dict]:
    """GET with UA, explicit timeouts, exponential backoff on 5xx/timeout/429.

    Returns parsed JSON, or None on 404 (concept/company not present) / exhaustion.
    Never retries 4xx other than 429 (per the project retry gate).
    """
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = session.get(url, headers={"User-Agent": SEC_UA}, timeout=_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as exc:
            logger.warning("EDGAR transient error %s on %s (attempt %d)", exc, url, attempt + 1)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue
        if resp.status_code == 404:
            return None
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue
        if resp.status_code != 200:
            logger.warning("EDGAR %d on %s", resp.status_code, url)
            return None
        try:
            return resp.json()
        except ValueError:
            return None
    return None


def load_cik_map(session: Optional[requests.Session] = None) -> Dict[str, str]:
    """Map ticker -> zero-padded 10-digit CIK from SEC's official registry."""
    sess = session or requests.Session()
    data = _get_json(sess, CIK_MAP_URL)
    if not data:
        return {}
    out: Dict[str, str] = {}
    for row in data.values():
        tkr = str(row.get("ticker", "")).upper()
        cik = row.get("cik_str")
        if tkr and cik is not None:
            out[tkr] = f"{int(cik):010d}"
    return out


def _earliest_filed(obs_list: Sequence[dict], *, annual: bool, flow: bool) -> Dict[str, Tuple[str, float]]:
    """Reduce raw XBRL observations to {period_end: (earliest_filed, as_reported_val)}.

    annual: keep only 10-K filings (form startswith '10-K'). flow concepts must span
    a full year (end-start > 300d); stock concepts are instantaneous (no 'start').
    For each period_end keep the EARLIEST-filed observation (as-originally-reported).
    """
    from datetime import date

    best: Dict[str, Tuple[str, float]] = {}
    for o in obs_list:
        form = str(o.get("form", ""))
        if annual and not form.startswith("10-K"):
            continue
        end = o.get("end")
        filed = o.get("filed")
        val = o.get("val")
        if end is None or filed is None or not isinstance(val, (int, float)):
            continue
        if flow:
            start = o.get("start")
            if start is None:
                continue
            try:
                span = (date.fromisoformat(end) - date.fromisoformat(start)).days
            except ValueError:
                continue
            if span <= 300:          # quarterly/YTD partial — skip, want full-year flow
                continue
        else:
            if o.get("start") is not None:   # stock concept must be instantaneous
                continue
            if str(o.get("fp", "")) != "FY":
                continue
        prev = best.get(end)
        if prev is None or filed < prev[0]:
            best[end] = (filed, float(val))
    return best


def _merge_concepts(usgaap: dict, names: Sequence[str], *, flow: bool) -> Dict[str, Tuple[str, float]]:
    """MERGE all synonym tags into one PIT annual series — for each period end take
    the EARLIEST-filed value across ANY synonym. (Filers switch tags across eras,
    e.g. SalesRevenueNet -> RevenueFromContractWithCustomer..., so picking only the
    first non-empty tag silently drops years and biases coverage.)"""
    merged: Dict[str, Tuple[str, float]] = {}
    for name in names:
        node = usgaap.get(name)
        if not node:
            continue
        units = node.get("units", {}).get("USD")
        if not units:
            continue
        for end, (filed, val) in _earliest_filed(units, annual=True, flow=flow).items():
            prev = merged.get(end)
            if prev is None or filed < prev[0]:
                merged[end] = (filed, val)
    return merged


def quality_records_from_facts(facts: dict) -> List[dict]:
    """Build PIT annual quality records from one company's companyfacts payload.

    Each record: {report_period (FY end), filed (PIT date = latest input filing for
    that year), source, pit, gross_margin?, net_margin?, debt_to_equity?}. A ratio
    is NaN when an input is missing; its availability is the LATEST filing among its
    inputs (a ratio is knowable only once every input is public — no lookahead).
    """
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    rev = _merge_concepts(usgaap, _REVENUE_CONCEPTS, flow=True)
    gp = _merge_concepts(usgaap, _GROSS_PROFIT, flow=True)
    cost = _merge_concepts(usgaap, _COST_CONCEPTS, flow=True)
    ni = _merge_concepts(usgaap, _NET_INCOME, flow=True)
    liab = _merge_concepts(usgaap, _LIABILITIES, flow=False)
    eq = _merge_concepts(usgaap, _EQUITY, flow=False)

    # Gross profit: direct GrossProfit tag, else derive Revenues - CostOfRevenue
    # (most filers report cost, not a GrossProfit line). Availability = later filing.
    gp_eff: Dict[str, Tuple[str, float]] = dict(gp)
    for end, (rf, rv) in rev.items():
        if end not in gp_eff and end in cost:
            cf, cv = cost[end]
            gp_eff[end] = (max(rf, cf), rv - cv)

    ends = set(rev) | set(gp_eff) | set(ni) | set(liab) | set(eq)
    records: List[dict] = []
    for end in sorted(ends):
        comp: Dict[str, float] = {}
        fileds: List[str] = []

        def _ratio(numkey, num, den, label):
            if end in num and end in den:
                nf, nv = num[end]
                df, dv = den[end]
                if dv not in (0, 0.0):
                    comp[label] = nv / dv
                    fileds.extend([nf, df])

        _ratio("gm", gp_eff, rev, "gross_margin")
        _ratio("nm", ni, rev, "net_margin")
        _ratio("de", liab, eq, "debt_to_equity")
        if not comp or not fileds:
            continue
        records.append({
            "report_period": end,
            "filed": max(fileds),         # PIT: knowable only when ALL inputs public
            "source": "EDGAR-XBRL",
            "pit": True,
            **comp,
        })
    return records


def build_pit_dump(
    tickers: Sequence[str],
    *,
    out_path: Optional[Path] = None,
    session: Optional[requests.Session] = None,
) -> Dict[str, List[dict]]:
    """Fetch EDGAR companyfacts for each ticker -> PIT annual quality records.

    Returns {ticker: [record,...]} and (optionally) writes it atomically. Tickers
    without a CIK or without usable XBRL fundamentals are simply absent (downstream
    treats absence as NaN/NOT_EVALUATED — never fabricated). One request per company.
    """
    sess = session or requests.Session()
    cik_map = load_cik_map(sess)
    dump: Dict[str, List[dict]] = {}
    missing_cik = 0
    for i, tkr in enumerate(tickers):
        cik = cik_map.get(str(tkr).upper().replace("-", "."))  # BRK-B XBRL uses BRK.B? try raw too
        if cik is None:
            cik = cik_map.get(str(tkr).upper())
        if cik is None:
            missing_cik += 1
            continue
        time.sleep(_REQUEST_SPACING_S)
        facts = _get_json(sess, COMPANYFACTS_URL.format(cik=cik))
        if not facts:
            continue
        recs = quality_records_from_facts(facts)
        if recs:
            dump[str(tkr)] = recs
        if (i + 1) % 50 == 0:
            logger.info("EDGAR fundamentals: %d/%d tickers processed", i + 1, len(tickers))
    logger.info("EDGAR dump: %d tickers with PIT fundamentals (%d missing CIK)",
                len(dump), missing_cik)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(dump, fh, indent=2, sort_keys=True)
            fh.flush()
        tmp.replace(out_path)
    return dump
