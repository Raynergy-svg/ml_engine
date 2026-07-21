"""Track B (equity research-alpha) — fetch + blind pipeline for a bounded pilot run.

See docs/experiment-equity-research-alpha-prereg-2026-06-30.md. This script does
STEP 1 of the offline pipeline only: fetch PIT 10-K text via
src.equity.research.pit_text_loader, entity-blind it via
src.equity.research.entity_blinder, and write each blinded task to disk as an
ANONYMOUS file (ticker withheld from the file the scorer reads) plus a separate
index the scorer does not open until scoring is complete. No LLM call happens
here — this module is offline/deterministic, matching the frozen design's
"harness never calls a model" discipline extended to this data-prep step.

SCALE-DOWN FROM THE FROZEN UNIVERSE (documented, not hidden): the pre-reg's
frozen universe is "all PIT S&P 500 members, ~2012-01 -> as-of" — for the LLM-
scorer substitution (this session acting as the scorer, since no
ANTHROPIC_API_KEY is configured for an automated fan-out), that is many
thousands of filings, infeasible to hand-score in one sitting. This script
runs a BOUNDED PILOT: a systematically-chosen (not outcome-chosen) 12-name
mega-cap subset, 3 annual 10-Ks per name (FY2023/FY2024/FY2025), fetched BEFORE
any scoring or result is seen. The selection rule and universe are frozen in
this file's PILOT_TICKERS constant before the first fetch call runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

from src.equity.edgar_fundamentals import load_cik_map
from src.equity.research.entity_blinder import IssuerBundle
from src.equity.research.pit_text_loader import load_pit_filing
from src.equity.research.scorer import prepare_scoring_task, scoring_task_to_prompt

# --- FROZEN pilot universe (decided before any fetch; see module docstring) ---
# Rule: 12 of the largest S&P 500 constituents by market cap, spread across
# GICS sectors, restricted to single-CIK straightforward EDGAR filers (excludes
# dual-class/trust structures that complicate blinding). NOT chosen by any
# expected result.
PILOT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
    "JPM", "JNJ", "XOM", "PG", "HD", "UNH",
]

# Anchor as-of dates: ~2 months after the typical accelerated-filer 10-K
# deadline (60 days post-FYE for a Dec-FYE large filer), so each anchor
# reliably resolves to that fiscal year's freshly-filed 10-K.
ANCHORS = ["2024-04-15", "2025-04-15", "2026-04-15"]

# Best-effort public deny-list metadata per issuer (legal name variants, HQ,
# CEO). The blinder's own residual-fingerprint scan catches what this misses;
# per prereg §8 the blinder is a noise-reducer, not the load-bearing control.
ISSUER_META = {
    "AAPL": IssuerBundle(
        company_names=("Apple Inc.", "Apple Computer, Inc.", "Apple"),
        tickers=("AAPL",), cik=None, exchanges=("Nasdaq",),
        person_names=("Tim Cook", "Timothy D. Cook"),
        hq_locations=("Cupertino, California", "Cupertino"),
    ),
    "MSFT": IssuerBundle(
        company_names=("Microsoft Corporation", "Microsoft"),
        tickers=("MSFT",), cik=None, exchanges=("Nasdaq",),
        person_names=("Satya Nadella",),
        hq_locations=("Redmond, Washington", "Redmond"),
    ),
    "GOOGL": IssuerBundle(
        company_names=("Alphabet Inc.", "Google Inc.", "Google LLC", "Alphabet"),
        tickers=("GOOGL", "GOOG"), cik=None, exchanges=("Nasdaq",),
        person_names=("Sundar Pichai",),
        hq_locations=("Mountain View, California", "Mountain View"),
    ),
    "AMZN": IssuerBundle(
        company_names=("Amazon.com, Inc.", "Amazon.com", "Amazon"),
        tickers=("AMZN",), cik=None, exchanges=("Nasdaq",),
        person_names=("Andy Jassy", "Andrew R. Jassy", "Jeff Bezos"),
        hq_locations=("Seattle, Washington", "Seattle"),
    ),
    "NVDA": IssuerBundle(
        company_names=("NVIDIA Corporation", "NVIDIA"),
        tickers=("NVDA",), cik=None, exchanges=("Nasdaq",),
        person_names=("Jensen Huang", "Jen-Hsun Huang"),
        hq_locations=("Santa Clara, California", "Santa Clara"),
    ),
    "META": IssuerBundle(
        company_names=("Meta Platforms, Inc.", "Facebook, Inc.", "Meta"),
        tickers=("META", "FB"), cik=None, exchanges=("Nasdaq",),
        person_names=("Mark Zuckerberg",),
        hq_locations=("Menlo Park, California", "Menlo Park"),
    ),
    "JPM": IssuerBundle(
        company_names=("JPMorgan Chase & Co.", "JPMorgan Chase", "J.P. Morgan"),
        tickers=("JPM",), cik=None, exchanges=("New York Stock Exchange", "NYSE"),
        person_names=("Jamie Dimon", "James Dimon"),
        hq_locations=("New York, New York",),
    ),
    "JNJ": IssuerBundle(
        company_names=("Johnson & Johnson",),
        tickers=("JNJ",), cik=None, exchanges=("New York Stock Exchange", "NYSE"),
        person_names=("Joaquin Duato",),
        hq_locations=("New Brunswick, New Jersey", "New Brunswick"),
    ),
    "XOM": IssuerBundle(
        company_names=("Exxon Mobil Corporation", "ExxonMobil", "Exxon"),
        tickers=("XOM",), cik=None, exchanges=("New York Stock Exchange", "NYSE"),
        person_names=("Darren Woods", "Darren W. Woods"),
        hq_locations=("Spring, Texas", "Irving, Texas"),
    ),
    "PG": IssuerBundle(
        company_names=("The Procter & Gamble Company", "Procter & Gamble", "P&G"),
        tickers=("PG",), cik=None, exchanges=("New York Stock Exchange", "NYSE"),
        person_names=("Jon Moeller", "Jon R. Moeller"),
        hq_locations=("Cincinnati, Ohio", "Cincinnati"),
    ),
    "HD": IssuerBundle(
        company_names=("The Home Depot, Inc.", "Home Depot"),
        tickers=("HD",), cik=None, exchanges=("New York Stock Exchange", "NYSE"),
        person_names=("Ted Decker", "Edward G. Decker"),
        hq_locations=("Atlanta, Georgia", "Atlanta"),
    ),
    "UNH": IssuerBundle(
        company_names=("UnitedHealth Group Incorporated", "UnitedHealth Group"),
        tickers=("UNH",), cik=None, exchanges=("New York Stock Exchange", "NYSE"),
        person_names=("Andrew Witty", "Stephen Hemsley", "Sir Andrew Witty"),
        hq_locations=("Minnetonka, Minnesota", "Minnetonka"),
    ),
}

OUT_DIR = Path("trained_data/research/track_b_pilot_2026_07_01")
TASKS_DIR = OUT_DIR / "blinded_tasks"       # what the scorer reads (no ticker)
CACHE_DIR = OUT_DIR / "edgar_cache"
INDEX_PATH = OUT_DIR / "_index.json"        # ticker/as_of map — NOT read until after scoring
RAW_META_PATH = OUT_DIR / "fetch_log.json"


def main() -> None:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    cik_map = load_cik_map(session=session)
    print(f"CIK map loaded: {len(cik_map)} tickers")

    index: dict = {}
    fetch_log: list = []
    seen_filings: dict = {}  # (ticker, filed) -> task_id, to dedupe identical 10-Ks across anchors
    task_id = 0

    for ticker in PILOT_TICKERS:
        cik = cik_map.get(ticker)
        if not cik:
            fetch_log.append({"ticker": ticker, "status": "NO_CIK"})
            print(f"{ticker}: NO CIK FOUND")
            continue
        bundle = ISSUER_META[ticker]
        bundle = IssuerBundle(
            company_names=bundle.company_names,
            former_company_names=bundle.former_company_names,
            tickers=bundle.tickers,
            cik=cik,
            exchanges=bundle.exchanges,
            person_names=bundle.person_names,
            product_names=bundle.product_names,
            hq_locations=bundle.hq_locations,
        )
        for anchor in ANCHORS:
            filing = load_pit_filing(
                ticker, cik, anchor, forms=("10-K",),
                session=session, cache_dir=CACHE_DIR,
            )
            if filing is None:
                fetch_log.append(
                    {"ticker": ticker, "anchor": anchor, "status": "NO_FILING"}
                )
                print(f"{ticker} @ {anchor}: NO FILING")
                continue
            dedupe_key = (ticker, filing.filed)
            if dedupe_key in seen_filings:
                fetch_log.append({
                    "ticker": ticker, "anchor": anchor, "status": "DUP_OF",
                    "dup_of_task": seen_filings[dedupe_key],
                    "filed": filing.filed,
                })
                print(f"{ticker} @ {anchor}: duplicate of {seen_filings[dedupe_key]} (filed {filing.filed})")
                continue

            # max_chars is NOT a prereg-frozen knob (only weights/quintile/cadence/
            # vol-target/costs are frozen, §3 "Frozen knobs"). The scorer.py default
            # (12000) was found to be consumed almost entirely by 10-K cover-page +
            # table-of-contents boilerplate, leaving near-zero real Business/Risk
            # Factors/MD&A prose to score. Raised BEFORE any scoring/result was seen.
            task = prepare_scoring_task(filing, bundle, blind=True, max_chars=45000)
            prompt = scoring_task_to_prompt(task)

            tid = f"task_{task_id:03d}"
            (TASKS_DIR / f"{tid}.txt").write_text(prompt, encoding="utf-8")
            index[tid] = {
                "ticker": ticker,
                "as_of": filing.filed,   # true PIT date: the actual SEC filing date
                "form": filing.form,
                "query_anchor": anchor,
                "audit": task.audit,
                "truncated": task.truncated,
                "original_chars": task.original_chars,
                "scored_chars": task.scored_chars,
            }
            seen_filings[dedupe_key] = tid
            fetch_log.append({
                "ticker": ticker, "anchor": anchor, "status": "OK", "task_id": tid,
                "filed": filing.filed, "form": filing.form,
            })
            print(f"{ticker} @ {anchor}: {tid} (filed {filing.filed}, {task.scored_chars} chars)")
            task_id += 1

    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    RAW_META_PATH.write_text(json.dumps(fetch_log, indent=2), encoding="utf-8")
    print(f"\n{task_id} distinct tasks written to {TASKS_DIR}")
    print(f"Index (ticker/as_of mapping, NOT to be read before scoring): {INDEX_PATH}")


if __name__ == "__main__":
    main()
