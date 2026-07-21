"""Track B (equity research-alpha) — SCALE-UP fetch+blind, strictly post-cutoff.

See docs/experiment-equity-research-alpha-prereg-2026-06-30.md and the 2026-07-01
pilot (§7 of that doc). This script replaces the pilot's 12-name bounded universe
with the broadest reachable S&P 500 universe, restricted to filings that are
STRICTLY POST-CUTOFF: ``filed >= MODEL_CUTOFF``. This is a materially stronger
lookahead control than the pilot's full/pre/post split — every filing fetched
here is one the research model's pretraining literally cannot contain, because
it did not exist as of the model's stated knowledge cutoff.

Universe construction (frozen BEFORE any fetch, matching prereg §5's "no
variants" discipline extended to this scale-up):
  1. Current S&P 500 constituents from Wikipedia (src.equity.sp500_membership),
     503 tickers as of fetch time (2026-07-02) — the broadest free, PIT-plausible
     membership source available without a paid feed.
  2. For each ticker: CIK via SEC's company_tickers.json (load_cik_map), most
     recent 10-Q/10-K with ``filed <= AS_OF`` via the existing PIT loader
     (src.equity.research.pit_text_loader.load_pit_filing).
  3. KEEP iff ``filed >= MODEL_CUTOFF``. A name with no qualifying filing in the
     post-cutoff window (delisted, off-cycle fiscal year, EDGAR gap) is DROPPED
     (logged, never lookahead, never zero-filled) — mirrors the inference-
     contract abstention rule used everywhere else in this repo.
  4. Entity-blinding uses a MINIMAL auto-derived IssuerBundle (SEC legal name +
     ticker + CIK only — no hand-curated CEO/HQ/product denylist, which does not
     scale to ~500 names by hand). Per prereg §8 (independently confirmed by two
     reviewers on the pilot): the blinder is a noise-reducer, NOT the load-bearing
     control at this design's actual scale — the load-bearing control is the
     post-cutoff filter in step 3, which every fetched filing here already
     satisfies by construction. A thinner denylist only means residual leaks (if
     any) are caught by the audit as informational, not as a validity threat.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from src.equity.edgar_fundamentals import load_cik_map
from src.equity.research.entity_blinder import IssuerBundle
from src.equity.research.pit_text_loader import load_pit_filing
from src.equity.research.scorer import prepare_scoring_task, scoring_task_to_prompt
from src.equity.sp500_membership import _fetch_wiki_tables, _parse_current

MODEL_CUTOFF = "2026-02-01"   # Sonnet 5 stated knowledge cutoff = Jan 2026
AS_OF = "2026-07-01"          # fetch ceiling; strictly <= today (2026-07-02)

OUT_DIR = Path("trained_data/research/track_b_postcutoff_2026_07_02")
TASKS_DIR = OUT_DIR / "blinded_tasks"
CACHE_DIR = OUT_DIR / "edgar_cache"
INDEX_PATH = OUT_DIR / "_index.json"
RAW_META_PATH = OUT_DIR / "fetch_log.json"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_UA = "ml_engine-research dcertan84@gmail.com"


def _load_company_titles(session: requests.Session) -> dict:
    """ticker -> SEC legal name ('title'), from the same file load_cik_map reads."""
    resp = session.get(COMPANY_TICKERS_URL, headers={"User-Agent": SEC_UA}, timeout=(5, 30))
    resp.raise_for_status()
    data = resp.json()
    out = {}
    for row in data.values():
        tkr = str(row.get("ticker", "")).upper()
        title = row.get("title")
        if tkr and title:
            out[tkr] = str(title)
    return out


def main() -> None:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    print("Fetching current S&P 500 constituent list (Wikipedia)...")
    cons, _chg = _fetch_wiki_tables()
    universe = _parse_current(cons)
    print(f"Universe: {len(universe)} tickers")

    print("Loading CIK map + company titles (SEC company_tickers.json)...")
    cik_map = load_cik_map(session=session)
    titles = _load_company_titles(session)
    print(f"CIK map: {len(cik_map)} tickers; titles: {len(titles)}")

    index: dict = {}
    fetch_log: list = []
    task_id = 0

    for i, ticker in enumerate(universe):
        cik = cik_map.get(ticker.replace("-", "."))  # SEC map uses '.' for class shares sometimes
        if not cik:
            cik = cik_map.get(ticker)
        if not cik:
            fetch_log.append({"ticker": ticker, "status": "NO_CIK"})
            continue

        bundle = IssuerBundle(
            company_names=(titles.get(ticker, ticker),),
            tickers=(ticker,),
            cik=cik,
        )

        filing = load_pit_filing(
            ticker, cik, AS_OF, forms=("10-Q", "10-K"),
            session=session, cache_dir=CACHE_DIR,
        )
        if filing is None:
            fetch_log.append({"ticker": ticker, "status": "NO_FILING"})
            continue
        if filing.filed < MODEL_CUTOFF:
            fetch_log.append({
                "ticker": ticker, "status": "PRE_CUTOFF_SKIPPED",
                "filed": filing.filed, "form": filing.form,
            })
            continue

        task = prepare_scoring_task(filing, bundle, blind=True, max_chars=45000)
        prompt = scoring_task_to_prompt(task)

        tid = f"task_{task_id:04d}"
        (TASKS_DIR / f"{tid}.txt").write_text(prompt, encoding="utf-8")
        index[tid] = {
            "ticker": ticker,
            "as_of": filing.filed,
            "form": filing.form,
            "audit": task.audit,
            "truncated": task.truncated,
            "original_chars": task.original_chars,
            "scored_chars": task.scored_chars,
        }
        fetch_log.append({
            "ticker": ticker, "status": "OK", "task_id": tid,
            "filed": filing.filed, "form": filing.form,
            "scored_chars": task.scored_chars,
        })
        print(f"[{i+1}/{len(universe)}] {ticker} @ {filing.filed} ({filing.form}): "
              f"{tid} ({task.scored_chars} chars)")
        task_id += 1

        if (i + 1) % 25 == 0:
            INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
            RAW_META_PATH.write_text(json.dumps(fetch_log, indent=2), encoding="utf-8")

    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    RAW_META_PATH.write_text(json.dumps(fetch_log, indent=2), encoding="utf-8")

    n_ok = sum(1 for f in fetch_log if f["status"] == "OK")
    n_no_cik = sum(1 for f in fetch_log if f["status"] == "NO_CIK")
    n_no_filing = sum(1 for f in fetch_log if f["status"] == "NO_FILING")
    n_pre_cutoff = sum(1 for f in fetch_log if f["status"] == "PRE_CUTOFF_SKIPPED")
    print(f"\n=== DONE ===\nOK: {n_ok}  NO_CIK: {n_no_cik}  NO_FILING: {n_no_filing}  "
          f"PRE_CUTOFF_SKIPPED (excluded, honest): {n_pre_cutoff}")
    print(f"universe={len(universe)} -> {task_id} strictly-post-cutoff tasks written to {TASKS_DIR}")


if __name__ == "__main__":
    main()
