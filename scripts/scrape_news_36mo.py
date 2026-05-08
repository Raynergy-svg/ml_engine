"""Stage 4-A historical news backfill — 36 months × 7 pairs sequential.

Run from repo root:
    python scripts/scrape_news_36mo.py

Output: trained_data/news/{PAIR}_ff_events.parquet (one per pair).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# stream INFO+ to stdout so the watchdog sees the per-week heartbeats
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("scrape_36mo")

CACHE_DIR = REPO / "trained_data" / "news"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
]

from src.data.news.source import ForexFactoryNewsSource  # noqa: E402

src = ForexFactoryNewsSource(cache_dir=str(CACHE_DIR))

now = datetime.now(timezone.utc)
since = now - timedelta(days=365 * 3)  # 36 months
log.info("Backfill window: %s -> %s (%.1f days)",
         since.isoformat(), now.isoformat(), (now - since).days)

results: list[dict] = []
t_global = time.monotonic()

for pair in PAIRS:
    t0 = time.monotonic()
    log.info("=" * 60)
    log.info("PAIR START: %s", pair)
    log.info("=" * 60)
    n_events = 0
    err: str | None = None
    try:
        events = src.fetch_events_historical(pair, since=since, until=now)
        n_events = len(events)
    except Exception as e:  # noqa: BLE001 — runner must capture/continue
        err = f"{type(e).__name__}: {e}"
        log.exception("Pair %s FAILED: %s", pair, err)

    cache_path = CACHE_DIR / f"{pair}_ff_events.parquet"
    cache_kb = cache_path.stat().st_size / 1024 if cache_path.exists() else 0.0

    first_ts = last_ts = "n/a"
    if cache_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(cache_path)
            if not df.empty and "ts_utc" in df.columns:
                first_ts = str(df["ts_utc"].min())
                last_ts = str(df["ts_utc"].max())
        except Exception as e:  # noqa: BLE001
            log.warning("Could not read parquet for span report: %s", e)

    elapsed = time.monotonic() - t0
    line = (
        f"{pair}: {n_events} events, cache={cache_kb:.1f} KB, "
        f"span {first_ts} -> {last_ts}, elapsed={elapsed:.1f}s"
        + (f" ERR={err}" if err else "")
    )
    log.info("PAIR DONE: %s", line)
    results.append({
        "pair": pair, "n_events": n_events, "cache_kb": cache_kb,
        "first_ts": first_ts, "last_ts": last_ts,
        "elapsed_sec": elapsed, "err": err,
    })

total_elapsed = time.monotonic() - t_global
log.info("=" * 60)
log.info("ALL PAIRS DONE — total wall-clock %.1fs (%.1fmin)",
         total_elapsed, total_elapsed / 60.0)
total_events = sum(r["n_events"] for r in results)
log.info("Total events across 7 pairs: %d", total_events)
for r in results:
    log.info("  %-8s  %6d events  %7.1f KB  %6.1fs%s",
             r["pair"], r["n_events"], r["cache_kb"], r["elapsed_sec"],
             "  ERR=" + r["err"] if r["err"] else "")

# Final summary as a sentinel string the parent agent grep'd for.
print("\n=== SCRAPE_36MO_SUMMARY ===")
for r in results:
    print(f"PAIR_RESULT {r['pair']} events={r['n_events']} "
          f"kb={r['cache_kb']:.1f} elapsed={r['elapsed_sec']:.1f}s "
          f"err={r['err'] or 'none'}")
print(f"TOTAL_EVENTS {total_events}")
print(f"TOTAL_ELAPSED_SEC {total_elapsed:.1f}")
print("=== END_SUMMARY ===")
