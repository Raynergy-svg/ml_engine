"""Demo: AXIOM Hedge Layer Phase 4 raw-vs-hedged scorecard, illustrated with
a real historical window.

Read-only demonstration script — no broker/network I/O, no state.json
touch, not wired into any live loop, never writes to the production ledger
(``trained_data/hedge/raw_vs_hedged_ledger.jsonl``).

The cached S&P constituent price panel (``market_data/equity/
sp500_prices.parquet``) was last refreshed 2026-06-24 — stale relative to
"today" (2026-07-07+), so a genuine forward mark from the CURRENT book
snapshot has no price data to mark against yet (this is exactly what
``hedged_shadow_lane`` reports honestly as ``insufficient_forward_price_data``
when run for real — see ``run_all()``). To still show what a POPULATED
scorecard looks like, this script marks the REAL current equity-harvester
book (``trained_data/equity/rebalance_state.json``) forward through the
LAST FEW cached trading days it does have data for (2026-06-17 ->
2026-06-24) — real prices, real weights, just a past window rather than
"from now". Every row is clearly labeled as a historical-window demo, never
mixed into the production ledger.

Run: python scripts/hedge_shadow_lane_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.hedge import hedge_scorecard as hsc  # noqa: E402
from src.hedge import hedged_shadow_lane as hsl  # noqa: E402
from src.hedge.exposure_tags import (  # noqa: E402
    CURRENCY_BUCKET_MAP_PATH,
    SECTOR_BUCKET_MAP_PATH,
    load_bucket_map,
)

N_DEMO_DAYS = 6  # tail window of the cached panel to walk forward through


def _equity_harvester_history() -> list:
    real_book = hsl.load_equity_harvester_book()
    if real_book is None:
        print("equity harvester: no live book found (rebalance_state.json missing/empty) — skipping demo")
        return []

    panel = hsl.load_equity_price_panel()
    if panel.empty:
        print("equity price panel missing/empty — skipping demo")
        return []

    sector_map = load_bucket_map(SECTOR_BUCKET_MAP_PATH)
    currency_map = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)

    dates = list(panel.index[-(N_DEMO_DAYS + 1):])
    rows = []
    for i in range(len(dates) - 1):
        asof = str(dates[i].date())
        # Reuses the REAL current weights held static across this short
        # historical window (the harvester rebalances ~every 21 days, so
        # this is a reasonable approximation of "the book that was live"
        # over a 6-day tail, not a claim that the book was IDENTICAL then).
        book = hsl.BookSnapshot(
            strategy="equity_harvester", asset_class="equity", asof_date=asof,
            weights=real_book.weights, raw_net_return=None, raw_gross_return=None,
            raw_cost=None, return_source="must_compute",
            meta={"demo": "historical_window", "note": "current live weights held static over a past window"})
        row = hsl.run_cycle_for_strategy(
            "equity_harvester", price_panel=panel, sector_bucket_map=sector_map,
            currency_bucket_map=currency_map, persist=False, book=book)
        if row is not None:
            rows.append(row)
    return rows


def main() -> None:
    print("=== AXIOM Hedge Layer Phase 4 — raw-vs-hedged demo ===")
    print(f"(historical window demo, NOT written to {hsl.RAW_VS_HEDGED_LEDGER_PATH})\n")

    rows = _equity_harvester_history()
    if not rows:
        return

    print(f"--- equity_harvester: {len(rows)} historical cycles "
          f"({rows[0]['asof_date']} -> {rows[-1]['asof_date']}) ---")
    for row in rows:
        applied = row["hedge"]["applied_proposal"]
        style = applied["style"] if applied else None
        print(f"  {row['asof_date']}: raw={row['raw']['net_return']:+.5f}  "
              f"hedge={style}  hedged_gross={row['hedged']['gross_return']}  "
              f"cost_known={row['hedge']['overlay_cost_known']}")

    card = hsc.compute_strategy_scorecard(rows)
    print("\n--- scorecard (alpha-vs-beta readout) ---")
    print(json.dumps(card, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
