"""Demo: AXIOM Hedge Layer exposure report on the USD-pileup example.

Read-only demonstration script — no broker/network I/O, no state.json touch,
not wired into any live loop. Prints the exposure report for the motivating
failure case: "Long USD/CAD + USD/CHF + USD/JPY looked like 3 trades but was
one USD macro bet" (the -$816 FX day).

Run: python scripts/hedge_exposure_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.hedge.portfolio_exposure import (  # noqa: E402
    ExposurePosition,
    build_exposure_report,
)


def main() -> None:
    usd_pileup = [
        ExposurePosition("fx", "USD_CAD", "long", 1.0, source="shadow_open"),
        ExposurePosition("fx", "USD_CHF", "long", 1.0, source="shadow_open"),
        ExposurePosition("fx", "USD_JPY", "long", 1.0, source="shadow_open"),
    ]
    report = build_exposure_report(usd_pileup)

    print("=== AXIOM Hedge Layer — sample exposure report (USD-pileup) ===")
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    print()
    print("--- narrative ---")
    for line in report.narrative():
        print(f"- {line}")


if __name__ == "__main__":
    main()
