"""Demo: AXIOM Hedge Layer Phase 3 hedge-proposal reports.

Read-only demonstration script — no broker/network I/O, no state.json touch,
not wired into any live loop. Prints structured hedge-proposal JSON for two
cases: the USD-pileup FX headline case and a Technology-sector equity sleeve.

Run: python scripts/hedge_candidates_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.hedge.hedge_candidates import build_hedge_report  # noqa: E402
from src.hedge.portfolio_exposure import (  # noqa: E402
    ExposurePosition,
    build_exposure_report,
)


def main() -> None:
    print("=== AXIOM Hedge Layer Phase 3 — USD-pileup FX hedge report ===")
    fx_open = [
        ExposurePosition("fx", "USD_CAD", "long", 1.0, source="shadow_open"),
        ExposurePosition("fx", "USD_CHF", "long", 1.0, source="shadow_open"),
    ]
    fx_candidate = ExposurePosition("fx", "USD_JPY", "long", 1.0, source="candidate")
    fx_report = build_exposure_report(fx_open, fx_candidate)
    print(json.dumps(
        build_hedge_report(fx_candidate, fx_report,
                           relative_strength={"CAD": 0.10, "CHF": -0.05, "JPY": 0.02}),
        indent=2, sort_keys=True))

    print()
    print("=== AXIOM Hedge Layer Phase 3 — Technology-sector equity hedge report ===")
    eq_open = [
        ExposurePosition("equity", "AAPL", "long", 1.0, source="shadow_open"),
        ExposurePosition("equity", "MSFT", "long", 1.0, source="shadow_open"),
    ]
    eq_candidate = ExposurePosition("equity", "NVDA", "long", 1.0, source="candidate")
    eq_report = build_exposure_report(eq_open, eq_candidate)
    print(json.dumps(
        build_hedge_report(eq_candidate, eq_report,
                           relative_strength={"AAPL": 0.10, "MSFT": -0.05, "NVDA": 0.20}),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
