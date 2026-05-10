"""Fit synthetic Platt scaling parameters from a reasonable prior.

Phase 44 — bridges the gap between "no calibration data yet" and "real refit
from journal outcomes". Writes default Platt sigmoid params that map raw agent
vote scores to win probabilities under a sensible monotone prior.

The file written is in the same schema as ConfidenceCalibrationSystem expects
(see src/scanner/confidence_calibration.py:_load_calibration ~line 822). Top
level: ``{"version": 1, "platt_params": {regime_name: {coef, intercept}}, ...}``

Default coef=4.0, intercept=-2.0 produces:
- raw 0.5 -> calibrated 0.5  (no shift at the midpoint)
- raw 0.7 -> calibrated 0.69 (mild compression toward the win side)
- raw 0.3 -> calibrated 0.31 (mild compression toward the loss side)

This is intentionally conservative. Real refitting from trade outcomes will
overwrite these values; until then they at least make _platt_scale return
``is_calibrated=True`` and a sensible monotone mapping rather than the raw
identity passthrough.

Atomic write via tmp + os.rename so a partial-read by a running process can
never see a half-written file.

Usage:
    python scripts/fit_platt_synthetic.py
    python scripts/fit_platt_synthetic.py --coef 4.0 --intercept -2.0
    python scripts/fit_platt_synthetic.py --regime _global \\
        --output trained_data/models/calibration_params.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


DEFAULT_OUTPUT = "trained_data/models/calibration_params.json"
DEFAULT_REGIME = "_global"
DEFAULT_COEF = 4.0
DEFAULT_INTERCEPT = -2.0


def sigmoid(x: float) -> float:
    """Numerically-stable sigmoid for the sample-table preview."""
    if x >= 0:
        ex = math.exp(-x)
        return 1.0 / (1.0 + ex)
    ex = math.exp(x)
    return ex / (1.0 + ex)


def platt(raw_score: float, coef: float, intercept: float) -> float:
    """Apply Platt sigmoid: P = 1 / (1 + exp(-(coef*raw + intercept)))."""
    return sigmoid(coef * raw_score + intercept)


def build_payload(regime: str, coef: float, intercept: float) -> Dict:
    """Produce the JSON payload in ConfidenceCalibrationSystem's schema."""
    return {
        "version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platt_params": {
            regime: {
                "coef": float(coef),
                "intercept": float(intercept),
            }
        },
        "trade_history": [],
        "metadata": {
            "n_trades": 0,
            "n_regimes": 1,
            "source": "synthetic_prior",
            "rationale": (
                "Synthetic Platt prior used until trade history accumulates "
                "enough outcomes for refit. Maps raw 0.5 -> 0.5; pushes 0.7 "
                "to ~0.69 and 0.3 to ~0.31 under coef=4.0, intercept=-2.0."
            ),
        },
    }


def atomic_write_json(path: Path, payload: Dict) -> None:
    """Write JSON atomically: tmp + os.rename. Caller catches OSError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


def print_sample_table(coef: float, intercept: float) -> None:
    """Smoke-check the sigmoid: print a small input/output table."""
    print(f"\nPlatt sigmoid sample (coef={coef}, intercept={intercept}):")
    print(f"  {'raw':>8s}  ->  {'calibrated':>10s}")
    for raw in (0.10, 0.30, 0.50, 0.582, 0.70, 0.90):
        cal = platt(raw, coef, intercept)
        print(f"  {raw:>8.3f}  ->  {cal:>10.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Platt scaling params for confidence calibration.",
    )
    parser.add_argument(
        "--coef", type=float, default=DEFAULT_COEF,
        help=f"Sigmoid slope coefficient (default {DEFAULT_COEF}).",
    )
    parser.add_argument(
        "--intercept", type=float, default=DEFAULT_INTERCEPT,
        help=f"Sigmoid intercept (default {DEFAULT_INTERCEPT}).",
    )
    parser.add_argument(
        "--regime", type=str, default=DEFAULT_REGIME,
        help=f"Regime key in platt_params (default '{DEFAULT_REGIME}').",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sample table only; do not write the file.",
    )
    args = parser.parse_args()

    payload = build_payload(args.regime, args.coef, args.intercept)
    print_sample_table(args.coef, args.intercept)

    if args.dry_run:
        print("\n--dry-run: not writing.")
        return 0

    out = Path(args.output)
    try:
        atomic_write_json(out, payload)
    except OSError as exc:
        print(f"ERROR: failed to write {out}: {exc}", file=sys.stderr)
        return 1

    size = out.stat().st_size
    print(f"\nWrote {out} ({size} bytes)")
    print(f"Schema: {{'platt_params': {{'{args.regime}': "
          f"{{'coef': {args.coef}, 'intercept': {args.intercept}}}}}, ...}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
