"""Iteratively retrain each remaining pair at M15 65k candles, paired
with EUR_USD as the strong-signal donor. Capture per-pair holdout
results to a JSON summary.

The 5-pair joint retrain at M15 failed (51.2% holdout) because the
5-pair group spanned 5 heterogeneous market regimes (USD vs Asian +
USD vs Commodity). The 2-pair smoke EUR_USD+GBP_USD passed at 70%
because both are EUR/GBP-quoted dollar pairs (highly correlated).

Hypothesis: pairing each target with a STRONG-SIGNAL DONOR (EUR_USD)
will let correlation transfer surface per-pair signal even when the
target alone doesn't have enough.

Output: trained_data/m15_pair_expansion_report.json with per-pair
M15 holdout accuracy, gate-pass status, and aggregate stats.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("expand_pairs")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Pairs to test (14 — EUR_USD/GBP_USD already validated)
TARGETS = [
    "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_AUD", "EUR_JPY", "EUR_CHF",
    "GBP_JPY", "GBP_AUD", "GBP_CHF",
    "AUD_JPY", "AUD_NZD",
]

DONOR = "EUR_USD"
GATE = 0.52


def parse_holdout(log_path: Path) -> dict:
    """Parse M15/H1/H4 holdout accuracy from a retrain log."""
    text = log_path.read_text(errors="ignore")
    out = {"M15": None, "H1": None, "H4": None, "primary_tf": None,
           "primary_acc": None, "aggregate": None, "promoted": False,
           "promotion_reason": None}
    # Per-TF lines: "Hold-out validation [M15]: 68.8% accuracy (275/400) PASSED"
    for tf in ("H1", "M15", "H4"):
        m = re.search(rf"Hold-out validation \[{tf}\]: (\d+\.\d+)% accuracy", text)
        if m:
            out[tf] = float(m.group(1)) / 100.0
    # Primary TF / aggregate: "Holdout primary TF=M15 accuracy=0.688 (400 samples) | aggregate=0.583"
    m = re.search(r"Holdout primary TF=(\w+) accuracy=([\d.]+).*?aggregate=([\d.]+)", text)
    if m:
        out["primary_tf"] = m.group(1)
        out["primary_acc"] = float(m.group(2))
        out["aggregate"] = float(m.group(3))
    # Promotion verdict
    m = re.search(r"decision\s*:\s*(\w+).*?reason\s*:\s*([^\n]+)", text, re.DOTALL)
    if m:
        out["promoted"] = (m.group(1).strip().upper() == "PROMOTE")
        out["promotion_reason"] = m.group(2).strip()
    return out


def run_one(target: str, donor: str, candles: int = 65000) -> dict:
    log_name = f"logs/expand_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = REPO_ROOT / log_name
    cmd = [
        "python3", "scripts/scheduled_retrain.py",
        "--pairs", f"{target},{donor}",
        "--granularity", "M15",
        "--candles", str(candles),
    ]
    log.info("[%s] starting: %s", target, " ".join(cmd))
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT,
                              timeout=900)  # 15-min cap per pair
    elapsed = time.time() - t0
    log.info("[%s] done in %.0fs (exit=%d), parsing log", target, elapsed, proc.returncode)
    holdout = parse_holdout(log_path)
    return {
        "target": target,
        "donor": donor,
        "exit_code": proc.returncode,
        "duration_sec": elapsed,
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        **holdout,
        "passed_gate": (holdout.get("M15") or 0) >= GATE,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", default=",".join(TARGETS),
                    help="Comma-separated target pairs to iterate")
    ap.add_argument("--donor", default=DONOR, help="Donor pair (high-signal)")
    ap.add_argument("--candles", type=int, default=65000)
    ap.add_argument("--report", type=Path,
                    default=REPO_ROOT / "trained_data" / "m15_pair_expansion_report.json")
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    log.info("Iterating %d targets with donor=%s, candles=%d",
             len(targets), args.donor, args.candles)

    results = []
    args.report.parent.mkdir(parents=True, exist_ok=True)

    for i, target in enumerate(targets, 1):
        log.info("=== %d/%d: %s ===", i, len(targets), target)
        try:
            res = run_one(target, args.donor, args.candles)
        except subprocess.TimeoutExpired:
            log.error("[%s] timeout >15min, skipping", target)
            res = {"target": target, "donor": args.donor, "error": "timeout",
                   "passed_gate": False, "M15": None}
        except Exception as e:
            log.error("[%s] error: %s", target, e)
            res = {"target": target, "donor": args.donor, "error": str(e),
                   "passed_gate": False, "M15": None}
        results.append(res)
        # Persist intermediate results so partial progress survives crashes
        args.report.write_text(json.dumps({
            "donor": args.donor,
            "candles": args.candles,
            "gate": GATE,
            "completed": i,
            "total": len(targets),
            "results": results,
            "ts": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True))

    # Final summary
    passed = [r for r in results if r.get("passed_gate")]
    failed = [r for r in results if not r.get("passed_gate")]
    log.info("\n%s", "=" * 60)
    log.info("PAIR EXPANSION RESULTS: %d/%d passed M15 gate (%.1f%%)",
             len(passed), len(results), 100 * len(passed) / max(len(results), 1))
    log.info("PASSED:")
    for r in sorted(passed, key=lambda r: -(r.get("M15") or 0)):
        log.info("  %-10s M15=%.1f%%", r["target"], 100 * (r.get("M15") or 0))
    log.info("FAILED:")
    for r in sorted(failed, key=lambda r: -(r.get("M15") or 0)):
        m15 = r.get("M15")
        log.info("  %-10s M15=%s", r["target"], f"{100*m15:.1f}%" if m15 else "ERROR")
    log.info("Report: %s", args.report)
    return 0 if len(passed) > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
