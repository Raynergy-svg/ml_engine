#!/usr/bin/env python3
"""
analyze_dry_run.py — US-604

Reads trained_data/dry_run_validation.jsonl and prints:
- Total cycles (distinct timestamps)
- Total per-pair candidates evaluated
- Distribution of block_reasons (histogram)
- Percent of candidates that would_submit in LIVE mode
- Per-gate firing rate

Run:
    python scripts/analyze_dry_run.py
    python scripts/analyze_dry_run.py --path other.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, cast


def _iter_rows(path: Path) -> Iterable[dict]:
    if not path.exists():
        print(f"[!] {path} does not exist yet. Nothing to analyze.", file=sys.stderr)
        return
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] line {lineno} bad JSON: {exc}", file=sys.stderr)


def analyze(rows: List[dict]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "total_candidates": 0,
            "total_cycles": 0,
            "per_pair": {},
            "block_reason_hist": {},
            "would_submit_pct": 0.0,
            "gate_firing_rate": {},
            "regime_dist": {},
        }

    cycles = {r.get("ts") for r in rows if r.get("ts")}
    per_pair = Counter(r.get("pair", "UNKNOWN") for r in rows)
    regimes = Counter(r.get("regime", "UNKNOWN") for r in rows)

    reason_hist: Counter = Counter()
    for r in rows:
        for reason in r.get("block_reasons", []) or []:
            reason_hist[reason] += 1

    would_submit = sum(1 for r in rows if r.get("would_submit"))
    pct = (would_submit / total) * 100.0

    gate_counts: Counter = Counter()
    for r in rows:
        gates = r.get("gates_fired") or {}
        for g, fired in gates.items():
            if fired:
                gate_counts[g] += 1

    gate_rate = {g: (gate_counts[g] / total) for g in gate_counts}

    return {
        "total_candidates": total,
        "total_cycles": len(cycles),
        "per_pair": dict(per_pair.most_common()),
        "block_reason_hist": dict(reason_hist.most_common()),
        "would_submit_count": would_submit,
        "would_submit_pct": round(pct, 2),
        "gate_firing_rate": {g: round(v, 4) for g, v in gate_rate.items()},
        "regime_dist": dict(regimes.most_common()),
    }


def _print_report(summary: Dict[str, Any], path: Path) -> None:
    print("=" * 60)
    print(f"Dry-Run Validation Report — {path}")
    print("=" * 60)
    print(f"Total scan cycles:         {summary['total_cycles']}")
    print(f"Total candidates:          {summary['total_candidates']}")
    print(
        f"Would-submit (LIVE mode):  "
        f"{summary.get('would_submit_count', 0)} "
        f"({summary['would_submit_pct']}%)"
    )

    print("\nPer-pair candidates:")
    for pair, count in (summary["per_pair"] or {}).items():
        print(f"  {pair:<10} {count}")

    print("\nRegime distribution:")
    for reg, count in (summary["regime_dist"] or {}).items():
        print(f"  {reg:<10} {count}")

    print("\nBlock-reason histogram:")
    hist = summary["block_reason_hist"] or {}
    if not hist:
        print("  (no blocks — all candidates tradeable)")
    else:
        for reason, count in hist.items():
            print(f"  {reason:<22} {count}")

    print("\nPer-gate firing rate:")
    rates = summary["gate_firing_rate"] or {}
    if not rates:
        print("  (no gate firings recorded)")
    else:
        for gate, rate in sorted(rates.items(), key=lambda kv: -kv[1]):
            print(f"  {gate:<22} {rate:.2%}")

    print("=" * 60)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default="trained_data/dry_run_validation.jsonl",
        help="Path to the validation JSONL file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human report.",
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    rows = list(_iter_rows(path))
    summary = analyze(rows)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_report(summary, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
