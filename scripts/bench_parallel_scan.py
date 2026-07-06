"""
Benchmark + correctness validation for the two-phase parallel scanner.

Runs N cycles of Scanner.scan() in sequential mode and parallel mode
against real models loaded from `trained_data/models/joint/` and OHLCV
data served by the local CSV fallback in `market_data/`.

Outputs:
  - Console summary (per-cycle wall-clock min/p50/p95, speedup, decision-
    match counts).
  - JSON dump under `trained_data/_benchmarks/parallel_scan_<ts>.json`.

Usage:
  python scripts/bench_parallel_scan.py [--cycles 5] [--workers 7]

Notes:
  - This benchmark intentionally does NOT touch OANDA. It uses the CSV
    fallback at `src/scanner/engine.py:_load_local_csv` which engages
    automatically when no broker/OANDA client is available.
  - Pairs are restricted to those with H1 CSVs in `market_data/`.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Pairs with H1 CSVs in market_data/ — discovered via:
#   ls market_data/ | sed 's/oanda_\([A-Z_]*\)_H1.*/\1/' | sort -u
PAIRS_WITH_CSV = ["AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY"]


@dataclass
class CycleTiming:
    cycle: int
    mode: str  # "sequential" or "parallel"
    wall_secs: float
    pairs_scanned: int
    decisions: Dict[str, str] = field(default_factory=dict)  # pair -> direction


@dataclass
class BenchmarkResult:
    cycles: int
    workers: int
    pairs: List[str]
    sequential_walls: List[float] = field(default_factory=list)
    parallel_walls: List[float] = field(default_factory=list)
    decision_match_per_cycle: List[bool] = field(default_factory=list)
    mismatches: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _build_scanner(pairs: List[str]) -> Any:
    """Construct a Scanner using local CSV fallback for OHLCV."""
    from src.scanner.config import ScannerConfig
    from src.scanner.engine import Scanner

    cfg = ScannerConfig(
        pairs=pairs,
        granularity="H1",
        min_candles=100,
    )
    # No oanda_client → init_oanda_client will fail → CSV fallback engages.
    return Scanner(config=cfg, oanda_client=None)


def _extract_decisions(analyses: List[Any]) -> Dict[str, str]:
    """Pull (pair → direction) for cycle-to-cycle comparison."""
    decisions: Dict[str, str] = {}
    for a in analyses or []:
        pair = getattr(a, "pair", None)
        direction = getattr(a, "direction", None)
        if pair is not None:
            decisions[pair] = str(direction) if direction is not None else "?"
    return decisions


def _run_cycle(scanner: Any, mode: str) -> CycleTiming:
    if mode == "parallel":
        os.environ["BUDDY_PARALLEL_SCAN"] = "1"
    else:
        os.environ.pop("BUDDY_PARALLEL_SCAN", None)

    start = time.monotonic()
    result = scanner.scan()
    wall = time.monotonic() - start

    # Scanner.scan() returns a ScanResult; pull the analyses list off it.
    analyses = getattr(result, "analyses", result)
    decisions = _extract_decisions(analyses)
    return CycleTiming(
        cycle=0,
        mode=mode,
        wall_secs=wall,
        pairs_scanned=len(decisions),
        decisions=decisions,
    )


def run_benchmark(cycles: int, workers: Optional[int] = None) -> BenchmarkResult:
    if workers:
        os.environ["BUDDY_PARALLEL_SCAN_WORKERS"] = str(workers)

    pairs = PAIRS_WITH_CSV
    print(f"Building scanner with {len(pairs)} pairs (CSV fallback)...")
    scanner = _build_scanner(pairs)

    result = BenchmarkResult(
        cycles=cycles,
        workers=workers or 0,
        pairs=pairs,
    )

    # Warm-up cycle (loads models, primes caches) — discarded.
    print("Warm-up cycle...")
    try:
        _run_cycle(scanner, "sequential")
    except Exception as e:
        print(f"Warm-up failed: {e}")
        result.errors.append(f"warmup: {e}")
        return result

    # Alternate sequential / parallel each iteration so any drift in
    # data/cache state hits both modes equally.
    for i in range(cycles):
        print(f"Cycle {i+1}/{cycles}...")
        try:
            seq = _run_cycle(scanner, "sequential")
            par = _run_cycle(scanner, "parallel")

            result.sequential_walls.append(seq.wall_secs)
            result.parallel_walls.append(par.wall_secs)

            match = seq.decisions == par.decisions
            result.decision_match_per_cycle.append(match)
            if not match:
                # Diff
                diff = {
                    "cycle": i + 1,
                    "seq_only": {p: d for p, d in seq.decisions.items() if par.decisions.get(p) != d},
                    "par_only": {p: d for p, d in par.decisions.items() if seq.decisions.get(p) != d},
                }
                result.mismatches.append(diff)
            print(f"  seq={seq.wall_secs:.3f}s par={par.wall_secs:.3f}s match={match}")
        except Exception as e:
            print(f"  cycle failed: {e}")
            result.errors.append(f"cycle {i+1}: {e}")

    return result


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[max(0, min(k, len(s) - 1))]


def _print_summary(r: BenchmarkResult) -> None:
    print("\n" + "=" * 60)
    print("PARALLEL SCAN BENCHMARK")
    print("=" * 60)
    print(f"Cycles:  {r.cycles}")
    print(f"Pairs:   {len(r.pairs)} ({', '.join(r.pairs)})")
    print(f"Workers: {r.workers or 'default'}")
    print()

    if not r.sequential_walls or not r.parallel_walls:
        print("No timing data captured.")
        if r.errors:
            print("Errors:")
            for e in r.errors:
                print(f"  - {e}")
        return

    seq_p50 = _percentile(r.sequential_walls, 50)
    par_p50 = _percentile(r.parallel_walls, 50)
    seq_p95 = _percentile(r.sequential_walls, 95)
    par_p95 = _percentile(r.parallel_walls, 95)
    speedup = (seq_p50 / par_p50) if par_p50 > 0 else float("nan")

    print(f"{'Mode':<14}{'min':>10}{'p50':>10}{'p95':>10}{'max':>10}")
    print("-" * 54)
    print(
        f"{'sequential':<14}"
        f"{min(r.sequential_walls):>10.3f}"
        f"{seq_p50:>10.3f}"
        f"{seq_p95:>10.3f}"
        f"{max(r.sequential_walls):>10.3f}"
    )
    print(
        f"{'parallel':<14}"
        f"{min(r.parallel_walls):>10.3f}"
        f"{par_p50:>10.3f}"
        f"{par_p95:>10.3f}"
        f"{max(r.parallel_walls):>10.3f}"
    )
    print()
    print(f"Speedup (p50): {speedup:.2f}x")
    print()

    matches = sum(1 for m in r.decision_match_per_cycle if m)
    total = len(r.decision_match_per_cycle)
    print(f"Decision parity: {matches}/{total} cycles matched")
    if r.mismatches:
        print("Mismatches:")
        for m in r.mismatches:
            print(f"  cycle {m['cycle']}: seq_only={m['seq_only']} par_only={m['par_only']}")

    if r.errors:
        print(f"\nErrors: {len(r.errors)}")
        for e in r.errors:
            print(f"  - {e}")


def _save_json(r: BenchmarkResult) -> Path:
    out_dir = PROJECT_ROOT / "trained_data" / "_benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"parallel_scan_{ts}.json"
    with open(path, "w") as f:
        json.dump(asdict(r), f, indent=2, default=str)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0, help="0 = default sizing")
    args = parser.parse_args()

    workers = args.workers if args.workers > 0 else None
    result = run_benchmark(cycles=args.cycles, workers=workers)
    _print_summary(result)
    json_path = _save_json(result)
    print(f"\nJSON dump: {json_path}")

    # Exit non-zero if any cycle's decisions don't match — that's a
    # correctness regression and must block.
    if result.mismatches:
        print("\nFAIL: parallel decisions diverged from sequential on at least one cycle.")
        return 2
    if result.errors:
        print("\nWARN: errors during run; review JSON dump.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
