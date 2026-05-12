#!/usr/bin/env python3
"""Inspect and roll back versioned snapshots of agent_weights.json.

Examples::

    # List the 20 most-recent versions (default action)
    python scripts/rollback_weights.py

    # Show a longer history
    python scripts/rollback_weights.py --list --limit 100

    # Preview a rollback without writing
    python scripts/rollback_weights.py --to a3f9 --dry-run

    # Restore a snapshot (prompts for confirmation unless --yes)
    python scripts/rollback_weights.py --to a3f9 --yes

    # Show the delta a rollback would apply
    python scripts/rollback_weights.py --diff a3f9

History entries are written by :mod:`src.scanner.weights_history` on every
successful write to ``trained_data/models/agent_weights.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.scanner.weights_history import (  # noqa: E402
    HISTORY_PATH,
    find_version,
    list_versions,
    restore_version,
)

DEFAULT_WEIGHTS_PATH = _PROJECT_ROOT / "trained_data" / "models" / "agent_weights.json"


def _load_current_weights(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _flatten_weights(weights: Any) -> Dict[str, float]:
    """Flatten a regime-keyed weights dict to {regime/agent: float}.

    Handles both shapes:
      - ``{"LOW": {"trend": 1.0, ...}, "_global": {...}, ...}``
      - ``{"weights": {...}}`` (the walkforward_retrainer envelope)
    Non-numeric leaves and ``_meta`` keys are skipped.
    """
    if not isinstance(weights, dict):
        return {}
    if "weights" in weights and isinstance(weights["weights"], dict) and len(weights) <= 3:
        return _flatten_weights(weights["weights"])
    flat: Dict[str, float] = {}
    for regime, row in weights.items():
        if regime.startswith("_"):
            continue
        if not isinstance(row, dict):
            continue
        for agent, val in row.items():
            try:
                flat[f"{regime}/{agent}"] = float(val)
            except (TypeError, ValueError):
                continue
    return flat


def _delta_summary(snapshot: Dict[str, Any], current: Dict[str, Any], top_n: int = 10) -> List[str]:
    """Return human-readable diff lines, top-N by absolute delta."""
    snap = _flatten_weights(snapshot)
    cur = _flatten_weights(current)
    keys = set(snap) | set(cur)
    deltas = []
    for k in keys:
        a, b = snap.get(k), cur.get(k)
        if a is None:
            deltas.append((k, float("inf"), None, b, "ADDED in current"))
        elif b is None:
            deltas.append((k, float("inf"), a, None, "REMOVED from current"))
        else:
            d = b - a
            if abs(d) >= 1e-6:
                deltas.append((k, abs(d), a, b, f"{a:.4f} -> {b:.4f}   (Δ {d:+.4f})"))
    deltas.sort(key=lambda t: t[1], reverse=True)
    if not deltas:
        return ["  (no differences)"]
    return [f"  {k:<35s} {desc}" for k, _, _, _, desc in deltas[:top_n]]


def _print_versions(versions: List[Dict[str, Any]], current: Dict[str, Any], history_path: Path) -> None:
    if not versions:
        print(f"No history entries found at {history_path}")
        return

    print(f"{'#':>3}  {'TIMESTAMP':<28} {'HASH':<10} {'SOURCE':<32} {'Δ AGENTS':>9}")
    print("-" * 90)
    for i, rec in enumerate(versions):
        h = (rec.get("snapshot_hash") or "")[:8]
        snap = rec.get("weights_snapshot") or {}
        snap_flat = _flatten_weights(snap)
        cur_flat = _flatten_weights(current)
        n_diff = sum(
            1 for k in set(snap_flat) | set(cur_flat)
            if abs(snap_flat.get(k, 0.0) - cur_flat.get(k, 0.0)) >= 1e-6
        )
        print(
            f"{i:>3}  {rec.get('ts_iso','?'):<28} {h:<10} "
            f"{(rec.get('source') or '')[:32]:<32} {n_diff:>9}"
        )


def _print_diff(hash_prefix: str, current: Dict[str, Any]) -> int:
    rec = find_version(hash_prefix)
    if rec is None:
        print(f"No history entry matches hash prefix '{hash_prefix}'")
        return 2
    print(f"Diff: snapshot {rec.get('snapshot_hash','?')[:8]} → current agent_weights.json")
    print(f"  source: {rec.get('source')}")
    print(f"  ts:     {rec.get('ts_iso')}")
    print()
    for line in _delta_summary(rec.get("weights_snapshot") or {}, current):
        print(line)
    return 0


def _do_restore(hash_prefix: str, weights_path: Path, *, yes: bool, dry_run: bool) -> int:
    rec = find_version(hash_prefix)
    if rec is None:
        print(f"No history entry matches hash prefix '{hash_prefix}'")
        return 2
    short = (rec.get("snapshot_hash") or "")[:8]
    print(
        f"Restore plan: write snapshot {short} (source={rec.get('source')}, "
        f"ts={rec.get('ts_iso')}) to {weights_path}"
    )
    print("Delta vs current (top 10):")
    for line in _delta_summary(rec.get("weights_snapshot") or {}, _load_current_weights(weights_path)):
        print(line)

    if dry_run:
        print("\n--dry-run: not writing.")
        return 0
    if not yes:
        try:
            confirm = input("\nProceed? [y/N] ").strip().lower()
        except EOFError:
            confirm = ""
        if confirm not in {"y", "yes"}:
            print("Aborted.")
            return 1

    try:
        restore_version(hash_prefix, weights_path=weights_path)
    except (LookupError, ValueError, OSError) as e:
        print(f"Restore failed: {e}")
        return 2
    print(f"Restored. New history entry tagged rollback:{short}.")
    return 0


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect / roll back agent_weights.json history.")
    p.add_argument("--list", action="store_true", help="List recent versions (default action).")
    p.add_argument("--limit", type=int, default=20, help="Max versions to show (default: 20).")
    p.add_argument("--to", metavar="HASH_PREFIX", help="Restore the snapshot matching this hash prefix (>=4 chars).")
    p.add_argument("--diff", metavar="HASH_PREFIX", help="Show delta between this snapshot and current weights.")
    p.add_argument("--dry-run", action="store_true", help="Preview --to without writing.")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt on --to.")
    p.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH,
                   help="Path to agent_weights.json (default: trained_data/models/agent_weights.json).")
    p.add_argument("--history-path", type=Path, default=HISTORY_PATH,
                   help="Path to weight_history.jsonl.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    current = _load_current_weights(args.weights_path)

    if args.diff:
        return _print_diff(args.diff, current)
    if args.to:
        return _do_restore(args.to, args.weights_path, yes=args.yes, dry_run=args.dry_run)

    versions = list_versions(limit=args.limit, path=args.history_path)
    _print_versions(versions, current, args.history_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
