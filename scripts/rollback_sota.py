#!/usr/bin/env python3
"""CLI: Rollback SOTA model to a previous version.

Usage:
    python scripts/rollback_sota.py --list
    python scripts/rollback_sota.py --to 20260615_123000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sota_core.model_versioning import list_versions, rollback_to_version


def main() -> None:
    parser = argparse.ArgumentParser(description="SOTA model rollback")
    parser.add_argument("--list", action="store_true", help="List available versions")
    parser.add_argument("--to", type=str, help="Rollback to version timestamp")
    args = parser.parse_args()

    if args.list:
        versions = list_versions()
        if not versions:
            print("No versions found.")
            return
        print(f"{'Timestamp':<20} {'Saved At':<26} {'Data Range'}")
        for v in versions:
            print(f"{v['timestamp']:<20} {v['saved_at']:<26} {v['lineage_summary']}")
        return

    if args.to:
        ok = rollback_to_version(args.to)
        sys.exit(0 if ok else 1)

    parser.print_help()


if __name__ == "__main__":
    main()
