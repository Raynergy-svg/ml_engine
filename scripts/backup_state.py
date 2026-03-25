#!/usr/bin/env python3
"""Automated Backup — Backs up critical state files to timestamped archives.

Backs up:
  - trained_data/trade_journal_rl.json (trade history)
  - trained_data/models/agent_weights.json (RL weights)
  - trained_data/pair_accuracy.json (accuracy gate state)
  - .claude/state.json (session state)

Usage:
    python scripts/backup_state.py           # Create backup
    python scripts/backup_state.py --list    # List backups
    python scripts/backup_state.py --prune   # Remove backups older than 30 days

Designed to run as a cron job or at the start of each watch mode session.
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = PROJECT_ROOT / "trained_data" / "backups"

# Files to back up
BACKUP_TARGETS = [
    "trained_data/trade_journal_rl.json",
    "trained_data/models/agent_weights.json",
    "trained_data/pair_accuracy.json",
    ".claude/state.json",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_backup() -> Path:
    """Create a timestamped backup of all critical state files.

    Returns:
        Path to the backup directory
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_subdir = BACKUP_DIR / timestamp
    backup_subdir.mkdir(parents=True, exist_ok=True)

    backed_up = 0
    for rel_path in BACKUP_TARGETS:
        src = PROJECT_ROOT / rel_path
        if src.exists():
            # Preserve directory structure in backup
            dest = backup_subdir / rel_path.replace("/", "__")
            try:
                shutil.copy2(str(src), str(dest))
                backed_up += 1
                logger.debug(f"  Backed up: {rel_path}")
            except Exception as e:
                logger.warning(f"  Failed to back up {rel_path}: {e}")
        else:
            logger.debug(f"  Skipped (not found): {rel_path}")

    # Write backup manifest
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files_backed_up": backed_up,
        "files_total": len(BACKUP_TARGETS),
    }
    (backup_subdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info(f"Backup created: {backup_subdir.name} ({backed_up}/{len(BACKUP_TARGETS)} files)")
    return backup_subdir


def list_backups() -> list:
    """List all existing backups."""
    if not BACKUP_DIR.exists():
        return []

    backups = sorted(BACKUP_DIR.iterdir(), reverse=True)
    results = []
    for b in backups:
        if b.is_dir():
            manifest_path = b / "manifest.json"
            manifest = {}
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                except Exception:
                    pass
            results.append({
                "name": b.name,
                "path": str(b),
                "files": manifest.get("files_backed_up", "?"),
                "timestamp": manifest.get("timestamp", b.name),
            })
    return results


def prune_backups(max_age_days: int = 30, keep_min: int = 5) -> int:
    """Remove backups older than max_age_days, keeping at least keep_min.

    Returns:
        Number of backups removed
    """
    backups = list_backups()
    if len(backups) <= keep_min:
        logger.info(f"Only {len(backups)} backups exist, keeping all (min={keep_min})")
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    removed = 0

    for backup in backups[keep_min:]:  # Never prune the most recent keep_min
        try:
            ts_str = backup["name"]
            ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            if ts < cutoff:
                shutil.rmtree(backup["path"])
                removed += 1
                logger.info(f"Pruned old backup: {backup['name']}")
        except Exception as e:
            logger.debug(f"Prune skipped {backup['name']}: {e}")

    return removed


def main():
    parser = argparse.ArgumentParser(description="Backup critical ML Engine state files")
    parser.add_argument("--list", action="store_true", help="List existing backups")
    parser.add_argument("--prune", action="store_true", help="Remove backups older than 30 days")
    parser.add_argument("--max-age", type=int, default=30, help="Max backup age in days (for --prune)")
    args = parser.parse_args()

    if args.list:
        backups = list_backups()
        if not backups:
            print("No backups found.")
            return 0
        print(f"{'Name':20s} {'Files':>6s}  {'Timestamp'}")
        print("-" * 60)
        for b in backups:
            print(f"{b['name']:20s} {str(b['files']):>6s}  {b['timestamp']}")
        print(f"\nTotal: {len(backups)} backups in {BACKUP_DIR}")
        return 0

    if args.prune:
        removed = prune_backups(max_age_days=args.max_age)
        print(f"Pruned {removed} old backups")
        return 0

    # Default: create backup
    backup_path = create_backup()
    print(f"Backup created: {backup_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
