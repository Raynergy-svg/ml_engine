#!/usr/bin/env python3
"""
Manual drift checking utility.

Allows you to query the current state of drift detection without running a full scan.
Useful for debugging or manual monitoring between scans.

Usage:
    python scripts/manual_drift_check.py              # Show current stats
    python scripts/manual_drift_check.py --check      # Check for drift and optionally trigger
    python scripts/manual_drift_check.py --stats      # Show detailed statistics
    python scripts/manual_drift_check.py --clear      # Clear drift request file
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Manual drift detection checker",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check for drift based on trade journal",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed accuracy statistics",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the pending retrain request",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.52,
        help="Per-pair accuracy threshold (default: 0.52)",
    )
    parser.add_argument(
        "--global-threshold",
        type=float,
        default=0.53,
        help="Global accuracy threshold (default: 0.53)",
    )

    args = parser.parse_args()

    if args.clear:
        clear_request()
        return 0

    # Load trade journal
    journal_path = PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"
    if not journal_path.exists():
        logger.error(f"Trade journal not found: {journal_path}")
        return 1

    try:
        entries = json.loads(journal_path.read_text())
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse trade journal: {e}")
        return 1

    # Extract closed trades
    closed = [e for e in entries if e.get("outcome") is not None]
    if not closed:
        logger.warning("No closed trades in journal yet")
        return 0

    # Calculate per-pair accuracy
    pair_accuracy = {}
    for entry in closed:
        pair = entry.get("pair")
        correct = entry.get("outcome", {}).get("prediction_correct", False)

        if pair not in pair_accuracy:
            pair_accuracy[pair] = {"correct": 0, "total": 0}

        pair_accuracy[pair]["total"] += 1
        if correct:
            pair_accuracy[pair]["correct"] += 1

    # Calculate rolling accuracy (last 50 per pair)
    pair_rolling = {}
    for entry in closed[-50:]:  # Last 50 trades overall for rolling global
        pair = entry.get("pair")
        correct = entry.get("outcome", {}).get("prediction_correct", False)

        if pair not in pair_rolling:
            pair_rolling[pair] = {"correct": 0, "total": 0}

        pair_rolling[pair]["total"] += 1
        if correct:
            pair_rolling[pair]["correct"] += 1

    # Global accuracy
    global_correct = sum(
        1 for e in closed if e.get("outcome", {}).get("prediction_correct")
    )
    global_total = len(closed)
    global_accuracy = global_correct / global_total if global_total > 0 else 0

    # Display report
    logger.info("\n" + "=" * 80)
    logger.info(f"DRIFT CHECK REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    logger.info(f"\nGlobal Stats:")
    logger.info(f"  Total trades: {global_total}")
    logger.info(f"  Correct predictions: {global_correct}")
    logger.info(f"  Global accuracy: {global_accuracy:.1%}")
    logger.info(f"  Threshold: {args.global_threshold:.1%}")

    if global_accuracy < args.global_threshold:
        logger.info(f"  ⚠️  BELOW THRESHOLD — global drift detected!")
    else:
        logger.info(f"  ✓ Above threshold")

    logger.info(f"\nPer-Pair Accuracy (all-time):")
    logger.info(f"  {'Pair':12} {'Accuracy':>12} {'Count':>8} {'Status'}")
    logger.info(f"  {'-' * 50}")

    pairs_below = []
    for pair in sorted(pair_accuracy.keys()):
        data = pair_accuracy[pair]
        acc = data["correct"] / data["total"]
        status = "✓ OK" if acc >= args.threshold else "⚠️  BELOW"

        logger.info(
            f"  {pair:12} {acc:>11.1%} {data['total']:>8} {status}"
        )

        if acc < args.threshold:
            pairs_below.append(pair)

    if args.stats:
        logger.info(f"\nDetailed Statistics:")
        logger.info(f"  Minimum pair count for evaluation: 10 trades")
        logger.info(f"  Per-pair threshold: {args.threshold:.1%}")
        logger.info(f"  Global threshold: {args.global_threshold:.1%}")

    logger.info(f"\n{'=' * 80}")

    if pairs_below:
        logger.info(f"\n⚠️  {len(pairs_below)} pair(s) below threshold:")
        for pair in pairs_below:
            logger.info(f"   - {pair}")

    if args.check:
        logger.info(f"\nDrift Check:")
        if global_accuracy < args.global_threshold:
            logger.info(
                f"  ⚠️  GLOBAL DRIFT DETECTED (acc={global_accuracy:.1%} < {args.global_threshold:.1%})"
            )
            logger.info(f"  Action: Would trigger full retrain of all pairs")
        elif pairs_below:
            logger.info(
                f"  ⚠️  PAIR DRIFT DETECTED ({len(pairs_below)} pairs below threshold)"
            )
            logger.info(f"  Action: Would trigger retrain of {pairs_below}")
        else:
            logger.info(f"  ✓ No drift detected")

        # Check if request file exists
        request_file = PROJECT_ROOT / "trained_data" / "retrain_request.json"
        if request_file.exists():
            logger.info(f"\n⚠️  Pending retrain request exists:")
            try:
                request = json.loads(request_file.read_text())
                logger.info(f"  Pairs: {request.get('pairs', [])}")
                logger.info(f"  Priority: {request.get('priority')}")
                logger.info(f"  Reason: {request.get('reason')}")
                logger.info(f"  Triggered at: {request.get('triggered_at')}")
            except Exception as e:
                logger.error(f"  Failed to read request: {e}")
        else:
            logger.info(f"\n✓ No pending retrain request")

    logger.info("=" * 80)
    return 0


def clear_request():
    """Clear the pending retrain request."""
    request_file = PROJECT_ROOT / "trained_data" / "retrain_request.json"

    if not request_file.exists():
        logger.info("No retrain request file found")
        return

    try:
        request_file.unlink()
        logger.info(f"✓ Cleared retrain request: {request_file}")
    except Exception as e:
        logger.error(f"Failed to clear request: {e}")


if __name__ == "__main__":
    sys.exit(main())
