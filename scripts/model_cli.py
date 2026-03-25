#!/usr/bin/env python3
"""
Model Manager CLI — Command-line interface for model A/B testing and promotion.

Usage:
    python scripts/model_cli.py register <version_tag> <model_path> [--accuracy FLOAT] [--candidate]
    python scripts/model_cli.py status
    python scripts/model_cli.py promote <version_tag>
    python scripts/model_cli.py rollback <version_tag>
    python scripts/model_cli.py history [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scanner.automation.model_manager import ModelManager


def cmd_register(args):
    """Register a new model version."""
    mm = ModelManager()

    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"✗ Error: Model path does not exist: {args.model_path}", file=sys.stderr)
        return 1

    # Register version
    version = mm.register_version(
        version_tag=args.version_tag,
        model_path=str(model_path.absolute()),
        accuracy=args.accuracy or 0.0,
        is_candidate=args.candidate,
        metadata={"registered_by": "model_cli"},
    )

    if version:
        tag_type = "[CANDIDATE]" if args.candidate else "[PRODUCTION]"
        print(f"✓ Registered {tag_type} {args.version_tag}")
        print(f"  Path: {version.model_path}")
        print(f"  Accuracy: {version.accuracy:.4f}")
        print(f"  Registered at: {version.registered_at}")
        return 0
    else:
        print(f"✗ Failed to register {args.version_tag}", file=sys.stderr)
        return 1


def cmd_status(args):
    """Show current model status."""
    mm = ModelManager()
    status = mm.get_status()

    print("\n" + "=" * 70)
    print("MODEL A/B TEST STATUS")
    print("=" * 70)

    print(f"\nTotal Versions Registered: {status['total_versions']}")

    if status["production"]:
        prod = status["production"]
        print(f"\n[PRODUCTION]")
        print(f"  Version: {prod['version_tag']}")
        print(f"  Validation Accuracy: {prod['accuracy']:.4f}")
        print(f"  Shadow Test Accuracy: {prod['shadow_accuracy']:.4f}")
        print(f"  Shadow Tests: {prod['shadow_scans']}")
        print(f"  Registered: {prod['registered_at']}")

    if status["candidate"]:
        cand = status["candidate"]
        progress = status.get("shadow_test_progress", {})

        print(f"\n[CANDIDATE]")
        print(f"  Version: {cand['version_tag']}")
        print(f"  Validation Accuracy: {cand['accuracy']:.4f}")
        print(f"  Shadow Test Accuracy: {cand['shadow_accuracy']:.4f}")
        print(f"  Shadow Tests: {cand['shadow_scans']}")
        print(f"  Registered: {cand['registered_at']}")

        if progress:
            print(f"\n[SHADOW TEST PROGRESS]")
            print(f"  Scans Completed: {progress['scans_completed']}/{progress['scans_required']}")
            print(f"  Progress: {progress['progress_pct']:.1f}%")
            print(f"  Improvement vs Incumbent: {progress['improvement_vs_incumbent']:+.4f}")
            print(f"  Improvement Threshold: {progress['improvement_threshold']:.4f}")
            print(f"  Ready for Promotion: {'YES ✓' if progress['promotion_ready'] else 'NO'}")
    else:
        print(f"\n[CANDIDATE] None — no candidate model registered for shadow testing")

    print("\n" + "=" * 70 + "\n")
    return 0


def cmd_promote(args):
    """Promote a candidate model to production."""
    mm = ModelManager()

    if args.version_tag not in mm._versions:
        print(f"✗ Version not found: {args.version_tag}", file=sys.stderr)
        return 1

    print(f"\nPromoting {args.version_tag} to production...")

    if mm.promote(args.version_tag):
        status = mm.get_status()
        prod = status["production"]
        print(f"✓ Successfully promoted {args.version_tag}")
        print(f"  New production accuracy: {prod['shadow_accuracy']:.4f}")
        return 0
    else:
        print(f"✗ Promotion failed", file=sys.stderr)
        return 1


def cmd_rollback(args):
    """Rollback to a previous model version."""
    mm = ModelManager()

    if args.version_tag not in mm._versions:
        print(f"✗ Version not found: {args.version_tag}", file=sys.stderr)
        return 1

    print(f"\nRolling back to {args.version_tag}...")

    if mm.rollback(args.version_tag):
        print(f"✓ Successfully rolled back to {args.version_tag}")
        return 0
    else:
        print(f"✗ Rollback failed", file=sys.stderr)
        return 1


def cmd_history(args):
    """Show version history."""
    mm = ModelManager()

    history_path = mm.version_history_path
    if not history_path.exists():
        print("No version history found")
        return 0

    print(f"\nVersion History ({history_path}):\n")

    entries = []
    try:
        with open(history_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading history: {e}", file=sys.stderr)
        return 1

    # Show most recent entries
    limit = args.limit or 20
    recent = entries[-limit:]

    for entry in reversed(recent):
        action = entry.get("action", "unknown")
        timestamp = entry.get("logged_at") or entry.get("registered_at", "?")

        if action == "registered":
            version = entry.get("version_tag")
            is_cand = entry.get("is_candidate", False)
            tag = "[CANDIDATE]" if is_cand else "[VERSION]"
            accuracy = entry.get("accuracy", 0)
            print(f"  {tag} {version:<20} accuracy={accuracy:.4f}")

        elif action == "promotion":
            cand = entry.get("candidate_version")
            cand_acc = entry.get("candidate_accuracy", 0)
            inc_acc = entry.get("incumbent_accuracy", 0)
            improvement = cand_acc - inc_acc
            print(f"  [PROMOTION] → {cand:<20} improvement={improvement:+.4f}")

        elif action == "rollback":
            version = entry.get("rollback_to_version")
            print(f"  [ROLLBACK] ← {version}")

        elif action == "shadow_test":
            pair = entry.get("pair")
            cand_correct = entry.get("candidate_correct")
            inc_correct = entry.get("incumbent_correct")
            result = "CAND WIN" if (cand_correct and not inc_correct) else "INC WIN" if (inc_correct and not cand_correct) else "BOTH OK" if (cand_correct and inc_correct) else "BOTH FAIL"
            print(f"  [SHADOW] {pair:<10} {result}")

        elif action in ("backup", "backup_before_rollback"):
            version = entry.get("version_tag")
            print(f"  [BACKUP]   {version}")

    print(f"\nShowing {len(recent)} of {len(entries)} total entries")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Model Manager CLI — Manage model versions and A/B testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/model_cli.py register v1.1.0 trained_data/models/buddy_tf_candidate.keras --candidate
  python scripts/model_cli.py status
  python scripts/model_cli.py promote v1.1.0
  python scripts/model_cli.py history --limit 30
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True

    # register command
    reg_parser = subparsers.add_parser("register", help="Register a new model version")
    reg_parser.add_argument("version_tag", help="Version tag (e.g., v1.1.0)")
    reg_parser.add_argument("model_path", help="Path to model file or directory")
    reg_parser.add_argument("--accuracy", type=float, help="Validation accuracy (0-1)")
    reg_parser.add_argument("--candidate", action="store_true", help="Mark as candidate for shadow testing")
    reg_parser.set_defaults(func=cmd_register)

    # status command
    status_parser = subparsers.add_parser("status", help="Show current model status")
    status_parser.set_defaults(func=cmd_status)

    # promote command
    promote_parser = subparsers.add_parser("promote", help="Promote candidate to production")
    promote_parser.add_argument("version_tag", help="Version tag to promote")
    promote_parser.set_defaults(func=cmd_promote)

    # rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to previous version")
    rollback_parser.add_argument("version_tag", help="Version tag to rollback to")
    rollback_parser.set_defaults(func=cmd_rollback)

    # history command
    history_parser = subparsers.add_parser("history", help="Show version history")
    history_parser.add_argument("--limit", type=int, default=20, help="Number of entries to show (default: 20)")
    history_parser.set_defaults(func=cmd_history)

    args = parser.parse_args()

    try:
        return args.func(args)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
