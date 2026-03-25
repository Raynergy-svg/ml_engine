"""Aura CLI entry point.

Usage:
    python -m src.aura.cli.main              # Interactive mode
    python -m src.aura.cli.main --demo       # Run the demo scenario (Maya career decision)
    python -m src.aura.cli.main --status     # Show bridge status and exit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.aura.cli.companion import AuraCompanion, CYAN, GREEN, YELLOW, RED, DIM, BOLD, RESET

logger = logging.getLogger(__name__)


def run_interactive(companion: AuraCompanion) -> None:
    """Run the interactive conversation loop."""
    companion.start_session()

    while True:
        try:
            user_input = input(f"{GREEN}You: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        response = companion.process_input(user_input)

        if response == "__QUIT__":
            break

        print(f"{CYAN}Aura: {RESET}{response}")
        print()

    companion.end_session()


def run_demo(companion: AuraCompanion) -> None:
    """Run the Maya career decision demo scenario from PRD v2.2 §19.

    This demonstrates the feedback bridge in action:
    1. Maya talks to Aura about career stress
    2. Aura detects elevated stress → readiness drops
    3. Buddy reads low readiness → reduces position sizes
    4. Maya overrides Buddy → Aura correlates override with stress
    5. Outcome: loss → both systems learn
    """
    companion.start_session()

    demo_messages = [
        "I've been really stressed about the career decision. Can't stop thinking about whether to go independent.",
        "I didn't sleep well last night. The anxiety about leaving my job keeps me up.",
        "I saw a good setup on EUR/USD and took the trade anyway even though Buddy said no. Just felt like I needed a win.",
        "The trade went against me. Lost 40 pips. I knew Buddy was right but I overrode it.",
        "Maybe I should step back from trading when I'm this stressed about the career thing.",
    ]

    print(f"{BOLD}═══ Demo: Maya Career Decision Scenario ═══{RESET}")
    print(f"{DIM}(From PRD v2.2 §19 — demonstrating the feedback bridge){RESET}\n")

    for i, msg in enumerate(demo_messages, 1):
        print(f"{DIM}--- Message {i}/{len(demo_messages)} ---{RESET}")
        print(f"{GREEN}Maya: {RESET}{msg}")
        print()

        response = companion.process_input(msg)
        print(f"{CYAN}Aura: {RESET}{response}")
        print()

        # Show readiness after each message
        readiness_response = companion.process_input("/readiness")
        print(f"{DIM}{readiness_response}{RESET}")
        print()

    # Show final bridge and graph status
    print(f"\n{BOLD}═══ Final Status ═══{RESET}")
    print(companion.process_input("/bridge"))
    print()
    print(companion.process_input("/graph"))

    companion.end_session()


def main():
    parser = argparse.ArgumentParser(description="Aura — Human Intelligence Engine")
    parser.add_argument("--demo", action="store_true", help="Run the Maya demo scenario")
    parser.add_argument("--status", action="store_true", help="Show bridge status and exit")
    parser.add_argument("--db-path", type=str, default=None, help="Path to self-model database")
    parser.add_argument("--bridge-dir", type=str, default=None, help="Path to bridge signal directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    db_path = Path(args.db_path) if args.db_path else None
    bridge_dir = Path(args.bridge_dir) if args.bridge_dir else None

    companion = AuraCompanion(db_path=db_path, bridge_dir=bridge_dir)

    if args.status:
        status = companion.bridge.get_bridge_status()
        import json
        print(json.dumps(status, indent=2, default=str))
        return

    if args.demo:
        run_demo(companion)
    else:
        run_interactive(companion)


if __name__ == "__main__":
    main()
