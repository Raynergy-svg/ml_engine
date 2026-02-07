"""Main entry point for CLI module.

This module provides the main() function that serves as the entry point
for the ML Engine Trading Bot CLI. It handles argument parsing and
dispatches to the appropriate command handlers.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Main CLI entry point.

    This is a temporary stub that delegates to the original main.py
    while the full CLI module is being built out.
    """
    # During the migration, delegate to original main.py
    # This allows incremental migration without breaking existing functionality
    try:
        # Import original main module
        import main as original_main

        original_main.main()
    except ImportError:
        print("Error: Could not import main module", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
