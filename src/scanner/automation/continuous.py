"""
Continuous Scanner - Run scans in a loop with configurable intervals.

Provides watch mode functionality for real-time market monitoring.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from src.scanner.engine import Scanner
    from src.scanner.results import ScanResult

try:
    from rich.console import Console
    console = Console()
except ImportError:
    console = None

logger = logging.getLogger(__name__)


@dataclass
class ContinuousConfig:
    """Configuration for continuous scanning."""

    interval_minutes: int = 5
    auto_execute: bool = False
    top_n: int = 5
    pairs: Optional[List[str]] = None
    granularity: str = "H1"
    enable_maintenance: bool = True
    max_scans: Optional[int] = None  # None = unlimited


class ContinuousScanner:
    """
    Run scans in a loop with configurable intervals.

    Features:
    - Graceful shutdown on Ctrl+C
    - Idle-time maintenance (gate retraining, journal sync)
    - Auto-execution of passing trades
    - Progress indication between scans

    Example:
        >>> scanner = Scanner(config)
        >>> continuous = ContinuousScanner(scanner)
        >>> continuous.run(interval_minutes=5, auto_execute=True)
    """

    def __init__(
        self,
        scanner: "Scanner",
        config: Optional[ContinuousConfig] = None,
    ):
        self.scanner = scanner
        self.config = config or ContinuousConfig()
        self._running = False
        self._scan_count = 0
        self._maintenance = None

        # Initialize maintenance if enabled
        if self.config.enable_maintenance:
            from src.scanner.automation.maintenance import IdleMaintenance
            self._maintenance = IdleMaintenance()

    def run(
        self,
        pairs: Optional[List[str]] = None,
        granularity: Optional[str] = None,
        interval_minutes: Optional[int] = None,
        auto_execute: bool = False,
        top_n: Optional[int] = None,
        on_scan_complete: Optional[Callable[["ScanResult"], None]] = None,
    ) -> int:
        """
        Run continuous scanning loop.

        Args:
            pairs: List of pairs to scan (overrides config)
            granularity: Timeframe (overrides config)
            interval_minutes: Minutes between scans (overrides config)
            auto_execute: Auto-execute passing trades (overrides config)
            top_n: Number of top results (overrides config)
            on_scan_complete: Callback after each scan

        Returns:
            Number of scans completed
        """
        # Apply overrides
        pairs = pairs or self.config.pairs
        granularity = granularity or self.config.granularity
        interval_minutes = interval_minutes or self.config.interval_minutes
        top_n = top_n or self.config.top_n

        # Setup signal handler for graceful shutdown
        self._running = True
        self._setup_signal_handler()

        self._scan_count = 0

        if console:
            console.print("\n[bold cyan]🔄 CONTINUOUS SCAN MODE[/bold cyan]")
            console.print(f"[dim]Scanning every {interval_minutes} minutes. Press Ctrl+C to stop.[/dim]")
            console.print("=" * 70)

        while self._running:
            self._scan_count += 1

            # Check max scans limit
            if self.config.max_scans and self._scan_count > self.config.max_scans:
                if console:
                    console.print(f"\n[yellow]Max scans ({self.config.max_scans}) reached[/yellow]")
                break

            try:
                if console:
                    now = datetime.now().strftime("%H:%M:%S")
                    console.print(f"\n[bold]── Scan #{self._scan_count} at {now} ──[/bold]")

                # Run scan
                result = self.scanner.scan(
                    pairs=pairs,
                    max_workers=4,
                )

                # Display results
                from src.scanner.display import ScannerDisplay
                display = ScannerDisplay()
                account_info = self.scanner.get_account_info()
                display.show_result(result, account_info=account_info)

                # Callback if provided
                if on_scan_complete:
                    on_scan_complete(result)

                # Auto-execute if enabled
                if auto_execute:
                    tradeable = [a for a in result.analyses if a.gates_passed]
                    if tradeable:
                        if console:
                            console.print(f"\n[green]Auto-executing {len(tradeable)} trade(s)...[/green]")
                        self.scanner.execute_trades(
                            analyses=tradeable,
                        )

                # Idle maintenance
                if self._maintenance:
                    self._maintenance.run_if_needed()

            except Exception as e:
                logger.error(f"Scan error: {e}")
                if console:
                    console.print(f"[red]Scan error: {e}[/red]")

            if self._running:
                self._sleep_with_progress(interval_minutes)

        if console:
            console.print(f"\n[green]✓ Continuous scan stopped after {self._scan_count} scans[/green]")

        return self._scan_count

    def stop(self):
        """Stop the continuous scan loop."""
        self._running = False
        if console:
            console.print("\n[yellow]Stopping continuous scan...[/yellow]")

    def _setup_signal_handler(self):
        """Setup Ctrl+C handler for graceful shutdown."""
        def handler(sig, frame):
            self.stop()

        signal.signal(signal.SIGINT, handler)

    def _sleep_with_progress(self, minutes: int):
        """Sleep with progress indication, allowing Ctrl+C to interrupt."""
        if console:
            console.print(f"\n[dim]Next scan in {minutes} minutes...[/dim]")

        # Sleep in 1-second increments to allow Ctrl+C
        for _ in range(minutes * 60):
            if not self._running:
                break
            time.sleep(1)
