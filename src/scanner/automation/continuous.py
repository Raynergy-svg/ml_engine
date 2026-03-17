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

                # Auto-execute if enabled (use is_tradeable not gates_passed)
                if auto_execute:
                    tradeable = [a for a in result.analyses if a.is_tradeable]
                    tradeable = self._filter_correlated_exposure(tradeable)
                    if tradeable:
                        if console:
                            console.print(f"\n[green]Auto-executing {len(tradeable)} trade(s)...[/green]")
                        self.scanner.execute_trades(
                            analyses=tradeable,
                        )

                # Log scan cycle for analytics
                self._log_scan_cycle(result, auto_execute)

                # Smart trading loop: monitor, drawdown guardian, RL sync
                self._run_smart_loop()

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

    def _log_scan_cycle(
        self,
        result: "ScanResult",
        auto_execute: bool,
    ) -> None:
        """Append a lightweight record of this scan cycle for post-hoc analytics."""
        import json
        from pathlib import Path

        log_path = Path("trained_data/scan_cycle_log.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        tradeable = [a for a in result.analyses if a.is_tradeable]
        record = {
            "timestamp": datetime.now().isoformat(),
            "scan_number": self._scan_count,
            "pairs_scanned": len(result.analyses),
            "tradeable_count": len(tradeable),
            "tradeable_pairs": [a.pair for a in tradeable],
            "auto_execute": auto_execute,
            "top_score": round(max((a.overall_score for a in result.analyses), default=0), 4),
            "model_type": result.model_type,
        }

        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.debug(f"Scan cycle log error: {e}")

    def _filter_correlated_exposure(self, tradeable: list) -> list:
        """Filter out trades that would double exposure on correlated pairs already open."""
        try:
            from src.scanner.execution import ExecutionManager
            from src.training.correlation_group_config import get_correlation_group

            em = ExecutionManager()
            open_trades = em.monitor_open_trades()
            if not open_trades:
                return tradeable

            # Build set of correlation groups currently exposed
            open_groups: set = set()
            open_pairs: set = set()
            for t in open_trades:
                pair = t.get("pair", "")
                open_pairs.add(pair)
                group = get_correlation_group(pair)
                if group and group.master_pair:
                    open_groups.add(group.master_pair)

            filtered = []
            for a in tradeable:
                if a.pair in open_pairs:
                    if console:
                        console.print(f"  [dim]Skipping {a.pair}: already open[/dim]")
                    continue
                group = get_correlation_group(a.pair)
                if group and group.master_pair in open_groups:
                    if console:
                        console.print(
                            f"  [dim]Skipping {a.pair}: correlated with open "
                            f"{group.master_pair} group[/dim]"
                        )
                    continue
                filtered.append(a)

            return filtered

        except Exception as e:
            logger.debug(f"Correlation filter error: {e}")
            return tradeable

    def _run_smart_loop(self) -> None:
        """Run the smart trading loop: monitor open trades, drawdown guardian, RL sync."""
        try:
            from src.scanner.execution import ExecutionManager

            em = ExecutionManager()

            # 1. Monitor open trades
            statuses = em.monitor_open_trades()
            if statuses and console:
                console.print(f"\n[dim]Open trades: {len(statuses)}[/dim]")
                for s in statuses:
                    pl_color = "green" if s["unrealized_pl"] >= 0 else "red"
                    console.print(
                        f"  [{pl_color}]{s['pair']} {s['direction']} "
                        f"P/L ${s['unrealized_pl']:.2f} "
                        f"(SL {s['sl_dist_pips']:.0f}p / TP {s['tp_dist_pips']:.0f}p, "
                        f"{s['time_in_minutes']}m)[/{pl_color}]"
                    )

            # 2. Drawdown guardian
            modifications = em.apply_drawdown_guardian()
            if modifications and console:
                for mod in modifications:
                    console.print(f"  [yellow]Guardian: {mod}[/yellow]")

            # 3. RL feedback sync
            rl_result = em.sync_closed_trades_rl()
            if rl_result.get("trades_synced", 0) > 0 and console:
                console.print(
                    f"  [cyan]RL sync: {rl_result['detail']}[/cyan]"
                )
                if rl_result.get("new_weights"):
                    for agent, w in rl_result["new_weights"].items():
                        console.print(f"    {agent}: {w:.3f}")

            # 4. Agent weight decay toward baseline
            try:
                from src.scanner.agents import ScannerAgentTeam
                team = ScannerAgentTeam(self.scanner.config)
                decayed = team.apply_weight_decay(decay_rate=0.02)
                if decayed and console:
                    console.print(f"  [dim]Weight decay applied ({len(decayed)} agents)[/dim]")
            except Exception as decay_err:
                logger.debug(f"Weight decay error: {decay_err}")

        except Exception as e:
            logger.debug(f"Smart loop error: {e}")

    def _sleep_with_progress(self, minutes: int):
        """Sleep with progress indication, allowing Ctrl+C to interrupt."""
        if console:
            console.print(f"\n[dim]Next scan in {minutes} minutes...[/dim]")

        # Sleep in 1-second increments to allow Ctrl+C
        for _ in range(minutes * 60):
            if not self._running:
                break
            time.sleep(1)
