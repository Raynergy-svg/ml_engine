"""
Scanner Display Module.

Rich-based display for scanner output with Live updates.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.box import SIMPLE_HEAVY
from rich.text import Text

from .results import PairAnalysis, ScanResult

logger = logging.getLogger(__name__)


class ScannerDisplay:
    """Rich-based display for scanner output.

    Features:
    - Real-time Live display updates
    - Color-coded gate status
    - Account info header
    - Correlation warnings
    """

    def __init__(self, console: Optional[Console] = None):
        """Initialize display.

        Args:
            console: Rich Console instance (creates new if None)
        """
        self.console = console or Console()
        self._live: Optional[Live] = None
        self._current_analyses: List[PairAnalysis] = []
        self._account_info: Dict[str, Any] = {}
        self._model_type: str = "unknown"
        self._granularity: str = "H1"
        self._scan_start_time: Optional[datetime] = None

    def _format_direction(self, direction: str, confidence: float) -> Text:
        """Format direction with color based on confidence."""
        if direction == "LONG":
            color = "green" if confidence >= 0.6 else "yellow"
            return Text(f"▲ {direction}", style=f"bold {color}")
        elif direction == "SHORT":
            color = "red" if confidence >= 0.6 else "yellow"
            return Text(f"▼ {direction}", style=f"bold {color}")
        else:
            return Text("● HOLD", style="dim")

    def _format_confidence(self, confidence: float) -> Text:
        """Format confidence percentage with color."""
        pct = confidence * 100
        if pct >= 70:
            return Text(f"{pct:.0f}%", style="bold green")
        elif pct >= 50:
            return Text(f"{pct:.0f}%", style="yellow")
        else:
            return Text(f"{pct:.0f}%", style="dim")

    def _format_gate(self, passed: bool, value: Optional[float] = None) -> Text:
        """Format gate status."""
        if passed:
            if value is not None:
                return Text(f"\u2713{value:.0f}" if value >= 1 else f"\u2713{value:.2f}", style="green")
            return Text("\u2713", style="green")
        else:
            if value is not None:
                return Text(f"\u2717{value:.0f}" if value >= 1 else f"\u2717{value:.2f}", style="red")
            return Text("✗", style="red")

    def _format_gate_compact(self, passed: bool) -> Text:
        """Format gate as single checkmark/cross."""
        if passed:
            return Text("\u2713", style="green")
        return Text("\u2717", style="red")

    def _format_error(self, error: str) -> Text:
        """Format error message."""
        return Text(f"⚠ {error[:30]}", style="dim red")

    def generate_table(self) -> Table:
        """Generate results table for Live display.

        Returns:
            Rich Table with current scan results
        """
        table = Table(
            title=None,
            show_header=True,
            header_style="bold cyan",
            expand=False,
            padding=(0, 1),
            box=SIMPLE_HEAVY,
        )

        # Columns – compact layout for 80-char terminals
        # Total: 7+1+7+1+4+1+3+1+3+1+3+1+3+1+9+1+15 = ~62 data + separators
        table.add_column("Pair", style="bold", no_wrap=True)
        table.add_column("Dir", justify="center", no_wrap=True)
        table.add_column("Conf", justify="right", no_wrap=True)
        table.add_column("M", justify="center", no_wrap=True)  # Momentum
        table.add_column("A", justify="center", no_wrap=True)  # ADX
        table.add_column("R", justify="center", no_wrap=True)  # Risk
        table.add_column("G", justify="center", no_wrap=True)  # Gates
        table.add_column("Price", justify="right", no_wrap=True)
        table.add_column("Note")

        # Sort analyses: tradeable first, then by confidence
        sorted_analyses = sorted(
            self._current_analyses,
            key=lambda x: (x.is_tradeable, x.confidence),
            reverse=True,
        )

        for analysis in sorted_analyses:
            # Handle hard errors (no data at all)
            if analysis.error and analysis.current_price < 0.0001:
                table.add_row(
                    analysis.pair.replace("_", "/"),
                    Text("-", style="dim"),
                    Text("-", style="dim"),
                    Text("-", style="dim"),
                    Text("-", style="dim"),
                    Text("-", style="dim"),
                    Text("-", style="dim"),
                    Text("-", style="dim"),
                    self._format_error(analysis.error),
                )
                continue

            # Format each column
            pair_text = analysis.pair.replace("_", "/")
            if analysis.gates_passed:
                pair_text = f"★ {pair_text}"

            # Gates summary
            gates_text = analysis.gate_summary
            gates_style = "green bold" if analysis.gates_passed else "dim"

            # Note (warnings or trade suggestion)
            note = ""
            if analysis.error:
                note = analysis.error
            elif analysis.gates_passed:
                note = f"SL:{analysis.sl_pips:.0f} TP:{analysis.tp_pips:.0f}"
            elif not analysis.momentum_passed:
                note = "low momentum"
            elif not analysis.confidence_passed:
                note = "low ADX"
            elif not analysis.risk_passed:
                note = "high risk"

            table.add_row(
                pair_text,
                self._format_direction(analysis.direction, analysis.confidence),
                self._format_confidence(analysis.confidence),
                self._format_gate_compact(analysis.momentum_passed),
                self._format_gate_compact(analysis.confidence_passed),
                self._format_gate_compact(analysis.risk_passed),
                Text(gates_text, style=gates_style),
                f"{analysis.current_price:.4f}" if analysis.current_price else "-",
                note,
            )

        return table

    def generate_header(self) -> Panel:
        """Generate header panel with account info."""
        # Account info
        nav = self._account_info.get("nav", 0)
        open_trades = self._account_info.get("open_trades", 0)
        unrealized_pl = self._account_info.get("unrealized_pl", 0)

        # Build header text
        header_parts = [
            "[bold cyan]📡 BUDDY SCANNER[/bold cyan]",
            f"[green]${nav:,.0f}[/green]" if nav > 0 else "",
            f"[dim]{open_trades} trades[/dim]" if open_trades > 0 else "",
        ]

        if unrealized_pl != 0:
            pl_color = "green" if unrealized_pl > 0 else "red"
            header_parts.append(f"[{pl_color}]P/L: ${unrealized_pl:+,.2f}[/{pl_color}]")

        header_parts.append(f"[dim]{self._model_type} | {self._granularity}[/dim]")

        header = " | ".join([p for p in header_parts if p])

        return Panel(header, border_style="cyan")

    def show_scanning_progress(self, pairs: List[str]) -> Progress:
        """Create progress display for scanning phase.

        Args:
            pairs: List of pairs being scanned

        Returns:
            Rich Progress instance
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        )

    def start_live(self) -> Live:
        """Start Live display for incremental updates.

        Returns:
            Rich Live instance
        """
        self._scan_start_time = datetime.now(timezone.utc)
        self._live = Live(
            self.generate_table(),
            console=self.console,
            refresh_per_second=4,
        )
        self._live.start()
        return self._live

    def update_live(self, analysis: PairAnalysis) -> None:
        """Update Live display with new analysis.

        Args:
            analysis: New or updated pair analysis
        """
        # Update or add analysis
        for i, existing in enumerate(self._current_analyses):
            if existing.pair == analysis.pair:
                self._current_analyses[i] = analysis
                break
        else:
            self._current_analyses.append(analysis)

        # Update Live display
        if self._live is not None:
            self._live.update(self.generate_table())

    def stop_live(self) -> None:
        """Stop Live display."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def show_result(
        self,
        result: ScanResult,
        account_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Display complete scan result.

        Args:
            result: Scan result with all analyses
            account_info: Optional account information
        """
        self._current_analyses = result.analyses
        self._model_type = result.model_type
        self._granularity = result.granularity

        if account_info:
            self._account_info = account_info

        # Print header
        self.console.print()
        self.console.print(self.generate_header())
        self.console.print()

        # Print table
        self.console.print(self.generate_table())

        # Summary
        tradeable = result.tradeable_pairs
        if tradeable:
            self.console.print()
            self.console.print(
                f"[bold green]✓ {len(tradeable)} tradeable:[/bold green] "
                f"{', '.join(p.pair.replace('_', '/') for p in tradeable)}"
            )
        else:
            self.console.print()
            self.console.print("[dim]No tradeable opportunities found[/dim]")

        # Show warnings for common issues
        error_pairs = [a for a in result.analyses if a.error]
        no_data = [a for a in error_pairs if "No data" in (a.error or "")]
        no_models = [a for a in error_pairs if "No models" in (a.error or "")]

        if no_data:
            self.console.print(
                "[yellow]⚠ Some pairs have no data. Check OANDA credentials "
                "or place CSV files in market_data/[/yellow]"
            )
        if no_models:
            self.console.print(
                "[yellow]⚠ Models not loaded. Ensure you're in the correct conda env "
                "(tf-metal / intel) with tensorflow, xgboost, etc. installed[/yellow]"
            )

        # Scan metadata
        self.console.print()
        self.console.print(
            f"[dim]Scanned {len(result.analyses)} pairs "
            f"at {result.scan_time.strftime('%H:%M:%S')} UTC[/dim]"
        )

    def show_session_warning(self, current_hour: int, session_range: str) -> bool:
        """Show session timing warning and prompt.

        Args:
            current_hour: Current UTC hour
            session_range: Active session range string

        Returns:
            True if user wants to continue, False otherwise
        """
        self.console.print()
        self.console.print("[bold yellow]⚠️  OUTSIDE OPTIMAL TRADING HOURS[/bold yellow]")
        self.console.print(f"[yellow]Current time: {current_hour}:00 UTC[/yellow]")
        self.console.print(f"[dim]Best hours: {session_range} (London/NY overlap)[/dim]")
        self.console.print()

        # Non-interactive mode
        try:
            response = self.console.input("[yellow]Continue anyway? [y/N]: [/yellow]")
            return response.strip().lower() in ('y', 'yes')
        except (EOFError, KeyboardInterrupt):
            return False

    def show_error(self, message: str) -> None:
        """Display error message.

        Args:
            message: Error message to display
        """
        self.console.print(f"[bold red]Error:[/bold red] {message}")

    def show_success(self, message: str) -> None:
        """Display success message.

        Args:
            message: Success message to display
        """
        self.console.print(f"[bold green]✓[/bold green] {message}")
