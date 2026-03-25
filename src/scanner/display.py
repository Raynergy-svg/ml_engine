"""
Scanner Display Module.

Rich-based display for scanner output with Live updates.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
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
        self._is_scanning: bool = False
        self._planner_orange = "color(215)"
        self._planner_sand = "color(223)"
        self._planner_cyan = "color(117)"
        self._planner_slate = "color(246)"

    def _format_direction(self, direction: str, confidence: float) -> Text:
        """Format direction with color based on confidence."""
        if direction == "LONG":
            color = self._planner_cyan if confidence >= 0.6 else self._planner_sand
            return Text(f"▲ {direction}", style=f"bold {color}")
        elif direction == "SHORT":
            color = self._planner_orange if confidence >= 0.6 else self._planner_sand
            return Text(f"▼ {direction}", style=f"bold {color}")
        else:
            return Text("● HOLD", style=self._planner_slate)

    def _format_confidence(self, confidence: float) -> Text:
        """Format confidence percentage with color."""
        pct = confidence * 100
        if pct >= 70:
            return Text(f"{pct:.0f}%", style=f"bold {self._planner_cyan}")
        elif pct >= 50:
            return Text(f"{pct:.0f}%", style=self._planner_sand)
        else:
            return Text(f"{pct:.0f}%", style=self._planner_slate)

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

    def _format_agent_compact(self, analysis: PairAnalysis) -> Text:
        """Format sub-inference agent consensus."""
        if analysis.agent_total <= 0:
            return Text("-", style=self._planner_slate)
        mark = "\u2713" if analysis.agent_passed else "\u2717"
        color = self._planner_cyan if analysis.agent_passed else self._planner_sand
        return Text(f"{mark}{analysis.agent_votes}/{analysis.agent_total}", style=color)

    def _format_pair_label(self, analysis: PairAnalysis) -> Text:
        """Format pair label with a compact visual badge."""
        pair = analysis.pair.replace("_", "/")
        text = Text()
        if analysis.error:
            text.append("! ", style=f"bold {self._planner_orange}")
        elif analysis.is_tradeable:
            text.append("● ", style=f"bold {self._planner_cyan}")
        elif analysis.gates_passed:
            text.append("◐ ", style=f"bold {self._planner_cyan}")
        elif analysis.direction in {"LONG", "SHORT"}:
            text.append("○ ", style=self._planner_sand)
        else:
            text.append("· ", style=self._planner_slate)
        text.append(pair, style=f"bold {self._planner_sand}")
        return text

    def _format_checks_compact(self, analysis: PairAnalysis) -> Text:
        """Format M/A/R gates as a compact labeled strip."""
        checks = Text()
        gate_states = [
            ("M", analysis.momentum_passed),
            ("A", analysis.confidence_passed),
            ("R", analysis.risk_passed),
        ]
        for idx, (label, passed) in enumerate(gate_states):
            style = f"bold {self._planner_cyan}" if passed else f"bold {self._planner_orange}"
            checks.append(label, style=style)
            checks.append("✓" if passed else "✗", style=style)
            if idx < len(gate_states) - 1:
                checks.append(" ", style=self._planner_slate)
        return checks

    def _format_status(self, analysis: PairAnalysis) -> Text:
        """Format overall scanner state for a pair."""
        if analysis.error:
            return Text("ERROR", style=f"bold {self._planner_orange}")
        if analysis.is_tradeable:
            if getattr(analysis, "agent_promoted", False):
                return Text("PROMOTED", style=f"bold {self._planner_cyan}")
            return Text("READY", style=f"bold {self._planner_cyan}")
        if analysis.gates_passed:
            return Text("FILTERED", style=f"bold {self._planner_sand}")
        if analysis.direction in {"LONG", "SHORT"}:
            return Text("WATCH", style=self._planner_sand)
        return Text("HOLD", style=self._planner_slate)

    def _format_price(self, analysis: PairAnalysis) -> str:
        """Format price with pair-appropriate precision."""
        if not analysis.current_price:
            return "-"
        decimals = 3 if analysis.pair.endswith("JPY") else 4
        return f"{analysis.current_price:.{decimals}f}"

    def _format_error(self, error: str) -> Text:
        """Format error message with full context."""
        return Text(f"⚠ {error[:80]}", style=self._planner_orange)

    def _generate_descriptive_note(self, analysis: PairAnalysis) -> str:
        """Generate a descriptive note based on analysis state.

        Provides specific, actionable notes explaining:
        - Why trade-ready pairs are attractive
        - What's preventing execution for watchlist pairs
        - Specific signals detected (momentum, trend, regime)

        Args:
            analysis: PairAnalysis with gate and signal data

        Returns:
            Descriptive note string
        """
        # Handle errors first
        if analysis.error:
            return analysis.error

        # === TRADE-READY PAIRS (gates_passed=True) ===
        if analysis.gates_passed:
            return self._generate_trade_ready_note(analysis)

        # === WATCHLIST PAIRS (gates_passed=False) ===
        return self._generate_watchlist_note(analysis)

    def _generate_trade_ready_note(self, analysis: PairAnalysis) -> str:
        """Generate note for trade-ready pairs (all gates passed)."""
        notes = []

        # Position sizing info
        notes.append(f"SL:{analysis.sl_pips:.0f} TP:{analysis.tp_pips:.0f}")

        # Signal strength description
        strength_desc = self._get_strength_description(analysis)
        if strength_desc:
            notes.append(strength_desc)

        # Agent consensus
        if analysis.agent_total > 0:
            agent_state = "confirmed" if analysis.agent_passed else "weak"
            notes.append(f"agent {agent_state} ({analysis.agent_votes}/{analysis.agent_total})")
        else:
            notes.append("agent n/a")

        if analysis.why_trade:
            notes.append(analysis.why_trade[0])

        # Execution path if promoted
        if getattr(analysis, "agent_promoted", False):
            source = (analysis.master_pair or analysis.pair).replace("_", "/")
            notes.append(f"exec via {source}")

        return " | ".join(notes)

    def _generate_watchlist_note(self, analysis: PairAnalysis) -> str:
        """Generate note for watchlist pairs (not trade-ready)."""
        if analysis.why_no_trade:
            return analysis.why_no_trade[0]
        if analysis.why_trade and analysis.direction in {"LONG", "SHORT"}:
            return analysis.why_trade[0]

        # Check agent-only passes first
        if analysis.agent_total > 0 and analysis.agent_passed:
            return f"agent confirmed ({analysis.agent_votes}/{analysis.agent_total})"
        if analysis.agent_total > 0 and not analysis.agent_passed:
            return f"agent weak ({analysis.agent_votes}/{analysis.agent_total})"

        # Check individual gate failures (M, A, R)
        if not analysis.momentum_passed:
            return self._momentum_failure_note(analysis)
        if not analysis.confidence_passed:
            return self._confidence_failure_note(analysis)
        if not analysis.risk_passed:
            return self._risk_failure_note(analysis)

        # All basic gates passed but gates_passed=False - check advanced filters
        if analysis.momentum_passed and analysis.confidence_passed and analysis.risk_passed:
            return self._advanced_filter_note(analysis)

        # Fallback for other cases
        if analysis.direction in {"LONG", "SHORT"} and analysis.confidence >= 0.50:
            return "watchlist setup"

        return "pending review"

    def _get_strength_description(self, analysis: PairAnalysis) -> str:
        """Get strength signal description for trade-ready pairs."""
        descriptions = []

        # Momentum acceleration signal
        if getattr(analysis, "momentum_acceleration", False):
            descriptions.append("momentum surge")
        elif analysis.momentum >= 0.6:
            descriptions.append("strong momentum")
        elif analysis.momentum >= 0.4:
            descriptions.append("trend aligned")

        # Confidence/trend strength
        ridge_conf = getattr(analysis, "ridge_confidence", 0)
        if ridge_conf >= 70:
            descriptions.append("strong trend")
        elif ridge_conf >= 55:
            descriptions.append("trend confirmed")

        # Directional clarity
        if analysis.confidence >= 0.65:
            descriptions.append("high conviction")
        elif analysis.confidence >= 0.55:
            descriptions.append("directional clarity")

        return descriptions[0] if descriptions else ""

    def _momentum_failure_note(self, analysis: PairAnalysis) -> str:
        """Generate note for momentum gate failure."""
        momentum = analysis.momentum

        if momentum < 0.15:
            return f"weak momentum ({momentum:.0%})"
        elif momentum < 0.20:
            return f"low momentum ({momentum:.0%})"
        else:
            return f"momentum below threshold ({momentum:.0%})"

    def _confidence_failure_note(self, analysis: PairAnalysis) -> str:
        """Generate note for confidence/ADX gate failure."""
        ridge_conf = getattr(analysis, "ridge_confidence", 0)

        if ridge_conf > 0 and ridge_conf < 40:
            return f"weak trend (ADX:{ridge_conf:.0f})"
        elif ridge_conf > 0 and ridge_conf < 50:
            return f"low ADX ({ridge_conf:.0f})"
        else:
            return "low ADX"

    def _risk_failure_note(self, analysis: PairAnalysis) -> str:
        """Generate note for risk gate failure."""
        drawdown = getattr(analysis, "drawdown", 0)
        rf_drawdown = getattr(analysis, "rf_drawdown", 0)
        risk_val = rf_drawdown if rf_drawdown > 0 else drawdown

        if risk_val > 0.05:
            return f"high risk (DD:{risk_val:.1%})"
        elif risk_val > 0.025:
            return f"elevated risk ({risk_val:.1%})"
        else:
            return "high risk"

    def _advanced_filter_note(self, analysis: PairAnalysis) -> str:
        """Generate note when basic gates pass but advanced filters block trade."""
        # Check volatility regime filter
        vol_regime = getattr(analysis, "volatility_regime", "UNKNOWN")
        vol_gate_passed = getattr(analysis, "volatility_gate_passed", True)

        if not vol_gate_passed or vol_regime in ("LOW", "NORMAL"):
            regime_note = vol_regime.lower() if vol_regime != "UNKNOWN" else "suboptimal"
            return f"volatility regime ({regime_note})"

        # Check if blocked by circuit breaker
        if getattr(analysis, "blocked_by_circuit_breaker", False):
            breakers = getattr(analysis, "circuit_breakers_triggered", [])
            if breakers:
                return f"circuit breaker ({breakers[0]})"
            return "circuit breaker"

        # Check execution quality
        if not getattr(analysis, "execution_quality_passed", True):
            spread = getattr(analysis, "spread_pips", 0)
            slippage = getattr(analysis, "est_slippage_pips", 0)
            if spread > 2:
                return f"wide spread ({spread:.1f} pips)"
            if slippage > 1:
                return f"high slippage ({slippage:.1f} pips)"
            return "execution quality"

        # Check transformer/meta-labeler gates (if available)
        # These are checked after volatility since they're optional
        direction = analysis.direction
        confidence = analysis.confidence

        # Provide context-specific notes based on available signals
        if confidence >= 0.55:
            if getattr(analysis, "momentum_acceleration", False):
                return f"accelerating {direction.lower()} | regime filter"
            return f"{direction.lower()} setup | regime filter"
        elif confidence >= 0.50:
            return f"emerging {direction.lower()} | regime filter"

        # Generic fallback with more context
        return "regime filter"

    def generate_table(self) -> Table:
        """Generate results table for Live display.

        Returns:
            Rich Table with current scan results
        """
        table = Table(
            title=None,
            show_header=True,
            header_style=f"bold {self._planner_orange}",
            expand=False,
            padding=(0, 1),
            box=SIMPLE_HEAVY,
            border_style=self._planner_sand,
        )

        table.add_column("Pair", no_wrap=True)
        table.add_column("Signal", justify="center", no_wrap=True)
        table.add_column("Conf", justify="right", no_wrap=True)
        table.add_column("Checks", justify="center", no_wrap=True)
        table.add_column("Agent", justify="center", no_wrap=True)
        table.add_column("State", justify="center", no_wrap=True)
        table.add_column("Price", justify="right", no_wrap=True)
        table.add_column("Why", overflow="fold")

        # Sort analyses: tradeable first, then by opportunity score.
        sorted_analyses = sorted(
            self._current_analyses,
            key=lambda x: (x.is_tradeable, x.overall_score, x.confidence),
            reverse=True,
        )

        if not sorted_analyses:
            status = Text()
            if self._is_scanning:
                status.append("◌ Collecting live scan results...", style=f"bold {self._planner_cyan}")
            else:
                status.append("No scan results yet", style=self._planner_slate)
            table.add_row(
                Text("waiting", style=self._planner_slate),
                Text("-", style=self._planner_slate),
                Text("-", style=self._planner_slate),
                Text("-", style=self._planner_slate),
                Text("-", style=self._planner_slate),
                Text("SCANNING" if self._is_scanning else "IDLE", style=self._planner_cyan if self._is_scanning else self._planner_slate),
                Text("-", style=self._planner_slate),
                status,
            )
            return table

        for analysis in sorted_analyses:
            # Handle errors — show what we know, with actionable context
            if analysis.error:
                # Still show direction/confidence if we have partial data
                has_partial = analysis.current_price > 0.0001 or analysis.confidence > 0.01
                table.add_row(
                    self._format_pair_label(analysis),
                    self._format_direction(analysis.direction, analysis.confidence) if has_partial else Text("--", style="dim"),
                    self._format_confidence(analysis.confidence) if has_partial else Text("--", style="dim"),
                    Text("--", style="dim"),
                    Text("--", style="dim"),
                    Text("ERROR", style=f"bold {self._planner_orange}"),
                    self._format_price(analysis) if has_partial else Text("--", style="dim"),
                    self._format_error(analysis.error),
                )
                continue

            # Generate descriptive note using new method
            note = self._generate_descriptive_note(analysis)

            table.add_row(
                self._format_pair_label(analysis),
                self._format_direction(analysis.direction, analysis.confidence),
                self._format_confidence(analysis.confidence),
                self._format_checks_compact(analysis),
                self._format_agent_compact(analysis),
                self._format_status(analysis),
                self._format_price(analysis),
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
            f"[bold {self._planner_orange}]BUDDY PLANNER SCANNER[/bold {self._planner_orange}]",
            f"[{self._planner_cyan}]${nav:,.0f}[/{self._planner_cyan}]" if nav > 0 else "",
            f"[{self._planner_slate}]{open_trades} trades[/{self._planner_slate}]" if open_trades > 0 else "",
        ]

        if unrealized_pl != 0:
            pl_color = self._planner_cyan if unrealized_pl > 0 else self._planner_orange
            header_parts.append(f"[{pl_color}]P/L: ${unrealized_pl:+,.2f}[/{pl_color}]")

        header_parts.append(f"[{self._planner_sand}]{self._model_type} | {self._granularity}[/{self._planner_sand}]")
        if self._is_scanning:
            header_parts.append(f"[{self._planner_cyan}]◌ scanning[/{self._planner_cyan}]")

        header = " | ".join([p for p in header_parts if p])
        legend = (
            f"[{self._planner_slate}]● ready  ○ watch  ◐ filtered  ! error"
            "   Checks: M momentum  A confidence  R risk"
            f"[/{self._planner_slate}]"
        )
        return Panel(Group(header, legend), border_style=self._planner_sand)

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
            BarColumn(bar_width=18),
            TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
            console=self.console,
            transient=True,
        )

    def start_live(self) -> Live:
        """Start Live display for incremental updates.

        Returns:
            Rich Live instance
        """
        self._scan_start_time = datetime.now(timezone.utc)
        self._is_scanning = True
        self._live = Live(
            Group(self.generate_header(), self.generate_table()),
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
            self._live.update(Group(self.generate_header(), self.generate_table()))

    def stop_live(self) -> None:
        """Stop Live display."""
        self._is_scanning = False
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
        self._is_scanning = False

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
        agent_ready = [a for a in result.analyses if a.agent_total > 0]
        agent_confirmed = [a for a in agent_ready if a.agent_passed]
        if tradeable:
            self.console.print()
            self.console.print(
                f"[bold green]✓ {len(tradeable)} tradeable:[/bold green] "
                f"{', '.join(p.pair.replace('_', '/') for p in tradeable)}"
            )
        else:
            self.console.print()
            self.console.print("[dim]No trade-ready setups. Showing best watchlist opportunities.[/dim]")
            watchlist = sorted(
                [a for a in result.analyses if a.error is None and a.direction in {"LONG", "SHORT"}],
                key=lambda a: (a.overall_score, a.confidence),
                reverse=True,
            )[:3]
            if watchlist:
                self.console.print(
                    "[dim]Watchlist: "
                    + ", ".join(f"{a.pair.replace('_', '/')}" for a in watchlist)
                    + "[/dim]"
                )
        if agent_ready:
            self.console.print(
                f"[dim]Agents: {len(agent_confirmed)}/{len(agent_ready)} confirmed "
                f"({len(agent_ready)} pairs evaluated)[/dim]"
            )

        # Show diagnostic summary for errors
        error_pairs = [a for a in result.analyses if a.error]
        if error_pairs:
            self.console.print()
            self.console.print(
                f"[bold yellow]⚠ {len(error_pairs)}/{len(result.analyses)} pairs failed[/bold yellow]"
            )

            # Group errors by category for cleaner output
            error_groups: Dict[str, List[str]] = {}
            for a in error_pairs:
                err = a.error or "Unknown"
                # Extract error category from the friendly message
                if "auth" in err.lower() or "token" in err.lower() or "Unauthorized" in err:
                    key = "AUTH"
                elif "connection" in err.lower() or "timeout" in err.lower():
                    key = "NETWORK"
                elif "No data" in err or "no candles" in err.lower():
                    key = "NO_DATA"
                elif "No models" in err or "Missing dependency" in err:
                    key = "MODELS"
                elif "shape" in err.lower() or "Type mismatch" in err:
                    key = "DATA_TYPE"
                else:
                    key = "OTHER"
                error_groups.setdefault(key, []).append(a.pair.replace("_", "/"))

            fixes = {
                "AUTH": "Check OANDA_API_TOKEN and OANDA_ACCOUNT_ID in .env",
                "NETWORK": "Check internet connection and OANDA API status",
                "NO_DATA": "Verify OANDA credentials or place CSV files in market_data/",
                "MODELS": "Run training first, or check conda env (tf-metal / intel)",
                "DATA_TYPE": "Model/feature version mismatch — retrain models",
                "OTHER": "Check logs for details",
            }

            for category, pairs_list in error_groups.items():
                pair_str = ", ".join(pairs_list[:5])
                if len(pairs_list) > 5:
                    pair_str += f" +{len(pairs_list) - 5} more"
                self.console.print(
                    f"  [yellow]{category}[/yellow] [{self._planner_slate}]({pair_str})[/{self._planner_slate}]"
                )
                self.console.print(
                    f"  [dim]  Fix: {fixes.get(category, 'Check logs')}[/dim]"
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
