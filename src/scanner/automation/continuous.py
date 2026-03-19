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

                # Filter out blocked pairs before scanning
                scan_pairs = pairs
                if scan_pairs and self.scanner.config.blocked_pairs:
                    filtered = [p for p in scan_pairs if p not in self.scanner.config.blocked_pairs]
                    if len(filtered) < len(scan_pairs):
                        skipped = [p for p in scan_pairs if p in self.scanner.config.blocked_pairs]
                        if console:
                            console.print(f"[dim]Skipping blocked pairs: {', '.join(skipped)}[/dim]")
                        scan_pairs = filtered

                # Run scan
                result = self.scanner.scan(
                    pairs=scan_pairs,
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

                # Log observations from scan results (US-008)
                try:
                    from src.scanner.automation.observation_log import ObservationLog
                    obs_log = ObservationLog()
                    obs_count = 0
                    for analysis in result.analyses:
                        obs_count += obs_log.log_from_analysis(analysis)
                    if obs_count > 0 and console:
                        console.print(f"  [dim]Observations: {obs_count} patterns logged[/dim]")
                except Exception as obs_err:
                    logger.debug(f"Observation logging error: {obs_err}")

                # Apply config tuning before next scan (US-005)
                try:
                    from src.scanner.automation.config_tuner import ConfigTuner
                    ct = ConfigTuner()
                    adjustments = ct.apply_to_config(self.scanner.config)
                    if adjustments and console:
                        console.print(f"  [dim]Config tuned: {len(adjustments)} adjustments[/dim]")
                except Exception as tune_err:
                    logger.debug(f"Pre-scan config tuning error: {tune_err}")

                # Smart trading loop: monitor, drawdown guardian, RL sync, learning
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
        """Run the smart trading loop: monitor, guardian, RL sync, learning, adaptation."""
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
            trades_synced = rl_result.get("trades_synced", 0)
            if trades_synced > 0 and console:
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

            # 5. Learning loop (US-012)
            self._run_learning_loop(em, trades_synced)

        except Exception as e:
            logger.debug(f"Smart loop error: {e}")

    def _run_learning_loop(self, em: object, trades_synced: int) -> None:
        """Step 5: Learning engine analysis, promotion, config tuning, metrics.

        Wraps all learning components in try/except to never crash the scan loop.
        """
        try:
            import json
            from pathlib import Path
            from src.scanner.automation.learning_engine import LearningEngine
            from src.scanner.automation.config_tuner import ConfigTuner
            from src.scanner.automation.improvement_tracker import ImprovementTracker
            from src.scanner.automation.state_engine import StateEngine

            le = LearningEngine()
            learnings_added = 0
            rules_promoted = 0
            config_adjustments = []

            # 5a. Analyze each newly synced trade
            if trades_synced > 0:
                journal_path = Path("trained_data/trade_journal_rl.json")
                if journal_path.exists():
                    entries = json.loads(journal_path.read_text())
                    closed = [e for e in entries if e.get("outcome") is not None]

                    for entry in closed:
                        insights = le.analyze_trade(entry)
                        if insights:
                            learnings_added += le.append_to_learnings(insights)

                        # Per-pair SL/TP adaptation (US-006)
                        le.update_pair_sl_tp(entry)

                        # LLM deep analysis for significant losses (US-009)
                        if getattr(self.scanner.config, "enable_llm_trade_analysis", False):
                            outcome = entry.get("outcome", {})
                            if outcome.get("realized_pl", 0) < -100:
                                pair_entries = [e for e in entries if e.get("pair") == entry.get("pair")]
                                llm_insights = le.deep_analyze_loss(entry, pair_entries)
                                if llm_insights:
                                    learnings_added += le.append_to_learnings(llm_insights)

            # 5b. Check for rule promotions
            promoted = le.check_promotions()
            rules_promoted = len(promoted)

            # 5c. Apply config tuning from promoted rules
            try:
                ct = ConfigTuner()
                config_adjustments = ct.apply_to_config(self.scanner.config)
            except Exception as tune_err:
                logger.debug(f"Config tuning error: {tune_err}")

            # 5d. Record session metrics
            if trades_synced > 0:
                try:
                    tracker = ImprovementTracker()
                    journal_path = Path("trained_data/trade_journal_rl.json")
                    all_entries = json.loads(journal_path.read_text()) if journal_path.exists() else []
                    tracker.record_session(
                        trades=all_entries,
                        learnings_added=learnings_added,
                        rules_promoted=rules_promoted,
                        config_adjustments=config_adjustments,
                    )
                except Exception as track_err:
                    logger.debug(f"Improvement tracking error: {track_err}")

            # 5e. Update portfolio snapshot
            try:
                se = StateEngine()
                se.update_portfolio_snapshot()
            except Exception as state_err:
                logger.debug(f"State update error: {state_err}")

            # 5f. Audit every 10th cycle
            try:
                se = StateEngine()
                cycle = se.increment_scan_cycle()
                if cycle % 10 == 0:
                    audit_result = le.audit()
                    if audit_result.get("actions") and console:
                        console.print(f"  [dim]Audit: {audit_result['actions']}[/dim]")
            except Exception as audit_err:
                logger.debug(f"Audit error: {audit_err}")

            # 5g. Merge accuracy-gated blocked pairs into scanner config (US-013)
            try:
                from src.scanner.automation.accuracy_gate import AccuracyGate
                ag = AccuracyGate(min_accuracy=0.55, min_trades=5)
                blocked_by_accuracy = ag.get_blocked_pairs()

                if blocked_by_accuracy:
                    # Merge with existing blocked_pairs from config
                    original_blocked = set(self.scanner.config.blocked_pairs or [])
                    new_blocked = set(blocked_by_accuracy)
                    merged = list(original_blocked | new_blocked)
                    self.scanner.config.blocked_pairs = merged

                    if console and blocked_by_accuracy:
                        console.print(
                            f"  [yellow]Accuracy gate blocking {len(blocked_by_accuracy)} "
                            f"pair(s): {', '.join(blocked_by_accuracy)}[/yellow]"
                        )
            except Exception as accuracy_err:
                logger.debug(f"Accuracy gate merge error: {accuracy_err}")

            # Log learning activity
            if (learnings_added > 0 or rules_promoted > 0) and console:
                console.print(
                    f"  [magenta]Learning: {learnings_added} insights captured, "
                    f"{rules_promoted} rules promoted[/magenta]"
                )
            if config_adjustments and console:
                console.print(
                    f"  [magenta]Config: {len(config_adjustments)} adjustments applied[/magenta]"
                )

        except Exception as e:
            logger.debug(f"Learning loop error: {e}")

    def _sleep_with_progress(self, minutes: int):
        """Sleep with progress indication, allowing Ctrl+C to interrupt."""
        if console:
            console.print(f"\n[dim]Next scan in {minutes} minutes...[/dim]")

        # Sleep in 1-second increments to allow Ctrl+C
        for _ in range(minutes * 60):
            if not self._running:
                break
            time.sleep(1)
