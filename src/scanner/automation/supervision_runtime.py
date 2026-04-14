"""Supervision-first runtime — replaces the 5-minute scan loop.

The main loop ticks every 30 seconds through 5 phases:
  IDLE → HUNTING → SUPERVISING → DIAGNOSING → ADAPTING

Scanner is triggered on-demand by the capacity model, not by a fixed timer.
Open trades are supervised at 30s intervals, not 5 minutes.
"""

import signal
import time
from typing import Any, Dict, List, Optional

import structlog

from src.scanner.automation.phase_manager import Phase, PhaseManager
from src.scanner.automation.capacity_model import CapacityModel, RearmTimer
from src.scanner.automation.supervision_tick import SupervisionTick
from src.scanner.automation.serial_executor import SerialExecutor
from src.scanner.automation.execution_funnel import ExecutionFunnel

logger = structlog.get_logger(__name__)


class SupervisionRuntime:
    """The new brain — supervision-first, scan on-demand.

    Replaces the while loop in ContinuousScanner.run() when
    config.enable_supervision_mode is True.
    """

    def __init__(
        self,
        scanner: Any,
        config: Any = None,
        console: Any = None,
    ):
        self._scanner = scanner
        self._config = config or getattr(scanner, "config", None)
        self._console = console
        self._running = False
        self._scan_count = 0

        # Tick interval
        self._tick_seconds = getattr(self._config, "supervision_tick_seconds", 30)

        # Phase manager
        self._phase = PhaseManager()

        # Capacity model
        self._capacity = CapacityModel(
            target_open=getattr(self._config, "target_open_positions", 3),
            max_open=getattr(self._config, "max_open_positions", 5),
            scan_trigger_threshold=getattr(self._config, "scan_trigger_threshold", 2),
        )

        # Rearm timer
        self._rearm = RearmTimer(
            base_interval=getattr(self._config, "rearm_base_interval", 300.0),
            min_interval=getattr(self._config, "rearm_min_interval", 120.0),
            max_interval=getattr(self._config, "rearm_max_interval", 1800.0),
        )

        # Execution manager (from scanner — attribute is _executor, lazy-initialized)
        # Force scanner to init its executor if it hasn't yet
        if hasattr(scanner, "_init_executor"):
            try:
                scanner._init_executor()
            except Exception as _init_err:
                logger.warning("supervision_runtime.executor_init_failed", error=str(_init_err))
        self._em = getattr(scanner, "_executor", None) or getattr(scanner, "_execution_manager", None)
        if self._em is None:
            logger.warning("supervision_runtime.no_execution_manager — trades will not execute")

        # Serial executor
        self._executor = SerialExecutor(
            execution_manager=self._em,
            scanner=scanner,
        )

        # Supervision core (from PR 1)
        self._supervision_core = None
        try:
            from src.scanner.automation.trade_slots import SlotManager
            from src.scanner.automation.close_diagnostician import CloseDiagnostician
            from src.scanner.automation.supervision_policy_engine import SupervisionPolicyEngine
            from src.scanner.automation.trigger_registry import TriggerRegistry
            from src.scanner.automation.supervision_core import SupervisionCore

            slot_mgr = SlotManager(
                max_concurrent=getattr(self._config, "max_open_positions", 5)
            )
            self._supervision_core = SupervisionCore(
                slot_manager=slot_mgr,
                diagnostician=CloseDiagnostician(),
                policy_engine=SupervisionPolicyEngine(),
                trigger_registry=TriggerRegistry(),
            )
        except Exception as e:
            logger.warning("supervision_runtime.core_init_failed", error=str(e))

        # Control plane (from Tier 7)
        self._control_plane = None
        try:
            from src.scanner.automation.trading_control_plane import TradingControlPlane
            # Reuse control plane if already created by ContinuousScanner
            # Otherwise create a minimal one
            self._control_plane = getattr(scanner, "_control_plane", None)
        except Exception:
            pass

        # Supervision tick — shares the execution manager reference
        self._tick = SupervisionTick(
            execution_manager=self._em,
            scanner=scanner,
            control_plane=self._control_plane,
            supervision_core=self._supervision_core,
        )

        if self._em is not None:
            logger.info(
                "supervision_runtime.ready",
                has_em=True,
                has_core=self._supervision_core is not None,
                has_control_plane=self._control_plane is not None,
            )
        else:
            logger.warning("supervision_runtime.degraded — no execution manager, will initialize on first scan")

        # Max scan passes per hunting phase
        # Default 1 on memory-constrained hardware (8GB M1): each scan pass
        # loads the full TCN/Ridge/RF ensemble for 15 pairs (~1.5-2GB peak).
        # A second pass before GC clears the first = OOM. The rearm timer
        # (default 5m) gates hunting frequency instead.
        # Increase via config max_scan_passes=2 only on machines with 16GB+.
        self._max_scan_passes = getattr(self._config, "max_scan_passes", 1)

        # Pairs
        self._pairs: List[str] = []
        self._auto_execute: bool = False
        self._top_n: int = 5

    def run(
        self,
        pairs: Optional[List[str]] = None,
        auto_execute: bool = False,
        top_n: int = 5,
        interval_minutes: Optional[int] = None,
    ) -> int:
        """Main supervision loop. Returns scan count on exit."""
        self._pairs = pairs or getattr(self._config, "pairs", None) or []
        self._auto_execute = auto_execute
        self._top_n = top_n
        self._running = True

        # Signal handling
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_signal)

        if self._console:
            self._console.print(
                f"\n[bold cyan]◆ SUPERVISION MODE[/bold cyan] — supervise-first, scan on-demand"
            )
            self._console.print(
                f"[dim]  {self._tick_seconds}s ticks · "
                f"target {self._capacity.target_open} positions · "
                f"max {self._capacity.max_open} · "
                f"rearm {self._rearm.base_interval / 60:.0f}m[/dim]"
            )
            self._console.print("[dim]  Press Ctrl+C to stop[/dim]")
            self._console.print("=" * 70)

        try:
            while self._running:
                tick_start = time.time()

                try:
                    self._run_tick()
                except Exception as e:
                    logger.error("supervision_runtime.tick_error", error=str(e))

                # Sleep remainder of tick interval
                elapsed = time.time() - tick_start
                sleep_time = max(0, self._tick_seconds - elapsed)
                if sleep_time > 0 and self._running:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            self._shutdown()

        return self._scan_count

    def _run_tick(self) -> None:
        """Execute one tick of the supervision loop."""
        phase = self._phase.current_phase

        # Check phase timeout
        timeout_target = self._phase.check_timeout()
        if timeout_target is not None:
            self._phase.transition_to(timeout_target, "phase_timeout")
            phase = self._phase.current_phase

        # Phase-specific logic
        if phase == Phase.IDLE:
            self._tick_idle()
        elif phase == Phase.HUNTING:
            self._tick_hunting()
        elif phase == Phase.SUPERVISING:
            self._tick_supervising()
        elif phase == Phase.DIAGNOSING:
            # Auto-transition to ADAPTING (synchronous)
            self._phase.transition_to(Phase.ADAPTING, "diagnosis_complete")
        elif phase == Phase.ADAPTING:
            self._tick_adapting()

        # Console status
        if self._console and self._tick.tick_count % 4 == 0:  # Every ~2 min
            self._print_status()

    def _tick_idle(self) -> None:
        """IDLE: health checks, check if we should start hunting."""
        open_count = self._tick.open_trade_count

        # If we have open trades, we should be supervising
        if open_count > 0:
            self._phase.transition_to(Phase.SUPERVISING, "found_open_trades")
            return

        # Check if we should hunt
        if self._capacity.should_scan(open_count, session_active=True):
            if self._rearm.should_rearm():
                self._phase.transition_to(Phase.HUNTING, "capacity_available")

    def _tick_hunting(self) -> None:
        """HUNTING: bounded scan → serial execution → transition."""
        if not self._auto_execute:
            # No auto-execute → just scan and display
            self._run_scan_pass()
            self._phase.transition_to(Phase.IDLE, "scan_complete_no_execute")
            return

        # Run bounded scan passes
        total_executed = 0
        for pass_num in range(self._max_scan_passes):
            open_count = self._tick.open_trade_count + total_executed
            available = self._capacity.available_slots(open_count)
            if available <= 0:
                break

            # Pre-filter pairs
            open_pairs = {
                getattr(p, "pair", "") for p in
                (self._supervision_core.slot_manager.open_positions()
                 if self._supervision_core else [])
            }
            scannable = self._capacity.get_scannable_pairs(
                all_pairs=self._pairs,
                open_pairs=open_pairs,
                cooldown_pairs=set(),  # TODO: get from slot_manager
                blocked_pairs=set(),
                correlated_pairs=set(),
            )

            if not scannable:
                break

            # Scan
            result = self._run_scan_pass(pairs=scannable)
            if result is None:
                break

            # Get tradeable candidates
            tradeable = [a for a in result.analyses if getattr(a, "is_tradeable", False)]
            if not tradeable:
                break

            # Serial execution
            funnel = self._executor.execute(tradeable, max_trades=available)
            total_executed += funnel.executed

            if funnel.executed == 0:
                break  # Nothing executed, don't try again

        # Force GC after scan pass: ML inference (TCN/Ridge/RF) allocates
        # large numpy arrays that aren't freed until GC runs. On 8GB M1,
        # these must be reclaimed before the next tick or we OOM.
        import gc as _gc
        _gc.collect()

        self._rearm.record_scan()

        # Transition based on state
        if total_executed > 0 or self._tick.open_trade_count > 0:
            self._phase.transition_to(Phase.SUPERVISING, f"hunting_complete_{total_executed}_executed")
        else:
            self._phase.transition_to(Phase.IDLE, "hunting_complete_no_trades")

    def _tick_supervising(self) -> None:
        """SUPERVISING: run supervision tick, check for transitions."""
        result = self._tick.tick()

        # Close detected → diagnose
        if result["closes_detected"] > 0:
            self._phase.transition_to(Phase.DIAGNOSING, f"{result['closes_detected']}_closes")

            # Record outcome for rearm timer
            for d in result.get("diagnoses", []):
                was_profit = (d.get("realized_pl", 0) or 0) > 0
                self._rearm.record_outcome(was_profit)
            return

        # All flat → idle or hunt
        if self._tick.open_trade_count == 0:
            if self._capacity.should_scan(0, session_active=True) and self._rearm.should_rearm():
                self._phase.transition_to(Phase.HUNTING, "all_flat_rearm")
            else:
                self._phase.transition_to(Phase.IDLE, "all_flat")
            return

        # Has capacity → hunt (while supervising existing)
        available = self._capacity.available_slots(self._tick.open_trade_count)
        if available > 0 and self._rearm.should_rearm():
            self._phase.transition_to(Phase.HUNTING, f"rearm_{available}_slots")

    def _tick_adapting(self) -> None:
        """ADAPTING: post-diagnosis, decide next phase."""
        # RL sync happens in the Tier 7 event handlers (already wired)
        # Supervision core has already applied adaptive policies in process_close_event

        # Transition based on state
        if self._tick.open_trade_count > 0:
            available = self._capacity.available_slots(self._tick.open_trade_count)
            if available > 0 and self._rearm.should_rearm():
                self._phase.transition_to(Phase.HUNTING, "post_adapt_hunt")
            else:
                self._phase.transition_to(Phase.SUPERVISING, "post_adapt_supervise")
        else:
            self._phase.transition_to(Phase.IDLE, "post_adapt_idle")

    def _run_scan_pass(self, pairs: Optional[List[str]] = None) -> Any:
        """Run one scan pass. Returns ScanResult or None."""
        scan_pairs = pairs or self._pairs
        if not scan_pairs:
            return None

        self._scan_count += 1
        try:
            result = self._scanner.scan(
                pairs=scan_pairs,
                max_workers=4,
            )
            return result
        except Exception as e:
            logger.error("supervision_runtime.scan_error", error=str(e))
            return None

    def _print_status(self) -> None:
        """Print supervision status using the watch header for visual consistency."""
        if not self._console:
            return

        phase = self._phase.current_phase.value.upper()
        open_n = self._tick.open_trade_count
        scans = self._scan_count
        interval = round(self._rearm.current_interval / 60, 1)

        # Use the standard watch header with SUPERVISE mode label
        try:
            from src.scanner.cli_display import render_watch_header
            nav = None
            pnl = None
            if self._em:
                try:
                    nav = self._em.get_nav() if hasattr(self._em, 'get_nav') else None
                except Exception:
                    pass
            render_watch_header(
                nav=nav,
                open_count=open_n,
                unrealized_pnl=pnl,
                cycle_num=scans,
                uptime_s=self._tick.tick_count * self._tick_seconds,
                next_scan_s=max(0, self._rearm.current_interval - (time.time() - self._rearm._last_scan_time)),
                mode=f"SUPERVISE · {phase}",
            )
        except Exception:
            # Fallback to simple line if cli_display not available
            self._console.print(
                f"  [dim]◆ {phase} · {open_n} open · {scans} scans · rearm {interval}m[/dim]"
            )

    def _handle_signal(self, signum: int, frame: Any) -> None:
        self._running = False

    def _shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info(
            "supervision_runtime.shutdown",
            phase=self._phase.current_phase.value,
            scans=self._scan_count,
            ticks=self._tick.tick_count,
        )
        self._running = False
        self._write_session_handoff()

    def _write_session_handoff(self) -> None:
        """Write a human-readable session handoff for Claude to read on next invocation.

        This file is raw facts — what happened this session.
        Claude reads it alongside .claude/brain/briefing.md to get full context.
        """
        import os
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        handoff_path = Path(__file__).parents[3] / ".claude" / "brain" / "session_handoff.md"
        handoff_path.parent.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()

        # Gather portfolio state
        nav = None
        open_trades = []
        realized_pl = None
        try:
            if self._em and hasattr(self._em, "get_nav"):
                nav = self._em.get_nav()
        except Exception:
            pass

        try:
            if self._em and hasattr(self._em, "monitor_open_trades"):
                open_trades = self._em.monitor_open_trades(evaluate_exits=False) or []
        except Exception:
            pass

        # Gather recent journal entries (last 10)
        recent_trades: list = []
        try:
            journal_path = Path(__file__).parents[3] / "trained_data" / "trade_journal_rl.json"
            if journal_path.exists():
                with open(journal_path, "r") as f:
                    journal = json.load(f)
                if isinstance(journal, list):
                    recent_trades = journal[-10:]
        except Exception:
            pass

        # Build handoff markdown
        lines = [
            "# Session Handoff",
            f"> Written by Buddy at shutdown. Raw facts only.",
            f"> Timestamp: {now}",
            f"> Read this alongside `.claude/brain/briefing.md`.",
            "",
            "---",
            "",
            "## Runtime Summary",
            f"- Phase at shutdown: `{self._phase.current_phase.value}`",
            f"- Scan passes this session: {self._scan_count}",
            f"- Supervision ticks: {self._tick.tick_count}",
            f"- Open trades at shutdown: {len(open_trades)}",
            "",
            "## Portfolio State",
        ]

        if nav is not None:
            lines.append(f"- NAV: ${nav:,.2f}")
        else:
            lines.append("- NAV: (unavailable — EM not connected)")

        if open_trades:
            lines.append(f"- Open positions: {len(open_trades)}")
            for t in open_trades:
                pair = t.get("pair", "?")
                direction = t.get("direction", "?")
                pl = t.get("unrealized_pl", t.get("unrealized_pnl", "?"))
                lines.append(f"  - {pair} {direction} | unrealized P/L: {pl}")
        else:
            lines.append("- Open positions: 0")

        lines += [
            "",
            "## Last 10 Journal Entries",
        ]

        if recent_trades:
            for t in recent_trades:
                pair = t.get("pair", "?")
                direction = t.get("direction", "?")
                conf = t.get("confidence", 0)
                outcome = t.get("outcome") or {}
                pl = outcome.get("realized_pl", "?")
                won = outcome.get("trade_won", None)
                exit_reason = outcome.get("exit_reason", "?")
                disagreement = (
                    t.get("regime", {}).get("model_disagreement")
                    or (t.get("agents", {}).get("agent_reasons") or [{}])[-1]
                    .get("metadata", {}).get("model_disagreement", "?")
                )
                result_icon = "✅" if won else ("❌" if won is False else "⏳")
                ts = t.get("timestamp", "?")[:10]
                lines.append(
                    f"- {ts} | {pair} {direction} | conf={conf:.0%} | "
                    f"disagreement={disagreement} | {result_icon} {exit_reason} | P/L: {pl}"
                )
        else:
            lines.append("- (no journal entries found)")

        lines += [
            "",
            "## Config State",
            f"- Rearm interval: {self._rearm.current_interval:.0f}s",
            f"- Target open positions: {self._capacity.target_open}",
            f"- Max open positions: {self._capacity.max_open}",
            "",
            "---",
            "_End of handoff. Claude: read briefing.md next._",
        ]

        try:
            with open(handoff_path, "w") as f:
                f.write("\n".join(lines))
            logger.info("supervision_runtime.handoff_written", path=str(handoff_path))
        except Exception as e:
            logger.warning("supervision_runtime.handoff_write_failed", error=str(e))
