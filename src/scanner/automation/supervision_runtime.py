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

        # Execution manager (from scanner)
        self._em = getattr(scanner, "_execution_manager", None)

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

        # Supervision tick
        self._tick = SupervisionTick(
            execution_manager=self._em,
            scanner=scanner,
            control_plane=self._control_plane,
            supervision_core=self._supervision_core,
        )

        # Max scan passes per hunting phase
        self._max_scan_passes = getattr(self._config, "max_scan_passes", 2)

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
                f"\n[bold cyan]SUPERVISION MODE[/bold cyan] — "
                f"[dim]{self._tick_seconds}s ticks, "
                f"target {self._capacity.target_open} positions, "
                f"max {self._capacity.max_open}[/dim]"
            )

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
                (self._supervision_core._slot_manager.open_positions()
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
        """Print concise status to console."""
        if not self._console:
            return
        phase = self._phase.current_phase.value.upper()
        open_n = self._tick.open_trade_count
        scans = self._scan_count
        interval = round(self._rearm.current_interval / 60, 1)
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
