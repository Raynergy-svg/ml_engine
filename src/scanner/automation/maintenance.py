"""
Idle Maintenance - Background retraining and journal sync.

Runs during idle periods between scans to keep models fresh.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.scanner.automation.background_activity import get_background_activity_tracker

try:
    from rich.console import Console
    console = Console()
except ImportError:
    console = None

logger = logging.getLogger(__name__)


@dataclass
class MaintenanceConfig:
    """Configuration for idle maintenance."""

    retrain_timeout_seconds: int = 300  # 5 minutes
    retrain_candles: int = 3000
    min_trades_before_retrain: int = 50
    retrain_cooldown_hours: int = 24


class IdleMaintenance:
    """
    Run maintenance tasks during idle periods.

    Tasks:
    - Journal sync with OANDA
    - Gate model retraining (XGB, RF, Ridge)
    - Drift detection checks

    Example:
        >>> maintenance = IdleMaintenance()
        >>> maintenance.run_if_needed()  # Runs sync and checks retrain
        >>> maintenance.trigger_background_retrain()  # Force background retrain
    """

    def __init__(self, config: Optional[MaintenanceConfig] = None):
        self.config = config or MaintenanceConfig()
        self._last_retrain: Optional[datetime] = None
        self._retrain_thread: Optional[threading.Thread] = None
        self._retrain_process: Optional[subprocess.Popen] = None
        self._stop_requested = threading.Event()

    def run_if_needed(self, oanda_client=None):
        """
        Run maintenance tasks if needed.

        Checks:
        1. Syncs trade journal with OANDA
        2. Checks if gate retraining is needed (50+ trades since last)
        3. Triggers background retrain if conditions met

        Args:
            oanda_client: Optional OANDA client for journal sync
        """
        try:
            if self._stop_requested.is_set():
                return
            self._sync_journal(oanda_client)
            self._check_and_retrain()
        except Exception as e:
            logger.debug(f"Idle maintenance failed: {e}")

    def _sync_journal(self, oanda_client=None):
        """Sync trade journal with OANDA closed trades."""
        try:
            from src.utils.trade_journal import TradeJournal

            journal = TradeJournal()

            # Get OANDA client if not provided
            if oanda_client is None:
                try:
                    from src.utils.oanda_practice import get_oanda_client
                    oanda_client = get_oanda_client()
                except Exception:
                    return  # Can't sync without client

            updated = journal.update_from_oanda(oanda_client)
            if updated > 0:
                logger.info(f"Journal synced: {updated} trade(s) updated")

        except Exception as e:
            logger.debug(f"Journal sync failed: {e}")

    def _check_and_retrain(self):
        """Check if retraining is needed and trigger if so."""
        try:
            if self._stop_requested.is_set():
                return
            # Check cooldown
            if self._last_retrain:
                cooldown = timedelta(hours=self.config.retrain_cooldown_hours)
                if datetime.now() - self._last_retrain < cooldown:
                    return

            # Check if market intelligence says retrain needed
            try:
                from market_intelligence import MarketIntelligence

                intel = MarketIntelligence(
                    enable_sentiment=False,
                    enable_calendar=False,
                    enable_online_learning=True,
                )

                if intel.should_update_model():
                    self.trigger_background_retrain()
                    intel.mark_model_updated()

            except ImportError:
                logger.debug("MarketIntelligence not available for retrain check")

        except Exception as e:
            logger.debug(f"Retrain check failed: {e}")

    def trigger_background_retrain(self):
        """
        Spawn background thread to retrain gate models.

        Runs retrain_gates.py as a subprocess to avoid blocking.
        """
        # Don't start another if one is running
        if self._retrain_thread and self._retrain_thread.is_alive():
            logger.debug("Background retrain already in progress")
            return

        def _retrain_task():
            tracker = get_background_activity_tracker()
            activity_id = tracker.start_activity(
                kind="maintenance_retrain",
                title="Idle maintenance gate retrain",
                source="idle_maintenance",
                metadata={"timeout_seconds": self.config.retrain_timeout_seconds},
            )
            try:
                if self._stop_requested.is_set():
                    tracker.update_activity(activity_id, status="canceled", summary="Shutdown requested before retrain started")
                    return

                # Disk-pressure precondition (Fix #8). ENOSPC inside the
                # retrain subprocess produces a silent SIGSEGV mid-train AND
                # can corrupt half-written checkpoints (joblib/keras write
                # incrementally; a truncated artifact then segfaults on the
                # next inference load). Skip cleanly here — do NOT update
                # _last_retrain, so the 24h cooldown stays reserved for
                # successful runs and the next maintenance cycle can
                # re-attempt as soon as conditions improve. The introspecting
                # variant is required: a raise would kill this daemon thread
                # and wedge the cooldown permanently.
                from src.scanner.runtime_guards import get_disk_pressure_status
                disk_ok, free_mib, threshold_mib = get_disk_pressure_status()
                if not disk_ok:
                    msg = (
                        f"maintenance retrain skipped: disk pressure "
                        f"({free_mib} MiB free < {threshold_mib} MiB threshold)"
                    )
                    logger.warning(msg)
                    tracker.fail_activity(
                        activity_id,
                        error=msg,
                        summary="Retrain skipped due to disk pressure",
                        metadata={
                            "free_mib": free_mib,
                            "threshold_mib": threshold_mib,
                            "reason": "disk_pressure",
                        },
                    )
                    return

                logger.info("🔄 Background gate retraining started...")
                tracker.append_event(activity_id, "Background gate retraining started")

                # Find retrain_gates.py
                retrain_script = Path(__file__).parents[3] / "retrain_gates.py"
                if not retrain_script.exists():
                    # Try main.py retrain-gates command instead
                    main_script = Path(__file__).parents[3] / "main.py"
                    if main_script.exists():
                        cmd = [sys.executable, str(main_script), "retrain-gates"]
                        cwd = str(main_script.parent)
                    else:
                        logger.warning("No retrain script found")
                        tracker.fail_activity(activity_id, error="No retrain script found", summary="Could not locate retrain command")
                        return
                else:
                    cmd = [
                        sys.executable,
                        str(retrain_script),
                        "--candles",
                        str(self.config.retrain_candles),
                    ]
                    cwd = str(retrain_script.parent)
                tracker.update_activity(activity_id, metadata={"command": cmd, "cwd": cwd})

                self._retrain_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd,
                )
                tracker.update_activity(
                    activity_id,
                    metadata={"pid": self._retrain_process.pid},
                    summary=f"Running retrain process PID {self._retrain_process.pid}",
                )
                deadline = time.time() + self.config.retrain_timeout_seconds
                while self._retrain_process.poll() is None:
                    if self._stop_requested.is_set():
                        logger.info("Maintenance shutdown: terminating background retrain")
                        self._retrain_process.terminate()
                        try:
                            self._retrain_process.wait(timeout=2)
                        except Exception:
                            self._retrain_process.kill()
                        tracker.update_activity(activity_id, status="canceled", summary="Maintenance retrain canceled during shutdown")
                        return
                    if time.time() >= deadline:
                        self._retrain_process.kill()
                        raise subprocess.TimeoutExpired(cmd=cmd, timeout=self.config.retrain_timeout_seconds)
                    time.sleep(0.2)

                _, stderr = self._retrain_process.communicate()
                if self._retrain_process.returncode == 0:
                    logger.info("✅ Background gate retraining complete")
                    self._last_retrain = datetime.now()
                    tracker.complete_activity(
                        activity_id,
                        summary="Background gate retraining complete",
                        metadata={"returncode": self._retrain_process.returncode},
                    )
                else:
                    logger.warning(f"Gate retraining failed: {stderr[:200]}")
                    tracker.fail_activity(
                        activity_id,
                        error=stderr[:200] or f"Process exited with {self._retrain_process.returncode}",
                        summary="Gate retraining failed",
                        metadata={"returncode": self._retrain_process.returncode},
                    )

            except subprocess.TimeoutExpired:
                logger.warning(f"Gate retraining timed out ({self.config.retrain_timeout_seconds}s)")
                tracker.fail_activity(
                    activity_id,
                    error=f"Timed out after {self.config.retrain_timeout_seconds}s",
                    summary="Gate retraining timed out",
                )
            except Exception as e:
                logger.warning(f"Background retrain failed: {e}")
                tracker.fail_activity(activity_id, error=str(e), summary="Background retrain failed")
            finally:
                self._retrain_process = None

        # Spawn daemon thread
        self._retrain_thread = threading.Thread(target=_retrain_task, daemon=True)
        self._retrain_thread.start()

        if console:
            console.print("[dim]🔄 Background gate retraining triggered[/dim]")

    def stop(self) -> None:
        """Request maintenance shutdown and stop active retraining work."""
        self._stop_requested.set()

        proc = self._retrain_process
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception as exc:
                    logger.debug("Maintenance retrain kill failed: %s", exc)

        if self._retrain_thread and self._retrain_thread.is_alive():
            self._retrain_thread.join(timeout=2)

    def run_sync_retrain(self):
        """
        Run gate retraining synchronously (blocking).

        Use this during dedicated maintenance windows, not between scans.
        """
        try:
            # Disk-pressure precondition (Fix #8). Same failure mode as the
            # async path: ENOSPC mid-train SIGSEGVs the child + corrupts
            # partial checkpoints. The sync variant must NOT raise — callers
            # expect a bool return — so use the introspecting status helper
            # and bail with `return False`. _last_retrain is left untouched
            # so the cooldown is not wedged by a transient disk fill.
            from src.scanner.runtime_guards import get_disk_pressure_status
            disk_ok, free_mib, threshold_mib = get_disk_pressure_status()
            if not disk_ok:
                logger.warning(
                    "sync retrain skipped: disk pressure "
                    "(%d MiB free < %d MiB threshold)",
                    free_mib, threshold_mib,
                )
                return False

            if console:
                console.print("[dim]🔄 Running synchronous gate retraining...[/dim]")

            retrain_script = Path(__file__).parents[3] / "retrain_gates.py"
            main_script = Path(__file__).parents[3] / "main.py"

            if retrain_script.exists():
                cmd = [
                    sys.executable,
                    str(retrain_script),
                    "--candles",
                    str(self.config.retrain_candles),
                ]
                cwd = str(retrain_script.parent)
            elif main_script.exists():
                cmd = [sys.executable, str(main_script), "retrain-gates"]
                cwd = str(main_script.parent)
            else:
                logger.warning("No retrain script found")
                return False

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.retrain_timeout_seconds,
                cwd=cwd,
            )

            if result.returncode == 0:
                self._last_retrain = datetime.now()
                if console:
                    console.print("[green]✅ Gate models retrained[/green]")
                return True
            else:
                logger.warning(f"Sync retrain failed: {result.stderr[:200]}")
                return False

        except subprocess.TimeoutExpired:
            logger.warning("Sync retrain timed out")
            return False
        except Exception as e:
            logger.warning(f"Sync retrain failed: {e}")
            return False
