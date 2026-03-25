"""Module activation map and async dispatch.

US-116: Track which modules are initialized, called, and orphaned.
Provides configurable cycle frequency so modules can run every N cycles
instead of every scan.

Approach:
1. Register modules with their run frequency (every N cycles)
2. Track activation: initialized vs actually called in scan cycle
3. Build dependency graph for module ordering
4. Identify orphaned modules (initialized but never dispatched)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "module_activation_log.json"


class ModuleRecord:
    """Tracks a registered module's activation state."""

    def __init__(
        self,
        name: str,
        instance: Any,
        run_every_n: int = 1,
        dependencies: Optional[List[str]] = None,
    ):
        self.name = name
        self.instance = instance
        self.run_every_n = max(1, run_every_n)
        self.dependencies = dependencies or []

        # State
        self.initialized = True
        self.enabled = True
        self.total_runs: int = 0
        self.total_skips: int = 0
        self.last_run_cycle: int = 0
        self.last_error: Optional[str] = None
        self.status: str = "active"  # active, orphaned, disabled, error

    def should_run(self, current_cycle: int) -> bool:
        """Check if module should run this cycle."""
        if not self.enabled:
            return False
        return (current_cycle - self.last_run_cycle) >= self.run_every_n

    def mark_run(self, cycle: int) -> None:
        self.last_run_cycle = cycle
        self.total_runs += 1
        self.status = "active"

    def mark_skip(self) -> None:
        self.total_skips += 1

    def mark_error(self, error: str) -> None:
        self.last_error = error
        self.status = "error"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "enabled": self.enabled,
            "run_every_n": self.run_every_n,
            "total_runs": self.total_runs,
            "total_skips": self.total_skips,
            "last_run_cycle": self.last_run_cycle,
            "last_error": self.last_error,
            "dependencies": self.dependencies,
        }


class ModuleActivationMap:
    """Tracks module activation and provides async dispatch scheduling.

    Solves the orphaned module problem: 22+ modules initialized in
    engine.py but never called in the scan cycle. This map registers
    all modules, tracks which ones actually run, and provides
    configurable dispatch intervals.

    Usage:
        activation = ModuleActivationMap()
        activation.register_module("threshold_optimizer", optimizer, run_every_n=5)
        if activation.should_run("threshold_optimizer", cycle=42):
            optimizer.optimize_thresholds("NORMAL")
            activation.mark_run("threshold_optimizer", cycle=42)
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or _LOG_PATH

        self._modules: Dict[str, ModuleRecord] = {}
        self._current_cycle: int = 0
        self._activation_log: List[Dict[str, Any]] = []
        self._total_cycles: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_module(
        self,
        name: str,
        instance: Any,
        run_every_n: int = 1,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Register a module for activation tracking.

        Args:
            name: Module identifier.
            instance: Module instance (for health checks).
            run_every_n: Run every N scan cycles (1 = every cycle).
            dependencies: List of module names this depends on.
        """
        self._modules[name] = ModuleRecord(
            name=name,
            instance=instance,
            run_every_n=run_every_n,
            dependencies=dependencies,
        )

    def should_run(self, name: str, current_cycle: int) -> bool:
        """Check if module should run this cycle.

        Args:
            name: Module name.
            current_cycle: Current scan cycle number.

        Returns:
            True if module should execute this cycle.
        """
        record = self._modules.get(name)
        if not record:
            return False

        # Check dependencies: all must be active
        for dep in record.dependencies:
            dep_record = self._modules.get(dep)
            if dep_record and dep_record.status in ("error", "disabled"):
                return False

        return record.should_run(current_cycle)

    def mark_run(self, name: str, cycle: int) -> None:
        """Mark a module as having run this cycle."""
        record = self._modules.get(name)
        if record:
            record.mark_run(cycle)

    def mark_error(self, name: str, error: str) -> None:
        """Mark a module as having errored."""
        record = self._modules.get(name)
        if record:
            record.mark_error(error)

    def disable_module(self, name: str) -> None:
        """Disable a module (won't run until re-enabled)."""
        record = self._modules.get(name)
        if record:
            record.enabled = False
            record.status = "disabled"

    def enable_module(self, name: str) -> None:
        """Re-enable a disabled module."""
        record = self._modules.get(name)
        if record:
            record.enabled = True
            record.status = "active"

    def get_activation_report(self) -> Dict[str, Any]:
        """Get report on all modules' activation status.

        Returns:
            Dict with modules by status category.
        """
        active = []
        orphaned = []
        disabled = []
        errored = []

        for name, record in self._modules.items():
            info = record.to_dict()
            if record.status == "disabled":
                disabled.append(info)
            elif record.status == "error":
                errored.append(info)
            elif record.total_runs == 0 and self._total_cycles > 5:
                record.status = "orphaned"
                orphaned.append(info)
            else:
                active.append(info)

        return {
            "total_modules": len(self._modules),
            "active": active,
            "orphaned": orphaned,
            "disabled": disabled,
            "errored": errored,
            "total_cycles": self._total_cycles,
        }

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get module dependency graph."""
        return {
            name: record.dependencies
            for name, record in self._modules.items()
        }

    def get_modules_to_run(self, cycle: int) -> List[str]:
        """Get list of modules that should run this cycle.

        Returns modules in dependency order.
        """
        to_run = []
        for name in self._modules:
            if self.should_run(name, cycle):
                to_run.append(name)

        # Simple topological sort: dependencies first
        ordered = []
        visited: Set[str] = set()

        def visit(name: str) -> None:
            if name in visited or name not in self._modules:
                return
            visited.add(name)
            for dep in self._modules[name].dependencies:
                if dep in to_run:
                    visit(dep)
            ordered.append(name)

        for name in to_run:
            visit(name)

        return ordered

    def complete_cycle(self, cycle: int) -> None:
        """Mark a scan cycle as complete. Log activation state."""
        self._current_cycle = cycle
        self._total_cycles += 1

        # Mark modules that didn't run as skipped
        for name, record in self._modules.items():
            if record.last_run_cycle != cycle and record.enabled:
                record.mark_skip()

        # Periodic logging
        if self._total_cycles % 10 == 0:
            report = self.get_activation_report()
            self._activation_log.append({
                "cycle": cycle,
                "active": len(report["active"]),
                "orphaned": len(report["orphaned"]),
                "errored": len(report["errored"]),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if len(self._activation_log) > 100:
                self._activation_log = self._activation_log[-50:]

    def get_status(self) -> Dict[str, Any]:
        """Get activation map status."""
        return {
            "total_modules": len(self._modules),
            "total_cycles": self._total_cycles,
            "current_cycle": self._current_cycle,
            "recent_logs": self._activation_log[-5:],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist activation state."""
        try:
            module_states = {
                name: {
                    "total_runs": record.total_runs,
                    "total_skips": record.total_skips,
                    "last_run_cycle": record.last_run_cycle,
                    "status": record.status,
                    "enabled": record.enabled,
                }
                for name, record in self._modules.items()
            }

            data = {
                "version": 1,
                "total_cycles": self._total_cycles,
                "current_cycle": self._current_cycle,
                "module_states": module_states,
                "activation_log": self._activation_log[-50:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"ModuleActivationMap: Failed to save: {e}")

    def _load_state(self) -> None:
        """Load persisted state."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            if isinstance(data, dict):
                self._total_cycles = data.get("total_cycles", 0)
                self._current_cycle = data.get("current_cycle", 0)
                self._activation_log = data.get("activation_log", [])
        except Exception as e:
            logger.debug(f"ModuleActivationMap: Failed to load: {e}")
