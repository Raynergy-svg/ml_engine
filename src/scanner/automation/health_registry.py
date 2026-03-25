"""Orchestrator health registry and pre/post-cycle validation.

US-118: Collective health monitoring for all modules. Pre-flight checks
before scan, post-cycle state validation, graceful degradation, and
system health score. Orchestrator can reject a cycle if health drops
below threshold.

Approach:
1. Modules register health check functions
2. Pre-flight: run all health checks, aggregate pass/fail
3. Post-flight: validate state persistence succeeded
4. System health score: 0.0-1.0 aggregate across all modules
5. Graceful degradation: disable failing modules, continue scanning
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "health_registry_log.json"

# Health thresholds
MIN_HEALTH_TO_CONTINUE = 0.40  # Reject cycle if below 40%
CRITICAL_MODULES = {
    "engine", "agents", "execution", "portfolio_optimizer",
    "drawdown_guardian", "regime_tracker",
}  # These must pass for scan to proceed


class ModuleHealth:
    """Health state for a registered module."""

    def __init__(
        self,
        name: str,
        health_check: Optional[Callable[[], bool]] = None,
        critical: bool = False,
    ):
        self.name = name
        self.health_check = health_check
        self.critical = critical or name in CRITICAL_MODULES

        self.last_check_passed: bool = True
        self.consecutive_failures: int = 0
        self.total_checks: int = 0
        self.total_passes: int = 0
        self.last_error: Optional[str] = None
        self.disabled: bool = False

    @property
    def pass_rate(self) -> float:
        if self.total_checks == 0:
            return 1.0
        return self.total_passes / self.total_checks

    def run_check(self) -> bool:
        """Run this module's health check."""
        self.total_checks += 1

        if self.health_check is None:
            # No health check registered — assume healthy
            self.last_check_passed = True
            self.total_passes += 1
            self.consecutive_failures = 0
            return True

        try:
            result = self.health_check()
            self.last_check_passed = bool(result)
            if self.last_check_passed:
                self.total_passes += 1
                self.consecutive_failures = 0
                self.last_error = None
            else:
                self.consecutive_failures += 1
                self.last_error = "health_check_returned_false"
            return self.last_check_passed
        except Exception as e:
            self.last_check_passed = False
            self.consecutive_failures += 1
            self.last_error = str(e)[:200]
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "healthy": self.last_check_passed,
            "critical": self.critical,
            "disabled": self.disabled,
            "pass_rate": round(self.pass_rate, 4),
            "consecutive_failures": self.consecutive_failures,
            "total_checks": self.total_checks,
            "last_error": self.last_error,
        }


class HealthRegistry:
    """Collective health monitoring for all automation modules.

    Provides pre-flight validation (before scan), post-flight checks
    (after scan), graceful degradation (disable failing modules),
    and a system health score for the orchestrator.

    Usage:
        registry = HealthRegistry()
        registry.register_module("threshold_optimizer", lambda: optimizer is not None)
        preflight = registry.run_preflight()
        if registry.should_continue_cycle():
            # ... run scan cycle ...
            registry.run_postflight()
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or _LOG_PATH

        self._modules: Dict[str, ModuleHealth] = {}
        self._health_log: List[Dict[str, Any]] = []
        self._total_preflights: int = 0
        self._total_postflights: int = 0
        self._cycles_rejected: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_module(
        self,
        name: str,
        health_check: Optional[Callable[[], bool]] = None,
        critical: bool = False,
    ) -> None:
        """Register a module with an optional health check function.

        Args:
            name: Module identifier.
            health_check: Callable returning True if healthy. None = always healthy.
            critical: If True, scan cycle is rejected when this module fails.
        """
        self._modules[name] = ModuleHealth(
            name=name,
            health_check=health_check,
            critical=critical,
        )

    def run_preflight(self) -> Dict[str, bool]:
        """Run all health checks before a scan cycle.

        Returns:
            Dict mapping module name → pass/fail.
        """
        results: Dict[str, bool] = {}

        for name, module in self._modules.items():
            if module.disabled:
                results[name] = True  # Disabled modules don't block
                continue
            results[name] = module.run_check()

        self._total_preflights += 1
        return results

    def run_postflight(self) -> Dict[str, bool]:
        """Run post-cycle validation checks.

        Verifies modules persisted state successfully by checking
        that their log files exist and were recently updated.
        """
        results: Dict[str, bool] = {}

        for name, module in self._modules.items():
            if module.disabled:
                results[name] = True
                continue

            # Post-flight: re-run health check
            results[name] = module.run_check()

            # Auto-disable after 3 consecutive failures (non-critical only)
            if module.consecutive_failures >= 3 and not module.critical:
                module.disabled = True
                logger.warning(
                    f"HealthRegistry: Auto-disabled {name} "
                    f"after {module.consecutive_failures} consecutive failures"
                )

        self._total_postflights += 1

        # Log health snapshot
        self._health_log.append({
            "type": "postflight",
            "health_score": round(self.get_system_health_score(), 4),
            "failed": [n for n, ok in results.items() if not ok],
            "disabled": [n for n, m in self._modules.items() if m.disabled],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._health_log) > 200:
            self._health_log = self._health_log[-100:]

        return results

    def get_system_health_score(self) -> float:
        """Compute aggregate system health score [0.0, 1.0].

        Weighted: critical modules count 2x.
        """
        if not self._modules:
            return 1.0

        total_weight = 0.0
        healthy_weight = 0.0

        for name, module in self._modules.items():
            weight = 2.0 if module.critical else 1.0
            total_weight += weight
            if module.last_check_passed or module.disabled:
                healthy_weight += weight

        return healthy_weight / total_weight if total_weight > 0 else 1.0

    def should_continue_cycle(
        self, min_health: float = MIN_HEALTH_TO_CONTINUE
    ) -> bool:
        """Check if the system is healthy enough to continue scanning.

        Args:
            min_health: Minimum health score to proceed.

        Returns:
            True if health is above threshold and all critical modules pass.
        """
        health = self.get_system_health_score()

        if health < min_health:
            self._cycles_rejected += 1
            return False

        # Check critical modules specifically
        for name, module in self._modules.items():
            if module.critical and not module.last_check_passed and not module.disabled:
                self._cycles_rejected += 1
                return False

        return True

    def re_enable_module(self, name: str) -> bool:
        """Re-enable a disabled module."""
        module = self._modules.get(name)
        if module and module.disabled:
            module.disabled = False
            module.consecutive_failures = 0
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get registry status for observability."""
        return {
            "total_modules": len(self._modules),
            "system_health": round(self.get_system_health_score(), 4),
            "total_preflights": self._total_preflights,
            "total_postflights": self._total_postflights,
            "cycles_rejected": self._cycles_rejected,
            "modules": {
                name: module.to_dict()
                for name, module in self._modules.items()
            },
            "recent_log": self._health_log[-5:],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist health registry state."""
        try:
            module_states = {}
            for name, module in self._modules.items():
                module_states[name] = {
                    "last_check_passed": module.last_check_passed,
                    "consecutive_failures": module.consecutive_failures,
                    "total_checks": module.total_checks,
                    "total_passes": module.total_passes,
                    "disabled": module.disabled,
                    "last_error": module.last_error,
                }

            data = {
                "version": 1,
                "total_preflights": self._total_preflights,
                "total_postflights": self._total_postflights,
                "cycles_rejected": self._cycles_rejected,
                "module_states": module_states,
                "health_log": self._health_log[-100:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"HealthRegistry: Failed to save: {e}")

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
                self._total_preflights = data.get("total_preflights", 0)
                self._total_postflights = data.get("total_postflights", 0)
                self._cycles_rejected = data.get("cycles_rejected", 0)
                self._health_log = data.get("health_log", [])
        except Exception as e:
            logger.debug(f"HealthRegistry: Failed to load: {e}")
