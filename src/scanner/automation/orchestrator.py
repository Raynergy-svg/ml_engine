"""ML Agent Orchestrator — full scan→trade→learn→tune loop.

Coordinates all automation modules into a single improvement cycle.
This is the brain of Buddy's autonomous evolution.

Architecture:
    Scanner → Agents → Gates → Execution → OANDA
        ↑                                      ↓
        └── Config Tuner ← Rules ← Learnings ← RL Feedback ← Trade Outcomes
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Result of a single orchestration cycle."""

    cycle_id: str = ""
    timestamp: str = ""
    scan_duration_secs: float = 0.0
    pairs_scanned: int = 0
    tradeable_setups: int = 0
    trades_executed: int = 0
    learnings_extracted: int = 0
    rules_promoted: int = 0
    config_adjustments: int = 0
    observations_logged: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "scan_duration_secs": self.scan_duration_secs,
            "pairs_scanned": self.pairs_scanned,
            "tradeable_setups": self.tradeable_setups,
            "trades_executed": self.trades_executed,
            "learnings_extracted": self.learnings_extracted,
            "rules_promoted": self.rules_promoted,
            "config_adjustments": self.config_adjustments,
            "observations_logged": self.observations_logged,
            "errors": self.errors,
        }


class Orchestrator:
    """Coordinates the full scan→trade→learn→tune improvement cycle.

    Each call to `run_cycle()` performs:
    1. SCAN  — Run multi-pair scanner with agent consensus
    2. LOG   — Record observations from scan (even non-tradeable)
    3. TRADE — Execute qualifying setups via OANDA (if auto_execute)
    4. SYNC  — Sync closed trades from OANDA + run RL weight updates
    5. LEARN — Extract learnings from trade outcomes
    6. PROMOTE — Check if any learnings should become rules
    7. TUNE  — Apply rules to config parameters
    8. STATE — Update session state
    9. TRACK — Record improvement metrics
    """

    def __init__(self, project_root: Optional[str] = None, auto_execute: bool = False):
        self._root = Path(project_root) if project_root else Path.cwd()
        self._auto_execute = auto_execute
        self._cycle_count = 0
        self._session_start = datetime.utcnow()

        # Initialize all automation modules (lazy — they handle missing deps gracefully)
        self._state = None
        self._learner = None
        self._tuner = None
        self._tracker = None
        self._observer = None

    def _init_modules(self):
        """Lazy-initialize automation modules."""
        if self._state is not None:
            return  # Already initialized

        root_str = str(self._root)

        try:
            from src.scanner.automation.state_engine import StateEngine
            self._state = StateEngine(base_dir=root_str)
        except Exception as e:
            logger.warning(f"StateEngine init failed: {e}")
            self._state = None

        try:
            from src.scanner.automation.learning_engine import LearningEngine
            self._learner = LearningEngine(project_root=root_str)
        except Exception as e:
            logger.warning(f"LearningEngine init failed: {e}")
            self._learner = None

        try:
            from src.scanner.automation.config_tuner import ConfigTuner
            self._tuner = ConfigTuner(project_root=root_str)
        except Exception as e:
            logger.warning(f"ConfigTuner init failed: {e}")
            self._tuner = None

        try:
            from src.scanner.automation.improvement_tracker import ImprovementTracker
            self._tracker = ImprovementTracker(project_root=root_str)
        except Exception as e:
            logger.warning(f"ImprovementTracker init failed: {e}")
            self._tracker = None

        try:
            from src.scanner.automation.observation_log import ObservationLog
            self._observer = ObservationLog(project_root=root_str)
        except Exception as e:
            logger.warning(f"ObservationLog init failed: {e}")
            self._observer = None

    def run_cycle(
        self,
        profile: str = "balanced",
        pairs: Optional[List[str]] = None,
        granularity: str = "H1",
    ) -> OrchestrationResult:
        """Execute a single full improvement cycle.

        Args:
            profile: Scan profile (balanced, conservative, aggressive, smart)
            pairs: Currency pairs to scan (None = all defaults)
            granularity: OANDA granularity (H1, M15, H4, D)

        Returns:
            OrchestrationResult with metrics from the cycle.
        """
        self._init_modules()
        self._cycle_count += 1
        result = OrchestrationResult(
            cycle_id=f"cycle_{self._cycle_count}_{datetime.utcnow().strftime('%H%M%S')}",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        # ── Step 1: SCAN ─────────────────────────────────────────
        scan_result = None
        t0 = time.time()
        try:
            from src.scanner.config import ScannerConfig
            from src.scanner.engine import Scanner

            config = ScannerConfig(profile=profile)

            # Apply any tuner adjustments
            if self._tuner:
                try:
                    adjustments = self._tuner.apply_to_config(config)
                    result.config_adjustments = len(adjustments)
                except Exception as e:
                    logger.debug(f"Config tuner skipped: {e}")

            scanner = Scanner(config)
            scan_result = scanner.scan(pairs=pairs, granularity=granularity)
            result.scan_duration_secs = round(time.time() - t0, 1)

            if scan_result and scan_result.analyses:
                result.pairs_scanned = len(scan_result.analyses)
                result.tradeable_setups = sum(
                    1 for a in scan_result.analyses
                    if getattr(a, "gates_passed", False) and getattr(a, "is_tradeable", False)
                )
        except Exception as e:
            result.errors.append(f"scan_failed: {e}")
            logger.error(f"Scan failed: {e}")

        # ── Step 2: LOG OBSERVATIONS ─────────────────────────────
        if scan_result and self._observer:
            try:
                result.observations_logged = self._observer.log_scan_observations(
                    scan_result.analyses
                )
            except Exception as e:
                result.errors.append(f"observation_log_failed: {e}")

        # ── Step 3: RECORD SCAN TO STATE ─────────────────────────
        if self._state:
            try:
                self._state.record_scan_cycle(
                    scan_result_count=result.pairs_scanned,
                    tradeable_count=result.tradeable_setups,
                    duration_secs=result.scan_duration_secs,
                )
            except Exception as e:
                logger.debug(f"State scan record skipped: {e}")

        # ── Step 4: SYNC CLOSED TRADES + RL ──────────────────────
        closed_trades = []
        try:
            from src.scanner.execution import ExecutionManager
            em = ExecutionManager()
            closed_trades = em.sync_closed_trades_rl() or []
        except Exception as e:
            logger.debug(f"RL sync skipped: {e}")

        # ── Step 5: EXTRACT LEARNINGS ────────────────────────────
        if closed_trades and self._learner:
            for trade in closed_trades:
                try:
                    insights = self._learner.analyze_trade(trade)
                    if insights:
                        self._learner.append_to_learnings(insights)
                        result.learnings_extracted += len(insights)
                except Exception as e:
                    result.errors.append(f"learning_extraction_failed: {e}")

        # ── Step 6: CHECK RULE PROMOTIONS ────────────────────────
        if self._learner and result.learnings_extracted > 0:
            try:
                promotions = self._learner.check_promotions()
                result.rules_promoted = len(promotions)
            except Exception as e:
                logger.debug(f"Promotion check skipped: {e}")

        # ── Step 7: CONSOLIDATE IF NEEDED ────────────────────────
        if self._learner:
            try:
                audit = self._learner.audit()
                if audit.get("total_learnings", 0) > 30:
                    self._learner.consolidate()
            except Exception as e:
                logger.debug(f"Consolidation skipped: {e}")

        # ── Step 8: UPDATE STATE ─────────────────────────────────
        if self._state:
            try:
                self._state.save_state(
                    status="running" if self._auto_execute else "scanning",
                    last_cycle=result.to_dict(),
                )
            except Exception as e:
                logger.debug(f"State update skipped: {e}")

        # ── Step 9: TRACK IMPROVEMENT ────────────────────────────
        if self._tracker:
            try:
                self._tracker.record_session(
                    session_id=result.cycle_id,
                    scan_count=1,
                    trade_count=len(closed_trades),
                    wins=sum(1 for t in closed_trades if (t.get("pnl_pips", 0) or 0) > 0),
                    losses=sum(1 for t in closed_trades if (t.get("pnl_pips", 0) or 0) <= 0),
                    total_pnl=sum(t.get("pnl_usd", 0) or 0 for t in closed_trades),
                    learnings_extracted=result.learnings_extracted,
                    rules_promoted=result.rules_promoted,
                    config_adjustments=result.config_adjustments,
                )
            except Exception as e:
                logger.debug(f"Improvement tracking skipped: {e}")

        return result

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for the dashboard.

        Returns a dict suitable for JSON serialization with:
        - Module availability (which components are active)
        - Last cycle results
        - Improvement trends
        - Learning stats
        """
        self._init_modules()

        status: Dict[str, Any] = {
            "modules": {
                "state_engine": self._state is not None,
                "learning_engine": self._learner is not None,
                "config_tuner": self._tuner is not None,
                "improvement_tracker": self._tracker is not None,
                "observation_log": self._observer is not None,
            },
            "session": {
                "started": self._session_start.isoformat() + "Z",
                "cycles_completed": self._cycle_count,
                "auto_execute": self._auto_execute,
            },
        }

        # Add state info
        if self._state:
            try:
                status["state"] = self._state.load_state()
            except Exception:
                status["state"] = {}

        # Add improvement trend
        if self._tracker:
            try:
                status["improvement_trend"] = self._tracker.get_trend()
            except Exception:
                status["improvement_trend"] = {}

        # Add learning audit
        if self._learner:
            try:
                status["learning_audit"] = self._learner.audit()
            except Exception:
                status["learning_audit"] = {}

        return status

    def get_improvement_report(self) -> str:
        """Generate a human-readable improvement report."""
        self._init_modules()
        if self._tracker:
            try:
                return self._tracker.generate_report()
            except Exception:
                pass
        return "Improvement tracker not available."
