"""30-second supervision tick — the heartbeat of the supervision-first runtime.

Wraps existing ExecutionManager methods (monitor_open_trades, apply_drawdown_guardian,
update_trailing_stops) into a unified tick cycle. Detects newly closed trades by
diffing position state between ticks.
"""

from typing import Any, Dict, List, Optional, Set

import structlog

logger = structlog.get_logger(__name__)


class SupervisionTick:
    """Called every 30s to supervise open positions and detect close events."""

    def __init__(
        self,
        execution_manager: Any = None,
        scanner: Any = None,
        control_plane: Any = None,
        supervision_core: Any = None,
    ):
        self._em = execution_manager
        self._scanner = scanner
        self._control_plane = control_plane
        self._supervision_core = supervision_core
        self._previous_open_trades: Set[str] = set()
        self._tick_count: int = 0

    def tick(self) -> Dict[str, Any]:
        """Run one supervision cycle. Returns a status dict."""
        self._tick_count += 1
        result: Dict[str, Any] = {
            "tick": self._tick_count,
            "positions_checked": 0,
            "exits_triggered": 0,
            "sl_modifications": 0,
            "closes_detected": 0,
            "diagnoses": [],
            "health_status": {},
        }

        # 1. Control plane health check
        if self._control_plane is not None:
            try:
                result["health_status"] = self._control_plane.check_health()
            except Exception as e:
                logger.debug("supervision_tick.health_error", error=str(e))

        if self._em is None:
            return result

        # 2. Monitor open trades
        current_open: Set[str] = set()
        try:
            statuses = self._em.monitor_open_trades()
            if statuses:
                result["positions_checked"] = len(statuses)
                for s in statuses:
                    tid = s.get("trade_id") or s.get("id", "")
                    if tid:
                        current_open.add(str(tid))
        except Exception as e:
            logger.debug("supervision_tick.monitor_error", error=str(e))

        # 3. Drawdown guardian
        try:
            guardian_msgs = self._em.apply_drawdown_guardian()
            if guardian_msgs:
                result["sl_modifications"] += len(guardian_msgs)
        except Exception as e:
            logger.debug("supervision_tick.guardian_error", error=str(e))

        # 4. Trailing stop updates
        try:
            trail_msgs = self._em.update_trailing_stops()
            if trail_msgs:
                result["sl_modifications"] += len(trail_msgs)
        except Exception as e:
            logger.debug("supervision_tick.trailing_error", error=str(e))

        # 5. Detect newly closed trades (diff against previous tick)
        closed_trades = self._previous_open_trades - current_open
        result["closes_detected"] = len(closed_trades)

        # 6. Process each close through supervision core
        if closed_trades and self._supervision_core is not None:
            for tid in closed_trades:
                try:
                    # Build trade_outcome from journal
                    outcome = self._get_trade_outcome(tid)
                    if outcome:
                        diagnosis = self._supervision_core.process_close_event(outcome)
                        result["diagnoses"].append({
                            "trade_id": tid,
                            "pair": outcome.get("pair", ""),
                            "close_reason": getattr(diagnosis, "close_reason", None),
                            "realized_pl": outcome.get("realized_pl", 0),
                        })
                except Exception as e:
                    logger.warning(
                        "supervision_tick.close_processing_error",
                        trade_id=tid,
                        error=str(e),
                    )

        # 7. Supervision core tick (expire policies, check cooldowns)
        if self._supervision_core is not None:
            try:
                self._supervision_core.tick()
            except Exception as e:
                logger.debug("supervision_tick.core_tick_error", error=str(e))

        # Update previous state for next tick
        self._previous_open_trades = current_open

        if result["closes_detected"] > 0:
            logger.info(
                "supervision_tick.closes_detected",
                count=result["closes_detected"],
                trades=list(closed_trades),
            )

        return result

    def _get_trade_outcome(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Look up trade outcome from journal for close diagnosis."""
        try:
            import json
            from pathlib import Path
            journal_path = Path("trained_data/trade_journal_rl.json")
            if not journal_path.exists():
                return None
            entries = json.loads(journal_path.read_text())
            for entry in reversed(entries):
                if str(entry.get("trade_id", "")) == trade_id:
                    outcome = entry.get("outcome", {})
                    if outcome:
                        # Merge entry-level and outcome-level fields
                        merged = dict(entry)
                        merged.update(outcome)
                        return merged
                    return entry
        except Exception as e:
            logger.debug("supervision_tick.journal_lookup_error", error=str(e))
        return None

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def open_trade_count(self) -> int:
        return len(self._previous_open_trades)
