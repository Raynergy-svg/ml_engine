"""
Tranche Tracker — Phase 51 (US-318).

Tracks partial profit tranche state per trade and orchestrates
breakeven SL movement after tranche closures.

After first TAKE_PARTIAL at 1.0R: move SL to breakeven (entry_price).
After second TAKE_PARTIAL at 1.5R: move SL to entry + 0.5R (lock profit).

Persists tranche state to JSON for cross-restart continuity.

Usage:
    tracker = TrancheTracker()
    tracker.record_tranche_close(trade_id, tranche_idx=0, r_at_close=1.05)
    sl_action = tracker.get_sl_adjustment(trade_id, entry_price, sl_distance)
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrancheState:
    """State for a single trade's tranche tracking."""
    trade_id: str
    pair: str
    entry_price: float
    original_sl: float
    direction: str  # "LONG" or "SHORT"
    tranches_closed: List[int] = field(default_factory=list)
    r_at_close: List[float] = field(default_factory=list)
    current_sl: float = 0.0
    sl_moved_to_breakeven: bool = False
    sl_moved_to_profit_lock: bool = False


@dataclass
class SLAdjustment:
    """Recommended SL adjustment after tranche close."""
    should_move: bool = False
    new_sl: float = 0.0
    reason: str = ""
    adjustment_type: str = ""  # "breakeven", "profit_lock", "none"


class TrancheTracker:
    """Tracks partial profit tranches and SL adjustments per trade."""

    def __init__(self, state_path: str = "trained_data/tranche_state.json"):
        self.state_path = state_path
        self._states: Dict[str, TrancheState] = {}
        self._load_state()

    def register_trade(
        self,
        trade_id: str,
        pair: str,
        entry_price: float,
        original_sl: float,
        direction: str,
    ) -> None:
        """Register a new trade for tranche tracking."""
        self._states[trade_id] = TrancheState(
            trade_id=trade_id,
            pair=pair,
            entry_price=entry_price,
            original_sl=original_sl,
            direction=direction.upper(),
            current_sl=original_sl,
        )
        self._save_state()
        logger.debug(
            "Phase 51: Registered trade %s (%s %s) for tranche tracking",
            trade_id, pair, direction,
        )

    def record_tranche_close(
        self, trade_id: str, tranche_idx: int, r_at_close: float
    ) -> SLAdjustment:
        """Record a tranche closure and compute SL adjustment.

        Args:
            trade_id: OANDA trade ID
            tranche_idx: Which tranche was closed (0, 1, 2)
            r_at_close: R-multiple at time of close

        Returns:
            SLAdjustment with recommended new SL.
        """
        state = self._states.get(trade_id)
        if state is None:
            logger.warning("Phase 51: Trade %s not registered for tranche tracking", trade_id)
            return SLAdjustment(reason="Trade not registered")

        # Record the close
        if tranche_idx not in state.tranches_closed:
            state.tranches_closed.append(tranche_idx)
            state.r_at_close.append(r_at_close)

        # Compute SL distance (entry to original SL)
        sl_distance = abs(state.entry_price - state.original_sl)

        adjustment = SLAdjustment()

        n_closed = len(state.tranches_closed)

        if n_closed >= 1 and not state.sl_moved_to_breakeven:
            # After first tranche: move SL to breakeven
            new_sl = state.entry_price
            state.sl_moved_to_breakeven = True
            state.current_sl = new_sl
            adjustment = SLAdjustment(
                should_move=True,
                new_sl=new_sl,
                reason=f"Tranche {tranche_idx} closed at {r_at_close:.2f}R — SL to breakeven",
                adjustment_type="breakeven",
            )
            logger.info(
                "Phase 51 US-318: %s %s — SL moved to breakeven %.5f after tranche %d",
                state.pair, trade_id, new_sl, tranche_idx,
            )

        elif n_closed >= 2 and not state.sl_moved_to_profit_lock:
            # After second tranche: move SL to entry + 0.5R (lock profit)
            if state.direction == "LONG":
                new_sl = state.entry_price + (sl_distance * 0.5)
            else:
                new_sl = state.entry_price - (sl_distance * 0.5)

            state.sl_moved_to_profit_lock = True
            state.current_sl = new_sl
            adjustment = SLAdjustment(
                should_move=True,
                new_sl=new_sl,
                reason=f"Tranche {tranche_idx} closed at {r_at_close:.2f}R — SL to entry+0.5R (profit lock)",
                adjustment_type="profit_lock",
            )
            logger.info(
                "Phase 51 US-318: %s %s — SL locked at %.5f (+0.5R) after tranche %d",
                state.pair, trade_id, new_sl, tranche_idx,
            )

        self._save_state()
        return adjustment

    def get_state(self, trade_id: str) -> Optional[TrancheState]:
        """Get tranche state for a trade."""
        return self._states.get(trade_id)

    def remove_trade(self, trade_id: str) -> None:
        """Remove a closed trade from tracking."""
        if trade_id in self._states:
            del self._states[trade_id]
            self._save_state()

    def get_active_trades(self) -> Dict[str, TrancheState]:
        """Return all actively tracked trades."""
        return dict(self._states)

    def _load_state(self) -> None:
        """Load tranche state from disk."""
        try:
            with open(self.state_path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            for tid, sdata in data.get("trades", {}).items():
                if isinstance(sdata, dict):
                    self._states[tid] = TrancheState(
                        trade_id=sdata.get("trade_id", tid),
                        pair=sdata.get("pair", ""),
                        entry_price=sdata.get("entry_price", 0.0),
                        original_sl=sdata.get("original_sl", 0.0),
                        direction=sdata.get("direction", "LONG"),
                        tranches_closed=sdata.get("tranches_closed", []),
                        r_at_close=sdata.get("r_at_close", []),
                        current_sl=sdata.get("current_sl", 0.0),
                        sl_moved_to_breakeven=sdata.get("sl_moved_to_breakeven", False),
                        sl_moved_to_profit_lock=sdata.get("sl_moved_to_profit_lock", False),
                    )
            logger.debug("Phase 51: Loaded tranche state for %d trades", len(self._states))
        except (FileNotFoundError, json.JSONDecodeError):
            self._states = {}

    def _save_state(self) -> None:
        """Save tranche state atomically."""
        try:
            os.makedirs(os.path.dirname(self.state_path) if os.path.dirname(self.state_path) else ".", exist_ok=True)
            data = {
                "version": 1,
                "trades": {tid: asdict(s) for tid, s in self._states.items()},
            }
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp, self.state_path)
        except Exception as e:
            logger.error("Phase 51: Failed to save tranche state: %s", e)
