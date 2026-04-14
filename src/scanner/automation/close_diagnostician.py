"""Close-event diagnosis: structured reasoning about WHY a trade closed."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class CloseReason(Enum):
    SL_HIT = "sl_hit"
    TP_HIT = "tp_hit"
    TRAILING_SL = "trailing_sl"
    MANUAL = "manual"
    TIMEOUT = "timeout"
    DRAWDOWN_GUARDIAN = "drawdown_guardian"
    REGIME_CHANGE = "regime_change"
    BREAKEVEN = "breakeven"


_EXIT_REASON_MAP = {
    "sl_hit": CloseReason.SL_HIT,
    "sl": CloseReason.SL_HIT,
    "tp_hit": CloseReason.TP_HIT,
    "tp": CloseReason.TP_HIT,
    "trailing": CloseReason.TRAILING_SL,
    "trailing_sl": CloseReason.TRAILING_SL,
    "timeout": CloseReason.TIMEOUT,
    "drawdown": CloseReason.DRAWDOWN_GUARDIAN,
    "drawdown_guardian": CloseReason.DRAWDOWN_GUARDIAN,
    "regime_change": CloseReason.REGIME_CHANGE,
    "regime": CloseReason.REGIME_CHANGE,
    "breakeven": CloseReason.BREAKEVEN,
    "manual": CloseReason.MANUAL,
}


@dataclass
class CloseDiagnosis:
    trade_id: str
    pair: str
    direction: str
    close_reason: CloseReason
    entry_price: float
    exit_price: float
    realized_pl: float
    pnl_pips: float
    duration_minutes: float
    regime_at_entry: str
    regime_at_exit: str
    regime_changed: bool
    confidence_at_entry: float
    agents_correct: List[str] = field(default_factory=list)
    agents_wrong: List[str] = field(default_factory=list)
    entry_still_valid_1h: Optional[bool] = None
    market_moved_after_close_pips: Optional[float] = None
    timestamp: str = ""


class CloseDiagnostician:
    """Maps raw trade journal entries to structured CloseDiagnosis."""

    def diagnose(self, trade_outcome: dict) -> CloseDiagnosis:
        close_reason = self._infer_close_reason(
            trade_outcome.get("exit_reason", "")
        )

        regime_at_entry = trade_outcome.get("regime_at_entry", "unknown")
        regime_at_exit = trade_outcome.get("regime_at_exit", "unknown")
        regime_changed = (
            regime_at_entry != regime_at_exit
            if regime_at_entry != "unknown" and regime_at_exit != "unknown"
            else False
        )

        agents_correct, agents_wrong = self._split_agents(trade_outcome)

        return CloseDiagnosis(
            trade_id=str(trade_outcome.get("trade_id", "")),
            pair=trade_outcome.get("pair", ""),
            direction=trade_outcome.get("direction", ""),
            close_reason=close_reason,
            entry_price=float(trade_outcome.get("entry_price", 0.0)),
            exit_price=float(trade_outcome.get("exit_price", 0.0)),
            realized_pl=float(trade_outcome.get("realized_pl", 0.0)),
            pnl_pips=float(trade_outcome.get("pnl_pips", 0.0)),
            duration_minutes=float(trade_outcome.get("duration_minutes", 0.0)),
            regime_at_entry=regime_at_entry,
            regime_at_exit=regime_at_exit,
            regime_changed=regime_changed,
            confidence_at_entry=float(
                trade_outcome.get("confidence_at_entry", 0.0)
            ),
            agents_correct=agents_correct,
            agents_wrong=agents_wrong,
            entry_still_valid_1h=trade_outcome.get("entry_still_valid_1h"),
            market_moved_after_close_pips=trade_outcome.get(
                "market_moved_after_close_pips"
            ),
            timestamp=trade_outcome.get("timestamp", ""),
        )

    def batch_diagnose(self, outcomes: List[dict]) -> List[CloseDiagnosis]:
        return [self.diagnose(o) for o in outcomes]

    @staticmethod
    def _infer_close_reason(exit_reason: str) -> CloseReason:
        key = exit_reason.strip().lower()
        return _EXIT_REASON_MAP.get(key, CloseReason.MANUAL)

    @staticmethod
    def _split_agents(trade_outcome: dict) -> tuple:
        agent_reasons = trade_outcome.get("agent_reasons", "")
        if not agent_reasons:
            return [], []

        trade_won = trade_outcome.get("trade_won", False)
        direction = trade_outcome.get("direction", "long").lower()

        agents_correct: List[str] = []
        agents_wrong: List[str] = []

        for entry in agent_reasons.split(","):
            entry = entry.strip()
            if not entry:
                continue

            if ":" in entry:
                agent_name, vote = entry.rsplit(":", 1)
                agent_name = agent_name.strip()
                vote = vote.strip().lower()
            else:
                agent_name = entry
                vote = ""

            agreed_with_direction = vote in (direction, "agree", "yes", "buy" if direction == "long" else "sell")

            if trade_won:
                if agreed_with_direction:
                    agents_correct.append(agent_name)
                else:
                    agents_wrong.append(agent_name)
            else:
                if agreed_with_direction:
                    agents_wrong.append(agent_name)
                else:
                    agents_correct.append(agent_name)

        return agents_correct, agents_wrong
