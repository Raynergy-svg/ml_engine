"""
Gate Attribution Engine — Phase 53 (US-328).

Tracks which pre-execution gates reject setups and measures rejection quality.
Instrumentation layer — does not change trading logic, adds visibility into
gate effectiveness.

For each gate decision (pass/reject), the engine records:
  - gate_name: which gate made the decision
  - reason: human-readable rejection reason (empty if passed)
  - confidence_at_decision: confidence score when gate evaluated
  - rejected: whether the gate blocked the trade
  - timestamp: when the decision was made

Aggregate metrics per gate per day:
  - rejections: total rejections
  - passes: total passes
  - rejection_pct: rejections / (rejections + passes)
  - good_rejection_rate: of rejected setups, what % would have lost
    (computed by back-matching rejection confidence against historical
     journal outcomes at similar confidence levels)

Usage:
    engine = GateAttributionEngine()
    engine.record_decision("spread_filter", rejected=True,
                           reason="spread 2.1 > threshold 1.5",
                           confidence=0.58, pair="EUR_USD")
    engine.record_decision("calendar_filter", rejected=False,
                           confidence=0.58, pair="EUR_USD")
    report = engine.generate_report()
    # {
    #   "spread_filter": {"rejections": 14, "passes": 88, ...},
    #   "calendar_filter": {"rejections": 3, "passes": 99, ...},
    # }
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Persistence path ────────────────────────────────────────────────
_DEFAULT_PERSISTENCE_PATH = "trained_data/gate_attribution.json"
_MAX_DECISION_LOG_SIZE = 2000  # Rolling window of raw decisions


@dataclass
class GateDecision:
    """A single gate decision record."""
    gate_name: str
    rejected: bool
    reason: str = ""
    confidence: float = 0.0
    pair: str = ""
    timestamp: float = 0.0  # Unix epoch

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "rejected": self.rejected,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "pair": self.pair,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GateDecision":
        return cls(
            gate_name=str(d.get("gate_name", "")),
            rejected=bool(d.get("rejected", False)),
            reason=str(d.get("reason", "")),
            confidence=float(d.get("confidence", 0.0)),
            pair=str(d.get("pair", "")),
            timestamp=float(d.get("timestamp", 0.0)),
        )


@dataclass
class GateStats:
    """Aggregated stats for a single gate."""
    gate_name: str = ""
    rejections: int = 0
    passes: int = 0
    rejection_pct: float = 0.0
    good_rejection_rate: float = 0.0  # Of rejected, what % would have lost
    avg_rejection_confidence: float = 0.0
    avg_pass_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "rejections": self.rejections,
            "passes": self.passes,
            "rejection_pct": round(self.rejection_pct, 4),
            "good_rejection_rate": round(self.good_rejection_rate, 4),
            "avg_rejection_confidence": round(self.avg_rejection_confidence, 4),
            "avg_pass_confidence": round(self.avg_pass_confidence, 4),
        }


@dataclass
class GateAttributionConfig:
    """Configuration for gate attribution engine."""
    persistence_path: str = _DEFAULT_PERSISTENCE_PATH
    max_decisions: int = _MAX_DECISION_LOG_SIZE
    # Thresholds for flagging gates
    bad_gate_threshold: float = 0.30  # <30% good rejection = killing good trades
    good_gate_threshold: float = 0.70  # >70% good rejection = protecting profit


class GateAttributionEngine:
    """Tracks gate decisions and measures rejection quality.

    Provides visibility into which gates are rejecting setups and whether
    those rejections are protecting profit (good) or killing opportunity (bad).
    """

    def __init__(self, config: Optional[GateAttributionConfig] = None):
        self.config = config or GateAttributionConfig()
        self._decisions: List[GateDecision] = []
        self._daily_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        # Structure: {date_str: {gate_name: {"rejections": N, "passes": N}}}
        self._loaded = False
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        path = self.config.persistence_path
        if not os.path.exists(path):
            self._loaded = True
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            # Validate structure
            if not isinstance(data, dict):
                logger.warning("Phase 53 (US-328): gate attribution file invalid structure, starting fresh")
                self._loaded = True
                return
            # Load decisions
            raw_decisions = data.get("decisions", [])
            if isinstance(raw_decisions, list):
                self._decisions = [
                    GateDecision.from_dict(d) for d in raw_decisions
                    if isinstance(d, dict)
                ]
            # Load daily counts
            raw_daily = data.get("daily_counts", {})
            if isinstance(raw_daily, dict):
                self._daily_counts = raw_daily
            self._loaded = True
            logger.info(
                "Phase 53 (US-328): Loaded gate attribution state — %d decisions, %d days",
                len(self._decisions), len(self._daily_counts),
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Phase 53 (US-328): Could not load gate attribution state: %s", e)
            self._loaded = True

    def save_state(self) -> None:
        """Persist state to disk with retry + atomic write (US-342)."""
        path = self.config.persistence_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "decisions": [d.to_dict() for d in self._decisions],
                "daily_counts": self._daily_counts,
            }
            try:
                from src.scanner.safe_io import resilient_save
                resilient_save(path, data, context="gate_attribution")
            except ImportError:
                tmp_path = path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                os.rename(tmp_path, path)
        except Exception as e:
            logger.error("Phase 53 (US-328): Failed to save gate attribution state: %s", e)

    def record_decision(
        self,
        gate_name: str,
        rejected: bool,
        reason: str = "",
        confidence: float = 0.0,
        pair: str = "",
    ) -> None:
        """Record a gate decision.

        Args:
            gate_name: Identifier for the gate (e.g. "spread_filter", "drawdown_adapter")
            rejected: True if the gate blocked the trade
            reason: Human-readable rejection reason (empty for passes)
            confidence: Confidence score at the time of evaluation
            pair: Instrument being evaluated
        """
        decision = GateDecision(
            gate_name=gate_name,
            rejected=rejected,
            reason=reason if rejected else "",
            confidence=confidence,
            pair=pair,
            timestamp=time.time(),
        )
        self._decisions.append(decision)

        # Trim rolling window
        if len(self._decisions) > self.config.max_decisions:
            self._decisions = self._decisions[-self.config.max_decisions:]

        # Update daily counts
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if date_str not in self._daily_counts:
            self._daily_counts[date_str] = {}
        if gate_name not in self._daily_counts[date_str]:
            self._daily_counts[date_str][gate_name] = {"rejections": 0, "passes": 0}

        if rejected:
            self._daily_counts[date_str][gate_name]["rejections"] += 1
        else:
            self._daily_counts[date_str][gate_name]["passes"] += 1

    def compute_good_rejection_rate(
        self,
        gate_name: str,
        journal_path: str = "trained_data/trade_journal_rl.json",
    ) -> float:
        """Compute good rejection rate for a gate.

        For rejected setups, estimate what % would have lost by checking
        historical journal outcomes at similar confidence levels.

        A "good rejection" is one where the confidence level of the rejected
        trade corresponds to a losing outcome in historical data.

        Args:
            gate_name: Gate to analyze
            journal_path: Path to trade journal

        Returns:
            Fraction of rejections that were "good" (0.0-1.0)
        """
        # Get rejected decisions for this gate
        rejected = [
            d for d in self._decisions
            if d.gate_name == gate_name and d.rejected
        ]
        if not rejected:
            return 0.0

        # Load journal outcomes
        try:
            if not os.path.exists(journal_path):
                return 0.0
            with open(journal_path, "r") as f:
                journal = json.load(f)
            if not isinstance(journal, list):
                return 0.0
        except (json.JSONDecodeError, OSError):
            return 0.0

        # Build confidence → win_rate lookup from journal
        # Group outcomes by confidence decile
        conf_outcomes: Dict[int, List[bool]] = {}
        for trade in journal:
            if not isinstance(trade, dict):
                continue
            conf = float(trade.get("confidence", trade.get("meta_confidence", 0.0)))
            outcome = trade.get("outcome", trade.get("result", ""))
            if not conf or not outcome:
                continue
            decile = min(int(conf * 10), 9)  # 0-9
            is_win = str(outcome).lower() in ("win", "profit", "tp_hit", "take_profit")
            if decile not in conf_outcomes:
                conf_outcomes[decile] = []
            conf_outcomes[decile].append(is_win)

        if not conf_outcomes:
            return 0.0

        # For each rejection, check if trades at that confidence level tend to lose
        good_rejections = 0
        for decision in rejected:
            decile = min(int(decision.confidence * 10), 9)
            outcomes = conf_outcomes.get(decile, [])
            if not outcomes:
                # No data at this confidence — assume neutral (not good)
                continue
            win_rate = sum(outcomes) / len(outcomes)
            # If win rate at this confidence < 50%, rejection was "good"
            if win_rate < 0.50:
                good_rejections += 1

        return good_rejections / len(rejected)

    def generate_report(
        self,
        journal_path: str = "trained_data/trade_journal_rl.json",
    ) -> Dict[str, Dict[str, Any]]:
        """Generate summary report of gate effectiveness.

        Returns:
            Dict of {gate_name: {rejections, passes, rejection_pct,
                                  good_rejection_rate, avg_rejection_confidence,
                                  avg_pass_confidence, effectiveness_flag}}
        """
        # Aggregate across all decisions
        gate_data: Dict[str, Dict[str, list]] = {}
        for d in self._decisions:
            if d.gate_name not in gate_data:
                gate_data[d.gate_name] = {
                    "rejection_confs": [],
                    "pass_confs": [],
                }
            if d.rejected:
                gate_data[d.gate_name]["rejection_confs"].append(d.confidence)
            else:
                gate_data[d.gate_name]["pass_confs"].append(d.confidence)

        report: Dict[str, Dict[str, Any]] = {}
        for gate_name, data in gate_data.items():
            rej_count = len(data["rejection_confs"])
            pass_count = len(data["pass_confs"])
            total = rej_count + pass_count

            avg_rej_conf = (
                sum(data["rejection_confs"]) / rej_count
                if rej_count > 0 else 0.0
            )
            avg_pass_conf = (
                sum(data["pass_confs"]) / pass_count
                if pass_count > 0 else 0.0
            )

            good_rej_rate = self.compute_good_rejection_rate(gate_name, journal_path)

            # Determine effectiveness flag
            if rej_count == 0:
                flag = "NO_REJECTIONS"
            elif good_rej_rate < self.config.bad_gate_threshold:
                flag = "KILLING_GOOD_TRADES"  # <30% good rejections
            elif good_rej_rate > self.config.good_gate_threshold:
                flag = "PROTECTING_PROFIT"  # >70% good rejections
            else:
                flag = "NEUTRAL"

            report[gate_name] = {
                "rejections": rej_count,
                "passes": pass_count,
                "rejection_pct": round(rej_count / total, 4) if total > 0 else 0.0,
                "good_rejection_rate": round(good_rej_rate, 4),
                "avg_rejection_confidence": round(avg_rej_conf, 4),
                "avg_pass_confidence": round(avg_pass_conf, 4),
                "effectiveness_flag": flag,
            }

        return report

    def get_daily_summary(self, date_str: Optional[str] = None) -> Dict[str, Dict[str, int]]:
        """Get gate rejection/pass counts for a specific day.

        Args:
            date_str: Date in YYYY-MM-DD format. Defaults to today.

        Returns:
            {gate_name: {"rejections": N, "passes": N}}
        """
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return dict(self._daily_counts.get(date_str, {}))

    def get_flagged_gates(
        self,
        journal_path: str = "trained_data/trade_journal_rl.json",
    ) -> Dict[str, List[str]]:
        """Identify gates that may need tuning.

        Returns:
            {"killing_good_trades": [gate_names...],
             "protecting_profit": [gate_names...]}
        """
        report = self.generate_report(journal_path)
        result: Dict[str, List[str]] = {
            "killing_good_trades": [],
            "protecting_profit": [],
        }
        for gate_name, stats in report.items():
            flag = stats.get("effectiveness_flag", "")
            if flag == "KILLING_GOOD_TRADES":
                result["killing_good_trades"].append(gate_name)
            elif flag == "PROTECTING_PROFIT":
                result["protecting_profit"].append(gate_name)
        return result

    def get_decisions_count(self) -> int:
        """Return total number of recorded decisions."""
        return len(self._decisions)

    def get_gate_names(self) -> List[str]:
        """Return list of unique gate names seen."""
        return sorted(set(d.gate_name for d in self._decisions))

    def clear(self) -> None:
        """Clear all recorded decisions and daily counts."""
        self._decisions.clear()
        self._daily_counts.clear()
