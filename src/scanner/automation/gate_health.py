"""Gate model health tracker with degradation detection.

US-063: Track per-gate rolling accuracy during live inference.
When gate accuracy drops below threshold, flag for heuristic fallback.

Usage:
    tracker = GateHealthTracker()
    tracker.record_prediction("momentum", predicted=True, actual=True)
    health = tracker.get_gate_health()
    # health = {"momentum": {"accuracy": 0.62, "degraded": False, ...}}
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GATE_ACCURACY_PATH = _PROJECT_ROOT / "trained_data" / "models" / "gate_accuracy.json"

# Gate types and their heuristic fallbacks
GATE_FALLBACKS = {
    "momentum": "adx_threshold",     # Fallback: ADX > 25 = momentum present
    "confidence": "vote_threshold",   # Fallback: weighted_vote > 0.65
    "risk": "atr_threshold",          # Fallback: ATR within 1.5x historical
    "meta_labeler": "ensemble_agree", # Fallback: 3/5 models agree
    "tcn_regime": "spread_based",     # Fallback: spread < 2x normal = not extreme
}

DEGRADATION_THRESHOLD = 0.50  # Below 50% = worse than random
MIN_SAMPLES = 50  # Need 50+ predictions before flagging


class GateHealthTracker:
    """Tracks per-gate prediction accuracy with rolling window.

    Records gate predictions and compares against actual trade outcomes.
    When a gate's accuracy drops below threshold, flags it for fallback.
    """

    def __init__(
        self,
        window_size: int = 50,
        degradation_threshold: float = DEGRADATION_THRESHOLD,
        persistence_path: Optional[Path] = None,
    ):
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.persistence_path = persistence_path or GATE_ACCURACY_PATH

        # Rolling window of (predicted, actual) per gate
        self._predictions: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._degraded_gates: set = set()
        self._load()

    def _load(self) -> None:
        """Load persisted gate accuracy data."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            for gate, records in data.get("predictions", {}).items():
                dq = deque(maxlen=self.window_size)
                for r in records[-self.window_size:]:
                    dq.append(tuple(r))
                self._predictions[gate] = dq

            self._degraded_gates = set(data.get("degraded_gates", []))
        except Exception as e:
            logger.debug(f"Failed to load gate accuracy: {e}")

    def _save(self) -> None:
        """Persist gate accuracy data."""
        data = {
            "predictions": {
                gate: list(preds)
                for gate, preds in self._predictions.items()
            },
            "degraded_gates": list(self._degraded_gates),
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
                self.persistence_path.write_text(
                    json.dumps(data, indent=2), encoding="utf-8"
                )
        except Exception as e:
            logger.debug(f"Failed to save gate accuracy: {e}")

    def record_prediction(
        self,
        gate_name: str,
        predicted: bool,
        actual: bool,
    ) -> None:
        """Record a gate prediction and actual outcome.

        Args:
            gate_name: Name of the gate (momentum, confidence, risk, etc.)
            predicted: What the gate predicted (pass/fail)
            actual: What actually happened (trade won/lost)
        """
        correct = predicted == actual
        self._predictions[gate_name].append((int(predicted), int(actual), int(correct)))

        # Check for degradation after enough samples
        preds = self._predictions[gate_name]
        if len(preds) >= MIN_SAMPLES:
            accuracy = sum(p[2] for p in preds) / len(preds)

            if accuracy < self.degradation_threshold:
                if gate_name not in self._degraded_gates:
                    self._degraded_gates.add(gate_name)
                    logger.warning(
                        f"Gate DEGRADED: {gate_name} accuracy={accuracy:.1%} "
                        f"< {self.degradation_threshold:.0%} over {len(preds)} samples. "
                        f"Recommending fallback: {GATE_FALLBACKS.get(gate_name, 'none')}"
                    )
                    # Log observation
                    try:
                        from src.scanner.automation.observation_log import ObservationLog
                        ObservationLog().log_observation(
                            pair="ALL",
                            category="gate_degradation",
                            description=(
                                f"Gate {gate_name} degraded: accuracy={accuracy:.1%} "
                                f"over {len(preds)} predictions"
                            ),
                            metadata={
                                "gate": gate_name,
                                "accuracy": round(accuracy, 4),
                                "samples": len(preds),
                                "fallback": GATE_FALLBACKS.get(gate_name),
                            },
                        )
                    except Exception:
                        pass
            elif gate_name in self._degraded_gates:
                # Recovered
                self._degraded_gates.discard(gate_name)
                logger.info(
                    f"Gate RECOVERED: {gate_name} accuracy={accuracy:.1%} "
                    f">= {self.degradation_threshold:.0%}"
                )

        # Persist every 10 recordings
        total_records = sum(len(d) for d in self._predictions.values())
        if total_records % 10 == 0:
            self._save()

    def is_degraded(self, gate_name: str) -> bool:
        """Check if a specific gate is degraded."""
        return gate_name in self._degraded_gates

    def get_gate_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health summary for all tracked gates.

        Returns:
            Dict mapping gate name → health metrics.
        """
        health: Dict[str, Dict[str, Any]] = {}
        for gate, preds in self._predictions.items():
            n = len(preds)
            accuracy = sum(p[2] for p in preds) / n if n > 0 else 0.0
            health[gate] = {
                "accuracy": round(accuracy, 4),
                "samples": n,
                "degraded": gate in self._degraded_gates,
                "fallback": GATE_FALLBACKS.get(gate),
                "sufficient_data": n >= MIN_SAMPLES,
            }
        return health

    def get_fallback_recommendation(self, gate_name: str) -> Optional[str]:
        """Get the recommended fallback heuristic for a degraded gate."""
        if gate_name in self._degraded_gates:
            return GATE_FALLBACKS.get(gate_name)
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get tracker status for observability."""
        return {
            "gates_tracked": len(self._predictions),
            "degraded_gates": list(self._degraded_gates),
            "gate_health": self.get_gate_health(),
        }
