"""Historical replay validator with drift detection.

US-120: Run current scan logic against historical data to validate
model drift and gate thresholds. Provides would-be P/L simulation,
drift detection, and A/B replay testing.

Approach:
1. Accept historical candle data and current scan config
2. Simulate scan → agent → gate → execution pipeline on historical data
3. Compute would-be P/L, Sharpe, max drawdown
4. Compare recent replay accuracy to baseline → detect drift
5. A/B test two configs on same historical data → pick winner
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "replay_validator_log.json"

# Replay parameters
MIN_CANDLES = 100      # Minimum candles for a valid replay
MAX_REPLAY_TRADES = 500  # Cap per replay run

# Drift detection
DRIFT_THRESHOLD = 0.10  # 10% accuracy drop = drift alert
BASELINE_WINDOW = 50    # Baseline computed from first N trades


class ReplayResult:
    """Container for a single replay run's results."""

    def __init__(self, pair: str, config_label: str = "default"):
        self.pair = pair
        self.config_label = config_label
        self.trades: List[Dict[str, Any]] = []
        self.signals_generated: int = 0
        self.signals_passed_gates: int = 0

    @property
    def total_pnl(self) -> float:
        return sum(t.get("pnl_pips", 0.0) for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.get("pnl_pips", 0) > 0)
        return wins / len(self.trades)

    @property
    def sharpe(self) -> float:
        pnls = [t.get("pnl_pips", 0.0) for t in self.trades]
        if len(pnls) < 2:
            return 0.0
        mean = float(np.mean(pnls))
        std = float(np.std(pnls))
        return mean / std if std > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in self.trades:
            cumulative += t.get("pnl_pips", 0.0)
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)
        return max_dd

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "config_label": self.config_label,
            "total_trades": len(self.trades),
            "total_pnl": round(self.total_pnl, 2),
            "win_rate": round(self.win_rate, 4),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 2),
            "signals_generated": self.signals_generated,
            "signals_passed_gates": self.signals_passed_gates,
            "gate_pass_rate": round(
                self.signals_passed_gates / max(self.signals_generated, 1), 4
            ),
        }


class ReplayValidator:
    """Validates trading logic against historical data.

    Bridges offline backtesting with the live system. Runs current
    scan configuration through historical candles to detect drift,
    validate gate thresholds, and A/B test config changes.

    Usage:
        validator = ReplayValidator()
        result = validator.replay_scan("EUR_USD", historical_candles, config)
        metrics = validator.compute_would_be_pnl(result)
        drift = validator.detect_drift(result, baseline_accuracy=0.55)
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or _LOG_PATH

        self._replay_log: List[Dict[str, Any]] = []
        self._drift_alerts: List[Dict[str, Any]] = []
        self._total_replays: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _pip_size(pair: str) -> float:
        """Return pip size for a currency pair."""
        return 0.01 if "JPY" in pair.upper() else 0.0001

    def replay_scan(
        self,
        pair: str,
        historical_candles: List[Dict[str, Any]],
        scan_config: Optional[Dict[str, Any]] = None,
        config_label: str = "default",
    ) -> ReplayResult:
        """Run scan logic on historical candle data.

        Args:
            pair: Currency pair.
            historical_candles: List of candle dicts with OHLCV data.
            scan_config: Override scan configuration (thresholds, etc.).
            config_label: Label for this config (for A/B comparison).

        Returns:
            ReplayResult with simulated trades and metrics.
        """
        result = ReplayResult(pair=pair, config_label=config_label)

        if len(historical_candles) < MIN_CANDLES:
            logger.debug(
                f"ReplayValidator: Too few candles ({len(historical_candles)} < {MIN_CANDLES})"
            )
            return result

        config = scan_config or {}
        confidence_threshold = config.get("confidence_threshold", 0.55)
        rr_threshold = config.get("rr_threshold", 1.2)
        atr_sl_mult = config.get("atr_sl_multiplier", 1.5)
        atr_tp_mult = config.get("atr_tp_multiplier", 2.0)
        pip_size = self._pip_size(pair)

        # Simulate scan through candles
        lookback = 20  # ATR lookback period

        for i in range(lookback, len(historical_candles)):
            if len(result.trades) >= MAX_REPLAY_TRADES:
                break

            candle = historical_candles[i]
            recent = historical_candles[max(0, i - lookback):i]

            # Compute simplified ATR from recent candles (raw price units)
            atr_raw = self._compute_atr(recent)
            if atr_raw == 0:
                continue

            # Convert ATR to pips
            atr_pips = atr_raw / pip_size

            # Simulate signal generation
            signal = self._simulate_signal(candle, recent)
            if signal is None:
                continue

            result.signals_generated += 1

            # Apply gate checks
            confidence = signal.get("confidence", 0.0)
            if confidence < confidence_threshold:
                continue

            sl_pips = atr_pips * atr_sl_mult
            tp_pips = atr_pips * atr_tp_mult

            if tp_pips / max(sl_pips, 0.0001) < rr_threshold:
                continue

            # SL/TP in raw price for outcome simulation
            sl_raw = atr_raw * atr_sl_mult
            tp_raw = atr_raw * atr_tp_mult

            result.signals_passed_gates += 1

            # Simulate trade outcome using future candles
            outcome = self._simulate_outcome(
                historical_candles, i, signal["direction"],
                sl_raw, tp_raw
            )

            # Convert outcome P/L from raw price to pips
            pnl_pips = outcome["pnl_pips"] / pip_size if outcome["pnl_pips"] != 0 else 0.0

            result.trades.append({
                "candle_idx": i,
                "direction": signal["direction"],
                "confidence": round(confidence, 4),
                "sl_pips": round(sl_pips, 1),
                "tp_pips": round(tp_pips, 1),
                "pnl_pips": round(pnl_pips, 1),
                "won": pnl_pips > 0,
                "bars_held": outcome.get("bars_held", 0),
            })

        self._total_replays += 1

        # Log replay
        self._replay_log.append({
            "pair": pair,
            "config_label": config_label,
            "n_candles": len(historical_candles),
            "result": result.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._replay_log) > 100:
            self._replay_log = self._replay_log[-50:]

        return result

    def compute_would_be_pnl(
        self, replay_result: ReplayResult
    ) -> Dict[str, Any]:
        """Compute would-be P/L metrics from a replay.

        Returns:
            Dict with total_pnl, sharpe, max_drawdown, win_rate.
        """
        return replay_result.to_dict()

    def detect_drift(
        self,
        replay_result: ReplayResult,
        baseline_accuracy: float = 0.55,
    ) -> Dict[str, Any]:
        """Detect model drift by comparing replay accuracy to baseline.

        Args:
            replay_result: Result from replay_scan.
            baseline_accuracy: Expected accuracy from historical performance.

        Returns:
            Dict with drift_score, is_drifting, and alert details.
        """
        if not replay_result.trades:
            return {"drift_score": 0.0, "is_drifting": False, "reason": "no_trades"}

        current_accuracy = replay_result.win_rate
        drift_score = baseline_accuracy - current_accuracy

        is_drifting = drift_score > DRIFT_THRESHOLD

        result = {
            "drift_score": round(drift_score, 4),
            "is_drifting": is_drifting,
            "current_accuracy": round(current_accuracy, 4),
            "baseline_accuracy": round(baseline_accuracy, 4),
            "pair": replay_result.pair,
            "n_trades": len(replay_result.trades),
        }

        if is_drifting:
            self._drift_alerts.append({
                **result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if len(self._drift_alerts) > 50:
                self._drift_alerts = self._drift_alerts[-25:]

        return result

    def compare_configs(
        self,
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
        pair: str,
        historical_candles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """A/B test two configurations on the same historical data.

        Returns:
            Dict with both results and recommendation.
        """
        result_a = self.replay_scan(pair, historical_candles, config_a, "config_a")
        result_b = self.replay_scan(pair, historical_candles, config_b, "config_b")

        metrics_a = result_a.to_dict()
        metrics_b = result_b.to_dict()

        # Score comparison: weighted by Sharpe (40%), win_rate (30%), total_pnl (30%)
        score_a = (
            0.4 * result_a.sharpe
            + 0.3 * result_a.win_rate
            + 0.3 * (result_a.total_pnl / max(abs(result_a.total_pnl) + abs(result_b.total_pnl), 1))
        )
        score_b = (
            0.4 * result_b.sharpe
            + 0.3 * result_b.win_rate
            + 0.3 * (result_b.total_pnl / max(abs(result_a.total_pnl) + abs(result_b.total_pnl), 1))
        )

        winner = "config_a" if score_a >= score_b else "config_b"

        return {
            "config_a": metrics_a,
            "config_b": metrics_b,
            "score_a": round(score_a, 4),
            "score_b": round(score_b, 4),
            "winner": winner,
            "margin": round(abs(score_a - score_b), 4),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get validator status for observability."""
        return {
            "total_replays": self._total_replays,
            "drift_alerts": len(self._drift_alerts),
            "recent_replays": self._replay_log[-5:],
            "recent_drift_alerts": self._drift_alerts[-3:],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_atr(self, candles: List[Dict[str, Any]]) -> float:
        """Compute Average True Range from candles."""
        if len(candles) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i].get("high", candles[i].get("mid", {}).get("h", 0))
            low = candles[i].get("low", candles[i].get("mid", {}).get("l", 0))
            prev_close = candles[i - 1].get("close", candles[i - 1].get("mid", {}).get("c", 0))

            try:
                high = float(high)
                low = float(low)
                prev_close = float(prev_close)
            except (TypeError, ValueError):
                continue

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        return float(np.mean(true_ranges)) if true_ranges else 0.0

    def _simulate_signal(
        self,
        candle: Dict[str, Any],
        recent: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Simulate signal generation from candle data.

        Uses simplified momentum-based signal for replay purposes.
        """
        if len(recent) < 5:
            return None

        # Get close prices
        closes = []
        for c in recent[-10:]:
            close = c.get("close", c.get("mid", {}).get("c"))
            try:
                closes.append(float(close))
            except (TypeError, ValueError):
                continue

        if len(closes) < 5:
            return None

        # Simple momentum signal
        short_ma = float(np.mean(closes[-5:]))
        long_ma = float(np.mean(closes))

        if short_ma == 0:
            return None

        momentum = (short_ma - long_ma) / long_ma

        # Only generate signal on meaningful momentum
        if abs(momentum) < 0.0005:
            return None

        direction = "LONG" if momentum > 0 else "SHORT"
        confidence = min(0.50 + abs(momentum) * 100, 0.90)

        return {
            "direction": direction,
            "confidence": confidence,
            "momentum": momentum,
        }

    def _simulate_outcome(
        self,
        candles: List[Dict[str, Any]],
        entry_idx: int,
        direction: str,
        sl_pips: float,
        tp_pips: float,
        max_bars: int = 50,
    ) -> Dict[str, Any]:
        """Simulate trade outcome from entry point.

        Walk forward through candles until SL or TP is hit.
        """
        entry_candle = candles[entry_idx]
        entry_price = float(
            entry_candle.get("close", entry_candle.get("mid", {}).get("c", 0))
        )

        if entry_price == 0:
            return {"pnl_pips": 0.0, "bars_held": 0}

        # sl_pips and tp_pips here are in raw price units (ATR * mult)
        for bars in range(1, min(max_bars, len(candles) - entry_idx)):
            future = candles[entry_idx + bars]
            high = float(future.get("high", future.get("mid", {}).get("h", entry_price)))
            low = float(future.get("low", future.get("mid", {}).get("l", entry_price)))

            if direction in ("BUY", "LONG"):
                if high - entry_price >= tp_pips:
                    return {"pnl_pips": tp_pips, "bars_held": bars}
                if entry_price - low >= sl_pips:
                    return {"pnl_pips": -sl_pips, "bars_held": bars}
            else:  # SELL / SHORT
                if entry_price - low >= tp_pips:
                    return {"pnl_pips": tp_pips, "bars_held": bars}
                if high - entry_price >= sl_pips:
                    return {"pnl_pips": -sl_pips, "bars_held": bars}

        # Timeout: close at current price
        last = candles[min(entry_idx + max_bars, len(candles) - 1)]
        last_close = float(last.get("close", last.get("mid", {}).get("c", entry_price)))

        if direction in ("BUY", "LONG"):
            pnl = last_close - entry_price
        else:
            pnl = entry_price - last_close

        return {"pnl_pips": pnl, "bars_held": max_bars}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist validator state."""
        try:
            data = {
                "version": 1,
                "total_replays": self._total_replays,
                "replay_log": self._replay_log[-50:],
                "drift_alerts": self._drift_alerts[-25:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"ReplayValidator: Failed to save: {e}")

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
                self._total_replays = data.get("total_replays", 0)
                self._replay_log = data.get("replay_log", [])
                self._drift_alerts = data.get("drift_alerts", [])
        except Exception as e:
            logger.debug(f"ReplayValidator: Failed to load: {e}")
