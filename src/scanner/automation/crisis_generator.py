"""Synthetic crisis event generator for RL training.

US-096: Generate synthetic rare market events (flash crashes, central bank
surprises, correlation breaks) using perturbation-based augmentation of
historical data. Feed synthetic scenarios into RL training loop to
improve drawdown resilience.

Based on 2024 HTGAN and SMOGAN research.

Event types:
1. Flash crash: sudden price drop (5-15%) followed by partial recovery
2. Volatility spike: ATR expansion (3-10x) sustained for N bars
3. Correlation break: normally correlated pairs diverge sharply

All generators take base OHLCV data and return perturbed versions that
preserve statistical properties while injecting the crisis pattern.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SCENARIO_PATH = _PROJECT_ROOT / "trained_data" / "synthetic_scenarios.json"


@dataclass
class SyntheticScenario:
    """A generated synthetic crisis scenario."""

    scenario_type: str = ""  # flash_crash, volatility_spike, correlation_break
    severity: float = 1.0  # 0-1 scale
    n_bars: int = 0
    perturbed_data: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scenario_type": self.scenario_type,
            "severity": round(self.severity, 3),
            "n_bars": self.n_bars,
            "metadata": self.metadata,
            "has_data": self.perturbed_data is not None,
        }


@dataclass
class ReplayResult:
    """Result of replaying a scenario through trading rules."""

    pnl_pips: float = 0.0
    max_drawdown_pips: float = 0.0
    n_trades: int = 0
    survived: bool = True  # True if drawdown stayed within limits

    def to_dict(self) -> dict:
        return {
            "pnl_pips": round(self.pnl_pips, 2),
            "max_drawdown_pips": round(self.max_drawdown_pips, 2),
            "n_trades": self.n_trades,
            "survived": self.survived,
        }


class CrisisEventGenerator:
    """Generate synthetic crisis events for RL robustness training.

    Creates perturbed market data that simulates rare events, allowing
    the RL system to train on crisis scenarios without waiting for them
    to occur naturally.

    Usage:
        gen = CrisisEventGenerator()
        scenario = gen.generate_flash_crash(base_candles, severity=0.7)
        result = gen.replay_scenario(scenario, trading_rules)
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None,
        seed: Optional[int] = None,
    ):
        self.persistence_path = Path(persistence_path) if persistence_path else _SCENARIO_PATH
        self._rng = np.random.default_rng(seed)
        self._generated_scenarios: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API: Event Generators
    # ------------------------------------------------------------------

    def generate_flash_crash(
        self,
        base_candles: np.ndarray,
        severity: float = 0.5,  # clamped to 0-1
        crash_bar_offset: Optional[int] = None,
        recovery_ratio: float = 0.6,
    ) -> SyntheticScenario:
        """Generate a flash crash event.

        Injects a sudden price drop at crash_bar_offset, followed by
        partial recovery. Preserves volume structure but amplifies
        volume around the crash.

        Args:
            base_candles: OHLCV array shape (n_bars, 5) — columns: O, H, L, C, V
            severity: 0-1 scale. 0.5 = ~7% drop, 1.0 = ~15% drop.
            crash_bar_offset: Which bar to inject crash at (random if None).
            recovery_ratio: How much of the drop recovers (0-1).

        Returns:
            SyntheticScenario with perturbed OHLCV data.
        """
        data = np.array(base_candles, dtype=np.float64, copy=True)
        n_bars = len(data)
        severity = max(0.0, min(1.0, severity))

        if n_bars < 10:
            return SyntheticScenario(scenario_type="flash_crash", n_bars=0)

        # Determine crash location
        if crash_bar_offset is None:
            # Place in middle third to allow both pre and post context
            crash_bar_offset = int(self._rng.integers(n_bars // 3, 2 * n_bars // 3))

        crash_bar = min(crash_bar_offset, n_bars - 5)

        # Drop magnitude: severity maps to 5-15% price drop
        drop_pct = 0.05 + severity * 0.10
        pre_crash_price = data[crash_bar, 3]  # Close

        # Apply crash
        crash_drop = pre_crash_price * drop_pct
        recovery_bars = min(5, n_bars - crash_bar - 1)

        for i in range(crash_bar, min(crash_bar + 2, n_bars)):
            # Immediate crash bars
            data[i, 0] -= crash_drop * (0.3 if i == crash_bar else 0.7)  # Open
            data[i, 1] -= crash_drop * 0.1  # High (still spikes down)
            data[i, 2] -= crash_drop * 1.1  # Low (wick extends further)
            data[i, 3] -= crash_drop  # Close
            data[i, 4] *= 3.0 + severity * 5.0  # Volume spike
            # Enforce OHLC consistency: High >= max(O,C), Low <= min(O,C)
            data[i, 1] = max(data[i, 1], data[i, 0], data[i, 3])
            data[i, 2] = min(data[i, 2], data[i, 0], data[i, 3])

        # Recovery phase
        recovery_amount = crash_drop * recovery_ratio
        for i in range(1, recovery_bars + 1):
            idx = crash_bar + 2 + i - 1
            if idx >= n_bars:
                break
            recovery_frac = i / recovery_bars
            recovery_at_bar = recovery_amount * recovery_frac
            data[idx, 0] += recovery_at_bar * 0.8
            data[idx, 1] += recovery_at_bar * 1.1
            data[idx, 2] += recovery_at_bar * 0.5
            data[idx, 3] += recovery_at_bar
            data[idx, 4] *= 1.5  # Elevated but declining volume
            # Enforce OHLC consistency
            data[idx, 1] = max(data[idx, 1], data[idx, 0], data[idx, 3])
            data[idx, 2] = min(data[idx, 2], data[idx, 0], data[idx, 3])

        # Clamp all prices to non-negative (financial data invariant)
        data[:, :4] = np.maximum(data[:, :4], 1e-6)

        scenario = SyntheticScenario(
            scenario_type="flash_crash",
            severity=severity,
            n_bars=n_bars,
            perturbed_data=data,
            metadata={
                "crash_bar": crash_bar,
                "drop_pct": round(drop_pct, 4),
                "recovery_ratio": recovery_ratio,
                "pre_crash_price": round(pre_crash_price, 5),
            },
        )

        self._log_scenario(scenario)
        return scenario

    def generate_volatility_spike(
        self,
        base_candles: np.ndarray,
        spike_magnitude: float = 3.0,
        spike_duration_bars: int = 10,
        spike_offset: Optional[int] = None,
    ) -> SyntheticScenario:
        """Generate a volatility spike event.

        Expands ATR (high-low range) by spike_magnitude for
        spike_duration_bars, with gradual ramp-up and ramp-down.

        Args:
            base_candles: OHLCV array shape (n_bars, 5).
            spike_magnitude: Multiplier for ATR (3.0 = 3x normal volatility).
            spike_duration_bars: How many bars the spike lasts.
            spike_offset: Starting bar for spike (random if None).

        Returns:
            SyntheticScenario with perturbed OHLCV data.
        """
        data = np.array(base_candles, dtype=np.float64, copy=True)
        n_bars = len(data)
        spike_magnitude = max(1.0, spike_magnitude)

        if n_bars < spike_duration_bars + 5:
            return SyntheticScenario(scenario_type="volatility_spike", n_bars=0)

        if spike_offset is None:
            spike_offset = int(self._rng.integers(5, n_bars - spike_duration_bars))

        # Compute baseline ATR
        ranges = data[:, 1] - data[:, 2]  # High - Low
        baseline_atr = float(np.mean(ranges[ranges > 0])) if np.any(ranges > 0) else 0.001

        for i in range(spike_duration_bars):
            idx = spike_offset + i
            if idx >= n_bars:
                break

            # Ramp: peak in middle, taper at edges
            progress = i / max(spike_duration_bars - 1, 1)
            ramp = np.sin(progress * np.pi)  # Peaks at 0.5
            mult = 1.0 + (spike_magnitude - 1.0) * ramp

            mid = (data[idx, 1] + data[idx, 2]) / 2.0
            half_range = baseline_atr * mult / 2.0

            # Expand range symmetrically around midpoint
            data[idx, 1] = mid + half_range  # High
            data[idx, 2] = mid - half_range  # Low

            # Add random directional noise
            noise = self._rng.normal(0, baseline_atr * 0.3)
            data[idx, 0] += noise  # Open
            data[idx, 3] += noise  # Close

            # Ensure OHLC consistency
            data[idx, 1] = max(data[idx, 1], data[idx, 0], data[idx, 3])
            data[idx, 2] = min(data[idx, 2], data[idx, 0], data[idx, 3])

            # Volume spike proportional to volatility
            data[idx, 4] *= mult * 0.8

        # Clamp all prices to non-negative (financial data invariant)
        data[:, :4] = np.maximum(data[:, :4], 1e-6)

        scenario = SyntheticScenario(
            scenario_type="volatility_spike",
            severity=min(1.0, spike_magnitude / 10.0),
            n_bars=n_bars,
            perturbed_data=data,
            metadata={
                "spike_magnitude": spike_magnitude,
                "spike_duration_bars": spike_duration_bars,
                "spike_offset": spike_offset,
                "baseline_atr": round(baseline_atr, 6),
            },
        )

        self._log_scenario(scenario)
        return scenario

    def generate_correlation_break(
        self,
        pair_data_dict: Dict[str, np.ndarray],
        target_pairs: Optional[List[str]] = None,
        break_magnitude: float = 0.5,
    ) -> SyntheticScenario:
        """Generate a correlation break event.

        Takes normally correlated pairs and injects divergence in the
        second half of the data, simulating a regime break.

        Args:
            pair_data_dict: Dict mapping pair_name → OHLCV array.
            target_pairs: Which pairs to decorrelate (first 2 if None).
            break_magnitude: How much to decorrelate (0=no change, 1=full inversion).

        Returns:
            SyntheticScenario with perturbed multi-pair data.
        """
        if len(pair_data_dict) < 2:
            return SyntheticScenario(scenario_type="correlation_break", n_bars=0)

        pairs = target_pairs or list(pair_data_dict.keys())[:2]
        if len(pairs) < 2 or pairs[0] not in pair_data_dict or pairs[1] not in pair_data_dict:
            return SyntheticScenario(scenario_type="correlation_break", n_bars=0)

        # Work on the second pair — first pair stays as anchor
        data_1 = np.array(pair_data_dict[pairs[0]], dtype=np.float64, copy=True)
        data_2 = np.array(pair_data_dict[pairs[1]], dtype=np.float64, copy=True)

        n_bars = min(len(data_1), len(data_2))
        if n_bars < 20:
            return SyntheticScenario(scenario_type="correlation_break", n_bars=0)

        # Break point: middle of data
        break_point = n_bars // 2

        # Compute returns of pair 1 (anchor) after break
        pair1_returns = np.diff(data_1[break_point:, 3])  # Close returns

        # Inject inverted returns into pair 2 (scaled by break_magnitude)
        for i in range(len(pair1_returns)):
            idx = break_point + 1 + i
            if idx >= n_bars:
                break

            # Inversion: add negative of pair1's return to pair2
            inversion = -pair1_returns[i] * break_magnitude
            data_2[idx, 0] += inversion  # Open
            data_2[idx, 3] += inversion  # Close
            data_2[idx, 1] = max(data_2[idx, 1], data_2[idx, 0], data_2[idx, 3])
            data_2[idx, 2] = min(data_2[idx, 2], data_2[idx, 0], data_2[idx, 3])

        # Build combined output
        perturbed_dict = dict(pair_data_dict)
        perturbed_dict[pairs[1]] = data_2

        scenario = SyntheticScenario(
            scenario_type="correlation_break",
            severity=break_magnitude,
            n_bars=n_bars,
            perturbed_data=None,  # Multi-pair stored in metadata
            metadata={
                "target_pairs": pairs,
                "break_point": break_point,
                "break_magnitude": break_magnitude,
                "perturbed_pair_data": {
                    p: d.tolist() for p, d in perturbed_dict.items()
                } if n_bars < 100 else {"note": "data too large to serialize"},
            },
        )

        self._log_scenario(scenario)
        return scenario

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay_scenario(
        self,
        scenario: SyntheticScenario,
        trading_rules: Optional[Dict[str, Any]] = None,
    ) -> ReplayResult:
        """Replay a synthetic scenario through simplified trading rules.

        Simulates basic trend-following on the perturbed data to
        estimate P/L and drawdown behavior.

        Args:
            scenario: The synthetic scenario to replay.
            trading_rules: Optional rule parameters (sl_pips, tp_pips, etc.)

        Returns:
            ReplayResult with simulated P/L and drawdown.
        """
        result = ReplayResult()

        if scenario.perturbed_data is None or len(scenario.perturbed_data) < 10:
            return result

        data = scenario.perturbed_data
        rules = trading_rules or {}
        sl_pips = rules.get("sl_pips", 20.0)
        tp_pips = rules.get("tp_pips", 30.0)
        max_dd_pips = rules.get("max_drawdown_pips", 50.0)

        total_pnl = 0.0
        max_drawdown = 0.0
        peak_pnl = 0.0
        n_trades = 0

        # Simple simulation: enter on close[i], exit at SL/TP check on close[i+1:]
        i = 0
        while i < len(data) - 5:
            close = data[i, 3]
            # Simple momentum signal: positive = BUY
            if i > 2:
                momentum = data[i, 3] - data[i - 3, 3]
            else:
                momentum = 0.0

            if abs(momentum) < 0.00001:
                i += 1
                continue

            direction = 1 if momentum > 0 else -1
            entry = close
            n_trades += 1

            # Scan forward for exit
            for j in range(i + 1, min(i + 20, len(data))):
                pnl = (data[j, 3] - entry) * direction * 10000  # Pips

                if pnl >= tp_pips:
                    total_pnl += tp_pips
                    break
                elif pnl <= -sl_pips:
                    total_pnl -= sl_pips
                    break
            else:
                # Time exit
                final_pnl = (data[min(i + 19, len(data) - 1), 3] - entry) * direction * 10000
                total_pnl += final_pnl

            # Drawdown tracking
            peak_pnl = max(peak_pnl, total_pnl)
            dd = peak_pnl - total_pnl
            max_drawdown = max(max_drawdown, dd)

            if max_drawdown > max_dd_pips:
                result.survived = False

            i += 5  # Skip forward to avoid overlapping trades

        result.pnl_pips = total_pnl
        result.max_drawdown_pips = max_drawdown
        result.n_trades = n_trades

        return result

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_scenario_batch(
        self,
        base_candles: np.ndarray,
        n_scenarios: int = 10,
    ) -> List[SyntheticScenario]:
        """Generate a batch of mixed crisis scenarios.

        Args:
            base_candles: Base OHLCV data.
            n_scenarios: Number of scenarios to generate.

        Returns:
            List of SyntheticScenario objects.
        """
        scenarios: List[SyntheticScenario] = []
        scenario_types = ["flash_crash", "volatility_spike"]

        for i in range(n_scenarios):
            stype = scenario_types[i % len(scenario_types)]
            severity = 0.3 + self._rng.random() * 0.7  # 0.3 to 1.0

            if stype == "flash_crash":
                s = self.generate_flash_crash(base_candles, severity=severity)
            else:
                mag = 2.0 + severity * 8.0  # 2-10x
                s = self.generate_volatility_spike(base_candles, spike_magnitude=mag)

            if s.n_bars > 0:
                scenarios.append(s)

        logger.info(f"CrisisGenerator: Generated {len(scenarios)} scenarios")
        # Auto-persist after batch generation
        self.save_log()
        return scenarios

    def get_status(self) -> Dict[str, Any]:
        """Get generator status for observability."""
        return {
            "total_generated": len(self._generated_scenarios),
            "by_type": self._count_by_type(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _count_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self._generated_scenarios:
            stype = s.get("scenario_type", "unknown")
            counts[stype] = counts.get(stype, 0) + 1
        return counts

    def _log_scenario(self, scenario: SyntheticScenario) -> None:
        self._generated_scenarios.append({
            "scenario_type": scenario.scenario_type,
            "severity": round(scenario.severity, 3),
            "n_bars": scenario.n_bars,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # Keep bounded
        if len(self._generated_scenarios) > 200:
            self._generated_scenarios = self._generated_scenarios[-100:]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_log(self) -> None:
        """Persist scenario generation log."""
        try:
            data = {
                "version": 1,
                "scenarios": self._generated_scenarios[-100:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"CrisisGenerator: Failed to save log: {e}")
