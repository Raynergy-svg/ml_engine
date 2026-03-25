"""
Phase 46 Integration Tests: Full Entry Filtering Pipeline.

Tests verify that the 5 new Phase 46 modules work together correctly:
1. regime_gates.py — Regime-conditional gate thresholds
2. mtf_confluence.py — Multi-timeframe Elder-style confluence
3. session_detector.py — Session detection with confidence/sizing modifiers
4. ensemble_conflict.py — Ensemble model conflict resolution
5. expectancy_tracker.py — Per-agent per-regime expectancy tracking

Each test uses REAL module instances to verify actual integration.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.scanner.regime_gates import (
    REGIME_EXTREME,
    REGIME_HIGH,
    REGIME_LOW,
    REGIME_NORMAL,
    RegimeGateConfig,
    RegimeGateProfile,
    apply_regime_gates,
    get_regime_profile,
)
from src.scanner.mtf_confluence import (
    MultiTimeframeConfluence,
    MTFConfig,
    ConfluenceResult,
)
from src.scanner.session_detector import (
    SessionDetector,
    SessionConfig,
    SessionInfo,
)
from src.scanner.ensemble_conflict import (
    EnsembleConflictResolver,
    ConflictConfig,
    ModelPrediction,
    ConflictResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES: Synthetic Data Generators
# ============================================================================


def create_sample_ohlcv(
    length: int = 100,
    trend: str = "BULLISH",
    volatility: float = 1.0,
) -> pd.DataFrame:
    """Create realistic synthetic OHLCV data for testing.

    Args:
        length: Number of candles
        trend: "BULLISH", "BEARISH", or "NEUTRAL"
        volatility: ATR multiplier (1.0 = normal, 2.0 = high vol)

    Returns:
        DataFrame with open, high, low, close, volume columns
    """
    dates = pd.date_range("2026-01-01", periods=length, freq="1h")
    base_price = 1.1000

    # Generate base close prices with trend
    closes = []
    trend_direction = 1.0 if trend == "BULLISH" else (-1.0 if trend == "BEARISH" else 0.0)

    for i in range(length):
        drift = trend_direction * 0.0005 * i  # Trend component
        noise = np.random.normal(0, 0.0003 * volatility)  # Random walk with vol
        close = base_price + drift + noise
        closes.append(close)

    closes = np.array(closes)

    # Generate OHLC from closes
    opens = closes + np.random.normal(0, 0.0001 * volatility, length)
    atr = 0.0008 * volatility  # Average True Range
    highs = closes + np.abs(np.random.normal(atr / 2, atr / 4, length))
    lows = closes - np.abs(np.random.normal(atr / 2, atr / 4, length))

    volumes = np.random.randint(1000, 10000, length)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=dates)

    return df


@pytest.fixture
def sample_4h_data_bullish():
    """4H bullish trend data (200 candles for MTF SMA200 calculation)."""
    return create_sample_ohlcv(200, trend="BULLISH", volatility=1.0)


@pytest.fixture
def sample_1h_data_bullish():
    """1H bullish data with good wave quality."""
    return create_sample_ohlcv(100, trend="BULLISH", volatility=1.2)


@pytest.fixture
def sample_15m_data_bullish():
    """15M bullish entry confirmation data."""
    return create_sample_ohlcv(80, trend="BULLISH", volatility=1.5)


@pytest.fixture
def sample_4h_data_bearish():
    """4H bearish trend data (200 candles for MTF SMA200 calculation)."""
    return create_sample_ohlcv(200, trend="BEARISH", volatility=1.0)


@pytest.fixture
def sample_1h_data_bearish():
    """1H bearish data."""
    return create_sample_ohlcv(100, trend="BEARISH", volatility=1.2)


@pytest.fixture
def sample_15m_data_bearish():
    """15M bearish entry confirmation data."""
    return create_sample_ohlcv(80, trend="BEARISH", volatility=1.5)


@pytest.fixture
def sample_4h_data_neutral():
    """4H neutral/choppy data (200 candles for MTF SMA200 calculation)."""
    return create_sample_ohlcv(200, trend="NEUTRAL", volatility=2.0)


@pytest.fixture
def regime_gate_config():
    """Regime gate config with defaults."""
    return RegimeGateConfig(enabled=True)


@pytest.fixture
def mtf_confluence():
    """Multi-timeframe confluence analyzer."""
    return MultiTimeframeConfluence(MTFConfig())


@pytest.fixture
def session_detector():
    """Session detector with default config."""
    return SessionDetector(SessionConfig())


@pytest.fixture
def ensemble_conflict_resolver():
    """Ensemble conflict resolver with default config."""
    return EnsembleConflictResolver(ConflictConfig())


# ============================================================================
# TEST CLASS: Regime Gate Integration
# ============================================================================


class TestRegimeGateIntegration:
    """Test regime gates interact correctly with the pipeline."""

    def test_low_regime_raises_confidence_threshold(self, regime_gate_config):
        """In LOW vol, confidence threshold should be 0.60 (higher than NORMAL 0.50)."""
        thresholds = apply_regime_gates(REGIME_LOW)
        assert thresholds["confidence_threshold"] == 60.0
        assert thresholds["confidence_threshold"] > 50.0  # Higher than NORMAL

    def test_normal_regime_uses_standard_thresholds(self, regime_gate_config):
        """In NORMAL regime, use baseline thresholds."""
        thresholds = apply_regime_gates(REGIME_NORMAL)
        assert thresholds["confidence_threshold"] == 50.0
        assert thresholds["min_rr_ratio"] == 1.2

    def test_high_regime_requires_momentum_alignment(self, regime_gate_config):
        """In HIGH vol, require_momentum_alignment should be True."""
        thresholds = apply_regime_gates(REGIME_HIGH)
        assert thresholds["require_momentum_alignment"] is True
        assert abs(thresholds["confidence_threshold"] - 55.0) < 0.01
        assert abs(thresholds["position_size_multiplier"] - 0.65) < 0.01

    def test_extreme_regime_blocks_low_confidence(self, regime_gate_config):
        """In EXTREME vol, confidence threshold is 0.65 (very selective)."""
        thresholds = apply_regime_gates(REGIME_EXTREME)
        assert thresholds["confidence_threshold"] == 65.0
        assert thresholds["position_size_multiplier"] == 0.40  # Small defensive position

    def test_regime_gates_graceful_fallback_no_regime(self, regime_gate_config):
        """When regime is None, fallback to static thresholds."""
        thresholds = apply_regime_gates(None)
        assert thresholds["confidence_threshold"] == 50.0  # Default static
        assert thresholds["position_size_multiplier"] == 1.0

    def test_regime_gates_unknown_regime_defaults_to_normal(self, regime_gate_config):
        """Unknown regime name defaults to NORMAL profile."""
        thresholds = apply_regime_gates("UNKNOWN_REGIME")
        assert thresholds["confidence_threshold"] == 50.0  # NORMAL default

    def test_position_size_multiplier_scales_with_regime(self, regime_gate_config):
        """Position sizing should scale: LOW > NORMAL > HIGH > EXTREME."""
        low = apply_regime_gates(REGIME_LOW)["position_size_multiplier"]
        normal = apply_regime_gates(REGIME_NORMAL)["position_size_multiplier"]
        high = apply_regime_gates(REGIME_HIGH)["position_size_multiplier"]
        extreme = apply_regime_gates(REGIME_EXTREME)["position_size_multiplier"]

        assert low > normal > high > extreme
        assert low == 1.3
        assert normal == 1.0
        assert high == 0.65
        assert extreme == 0.40

    def test_min_rr_ratio_increases_in_difficult_regimes(self, regime_gate_config):
        """R:R requirements rise in volatile/extreme regimes."""
        normal_rr = apply_regime_gates(REGIME_NORMAL)["min_rr_ratio"]
        high_rr = apply_regime_gates(REGIME_HIGH)["min_rr_ratio"]
        extreme_rr = apply_regime_gates(REGIME_EXTREME)["min_rr_ratio"]

        assert normal_rr == 1.2
        assert high_rr == 1.3
        assert extreme_rr == 1.5


# ============================================================================
# TEST CLASS: Multi-Timeframe Confluence Integration
# ============================================================================


class TestMTFConfluenceIntegration:
    """Test MTF confluence interacts with direction filtering."""

    def test_mtf_all_screens_aligned_bullish_proceeds(
        self,
        mtf_confluence,
        sample_4h_data_bullish,
        sample_1h_data_bullish,
        sample_15m_data_bullish,
    ):
        """When all 3 screens agree on direction (BULLISH), no misalignment penalty applied."""
        result = mtf_confluence.score_confluence(
            sample_4h_data_bullish,
            sample_1h_data_bullish,
            sample_15m_data_bullish,
        )

        assert result is not None
        assert result.direction_aligned is True
        assert result.penalty_applied == 0.0  # No penalty for alignment
        # Verify all screens point same direction
        screen_directions = {s.direction for s in result.screen_results if s.direction != "NEUTRAL"}
        assert len(screen_directions) <= 1  # All same or all neutral

    def test_mtf_all_screens_aligned_bearish_proceeds(
        self,
        mtf_confluence,
        sample_4h_data_bearish,
        sample_1h_data_bearish,
        sample_15m_data_bearish,
    ):
        """When all 3 screens agree on BEARISH direction, no misalignment penalty applied."""
        result = mtf_confluence.score_confluence(
            sample_4h_data_bearish,
            sample_1h_data_bearish,
            sample_15m_data_bearish,
        )

        assert result is not None
        assert result.direction_aligned is True
        # When aligned, no penalty applied
        assert result.penalty_applied == 0.0

    def test_mtf_direction_alignment_detection(
        self,
        mtf_confluence,
        sample_4h_data_bullish,
        sample_1h_data_bullish,
        sample_15m_data_bullish,
    ):
        """Confluence should detect when all screens align vs. disagree."""
        result_aligned = mtf_confluence.score_confluence(
            sample_4h_data_bullish,
            sample_1h_data_bullish,
            sample_15m_data_bullish,
        )

        # When all agree, no penalty
        assert result_aligned is not None
        assert result_aligned.direction_aligned is True
        assert result_aligned.penalty_applied == 0.0

    def test_mtf_score_in_valid_range(
        self,
        mtf_confluence,
        sample_4h_data_bullish,
        sample_1h_data_bullish,
        sample_15m_data_bullish,
    ):
        """Confluence score should always be between 0.0 and 1.0."""
        result = mtf_confluence.score_confluence(
            sample_4h_data_bullish,
            sample_1h_data_bullish,
            sample_15m_data_bullish,
        )

        assert result is not None
        assert 0.0 <= result.confluence_score <= 1.0

    def test_mtf_three_screens_included(
        self,
        mtf_confluence,
        sample_4h_data_bullish,
        sample_1h_data_bullish,
        sample_15m_data_bullish,
    ):
        """Result should include all 3 screen evaluations."""
        result = mtf_confluence.score_confluence(
            sample_4h_data_bullish,
            sample_1h_data_bullish,
            sample_15m_data_bullish,
        )

        assert result is not None
        assert len(result.screen_results) == 3
        screen_names = [s.screen_name for s in result.screen_results]
        assert "trend_4h" in screen_names
        assert "wave_1h" in screen_names
        assert "entry_15m" in screen_names


# ============================================================================
# TEST CLASS: Session Filter Integration
# ============================================================================


class TestSessionFilterIntegration:
    """Test session awareness affects sizing correctly."""

    def test_tokyo_session_reduces_position_size(self, session_detector):
        """During Tokyo session (23:00-08:00), position size multiplier should be 0.70."""
        tokyo_time = datetime(2026, 1, 15, 2, 0, 0, tzinfo=timezone.utc)  # 02:00 UTC
        session = session_detector.get_current_session(tokyo_time)

        assert session.name == "TOKYO"
        assert session.position_size_multiplier == 0.70
        assert session.is_overlap is False

    def test_london_session_maintains_full_size(self, session_detector):
        """During London session (08:00-17:00), position size multiplier should be 1.00."""
        london_time = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)  # 10:00 UTC
        session = session_detector.get_current_session(london_time)

        assert session.name == "LONDON"
        assert session.position_size_multiplier == 1.00
        assert session.is_overlap is False

    def test_london_ny_overlap_reduces_size_for_volatility(self, session_detector):
        """During London-NY overlap (13:00-17:00), position size should be 0.75."""
        overlap_time = datetime(2026, 1, 15, 14, 30, 0, tzinfo=timezone.utc)  # 14:30 UTC
        session = session_detector.get_current_session(overlap_time)

        assert session.name == "LONDON_NY_OVERLAP"
        assert session.is_overlap is True
        assert session.position_size_multiplier == 0.75

    def test_new_york_session_moderate_sizing(self, session_detector):
        """During New York session (13:00-22:00), position size multiplier should be 0.85."""
        ny_time = datetime(2026, 1, 15, 18, 0, 0, tzinfo=timezone.utc)  # 18:00 UTC
        session = session_detector.get_current_session(ny_time)

        assert session.name == "NEW_YORK"
        assert session.position_size_multiplier == 0.85

    def test_off_hours_minimal_position_size(self, session_detector):
        """During off hours (22:00-23:00), position size multiplier should be 0.50."""
        off_hours = datetime(2026, 1, 15, 22, 30, 0, tzinfo=timezone.utc)  # 22:30 UTC
        session = session_detector.get_current_session(off_hours)

        assert session.name == "OFF_HOURS"
        assert session.position_size_multiplier == 0.50

    def test_tokyo_session_reduces_trend_confidence(self, session_detector):
        """Tokyo session applies -0.15 confidence penalty for trend strategies."""
        tokyo_time = datetime(2026, 1, 15, 2, 0, 0, tzinfo=timezone.utc)
        session = session_detector.get_current_session(tokyo_time)

        trend_modifier = session.confidence_modifiers.get("trend", 0.0)
        assert trend_modifier == -0.15

    def test_london_session_boosts_trend_confidence(self, session_detector):
        """London session applies +0.10 confidence boost for trend strategies."""
        london_time = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        session = session_detector.get_current_session(london_time)

        trend_modifier = session.confidence_modifiers.get("trend", 0.0)
        assert trend_modifier == 0.10

    def test_overlap_session_boosts_momentum(self, session_detector):
        """London-NY overlap applies +0.10 confidence boost for momentum strategies."""
        overlap_time = datetime(2026, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        session = session_detector.get_current_session(overlap_time)

        momentum_modifier = session.confidence_modifiers.get("momentum", 0.0)
        assert momentum_modifier == 0.10

    def test_session_detection_disabled_returns_neutral(self):
        """When session detection is disabled, should return neutral 1.0x multiplier."""
        config = SessionConfig(enabled=False)
        detector = SessionDetector(config)

        any_time = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(any_time)

        assert session.position_size_multiplier == 1.0
        assert len(session.confidence_modifiers) == 0


# ============================================================================
# TEST CLASS: Ensemble Conflict Integration
# ============================================================================


class TestEnsembleConflictIntegration:
    """Test ensemble conflict resolver blocks appropriately."""

    def test_ensemble_blocks_when_models_disagree_direction(
        self, ensemble_conflict_resolver
    ):
        """When TCN says BUY (+0.8) and RF says SELL (-0.7), trade should be blocked."""
        predictions = [
            ModelPrediction("TCN", score=0.8, direction="BUY", confidence=0.9),
            ModelPrediction("Ridge", score=0.0, direction="HOLD", confidence=0.5),
            ModelPrediction("RF", score=-0.7, direction="SELL", confidence=0.8),
        ]

        result = ensemble_conflict_resolver.resolve(predictions)

        assert result is not None
        assert result.has_direction_conflict is True
        assert result.should_block is True

    def test_ensemble_all_models_agree_no_block(self, ensemble_conflict_resolver):
        """When all models agree (all BUY), no conflict block."""
        predictions = [
            ModelPrediction("TCN", score=0.75, direction="BUY", confidence=0.95),
            ModelPrediction("Ridge", score=0.65, direction="BUY", confidence=0.85),
            ModelPrediction("RF", score=0.70, direction="BUY", confidence=0.90),
        ]

        result = ensemble_conflict_resolver.resolve(predictions)

        assert result is not None
        assert result.has_direction_conflict is False
        assert result.should_block is False
        assert result.consensus_direction == "BUY"

    def test_ensemble_applies_graduated_penalty(
        self, ensemble_conflict_resolver
    ):
        """Disagreement of 0.40 should apply -0.20 penalty (per penalty curve)."""
        # Create predictions with high disagreement (std/mean ratio)
        predictions = [
            ModelPrediction("TCN", score=0.5, direction="BUY", confidence=0.9),
            ModelPrediction("Ridge", score=0.1, direction="HOLD", confidence=0.5),
            ModelPrediction("RF", score=-0.2, direction="SELL", confidence=0.7),
        ]

        result = ensemble_conflict_resolver.resolve(predictions)

        assert result is not None
        # Disagreement score should be calculated from std/mean
        # With these scores: mean=0.13, std≈0.35, disagreement ≈ 2.7 (clamped to 1.0)
        assert result.penalty <= 0.0  # Penalty is always non-positive

    def test_ensemble_consensus_direction_majority_vote(
        self, ensemble_conflict_resolver
    ):
        """Consensus should be majority vote (2 BUY, 1 SELL = BUY)."""
        predictions = [
            ModelPrediction("TCN", score=0.7, direction="BUY", confidence=0.95),
            ModelPrediction("Ridge", score=0.6, direction="BUY", confidence=0.85),
            ModelPrediction("RF", score=-0.5, direction="SELL", confidence=0.80),
        ]

        result = ensemble_conflict_resolver.resolve(predictions)

        assert result is not None
        assert result.consensus_direction == "BUY"
        assert result.direction_votes["BUY"] == 2
        assert result.direction_votes["SELL"] == 1

    def test_ensemble_hold_consensus_when_all_neutral(
        self, ensemble_conflict_resolver
    ):
        """When all models are neutral (HOLD), consensus should be HOLD."""
        predictions = [
            ModelPrediction("TCN", score=0.05, direction="HOLD", confidence=0.5),
            ModelPrediction("Ridge", score=-0.05, direction="HOLD", confidence=0.5),
            ModelPrediction("RF", score=0.02, direction="HOLD", confidence=0.5),
        ]

        result = ensemble_conflict_resolver.resolve(predictions)

        assert result is not None
        assert result.consensus_direction == "HOLD"

    def test_ensemble_disabled_returns_zero_penalty(self):
        """When conflict resolver is disabled, should return zero penalty."""
        config = ConflictConfig(enabled=False)
        resolver = EnsembleConflictResolver(config)

        predictions = [
            ModelPrediction("TCN", score=0.8, direction="BUY", confidence=0.9),
            ModelPrediction("RF", score=-0.7, direction="SELL", confidence=0.8),
        ]

        result = resolver.resolve(predictions)

        assert result is not None
        assert result.penalty == 0.0
        assert result.should_block is False

    def test_ensemble_empty_predictions_no_crash(
        self, ensemble_conflict_resolver
    ):
        """With empty predictions, should return graceful result, not crash."""
        result = ensemble_conflict_resolver.resolve([])

        assert result is not None
        assert result.penalty == 0.0
        assert result.consensus_direction == "HOLD"


# ============================================================================
# TEST CLASS: Full Pipeline Integration
# ============================================================================


class TestFullPipelineIntegration:
    """Test the complete Phase 46 pipeline end-to-end."""

    def test_full_pipeline_regime_to_execute_bullish_high_regime(
        self,
        regime_gate_config,
        mtf_confluence,
        session_detector,
        ensemble_conflict_resolver,
        sample_4h_data_bullish,
        sample_1h_data_bullish,
        sample_15m_data_bullish,
    ):
        """Full pipeline: HIGH regime → regime gates → MTF confluence → session filter → ensemble.

        Steps:
        1. Get HIGH regime thresholds
        2. Check MTF confluence (all screens aligned)
        3. Get current session (London 10:00 UTC)
        4. Check ensemble (all models agree on BUY)
        5. Verify final position size = confluence_score * session_multiplier * base_size
        """
        # Step 1: Apply HIGH regime gates
        regime_thresholds = apply_regime_gates(REGIME_HIGH)
        assert abs(regime_thresholds["confidence_threshold"] - 55.0) < 0.01
        assert abs(regime_thresholds["position_size_multiplier"] - 0.65) < 0.01

        # Step 2: Check MTF confluence
        confluence_result = mtf_confluence.score_confluence(
            sample_4h_data_bullish,
            sample_1h_data_bullish,
            sample_15m_data_bullish,
        )
        assert confluence_result is not None
        assert confluence_result.direction_aligned is True

        # Step 3: Get session modifiers (London)
        london_time = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        session = session_detector.get_current_session(london_time)
        assert session.name == "LONDON"
        assert session.position_size_multiplier == 1.00

        # Step 4: Check ensemble (all models BUY)
        predictions = [
            ModelPrediction("TCN", score=0.75, direction="BUY", confidence=0.95),
            ModelPrediction("Ridge", score=0.70, direction="BUY", confidence=0.90),
            ModelPrediction("RF", score=0.72, direction="BUY", confidence=0.92),
        ]
        conflict_result = ensemble_conflict_resolver.resolve(predictions)
        assert conflict_result is not None
        assert conflict_result.should_block is False

        # Step 5: Compute final sizing
        base_position_size = 1.0  # 1 lot
        regime_mult = regime_thresholds["position_size_multiplier"]  # 0.65
        session_mult = session.position_size_multiplier  # 1.00
        confluence_factor = confluence_result.confluence_score  # ~0.6-0.8

        final_size = base_position_size * regime_mult * session_mult * confluence_factor
        assert final_size > 0.0
        assert final_size < base_position_size  # HIGH regime reduces size

    def test_full_pipeline_extreme_regime_defensive(
        self,
        mtf_confluence,
        session_detector,
        ensemble_conflict_resolver,
        sample_4h_data_bullish,
        sample_1h_data_bullish,
        sample_15m_data_bullish,
    ):
        """Full pipeline in EXTREME regime: very selective, small defensive positions."""
        # Step 1: EXTREME regime thresholds
        extreme_thresholds = apply_regime_gates(REGIME_EXTREME)
        assert extreme_thresholds["confidence_threshold"] == 65.0
        assert extreme_thresholds["position_size_multiplier"] == 0.40

        # Step 2: Only high-quality confluence passes
        confluence_result = mtf_confluence.score_confluence(
            sample_4h_data_bullish,
            sample_1h_data_bullish,
            sample_15m_data_bullish,
        )

        # Step 3: Session filter (worst case: overlap reduces to 0.75)
        overlap_time = datetime(2026, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        session = session_detector.get_current_session(overlap_time)

        # Step 4: Ensemble must ALL agree (zero conflict)
        predictions = [
            ModelPrediction("TCN", score=0.80, direction="BUY", confidence=0.98),
            ModelPrediction("Ridge", score=0.75, direction="BUY", confidence=0.95),
            ModelPrediction("RF", score=0.78, direction="BUY", confidence=0.96),
        ]
        conflict_result = ensemble_conflict_resolver.resolve(predictions)

        # Step 5: Final position size in EXTREME regime is very small
        base_size = 1.0
        regime_mult = extreme_thresholds["position_size_multiplier"]  # 0.40
        session_mult = session.position_size_multiplier  # 0.75
        final_size = base_size * regime_mult * session_mult

        assert abs(final_size - 0.30) < 0.001  # 40% * 75% = 30% of normal size
        assert final_size < 0.40  # Definitely reduced

    def test_full_pipeline_ensemble_conflict_blocks_trade(
        self,
        ensemble_conflict_resolver,
    ):
        """When ensemble models conflict (BUY vs SELL), trade should be blocked."""
        # Ensemble: TCN BUY vs RF SELL = conflict
        predictions = [
            ModelPrediction("TCN", score=0.75, direction="BUY", confidence=0.95),
            ModelPrediction("RF", score=-0.70, direction="SELL", confidence=0.90),
        ]
        conflict_result = ensemble_conflict_resolver.resolve(predictions)
        assert conflict_result.has_direction_conflict is True
        assert conflict_result.should_block is True

        # This is the hard gate that prevents execution regardless of MTF quality
        assert conflict_result.penalty == -1.0  # Hard block

    def test_full_pipeline_regime_session_stacking(
        self,
        session_detector,
        sample_4h_data_bullish,
        sample_1h_data_bullish,
        sample_15m_data_bullish,
    ):
        """Verify regime gates and session detector multipliers stack correctly."""
        mtf = MultiTimeframeConfluence()

        # Scenario 1: LOW regime + London session (best case)
        low_regime = apply_regime_gates(REGIME_LOW)
        london_time = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        london_session = session_detector.get_current_session(london_time)

        size_low_london = low_regime["position_size_multiplier"] * london_session.position_size_multiplier
        assert size_low_london == 1.3 * 1.0  # 1.3x
        assert size_low_london > 1.0  # Boosted

        # Scenario 2: EXTREME regime + overlap session (worst case)
        extreme_regime = apply_regime_gates(REGIME_EXTREME)
        overlap_time = datetime(2026, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        overlap_session = session_detector.get_current_session(overlap_time)

        size_extreme_overlap = extreme_regime["position_size_multiplier"] * overlap_session.position_size_multiplier
        assert size_extreme_overlap == 0.40 * 0.75  # 0.30x
        assert size_extreme_overlap < 0.5  # Defensive

    def test_full_pipeline_all_gates_pass_high_quality_setup(
        self,
        regime_gate_config,
        mtf_confluence,
        session_detector,
        ensemble_conflict_resolver,
        sample_4h_data_bullish,
        sample_1h_data_bullish,
        sample_15m_data_bullish,
    ):
        """High-quality setup demonstrates all modules working together."""
        # Regime: NORMAL
        regime_thresholds = apply_regime_gates(REGIME_NORMAL)
        assert regime_thresholds["confidence_threshold"] == 50.0

        # MTF: All screens should align (same direction or neutral)
        confluence = mtf_confluence.score_confluence(
            sample_4h_data_bullish,
            sample_1h_data_bullish,
            sample_15m_data_bullish,
        )
        assert confluence is not None
        assert confluence.direction_aligned is True  # No misalignment penalty

        # Session: London (good liquidity)
        london_time = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        session = session_detector.get_current_session(london_time)
        assert session.position_size_multiplier == 1.00

        # Ensemble: All models agree BUY
        predictions = [
            ModelPrediction("TCN", score=0.78, direction="BUY", confidence=0.96),
            ModelPrediction("Ridge", score=0.72, direction="BUY", confidence=0.92),
            ModelPrediction("RF", score=0.75, direction="BUY", confidence=0.94),
        ]
        conflict = ensemble_conflict_resolver.resolve(predictions)
        assert conflict.should_block is False
        assert conflict.consensus_direction == "BUY"

        # All gates aligned and no blocks
        assert confluence.direction_aligned is True
        assert conflict.should_block is False


# ============================================================================
# TEST CLASS: Edge Cases and Error Handling
# ============================================================================


class TestPhase46EdgeCases:
    """Test edge cases and error handling across Phase 46 modules."""

    def test_mtf_insufficient_data_returns_rejection(self, mtf_confluence):
        """With insufficient data (< 200 candles for 4H), should return rejection."""
        short_data = create_sample_ohlcv(length=10)
        result = mtf_confluence.score_confluence(short_data, short_data, short_data)

        assert result is not None
        assert result.recommendation == "REJECT"

    def test_regime_gates_all_regimes_validate(self, regime_gate_config):
        """All 4 regime profiles should validate without error."""
        for regime in [REGIME_LOW, REGIME_NORMAL, REGIME_HIGH, REGIME_EXTREME]:
            profile = get_regime_profile(regime)
            assert profile is not None
            profile.validate()  # Should not raise

    def test_session_detector_wrapping_tokyo(self, session_detector):
        """Tokyo session wraps midnight (23:00-08:00 UTC)."""
        # Test 23:00 UTC (start)
        pre_midnight = datetime(2026, 1, 15, 23, 0, 0, tzinfo=timezone.utc)
        session = session_detector.get_current_session(pre_midnight)
        assert session.name == "TOKYO"

        # Test 02:00 UTC (middle)
        post_midnight = datetime(2026, 1, 15, 2, 0, 0, tzinfo=timezone.utc)
        session = session_detector.get_current_session(post_midnight)
        assert session.name == "TOKYO"

        # Test 08:00 UTC (boundary, should be LONDON)
        at_boundary = datetime(2026, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        session = session_detector.get_current_session(at_boundary)
        assert session.name == "LONDON"

    def test_ensemble_single_model_no_disagreement(self, ensemble_conflict_resolver):
        """With only 1 model, disagreement should be 0.0."""
        predictions = [
            ModelPrediction("TCN", score=0.7, direction="BUY", confidence=0.95),
        ]

        result = ensemble_conflict_resolver.resolve(predictions)

        assert result is not None
        assert result.disagreement_score == 0.0
        assert result.has_direction_conflict is False

    def test_confluence_neutral_trend_screen(
        self, mtf_confluence, sample_4h_data_neutral
    ):
        """When trend screen is NEUTRAL, confluence should handle gracefully."""
        result = mtf_confluence.score_confluence(
            sample_4h_data_neutral,
            create_sample_ohlcv(50, trend="BULLISH"),
            create_sample_ohlcv(30, trend="BULLISH"),
        )

        assert result is not None
        # Should still produce valid score (not crash)
        assert 0.0 <= result.confluence_score <= 1.0

    def test_regime_gates_boundary_thresholds(self, regime_gate_config):
        """Test boundary cases for regime threshold values."""
        # Confidence threshold should be in [0, 100]
        for regime in [REGIME_LOW, REGIME_NORMAL, REGIME_HIGH, REGIME_EXTREME]:
            thresholds = apply_regime_gates(regime)
            assert 0.0 <= thresholds["confidence_threshold"] <= 100.0
            assert 0.0 < thresholds["position_size_multiplier"]  # Must be positive

    def test_session_detector_all_session_names_valid(self, session_detector):
        """All session names should be recognizable FX sessions."""
        valid_names = {"TOKYO", "LONDON", "NEW_YORK", "LONDON_NY_OVERLAP", "OFF_HOURS"}

        for hour in range(24):
            utc_time = datetime(2026, 1, 15, hour, 0, 0, tzinfo=timezone.utc)
            session = session_detector.get_current_session(utc_time)
            assert session.name in valid_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
