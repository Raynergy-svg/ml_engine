"""Unit tests for position sizing module."""

import pytest
from position_sizing import (
    PositionSizingConfig,
    PositionSize,
    DynamicPositionSizer,
    FixedPositionSizer,
    create_default_position_sizer,
    create_conservative_position_sizer,
)


class TestPositionSizingConfig:
    """Test PositionSizingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PositionSizingConfig()
        assert config.risk_per_trade_pct == 0.02
        assert config.min_confidence_threshold == 0.5
        assert config.max_position_multiplier == 3.0
        assert config.low_confidence_band == (0.5, 0.65)
        assert config.medium_confidence_band == (0.65, 0.8)
        assert config.high_confidence_band == (0.8, 1.0)
        assert config.low_confidence_multiplier == 0.5
        assert config.medium_confidence_multiplier == 1.0
        assert config.high_confidence_multiplier == 2.0
        assert config.max_position_pct == 0.10
        assert config.min_position_size == 1000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PositionSizingConfig(
            risk_per_trade_pct=0.01,
            min_confidence_threshold=0.6,
            max_position_multiplier=2.0,
            low_confidence_band=(0.6, 0.7),
            medium_confidence_band=(0.7, 0.85),
            high_confidence_band=(0.85, 1.0),
            low_confidence_multiplier=0.3,
            medium_confidence_multiplier=0.7,
            high_confidence_multiplier=1.5,
            max_position_pct=0.05,
            min_position_size=2000
        )
        assert config.risk_per_trade_pct == 0.01
        assert config.min_confidence_threshold == 0.6
        assert config.max_position_multiplier == 2.0
        assert config.low_confidence_band == (0.6, 0.7)
        assert config.medium_confidence_band == (0.7, 0.85)
        assert config.high_confidence_band == (0.85, 1.0)
        assert config.low_confidence_multiplier == 0.3
        assert config.medium_confidence_multiplier == 0.7
        assert config.high_confidence_multiplier == 1.5
        assert config.max_position_pct == 0.05
        assert config.min_position_size == 2000


class TestPositionSize:
    """Test PositionSize dataclass."""
    
    def test_position_size_creation(self):
        """Test PositionSize creation."""
        position = PositionSize(
            units=10000,
            confidence_level="high",
            position_multiplier=2.0,
            risk_amount=200.0,
            confidence_score=0.85,
            is_valid=True,
            reason="Valid"
        )
        
        assert position.units == 10000
        assert position.confidence_level == "high"
        assert position.position_multiplier == 2.0
        assert position.risk_amount == 200.0
        assert position.confidence_score == 0.85
        assert position.is_valid is True
        assert position.reason == "Valid"


class TestDynamicPositionSizer:
    """Test DynamicPositionSizer class."""
    
    def test_default_position_sizer_creation(self):
        """Test creation of default position sizer."""
        sizer = create_default_position_sizer()
        assert isinstance(sizer, DynamicPositionSizer)
        assert sizer.config.risk_per_trade_pct == 0.02
        assert sizer.config.min_confidence_threshold == 0.5
        assert sizer.config.max_position_multiplier == 3.0
    
    def test_conservative_position_sizer_creation(self):
        """Test creation of conservative position sizer."""
        sizer = create_conservative_position_sizer()
        assert isinstance(sizer, DynamicPositionSizer)
        assert sizer.config.risk_per_trade_pct == 0.01
        assert sizer.config.min_confidence_threshold == 0.6
        assert sizer.config.max_position_multiplier == 2.0
    
    def test_calculate_position_size_no_confidence(self):
        """Test position size calculation without confidence."""
        sizer = create_default_position_sizer()
        position = sizer.calculate_position_size(
            account_equity=100000.0,
            stop_loss_pips=20.0,
            instrument="USD_JPY"
        )
        
        assert position.units > 0
        assert position.confidence_level == "invalid"
        assert position.position_multiplier == 0.0
        assert position.is_valid is False
        assert "No confidence provided" in position.reason
    
    def test_calculate_position_size_low_confidence(self):
        """Test position size calculation with low confidence."""
        sizer = create_default_position_sizer()
        position = sizer.calculate_position_size(
            account_equity=100000.0,
            stop_loss_pips=20.0,
            instrument="USD_JPY",
            raw_confidence=0.55  # Low confidence
        )
        
        assert position.units > 0
        assert position.confidence_level == "low"
        assert position.position_multiplier == 0.5
        assert position.is_valid is True
        assert "0.550 -> low level" in position.reason
    
    def test_calculate_position_size_medium_confidence(self):
        """Test position size calculation with medium confidence."""
        sizer = create_default_position_sizer()
        position = sizer.calculate_position_size(
            account_equity=100000.0,
            stop_loss_pips=20.0,
            instrument="USD_JPY",
            raw_confidence=0.75  # Medium confidence
        )
        
        assert position.units > 0
        assert position.confidence_level == "medium"
        assert position.position_multiplier == 1.0
        assert position.is_valid is True
        assert "0.750 -> medium level" in position.reason
    
    def test_calculate_position_size_high_confidence(self):
        """Test position size calculation with high confidence."""
        sizer = create_default_position_sizer()
        position = sizer.calculate_position_size(
            account_equity=100000.0,
            stop_loss_pips=20.0,
            instrument="USD_JPY",
            raw_confidence=0.9  # High confidence
        )
        
        assert position.units > 0
        assert position.confidence_level == "high"
        assert position.position_multiplier == 2.0
        assert position.is_valid is True
        assert "0.900 -> high level" in position.reason
    
    def test_calculate_position_size_invalid_confidence(self):
        """Test position size calculation with invalid confidence."""
        sizer = create_default_position_sizer()
        position = sizer.calculate_position_size(
            account_equity=100000.0,
            stop_loss_pips=20.0,
            instrument="USD_JPY",
            raw_confidence=0.4  # Below threshold
        )

        assert position.units == 0
        assert position.confidence_level == "invalid"
        assert position.position_multiplier == 0.0
        assert position.is_valid is False
        assert "Confidence too low" in position.reason

    def test_calculate_position_size_calibrated_confidence(self):
        """Test position size calculation with calibrated confidence."""
        from confidence_calibration import CalibrationResult

        sizer = create_default_position_sizer()
        calibrated_confidence = CalibrationResult(
            original_confidence=0.8,
            calibrated_confidence=0.75,
            adjustment_factors={},
            calibration_method="platt",
            is_valid=True,
            reason="Valid",
        )

        position = sizer.calculate_position_size(
            account_equity=100000.0,
            stop_loss_pips=20.0,
            instrument="USD_JPY",
            calibrated_confidence=calibrated_confidence,
        )

        assert position.units > 0
        assert position.confidence_level == "medium"
        assert position.position_multiplier == 1.0
        assert position.is_valid is True
        assert "0.750 -> medium level" in position.reason

    def test_get_confidence_recommendation(self):
        """Test confidence recommendation."""
        sizer = create_default_position_sizer()

        assert "AVOID" in sizer.get_confidence_recommendation(0.4)
        assert "SMALL POSITION" in sizer.get_confidence_recommendation(0.6)
        assert "NORMAL POSITION" in sizer.get_confidence_recommendation(0.75)
        assert "LARGE POSITION" in sizer.get_confidence_recommendation(0.9)


class TestFixedPositionSizer:
    """Test FixedPositionSizer class."""
    
    def test_fixed_position_sizer_creation(self):
        """Test creation of fixed position sizer."""
        sizer = FixedPositionSizer(fixed_size=15000)
        assert isinstance(sizer, FixedPositionSizer)
        assert sizer.fixed_size == 15000
    
    def test_calculate_fixed_position_size(self):
        """Test fixed position size calculation."""
        sizer = FixedPositionSizer(fixed_size=15000)
        position = sizer.calculate_position_size(
            account_equity=100000.0,
            stop_loss_pips=20.0,
            instrument="USD_JPY",
            raw_confidence=0.8
        )
        
        assert position.units == 15000
        assert position.confidence_level == "fixed"
        assert position.position_multiplier == 1.0
        assert position.is_valid is True
        assert position.reason == "Fixed position size"


class TestPositionSizingConstraints:
    """Test position sizing constraints."""
    
    def test_minimum_position_size_constraint(self):
        """Test minimum position size constraint."""
        sizer = create_default_position_sizer()
        position = sizer.calculate_position_size(
            account_equity=1000.0,  # Very small account
            stop_loss_pips=20.0,
            instrument="USD_JPY",
            raw_confidence=0.8
        )
        
        assert position.units >= 1000  # Minimum position size
    
    def test_maximum_position_pct_constraint(self):
        """Test maximum position percentage constraint."""
        sizer = create_default_position_sizer()
        position = sizer.calculate_position_size(
            account_equity=1000000.0,  # Very large account
            stop_loss_pips=20.0,
            instrument="USD_JPY",
            raw_confidence=0.8
        )
        
        # Should be constrained by max_position_pct (10%)
        max_allowed = int(1000000.0 * 0.10 / 1000) * 1000
        assert position.units <= max_allowed
    
    def test_maximum_position_multiplier_constraint(self):
        """Test maximum position multiplier constraint."""
        sizer = create_default_position_sizer()
        position = sizer.calculate_position_size(
            account_equity=100000.0,
            stop_loss_pips=20.0,
            instrument="USD_JPY",
            raw_confidence=0.9  # High confidence, should get 2.0x multiplier
        )
        
        # Base position should be multiplied by 2.0 (high confidence multiplier)
        # but constrained by max_position_multiplier (3.0)
        assert position.position_multiplier == 2.0
        assert position.confidence_level == "high"


class TestPositionSizingInstrumentHandling:
    """Test position sizing with different instruments."""
    
    def test_usd_jpy_position_sizing(self):
        """Test position sizing for USD/JPY."""
        sizer = create_default_position_sizer()
        position = sizer.calculate_position_size(
            account_equity=100000.0,
            stop_loss_pips=20.0,
            instrument="USD_JPY",
            raw_confidence=0.8
        )
        
        assert position.units > 0
        assert position.is_valid is True
    
    def test_eur_usd_position_sizing(self):
        """Test position sizing for EUR/USD."""
        sizer = create_default_position_sizer()
        position = sizer.calculate_position_size(
            account_equity=100000.0,
            stop_loss_pips=20.0,
            instrument="EUR_USD",
            raw_confidence=0.8
        )
        
        assert position.units > 0
        assert position.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__])
# — Raynergy-svg —
