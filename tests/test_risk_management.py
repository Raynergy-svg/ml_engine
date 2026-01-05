"""Unit tests for risk management module."""

import pytest
from risk_management import (
    RiskManagementConfig,
    RiskManagementResult,
    ConfidenceBasedRiskManager,
    FixedRiskManager,
    create_default_risk_manager,
    create_conservative_risk_manager,
)


class TestRiskManagementConfig:
    """Test RiskManagementConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RiskManagementConfig()
        assert config.low_confidence_rr == 1.5
        assert config.medium_confidence_rr == 2.0
        assert config.high_confidence_rr == 3.0
        assert config.low_confidence_threshold == 0.5
        assert config.medium_confidence_threshold == 0.7
        assert config.high_confidence_threshold == 0.85
        assert config.min_rr_ratio == 1.0
        assert config.max_rr_ratio == 5.0
        assert config.low_confidence_sl_multiplier == 1.2
        assert config.medium_confidence_sl_multiplier == 1.0
        assert config.high_confidence_sl_multiplier == 0.8
        assert config.low_confidence_tp_multiplier == 0.8
        assert config.medium_confidence_tp_multiplier == 1.0
        assert config.high_confidence_tp_multiplier == 1.5
        assert config.max_stop_loss_pips == 100.0
        assert config.min_take_profit_pips == 5.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RiskManagementConfig(
            low_confidence_rr=1.2,
            medium_confidence_rr=1.5,
            high_confidence_rr=2.0,
            low_confidence_threshold=0.6,
            medium_confidence_threshold=0.8,
            high_confidence_threshold=0.9,
            min_rr_ratio=1.0,
            max_rr_ratio=3.0,
            low_confidence_sl_multiplier=1.5,
            medium_confidence_sl_multiplier=1.1,
            high_confidence_sl_multiplier=0.9,
            low_confidence_tp_multiplier=0.7,
            medium_confidence_tp_multiplier=0.9,
            high_confidence_tp_multiplier=1.2,
            max_stop_loss_pips=50.0,
            min_take_profit_pips=10.0
        )
        assert config.low_confidence_rr == 1.2
        assert config.medium_confidence_rr == 1.5
        assert config.high_confidence_rr == 2.0
        assert config.low_confidence_threshold == 0.6
        assert config.medium_confidence_threshold == 0.8
        assert config.high_confidence_threshold == 0.9
        assert config.min_rr_ratio == 1.0
        assert config.max_rr_ratio == 3.0
        assert config.low_confidence_sl_multiplier == 1.5
        assert config.medium_confidence_sl_multiplier == 1.1
        assert config.high_confidence_sl_multiplier == 0.9
        assert config.low_confidence_tp_multiplier == 0.7
        assert config.medium_confidence_tp_multiplier == 0.9
        assert config.high_confidence_tp_multiplier == 1.2
        assert config.max_stop_loss_pips == 50.0
        assert config.min_take_profit_pips == 10.0


class TestRiskManagementResult:
    """Test RiskManagementResult dataclass."""
    
    def test_result_creation(self):
        """Test RiskManagementResult creation."""
        result = RiskManagementResult(
            stop_loss_pips=20.0,
            take_profit_pips=40.0,
            risk_reward_ratio=2.0,
            confidence_level="medium",
            sl_adjustment_factor=1.0,
            tp_adjustment_factor=1.0,
            is_valid=True,
            reason="Valid"
        )
        
        assert result.stop_loss_pips == 20.0
        assert result.take_profit_pips == 40.0
        assert result.risk_reward_ratio == 2.0
        assert result.confidence_level == "medium"
        assert result.sl_adjustment_factor == 1.0
        assert result.tp_adjustment_factor == 1.0
        assert result.is_valid is True
        assert result.reason == "Valid"


class TestConfidenceBasedRiskManager:
    """Test ConfidenceBasedRiskManager class."""
    
    def test_default_risk_manager_creation(self):
        """Test creation of default risk manager."""
        manager = create_default_risk_manager()
        assert isinstance(manager, ConfidenceBasedRiskManager)
        assert manager.config.low_confidence_rr == 1.5
        assert manager.config.medium_confidence_rr == 2.0
        assert manager.config.high_confidence_rr == 3.0
    
    def test_conservative_risk_manager_creation(self):
        """Test creation of conservative risk manager."""
        manager = create_conservative_risk_manager()
        assert isinstance(manager, ConfidenceBasedRiskManager)
        assert manager.config.low_confidence_rr == 1.2
        assert manager.config.medium_confidence_rr == 1.5
        assert manager.config.high_confidence_rr == 2.0
    
    def test_calculate_risk_levels_no_confidence(self):
        """Test risk level calculation without confidence."""
        manager = create_default_risk_manager()
        result = manager.calculate_risk_levels(
            entry_price=1.2000,
            calibrated_confidence=None,
            raw_confidence=None
        )
        
        assert result.stop_loss_pips == 0.0
        assert result.take_profit_pips == 0.0
        assert result.risk_reward_ratio == 0.0
        assert result.confidence_level == "invalid"
        assert result.sl_adjustment_factor == 0.0
        assert result.tp_adjustment_factor == 0.0
        assert result.is_valid is False
        assert "No confidence provided" in result.reason
    
    def test_calculate_risk_levels_low_confidence(self):
        """Test risk level calculation with low confidence."""
        manager = create_default_risk_manager()
        result = manager.calculate_risk_levels(
            entry_price=1.2000,
            calibrated_confidence=None,
            raw_confidence=0.6  # Low confidence
        )
        
        assert result.stop_loss_pips > 0
        assert result.take_profit_pips > 0
        assert result.risk_reward_ratio == 1.5  # Low confidence R:R
        assert result.confidence_level == "low"
        assert result.sl_adjustment_factor == 1.2  # Wider stops
        assert result.tp_adjustment_factor == 0.8  # Tighter targets
        assert result.is_valid is True
        assert "0.600 -> low level" in result.reason
    
    def test_calculate_risk_levels_medium_confidence(self):
        """Test risk level calculation with medium confidence."""
        manager = create_default_risk_manager()
        result = manager.calculate_risk_levels(
            entry_price=1.2000,
            calibrated_confidence=None,
            raw_confidence=0.75  # Medium confidence
        )
        
        assert result.stop_loss_pips > 0
        assert result.take_profit_pips > 0
        assert result.risk_reward_ratio == 2.0  # Medium confidence R:R
        assert result.confidence_level == "medium"
        assert result.sl_adjustment_factor == 1.0  # Standard stops
        assert result.tp_adjustment_factor == 1.0  # Standard targets
        assert result.is_valid is True
        assert "0.750 -> medium level" in result.reason
    
    def test_calculate_risk_levels_high_confidence(self):
        """Test risk level calculation with high confidence."""
        manager = create_default_risk_manager()
        result = manager.calculate_risk_levels(
            entry_price=1.2000,
            calibrated_confidence=None,
            raw_confidence=0.9  # High confidence
        )
        
        assert result.stop_loss_pips > 0
        assert result.take_profit_pips > 0
        assert result.risk_reward_ratio == 3.0  # High confidence R:R
        assert result.confidence_level == "high"
        assert result.sl_adjustment_factor == 0.8  # Tighter stops
        assert result.tp_adjustment_factor == 1.5  # Wider targets
        assert result.is_valid is True
        assert "0.900 -> high level" in result.reason
    
    def test_calculate_risk_levels_invalid_confidence(self):
        """Test risk level calculation with invalid confidence."""
        manager = create_default_risk_manager()
        result = manager.calculate_risk_levels(
            entry_price=1.2000,
            calibrated_confidence=None,
            raw_confidence=0.4  # Below threshold
        )
        
        assert result.stop_loss_pips == 0.0
        assert result.take_profit_pips == 0.0
        assert result.risk_reward_ratio == 0.0
        assert result.confidence_level == "invalid"
        assert result.sl_adjustment_factor == 0.0
        assert result.tp_adjustment_factor == 0.0
        assert result.is_valid is False
        assert "Below threshold" in result.reason
    
    def test_calculate_risk_levels_with_base_values(self):
        """Test risk level calculation with base SL/TP values."""
        manager = create_default_risk_manager()
        result = manager.calculate_risk_levels(
            entry_price=1.2000,
            calibrated_confidence=None,
            raw_confidence=0.8,  # High confidence
            base_stop_loss_pips=25.0,
            base_take_profit_pips=75.0
        )
        
        assert result.stop_loss_pips > 0
        assert result.take_profit_pips > 0
        assert result.risk_reward_ratio == 3.0  # High confidence R:R
        assert result.confidence_level == "high"
        assert result.sl_adjustment_factor == 0.8  # Tighter stops
        assert result.tp_adjustment_factor == 1.5  # Wider targets
        assert result.is_valid is True
    
    def test_calculate_risk_levels_with_instrument_constraints(self):
        """Test risk level calculation with instrument-specific constraints."""
        manager = create_default_risk_manager()
        result = manager.calculate_risk_levels(
            entry_price=150.00,
            calibrated_confidence=None,
            raw_confidence=0.8,
            instrument="USD_JPY"
        )
        
        # JPY pairs should have different minimum stop loss
        assert result.stop_loss_pips >= 10.0  # JPY minimum
        assert result.take_profit_pips >= 5.0  # General minimum
    
    def test_calculate_risk_levels_with_max_sl_constraint(self):
        """Test risk level calculation with maximum stop loss constraint."""
        manager = create_default_risk_manager()
        result = manager.calculate_risk_levels(
            entry_price=1.2000,
            calibrated_confidence=None,
            raw_confidence=0.8,
            base_stop_loss_pips=150.0  # Above max
        )
        
        assert result.stop_loss_pips <= 100.0  # Max constraint
    
    def test_get_confidence_recommendation(self):
        """Test confidence recommendation."""
        manager = create_default_risk_manager()
        
        assert "AGGRESSIVE" in manager.get_risk_recommendation(0.9)
        assert "MODERATE" in manager.get_risk_recommendation(0.75)
        assert "CONSERVATIVE" in manager.get_risk_recommendation(0.6)
        assert "AVOID" in manager.get_risk_recommendation(0.4)


class TestFixedRiskManager:
    """Test FixedRiskManager class."""
    
    def test_fixed_risk_manager_creation(self):
        """Test creation of fixed risk manager."""
        manager = FixedRiskManager(fixed_sl_pips=25.0, fixed_tp_pips=50.0)
        assert isinstance(manager, FixedRiskManager)
        assert manager.fixed_sl_pips == 25.0
        assert manager.fixed_tp_pips == 50.0
    
    def test_calculate_fixed_risk_levels(self):
        """Test fixed risk level calculation."""
        manager = FixedRiskManager(fixed_sl_pips=25.0, fixed_tp_pips=50.0)
        result = manager.calculate_risk_levels(
            entry_price=1.2000,
            calibrated_confidence=None,
            raw_confidence=0.8
        )
        
        assert result.stop_loss_pips == 25.0
        assert result.take_profit_pips == 50.0
        assert result.risk_reward_ratio == 2.0
        assert result.confidence_level == "fixed"
        assert result.sl_adjustment_factor == 1.0
        assert result.tp_adjustment_factor == 1.0
        assert result.is_valid is True
        assert result.reason == "Fixed risk levels"


class TestRiskManagementConfidenceLevels:
    """Test confidence level determination."""
    
    def test_get_confidence_level(self):
        """Test confidence level determination."""
        manager = create_default_risk_manager()
        
        assert manager._get_confidence_level(0.4) == "invalid"
        assert manager._get_confidence_level(0.55) == "low"
        assert manager._get_confidence_level(0.75) == "medium"
        assert manager._get_confidence_level(0.9) == "high"
    
    def test_get_adjustment_factors(self):
        """Test adjustment factor calculation."""
        manager = create_default_risk_manager()
        
        sl_factor, tp_factor = manager._get_adjustment_factors("low")
        assert sl_factor == 1.2
        assert tp_factor == 0.8
        
        sl_factor, tp_factor = manager._get_adjustment_factors("medium")
        assert sl_factor == 1.0
        assert tp_factor == 1.0
        
        sl_factor, tp_factor = manager._get_adjustment_factors("high")
        assert sl_factor == 0.8
        assert tp_factor == 1.5
        
        sl_factor, tp_factor = manager._get_adjustment_factors("invalid")
        assert sl_factor == 1.0
        assert tp_factor == 1.0
    
    def test_get_base_risk_reward_ratio(self):
        """Test base risk-reward ratio calculation."""
        manager = create_default_risk_manager()
        
        assert manager._get_base_risk_reward_ratio("low") == 1.5
        assert manager._get_base_risk_reward_ratio("medium") == 2.0
        assert manager._get_base_risk_reward_ratio("high") == 3.0
        assert manager._get_base_risk_reward_ratio("invalid") == 1.0


class TestRiskManagementConstraints:
    """Test risk management constraints."""
    
    def test_min_take_profit_constraint(self):
        """Test minimum take profit constraint."""
        manager = create_default_risk_manager()
        result = manager.calculate_risk_levels(
            entry_price=1.2000,
            calibrated_confidence=None,
            raw_confidence=0.8,
            base_take_profit_pips=2.0  # Below minimum
        )
        
        assert result.take_profit_pips >= 5.0  # Minimum constraint
    
    def test_max_stop_loss_constraint(self):
        """Test maximum stop loss constraint."""
        manager = create_default_risk_manager()
        result = manager.calculate_risk_levels(
            entry_price=1.2000,
            calibrated_confidence=None,
            raw_confidence=0.8,
            base_stop_loss_pips=150.0  # Above maximum
        )
        
        assert result.stop_loss_pips <= 100.0  # Maximum constraint


class TestRiskManagementInstrumentHandling:
    """Test risk management with different instruments."""
    
    def test_usd_jpy_risk_management(self):
        """Test risk management for USD/JPY."""
        manager = create_default_risk_manager()
        result = manager.calculate_risk_levels(
            entry_price=150.00,
            calibrated_confidence=None,
            raw_confidence=0.8,
            instrument="USD_JPY"
        )
        
        assert result.stop_loss_pips >= 10.0  # JPY minimum
        assert result.take_profit_pips >= 5.0
        assert result.is_valid is True
    
    def test_eur_usd_risk_management(self):
        """Test risk management for EUR/USD."""
        manager = create_default_risk_manager()
        result = manager.calculate_risk_levels(
            entry_price=1.2000,
            calibrated_confidence=None,
            raw_confidence=0.8,
            instrument="EUR_USD"
        )
        
        assert result.stop_loss_pips >= 5.0  # General minimum
        assert result.take_profit_pips >= 5.0
        assert result.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__])
# — Raynergy-svg —
