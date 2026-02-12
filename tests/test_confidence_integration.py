"""Integration tests for confidence-based trading system.

This module tests the end-to-end integration of confidence calibration,
position sizing, and risk management components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

pytest.importorskip("tensorflow")

from src.risk.confidence_calibration import (
    create_default_calibrator,
    create_adjuster,
)
from src.risk.position_sizing import (
    create_default_position_sizer,
)
from src.risk.risk_management import (
    create_default_risk_manager,
)
from src.utils.unified_talk import TalkContext


class TestConfidenceIntegration:
    """Test integration of confidence-based trading components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calibrator = create_default_calibrator()
        self.adjuster = create_adjuster()
        self.position_sizer = create_default_position_sizer()
        self.risk_manager = create_default_risk_manager()
        
        # Mock TalkContext
        self.mock_ctx = Mock(spec=TalkContext)
        self.mock_ctx.last_result = {
            'prediction': 1.2050,
            'last_close': 1.2000,
            'state_probs': [0.3, 0.7],  # 70% bullish
            'risk': 0.3,
            'trend': 0.1
        }
        self.mock_ctx.oanda_instrument = "EUR_USD"
        self.mock_ctx.stop_loss_pips = 20.0
        self.mock_ctx.take_profit_pips = 40.0
        self.mock_ctx.use_stop_loss = True
        self.mock_ctx.use_take_profit = True
    
    def test_full_confidence_pipeline(self):
        """Test the complete confidence pipeline from raw confidence to trade execution."""
        # 1. Start with raw confidence from model
        raw_confidence = 0.85
        prediction_direction = 0.7  # 20% away from neutral
        tier2_win_probability = 0.65  # Above 50% threshold
        
        # 2. Apply confidence adjustment
        adjusted_result = self.adjuster.adjust_confidence(
            original_confidence=raw_confidence,
            prediction_direction=prediction_direction,
            tier2_win_probability=tier2_win_probability,
            trading_context='forex'
        )
        
        assert adjusted_result.is_valid is True
        assert adjusted_result.confidence_score < raw_confidence  # Should be reduced due to trading context
        
        # 3. Apply confidence calibration (if we had historical data)
        # For this test, we'll use the adjusted confidence as calibrated
        calibrated_confidence = adjusted_result.confidence_score
        
        # 4. Calculate position size
        position_result = self.position_sizer.calculate_position_size(
            account_equity=100000.0,
            stop_loss_pips=20.0,
            instrument="EUR_USD",
            raw_confidence=calibrated_confidence
        )
        
        assert position_result.is_valid is True
        assert position_result.units > 0
        assert position_result.confidence_level == "high"  # Should be high due to 0.85 confidence
        
        # 5. Calculate risk levels
        risk_result = self.risk_manager.calculate_risk_levels(
            entry_price=1.2000,
            raw_confidence=calibrated_confidence,
            base_stop_loss_pips=20.0,
            base_take_profit_pips=40.0,
            instrument="EUR_USD"
        )
        
        assert risk_result.is_valid is True
        assert risk_result.confidence_level == "high"
        assert risk_result.risk_reward_ratio == 3.0  # High confidence R:R
        
        # 6. Verify the complete pipeline
        assert calibrated_confidence > 0.5  # Above minimum threshold
        assert position_result.units > 1000  # Above minimum position size
        assert risk_result.stop_loss_pips > 0
        assert risk_result.take_profit_pips > 0
        assert risk_result.risk_reward_ratio >= 1.5  # Reasonable R:R
    
    def test_low_confidence_pipeline(self):
        """Test pipeline with low confidence - should result in no trade."""
        raw_confidence = 0.45  # Below threshold
        prediction_direction = 0.55  # Slightly directional
        tier2_win_probability = 0.4  # Below 50%
        
        # 1. Apply confidence adjustment
        adjusted_result = self.adjuster.adjust_confidence(
            original_confidence=raw_confidence,
            prediction_direction=prediction_direction,
            tier2_win_probability=tier2_win_probability,
            trading_context='forex'
        )
        
        assert adjusted_result.is_valid is False
        assert "Below threshold" in adjusted_result.reason
        
        # 2. Position sizing should return 0 units
        position_result = self.position_sizer.calculate_position_size(
            account_equity=100000.0,
            stop_loss_pips=20.0,
            instrument="EUR_USD",
            raw_confidence=raw_confidence
        )
        
        assert position_result.units == 0
        assert position_result.is_valid is False
        
        # 3. Risk management should return 0 values
        risk_result = self.risk_manager.calculate_risk_levels(
            entry_price=1.2000,
            raw_confidence=raw_confidence,
            instrument="EUR_USD"
        )
        
        assert risk_result.stop_loss_pips == 0.0
        assert risk_result.take_profit_pips == 0.0
        assert risk_result.is_valid is False
    
    def test_medium_confidence_pipeline(self):
        """Test pipeline with medium confidence."""
        raw_confidence = 0.75  # Medium confidence
        prediction_direction = 0.8  # Strong directional
        tier2_win_probability = 0.55  # Slightly above 50%
        
        # 1. Apply confidence adjustment
        adjusted_result = self.adjuster.adjust_confidence(
            original_confidence=raw_confidence,
            prediction_direction=prediction_direction,
            tier2_win_probability=tier2_win_probability,
            trading_context='forex'
        )
        
        assert adjusted_result.is_valid is True
        assert adjusted_result.confidence_score < raw_confidence  # Reduced due to trading context
        
        calibrated_confidence = adjusted_result.confidence_score
        
        # 2. Calculate position size
        position_result = self.position_sizer.calculate_position_size(
            account_equity=100000.0,
            stop_loss_pips=20.0,
            instrument="EUR_USD",
            raw_confidence=calibrated_confidence
        )
        
        assert position_result.is_valid is True
        assert position_result.confidence_level == "medium"
        assert position_result.position_multiplier == 1.0  # Medium confidence multiplier
        
        # 3. Calculate risk levels
        risk_result = self.risk_manager.calculate_risk_levels(
            entry_price=1.2000,
            raw_confidence=calibrated_confidence,
            base_stop_loss_pips=20.0,
            base_take_profit_pips=40.0,
            instrument="EUR_USD"
        )
        
        assert risk_result.is_valid is True
        assert risk_result.confidence_level == "medium"
        assert risk_result.risk_reward_ratio == 2.0  # Medium confidence R:R
        assert risk_result.sl_adjustment_factor == 1.0  # Standard stops
        assert risk_result.tp_adjustment_factor == 1.0  # Standard targets
    
    def test_confidence_calibration_integration(self):
        """Test integration with confidence calibration."""
        # Simulate historical data for calibration
        np.random.seed(42)
        historical_confidences = np.random.uniform(0.5, 1.0, 50)
        historical_outcomes = np.random.binomial(1, historical_confidences)
        
        # Fit calibrator
        self.calibrator.fit(historical_confidences, historical_outcomes)
        
        # Test calibration of new confidence
        test_confidence = 0.8
        calibration_result = self.calibrator.calibrate_confidence(test_confidence)
        
        assert calibration_result.original_confidence == test_confidence
        assert calibration_result.is_valid is True
        
        # Use calibrated confidence in position sizing
        position_result = self.position_sizer.calculate_position_size(
            account_equity=100000.0,
            stop_loss_pips=20.0,
            instrument="EUR_USD",
            calibrated_confidence=calibration_result
        )
        
        assert position_result.is_valid is True
        assert position_result.units > 0
    
    def test_position_sizing_constraints_integration(self):
        """Test position sizing constraints with confidence integration."""
        # Test with very large account equity
        position_result = self.position_sizer.calculate_position_size(
            account_equity=10000000.0,  # $10M account
            stop_loss_pips=20.0,
            instrument="EUR_USD",
            raw_confidence=0.9  # High confidence
        )
        
        # Should be constrained by max_position_pct (10%) using equity-based formula
        # Code: max_position_from_equity = int(account_equity * max_position_pct * 100)
        max_allowed = int(10000000.0 * 0.10 * 100)
        assert position_result.units <= max_allowed
        assert position_result.confidence_level == "high"
        
        # Test with very small account equity
        position_result = self.position_sizer.calculate_position_size(
            account_equity=1000.0,  # $1K account
            stop_loss_pips=20.0,
            instrument="EUR_USD",
            raw_confidence=0.9
        )
        
        # Should be constrained by minimum position size
        assert position_result.units >= 1000
        assert position_result.confidence_level == "high"
    
    def test_risk_management_constraints_integration(self):
        """Test risk management constraints with confidence integration."""
        # Test with very large base stop loss
        risk_result = self.risk_manager.calculate_risk_levels(
            entry_price=1.2000,
            raw_confidence=0.9,
            base_stop_loss_pips=200.0,  # Above max
            base_take_profit_pips=600.0,
            instrument="EUR_USD"
        )
        
        # Should be constrained by max_stop_loss_pips
        assert risk_result.stop_loss_pips <= 100.0
        assert risk_result.risk_reward_ratio == 3.0  # High confidence R:R
        
        # Test with very small base take profit
        risk_result = self.risk_manager.calculate_risk_levels(
            entry_price=1.2000,
            raw_confidence=0.9,
            base_stop_loss_pips=10.0,
            base_take_profit_pips=2.0,  # Below minimum
            instrument="EUR_USD"
        )
        
        # Should be constrained by min_take_profit_pips
        assert risk_result.take_profit_pips >= 5.0
        assert risk_result.risk_reward_ratio == 3.0
    
    def test_different_instruments_integration(self):
        """Test integration with different instruments."""
        instruments = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
        
        for instrument in instruments:
            # Test position sizing
            position_result = self.position_sizer.calculate_position_size(
                account_equity=100000.0,
                stop_loss_pips=20.0,
                instrument=instrument,
                raw_confidence=0.8
            )
            
            assert position_result.is_valid is True
            assert position_result.units > 0
            
            # Test risk management
            risk_result = self.risk_manager.calculate_risk_levels(
                entry_price=1.2000 if "JPY" not in instrument else 120.0,
                raw_confidence=0.8,
                base_stop_loss_pips=20.0,
                base_take_profit_pips=40.0,
                instrument=instrument
            )
            
            assert risk_result.is_valid is True
            assert risk_result.stop_loss_pips > 0
            assert risk_result.take_profit_pips > 0
            
            # JPY pairs should have different minimum stop loss
            if "JPY" in instrument:
                assert risk_result.stop_loss_pips >= 10.0
            else:
                assert risk_result.stop_loss_pips >= 5.0
    
    def test_confidence_threshold_integration(self):
        """Test integration around confidence thresholds."""
        thresholds = [0.49, 0.50, 0.51, 0.64, 0.65, 0.66, 0.79, 0.80, 0.81]
        
        for confidence in thresholds:
            # Test position sizing
            position_result = self.position_sizer.calculate_position_size(
                account_equity=100000.0,
                stop_loss_pips=20.0,
                instrument="EUR_USD",
                raw_confidence=confidence
            )
            
            if confidence < 0.5:
                assert position_result.units == 0
                assert position_result.is_valid is False
            else:
                assert position_result.units > 0
                assert position_result.is_valid is True
            
            # Test risk management
            risk_result = self.risk_manager.calculate_risk_levels(
                entry_price=1.2000,
                raw_confidence=confidence,
                instrument="EUR_USD"
            )
            
            if confidence < 0.5:
                assert risk_result.stop_loss_pips == 0.0
                assert risk_result.take_profit_pips == 0.0
                assert risk_result.is_valid is False
            else:
                assert risk_result.stop_loss_pips > 0
                assert risk_result.take_profit_pips > 0
                assert risk_result.is_valid is True


class TestUnifiedTalkIntegration:
    """Test integration with unified_talk.py."""
    
    def test_talk_context_integration(self):
        """Test that TalkContext properly integrates confidence components."""
        from src.utils.unified_talk import _load_context
        
        # Create a minimal config for testing
        config = {
            'paths': {
                'model_dir': 'trained_data/models',
                'checkpoint_dir': 'trained_data/checkpoints'
            },
            'data': {
                'feature_columns': ['open', 'high', 'low', 'close', 'volume'],
                'sequence_length': 60
            },
            'device': 'cpu'
        }
        
        # Mock the config loading
        with patch('src.utils.unified_talk.load_config', return_value=config):
            with patch('src.utils.unified_talk.select_latest_checkpoint') as mock_select:
                mock_select.return_value = None  # No checkpoint for this test
                
                # This should still create the context with confidence components
                try:
                    ctx = _load_context('test_config.yaml')
                    # If we get here, the confidence components were created successfully
                    assert ctx.confidence_calibrator is not None
                    assert ctx.position_sizer is not None
                    assert ctx.risk_manager is not None
                except FileNotFoundError:
                    # Expected since we're mocking the checkpoint
                    pass
    
    def test_confidence_pipeline_in_talk_context(self):
        """Test confidence pipeline within TalkContext."""
        # Create mock TalkContext with confidence components
        ctx = Mock()
        ctx.confidence_calibrator = create_default_calibrator()
        ctx.position_sizer = create_default_position_sizer()
        ctx.risk_manager = create_default_risk_manager()
        ctx.last_result = {
            'prediction': 1.2050,
            'last_close': 1.2000,
            'state_probs': [0.3, 0.7],
            'risk': 0.3,
            'trend': 0.1
        }
        ctx.oanda_instrument = "EUR_USD"
        ctx.stop_loss_pips = 20.0
        ctx.take_profit_pips = 40.0
        ctx.use_stop_loss = True
        ctx.use_take_profit = True
        
        # Simulate the confidence pipeline that would happen in trade execution
        if ctx.last_result and ctx.last_result.get('state_probs'):
            confidence = float(max(ctx.last_result['state_probs']))
            
            # Calculate position size
            position_result = ctx.position_sizer.calculate_position_size(
                account_equity=100000.0,
                stop_loss_pips=ctx.stop_loss_pips,
                instrument=ctx.oanda_instrument,
                raw_confidence=confidence
            )
            
            # Calculate risk levels
            risk_result = ctx.risk_manager.calculate_risk_levels(
                entry_price=ctx.last_result['last_close'],
                raw_confidence=confidence,
                base_stop_loss_pips=ctx.stop_loss_pips if ctx.use_stop_loss else None,
                base_take_profit_pips=ctx.take_profit_pips if ctx.use_take_profit else None,
                instrument=ctx.oanda_instrument
            )
            
            # Verify the pipeline worked
            assert position_result.is_valid is True
            assert risk_result.is_valid is True
            assert position_result.units > 0
            assert risk_result.stop_loss_pips > 0
            assert risk_result.take_profit_pips > 0


if __name__ == "__main__":
    pytest.main([__file__])
# — Raynergy-svg —
