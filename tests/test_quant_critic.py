"""Unit tests for quant critic module in openai_integration."""

import json
import sys
import pytest
from unittest.mock import patch, MagicMock

pytest.importorskip("openai_integration")

# Mock openai module before importing openai_integration
sys.modules['openai'] = MagicMock()
sys.modules['dotenv'] = MagicMock()

from openai_integration import (
    query_quant_critic,
    improve_trading_rationale,
)


class TestQueryQuantCritic:
    """Test query_quant_critic function."""

    def test_returns_dict_structure(self):
        """Test that function returns expected dict structure when API fails."""
        with patch('openai_integration.query_openai', return_value=None):
            result = query_quant_critic(
                ticker="USD_JPY",
                timeframe="H1",
                buddy_raw={"confidence": 0.85, "prediction": 1.234},
                current_response="Signal: BUY USD_JPY",
            )
            
            assert isinstance(result, dict)
            assert "feedback" in result
            assert "improved_rationale" in result
            assert "risk_assessment" in result
            assert "is_optimal" in result
            assert result["is_optimal"] is False

    def test_parses_valid_json_response(self):
        """Test that function correctly parses valid JSON response."""
        mock_response = json.dumps({
            "feedback": "The confidence seems high relative to the market conditions.",
            "improved_rationale": "BUY USD_JPY with caution due to high volatility.",
            "risk_assessment": "Consider reducing position size by 50%.",
            "is_optimal": False,
        })
        
        with patch('openai_integration.query_openai', return_value=mock_response):
            result = query_quant_critic(
                ticker="USD_JPY",
                timeframe="H1",
                buddy_raw={"confidence": 0.95},
                current_response="Signal: BUY USD_JPY",
            )
            
            assert result["feedback"] == "The confidence seems high relative to the market conditions."
            assert result["improved_rationale"] == "BUY USD_JPY with caution due to high volatility."
            assert result["risk_assessment"] == "Consider reducing position size by 50%."
            assert result["is_optimal"] is False

    def test_handles_optimal_response(self):
        """Test handling of optimal (no improvements needed) response."""
        mock_response = json.dumps({
            "feedback": "No improvements needed.",
            "improved_rationale": None,
            "risk_assessment": "Risk factors already well addressed.",
            "is_optimal": True,
        })
        
        with patch('openai_integration.query_openai', return_value=mock_response):
            result = query_quant_critic(
                ticker="EUR_USD",
                timeframe="M5",
                buddy_raw={"confidence": 0.75},
                current_response="Balanced signal with proper risk assessment.",
            )
            
            assert result["is_optimal"] is True
            assert result["improved_rationale"] is None

    def test_handles_invalid_json_response(self):
        """Test graceful handling of invalid JSON response."""
        invalid_response = "This is not valid JSON - free form feedback here."
        
        with patch('openai_integration.query_openai', return_value=invalid_response):
            result = query_quant_critic(
                ticker="GBP_USD",
                timeframe="H4",
                buddy_raw={"confidence": 0.5},
                current_response="Uncertain signal.",
            )
            
            # Should return the raw text as feedback
            assert result["feedback"] == invalid_response
            assert result["improved_rationale"] is None
            assert result["is_optimal"] is False


class TestImproveTradingRationale:
    """Test improve_trading_rationale convenience function."""

    def test_constructs_buddy_raw_correctly(self):
        """Test that buddy_raw is constructed with correct fields."""
        mock_response = json.dumps({
            "feedback": "Good analysis.",
            "improved_rationale": None,
            "risk_assessment": None,
            "is_optimal": True,
        })
        
        with patch('openai_integration.query_quant_critic') as mock_critic:
            mock_critic.return_value = json.loads(mock_response)
            
            improve_trading_rationale(
                ticker="USD_JPY",
                timeframe="H1",
                prediction=110.500,
                confidence=0.85,
                last_price=110.450,
            )
            
            # Check that query_quant_critic was called with correct structure
            call_args = mock_critic.call_args
            assert call_args is not None
            
            kwargs = call_args.kwargs
            assert kwargs["ticker"] == "USD_JPY"
            assert kwargs["timeframe"] == "H1"
            
            buddy_raw = kwargs["buddy_raw"]
            expected_prediction = 110.500
            expected_last_price = 110.450
            assert buddy_raw["prediction"] == expected_prediction
            assert buddy_raw["confidence"] == 0.85
            assert buddy_raw["last_price"] == expected_last_price
            assert buddy_raw["delta"] == pytest.approx(expected_prediction - expected_last_price, rel=1e-5)
            assert buddy_raw["direction"] == "long"

    def test_direction_short(self):
        """Test correct direction when prediction is below last price."""
        mock_response = json.dumps({
            "feedback": "Good analysis.",
            "improved_rationale": None,
            "risk_assessment": None,
            "is_optimal": True,
        })
        
        with patch('openai_integration.query_quant_critic') as mock_critic:
            mock_critic.return_value = json.loads(mock_response)
            
            improve_trading_rationale(
                ticker="EUR_USD",
                timeframe="M5",
                prediction=1.0800,
                confidence=0.70,
                last_price=1.0850,
            )
            
            buddy_raw = mock_critic.call_args.kwargs["buddy_raw"]
            assert buddy_raw["direction"] == "short"

    def test_includes_optional_features(self):
        """Test that optional features are included when provided."""
        mock_response = json.dumps({
            "feedback": "Feature analysis included.",
            "improved_rationale": None,
            "risk_assessment": None,
            "is_optimal": True,
        })
        
        with patch('openai_integration.query_quant_critic') as mock_critic:
            mock_critic.return_value = json.loads(mock_response)
            
            features = {"rsi": 65, "macd": 0.002, "atr": 0.0015}
            gate_results = {"confidence_gate": True, "momentum_gate": True}
            
            improve_trading_rationale(
                ticker="USD_JPY",
                timeframe="H1",
                prediction=110.500,
                confidence=0.85,
                last_price=110.450,
                features=features,
                gate_results=gate_results,
            )
            
            buddy_raw = mock_critic.call_args.kwargs["buddy_raw"]
            assert buddy_raw["key_features"] == features
            assert buddy_raw["gate_results"] == gate_results

    def test_returns_critic_result(self):
        """Test that the function returns the quant critic result."""
        expected_result = {
            "feedback": "Consider market volatility.",
            "improved_rationale": "Add volatility context.",
            "risk_assessment": "High ATR environment.",
            "is_optimal": False,
        }
        
        with patch('openai_integration.query_quant_critic', return_value=expected_result):
            result = improve_trading_rationale(
                ticker="USD_JPY",
                timeframe="H1",
                prediction=110.500,
                confidence=0.85,
                last_price=110.450,
            )
            
            assert result == expected_result
