"""Phase 37 (US-238 + US-239): Unit tests for GateEvaluator.

Tests model loading, evaluate_confidence, evaluate_momentum, evaluate_risk,
evaluate_volatility_regime, _compute_adx, _prepare_features_input, and
_extract_momentum_score. All tests use mocks — no real model files needed.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from src.scanner.gates import GateEvaluator, _predict_with_named_input_if_needed


# ---------------------------------------------------------------------------
# Helper: create a GateEvaluator with a temp model dir
# ---------------------------------------------------------------------------
def _make_evaluator(use_joint_only=True, min_vol_regime=2, create_joint_dir=True):
    tmpdir = tempfile.mkdtemp()
    model_dir = Path(tmpdir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    if use_joint_only and create_joint_dir:
        (model_dir / "joint").mkdir(parents=True, exist_ok=True)
    return GateEvaluator(
        model_dir=model_dir,
        use_joint_only=use_joint_only,
        min_volatility_regime=min_vol_regime,
    )


def _make_features(rows=100, include_ohlcv=True):
    """Create a mock feature DataFrame."""
    np.random.seed(42)
    close = 1.1000 + np.cumsum(np.random.randn(rows) * 0.001)
    data = {
        "close": close,
        "rsi": np.random.uniform(30, 70, rows),
        "macd": np.random.randn(rows) * 0.001,
        "adx": np.random.uniform(15, 40, rows),
        "atr": np.full(rows, 0.0015),
        "bb_upper": close + 0.003,
        "bb_lower": close - 0.003,
        "ema_20": close - 0.0003,
        "sma_200": close - 0.002,
        "volume": np.random.randint(100, 1000, rows),
    }
    if include_ohlcv:
        data["high"] = close + 0.002
        data["low"] = close - 0.002
        data["open"] = close - 0.0005
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests: GateEvaluator initialization
# ---------------------------------------------------------------------------
class TestGateEvaluatorInit:
    def test_default_init(self):
        ev = _make_evaluator()
        assert ev.use_joint_only is True
        assert ev.min_volatility_regime == 2
        assert ev._models_loaded is False

    def test_non_joint_mode(self):
        ev = _make_evaluator(use_joint_only=False)
        assert ev.model_dir == ev.base_model_dir

    def test_joint_mode_path(self):
        ev = _make_evaluator(use_joint_only=True)
        assert ev.model_dir.name == "joint"

    def test_min_vol_regime_clamped_low(self):
        ev = _make_evaluator(min_vol_regime=-5)
        assert ev.min_volatility_regime == 0

    def test_min_vol_regime_clamped_high(self):
        ev = _make_evaluator(min_vol_regime=10)
        assert ev.min_volatility_regime == 3

    def test_tcn_not_available_by_default(self):
        ev = _make_evaluator()
        assert ev.tcn_volatility_available is False

    def test_momentum_model_type_none(self):
        ev = _make_evaluator()
        assert ev._momentum_model_type == "none"


# ---------------------------------------------------------------------------
# Tests: load_models
# ---------------------------------------------------------------------------
class TestLoadModels:
    def test_load_models_missing_joint_dir(self):
        ev = _make_evaluator(create_joint_dir=False)
        with pytest.raises(FileNotFoundError, match="Joint model directory"):
            ev.load_models(require_tcn=False)

    def test_load_models_no_models_found(self):
        ev = _make_evaluator()
        # No TCN required, empty dir → should load but no models
        status = ev.load_models(require_tcn=False)
        assert status["momentum"] is False
        assert status["confidence"] is False
        assert status["risk"] is False
        assert ev._models_loaded is True

    def test_load_models_require_tcn_raises(self):
        ev = _make_evaluator()
        with pytest.raises(FileNotFoundError, match="TCN Volatility"):
            ev.load_models(require_tcn=True)


# ---------------------------------------------------------------------------
# Tests: _prepare_features_input
# ---------------------------------------------------------------------------
class TestPrepareFeatures:
    def test_valid_dataframe(self):
        ev = _make_evaluator()
        df = _make_features(rows=10)
        X = ev._prepare_features_input(df)
        assert X.shape[0] == 10
        assert X.ndim == 2

    def test_single_row(self):
        ev = _make_evaluator()
        df = _make_features(rows=1)
        X = ev._prepare_features_input(df)
        assert X.shape[0] == 1
        assert X.ndim == 2

    def test_none_raises(self):
        ev = _make_evaluator()
        with pytest.raises(ValueError, match="empty or None"):
            ev._prepare_features_input(None)

    def test_empty_raises(self):
        ev = _make_evaluator()
        with pytest.raises(ValueError, match="empty or None"):
            ev._prepare_features_input(pd.DataFrame())

    def test_with_target_feature_names(self):
        ev = _make_evaluator()
        df = _make_features(rows=5)
        X = ev._prepare_features_input(df, target_feature_names=["close", "rsi"])
        assert X.shape[1] == 2  # Only 2 features selected

    def test_missing_target_features_uses_available(self):
        ev = _make_evaluator()
        df = _make_features(rows=5)
        X = ev._prepare_features_input(
            df, target_feature_names=["close", "nonexistent_feature"]
        )
        assert X.shape[1] == 1  # Only 'close' found


# ---------------------------------------------------------------------------
# Tests: _extract_momentum_score
# ---------------------------------------------------------------------------
class TestExtractMomentumScore:
    def test_2d_multiclass(self):
        ev = _make_evaluator()
        proba = np.array([[0.3, 0.7]])
        score = ev._extract_momentum_score(proba)
        assert abs(score - 0.7) < 0.01

    def test_2d_single_class(self):
        ev = _make_evaluator()
        proba = np.array([[0.65]])
        score = ev._extract_momentum_score(proba)
        assert abs(score - 0.65) < 0.01

    def test_1d_array(self):
        ev = _make_evaluator()
        proba = np.array([0.42])
        score = ev._extract_momentum_score(proba)
        assert abs(score - 0.42) < 0.01

    def test_clamps_above_one(self):
        ev = _make_evaluator()
        proba = np.array([1.5])
        score = ev._extract_momentum_score(proba)
        assert score == 1.0

    def test_clamps_below_zero(self):
        ev = _make_evaluator()
        proba = np.array([-0.3])
        score = ev._extract_momentum_score(proba)
        assert score == 0.0

    def test_none_raises(self):
        ev = _make_evaluator()
        with pytest.raises(ValueError, match="empty or None"):
            ev._extract_momentum_score(None)

    def test_empty_raises(self):
        ev = _make_evaluator()
        with pytest.raises(ValueError, match="empty or None"):
            ev._extract_momentum_score(np.array([]))


# ---------------------------------------------------------------------------
# Tests: evaluate_momentum (no model loaded)
# ---------------------------------------------------------------------------
class TestEvaluateMomentumNoModel:
    def test_returns_neutral_when_no_model(self):
        ev = _make_evaluator()
        df = _make_features()
        score, accel = ev.evaluate_momentum(df)
        assert score == ev.NEUTRAL_MOMENTUM_SCORE
        assert accel is False

    def test_none_features_raises(self):
        ev = _make_evaluator()
        with pytest.raises(ValueError, match="cannot be None"):
            ev.evaluate_momentum(None)


# ---------------------------------------------------------------------------
# Tests: evaluate_momentum (with mock model)
# ---------------------------------------------------------------------------
class TestEvaluateMomentumWithModel:
    def test_catboost_primary(self):
        ev = _make_evaluator()
        mock_cb = MagicMock()
        mock_cb.predict_proba.return_value = np.array([[0.3, 0.7]])
        ev._catboost_momentum = mock_cb
        ev._momentum_model_type = "catboost"

        df = _make_features()
        score, accel = ev.evaluate_momentum(df)
        assert score > 0
        assert isinstance(accel, bool)

    def test_no_model_returns_neutral(self):
        """With no momentum models at all, should return neutral score."""
        ev = _make_evaluator()
        ev._catboost_momentum = None
        ev._xgboost_momentum = None

        df = _make_features()
        score, accel = ev.evaluate_momentum(df)
        assert score == ev.NEUTRAL_MOMENTUM_SCORE
        assert accel is False


# ---------------------------------------------------------------------------
# Tests: evaluate_confidence (no model)
# ---------------------------------------------------------------------------
class TestEvaluateConfidenceNoModel:
    def test_fallback_to_adx(self):
        ev = _make_evaluator()
        df = _make_features(rows=50)
        confidence = ev.evaluate_confidence(df)
        # Should return ADX value from features or computed ADX
        assert 0.0 <= confidence <= 100.0

    def test_single_row_features(self):
        ev = _make_evaluator()
        df = _make_features(rows=1)
        confidence = ev.evaluate_confidence(df)
        assert isinstance(confidence, float)


# ---------------------------------------------------------------------------
# Tests: evaluate_risk (no model)
# ---------------------------------------------------------------------------
class TestEvaluateRiskNoModel:
    def test_returns_default_without_model(self):
        ev = _make_evaluator()
        df = _make_features()
        drawdown, streak = ev.evaluate_risk(df)
        assert drawdown == 0.02
        assert streak == 0.5

    def test_with_mock_model(self):
        ev = _make_evaluator()
        mock_rf = MagicMock()
        mock_rf.predict.return_value = np.array([0.03])
        ev._rf_risk = mock_rf

        df = _make_features()
        drawdown, streak = ev.evaluate_risk(df)
        assert drawdown >= 0
        assert 0 <= streak <= 0.9


# ---------------------------------------------------------------------------
# Tests: evaluate_volatility_regime (no model)
# ---------------------------------------------------------------------------
class TestEvaluateVolatilityRegimeNoModel:
    def test_returns_low_without_model(self):
        ev = _make_evaluator()
        df = _make_features(rows=100)
        regime, conf, allowed = ev.evaluate_volatility_regime(df)
        assert regime == GateEvaluator.REGIME_LOW
        assert conf == 0.0
        assert allowed is False


# ---------------------------------------------------------------------------
# Tests: evaluate_volatility_regime (with mock model)
# ---------------------------------------------------------------------------
class TestEvaluateVolatilityRegimeWithModel:
    def test_insufficient_data(self):
        ev = _make_evaluator()
        ev._tcn_volatility = MagicMock()  # Model present
        df = _make_features(rows=10)  # Too few rows for seq_len=60
        regime, conf, allowed = ev.evaluate_volatility_regime(df, seq_len=60)
        assert regime == GateEvaluator.REGIME_LOW
        assert allowed is False

    def test_dual_head_dict_output(self):
        ev = _make_evaluator(min_vol_regime=2)
        feature_names = [f"vol_feature_{i}" for i in range(6)]
        ev._tcn_feature_names = feature_names
        ev._tcn_volatility = MagicMock()
        ev._tcn_volatility.predict.return_value = {
            "classification": np.array([[0.05, 0.10, 0.80, 0.05]]),
            "regression": np.array([[0.25]]),
        }
        df = pd.DataFrame(
            {name: np.linspace(0.0, 1.0, 70) for name in feature_names}
        )

        regime, conf, allowed = ev.evaluate_volatility_regime(df, seq_len=60)

        assert regime == GateEvaluator.REGIME_HIGH
        assert conf == pytest.approx(0.80)
        assert allowed is True


# ---------------------------------------------------------------------------
# Tests: _compute_adx
# ---------------------------------------------------------------------------
class TestComputeADX:
    def test_valid_data(self):
        ev = _make_evaluator()
        df = _make_features(rows=50, include_ohlcv=True)
        adx = ev._compute_adx(df, period=14)
        assert adx is not None
        assert 0.0 <= adx <= 100.0

    def test_insufficient_data(self):
        ev = _make_evaluator()
        df = _make_features(rows=5, include_ohlcv=True)
        adx = ev._compute_adx(df, period=14)
        assert adx is None

    def test_missing_columns(self):
        ev = _make_evaluator()
        df = pd.DataFrame({"close": [1.0, 1.1, 1.2]})  # No high/low
        adx = ev._compute_adx(df, period=14)
        assert adx is None  # Should return None due to exception

    def test_period_1(self):
        ev = _make_evaluator()
        df = _make_features(rows=20, include_ohlcv=True)
        adx = ev._compute_adx(df, period=1)
        # With period=1, should still produce a value
        assert adx is not None


# ---------------------------------------------------------------------------
# Tests: _predict_with_named_input_if_needed (module-level helper)
# ---------------------------------------------------------------------------
class TestPredictWithNamedInput:
    def test_model_without_inputs_attr(self):
        mock_model = MagicMock()
        mock_model.inputs = None
        mock_model.predict.return_value = np.array([[0.7]])

        batch = np.random.randn(1, 10)
        result = _predict_with_named_input_if_needed(mock_model, batch)
        mock_model.predict.assert_called_once()
        assert result is not None

    def test_model_with_single_named_input(self):
        mock_model = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input_layer:0"
        mock_model.inputs = [mock_input]
        mock_model.predict.return_value = np.array([[0.8]])

        batch = np.random.randn(1, 10)
        result = _predict_with_named_input_if_needed(mock_model, batch)
        assert result is not None


# ---------------------------------------------------------------------------
# Tests: GateEvaluator constants
# ---------------------------------------------------------------------------
class TestGateConstants:
    def test_regime_constants(self):
        assert GateEvaluator.REGIME_LOW == 0
        assert GateEvaluator.REGIME_NORMAL == 1
        assert GateEvaluator.REGIME_HIGH == 2
        assert GateEvaluator.REGIME_EXTREME == 3

    def test_thresholds(self):
        assert 0.0 < GateEvaluator.ACCELERATION_THRESHOLD < 1.0
        assert 0.5 <= GateEvaluator.TRANSFORMER_THRESHOLD <= 0.8
        assert 0.5 <= GateEvaluator.META_LABELER_THRESHOLD <= 0.6


# ---------------------------------------------------------------------------
# Tests: Inference contract gap (no-mock, real-disk; 2026-05-28 fix)
# ---------------------------------------------------------------------------
class TestPrepareFeaturesContractGap:
    """Verify _prepare_features_input refuses on zero feature overlap, and
    evaluate_momentum abstains to neutral on the resulting ValueError rather
    than crashing or silently feeding the model arbitrary columns. Fixes the
    degenerate-fallback at gates.py:1395 that historically produced the
    "88% SHORT bias" pattern (see memory observation 1352 and the
    "Train↔Inference Contract Gates" rule in .claude/rules/improvement.md).
    """

    def _evaluator(self, tmp_path):
        models = tmp_path / "models"
        (models / "joint").mkdir(parents=True, exist_ok=True)
        return GateEvaluator(
            model_dir=models,
            use_joint_only=True,
        )

    def test_zero_overlap_raises_contract_gap(self, tmp_path):
        ev = self._evaluator(tmp_path)
        df = pd.DataFrame({"foo": [1.0], "bar": [2.0]})
        expected = ["macd_hist_momentum", "rsi_momentum", "adx"]
        with pytest.raises(ValueError, match="contract gap"):
            ev._prepare_features_input(df, expected)

    def test_partial_gap_warns_but_proceeds(self, tmp_path, caplog):
        ev = self._evaluator(tmp_path)
        df = pd.DataFrame({
            "macd_hist_momentum": [0.5],
            "rsi_momentum": [0.6],
            "extra_col": [1.0],
        })
        # 2 of 5 match — below the 50% threshold → loud warning, still proceeds.
        expected = ["macd_hist_momentum", "rsi_momentum", "adx", "atr_pct_10", "obv"]
        with caplog.at_level("WARNING"):
            X = ev._prepare_features_input(df, expected)
        assert X.shape == (1, 2)
        assert any("partial gap" in r.message for r in caplog.records), \
            f"expected partial-gap warning, got: {[r.message for r in caplog.records]}"

    def test_full_overlap_no_warn(self, tmp_path, caplog):
        ev = self._evaluator(tmp_path)
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        with caplog.at_level("WARNING"):
            X = ev._prepare_features_input(df, ["a", "b", "c"])
        assert X.shape == (1, 3)
        assert not any("contract" in r.message.lower() for r in caplog.records)

    def test_evaluate_momentum_abstains_on_contract_gap(self, tmp_path):
        ev = self._evaluator(tmp_path)
        # _xgb_feature_names is normally set by load_models(); simulate the
        # post-load state where the model expects names that the runtime
        # DataFrame doesn't have.
        ev._xgb_feature_names = ["macd_hist_momentum", "rsi_momentum"]
        df = pd.DataFrame({"foo": [1.0], "bar": [2.0]})  # zero overlap
        score, accel = ev.evaluate_momentum(df)
        assert score == ev.NEUTRAL_MOMENTUM_SCORE
        assert accel is False

    def test_evaluate_momentum_contract_gap_warning_suppressed_second_call(
        self, tmp_path, caplog
    ):
        ev = self._evaluator(tmp_path)
        ev._xgb_feature_names = ["macd_hist_momentum", "rsi_momentum"]
        df = pd.DataFrame({"foo": [1.0]})
        with caplog.at_level("WARNING"):
            ev.evaluate_momentum(df)
            ev.evaluate_momentum(df)
        # Both calls return neutral; warning should fire exactly once.
        gap_warnings = [
            r for r in caplog.records if "contract gap" in r.message.lower()
        ]
        assert len(gap_warnings) == 1, \
            f"expected one suppressed-after-first contract-gap warning, got {len(gap_warnings)}"
