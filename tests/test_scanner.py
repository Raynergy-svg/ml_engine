"""
Tests for src/scanner module.

Tests cover:
- ScannerConfig dataclass defaults and post_init behavior
- PairAnalysis scoring, gate summary, and tradeability logic
- ScanResult filtering and aggregation properties
- GateEvaluator existence
"""

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestScannerConfig:
    """Test ScannerConfig dataclass."""

    def test_default_pairs(self):
        from src.scanner.config import ScannerConfig, MAJOR_PAIRS
        cfg = ScannerConfig()
        assert cfg.pairs == MAJOR_PAIRS

    def test_default_thresholds(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig()
        assert cfg.min_tcn_probability == 0.60
        assert cfg.min_confidence == 50.0  # 0-100 scale (Ridge ADX score)
        assert cfg.min_momentum == 0.20
        assert cfg.max_drawdown_pct == 0.025

    def test_string_paths_converted(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig(config_path="config/config_improved_H1.yaml",
                            model_dir="trained_data/models")
        assert isinstance(cfg.config_path, Path)
        assert isinstance(cfg.model_dir, Path)
        assert cfg.config_path.is_absolute()
        assert cfg.model_dir.is_absolute()

    def test_pip_values(self):
        from src.scanner.config import ScannerConfig, PIP_VALUES
        cfg = ScannerConfig()
        assert cfg.get_pip_value("EUR_USD") == 0.0001
        assert cfg.get_pip_value("USD_JPY") == 0.01
        # Unknown pair falls back to 0.0001
        assert cfg.get_pip_value("UNKNOWN_PAIR") == 0.0001

    def test_session_filter_defaults(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig()
        assert cfg.enable_session_filter is True
        assert cfg.session_start_utc == 8
        assert cfg.session_end_utc == 21

    def test_from_cli_args_force_disables_session(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig.from_cli_args(force=True)
        assert cfg.enable_session_filter is False

    def test_profile_aggressive_lowers_gate_thresholds(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig()
        cfg.apply_profile("aggressive")
        assert cfg.profile == "aggressive"
        assert cfg.min_confidence == 45.0
        assert cfg.min_momentum == 0.12
        assert cfg.min_atr_pips == 3.0
        assert cfg.min_volatility_regime == 0

    def test_profile_conservative_tightens_gate_thresholds(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig()
        cfg.apply_profile("conservative")
        assert cfg.profile == "conservative"
        assert cfg.min_confidence == 58.0
        assert cfg.min_momentum == 0.28
        assert cfg.min_atr_pips == 6.0
        assert cfg.min_volatility_regime == 2

    def test_unknown_profile_raises(self):
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig()
        with pytest.raises(ValueError):
            cfg.apply_profile("ultra")


# ---------------------------------------------------------------------------
# Results tests
# ---------------------------------------------------------------------------

def _make_pair_analysis(**kwargs):
    """Create a PairAnalysis with sensible defaults, overridable via kwargs."""
    from src.scanner.results import PairAnalysis
    defaults = dict(
        pair="EUR_USD",
        direction="LONG",
        confidence=0.70,
        gates_passed=True,
        volatility_gate_passed=True,
        tcn_probability=0.65,
        ridge_confidence=60.0,
        xgb_momentum=0.40,
        momentum=0.40,
        drawdown=0.01,
        rf_drawdown=0.01,
        trend_strength=0.50,
        confidence_passed=True,
        momentum_passed=True,
        risk_passed=True,
    )
    defaults.update(kwargs)
    return PairAnalysis(**defaults)


class TestPairAnalysis:
    """Test PairAnalysis dataclass."""

    def test_is_tradeable_all_gates_passed(self):
        pa = _make_pair_analysis(gates_passed=True, direction="LONG")
        assert pa.is_tradeable is True

    def test_not_tradeable_hold(self):
        """HOLD should never be treated as executable or backtestable."""
        pa = _make_pair_analysis(gates_passed=True, direction="HOLD")
        assert pa.is_tradeable is False

    def test_not_tradeable_direction_none(self):
        pa = _make_pair_analysis(gates_passed=True, direction=None)
        assert pa.is_tradeable is False

    def test_not_tradeable_gates_failed(self):
        pa = _make_pair_analysis(gates_passed=False, direction="LONG")
        assert pa.is_tradeable is False

    def test_not_tradeable_with_error(self):
        pa = _make_pair_analysis(gates_passed=True, direction="LONG",
                                 error="Model load failed")
        assert pa.is_tradeable is False

    def test_overall_score_range(self):
        pa = _make_pair_analysis()
        score = pa.overall_score
        assert 0.0 <= score <= 1.0

    def test_overall_score_zero_on_error(self):
        pa = _make_pair_analysis(error="something broke")
        assert pa.overall_score == 0.0

    def test_gate_summary_all_passed(self):
        pa = _make_pair_analysis(
            confidence_passed=True,
            confidence_gate_passed=True,
            momentum_passed=True,
            momentum_gate_passed=True,
            risk_passed=True,
            risk_gate_passed=True,
        )
        # gate_summary counts confidence, momentum, risk (3 gates)
        assert pa.gate_summary == "3/3"

    def test_gate_summary_some_failed(self):
        pa = _make_pair_analysis(
            confidence_passed=False,
            confidence_gate_passed=False,
            momentum_passed=False,
            momentum_gate_passed=False,
            risk_passed=True,
            risk_gate_passed=True,
        )
        assert pa.gate_summary == "1/3"

    def test_to_dict_keys(self):
        pa = _make_pair_analysis()
        d = pa.to_dict()
        assert "pair" in d
        assert "direction" in d
        assert "confidence" in d
        assert "is_tradeable" in d
        assert "overall_score" in d
        assert "volatility_regime" in d

    def test_default_fields(self):
        from src.scanner.results import PairAnalysis
        pa = PairAnalysis(pair="TEST_PAIR")
        assert pa.direction == "HOLD"
        assert pa.confidence == 0.0
        assert pa.gates_passed is False
        assert pa.error is None


class TestScanResult:
    """Test ScanResult aggregation."""

    def _make_result(self):
        from src.scanner.results import ScanResult
        return ScanResult(analyses=[
            _make_pair_analysis(pair="EUR_USD", gates_passed=True,
                                direction="LONG", confidence=0.80),
            _make_pair_analysis(pair="GBP_USD", gates_passed=True,
                                direction="SHORT", confidence=0.65),
            _make_pair_analysis(pair="USD_JPY", gates_passed=False,
                                direction="HOLD"),
            _make_pair_analysis(pair="AUD_USD", direction="LONG",
                                error="Model not found"),
        ])

    def test_tradeable_count(self):
        sr = self._make_result()
        assert len(sr.tradeable) == 2

    def test_tradeable_sorted_by_score(self):
        sr = self._make_result()
        scores = [a.overall_score for a in sr.tradeable]
        assert scores == sorted(scores, reverse=True)

    def test_non_tradeable(self):
        sr = self._make_result()
        # USD_JPY has gates_passed=False and no error
        non_t = sr.non_tradeable
        assert any(a.pair == "USD_JPY" for a in non_t)

    def test_errors(self):
        sr = self._make_result()
        assert len(sr.errors) == 1
        assert sr.errors[0].pair == "AUD_USD"

    def test_success_rate(self):
        sr = self._make_result()
        # 3 out of 4 succeeded (1 error)
        assert sr.success_rate == pytest.approx(0.75)

    def test_get_top_n(self):
        sr = self._make_result()
        top = sr.get_top_n(1)
        assert len(top) == 1
        assert top[0].pair == "EUR_USD"  # highest confidence

    def test_to_dict(self):
        sr = self._make_result()
        d = sr.to_dict()
        assert "tradeable_count" in d
        assert "error_count" in d
        assert d["error_count"] == 1
        assert d["tradeable_count"] == 2

    def test_empty_result(self):
        from src.scanner.results import ScanResult
        sr = ScanResult()
        assert sr.success_rate == 0.0
        assert len(sr.tradeable) == 0


class TestScannerEngine:
    """Targeted engine behavior tests for scan robustness."""

    @staticmethod
    def _make_df(close: float = 1.2345) -> pd.DataFrame:
        periods = 220
        idx = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "open": [close] * periods,
                "high": [close + 0.0005] * periods,
                "low": [close - 0.0005] * periods,
                "close": [close] * periods,
                "volume": [100] * periods,
            },
            index=idx,
        )

    def test_scan_uses_config_pairs(self):
        from src.scanner.engine import Scanner
        from src.scanner.config import ScannerConfig
        from src.scanner.results import PairAnalysis

        class PairSelectionScanner(Scanner):
            def _load_yaml_config(self):
                return None

            def _init_oanda_client(self):
                return False

            def _init_gate_evaluator(self):
                return False

            def _init_feature_engineer(self):
                return True

            def _init_modular_ensemble(self):
                self._ensemble_loaded = False
                self._modular_ensemble = None
                return False

            def _scan_pair(self, pair: str):
                return PairAnalysis(pair=pair, direction="LONG", confidence=0.6, gates_passed=True)

            def _run_sub_inference_pass(self, analyses, on_pair_complete=None):
                return analyses

        cfg = ScannerConfig(pairs=["GBP_USD", "USD_JPY"])
        cfg.enable_session_filter = False
        scanner = PairSelectionScanner(cfg)

        result = scanner.scan(max_workers=1)

        assert [a.pair for a in result.analyses] == ["GBP_USD", "USD_JPY"]

    def test_scan_reports_technical_when_ensemble_not_loaded(self):
        from src.scanner.engine import Scanner
        from src.scanner.config import ScannerConfig
        from src.scanner.results import PairAnalysis

        class ModelTypeScanner(Scanner):
            def _load_yaml_config(self):
                return None

            def _init_oanda_client(self):
                return False

            def _init_gate_evaluator(self):
                self._gate_evaluator = None
                return False

            def _init_feature_engineer(self):
                return True

            def _init_modular_ensemble(self):
                self._modular_ensemble = object()
                self._ensemble_loaded = False
                self._ensemble_type = "ModularEnsembleInference"
                return False

            def _scan_pair(self, pair: str):
                return PairAnalysis(pair=pair, direction="LONG", confidence=0.6, gates_passed=True)

            def _run_sub_inference_pass(self, analyses, on_pair_complete=None):
                return analyses

        cfg = ScannerConfig(pairs=["EUR_USD"])
        cfg.enable_session_filter = False
        scanner = ModelTypeScanner(cfg)

        result = scanner.scan(max_workers=1)

        assert result.model_type == "technical"

    def test_cache_invalidates_when_profile_changes(self):
        from src.scanner.engine import Scanner
        from src.scanner.config import ScannerConfig

        class CacheAwareScanner(Scanner):
            def __init__(self, config):
                self.inference_calls = 0
                super().__init__(config)

            def _load_yaml_config(self):
                return None

            def _init_oanda_client(self):
                return False

            def _init_gate_evaluator(self):
                self._gate_evaluator = None
                return False

            def _init_feature_engineer(self):
                self._feature_engineer = lambda df: df.assign(
                    atr=0.001,
                    rsi=45.0,
                    adx=25.0,
                    sma_20=df["close"],
                    sma_50=df["close"],
                )
                return True

            def _init_modular_ensemble(self):
                self._modular_ensemble = None
                self._ensemble_loaded = False
                self._ensemble_type = None
                return False

            def _fetch_pair_data(self, pair: str, count: int = 500):
                return TestScannerEngine._make_df()

            def _run_inference(self, df_raw, df_feat, pair: str):
                self.inference_calls += 1
                return (
                    "LONG",
                    0.62,
                    61.0,
                    58.0,
                    True,
                    2,
                    True,
                    {
                        "momentum": 0.32,
                        "momentum_acceleration": False,
                        "momentum_passed": True,
                        "confidence_score": 58.0,
                        "confidence_passed": True,
                        "risk_passed": True,
                        "volatility_gate_passed": True,
                        "drawdown": 0.01,
                        "model_source_pair": pair,
                    },
                )

        cfg = ScannerConfig(pairs=["EUR_USD"])
        cfg.enable_session_filter = False
        cfg.incremental_enabled = True
        cfg.price_change_threshold = 1.0
        scanner = CacheAwareScanner(cfg)

        scanner._scan_pair("EUR_USD")
        scanner._scan_pair("EUR_USD")
        assert scanner.inference_calls == 1

        scanner.config.apply_profile("conservative")
        scanner._scan_pair("EUR_USD")
        assert scanner.inference_calls == 2

    def test_cached_results_do_not_leak_mutations(self):
        from src.scanner.engine import Scanner
        from src.scanner.config import ScannerConfig

        class CacheIsolationScanner(Scanner):
            def __init__(self, config):
                self.inference_calls = 0
                super().__init__(config)

            def _load_yaml_config(self):
                return None

            def _init_oanda_client(self):
                return False

            def _init_gate_evaluator(self):
                self._gate_evaluator = None
                return False

            def _init_feature_engineer(self):
                self._feature_engineer = lambda df: df.assign(
                    atr=0.001,
                    rsi=45.0,
                    adx=25.0,
                    sma_20=df["close"],
                    sma_50=df["close"],
                )
                return True

            def _init_modular_ensemble(self):
                self._modular_ensemble = None
                self._ensemble_loaded = False
                self._ensemble_type = None
                return False

            def _fetch_pair_data(self, pair: str, count: int = 500):
                return TestScannerEngine._make_df()

            def _run_inference(self, df_raw, df_feat, pair: str):
                self.inference_calls += 1
                return (
                    "LONG",
                    0.57,
                    57.0,
                    54.0,
                    True,
                    2,
                    True,
                    {
                        "momentum": 0.30,
                        "momentum_acceleration": False,
                        "momentum_passed": True,
                        "confidence_score": 54.0,
                        "confidence_passed": True,
                        "risk_passed": True,
                        "volatility_gate_passed": True,
                        "drawdown": 0.01,
                        "model_source_pair": pair,
                    },
                )

        cfg = ScannerConfig(pairs=["EUR_USD"])
        cfg.enable_session_filter = False
        cfg.incremental_enabled = True
        cfg.price_change_threshold = 1.0
        scanner = CacheIsolationScanner(cfg)

        first = scanner._scan_pair("EUR_USD")
        pristine_confidence = first.confidence
        first.agent_passed = True
        first.agent_promoted = True
        first.confidence = 0.99

        cached = scanner._scan_pair("EUR_USD")

        assert scanner.inference_calls == 1
        assert cached.agent_passed is False
        assert cached.agent_promoted is False
        assert cached.confidence == pytest.approx(pristine_confidence)

    def test_specialist_agents_populate_reasoning_fields(self):
        from src.scanner.engine import Scanner
        from src.scanner.config import ScannerConfig

        class SupportiveAgentScanner(Scanner):
            def _load_yaml_config(self):
                return None

            def _init_oanda_client(self):
                return False

            def _init_gate_evaluator(self):
                self._gate_evaluator = None
                return False

            def _init_feature_engineer(self):
                def feature_engineer(df):
                    feat = df.copy()
                    feat["atr"] = 0.0012
                    feat["rsi"] = 42.0
                    feat["adx"] = 33.0
                    feat["sma_20"] = feat["close"].rolling(20).mean().bfill()
                    feat["sma_50"] = feat["close"].rolling(50).mean().bfill()
                    feat["volume_ratio_20"] = 1.3
                    feat["returns"] = feat["close"].pct_change().fillna(0.0)
                    return feat

                self._feature_engineer = feature_engineer
                return True

            def _init_modular_ensemble(self):
                self._modular_ensemble = None
                self._ensemble_loaded = False
                self._ensemble_type = None
                return False

            def _fetch_pair_data(self, pair: str, count: int = 500):
                periods = 220
                idx = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
                close = pd.Series([1.10 + i * 0.0002 for i in range(periods)], index=idx)
                return pd.DataFrame(
                    {
                        "open": close - 0.0001,
                        "high": close + 0.0003,
                        "low": close - 0.0003,
                        "close": close,
                        "volume": [150] * periods,
                    },
                    index=idx,
                )

            def _run_inference(self, df_raw, df_feat, pair: str):
                return (
                    "LONG",
                    0.64,
                    64.0,
                    58.0,
                    True,
                    2,
                    True,
                    {
                        "momentum": 0.36,
                        "momentum_acceleration": True,
                        "momentum_passed": True,
                        "confidence_score": 58.0,
                        "confidence_passed": True,
                        "risk_passed": True,
                        "volatility_gate_passed": True,
                        "drawdown": 0.01,
                        "model_source_pair": pair,
                    },
                )

        cfg = ScannerConfig(pairs=["EUR_USD"])
        cfg.enable_session_filter = False
        scanner = SupportiveAgentScanner(cfg)

        result = scanner._scan_pair("EUR_USD")

        assert result is not None
        assert result.weighted_vote_score >= result.weighted_vote_threshold
        assert result.why_trade
        assert result.agent_reasons
        assert "trend_align" in result.agent_reason_codes
        assert 0.0 <= result.uncertainty_score <= 1.0
        assert result.execution_quality_passed is True

    def test_specialist_agents_can_block_low_quality_setup(self):
        from src.scanner.engine import Scanner
        from src.scanner.config import ScannerConfig

        class AdverseAgentScanner(Scanner):
            def _load_yaml_config(self):
                return None

            def _init_oanda_client(self):
                return False

            def _init_gate_evaluator(self):
                self._gate_evaluator = None
                return False

            def _init_feature_engineer(self):
                def feature_engineer(df):
                    feat = df.copy()
                    feat["atr"] = 0.00012
                    feat["rsi"] = 78.0
                    feat["adx"] = 19.0
                    feat["sma_20"] = feat["close"].rolling(20).mean().bfill()
                    feat["sma_50"] = feat["close"].rolling(50).mean().bfill()
                    feat["volume_ratio_20"] = 0.45
                    feat["returns"] = feat["close"].pct_change().fillna(0.0)
                    return feat

                self._feature_engineer = feature_engineer
                return True

            def _init_modular_ensemble(self):
                self._modular_ensemble = None
                self._ensemble_loaded = False
                self._ensemble_type = None
                return False

            def _fetch_pair_data(self, pair: str, count: int = 500):
                periods = 220
                idx = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
                close = pd.Series([1.30 - i * 0.00015 for i in range(periods)], index=idx)
                return pd.DataFrame(
                    {
                        "open": close + 0.0001,
                        "high": close + 0.0002,
                        "low": close - 0.0002,
                        "close": close,
                        "volume": [70] * periods,
                    },
                    index=idx,
                )

            def _run_inference(self, df_raw, df_feat, pair: str):
                return (
                    "LONG",
                    0.51,
                    52.0,
                    46.0,
                    True,
                    0,
                    True,
                    {
                        "momentum": 0.22,
                        "momentum_acceleration": False,
                        "momentum_passed": True,
                        "confidence_score": 46.0,
                        "confidence_passed": True,
                        "risk_passed": False,
                        "volatility_gate_passed": False,
                        "drawdown": 0.04,
                        "model_source_pair": pair,
                    },
                )

        cfg = ScannerConfig(pairs=["EUR_USD"])
        cfg.enable_session_filter = False
        scanner = AdverseAgentScanner(cfg)

        result = scanner._scan_pair("EUR_USD")

        assert result is not None
        assert result.gates_passed is False
        assert result.weighted_vote_score < result.weighted_vote_threshold or result.blocked_by_circuit_breaker
        assert result.why_no_trade
        assert result.blocked_by_circuit_breaker is True
        assert any(code in result.agent_reason_codes for code in {"risk_block", "uncertainty_block", "volatility_block"})


class TestScannerDisplayReasoning:
    def test_display_prefers_specialist_reasoning_note(self):
        from src.scanner.display import ScannerDisplay

        analysis = _make_pair_analysis(
            gates_passed=False,
            confidence=0.58,
            why_no_trade=["uncertainty high (0.71)"],
        )

        display = ScannerDisplay()

        assert display._generate_descriptive_note(analysis) == "uncertainty high (0.71)"


# ---------------------------------------------------------------------------
# Gate evaluator tests
# ---------------------------------------------------------------------------

class TestGateEvaluator:
    """Test GateEvaluator class exists and is importable."""

    def test_gate_evaluator_importable(self):
        from src.scanner.gates import GateEvaluator
        assert GateEvaluator is not None
