"""
Integration tests for CLI commands module.

Validates:
1. train_buddy re-export from cli.training matches the original
2. _MAJOR_PAIRS constant has expected FX pairs
3. Key buddy functions exist and are callable
4. Dispatch helpers have correct signatures
"""

from __future__ import annotations

import pytest
import sys

# Pre-load cli.commands to handle potential cascading import failures
# (cli/__init__.py -> candle_optimizer -> modular_trainers -> tensorflow).
# When tensorflow is not installed, the first import of any cli submodule
# fails because cli/__init__.py triggers the cascade. However, cli.commands
# itself gets loaded before the failure point, so subsequent imports work.
try:
    import cli.commands  # noqa: F401
except ModuleNotFoundError:
    pass


# ---------------------------------------------------------------------------
# Test: train_buddy re-export
# ---------------------------------------------------------------------------

class TestTrainBuddyReExport:
    """Verify train_buddy is re-exported from cli.training for backward compat."""

    def test_import_train_buddy_from_commands(self):
        """train_buddy should be importable from cli.commands."""
        from cli.commands import train_buddy
        assert callable(train_buddy), "train_buddy should be callable"

    def test_import_train_buddy_from_training(self):
        """train_buddy should be importable from cli.training."""
        from cli.training import train_buddy
        assert callable(train_buddy), "train_buddy should be callable"

    def test_same_function_object(self):
        """cli.commands.train_buddy should be the same object as cli.training.train_buddy."""
        from cli.commands import train_buddy as tb_commands
        from cli.training import train_buddy as tb_training
        assert tb_commands is tb_training, (
            "cli.commands.train_buddy should be re-exported from cli.training"
        )


# ---------------------------------------------------------------------------
# Test: _MAJOR_PAIRS constant
# ---------------------------------------------------------------------------

class TestMajorPairs:
    """Verify the _MAJOR_PAIRS constant."""

    def test_major_pairs_exist(self):
        """_MAJOR_PAIRS should be a non-empty list."""
        from cli.commands import _MAJOR_PAIRS
        assert isinstance(_MAJOR_PAIRS, list)
        assert len(_MAJOR_PAIRS) > 0

    def test_major_pairs_count(self):
        """Should have 7 major FX pairs."""
        from cli.commands import _MAJOR_PAIRS
        assert len(_MAJOR_PAIRS) == 7, (
            f"Expected 7 major pairs, got {len(_MAJOR_PAIRS)}: {_MAJOR_PAIRS}"
        )

    def test_eur_usd_in_pairs(self):
        """EUR_USD must be in the major pairs list."""
        from cli.commands import _MAJOR_PAIRS
        assert "EUR_USD" in _MAJOR_PAIRS

    def test_all_pairs_have_underscore_format(self):
        """All pairs should use OANDA underscore format (e.g., EUR_USD)."""
        from cli.commands import _MAJOR_PAIRS
        for pair in _MAJOR_PAIRS:
            assert "_" in pair, f"Pair {pair} missing underscore separator"
            parts = pair.split("_")
            assert len(parts) == 2, f"Pair {pair} should have exactly 2 currencies"
            assert len(parts[0]) == 3, f"Base currency in {pair} should be 3 chars"
            assert len(parts[1]) == 3, f"Quote currency in {pair} should be 3 chars"

    def test_major_pairs_content(self):
        """Verify all expected major pairs are present."""
        from cli.commands import _MAJOR_PAIRS

        expected = {"EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
                    "AUD_USD", "USD_CAD", "NZD_USD"}
        assert set(_MAJOR_PAIRS) == expected, (
            f"Unexpected pairs. Got {set(_MAJOR_PAIRS)}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Test: Key function existence
# ---------------------------------------------------------------------------

class TestKeyFunctions:
    """Verify key functions exist in cli.commands and have expected signatures."""

    def test_buddy_function_exists(self):
        """buddy() function should exist and be callable."""
        from cli.commands import buddy
        assert callable(buddy)

    def test_buddy_loop_function_exists(self):
        """buddy_loop() function should exist."""
        from cli.commands import buddy_loop
        assert callable(buddy_loop)

    def test_buddy_validate_function_exists(self):
        """buddy_validate() function should exist."""
        from cli.commands import buddy_validate
        assert callable(buddy_validate)

    def test_buddy_test_function_exists(self):
        """buddy_test() legacy function should exist."""
        from cli.commands import buddy_test
        assert callable(buddy_test)

    def test_dispatch_helpers_exist(self):
        """_dispatch_buddy and _dispatch_train_buddy should exist."""
        from cli.commands import _dispatch_buddy, _dispatch_train_buddy
        assert callable(_dispatch_buddy)
        assert callable(_dispatch_train_buddy)

    def test_normalize_command_args_exists(self):
        """_normalize_command_args helper should exist."""
        from cli.commands import _normalize_command_args
        assert callable(_normalize_command_args)

    def test_compute_force_units_exists(self):
        """_compute_force_units helper should exist."""
        from cli.commands import _compute_force_units
        assert callable(_compute_force_units)


# ---------------------------------------------------------------------------
# Test: _compute_force_units logic
# ---------------------------------------------------------------------------

class TestComputeForceUnits:
    """Test _compute_force_units helper function."""

    def test_none_inputs_return_none(self):
        """Both None -> None."""
        from cli.commands import _compute_force_units
        result = _compute_force_units(force_units_raw=None, force_margin_raw=None)
        assert result is None

    def test_force_units_raw_passthrough(self):
        """force_units_raw should be converted to int and returned."""
        from cli.commands import _compute_force_units
        result = _compute_force_units(force_units_raw=5000, force_margin_raw=None)
        assert result == 5000

    def test_force_units_raw_as_string(self):
        """String force_units_raw should be converted to int."""
        from cli.commands import _compute_force_units
        result = _compute_force_units(force_units_raw="2000", force_margin_raw=None)
        assert result == 2000

    def test_force_margin_conversion(self):
        """force_margin_raw should be converted to units via /0.05 formula."""
        from cli.commands import _compute_force_units
        result = _compute_force_units(force_units_raw=None, force_margin_raw=100.0)
        assert result == int(round(100.0 / 0.05))

    def test_force_units_takes_priority(self):
        """force_units_raw takes priority over force_margin_raw."""
        from cli.commands import _compute_force_units
        result = _compute_force_units(force_units_raw=3000, force_margin_raw=100.0)
        assert result == 3000


# ---------------------------------------------------------------------------
# Test: _normalize_command_args
# ---------------------------------------------------------------------------

class TestNormalizeCommandArgs:
    """Test command argument normalization."""

    def test_normalizes_uppercase_buddy(self):
        """'Buddy' command should be lowercased to 'buddy'."""
        from cli.commands import _normalize_command_args
        from types import SimpleNamespace

        args = SimpleNamespace(command="Buddy")
        _normalize_command_args(args)
        assert args.command == "buddy"

    def test_leaves_lowercase_unchanged(self):
        """'buddy' command should not be modified."""
        from cli.commands import _normalize_command_args
        from types import SimpleNamespace

        args = SimpleNamespace(command="buddy")
        _normalize_command_args(args)
        assert args.command == "buddy"

    def test_non_buddy_command_unchanged(self):
        """Other commands should not be modified."""
        from cli.commands import _normalize_command_args
        from types import SimpleNamespace

        args = SimpleNamespace(command="train")
        _normalize_command_args(args)
        assert args.command == "train"


# ---------------------------------------------------------------------------
# Test: Buddy interactive wizard check
# ---------------------------------------------------------------------------

class TestBuddyWizardCheck:
    """Test _maybe_run_buddy_interactive_wizard existence."""

    def test_wizard_check_function_exists(self):
        """_maybe_run_buddy_interactive_wizard should exist and be callable."""
        from cli.commands import _maybe_run_buddy_interactive_wizard
        assert callable(_maybe_run_buddy_interactive_wizard)

    def test_repl_launcher_exists(self):
        """_maybe_launch_buddy_repl should exist and be callable."""
        from cli.commands import _maybe_launch_buddy_repl
        assert callable(_maybe_launch_buddy_repl)

    def test_chat_launcher_exists(self):
        """_maybe_launch_buddy_chat should exist and be callable."""
        from cli.commands import _maybe_launch_buddy_chat
        assert callable(_maybe_launch_buddy_chat)

    def test_chat_launcher_opens_grounded_repl(self, monkeypatch):
        """buddy-chat should launch unified talk via the shared REPL helper."""
        from cli.commands import _maybe_launch_buddy_chat
        from types import SimpleNamespace

        called = {}

        monkeypatch.setattr(
            "cli.commands.launch_buddy_repl_from_wizard",
            lambda *args, **kwargs: called.update({"args": args, "kwargs": kwargs}),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["main.py", "buddy-chat", "--instrument", "EUR_USD"],
        )

        launched = _maybe_launch_buddy_chat(
            SimpleNamespace(
                command="buddy-chat",
                config="config/config_intel_optimized.yaml",
                model_path=None,
                instrument="EUR_USD",
                granularity="M5",
                candles=300,
                dry_run=False,
                all_features=False,
                verbose=False,
            )
        )

        assert launched is True
        assert called["kwargs"]["instrument"] == "EUR_USD"
        assert called["kwargs"]["execute"] is False


# ---------------------------------------------------------------------------
# Test: intelligent-mode wiring helpers
# ---------------------------------------------------------------------------

class TestBuddyIntelligentWiring:
    """Verify Buddy intelligent analysis is wired to action inputs."""

    def test_normalize_llm_news_sentiment(self):
        """Should map market-intel sentiment payload to LLM prompt shape."""
        from cli.commands import _normalize_llm_news_sentiment

        normalized = _normalize_llm_news_sentiment({
            "aggregate_label": "bullish",
            "aggregate_score": 0.42,
            "num_headlines": 7,
        })

        assert normalized == {
            "sentiment": "bullish",
            "score": 0.42,
            "headline_count": 7,
        }

    def test_run_buddy_intelligent_analysis_validates_trade(self, monkeypatch):
        """Tradable signals should go through LLM validation before action."""
        pd = pytest.importorskip("pandas")
        from cli.commands import _run_buddy_intelligent_analysis
        from types import SimpleNamespace

        calls: dict[str, object] = {}

        class FakeBuddyRawOutput:
            def __init__(self, **kwargs):
                self.payload = kwargs

        class FakeMultiModalContext:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def to_dict(self):
                return dict(self.kwargs)

        class FakeIntelligentMode:
            def get_full_intelligent_analysis(self, **kwargs):
                calls["reasoning"] = kwargs
                return {"reasoning": "Structured reasoning"}

            def get_adaptation_policy(self, **kwargs):
                calls["adaptive_policy"] = kwargs
                return SimpleNamespace(
                    approve=True,
                    size_multiplier=1.0,
                    reason="Adaptive memory neutral",
                    risk_flags=[],
                    sample_size=3,
                )

        def fake_validate_trade_with_risk_assessment(**kwargs):
            calls["validation"] = kwargs
            return SimpleNamespace(
                approve=False,
                size_multiplier=0.75,
                reason="event risk",
                risk_flags=["economic_event"],
            )

        fake_module = SimpleNamespace(
            BuddyRawOutput=FakeBuddyRawOutput,
            MultiModalContext=FakeMultiModalContext,
            get_intelligent_mode=lambda: FakeIntelligentMode(),
            validate_trade_with_risk_assessment=fake_validate_trade_with_risk_assessment,
            apply_adaptive_trade_policy=lambda validation, policy: validation,
        )
        monkeypatch.setitem(sys.modules, "buddy_intelligent_mode", fake_module)

        signal = SimpleNamespace(
            trade=True,
            direction="long",
            confidence=63.0,
            tcn_probability=0.61,
            ridge_confidence=52.0,
            xgb_momentum=0.24,
            rf_drawdown_pips=18.0,
            rf_streak_prob=0.22,
            meta_confidence=0.58,
            confidence_gate_passed=True,
            momentum_gate_passed=True,
            risk_gate_passed=True,
            regime_gate_passed=True,
            meta_gate_passed=True,
            reason=None,
            metadata={
                "intel_data": {
                    "sentiment": {
                        "aggregate_label": "bullish",
                        "aggregate_score": 0.25,
                        "num_headlines": 4,
                    }
                },
                "gate_observability": {"current": {"trade": True}},
                "stage_a": {"core_pass": True},
                "stage_b": {"score_passed": True},
                "gate_thresholds": {"min_tcn_probability": 0.53},
            },
        )
        df = pd.DataFrame([
            {
                "close": 1.1000,
                "atr": 0.0012,
                "atr_pct_14": 0.0011,
                "rsi": 58.0,
                "adx": 24.0,
            },
            {
                "close": 1.1015,
                "atr": 0.0013,
                "atr_pct_14": 0.0012,
                "rsi": 61.0,
                "adx": 27.0,
            },
        ])
        ensemble = SimpleNamespace(market_intel=object())

        result = _run_buddy_intelligent_analysis(
            instrument="EUR_USD",
            granularity="H1",
            signal=signal,
            df=df,
            ensemble=ensemble,
            intelligent=True,
            explain=True,
            llm_initialized=True,
        )

        assert result["llm_approved"] is False
        assert result["validation"].reason == "event risk"
        assert result["reasoning"]["reasoning"] == "Structured reasoning"
        assert result["adaptive_policy"].reason == "Adaptive memory neutral"
        assert isinstance(calls["reasoning"]["context"], FakeMultiModalContext)

    def test_buddy_execute_path_blocks_order_when_intelligent_rejects(self, monkeypatch, tmp_path, capsys):
        """The live Buddy command path should honor an intelligent-mode veto."""
        import json
        import pandas as pd
        from types import SimpleNamespace

        from cli.commands import buddy

        monkeypatch.chdir(tmp_path)
        meta_dir = tmp_path / "trained_data" / "models" / "EUR_USD"
        meta_dir.mkdir(parents=True)
        (meta_dir / "modular_ensemble.meta.json").write_text(json.dumps({"selected_pairs": ["EUR_USD"]}))

        orders: list[tuple[str, int]] = []

        class FakeProvider:
            is_available = True
            name = "fake"

        monkeypatch.setitem(
            sys.modules,
            "llm_providers",
            SimpleNamespace(initialize_buddy_llm=lambda provider=None: FakeProvider()),
        )

        class FakeTradeJournal:
            def sync_open_trades(self, client):
                return {"closed_updated": 0}

            def backfill_online_learning(self):
                return 0

            def log_trade(self, **kwargs):
                return None

        monkeypatch.setitem(
            sys.modules,
            "src.utils.trade_journal",
            SimpleNamespace(TradeJournal=FakeTradeJournal),
        )

        class FakeMarketIntelligence:
            def __init__(self, enable_online_learning=True):
                self.online_learner = SimpleNamespace(trade_buffer=[])

        monkeypatch.setitem(
            sys.modules,
            "market_intelligence",
            SimpleNamespace(MarketIntelligence=FakeMarketIntelligence),
        )

        class FakeClient:
            def get_account_summary(self):
                return {"account": {"NAV": "10000"}}

            def _request(self, *_args, **_kwargs):
                return {"trades": []}

            def get_candles(self, instrument, granularity, count, price):
                return {"candles": []}

            def create_market_order(self, *, instrument: str, units: int, **kwargs):
                orders.append((instrument, int(units)))
                return {"orderFillTransaction": {"tradeOpened": {"tradeID": "T-1"}, "price": "1.1010"}}

        fake_client = FakeClient()

        class FakeOandaPracticeClient:
            @classmethod
            def from_env(cls):
                return fake_client

        monkeypatch.setitem(
            sys.modules,
            "src.utils.oanda_practice",
            SimpleNamespace(OandaPracticeClient=FakeOandaPracticeClient),
        )

        monkeypatch.setitem(
            sys.modules,
            "src.utils.fx_paper",
            SimpleNamespace(
                candles_to_ohlcv_df=lambda _resp: pd.DataFrame(
                    [
                        {"close": 1.1000, "atr": 0.0010, "rsi": 56.0, "adx": 20.0},
                        {"close": 1.1010, "atr": 0.0012, "rsi": 60.0, "adx": 24.0},
                    ]
                )
            ),
        )

        class FakeFeatureEngineering:
            def __init__(self, *_args, **_kwargs):
                pass

            def create_features(self, df, include_all=True):
                return df

        monkeypatch.setitem(
            sys.modules,
            "src.data.feature_engineering",
            SimpleNamespace(FeatureEngineering=FakeFeatureEngineering),
        )

        signal = SimpleNamespace(
            trade=True,
            direction="long",
            size=1.0,
            confidence=63.0,
            tcn_probability=0.61,
            ridge_confidence=52.0,
            xgb_momentum=0.24,
            rf_drawdown_pips=18.0,
            rf_streak_prob=0.22,
            meta_confidence=0.58,
            confidence_gate_passed=True,
            momentum_gate_passed=True,
            risk_gate_passed=True,
            regime_gate_passed=True,
            meta_gate_passed=True,
            reason=None,
            metadata={
                "intel_data": {"sentiment": {"aggregate_label": "bullish", "aggregate_score": 0.2, "num_headlines": 3}},
                "gate_observability": {"current": {"trade": True}},
                "stage_a": {"core_pass": True},
                "stage_b": {"score_passed": True},
                "gate_thresholds": {"min_tcn_probability": 0.53},
            },
        )

        class FakeEnsemble:
            def __init__(self, *args, **kwargs):
                self.market_intel = object()

            def load_models(self, instrument=None):
                return None

            def get_model_load_report(self):
                return []

            def predict_verbose(self, df, equity=None, instrument=None, model_instrument=None):
                return {
                    "raw_signal": signal,
                    "gate_checks": ["gate ok"],
                    "decision": "→ TRADE",
                }

        monkeypatch.setitem(
            sys.modules,
            "src.core.modular_inference",
            SimpleNamespace(ModularEnsembleInference=FakeEnsemble),
        )

        class FakeBuddyRawOutput:
            def __init__(self, **kwargs):
                self.direction = kwargs.get("direction")
                self.confidence = kwargs.get("confidence", 0.0)
                self.probability = kwargs.get("probability")
                self.payload = kwargs

            def to_dict(self):
                return dict(self.payload)

        class FakeMultiModalContext:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def to_dict(self):
                return dict(self.kwargs)

        class FakeIntelligentMode:
            def get_full_intelligent_analysis(self, **kwargs):
                return {"reasoning": "Integrated reasoning"}

            def get_adaptation_policy(self, **kwargs):
                return SimpleNamespace(
                    approve=True,
                    size_multiplier=1.0,
                    reason="Adaptive memory neutral",
                    risk_flags=[],
                    sample_size=4,
                )

        def fake_validate_trade_with_risk_assessment(**kwargs):
            return SimpleNamespace(
                approve=False,
                size_multiplier=1.0,
                reason="event risk",
                risk_flags=["economic_event"],
            )

        monkeypatch.setitem(
            sys.modules,
            "buddy_intelligent_mode",
            SimpleNamespace(
                BuddyRawOutput=FakeBuddyRawOutput,
                MultiModalContext=FakeMultiModalContext,
                get_intelligent_mode=lambda: FakeIntelligentMode(),
                validate_trade_with_risk_assessment=fake_validate_trade_with_risk_assessment,
                apply_adaptive_trade_policy=lambda validation, policy: validation,
            ),
        )

        monkeypatch.setattr(
            "cli.commands.load_config",
            lambda *_args, **_kwargs: {"feature_engineering": {}},
        )
        monkeypatch.setattr(
            "cli.commands._build_inference_config_from_yaml",
            lambda **kwargs: (
                SimpleNamespace(
                    account_equity=10000.0,
                    use_rsi_extreme_gate=False,
                    use_trend_contra_gate=False,
                    risk_per_trade_pct=0.01,
                ),
                False,
            ),
        )
        monkeypatch.setattr("cli.commands._resolve_master_pair_for_buddy", lambda **kwargs: "EUR_USD")
        monkeypatch.setattr(sys, "argv", ["prog", "buddy", "--execute"])

        buddy(
            config_path="config.yaml",
            instrument="EUR_USD",
            granularity="H1",
            candles=100,
            execute=True,
            intelligent=True,
            explain=False,
            interactive_master=False,
        )

        out = capsys.readouterr().out
        from memory_client import MLEngineMemory

        runtime_state = MLEngineMemory(storage_path=tmp_path / "trained_data" / ".memory").get_runtime_state("buddy_runtime")
        assert orders == []
        assert "NO TRADE: LLM VALIDATION FAILED" in out
        assert "event risk" in out
        assert runtime_state["selected_llm_provider"] == "fake"
        assert runtime_state["last_no_trade_decision"]["reason"] == "event risk"
        assert runtime_state["last_no_trade_decision"]["decision_type"] == "intelligent_reject"

    def test_buddy_dry_run_signal_persists_last_trade_execution_decision(self, monkeypatch, tmp_path, capsys):
        """Trade signals should persist a positive execution rationale even in dry-run mode."""
        import json
        import pandas as pd
        from types import SimpleNamespace

        from cli.commands import buddy
        from memory_client import MLEngineMemory

        monkeypatch.chdir(tmp_path)
        meta_dir = tmp_path / "trained_data" / "models" / "EUR_USD"
        meta_dir.mkdir(parents=True)
        (meta_dir / "modular_ensemble.meta.json").write_text(json.dumps({"selected_pairs": ["EUR_USD"]}))

        class FakeProvider:
            is_available = True
            name = "fake"

        monkeypatch.setitem(
            sys.modules,
            "llm_providers",
            SimpleNamespace(initialize_buddy_llm=lambda provider=None: FakeProvider()),
        )

        class FakeTradeJournal:
            def sync_open_trades(self, client):
                return {"closed_updated": 0}

            def backfill_online_learning(self):
                return 0

            def log_trade(self, **kwargs):
                return None

        monkeypatch.setitem(
            sys.modules,
            "src.utils.trade_journal",
            SimpleNamespace(TradeJournal=FakeTradeJournal),
        )

        class FakeMarketIntelligence:
            def __init__(self, enable_online_learning=True):
                self.online_learner = SimpleNamespace(trade_buffer=[])

        monkeypatch.setitem(
            sys.modules,
            "market_intelligence",
            SimpleNamespace(MarketIntelligence=FakeMarketIntelligence),
        )

        class FakeClient:
            def get_account_summary(self):
                return {"account": {"NAV": "10000"}}

            def _request(self, *_args, **_kwargs):
                return {"trades": []}

            def get_candles(self, instrument, granularity, count, price):
                return {"candles": []}

        fake_client = FakeClient()

        class FakeOandaPracticeClient:
            @classmethod
            def from_env(cls):
                return fake_client

        monkeypatch.setitem(
            sys.modules,
            "src.utils.oanda_practice",
            SimpleNamespace(OandaPracticeClient=FakeOandaPracticeClient),
        )

        monkeypatch.setitem(
            sys.modules,
            "src.utils.fx_paper",
            SimpleNamespace(
                candles_to_ohlcv_df=lambda _resp: pd.DataFrame(
                    [
                        {"close": 1.1000, "atr": 0.0010, "rsi": 56.0, "adx": 20.0},
                        {"close": 1.1010, "atr": 0.0012, "rsi": 60.0, "adx": 24.0},
                    ]
                )
            ),
        )

        class FakeFeatureEngineering:
            def __init__(self, *_args, **_kwargs):
                pass

            def create_features(self, df, include_all=True):
                return df

        monkeypatch.setitem(
            sys.modules,
            "src.data.feature_engineering",
            SimpleNamespace(FeatureEngineering=FakeFeatureEngineering),
        )

        signal = SimpleNamespace(
            trade=True,
            direction="long",
            size=1.0,
            confidence=63.0,
            tcn_probability=0.61,
            ridge_confidence=52.0,
            xgb_momentum=0.24,
            rf_drawdown_pips=18.0,
            rf_streak_prob=0.22,
            meta_confidence=0.58,
            confidence_gate_passed=True,
            momentum_gate_passed=True,
            risk_gate_passed=True,
            regime_gate_passed=True,
            meta_gate_passed=True,
            reason=None,
            metadata={
                "intel_data": {"sentiment": {"aggregate_label": "bullish", "aggregate_score": 0.2, "num_headlines": 3}},
                "gate_observability": {"current": {"trade": True}},
                "stage_a": {"core_pass": True},
                "stage_b": {"score_passed": True},
                "gate_thresholds": {"min_tcn_probability": 0.53},
            },
        )

        class FakeEnsemble:
            def __init__(self, *args, **kwargs):
                self.market_intel = object()

            def load_models(self, instrument=None):
                return None

            def get_model_load_report(self):
                return []

            def predict_verbose(self, df, equity=None, instrument=None, model_instrument=None):
                return {
                    "raw_signal": signal,
                    "gate_checks": ["gate ok"],
                    "decision": "→ TRADE",
                }

        monkeypatch.setitem(
            sys.modules,
            "src.core.modular_inference",
            SimpleNamespace(ModularEnsembleInference=FakeEnsemble),
        )

        class FakeBuddyRawOutput:
            def __init__(self, **kwargs):
                self.direction = kwargs.get("direction")
                self.confidence = kwargs.get("confidence", 0.0)
                self.probability = kwargs.get("probability")
                self.payload = kwargs

            def to_dict(self):
                return dict(self.payload)

        class FakeMultiModalContext:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def to_dict(self):
                return dict(self.kwargs)

        class FakeIntelligentMode:
            def get_full_intelligent_analysis(self, **kwargs):
                return {"reasoning": "Integrated reasoning"}

            def get_adaptation_policy(self, **kwargs):
                return SimpleNamespace(
                    approve=True,
                    size_multiplier=1.0,
                    reason="Adaptive memory neutral",
                    risk_flags=[],
                    sample_size=4,
                )

        def fake_validate_trade_with_risk_assessment(**kwargs):
            return SimpleNamespace(
                approve=True,
                size_multiplier=1.0,
                reason="all clear",
                risk_flags=[],
            )

        monkeypatch.setitem(
            sys.modules,
            "buddy_intelligent_mode",
            SimpleNamespace(
                BuddyRawOutput=FakeBuddyRawOutput,
                MultiModalContext=FakeMultiModalContext,
                get_intelligent_mode=lambda: FakeIntelligentMode(),
                validate_trade_with_risk_assessment=fake_validate_trade_with_risk_assessment,
                apply_adaptive_trade_policy=lambda validation, policy: validation,
            ),
        )

        monkeypatch.setattr(
            "cli.commands.load_config",
            lambda *_args, **_kwargs: {"feature_engineering": {}},
        )
        monkeypatch.setattr(
            "cli.commands._build_inference_config_from_yaml",
            lambda **kwargs: (
                SimpleNamespace(
                    account_equity=10000.0,
                    use_rsi_extreme_gate=False,
                    use_trend_contra_gate=False,
                    risk_per_trade_pct=0.01,
                ),
                False,
            ),
        )
        monkeypatch.setattr("cli.commands._resolve_master_pair_for_buddy", lambda **kwargs: "EUR_USD")

        buddy(
            config_path="config.yaml",
            instrument="EUR_USD",
            granularity="H1",
            candles=100,
            execute=False,
            intelligent=True,
            explain=False,
            interactive_master=False,
        )

        out = capsys.readouterr().out
        runtime_state = MLEngineMemory(storage_path=tmp_path / "trained_data" / ".memory").get_runtime_state("buddy_runtime")

        assert "DRY RUN MODE" in out
        assert runtime_state["last_trade_execution_decision"]["decision_type"] == "dry_run_signal"
        assert runtime_state["last_trade_execution_decision"]["reason"] == "All model gates passed and intelligent review approved"
        assert runtime_state["last_trade_execution_decision"]["direction"] == "long"


# ---------------------------------------------------------------------------
# Test: post-trade feedback helpers
# ---------------------------------------------------------------------------

class TestPostTradeFeedbackHelpers:
    """Verify post-trade helper behavior."""

    def test_extract_first_json_object(self):
        """Should parse first JSON object from mixed text."""
        from cli.commands import _extract_first_json_object

        parsed = _extract_first_json_object(
            "prefix text {\"quality\":\"good\",\"summary\":\"ok\"} trailing text"
        )
        assert isinstance(parsed, dict)
        assert parsed["quality"] == "good"
        assert parsed["summary"] == "ok"

    def test_build_post_trade_feedback_rl_fallback(self):
        """When LLM is disabled, RL/fallback guidance should be returned."""
        from cli.commands import _build_post_trade_feedback

        feedback = _build_post_trade_feedback(
            instrument="EUR_USD",
            direction="long",
            entry_price=1.1000,
            fill_price=1.1001,
            stop_loss_price=1.0985,
            take_profit_price=1.1030,
            stop_loss_pips=15.0,
            take_profit_pips=30.0,
            confidence=62.0,
            tcn_probability=0.57,
            llm_enabled=False,
            rl_scenarios={
                "now": {"action": "hold", "confidence": 0.8, "reason": "RL suggests hold"},
                "if_good": {"action": "exit_profit", "confidence": 0.7, "reason": "Take profit zone"},
                "if_bad": {"action": "exit_loss", "confidence": 0.9, "reason": "Cut loss"},
            },
        )

        assert feedback["source"] == "rl"
        assert feedback["quality"] in {"good", "neutral", "bad"}
        assert "hold" in feedback["summary"].lower() or feedback["summary"]
        assert "profit" in feedback["take_profit"].lower()
