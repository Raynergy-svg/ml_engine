import yaml
from types import SimpleNamespace

import pandas as pd

from src.agent.tool_registry import _render_planner_scan_summary, build_default_tool_registry


def test_render_planner_scan_summary_session_block():
    summary = {
        "count": 2,
        "tradable_count": 0,
        "approved_count": 0,
        "all_rows": [
            {"pair": "EUR_USD", "direction": "HOLD", "error": "Outside trading session (4:00 UTC, active: 8-21 UTC)"},
            {"pair": "GBP_USD", "direction": "HOLD", "error": "Outside trading session (4:00 UTC, active: 8-21 UTC)"},
        ],
    }

    text = _render_planner_scan_summary(summary, granularity="M5").lower()
    assert "nothing is tradeable right now" in text
    assert "session filter blocked every setup" in text


def test_render_planner_scan_summary_market_closed():
    summary = {
        "count": 2,
        "tradable_count": 0,
        "approved_count": 0,
        "all_rows": [
            {"pair": "EUR_USD", "direction": "HOLD", "error": "FX market closed (weekend: Saturday 17:00 UTC; reopens Sunday 22:00 UTC)"},
            {"pair": "GBP_USD", "direction": "HOLD", "error": "FX market closed (weekend: Saturday 17:00 UTC; reopens Sunday 22:00 UTC)"},
        ],
    }

    text = _render_planner_scan_summary(summary, granularity="H1").lower()
    assert "nothing is tradeable right now" in text
    assert "fx market is closed" in text
    assert "session filter" not in text


def test_render_planner_scan_summary_approved_setups():
    summary = {
        "count": 5,
        "tradable_count": 2,
        "approved_count": 1,
        "all_rows": [
            {"pair": "EUR_USD", "direction": "LONG", "error": None},
            {"pair": "GBP_USD", "direction": "SHORT", "error": None},
        ],
    }

    text = _render_planner_scan_summary(summary, granularity="M5").lower()
    assert "1 approved setups" in text
    assert "best read" in text
    assert "eur/usd long" in text


def test_render_planner_scan_summary_directional_but_unapproved():
    summary = {
        "count": 5,
        "tradable_count": 3,
        "approved_count": 0,
        "all_rows": [
            {"pair": "EUR_USD", "direction": "LONG", "error": None},
            {"pair": "GBP_USD", "direction": "SHORT", "error": None},
            {"pair": "USD_CHF", "direction": "SHORT", "error": None},
        ],
    }

    text = _render_planner_scan_summary(summary, granularity="H1").lower()
    assert "3 directional setups" in text
    assert "none cleared the full approval gates yet" in text


def test_scan_market_tool_forwards_run_agent_confirmation(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"inference": {}}, sort_keys=False), encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_buddy_scan(*, config_path, pairs=None, granularity="H1", top_n=5, force=False, diversified=False, auto_execute=False, no_execute=False, clean_output=False, run_agent_confirmation=False, **_kwargs):
        captured.update(
            {
                "config_path": config_path,
                "pairs": pairs,
                "granularity": granularity,
                "top_n": top_n,
                "auto_execute": auto_execute,
                "run_agent_confirmation": run_agent_confirmation,
            }
        )
        return []

    import cli.buddy_scanning as buddy_scanning

    monkeypatch.setattr(buddy_scanning, "buddy_scan", _fake_buddy_scan)

    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    state: dict[str, object] = {}
    result = registry.execute(
        "scan_market",
        {
            "pairs": "EUR_USD,GBP_USD",
            "granularity": "H1",
            "top_n": 2,
            "run_agent_confirmation": True,
        },
        state,
    )

    assert result.ok is True
    assert captured["pairs"] == "EUR_USD,GBP_USD"
    assert captured["granularity"] == "H1"
    assert captured["top_n"] == 2
    assert captured["run_agent_confirmation"] is True
    assert isinstance(result.result, dict)
    assert result.result["granularity"] == "H1"
    assert isinstance(state.get("last_scan_summary"), dict)
    assert state["last_scan_summary"]["granularity"] == "H1"


def test_override_guardrail_rejects_unknown_guardrail(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"scan": {}}, sort_keys=False), encoding="utf-8")
    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    result = registry.execute(
        "override_guardrail",
        {
            "guardrail_name": "volatility",
            "action": "disable",
            "scope": "this_scan",
        },
        {},
    )
    assert result.ok is False
    assert "guardrail_name='session'" in (result.error or "")


def test_override_guardrail_this_scan_is_consumed_by_scan_market(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"scan": {"enable_session_filter": True, "session_start_utc": 8, "session_end_utc": 21}}, sort_keys=False),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def _fake_buddy_scan(*, session_filter_enabled_override=None, session_window_utc_override=None, force=False, **_kwargs):
        captured["session_filter_enabled_override"] = session_filter_enabled_override
        captured["session_window_utc_override"] = session_window_utc_override
        captured["force"] = force
        return []

    import cli.buddy_scanning as buddy_scanning

    monkeypatch.setattr(buddy_scanning, "buddy_scan", _fake_buddy_scan)
    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    state: dict[str, object] = {}

    override_result = registry.execute(
        "override_guardrail",
        {"guardrail_name": "session", "action": "disable", "scope": "this_scan"},
        state,
    )
    assert override_result.ok is True
    assert "guardrail_overrides_once" in state

    scan_result = registry.execute(
        "scan_market",
        {"granularity": "H1", "top_n": 1},
        state,
    )
    assert scan_result.ok is True
    assert captured["session_filter_enabled_override"] is False
    assert captured["force"] is True
    assert "guardrail_overrides_once" not in state


def test_override_guardrail_this_session_persists_across_scans(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"scan": {"enable_session_filter": True, "session_start_utc": 8, "session_end_utc": 21}}, sort_keys=False),
        encoding="utf-8",
    )
    seen: list[tuple[object, object]] = []

    def _fake_buddy_scan(*, session_filter_enabled_override=None, session_window_utc_override=None, **_kwargs):
        seen.append((session_filter_enabled_override, session_window_utc_override))
        return []

    import cli.buddy_scanning as buddy_scanning

    monkeypatch.setattr(buddy_scanning, "buddy_scan", _fake_buddy_scan)
    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    state: dict[str, object] = {}

    override_result = registry.execute(
        "override_guardrail",
        {"guardrail_name": "session", "action": "set_window_utc", "scope": "this_session", "window_start_utc": 2, "window_end_utc": 22},
        state,
    )
    assert override_result.ok is True

    for _ in range(2):
        scan_result = registry.execute(
            "scan_market",
            {"granularity": "H1", "top_n": 1},
            state,
        )
        assert scan_result.ok is True

    assert len(seen) == 2
    assert seen[0][0] is True
    assert tuple(seen[0][1]) == (2, 22)
    assert seen[1][0] is True
    assert tuple(seen[1][1]) == (2, 22)


def test_override_guardrail_permanent_requires_confirmation(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"scan": {"enable_session_filter": True, "session_start_utc": 8, "session_end_utc": 21}}, sort_keys=False),
        encoding="utf-8",
    )
    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    state: dict[str, object] = {}

    pending_result = registry.execute(
        "override_guardrail",
        {"guardrail_name": "session", "action": "disable", "scope": "permanent"},
        state,
    )
    assert pending_result.ok is True
    assert isinstance(state.get("pending_guardrail_override"), dict)

    cfg_after_pending = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert cfg_after_pending["scan"]["enable_session_filter"] is True

    token = str(pending_result.result["confirmation_token"])
    confirm_result = registry.execute(
        "override_guardrail",
        {"guardrail_name": "session", "action": "disable", "scope": "permanent", "confirm_token": token},
        state,
    )
    assert confirm_result.ok is True
    cfg_after_confirm = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert cfg_after_confirm["scan"]["enable_session_filter"] is False
    assert "pending_guardrail_override" not in state


def test_override_guardrail_cancel_clears_pending_without_mutation(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"scan": {"enable_session_filter": True, "session_start_utc": 8, "session_end_utc": 21}}, sort_keys=False),
        encoding="utf-8",
    )
    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    state: dict[str, object] = {}

    pending_result = registry.execute(
        "override_guardrail",
        {"guardrail_name": "session", "action": "disable", "scope": "permanent"},
        state,
    )
    assert pending_result.ok is True
    cancel_result = registry.execute(
        "override_guardrail",
        {"guardrail_name": "session", "action": "disable", "scope": "permanent", "confirm_token": "cancel"},
        state,
    )
    assert cancel_result.ok is True
    assert "pending_guardrail_override" not in state
    cfg_after_cancel = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert cfg_after_cancel["scan"]["enable_session_filter"] is True


def test_get_status_includes_guardrail_override_state(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"scan": {"enable_session_filter": True, "session_start_utc": 8, "session_end_utc": 21}}, sort_keys=False),
        encoding="utf-8",
    )

    class _FakeMemory:
        def get_runtime_state(self, _key):
            return {}

    import src.agent.tool_registry as tool_registry

    monkeypatch.setattr(tool_registry, "MLEngineMemory", _FakeMemory)
    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    state = {
        "active_instrument": "EUR_USD",
        "granularity": "H1",
        "execute_mode": "dry-run",
        "guardrail_overrides_session": {"session": {"guardrail_name": "session", "action": "disable", "scope": "this_session", "session_filter_enabled": False, "session_window_utc": None}},
    }
    result = registry.execute("get_status", {}, state)
    assert result.ok is True
    assert result.result["session_filter_effective"] is False
    assert result.result["session_window_effective"] == [8, 21]
    assert isinstance(result.result["guardrail_overrides_session"], dict)


def test_configure_pair_sentiment_gate_updates_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "inference": {
                    "min_tcn_probability": 0.60,
                    "min_confidence": 50.0,
                    "min_momentum": 0.20,
                    "max_drawdown_pct": 0.025,
                    "max_streak_prob": 0.95,
                    "risk_per_trade_pct": 0.02,
                    "pip_value": 10.0,
                    "min_stop_loss_pips": 10.0,
                    "min_position_lots": 0.01,
                    "position_sizing_mode": "fixed",
                    "rl_min_risk_utilization": 0.25,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    result = registry.execute(
        "configure_pair_sentiment_gate",
        {"pair": "AUD/NZD", "enabled": True, "directions": ["short"]},
        {},
    )

    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pair_cfg = updated["inference"]["by_pair"]["AUD_NZD"]
    assert result.ok is True
    assert pair_cfg["sentiment_block_enabled"] is True
    assert pair_cfg["sentiment_block_directions"] == ["short"]


def test_configure_correlation_sentiment_gate_updates_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"inference": {}}, sort_keys=False),
        encoding="utf-8",
    )

    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    result = registry.execute(
        "configure_correlation_sentiment_gate",
        {"master_pair": "AUD_NZD", "direction": "short", "enabled": True},
        {},
    )

    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    by_pair = updated["inference"]["by_pair"]
    assert result.ok is True
    assert by_pair["AUD_NZD"]["sentiment_block_directions"] == ["short"]
    assert by_pair["EUR_AUD"]["sentiment_block_directions"] == ["long"]
    assert by_pair["GBP_AUD"]["sentiment_block_directions"] == ["long"]


def test_retrain_meta_labelers_tool_with_mocked_retrainer(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"inference": {}}, sort_keys=False), encoding="utf-8")

    from src.training import correlation_group_config
    from src.training import meta_labeler_retrainer

    monkeypatch.setattr(correlation_group_config, "get_unique_master_pairs", lambda: ["AUD_NZD", "EUR_GBP"])

    class FakeRetrainer:
        def __init__(self, config=None, output_dir=None, **kwargs):
            self.output_dir = output_dir

        def load_trade_journal(self, path):
            return [
                SimpleNamespace(instrument="AUD_NZD"),
                SimpleNamespace(instrument="EUR_GBP"),
            ]

        def fit(self, samples, verbose=False):
            return {"ensemble_val_accuracy": 0.71, "ensemble_val_auc": 0.74}

        def save(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake")

    monkeypatch.setattr(meta_labeler_retrainer, "MetaLabelerRetrainer", FakeRetrainer)

    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    result = registry.execute(
        "retrain_meta_labelers",
        {"data_source": "journal", "min_samples": 1},
        {},
    )

    assert result.ok is True
    assert result.result["trained_count"] == 2
    assert (tmp_path / "trained_data" / "models" / "AUD_NZD" / "meta_labeler.pkl").exists()
    assert (tmp_path / "trained_data" / "models" / "EUR_GBP" / "meta_labeler.pkl").exists()


def test_retrain_meta_labelers_tool_uses_distinct_simulation_seeds(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"inference": {}}, sort_keys=False), encoding="utf-8")

    from src.training import correlation_group_config
    from src.training import meta_labeler_retrainer

    monkeypatch.setattr(correlation_group_config, "get_unique_master_pairs", lambda: ["AUD_NZD", "EUR_GBP"])

    seen_seeds = []

    class FakeRetrainer:
        def __init__(self, config=None, output_dir=None, **kwargs):
            self.output_dir = output_dir

        def simulate_trades_from_model(
            self,
            n_trades=0,
            directional_accuracy=0.0,
            short_accuracy=0.0,
            long_accuracy=0.0,
            short_bias=0.0,
            seed=None,
        ):
            seen_seeds.append(seed)
            return [SimpleNamespace(instrument=self.output_dir.name)]

        def fit(self, samples, verbose=False):
            return {"ensemble_val_accuracy": 0.71, "ensemble_val_auc": 0.74}

        def save(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake")

    monkeypatch.setattr(meta_labeler_retrainer, "MetaLabelerRetrainer", FakeRetrainer)

    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    result = registry.execute(
        "retrain_meta_labelers",
        {"data_source": "simulation", "min_samples": 1},
        {},
    )

    assert result.ok is True
    assert len(seen_seeds) == 2
    assert seen_seeds[0] != seen_seeds[1]


def test_retrain_meta_labelers_tool_uses_pair_specific_simulation_profile(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"inference": {}}, sort_keys=False), encoding="utf-8")

    from src.training import correlation_group_config
    from src.training import meta_labeler_retrainer

    monkeypatch.setattr(correlation_group_config, "get_unique_master_pairs", lambda: ["AUD_NZD"])

    captured = {}

    class FakeRetrainer:
        def __init__(self, config=None, output_dir=None, **kwargs):
            self.output_dir = output_dir

        def simulate_trades_from_model(
            self,
            n_trades=0,
            directional_accuracy=0.0,
            short_accuracy=0.0,
            long_accuracy=0.0,
            short_bias=0.0,
            seed=None,
        ):
            captured.update(
                {
                    "directional_accuracy": directional_accuracy,
                    "short_accuracy": short_accuracy,
                    "long_accuracy": long_accuracy,
                    "short_bias": short_bias,
                    "seed": seed,
                }
            )
            return [SimpleNamespace(instrument=self.output_dir.name)]

        def fit(self, samples, verbose=False):
            return {"ensemble_val_accuracy": 0.71, "ensemble_val_auc": 0.74}

        def save(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake")

    monkeypatch.setattr(meta_labeler_retrainer, "MetaLabelerRetrainer", FakeRetrainer)

    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    result = registry.execute(
        "retrain_meta_labelers",
        {"data_source": "simulation", "min_samples": 1},
        {},
    )

    assert result.ok is True
    assert captured["directional_accuracy"] == 0.8525
    assert captured["short_accuracy"] == 1.0
    assert round(captured["long_accuracy"], 4) == 0.6722
    assert captured["short_bias"] == 0.55


def test_retrain_meta_labelers_tool_uses_pair_specific_backtest_path(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"inference": {}}, sort_keys=False), encoding="utf-8")

    backtest_path = tmp_path / "aud_nzd_backtest.json"
    backtest_path.write_text('{"trades": [{"instrument": "AUD_NZD", "pnl": 1.0}]}', encoding="utf-8")

    from src.training import correlation_group_config
    from src.training import meta_labeler_retrainer

    monkeypatch.setattr(correlation_group_config, "get_unique_master_pairs", lambda: ["AUD_NZD"])

    seen_paths = []

    class FakeRetrainer:
        def __init__(self, config=None, output_dir=None, **kwargs):
            self.output_dir = output_dir

        def load_backtest_results(self, path):
            seen_paths.append(str(path))
            return [SimpleNamespace(instrument="AUD_NZD")]

        def fit(self, samples, verbose=False):
            return {"ensemble_val_accuracy": 0.71, "ensemble_val_auc": 0.74}

        def save(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake")

    monkeypatch.setattr(meta_labeler_retrainer, "MetaLabelerRetrainer", FakeRetrainer)

    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    result = registry.execute(
        "retrain_meta_labelers",
        {"data_source": "backtest", "min_samples": 1, "backtest_paths": {"AUD_NZD": str(backtest_path)}},
        {},
    )

    assert result.ok is True
    assert seen_paths == [str(backtest_path)]


def test_run_basket_backtest_tool_with_mocked_scanner(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"inference": {}}, sort_keys=False), encoding="utf-8")

    from src.scanner import config as scanner_config_module
    from src.scanner import engine as scanner_engine_module

    class FakeScannerConfig:
        lookback_candles = 200

        @classmethod
        def from_cli_args(cls, **kwargs):
            return cls()

    class FakeScanner:
        def __init__(self, config=None):
            self._backtester = SimpleNamespace(
                run=lambda df, direction, window=50: SimpleNamespace(
                    total_pnl=0.08,
                    win_rate=0.62,
                    sharpe=1.3,
                )
            )

        def scan_with_analysis(self, **kwargs):
            return SimpleNamespace(
                analyses=[
                    SimpleNamespace(
                        pair="EUR_USD",
                        direction="LONG",
                        confidence=0.72,
                        gates_passed=True,
                        error=None,
                    ),
                    SimpleNamespace(
                        pair="AUD_USD",
                        direction="SHORT",
                        confidence=0.68,
                        gates_passed=True,
                        error=None,
                    ),
                ]
            )

        def _fetch_pair_data(self, pair, count):
            return pd.DataFrame({"close": [1.0 + i * 0.001 for i in range(80)]})

    monkeypatch.setattr(scanner_config_module, "ScannerConfig", FakeScannerConfig)
    monkeypatch.setattr(scanner_engine_module, "Scanner", FakeScanner)

    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    result = registry.execute(
        "run_basket_backtest",
        {"baskets": ["euro", "aussie"], "granularity": "H1"},
        {},
    )

    assert result.ok is True
    assert result.result["eligible_pairs"] == ["EUR_USD", "AUD_USD"]
    assert len(result.result["equity_curve"]) > 0


def test_configure_aggressive_regime_sizing_updates_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"scan": {}}, sort_keys=False), encoding="utf-8")

    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    result = registry.execute(
        "configure_aggressive_regime_sizing",
        {
            "enabled": True,
            "high_vol_scale": 1.6,
            "extreme_vol_scale": 1.9,
            "min_meta_confidence": 0.55,
        },
        {},
    )

    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scan_cfg = updated["scan"]
    assert result.ok is True
    assert scan_cfg["aggressive_mode"] is True
    assert scan_cfg["regime_scaling_enabled"] is True
    assert scan_cfg["aggressive_scale_high_vol"] == 1.6
    assert scan_cfg["aggressive_scale_extreme_vol"] == 1.9
    assert scan_cfg["aggressive_min_meta_confidence"] == 0.55


def test_configure_scan_profile_updates_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"scan": {}}, sort_keys=False), encoding="utf-8")

    registry = build_default_tool_registry(str(config_path), root_dir=tmp_path)
    result = registry.execute(
        "configure_scan_profile",
        {"profile": "aggressive"},
        {},
    )

    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scan_cfg = updated["scan"]
    assert result.ok is True
    assert scan_cfg["profile"] == "aggressive"
    assert "aggressive" in result.human_summary.lower()
