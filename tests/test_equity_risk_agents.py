"""US-008 — Tests for :mod:`src.equity.risk_agents`.

No mocks. Real :class:`RiskAgentLayer`, real :class:`RiskAgentConfig`, real
on-disk SHIP_GATE.json + state file via ``tmp_path``. Each gate is driven
with edge cases (zero / negative / None inputs) and the composite
``RiskAgentLayer.evaluate`` is exercised end-to-end.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.equity.risk_agents import (
    PortfolioState,
    RiskAgentConfig,
    RiskAgentError,
    RiskAgentLayer,
    RiskDecision,
    adv_participation,
    concentration_check,
    drawdown_guardian,
    gross_exposure_cap,
    per_name_exposure_cap,
    rebalance_gate,
    vol_targeter,
)


UNIVERSE_HASH = "abc12345" * 8  # 64-char placeholder
OTHER_HASH = "fedc6789" * 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_ship_gate(
    path: Path,
    *,
    gate_pass: bool = True,
    universe_hash: str = UNIVERSE_HASH,
    malformed: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if malformed:
        path.write_text("{not valid json")
        return path
    payload = {
        "gate_pass": bool(gate_pass),
        "net_sharpe": 0.55,
        "max_dd": 0.20,
        "positive_years": 11,
        "total_years": 16,
        "universe_hash": universe_hash,
        "asof": "2026-06-19",
        "recommendation": "PASS — test fixture",
        "criteria_source": "test",
        "thresholds": {},
        "pipeline_version": "2026-06-18-eq1",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _state(**overrides) -> PortfolioState:
    base = dict(
        nav=1_000_000.0,
        peak_nav=1_000_000.0,
        current_weights={"AAPL": 0.05, "MSFT": 0.05},
        recent_returns=[],
        last_rebalance_asof=None,
        active_plan_open=False,
        halted=False,
    )
    base.update(overrides)
    return PortfolioState(**base)


def _config(**overrides) -> RiskAgentConfig:
    base = dict(
        gross_cap=1.0,
        per_name_cap=0.10,
        target_vol=0.12,
        max_lev=1.0,
        vol_lookback=21,
        dd_soft=0.10,
        dd_hard=0.20,
        concentration_cap=0.30,
        concentration_top_k=3,
        max_pairwise_correlation=0.95,
        adv_participation_cap=0.05,
        adv_lookback_days=20,
        cooldown_days=1,
    )
    base.update(overrides)
    return RiskAgentConfig(**base)


# ---------------------------------------------------------------------------
# RiskAgentConfig invariants
# ---------------------------------------------------------------------------


def test_config_rejects_nonsensical_thresholds():
    with pytest.raises(RiskAgentError):
        RiskAgentConfig(gross_cap=0.0)
    with pytest.raises(RiskAgentError):
        RiskAgentConfig(per_name_cap=2.0, gross_cap=1.0)
    with pytest.raises(RiskAgentError):
        RiskAgentConfig(target_vol=0.0)
    with pytest.raises(RiskAgentError):
        RiskAgentConfig(dd_soft=0.30, dd_hard=0.20)
    with pytest.raises(RiskAgentError):
        RiskAgentConfig(vol_lookback=1)
    with pytest.raises(RiskAgentError):
        RiskAgentConfig(adv_participation_cap=0.0)
    with pytest.raises(RiskAgentError):
        RiskAgentConfig(adv_participation_cap=1.5)
    with pytest.raises(RiskAgentError):
        RiskAgentConfig(concentration_top_k=0)
    with pytest.raises(RiskAgentError):
        RiskAgentConfig(cooldown_days=-1)


# ---------------------------------------------------------------------------
# Drawdown guardian — runs every cycle, three bands
# ---------------------------------------------------------------------------


def test_drawdown_guardian_pass_below_soft():
    config = _config()
    state = _state(nav=950_000.0, peak_nav=1_000_000.0)  # 5% DD
    v = drawdown_guardian(state, config)
    assert v.passed is True
    assert v.block_trade is False
    assert v.metadata["degross_factor"] == 1.0
    assert v.metadata["halt"] is False


def test_drawdown_guardian_soft_band_degross():
    config = _config()
    state = _state(nav=850_000.0, peak_nav=1_000_000.0)  # 15% DD
    v = drawdown_guardian(state, config)
    assert v.passed is True
    assert v.block_trade is False
    assert 0.0 < v.metadata["degross_factor"] < 1.0
    assert v.metadata["halt"] is False
    # 15% sits exactly midway between 10% soft and 20% hard -> factor 0.5
    assert v.metadata["degross_factor"] == pytest.approx(0.5)


def test_drawdown_guardian_hard_breach_halts():
    config = _config()
    state = _state(nav=750_000.0, peak_nav=1_000_000.0)  # 25% DD
    v = drawdown_guardian(state, config)
    assert v.passed is False
    assert v.block_trade is True
    assert v.metadata["halt"] is True
    assert v.metadata["degross_factor"] == 0.0


def test_drawdown_guardian_handles_zero_peak():
    # Defensive: peak_nav==0 means we have no high-watermark to compare to;
    # gate treats that as full drawdown (block).
    state = _state(nav=0.0, peak_nav=0.0)
    v = drawdown_guardian(state, _config())
    assert v.passed is False
    assert v.metadata["drawdown"] == 1.0


def test_drawdown_guardian_handles_negative_nav():
    state = _state(nav=-1.0, peak_nav=1_000_000.0)
    v = drawdown_guardian(state, _config())
    assert v.block_trade is True
    assert v.metadata["drawdown"] == 1.0


# ---------------------------------------------------------------------------
# Vol targeter
# ---------------------------------------------------------------------------


def test_vol_targeter_warmup_pass_through():
    config = _config(vol_lookback=21)
    state = _state(recent_returns=[0.001, -0.002])
    v = vol_targeter(state, config)
    assert v.passed is True
    assert v.reason_code == "vol_warmup"
    assert v.metadata["vol_factor"] == 1.0


def test_vol_targeter_zero_returns_caps_at_max_lev():
    config = _config(vol_lookback=5, max_lev=1.0)
    state = _state(recent_returns=[0.0] * 5)
    v = vol_targeter(state, config)
    assert v.passed is True
    assert v.metadata["vol_factor"] == pytest.approx(1.0)


def test_vol_targeter_high_vol_scales_down():
    config = _config(vol_lookback=5, target_vol=0.10, max_lev=1.0)
    # Daily 2% moves -> annualised ~32% >> 10% target -> factor < 1
    state = _state(recent_returns=[0.02, -0.02, 0.02, -0.02, 0.02])
    v = vol_targeter(state, config)
    assert v.passed is True
    assert v.metadata["vol_factor"] < 1.0
    assert v.metadata["realized_vol"] > 0.10


def test_vol_targeter_blocks_on_non_finite():
    config = _config(vol_lookback=3)
    state = _state(recent_returns=[0.01, math.nan, 0.01])
    v = vol_targeter(state, config)
    assert v.passed is False
    assert v.block_trade is True
    assert v.reason_code == "vol_inputs_nonfinite"


# ---------------------------------------------------------------------------
# Gross + per-name exposure caps
# ---------------------------------------------------------------------------


def test_gross_cap_allows_within():
    config = _config(gross_cap=1.0)
    v = gross_exposure_cap({"AAPL": 0.5, "MSFT": 0.5}, config)
    assert v.passed is True


def test_gross_cap_blocks_excess():
    config = _config(gross_cap=1.0)
    v = gross_exposure_cap({"AAPL": 0.6, "MSFT": 0.6}, config)
    assert v.passed is False
    assert v.block_trade is True
    assert v.reason_code == "gross_cap_exceeded"


def test_gross_cap_handles_empty():
    v = gross_exposure_cap({}, _config())
    assert v.passed is True
    assert v.metadata["gross"] == 0.0


def test_gross_cap_rejects_nonfinite_weights():
    with pytest.raises(RiskAgentError):
        gross_exposure_cap({"AAPL": math.inf}, _config())


def test_per_name_cap_allows_within():
    config = _config(per_name_cap=0.10)
    v = per_name_exposure_cap({"AAPL": 0.10, "MSFT": 0.05}, config)
    assert v.passed is True


def test_per_name_cap_blocks_excess():
    config = _config(per_name_cap=0.10)
    v = per_name_exposure_cap({"AAPL": 0.15, "MSFT": 0.05}, config)
    assert v.passed is False
    assert v.block_trade is True
    assert v.metadata["max_name"] == "AAPL"


def test_per_name_cap_handles_empty():
    v = per_name_exposure_cap({}, _config())
    assert v.passed is True


def test_per_name_cap_rejects_non_string_key():
    with pytest.raises(RiskAgentError):
        per_name_exposure_cap({"": 0.05}, _config())


# ---------------------------------------------------------------------------
# Concentration check
# ---------------------------------------------------------------------------


def test_concentration_within_top_k_cap():
    # 5 names equal weighted -> top-3 share = 0.6; cap=0.7 -> pass
    config = _config(concentration_cap=0.7, concentration_top_k=3)
    weights = {t: 0.05 for t in ["A", "B", "C", "D", "E"]}
    v = concentration_check(weights, config)
    assert v.passed is True


def test_concentration_blocks_above_cap():
    # 5 names, top-1 dominates -> top-3 share = ~0.94 > cap=0.5
    config = _config(concentration_cap=0.5, concentration_top_k=3)
    weights = {"A": 0.50, "B": 0.02, "C": 0.02, "D": 0.02, "E": 0.02}
    v = concentration_check(weights, config)
    assert v.passed is False
    assert v.block_trade is True


def test_concentration_empty_and_zero_gross():
    assert concentration_check({}, _config()).passed is True
    v = concentration_check({"A": 0.0, "B": 0.0}, _config())
    assert v.passed is True


def test_concentration_correlation_breach_blocks():
    config = _config(max_pairwise_correlation=0.80)
    weights = {"A": 0.10, "B": 0.10, "C": 0.10}
    corr = pd.DataFrame(
        [[1.0, 0.95, 0.10], [0.95, 1.0, 0.10], [0.10, 0.10, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    v = concentration_check(weights, config, correlations=corr)
    assert v.passed is False
    assert v.block_trade is True
    assert v.reason_code == "concentration_corr_exceeded"


def test_concentration_correlation_within_passes():
    config = _config(max_pairwise_correlation=0.95)
    weights = {"A": 0.10, "B": 0.10}
    corr = pd.DataFrame(
        [[1.0, 0.50], [0.50, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    v = concentration_check(weights, config, correlations=corr)
    assert v.passed is True


def test_concentration_correlation_nonfinite_blocks():
    config = _config()
    weights = {"A": 0.10, "B": 0.10}
    corr = pd.DataFrame(
        [[1.0, np.nan], [np.nan, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    v = concentration_check(weights, config, correlations=corr)
    assert v.passed is False
    assert v.reason_code == "concentration_corr_nonfinite"


def test_concentration_rejects_non_dataframe_correlations():
    with pytest.raises(RiskAgentError):
        concentration_check(
            {"A": 0.1, "B": 0.1}, _config(), correlations=[[1.0]]
        )


# ---------------------------------------------------------------------------
# ADV participation cap
# ---------------------------------------------------------------------------


def test_adv_participation_no_delta_passes():
    cur = {"AAPL": 0.05}
    tgt = {"AAPL": 0.05}
    adv = {"AAPL": 1_000_000_000.0}
    v = adv_participation(tgt, cur, nav=1_000_000.0, adv=adv, config=_config())
    assert v.passed is True
    assert v.metadata["n_names_traded"] == 0


def test_adv_participation_within_cap():
    cur = {"AAPL": 0.00}
    tgt = {"AAPL": 0.05}  # trade notional = 5% * $1M = $50k
    adv = {"AAPL": 100_000_000.0}  # 0.05% participation
    v = adv_participation(
        tgt, cur, nav=1_000_000.0, adv=adv,
        config=_config(adv_participation_cap=0.05),
    )
    assert v.passed is True
    assert v.metadata["max_participation"] < 0.001


def test_adv_participation_exceeds_cap():
    cur = {"AAPL": 0.00}
    tgt = {"AAPL": 0.50}  # $500k trade
    adv = {"AAPL": 1_000_000.0}  # 50% of ADV — way over 5% cap
    v = adv_participation(
        tgt, cur, nav=1_000_000.0, adv=adv,
        config=_config(adv_participation_cap=0.05),
    )
    assert v.passed is False
    assert v.block_trade is True
    assert v.reason_code == "adv_participation_exceeded"
    assert v.metadata["worst_ticker"] == "AAPL"


def test_adv_participation_missing_panel_blocks():
    cur = {"AAPL": 0.0}
    tgt = {"AAPL": 0.05}
    v = adv_participation(
        tgt, cur, nav=1_000_000.0, adv=None, config=_config()
    )
    assert v.passed is False
    assert v.reason_code == "adv_panel_missing"


def test_adv_participation_missing_ticker_blocks():
    cur = {}
    tgt = {"AAPL": 0.05}
    adv = {"MSFT": 1_000_000.0}  # AAPL absent
    v = adv_participation(
        tgt, cur, nav=1_000_000.0, adv=adv, config=_config()
    )
    assert v.passed is False
    assert v.reason_code == "adv_value_invalid"


def test_adv_participation_zero_adv_blocks():
    cur = {}
    tgt = {"AAPL": 0.05}
    adv = {"AAPL": 0.0}
    v = adv_participation(
        tgt, cur, nav=1_000_000.0, adv=adv, config=_config()
    )
    assert v.passed is False
    assert v.reason_code == "adv_value_invalid"


def test_adv_participation_negative_adv_blocks():
    cur = {}
    tgt = {"AAPL": 0.05}
    adv = {"AAPL": -1.0}
    v = adv_participation(
        tgt, cur, nav=1_000_000.0, adv=adv, config=_config()
    )
    assert v.passed is False
    assert v.reason_code == "adv_value_invalid"


def test_adv_participation_nan_adv_blocks():
    cur = {}
    tgt = {"AAPL": 0.05}
    adv = {"AAPL": math.nan}
    v = adv_participation(
        tgt, cur, nav=1_000_000.0, adv=adv, config=_config()
    )
    assert v.passed is False
    assert v.reason_code == "adv_value_invalid"


def test_adv_participation_zero_nav_blocks():
    v = adv_participation(
        {"AAPL": 0.05}, {}, nav=0.0,
        adv={"AAPL": 1_000_000.0}, config=_config(),
    )
    assert v.passed is False
    assert v.reason_code == "adv_nav_nonpositive"


def test_adv_participation_non_numeric_adv_raises():
    with pytest.raises(RiskAgentError):
        adv_participation(
            {"AAPL": 0.05}, {}, nav=1_000_000.0,
            adv={"AAPL": "not-a-number"}, config=_config(),
        )


# ---------------------------------------------------------------------------
# Rebalance gate
# ---------------------------------------------------------------------------


def test_rebalance_first_cycle_allowed():
    asof = pd.Timestamp("2026-06-21", tz="UTC")
    v = rebalance_gate(_state(), asof, _config())
    assert v.passed is True
    assert v.reason_code == "rebalance_first_cycle"


def test_rebalance_cooldown_blocks():
    asof = pd.Timestamp("2026-06-21", tz="UTC")
    state = _state(last_rebalance_asof="2026-06-21T00:00:00+00:00")
    v = rebalance_gate(state, asof, _config(cooldown_days=1))
    assert v.passed is False
    assert v.reason_code == "rebalance_cooldown_active"


def test_rebalance_allows_after_cooldown():
    asof = pd.Timestamp("2026-06-23", tz="UTC")
    state = _state(last_rebalance_asof="2026-06-21T00:00:00+00:00")
    v = rebalance_gate(state, asof, _config(cooldown_days=1))
    assert v.passed is True


def test_rebalance_blocks_when_plan_active():
    asof = pd.Timestamp("2026-06-25", tz="UTC")
    state = _state(
        last_rebalance_asof="2026-06-21T00:00:00+00:00",
        active_plan_open=True,
    )
    v = rebalance_gate(state, asof, _config(cooldown_days=1))
    assert v.passed is False
    assert v.reason_code == "rebalance_plan_active"


def test_rebalance_naive_asof_normalised_to_utc():
    asof = pd.Timestamp("2026-06-23")  # naive
    state = _state(last_rebalance_asof="2026-06-21T00:00:00+00:00")
    v = rebalance_gate(state, asof, _config(cooldown_days=1))
    assert v.passed is True


def test_rebalance_unparseable_last_raises():
    asof = pd.Timestamp("2026-06-23", tz="UTC")
    state = _state(last_rebalance_asof="not-a-date")
    with pytest.raises(RiskAgentError):
        rebalance_gate(state, asof, _config())


# ---------------------------------------------------------------------------
# Ship-gate guard
# ---------------------------------------------------------------------------


def test_ship_gate_missing_file_blocks(tmp_path):
    with pytest.raises(RiskAgentError) as exc:
        RiskAgentLayer.create(
            _config(),
            tmp_path / "missing" / "SHIP_GATE.json",
            UNIVERSE_HASH,
        )
    assert "not found" in str(exc.value)


def test_ship_gate_pass_false_blocks(tmp_path):
    gate = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate, gate_pass=False)
    with pytest.raises(RiskAgentError) as exc:
        RiskAgentLayer.create(_config(), gate, UNIVERSE_HASH)
    assert "gate_pass" in str(exc.value)


def test_ship_gate_hash_mismatch_blocks(tmp_path):
    gate = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate, universe_hash=OTHER_HASH)
    with pytest.raises(RiskAgentError) as exc:
        RiskAgentLayer.create(_config(), gate, UNIVERSE_HASH)
    assert "universe_hash mismatch" in str(exc.value)


def test_ship_gate_malformed_json_blocks(tmp_path):
    gate = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate, malformed=True)
    with pytest.raises(RiskAgentError):
        RiskAgentLayer.create(_config(), gate, UNIVERSE_HASH)


def test_ship_gate_blank_hash_blocks(tmp_path):
    gate = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate, universe_hash="")
    with pytest.raises(RiskAgentError):
        RiskAgentLayer.create(_config(), gate, UNIVERSE_HASH)


def test_ship_gate_clean_construct(tmp_path):
    gate = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate)
    layer = RiskAgentLayer.create(_config(), gate, UNIVERSE_HASH)
    assert layer.config.gross_cap == 1.0
    assert layer.universe_hash == UNIVERSE_HASH


def test_create_rejects_blank_universe_hash(tmp_path):
    gate = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate)
    with pytest.raises(RiskAgentError):
        RiskAgentLayer.create(_config(), gate, "")


def test_create_rejects_wrong_config_type(tmp_path):
    gate = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate)
    with pytest.raises(RiskAgentError):
        RiskAgentLayer.create(object(), gate, UNIVERSE_HASH)  # type: ignore


# ---------------------------------------------------------------------------
# Orchestrator composite — runs all gates, drawdown every cycle
# ---------------------------------------------------------------------------


def _layer(tmp_path: Path, **config_overrides) -> RiskAgentLayer:
    gate = tmp_path / "backtests" / "SHIP_GATE.json"
    _write_ship_gate(gate)
    return RiskAgentLayer.create(_config(**config_overrides), gate, UNIVERSE_HASH)


def test_evaluate_clean_cycle_passes(tmp_path):
    layer = _layer(tmp_path)
    decision = layer.evaluate(
        asof=pd.Timestamp("2026-06-21", tz="UTC"),
        target_weights={"AAPL": 0.05, "MSFT": 0.05},
        state=_state(),
        adv={"AAPL": 1e9, "MSFT": 1e9},
    )
    assert isinstance(decision, RiskDecision)
    assert decision.block_trade is False
    assert decision.halt is False
    assert decision.degross_factor == pytest.approx(1.0)
    # Drawdown guardian MUST be the first verdict every cycle.
    assert decision.verdicts[0].name == "drawdown_guardian"
    # All six explicit gates + drawdown guardian = 7 verdicts.
    names = [v.name for v in decision.verdicts]
    for required in [
        "drawdown_guardian",
        "vol_targeter",
        "gross_exposure_cap",
        "per_name_exposure_cap",
        "concentration_check",
        "adv_participation",
        "rebalance_gate",
    ]:
        assert required in names


def test_evaluate_hard_drawdown_halts_and_blocks(tmp_path):
    layer = _layer(tmp_path)
    state = _state(nav=700_000.0, peak_nav=1_000_000.0)  # 30% DD
    decision = layer.evaluate(
        asof=pd.Timestamp("2026-06-21", tz="UTC"),
        target_weights={"AAPL": 0.05},
        state=state,
        adv={"AAPL": 1e9},
    )
    assert decision.block_trade is True
    assert decision.halt is True
    assert decision.degross_factor == 0.0


def test_evaluate_soft_drawdown_degrosses_not_halts(tmp_path):
    layer = _layer(tmp_path)
    state = _state(nav=850_000.0, peak_nav=1_000_000.0)  # 15% DD
    decision = layer.evaluate(
        asof=pd.Timestamp("2026-06-21", tz="UTC"),
        target_weights={"AAPL": 0.05, "MSFT": 0.05},
        state=state,
        adv={"AAPL": 1e9, "MSFT": 1e9},
    )
    assert decision.block_trade is False
    assert decision.halt is False
    assert 0.0 < decision.degross_factor < 1.0


def test_evaluate_drawdown_runs_even_on_other_breach(tmp_path):
    """Drawdown guardian runs EVERY cycle — even when another gate blocks."""
    layer = _layer(tmp_path)
    decision = layer.evaluate(
        asof=pd.Timestamp("2026-06-21", tz="UTC"),
        # Per-name cap = 0.10; weight = 0.50 blows it.
        target_weights={"AAPL": 0.50},
        state=_state(),
        adv={"AAPL": 1e9},
    )
    assert decision.block_trade is True
    assert decision.verdicts[0].name == "drawdown_guardian"
    assert decision.verdicts[0].passed is True  # DD itself was fine


def test_evaluate_atomic_state_write(tmp_path):
    layer = _layer(tmp_path)
    state = _state(nav=1_050_000.0, peak_nav=1_050_000.0)
    state_path = tmp_path / "state" / "risk.json"
    layer.write_state(state, state_path)
    assert state_path.exists()
    payload = json.loads(state_path.read_text())
    assert payload["nav"] == 1_050_000.0
    assert payload["peak_nav"] == 1_050_000.0
    # No half-written .tmp left behind.
    leftover = list(state_path.parent.glob(".*.tmp"))
    assert leftover == []


def test_evaluate_round_trip_state(tmp_path):
    layer = _layer(tmp_path)
    state = _state(nav=900_000.0, peak_nav=1_000_000.0,
                   recent_returns=[0.001, -0.002, 0.003])
    p = tmp_path / "state" / "risk.json"
    layer.write_state(state, p)
    loaded = PortfolioState.from_dict(json.loads(p.read_text()))
    assert loaded.nav == 900_000.0
    assert loaded.peak_nav == 1_000_000.0
    assert loaded.recent_returns == [0.001, -0.002, 0.003]


def test_portfolio_state_from_dict_validates():
    with pytest.raises(RiskAgentError):
        PortfolioState.from_dict({"peak_nav": 1.0})  # missing nav
    with pytest.raises(RiskAgentError):
        PortfolioState.from_dict(
            {"nav": 1.0, "peak_nav": 1.0, "current_weights": "bad"}
        )
    with pytest.raises(RiskAgentError):
        PortfolioState.from_dict(
            {"nav": 1.0, "peak_nav": 1.0, "recent_returns": "bad"}
        )


def test_evaluate_adv_block_surfaces_in_decision(tmp_path):
    layer = _layer(tmp_path)
    decision = layer.evaluate(
        asof=pd.Timestamp("2026-06-21", tz="UTC"),
        target_weights={"AAPL": 0.05},
        state=_state(current_weights={}),
        adv={"AAPL": 1.0},  # vanishing ADV -> participation blow-up
    )
    assert decision.block_trade is True
    assert "adv_participation_exceeded" in decision.reasons


def test_evaluate_rebalance_plan_in_flight_blocks(tmp_path):
    layer = _layer(tmp_path)
    state = _state(
        last_rebalance_asof="2026-06-20T00:00:00+00:00",
        active_plan_open=True,
    )
    decision = layer.evaluate(
        asof=pd.Timestamp("2026-06-21", tz="UTC"),
        target_weights={"AAPL": 0.05},
        state=state,
        adv={"AAPL": 1e9},
    )
    assert decision.block_trade is True
    assert "rebalance_plan_active" in decision.reasons
