"""No-mock tests for the US-005 harvester ship-gate runner.

Project rules in effect: real ``UniverseSnapshot``, real ``run_portfolio_backtest``,
real ``evaluate_gate`` — no ``unittest.mock`` / ``MagicMock`` / ``patch``. All
disk writes go through ``tmp_path``. Synthetic 12-year daily panels exercise
both PASS and FAIL paths without depending on yfinance or live data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.equity.ship_gate import (
    CRITERIA_SOURCE,
    DEFAULT_DD_HARD,
    HarvesterShipGateReport,
    ShipGateError,
    build_equal_weight_panel,
    evaluate_harvester,
    run_and_persist,
    write_backtest_artifact,
    write_ship_gate,
)
from src.equity.universe import (
    UniverseRules,
    UniverseSnapshot,
    compute_universe_hash,
)


def _utc_bday_index(start: str, n: int) -> pd.DatetimeIndex:
    idx = pd.bdate_range(start=start, periods=n)
    return pd.DatetimeIndex(idx).tz_localize("UTC").normalize()


def _make_membership(idx: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(True, index=idx, columns=tickers)


def _make_snapshot(idx: pd.DatetimeIndex, tickers: list[str]) -> UniverseSnapshot:
    members = _make_membership(idx, tickers)
    h = compute_universe_hash(members)
    return UniverseSnapshot(
        membership=members,
        rules=UniverseRules(top_n=len(tickers)),
        built_at="2026-06-19T00:00:00+00:00",
        pipeline_version="test-2026-06-19",
        universe_hash=h,
        reconstitution_dates=(idx[0].date().isoformat(),),
    )


def _prices_from_drift(
    idx: pd.DatetimeIndex,
    tickers: list[str],
    *,
    mu_annual: float,
    sigma_annual: float,
    seed: int,
) -> pd.DataFrame:
    """Deterministic synthetic GBM-style price panel.

    Returns are i.i.d. normal with annualised drift ``mu_annual`` and vol
    ``sigma_annual``; identical seeds + identical mu/sigma reproduce the same
    panel bit-for-bit. Used to drive the harvester to known-PASS / known-FAIL
    regimes for the gate.
    """
    rng = np.random.default_rng(seed)
    n = len(idx)
    daily_mu = mu_annual / 252.0
    daily_sigma = sigma_annual / np.sqrt(252.0)
    arr = np.zeros((n, len(tickers)))
    arr[0] = 100.0
    for j in range(len(tickers)):
        rets = rng.normal(daily_mu, daily_sigma, n - 1)
        arr[1:, j] = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame(arr, index=idx, columns=tickers)


def test_build_equal_weight_panel_renormalises_unpriced_members():
    idx = _utc_bday_index("2025-01-02", 5)
    tickers = ["A", "B", "C"]
    members = _make_membership(idx, tickers)
    snapshot = _make_snapshot(idx, tickers)

    prices = pd.DataFrame(
        [[100.0, 100.0, np.nan],
         [101.0, 100.5, np.nan],
         [102.0, 101.0, np.nan],
         [101.5, 101.5, 200.0],
         [102.5, 102.0, 201.0]],
        index=idx,
        columns=tickers,
    )

    w = build_equal_weight_panel(snapshot, prices)
    # On rows 0..2, C has no price → A,B carry 0.5 each, C is 0.
    np.testing.assert_allclose(w.iloc[0].tolist(), [0.5, 0.5, 0.0])
    np.testing.assert_allclose(w.iloc[1].tolist(), [0.5, 0.5, 0.0])
    # On rows 3..4, all three priced → 1/3 each.
    np.testing.assert_allclose(w.iloc[3].tolist(), [1 / 3, 1 / 3, 1 / 3], atol=1e-12)
    # Every row sums to 1 (or 0 if no one is priced).
    np.testing.assert_allclose(w.sum(axis=1).tolist(), [1.0, 1.0, 1.0, 1.0, 1.0])
    # Members object on snapshot was not mutated.
    assert members.equals(snapshot.membership)


def test_evaluate_harvester_passes_on_calm_long_run():
    """A 12-year low-vol drifting panel clears every criterion in the bar."""
    idx = _utc_bday_index("2010-01-04", 252 * 12 + 10)
    tickers = ["AA", "BB", "CC", "DD", "EE"]
    snapshot = _make_snapshot(idx, tickers)
    # Moderate positive drift, modest vol → managed Sharpe well above 0.4
    # and drawdowns bounded by the 20% hard floor of the overlay.
    prices = _prices_from_drift(
        idx, tickers, mu_annual=0.10, sigma_annual=0.12, seed=42
    )

    report = evaluate_harvester(
        snapshot,
        prices,
        cost_bps=2.0,
        slippage_bps_per_pct_adv=0.0,
        adv=None,
    )

    assert isinstance(report, HarvesterShipGateReport)
    assert report.gate_pass is True
    assert report.universe_hash == snapshot.universe_hash
    assert report.criteria["walk_forward"]["passed"] is True
    assert report.criteria["history_length"]["passed"] is True
    assert report.criteria["max_drawdown"]["passed"] is True
    assert report.total_years >= 10
    assert "PASS" in report.recommendation
    # Overlay cannot exceed its hard floor by construction.
    assert report.max_dd <= DEFAULT_DD_HARD + 1e-6


def test_evaluate_harvester_fails_on_short_history():
    """A 2-year panel cannot clear MIN_TOTAL_YEARS=10 — gate must fail."""
    idx = _utc_bday_index("2024-01-02", 252 * 2)
    tickers = ["AA", "BB"]
    snapshot = _make_snapshot(idx, tickers)
    prices = _prices_from_drift(
        idx, tickers, mu_annual=0.10, sigma_annual=0.12, seed=7
    )

    report = evaluate_harvester(snapshot, prices, cost_bps=2.0)

    assert report.gate_pass is False
    assert report.criteria["history_length"]["passed"] is False
    assert "FAIL" in report.recommendation
    assert "EW-sector" in report.recommendation


def test_evaluate_harvester_with_adv_charges_slippage(tmp_path):
    """Providing an ADV panel makes slippage non-zero and lowers net Sharpe."""
    idx = _utc_bday_index("2010-01-04", 252 * 12)
    tickers = ["AA", "BB", "CC"]
    snapshot = _make_snapshot(idx, tickers)
    prices = _prices_from_drift(
        idx, tickers, mu_annual=0.08, sigma_annual=0.18, seed=11
    )

    # Free
    report_free = evaluate_harvester(
        snapshot, prices, cost_bps=0.0, slippage_bps_per_pct_adv=0.0, adv=None
    )

    # 5% ADV-of-NAV uniformly; 5bps per 1pp ADV → ~25 bps per trade-side.
    adv = pd.DataFrame(0.05, index=idx, columns=tickers)
    report_costly = evaluate_harvester(
        snapshot,
        prices,
        cost_bps=2.0,
        slippage_bps_per_pct_adv=5.0,
        adv=adv,
    )

    assert report_costly.net_sharpe <= report_free.net_sharpe + 1e-9


def test_write_ship_gate_atomic_and_payload_shape(tmp_path):
    idx = _utc_bday_index("2010-01-04", 252 * 12 + 5)
    tickers = ["AA", "BB", "CC"]
    snapshot = _make_snapshot(idx, tickers)
    prices = _prices_from_drift(
        idx, tickers, mu_annual=0.10, sigma_annual=0.12, seed=99
    )
    report = evaluate_harvester(snapshot, prices)

    out = tmp_path / "SHIP_GATE.json"
    written = write_ship_gate(report, out)
    assert written == out
    assert out.exists()
    # No leftover .tmp sibling.
    leftovers = list(tmp_path.glob("SHIP_GATE.json.tmp"))
    assert leftovers == []

    payload = json.loads(out.read_text())
    for required in (
        "gate_pass",
        "net_sharpe",
        "max_dd",
        "positive_years",
        "universe_hash",
        "asof",
    ):
        assert required in payload, f"SHIP_GATE missing {required!r}"
    assert payload["universe_hash"] == snapshot.universe_hash
    assert payload["criteria_source"] == CRITERIA_SOURCE
    assert isinstance(payload["thresholds"], dict)
    assert payload["thresholds"]["min_total_years"] == 10
    assert payload["thresholds"]["min_positive_years"] == 6


def test_write_backtest_artifact_is_timestamped_and_atomic(tmp_path):
    idx = _utc_bday_index("2010-01-04", 252 * 12)
    tickers = ["AA", "BB"]
    snapshot = _make_snapshot(idx, tickers)
    prices = _prices_from_drift(
        idx, tickers, mu_annual=0.08, sigma_annual=0.15, seed=1
    )
    report = evaluate_harvester(snapshot, prices)

    path = write_backtest_artifact(
        report, tmp_path, timestamp="20260619T120000Z"
    )
    assert path.name == "equity_harvester_singlestock_20260619T120000Z.json"
    assert path.exists()
    assert list(tmp_path.glob("*.tmp")) == []

    payload = json.loads(path.read_text())
    assert "summary" in payload
    assert "criteria" in payload
    assert "backtest_summary" in payload
    assert payload["overlay_params"]["target_vol"] == report.overlay_params["target_vol"]


def test_run_and_persist_writes_both_artifacts_and_ship_gate_name_is_fixed(tmp_path):
    idx = _utc_bday_index("2010-01-04", 252 * 12)
    tickers = ["AA", "BB", "CC"]
    snapshot = _make_snapshot(idx, tickers)
    prices = _prices_from_drift(
        idx, tickers, mu_annual=0.10, sigma_annual=0.12, seed=2026
    )

    report, artifact_path, ship_gate_path = run_and_persist(
        snapshot,
        prices,
        tmp_path,
        timestamp="20260619T134500Z",
        pipeline_version="test-vX",
    )

    assert artifact_path.name == "equity_harvester_singlestock_20260619T134500Z.json"
    assert ship_gate_path.name == "SHIP_GATE.json"
    assert artifact_path.exists()
    assert ship_gate_path.exists()
    payload = json.loads(ship_gate_path.read_text())
    assert payload["pipeline_version"] == "test-vX"
    # Downstream guard: if gate_pass is true here, universe_hash matches.
    assert payload["universe_hash"] == snapshot.universe_hash
    assert payload["gate_pass"] is report.gate_pass


def test_evaluate_harvester_raises_on_empty_prices():
    idx = _utc_bday_index("2025-01-02", 5)
    snapshot = _make_snapshot(idx, ["AA"])
    with pytest.raises(ShipGateError, match="empty"):
        evaluate_harvester(snapshot, pd.DataFrame())


def test_evaluate_harvester_raises_on_misaligned_dates():
    """Membership in 2010 + prices in 2030 → zero overlap → fail loud."""
    members_idx = _utc_bday_index("2010-01-04", 5)
    prices_idx = _utc_bday_index("2030-01-02", 5)
    snapshot = _make_snapshot(members_idx, ["AA"])
    prices = _prices_from_drift(
        prices_idx, ["AA"], mu_annual=0.0, sigma_annual=0.1, seed=0
    )
    with pytest.raises(ShipGateError, match="overlap"):
        evaluate_harvester(snapshot, prices)


def test_evaluate_harvester_rejects_invalid_execution_lag():
    idx = _utc_bday_index("2010-01-04", 252 * 12)
    snapshot = _make_snapshot(idx, ["AA"])
    prices = _prices_from_drift(
        idx, ["AA"], mu_annual=0.05, sigma_annual=0.15, seed=3
    )
    with pytest.raises(ShipGateError, match="execution_lag"):
        evaluate_harvester(snapshot, prices, execution_lag=0)


# ---------------------------------------------------------------------------
# Universe-scope self-description (2026-07-20). A headline net_sharpe is only
# interpretable against the universe it was measured on: the live artifact's
# curated 20-name PASS (0.906) becomes 0.740 full / 0.355 OOS -- a gate FAIL --
# once the wide universe is survivorship-corrected (2026-07-01 independent
# audit, docs/training-architecture-audit-2026-07-03.md:132-134). The payload
# must carry scope so a regenerated artifact cannot silently drop it.
# ---------------------------------------------------------------------------


def test_ship_gate_payload_states_universe_scope_and_carries_caveats(tmp_path):
    idx = _utc_bday_index("2010-01-04", 252 * 12 + 5)
    tickers = ["AA", "BB", "CC"]
    snapshot = _make_snapshot(idx, tickers)
    prices = _prices_from_drift(
        idx, tickers, mu_annual=0.10, sigma_annual=0.12, seed=99
    )
    report = evaluate_harvester(snapshot, prices)

    payload = json.loads(write_ship_gate(report, tmp_path / "SHIP_GATE.json").read_text())

    assert "universe_scope" in payload, "SHIP_GATE must state the universe it measured"
    # Derived from the snapshot itself, so it cannot drift from what ran.
    assert "3 distinct names" in payload["universe_scope"]
    assert str(len(idx)) in payload["universe_scope"]
    assert "caveats" in payload
    assert isinstance(payload["caveats"], list)


def test_universe_scope_is_derived_not_defaulted(tmp_path):
    """Scope must reflect the real universe width, not a placeholder."""
    idx = _utc_bday_index("2010-01-04", 252 * 12 + 5)
    wide = ["AA", "BB", "CC", "DD", "EE", "FF"]
    snapshot = _make_snapshot(idx, wide)
    prices = _prices_from_drift(idx, wide, mu_annual=0.10, sigma_annual=0.12, seed=7)

    report = evaluate_harvester(snapshot, prices)

    assert report.universe_scope != "unspecified"
    assert "6 distinct names" in report.universe_scope


def test_live_ship_gate_artifact_reports_both_universe_numbers():
    """The committed artifact must never show the flattering number alone.

    Pins the 2026-07-01 audit directive ("always report both") against the
    real on-disk file the arming path reads.
    """
    path = Path(__file__).resolve().parents[1] / "trained_data" / "backtests" / "SHIP_GATE.json"
    if not path.exists():
        pytest.skip("SHIP_GATE.json not present in this checkout")
    payload = json.loads(path.read_text())

    # arm() contract fields must still be intact.
    assert payload["gate_pass"] is True
    assert payload["net_sharpe"] == 0.906

    # ...and the failing wide-universe counterpart must travel with it.
    audit = payload["independent_audit"]
    assert audit["wide_universe_net_sharpe_oos"] == 0.355
    assert audit["wide_universe_gate_verdict"] == "FAIL"
    assert audit["wide_universe_net_sharpe_oos"] < payload["thresholds"]["min_net_sharpe"]
    assert any("SURVIVORSHIP" in c.upper() for c in payload["caveats"])
