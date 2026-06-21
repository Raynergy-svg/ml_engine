"""US-006 — Tests for :class:`HarvesterStrategy`.

No mocks. Real :class:`UniverseSnapshot`, real :class:`ScannerConfig`, real
on-disk ship-gate JSON via ``tmp_path``. Every test exercises the same code
path the runtime control loop will execute.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.equity.harvester_strategy import (
    HarvesterStrategy,
    HarvesterStrategyError,
)
from src.equity.strategy import (
    EquityStrategy,
    PER_NAME_ABS_CAP,
    StrategyError,
)
from src.equity.universe import UniverseRules, UniverseSnapshot
from src.factor.portfolio import GROSS_CAP
from src.scanner.config import ScannerConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def _make_panels(
    seed: int = 7,
    n_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build matching membership + price panels for a 4-name universe."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-01", periods=n_days, tz="UTC").normalize()

    # All names are members on every date — keeps the test deterministic.
    membership = pd.DataFrame(
        np.ones((n_days, len(TICKERS)), dtype=bool),
        index=dates,
        columns=TICKERS,
    )

    # Random-walk prices, no NaN, finite.
    rets = rng.normal(0.0, 0.008, size=(n_days, len(TICKERS)))
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0),
        index=dates,
        columns=TICKERS,
    )
    return membership, prices


def _make_snapshot(
    membership: pd.DataFrame,
    universe_hash: str = "deadbeef" * 8,
) -> UniverseSnapshot:
    return UniverseSnapshot(
        membership=membership,
        rules=UniverseRules(),
        built_at="2026-06-19T00:00:00Z",
        pipeline_version="2026-06-18-eq1",
        universe_hash=universe_hash,
    )


def _write_ship_gate(
    path: Path,
    *,
    gate_pass: bool = True,
    universe_hash: str = "deadbeef" * 8,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _build(tmp_path: Path, **overrides) -> HarvesterStrategy:
    """Build a strategy + ship gate against a fresh synthetic universe."""
    membership, prices = _make_panels()
    snapshot = _make_snapshot(membership)
    gate_path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate_path, universe_hash=snapshot.universe_hash)
    return HarvesterStrategy.create(
        snapshot=snapshot,
        prices=prices,
        ship_gate_path=gate_path,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


def test_harvester_satisfies_equity_strategy_protocol(tmp_path: Path):
    strategy = _build(tmp_path)
    assert isinstance(strategy, EquityStrategy)


# ---------------------------------------------------------------------------
# Determinism + Claude-free
# ---------------------------------------------------------------------------


def test_compute_target_weights_is_deterministic(tmp_path: Path):
    """Same inputs → same dict, every time."""
    strategy = _build(tmp_path)
    asof = strategy.snapshot.membership.index[-1]
    first = strategy.compute_target_weights(asof)
    second = strategy.compute_target_weights(asof)
    third = strategy.compute_target_weights(asof)
    assert first == second == third
    # Hash check — no float drift round-tripped.
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_compute_target_weights_returns_long_only_book(tmp_path: Path):
    strategy = _build(tmp_path)
    asof = strategy.snapshot.membership.index[-1]
    weights = strategy.compute_target_weights(asof)
    assert all(w >= 0.0 for w in weights.values())


def test_long_only_default_is_true(tmp_path: Path):
    strategy = _build(tmp_path)
    assert strategy.long_only is True


def test_gross_within_cap(tmp_path: Path):
    strategy = _build(tmp_path)
    asof = strategy.snapshot.membership.index[-1]
    weights = strategy.compute_target_weights(asof)
    gross = sum(abs(w) for w in weights.values())
    assert gross <= GROSS_CAP + 1e-9
    assert all(abs(w) <= PER_NAME_ABS_CAP + 1e-9 for w in weights.values())


# ---------------------------------------------------------------------------
# Causality — perturbing future rows must not change past output
# ---------------------------------------------------------------------------


def test_compute_is_causal_under_future_price_perturbation(tmp_path: Path):
    """Slicing to ``loc[:asof]`` means a future price change is invisible."""
    membership, prices = _make_panels()
    snapshot = _make_snapshot(membership)
    gate_path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate_path, universe_hash=snapshot.universe_hash)

    asof = membership.index[30]  # interior point
    strat_a = HarvesterStrategy.create(snapshot, prices, gate_path)
    w_a = strat_a.compute_target_weights(asof)

    # Mutate everything strictly after asof — should not affect w_a's result.
    perturbed = prices.copy()
    perturbed.iloc[50:] = perturbed.iloc[50:] * 5.0
    strat_b = HarvesterStrategy.create(snapshot, perturbed, gate_path)
    w_b = strat_b.compute_target_weights(asof)

    assert w_a == w_b


# ---------------------------------------------------------------------------
# Overlay scalar actually scales — drawdown circuit-breaker
# ---------------------------------------------------------------------------


def test_overlay_scales_exposure_below_unit(tmp_path: Path):
    """A drawdown beyond ``dd_hard`` collapses the book to zero."""
    membership, _prices = _make_panels()
    # Construct a price series that crashes 30% over the window.
    crash_dates = membership.index
    n = len(crash_dates)
    base = np.linspace(100.0, 70.0, n)
    crash_prices = pd.DataFrame(
        np.tile(base[:, None], (1, len(TICKERS))),
        index=crash_dates,
        columns=TICKERS,
    )
    snapshot = _make_snapshot(membership)
    gate_path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate_path, universe_hash=snapshot.universe_hash)

    strategy = HarvesterStrategy.create(
        snapshot, crash_prices, gate_path,
        dd_soft=0.05, dd_hard=0.10, max_lev=1.0,
    )
    weights = strategy.compute_target_weights(membership.index[-1])
    # Past hard drawdown → scalar=0 → book is flat.
    assert weights == {}


# ---------------------------------------------------------------------------
# Ship gate guard — fail-closed
# ---------------------------------------------------------------------------


def test_ship_gate_missing_file_blocks_construction(tmp_path: Path):
    membership, prices = _make_panels()
    snapshot = _make_snapshot(membership)
    missing = tmp_path / "does_not_exist" / "SHIP_GATE.json"
    with pytest.raises(HarvesterStrategyError, match="not found"):
        HarvesterStrategy.create(snapshot, prices, missing)


def test_ship_gate_false_blocks_construction(tmp_path: Path):
    membership, prices = _make_panels()
    snapshot = _make_snapshot(membership)
    gate_path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(
        gate_path, gate_pass=False, universe_hash=snapshot.universe_hash
    )
    with pytest.raises(HarvesterStrategyError, match="gate_pass"):
        HarvesterStrategy.create(snapshot, prices, gate_path)


def test_ship_gate_universe_hash_mismatch_blocks(tmp_path: Path):
    membership, prices = _make_panels()
    snapshot = _make_snapshot(membership, universe_hash="aaaa" * 16)
    gate_path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate_path, universe_hash="bbbb" * 16)
    with pytest.raises(HarvesterStrategyError, match="universe_hash mismatch"):
        HarvesterStrategy.create(snapshot, prices, gate_path)


def test_ship_gate_malformed_json_blocks(tmp_path: Path):
    membership, prices = _make_panels()
    snapshot = _make_snapshot(membership)
    gate_path = tmp_path / "SHIP_GATE.json"
    gate_path.write_text("{not valid json")
    with pytest.raises(HarvesterStrategyError, match="cannot read"):
        HarvesterStrategy.create(snapshot, prices, gate_path)


def test_ship_gate_missing_hash_blocks(tmp_path: Path):
    membership, prices = _make_panels()
    snapshot = _make_snapshot(membership)
    gate_path = tmp_path / "SHIP_GATE.json"
    gate_path.write_text(json.dumps({"gate_pass": True}))
    with pytest.raises(HarvesterStrategyError, match="universe_hash"):
        HarvesterStrategy.create(snapshot, prices, gate_path)


def test_ship_gate_non_object_payload_blocks(tmp_path: Path):
    membership, prices = _make_panels()
    snapshot = _make_snapshot(membership)
    gate_path = tmp_path / "SHIP_GATE.json"
    gate_path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(HarvesterStrategyError, match="not a JSON object"):
        HarvesterStrategy.create(snapshot, prices, gate_path)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_membership_rejected(tmp_path: Path):
    empty = pd.DataFrame(
        columns=TICKERS,
        index=pd.DatetimeIndex([], tz="UTC", name="date"),
    )
    snapshot = _make_snapshot(empty)
    _, prices = _make_panels()
    gate_path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate_path, universe_hash=snapshot.universe_hash)
    with pytest.raises(HarvesterStrategyError, match="empty membership"):
        HarvesterStrategy.create(snapshot, prices, gate_path)


def test_bad_dd_ordering_rejected(tmp_path: Path):
    with pytest.raises(HarvesterStrategyError, match="dd_soft"):
        _build(tmp_path, dd_soft=0.20, dd_hard=0.10)


def test_negative_target_vol_rejected(tmp_path: Path):
    with pytest.raises(HarvesterStrategyError, match="target_vol"):
        _build(tmp_path, target_vol=-0.05)


def test_compute_rejects_unknown_asof(tmp_path: Path):
    strategy = _build(tmp_path)
    with pytest.raises(HarvesterStrategyError, match="not present"):
        strategy.compute_target_weights(pd.Timestamp("1999-01-04"))


# ---------------------------------------------------------------------------
# from_config — 3-layer pattern (dataclass → profile dict → consumer getattr)
# ---------------------------------------------------------------------------


def test_from_config_reads_scanner_config_defaults(tmp_path: Path):
    """`from_config` reads the dataclass defaults via getattr."""
    membership, prices = _make_panels()
    snapshot = _make_snapshot(membership)
    gate_path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate_path, universe_hash=snapshot.universe_hash)

    cfg = ScannerConfig()
    strategy = HarvesterStrategy.from_config(
        cfg, snapshot, prices, ship_gate_path=gate_path
    )
    assert strategy.target_vol == pytest.approx(cfg.equity_harvester_target_vol)
    assert strategy.dd_soft == pytest.approx(cfg.equity_harvester_dd_soft)
    assert strategy.dd_hard == pytest.approx(cfg.equity_harvester_dd_hard)
    assert strategy.max_lev == pytest.approx(cfg.equity_harvester_max_lev)
    assert strategy.vol_lookback == cfg.equity_harvester_vol_lookback
    assert strategy.long_only is bool(cfg.equity_harvester_long_only)


def test_from_config_respects_profile_overrides(tmp_path: Path):
    """Apply ``balanced`` profile → overlay knobs propagate through to the
    consumer. This exercises the 3-layer pattern end to end:

    1. Dataclass default exists for ``equity_harvester_target_vol``.
    2. ``apply_profile('balanced')`` writes the profile-dict value to the
       config attribute.
    3. ``from_config`` reads it back via ``getattr``.
    """
    membership, prices = _make_panels()
    snapshot = _make_snapshot(membership)
    gate_path = tmp_path / "SHIP_GATE.json"
    _write_ship_gate(gate_path, universe_hash=snapshot.universe_hash)

    cfg = ScannerConfig()
    cfg.apply_profile("balanced")
    # Mutate post-profile to prove the value is read by `from_config`.
    cfg.equity_harvester_target_vol = 0.07
    cfg.equity_harvester_dd_soft = 0.04
    cfg.equity_harvester_dd_hard = 0.09
    cfg.equity_harvester_max_lev = 0.5
    cfg.equity_harvester_long_only = True

    strategy = HarvesterStrategy.from_config(
        cfg, snapshot, prices, ship_gate_path=gate_path
    )
    assert strategy.target_vol == pytest.approx(0.07)
    assert strategy.dd_soft == pytest.approx(0.04)
    assert strategy.dd_hard == pytest.approx(0.09)
    assert strategy.max_lev == pytest.approx(0.5)


def test_scanner_config_has_required_harvester_fields():
    """Layer 1: dataclass fields must exist (orphan keys are silently
    dropped by apply_profile, so this guards against typo drift)."""
    fields = ScannerConfig.__dataclass_fields__
    for name in (
        "equity_harvester_target_vol",
        "equity_harvester_dd_soft",
        "equity_harvester_dd_hard",
        "equity_harvester_max_lev",
        "equity_harvester_vol_lookback",
        "equity_harvester_long_only",
        "equity_harvester_ship_gate_path",
    ):
        assert name in fields, f"missing dataclass field {name}"


def test_balanced_profile_dict_carries_harvester_overrides():
    """Layer 2: profile dict carries entries that match dataclass field names."""
    from src.scanner.config import SCAN_PROFILES

    overrides = SCAN_PROFILES["balanced"]
    for name in (
        "equity_harvester_target_vol",
        "equity_harvester_dd_soft",
        "equity_harvester_dd_hard",
        "equity_harvester_max_lev",
        "equity_harvester_vol_lookback",
        "equity_harvester_long_only",
        "equity_harvester_ship_gate_path",
    ):
        assert name in overrides, f"missing profile override {name}"


# ---------------------------------------------------------------------------
# Atomic state writes
# ---------------------------------------------------------------------------


def test_write_state_is_atomic(tmp_path: Path):
    """A successful write leaves no leftover tmp siblings."""
    strategy = _build(tmp_path)
    asof = strategy.snapshot.membership.index[-1]
    weights = strategy.compute_target_weights(asof)

    state_path = tmp_path / "state" / "equity_target.json"
    strategy.write_state(weights, asof, state_path)

    assert state_path.exists()
    payload = json.loads(state_path.read_text())
    assert payload["weights"] == weights
    assert payload["n_names"] == len(weights)
    assert payload["long_only"] is True
    assert payload["universe_hash"] == strategy.universe_hash

    sibling_tmps = [
        p for p in state_path.parent.iterdir()
        if p.name.startswith(f".{state_path.name}") and p.name.endswith(".tmp")
    ]
    assert sibling_tmps == []


# ---------------------------------------------------------------------------
# Sanity — when scaled, validate_weights still accepts the book
# ---------------------------------------------------------------------------


def test_emitted_weights_pass_validate_weights(tmp_path: Path):
    """The seam contract: emitted weights MUST clear validate_weights."""
    strategy = _build(tmp_path)
    asof = strategy.snapshot.membership.index[-1]
    weights = strategy.compute_target_weights(asof)
    # No raise → validate_weights accepted the book.
    try:
        from src.equity.strategy import validate_weights
        validate_weights(weights, long_only=True)
    except StrategyError as exc:
        pytest.fail(f"emitted book failed seam validation: {exc}")
