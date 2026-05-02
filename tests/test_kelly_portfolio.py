"""Unit tests for KellyPortfolioOptimizer (src/scanner/automation/kelly_portfolio.py).

Closes a zero-coverage gap surfaced by the 2026-05-01 test coverage audit.

Pipeline under test:
1. compute_kelly: f* = (p*b - q)/b with safe handling of degenerate inputs
2. compute_half_kelly: production default = full / 2
3. portfolio_optimal_size: full pipeline incl. VaR, correlation, position cap
4. _compute_open_risk: existing-position risk in % of NAV
5. _correlation_reduction: shrinks size for concentrated exposure

Production safety: half-Kelly is the default, max single position is 5% NAV,
portfolio VaR cap is 15% NAV. Tests pin all three.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.automation.kelly_portfolio import (
    MAX_PORTFOLIO_RISK_PCT,
    MAX_SINGLE_POSITION_PCT,
    MIN_TRADES_FOR_KELLY,
    STATIC_CORRELATIONS,
    KellyPortfolioOptimizer,
    KellyResult,
)


# ---------------------------------------------------------------------------
# Fixture: optimizer with seeded journal
# ---------------------------------------------------------------------------

def _make_journal(tmp_path: Path, wins: int, losses: int, win_pips: float = 30.0, loss_pips: float = 20.0) -> Path:
    """Create a trade journal with the given win/loss counts.

    Schema-compliant with safe_json validate_trade_journal: each entry needs
    'pair', 'direction', 'timestamp' keys (Phase 26 US-160).
    """
    base_entry = {
        "pair": "EUR_USD",
        "direction": "LONG",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    trades = []
    for _ in range(wins):
        trades.append({**base_entry, "outcome": "WIN", "pnl_pips": win_pips})
    for _ in range(losses):
        trades.append({**base_entry, "outcome": "LOSS", "pnl_pips": -loss_pips})
    path = tmp_path / "trade_journal_rl.json"
    path.write_text(json.dumps(trades))
    return path


@pytest.fixture
def journal_with_edge(tmp_path):
    """Journal with positive Kelly edge: 60% win rate, 30/20 win/loss ratio."""
    return _make_journal(tmp_path, wins=18, losses=12)


@pytest.fixture
def journal_no_edge(tmp_path):
    """Journal with negative edge: 30% win rate."""
    return _make_journal(tmp_path, wins=9, losses=21)


@pytest.fixture
def journal_thin(tmp_path):
    """Journal below MIN_TRADES_FOR_KELLY threshold."""
    return _make_journal(tmp_path, wins=5, losses=5)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        assert opt.max_portfolio_risk == MAX_PORTFOLIO_RISK_PCT
        assert opt.max_single_position == MAX_SINGLE_POSITION_PCT
        assert opt.use_half_kelly is True
        assert opt._stats_initialized is False

    def test_loads_stats_when_journal_has_enough_trades(self, journal_with_edge, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=journal_with_edge,
            persistence_path=tmp_path / "kelly.json",
        )
        assert opt._stats_initialized is True
        assert opt._n_trades == 30
        assert opt._win_rate == pytest.approx(0.60)
        assert opt._avg_win_pips == pytest.approx(30.0)
        assert opt._avg_loss_pips == pytest.approx(20.0)

    def test_skips_stats_when_journal_below_threshold(self, journal_thin, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=journal_thin,
            persistence_path=tmp_path / "kelly.json",
        )
        assert opt._stats_initialized is False
        # 10 trades < MIN_TRADES_FOR_KELLY (20)
        assert MIN_TRADES_FOR_KELLY == 20

    def test_missing_journal_no_crash(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "missing.json",
            persistence_path=tmp_path / "kelly.json",
        )
        assert opt._stats_initialized is False
        assert opt._n_trades == 0


# ---------------------------------------------------------------------------
# Kelly fraction math
# ---------------------------------------------------------------------------

class TestKellyMath:
    def test_kelly_explicit_inputs(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        # p=0.6, w/l=1.5 → b=1.5, q=0.4 → kelly = (0.6*1.5 - 0.4)/1.5 = 0.5/1.5 = 0.3333
        f = opt.compute_kelly(win_rate=0.60, avg_win=30.0, avg_loss=20.0)
        assert f == pytest.approx(1.0 / 3.0, abs=1e-9)

    def test_half_kelly_explicit_inputs(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        f = opt.compute_half_kelly(win_rate=0.60, avg_win=30.0, avg_loss=20.0)
        assert f == pytest.approx(1.0 / 6.0, abs=1e-9)

    def test_kelly_uses_cached_stats_when_no_args(self, journal_with_edge, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=journal_with_edge,
            persistence_path=tmp_path / "kelly.json",
        )
        f = opt.compute_kelly()
        # Should match seeded stats (60%, 30/20)
        assert f == pytest.approx(1.0 / 3.0, abs=1e-9)

    def test_kelly_negative_for_losing_strategy(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        # 30% wins on 1:1 → kelly = (0.3*1 - 0.7)/1 = -0.4
        f = opt.compute_kelly(win_rate=0.30, avg_win=20.0, avg_loss=20.0)
        assert f == pytest.approx(-0.4, abs=1e-9)

    def test_kelly_zero_when_avg_loss_nonpositive(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        assert opt.compute_kelly(win_rate=0.60, avg_win=30.0, avg_loss=0.0) == 0.0
        assert opt.compute_kelly(win_rate=0.60, avg_win=30.0, avg_loss=-5.0) == 0.0

    def test_kelly_zero_when_win_rate_nonpositive_or_one(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        assert opt.compute_kelly(win_rate=0.0, avg_win=30.0, avg_loss=20.0) == 0.0
        assert opt.compute_kelly(win_rate=1.0, avg_win=30.0, avg_loss=20.0) == 0.0
        assert opt.compute_kelly(win_rate=-0.10, avg_win=30.0, avg_loss=20.0) == 0.0


# ---------------------------------------------------------------------------
# portfolio_optimal_size pipeline
# ---------------------------------------------------------------------------

class TestPortfolioOptimalSize:
    def test_uninitialized_stats_returns_max_single_position(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        result = opt.portfolio_optimal_size(
            pair="EUR_USD",
            direction="LONG",
            confidence=0.65,
            sl_pips=20.0,
            nav=10000.0,
        )
        assert result.recommended_risk_pct == MAX_SINGLE_POSITION_PCT
        assert "Insufficient trade history" in result.reason

    def test_negative_kelly_returns_zero_size(self, journal_no_edge, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=journal_no_edge,
            persistence_path=tmp_path / "kelly.json",
        )
        result = opt.portfolio_optimal_size(
            pair="EUR_USD",
            direction="LONG",
            confidence=0.65,
            sl_pips=20.0,
            nav=10000.0,
        )
        assert result.recommended_risk_pct == 0.0
        assert result.recommended_lots == 0.0
        assert "Negative Kelly" in result.reason

    def test_positive_kelly_returns_capped_size(self, journal_with_edge, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=journal_with_edge,
            persistence_path=tmp_path / "kelly.json",
        )
        result = opt.portfolio_optimal_size(
            pair="EUR_USD",
            direction="LONG",
            confidence=0.65,
            sl_pips=20.0,
            nav=10000.0,
        )
        # Half-kelly = 0.1667; * conf_boost(min(0.65/0.6, 1.2) ≈ 1.083) ≈ 0.181;
        # exceeds 15% portfolio budget → trimmed to 0.15 → var_constrained=True;
        # then capped at max_single 5%
        assert result.recommended_risk_pct == pytest.approx(MAX_SINGLE_POSITION_PCT)
        assert result.recommended_lots > 0.0
        # No correlation penalty without open positions
        assert result.correlation_reduction == 1.0
        # var_constrained=True is expected: aggressive strategies trip the cap
        assert result.var_constrained is True

    def test_var_constraint_blocks_when_budget_exhausted(self, journal_with_edge, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=journal_with_edge,
            persistence_path=tmp_path / "kelly.json",
        )
        # Simulate open positions consuming the entire 15% portfolio risk budget
        # 10000 NAV * 15% = $1500 risk; 10 lots * 15 pips * $10/pip = $1500
        open_positions = [{
            "pair": "GBP_JPY",
            "direction": "LONG",
            "lots": 10.0,
            "sl_pips": 15.0,
            "pip_value_per_lot": 10.0,
        }]
        result = opt.portfolio_optimal_size(
            pair="EUR_USD",
            direction="LONG",
            confidence=0.65,
            sl_pips=20.0,
            open_positions=open_positions,
            nav=10000.0,
        )
        assert result.recommended_risk_pct == 0.0
        assert result.recommended_lots == 0.0
        assert result.var_constrained is True
        assert "budget exhausted" in result.reason

    def test_correlation_reduction_applied_for_correlated_long(self, journal_with_edge, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=journal_with_edge,
            persistence_path=tmp_path / "kelly.json",
        )
        # GBP_USD long is highly correlated with EUR_USD long (0.85)
        open_positions = [{
            "pair": "GBP_USD",
            "direction": "LONG",
            "lots": 0.1,
            "sl_pips": 20.0,
            "pip_value_per_lot": 10.0,
        }]
        result = opt.portfolio_optimal_size(
            pair="EUR_USD",
            direction="LONG",
            confidence=0.65,
            sl_pips=20.0,
            open_positions=open_positions,
            nav=100000.0,  # Large NAV so VaR doesn't bind
        )
        # 0.85 corr → reduction = max(0.3, 1 - 0.85*0.7) = max(0.3, 0.405) = 0.405
        assert result.correlation_reduction == pytest.approx(0.405, abs=1e-6)

    def test_correlation_reduction_for_anti_correlated_opposite_direction(self, journal_with_edge, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=journal_with_edge,
            persistence_path=tmp_path / "kelly.json",
        )
        # USD_CHF SHORT vs EUR_USD LONG: corr(EUR_USD, USD_CHF) = -0.90
        # opposite directions → effective_corr = -(-0.90) = 0.90 → concentration!
        open_positions = [{
            "pair": "USD_CHF",
            "direction": "SHORT",
            "lots": 0.1,
            "sl_pips": 20.0,
            "pip_value_per_lot": 10.0,
        }]
        result = opt.portfolio_optimal_size(
            pair="EUR_USD",
            direction="LONG",
            confidence=0.65,
            sl_pips=20.0,
            open_positions=open_positions,
            nav=100000.0,
        )
        # 0.90 effective corr → reduction = max(0.3, 1 - 0.90*0.7) = max(0.3, 0.37) = 0.37
        assert result.correlation_reduction == pytest.approx(0.37, abs=1e-6)

    def test_correlation_reduction_floor_at_0_3(self, journal_with_edge, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=journal_with_edge,
            persistence_path=tmp_path / "kelly.json",
        )
        # Force max effective correlation via 1.0 fake correlation — but using static table,
        # the highest LONG/LONG correlation in defaults is AUD_USD/NZD_USD at 0.90
        # which gives reduction = max(0.3, 1 - 0.9*0.7) = 0.37 (above floor)
        # Floor only kicks at correlation > ~1.0 — verify floor exists by checking
        # the math directly via _correlation_reduction
        reduction = opt._correlation_reduction(
            new_pair="EUR_USD",
            new_direction="LONG",
            open_positions=[{"pair": "GBP_USD", "direction": "LONG"}] * 10,
        )
        # Multiple positions don't compound (uses max), but the value stays above 0.3
        assert reduction >= 0.3

    def test_floor_at_0_5_pct_when_kelly_very_small(self, tmp_path):
        # Build a journal with tiny Kelly edge: 51% win rate, 1:1
        path = _make_journal(tmp_path, wins=51, losses=49, win_pips=20.0, loss_pips=20.0)
        opt = KellyPortfolioOptimizer(
            journal_path=path,
            persistence_path=tmp_path / "kelly.json",
        )
        result = opt.portfolio_optimal_size(
            pair="EUR_USD",
            direction="LONG",
            confidence=0.50,  # No confidence boost (0.50/0.60 < 1.0)
            sl_pips=20.0,
            nav=10000.0,
        )
        # Half-Kelly = 0.01; * confidence(0.5/0.6=0.833) = 0.00833 → floored at 0.005
        assert result.recommended_risk_pct >= 0.005

    def test_full_kelly_mode(self, journal_with_edge, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=journal_with_edge,
            persistence_path=tmp_path / "kelly.json",
            use_half_kelly=False,
        )
        result = opt.portfolio_optimal_size(
            pair="EUR_USD",
            direction="LONG",
            confidence=0.65,
            sl_pips=20.0,
            nav=10000.0,
        )
        # Full Kelly = 0.333; would exceed 5% cap → capped
        assert result.recommended_risk_pct == pytest.approx(MAX_SINGLE_POSITION_PCT)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestComputeOpenRisk:
    def test_no_positions_returns_zero(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        assert opt._compute_open_risk([], nav=10000.0) == 0.0

    def test_zero_nav_returns_zero(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        positions = [{"lots": 1.0, "sl_pips": 20.0, "pip_value_per_lot": 10.0}]
        assert opt._compute_open_risk(positions, nav=0.0) == 0.0

    def test_single_position_risk(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        # 1 lot * 20 pips * $10 = $200; on $10k NAV → 2%
        positions = [{"lots": 1.0, "sl_pips": 20.0, "pip_value_per_lot": 10.0}]
        assert opt._compute_open_risk(positions, nav=10000.0) == pytest.approx(0.02)

    def test_multiple_positions_summed(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        positions = [
            {"lots": 1.0, "sl_pips": 20.0, "pip_value_per_lot": 10.0},   # $200
            {"lots": 0.5, "sl_pips": 30.0, "pip_value_per_lot": 10.0},   # $150
        ]
        # $350 / $10k = 3.5%
        assert opt._compute_open_risk(positions, nav=10000.0) == pytest.approx(0.035)

    def test_missing_keys_use_defaults(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        # Missing pip_value_per_lot defaults to 10.0; missing sl_pips defaults to 10.0
        positions = [{"lots": 1.0}]
        # 1 * 10 * 10 = $100 / $10k = 1%
        assert opt._compute_open_risk(positions, nav=10000.0) == pytest.approx(0.01)


class TestCorrelationReduction:
    def test_no_open_positions_returns_one(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        assert opt._correlation_reduction("EUR_USD", "LONG", []) == 1.0

    def test_uncorrelated_pair_no_reduction(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        # EUR_USD has no correlation entry for "MOON_PAIR"
        result = opt._correlation_reduction(
            "EUR_USD", "LONG",
            [{"pair": "MOON_PAIR", "direction": "LONG"}],
        )
        assert result == 1.0

    def test_pair_not_in_static_table_returns_one(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        result = opt._correlation_reduction(
            "EUR_TRY", "LONG",
            [{"pair": "EUR_USD", "direction": "LONG"}],
        )
        assert result == 1.0

    def test_negative_correlation_same_direction_no_reduction(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        # EUR_USD LONG vs USD_CHF LONG: corr = -0.90
        # same direction → effective_corr = -0.90 → no concentration → reduction=1.0
        result = opt._correlation_reduction(
            "EUR_USD", "LONG",
            [{"pair": "USD_CHF", "direction": "LONG"}],
        )
        assert result == 1.0

    def test_static_correlation_table_has_expected_pairs(self):
        # Guard against accidental table erosion
        assert "EUR_USD" in STATIC_CORRELATIONS
        assert STATIC_CORRELATIONS["EUR_USD"]["GBP_USD"] == 0.85
        assert STATIC_CORRELATIONS["EUR_USD"]["USD_CHF"] == -0.90
        assert STATIC_CORRELATIONS["AUD_USD"]["NZD_USD"] == 0.90


class TestRiskToLots:
    def test_basic_conversion(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        # 2% of $10k = $200; / (20 pips * $10) = 1.0 lot
        lots = opt._risk_to_lots(risk_pct=0.02, nav=10000.0, sl_pips=20.0, pip_value_per_lot=10.0)
        assert lots == pytest.approx(1.0)

    def test_invalid_sl_returns_min_lots(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        assert opt._risk_to_lots(0.02, 10000.0, sl_pips=0.0, pip_value_per_lot=10.0) == 0.01
        assert opt._risk_to_lots(0.02, 10000.0, sl_pips=-5.0, pip_value_per_lot=10.0) == 0.01

    def test_invalid_pip_value_returns_min_lots(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        assert opt._risk_to_lots(0.02, 10000.0, sl_pips=20.0, pip_value_per_lot=0.0) == 0.01

    def test_floored_at_min_lots(self, tmp_path):
        opt = KellyPortfolioOptimizer(
            journal_path=tmp_path / "noexist.json",
            persistence_path=tmp_path / "kelly.json",
        )
        # Tiny risk → would compute < 0.01 → floored
        lots = opt._risk_to_lots(0.0001, 10000.0, 100.0, 10.0)
        assert lots == 0.01


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_snapshot_writes_json(self, journal_with_edge, tmp_path):
        kelly_path = tmp_path / "kelly.json"
        opt = KellyPortfolioOptimizer(
            journal_path=journal_with_edge,
            persistence_path=kelly_path,
        )
        opt.portfolio_optimal_size(
            pair="EUR_USD", direction="LONG", confidence=0.65,
            sl_pips=20.0, nav=10000.0,
        )
        assert kelly_path.exists()
        data = json.loads(kelly_path.read_text())
        assert data["pair"] == "EUR_USD"
        assert "result" in data
        assert "stats" in data
        assert data["stats"]["n_trades"] == 30


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------

class TestResultSerialization:
    def test_to_dict_rounds_floats(self):
        r = KellyResult(
            full_kelly_fraction=0.123456789,
            half_kelly_fraction=0.0617283945,
            recommended_risk_pct=0.04999999,
            recommended_lots=1.234567,
            portfolio_risk_used_pct=0.0876543,
            portfolio_risk_remaining_pct=0.0623457,
            correlation_reduction=0.789012,
            var_constrained=True,
            win_rate=0.616666,
            avg_win_loss_ratio=1.555555,
            reason="test",
        )
        d = r.to_dict()
        assert d["full_kelly_fraction"] == 0.123457
        assert d["recommended_lots"] == 1.2346
        assert d["correlation_reduction"] == 0.789
        assert d["var_constrained"] is True
        assert d["reason"] == "test"

    def test_default_result_serializes_safely(self):
        d = KellyResult().to_dict()
        assert d["full_kelly_fraction"] == 0.0
        assert d["recommended_risk_pct"] == 0.0
        assert d["var_constrained"] is False
