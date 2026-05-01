"""Unit tests for AffinityPortfolioSizer (src/scanner/automation/affinity_portfolio.py).

Closes a zero-coverage gap surfaced by the 2026-05-01 test coverage audit.

The sizer reads pair_affinity_matrix.json (US-062 output) and turns affinity
scores into a continuous size multiplier:
  - effective_affinity ≤ -0.30 → linear reduction toward 0.50 floor
  - effective_affinity ≥ +0.30 → linear boost toward 1.20 ceiling
  - otherwise → 1.00 (neutral)

effective_affinity = 0.6 * mean + 0.4 * max — a conservative blend that
penalises any single concentrated correlation, not just the average.

Matrix path is a module-level constant; tests use monkeypatch to redirect
loading at instantiation time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.scanner.automation.affinity_portfolio as affinity_module
from src.scanner.automation.affinity_portfolio import (
    MAX_SIZE_MULTIPLIER,
    MIN_SIZE_MULTIPLIER,
    NEGATIVE_AFFINITY_THRESHOLD,
    NEUTRAL_MULTIPLIER,
    POSITIVE_AFFINITY_THRESHOLD,
    AffinityPortfolioSizer,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _write_matrix(tmp_path: Path, matrix: dict, key: str = "matrix") -> Path:
    """Write an affinity matrix JSON file at tmp_path."""
    path = tmp_path / "pair_affinity_matrix.json"
    path.write_text(json.dumps({key: matrix}))
    return path


@pytest.fixture
def matrix_with_strong_negative(tmp_path, monkeypatch):
    """Inject an affinity matrix where EUR_USD has strong negative affinity."""
    path = _write_matrix(tmp_path, {
        "EUR_USD": {"GBP_USD": -0.80, "USD_JPY": -0.60},
        "GBP_USD": {"EUR_USD": -0.80, "AUD_USD": 0.10},
    })
    monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
    return path


@pytest.fixture
def matrix_with_strong_positive(tmp_path, monkeypatch):
    path = _write_matrix(tmp_path, {
        "EUR_USD": {"GBP_USD": 0.80, "USD_JPY": 0.60},
    })
    monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
    return path


@pytest.fixture
def matrix_neutral(tmp_path, monkeypatch):
    path = _write_matrix(tmp_path, {
        "EUR_USD": {"GBP_USD": 0.0, "USD_JPY": 0.10},
    })
    monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
    return path


@pytest.fixture
def no_matrix(tmp_path, monkeypatch):
    """Point matrix path at a non-existent file."""
    monkeypatch.setattr(
        affinity_module, "_AFFINITY_MATRIX_PATH", tmp_path / "missing.json",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_thresholds(self):
        assert NEGATIVE_AFFINITY_THRESHOLD == -0.30
        assert POSITIVE_AFFINITY_THRESHOLD == 0.30
        assert MIN_SIZE_MULTIPLIER == 0.50
        assert MAX_SIZE_MULTIPLIER == 1.20
        assert NEUTRAL_MULTIPLIER == 1.0


# ---------------------------------------------------------------------------
# Construction & matrix loading
# ---------------------------------------------------------------------------

class TestMatrixLoad:
    def test_no_matrix_file(self, no_matrix, tmp_path):
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        assert s._matrix_loaded is False
        assert s._affinity_matrix == {}

    def test_loads_from_matrix_key(self, matrix_with_strong_negative, tmp_path):
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        assert s._matrix_loaded is True
        assert "EUR_USD" in s._affinity_matrix

    def test_loads_from_affinity_scores_key(self, tmp_path, monkeypatch):
        path = tmp_path / "pair_affinity_matrix.json"
        path.write_text(json.dumps({
            "affinity_scores": {"EUR_USD": {"GBP_USD": 0.5}},
        }))
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        assert s._matrix_loaded is True
        assert s._affinity_matrix["EUR_USD"]["GBP_USD"] == 0.5

    def test_loads_flat_matrix_format(self, tmp_path, monkeypatch):
        # No 'matrix' or 'affinity_scores' key — treats whole dict as matrix
        path = tmp_path / "pair_affinity_matrix.json"
        path.write_text(json.dumps({
            "EUR_USD": {"GBP_USD": 0.5},
            "GBP_USD": {"EUR_USD": 0.5},
        }))
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        assert s._matrix_loaded is True
        assert s._affinity_matrix["EUR_USD"]["GBP_USD"] == 0.5

    def test_handles_corrupt_matrix(self, tmp_path, monkeypatch):
        path = tmp_path / "pair_affinity_matrix.json"
        path.write_text("{ invalid json")
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        # Must not raise. safe_json returns default {} on corruption — the loader
        # then "loads" an empty matrix. The downstream gate (`not self._affinity_matrix`)
        # still falls back to neutral on empty matrices.
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        assert s._affinity_matrix == {}
        # Behavior must still be neutral
        assert s.get_affinity_multiplier("EUR_USD", ["GBP_USD"]) == NEUTRAL_MULTIPLIER

    def test_handles_non_dict_matrix(self, tmp_path, monkeypatch):
        path = tmp_path / "pair_affinity_matrix.json"
        path.write_text(json.dumps([1, 2, 3]))
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        assert s._matrix_loaded is False

    def test_reload_matrix_reflects_new_data(self, tmp_path, monkeypatch):
        path = tmp_path / "pair_affinity_matrix.json"
        path.write_text(json.dumps({"matrix": {"EUR_USD": {"GBP_USD": 0.5}}}))
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        assert s._affinity_matrix["EUR_USD"]["GBP_USD"] == 0.5

        # Update file & reload
        path.write_text(json.dumps({"matrix": {"EUR_USD": {"GBP_USD": -0.9}}}))
        ok = s.reload_matrix()
        assert ok is True
        assert s._affinity_matrix["EUR_USD"]["GBP_USD"] == -0.9


# ---------------------------------------------------------------------------
# get_affinity_multiplier — neutral / fallback paths
# ---------------------------------------------------------------------------

class TestNeutralPaths:
    def test_no_open_positions_returns_neutral(self, matrix_with_strong_negative, tmp_path):
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        assert s.get_affinity_multiplier("EUR_USD", []) == NEUTRAL_MULTIPLIER

    def test_no_matrix_returns_neutral(self, no_matrix, tmp_path):
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        assert s.get_affinity_multiplier("EUR_USD", ["GBP_USD"]) == NEUTRAL_MULTIPLIER

    def test_no_overlap_returns_neutral(self, matrix_with_strong_negative, tmp_path):
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        # Pair not in matrix at all
        assert s.get_affinity_multiplier("MOON_USD", ["GBP_USD"]) == NEUTRAL_MULTIPLIER

    def test_partial_overlap_uses_known_pairs_only(self, matrix_with_strong_negative, tmp_path):
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        # Open: GBP_USD (known, -0.80), MOON_USD (unknown). Should use only GBP_USD.
        # effective = 0.6 * -0.80 + 0.4 * -0.80 = -0.80
        # reduction = (0.80 - 0.30) / 0.70 = 0.714
        # mult = max(0.50, 1.0 - 0.714 * 0.50) = max(0.50, 0.643) = 0.643
        m = s.get_affinity_multiplier("EUR_USD", ["GBP_USD", "MOON_USD"])
        assert m == pytest.approx(0.643, abs=1e-3)


# ---------------------------------------------------------------------------
# Negative affinity → reduction
# ---------------------------------------------------------------------------

class TestNegativeAffinityReduction:
    def test_strong_negative_reduces_size(self, matrix_with_strong_negative, tmp_path):
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        # GBP_USD=-0.80, USD_JPY=-0.60 → mean=-0.70, max=-0.60
        # effective = 0.6*-0.70 + 0.4*-0.60 = -0.42 - 0.24 = -0.66
        # reduction = (0.66 - 0.30) / (1.0 - 0.30) = 0.36 / 0.70 = 0.514
        # mult = max(0.50, 1.0 - 0.514 * 0.50) = max(0.50, 0.743) = 0.743
        m = s.get_affinity_multiplier("EUR_USD", ["GBP_USD", "USD_JPY"])
        assert m < NEUTRAL_MULTIPLIER
        assert m == pytest.approx(0.743, abs=1e-3)

    def test_at_negative_threshold_returns_neutral(self, tmp_path, monkeypatch):
        # affinity exactly at -0.30 → boundary case (≤ threshold → reduction branch)
        # reduction = (0.30 - 0.30) / 0.70 = 0 → mult = 1.0 - 0 = 1.0
        path = _write_matrix(tmp_path, {"EUR_USD": {"GBP_USD": -0.30}})
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        m = s.get_affinity_multiplier("EUR_USD", ["GBP_USD"])
        assert m == pytest.approx(NEUTRAL_MULTIPLIER, abs=1e-6)

    def test_extreme_negative_floored_at_min(self, tmp_path, monkeypatch):
        # affinity = -1.0 → reduction = 0.70/0.70 = 1.0 → mult = 1.0 - 0.50 = 0.50
        path = _write_matrix(tmp_path, {"EUR_USD": {"GBP_USD": -1.0}})
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        m = s.get_affinity_multiplier("EUR_USD", ["GBP_USD"])
        assert m == MIN_SIZE_MULTIPLIER

    def test_just_below_threshold_no_reduction(self, tmp_path, monkeypatch):
        # affinity = -0.29 → above threshold (-0.30) → neutral band
        path = _write_matrix(tmp_path, {"EUR_USD": {"GBP_USD": -0.29}})
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        m = s.get_affinity_multiplier("EUR_USD", ["GBP_USD"])
        assert m == NEUTRAL_MULTIPLIER


# ---------------------------------------------------------------------------
# Positive affinity → boost
# ---------------------------------------------------------------------------

class TestPositiveAffinityBoost:
    def test_strong_positive_boosts_size(self, matrix_with_strong_positive, tmp_path):
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        # GBP_USD=0.80, USD_JPY=0.60 → mean=0.70, max=0.80
        # effective = 0.6*0.70 + 0.4*0.80 = 0.42 + 0.32 = 0.74
        # boost = (0.74 - 0.30) / 0.70 = 0.629
        # mult = min(1.20, 1.0 + 0.629 * 0.20) = min(1.20, 1.126) = 1.126
        m = s.get_affinity_multiplier("EUR_USD", ["GBP_USD", "USD_JPY"])
        assert m > NEUTRAL_MULTIPLIER
        assert m == pytest.approx(1.126, abs=1e-3)

    def test_at_positive_threshold_returns_neutral(self, tmp_path, monkeypatch):
        # affinity = 0.30 → boundary; in source: ≥ threshold → boost branch
        # boost = (0.30 - 0.30) / 0.70 = 0 → mult = 1.0 + 0 = 1.0
        path = _write_matrix(tmp_path, {"EUR_USD": {"GBP_USD": 0.30}})
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        m = s.get_affinity_multiplier("EUR_USD", ["GBP_USD"])
        assert m == pytest.approx(NEUTRAL_MULTIPLIER, abs=1e-6)

    def test_extreme_positive_capped_at_max(self, tmp_path, monkeypatch):
        path = _write_matrix(tmp_path, {"EUR_USD": {"GBP_USD": 1.0}})
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        m = s.get_affinity_multiplier("EUR_USD", ["GBP_USD"])
        assert m == MAX_SIZE_MULTIPLIER

    def test_just_below_positive_threshold_neutral(self, tmp_path, monkeypatch):
        path = _write_matrix(tmp_path, {"EUR_USD": {"GBP_USD": 0.29}})
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        m = s.get_affinity_multiplier("EUR_USD", ["GBP_USD"])
        assert m == NEUTRAL_MULTIPLIER


# ---------------------------------------------------------------------------
# Symmetric lookup
# ---------------------------------------------------------------------------

class TestSymmetricLookup:
    def test_lookup_uses_reverse_direction(self, tmp_path, monkeypatch):
        # Matrix only stores GBP_USD → EUR_USD edge; lookup for EUR_USD vs GBP_USD
        # should still find it via reverse direction.
        path = _write_matrix(tmp_path, {"GBP_USD": {"EUR_USD": -0.80}})
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        # Access via internal lookup method
        score = s._get_pair_affinity("EUR_USD", "GBP_USD")
        assert score == -0.80

    def test_returns_none_for_unknown_pair(self, tmp_path, monkeypatch):
        path = _write_matrix(tmp_path, {"EUR_USD": {"GBP_USD": 0.5}})
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        assert s._get_pair_affinity("MOON", "USD_JPY") is None

    def test_forward_direction_takes_priority(self, tmp_path, monkeypatch):
        # If both directions stored, forward wins
        path = _write_matrix(tmp_path, {
            "EUR_USD": {"GBP_USD": 0.5},
            "GBP_USD": {"EUR_USD": -0.5},  # contradictory reverse value
        })
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        # Forward EUR→GBP wins
        assert s._get_pair_affinity("EUR_USD", "GBP_USD") == 0.5


# ---------------------------------------------------------------------------
# Effective affinity blend (max-weighted)
# ---------------------------------------------------------------------------

class TestEffectiveBlend:
    def test_max_dominates_over_average(self, tmp_path, monkeypatch):
        # mean=0.0 (one +0.95, one -0.95), max=+0.95
        # effective = 0.6*0 + 0.4*0.95 = 0.38 → above POSITIVE_AFFINITY_THRESHOLD
        path = _write_matrix(tmp_path, {
            "EUR_USD": {"GBP_USD": 0.95, "USD_JPY": -0.95},
        })
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        m = s.get_affinity_multiplier("EUR_USD", ["GBP_USD", "USD_JPY"])
        # 0.38 → boost = 0.08/0.70 = 0.114 → mult = 1 + 0.114*0.20 = 1.023
        assert m > 1.0
        assert m == pytest.approx(1.023, abs=1e-3)

    def test_single_pair_effective_equals_score(self, tmp_path, monkeypatch):
        # mean = max = the only score → effective = 1.0 * score
        path = _write_matrix(tmp_path, {"EUR_USD": {"GBP_USD": -0.50}})
        monkeypatch.setattr(affinity_module, "_AFFINITY_MATRIX_PATH", path)
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        m = s.get_affinity_multiplier("EUR_USD", ["GBP_USD"])
        # effective = -0.50; reduction = 0.20/0.70 = 0.286
        # mult = max(0.50, 1.0 - 0.286*0.50) = 0.857
        assert m == pytest.approx(0.857, abs=1e-3)


# ---------------------------------------------------------------------------
# Decision logging
# ---------------------------------------------------------------------------

class TestDecisionLog:
    def test_decision_logged_when_matrix_active(self, matrix_with_strong_negative, tmp_path):
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        before = s._total_decisions
        s.get_affinity_multiplier("EUR_USD", ["GBP_USD"])
        assert s._total_decisions == before + 1
        assert len(s._decisions) == 1
        d = s._decisions[0]
        assert d["pair"] == "EUR_USD"
        assert d["open_positions"] == ["GBP_USD"]
        assert "avg_affinity" in d
        assert "max_affinity" in d
        assert "effective_affinity" in d
        assert "multiplier" in d

    def test_neutral_path_does_not_log(self, no_matrix, tmp_path):
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        s.get_affinity_multiplier("EUR_USD", ["GBP_USD"])
        # No decision logged when matrix unavailable (early return)
        assert s._total_decisions == 0

    def test_no_open_positions_does_not_log(self, matrix_with_strong_negative, tmp_path):
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        s.get_affinity_multiplier("EUR_USD", [])
        assert s._total_decisions == 0

    def test_decision_log_truncates_at_300(self, matrix_with_strong_negative, tmp_path):
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        for _ in range(350):
            s.get_affinity_multiplier("EUR_USD", ["GBP_USD"])
        # Spec: when len > 300, truncate to last 150
        assert 150 <= len(s._decisions) <= 300


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_keys(self, matrix_with_strong_negative, tmp_path):
        s = AffinityPortfolioSizer(persistence_path=tmp_path / "log.json")
        st = s.get_status()
        for key in ("matrix_loaded", "n_pairs_in_matrix", "total_decisions", "recent_decisions"):
            assert key in st
        assert st["matrix_loaded"] is True
        assert st["n_pairs_in_matrix"] == 2
        assert st["total_decisions"] == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_state_writes_json(self, matrix_with_strong_negative, tmp_path):
        path = tmp_path / "log.json"
        s = AffinityPortfolioSizer(persistence_path=path)
        s.get_affinity_multiplier("EUR_USD", ["GBP_USD"])
        s.save_state()
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert data["total_decisions"] == 1
        assert "decisions" in data

    def test_save_creates_parent_dir(self, no_matrix, tmp_path):
        path = tmp_path / "deep" / "log.json"
        s = AffinityPortfolioSizer(persistence_path=path)
        s.save_state()
        assert path.exists()

    def test_save_swallows_exceptions(self, no_matrix, tmp_path, monkeypatch):
        from pathlib import Path as _P
        path = tmp_path / "log.json"
        s = AffinityPortfolioSizer(persistence_path=path)

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(_P, "write_text", _boom)
        s.save_state()  # Must not raise

    def test_round_trip_preserves_decisions(self, matrix_with_strong_negative, tmp_path):
        path = tmp_path / "log.json"
        s1 = AffinityPortfolioSizer(persistence_path=path)
        for _ in range(5):
            s1.get_affinity_multiplier("EUR_USD", ["GBP_USD"])
        total_before = s1._total_decisions
        s1.save_state()

        s2 = AffinityPortfolioSizer(persistence_path=path)
        assert s2._total_decisions == total_before
        assert len(s2._decisions) == 5

    def test_load_handles_corrupt_state(self, matrix_with_strong_negative, tmp_path):
        path = tmp_path / "log.json"
        path.write_text("{ invalid")
        # Must not raise
        s = AffinityPortfolioSizer(persistence_path=path)
        assert s._total_decisions == 0
