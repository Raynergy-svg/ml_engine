"""Unit tests for EntropyPositionSizer (src/scanner/automation/entropy_sizer.py).

Closes a zero-coverage gap surfaced by the 2026-05-01 test coverage audit. The
module computes Shannon entropy across the 15-agent ensemble vote distribution
and turns it into a regime-conditioned position-size multiplier:
- Low entropy (consensus) → boost up to max_boost
- High entropy (disagreement) → penalty down to max_penalty
- Multiplier clamped to [min_multiplier, max_multiplier]

Critical because entropy multiplies position risk on every trade.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.scanner.automation.entropy_sizer import (
    DEFAULT_MAX_MULTIPLIER,
    DEFAULT_MIN_MULTIPLIER,
    REGIME_ENTROPY_THRESHOLDS,
    EntropyPositionSizer,
    EntropyResult,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        sizer = EntropyPositionSizer()
        assert sizer.min_multiplier == DEFAULT_MIN_MULTIPLIER
        assert sizer.max_multiplier == DEFAULT_MAX_MULTIPLIER
        assert sizer.persistence_path.name == "entropy_sizing_log.json"
        assert len(sizer._history) == 0

    def test_custom_multipliers(self):
        sizer = EntropyPositionSizer(min_multiplier=0.10, max_multiplier=2.00)
        assert sizer.min_multiplier == 0.10
        assert sizer.max_multiplier == 2.00

    def test_custom_persistence_path(self, tmp_path):
        path = tmp_path / "custom.json"
        sizer = EntropyPositionSizer(persistence_path=path)
        assert sizer.persistence_path == path


# ---------------------------------------------------------------------------
# Vote categorization & entropy math
# ---------------------------------------------------------------------------

class TestVoteCategorization:
    def test_empty_votes_returns_zero_result(self):
        sizer = EntropyPositionSizer()
        result = sizer.compute_ensemble_entropy({})
        assert result.n_agents == 0
        assert result.raw_entropy == 0.0
        assert result.normalized_entropy == 0.0
        assert result.vote_distribution == {}

    def test_bool_votes_categorized(self):
        sizer = EntropyPositionSizer()
        votes = {"a": True, "b": True, "c": False}
        result = sizer.compute_ensemble_entropy(votes)
        assert result.vote_distribution["FOR"] == 2
        assert result.vote_distribution["AGAINST"] == 1
        assert result.vote_distribution["NEUTRAL"] == 0
        assert result.n_agents == 3

    def test_str_votes_buy_sell_categorized(self):
        sizer = EntropyPositionSizer()
        votes = {"a": "BUY", "b": "buy", "c": "SELL"}
        # Note: "SELL" maps to FOR per the source (any of BUY/SELL/FOR/TRUE/1)
        # Let's verify what the actual categorization does
        result = sizer.compute_ensemble_entropy(votes)
        # Per source: BUY/SELL/FOR/TRUE/1 → FOR; AGAINST/FALSE/0/HOLD → AGAINST
        assert result.vote_distribution["FOR"] == 3
        assert result.vote_distribution["AGAINST"] == 0

    def test_str_votes_hold_categorized_as_against(self):
        sizer = EntropyPositionSizer()
        votes = {"a": "HOLD", "b": "AGAINST", "c": "FALSE"}
        result = sizer.compute_ensemble_entropy(votes)
        assert result.vote_distribution["AGAINST"] == 3

    def test_str_votes_unknown_become_neutral(self):
        sizer = EntropyPositionSizer()
        votes = {"a": "MAYBE", "b": "unknown"}
        result = sizer.compute_ensemble_entropy(votes)
        assert result.vote_distribution["NEUTRAL"] == 2

    def test_numeric_votes_threshold_at_pm_0_1(self):
        sizer = EntropyPositionSizer()
        votes = {"a": 0.5, "b": -0.5, "c": 0.05, "d": -0.05, "e": 0.0}
        result = sizer.compute_ensemble_entropy(votes)
        assert result.vote_distribution["FOR"] == 1
        assert result.vote_distribution["AGAINST"] == 1
        # 0.05, -0.05, 0.0 are all in dead zone
        assert result.vote_distribution["NEUTRAL"] == 3

    def test_unknown_type_becomes_neutral(self):
        sizer = EntropyPositionSizer()
        votes = {"a": None, "b": [1, 2]}
        result = sizer.compute_ensemble_entropy(votes)
        assert result.vote_distribution["NEUTRAL"] == 2


class TestEntropyMath:
    def test_perfect_consensus_zero_entropy(self):
        sizer = EntropyPositionSizer()
        # 15 agents, all FOR → entropy = 0 (perfect consensus)
        votes = {f"a{i}": True for i in range(15)}
        result = sizer.compute_ensemble_entropy(votes)
        assert result.raw_entropy == 0.0
        assert result.normalized_entropy == 0.0

    def test_50_50_split_max_entropy(self):
        sizer = EntropyPositionSizer()
        votes = {f"a{i}": (i % 2 == 0) for i in range(10)}
        result = sizer.compute_ensemble_entropy(votes)
        # H = -2 * 0.5 * log2(0.5) = 1.0 bit
        assert result.raw_entropy == pytest.approx(1.0, abs=1e-9)
        assert result.normalized_entropy == pytest.approx(1.0, abs=1e-9)

    def test_three_way_split_normalized_to_one(self):
        sizer = EntropyPositionSizer()
        # 5 FOR, 5 AGAINST, 5 NEUTRAL → max entropy across 3 categories
        votes = {}
        for i in range(5):
            votes[f"f{i}"] = True
        for i in range(5):
            votes[f"a{i}"] = False
        for i in range(5):
            votes[f"n{i}"] = "MAYBE"
        result = sizer.compute_ensemble_entropy(votes)
        # H = log2(3) ≈ 1.585; normalized by log2(3) → 1.0
        assert result.raw_entropy == pytest.approx(math.log2(3), abs=1e-9)
        assert result.normalized_entropy == pytest.approx(1.0, abs=1e-9)

    def test_single_vote_entropy_is_zero(self):
        sizer = EntropyPositionSizer()
        result = sizer.compute_ensemble_entropy({"a": True})
        assert result.raw_entropy == 0.0
        assert result.normalized_entropy == 0.0
        assert result.n_agents == 1


# ---------------------------------------------------------------------------
# Size multiplier mapping
# ---------------------------------------------------------------------------

class TestSizeMultiplier:
    def test_consensus_returns_max_boost_for_regime(self):
        sizer = EntropyPositionSizer()
        # Zero entropy → full boost
        for regime, params in REGIME_ENTROPY_THRESHOLDS.items():
            mult = sizer.get_size_multiplier(normalized_entropy=0.0, regime=regime)
            assert mult == pytest.approx(params["max_boost"], abs=1e-6)

    def test_max_disagreement_returns_max_penalty_for_regime(self):
        # Use a wider min_multiplier band so clamp doesn't override per-regime penalty
        sizer = EntropyPositionSizer(min_multiplier=0.10)
        # Entropy = 1.0 → minimum (max_penalty floor)
        for regime, params in REGIME_ENTROPY_THRESHOLDS.items():
            mult = sizer.get_size_multiplier(normalized_entropy=1.0, regime=regime)
            assert mult == pytest.approx(params["max_penalty"], abs=1e-6)

    def test_default_min_multiplier_clamps_extreme_penalty(self):
        """Default min_multiplier=0.30 floors the EXTREME max_penalty=0.25."""
        sizer = EntropyPositionSizer()
        mult = sizer.get_size_multiplier(normalized_entropy=1.0, regime="EXTREME")
        # Without clamp would be 0.25; with default 0.30 floor → 0.30
        assert mult == 0.30

    def test_at_neutral_threshold_multiplier_is_one(self):
        sizer = EntropyPositionSizer()
        # At neutral, boost path returns 1.0 + (max_boost-1)*0 = 1.0
        for regime, params in REGIME_ENTROPY_THRESHOLDS.items():
            mult = sizer.get_size_multiplier(
                normalized_entropy=params["neutral"], regime=regime,
            )
            assert mult == pytest.approx(1.0, abs=1e-6)

    def test_unknown_regime_falls_back_to_normal(self):
        sizer = EntropyPositionSizer()
        m_unknown = sizer.get_size_multiplier(normalized_entropy=0.5, regime="MOON")
        m_normal = sizer.get_size_multiplier(normalized_entropy=0.5, regime="NORMAL")
        assert m_unknown == m_normal

    def test_multiplier_clamped_to_min(self):
        sizer = EntropyPositionSizer(min_multiplier=0.50)
        # EXTREME max_penalty=0.25, but our min_multiplier=0.50 should override
        mult = sizer.get_size_multiplier(normalized_entropy=1.0, regime="EXTREME")
        assert mult == 0.50

    def test_multiplier_clamped_to_max(self):
        sizer = EntropyPositionSizer(max_multiplier=1.10)
        # NORMAL max_boost=1.15 but we cap at 1.10
        mult = sizer.get_size_multiplier(normalized_entropy=0.0, regime="NORMAL")
        assert mult == 1.10

    def test_no_input_returns_neutral_one(self):
        sizer = EntropyPositionSizer()
        mult = sizer.get_size_multiplier()
        assert mult == 1.0

    def test_entropy_result_input_takes_priority_over_normalized(self):
        sizer = EntropyPositionSizer()
        result = EntropyResult(normalized_entropy=0.0)
        # Even when normalized_entropy=1.0 is also passed, the result wins
        mult = sizer.get_size_multiplier(
            entropy_result=result,
            normalized_entropy=1.0,
            regime="NORMAL",
        )
        # 0.0 entropy → max boost (1.15)
        assert mult == pytest.approx(1.15)

    def test_low_regime_more_tolerant_than_extreme(self):
        """At the same entropy, LOW should give a higher (more permissive) multiplier."""
        sizer = EntropyPositionSizer()
        h = 0.45
        m_low = sizer.get_size_multiplier(normalized_entropy=h, regime="LOW")
        m_extreme = sizer.get_size_multiplier(normalized_entropy=h, regime="EXTREME")
        assert m_low > m_extreme


class TestMonotonicity:
    def test_multiplier_monotonically_decreasing_with_entropy(self):
        sizer = EntropyPositionSizer()
        last_mult = float("inf")
        for h in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            mult = sizer.get_size_multiplier(normalized_entropy=h, regime="NORMAL")
            assert mult <= last_mult + 1e-9
            last_mult = mult


# ---------------------------------------------------------------------------
# size_from_votes (end-to-end)
# ---------------------------------------------------------------------------

class TestSizeFromVotes:
    def test_consensus_votes_yield_high_multiplier(self):
        sizer = EntropyPositionSizer()
        votes = {f"a{i}": True for i in range(15)}
        result = sizer.size_from_votes(votes, regime="NORMAL")
        assert result.normalized_entropy == 0.0
        assert result.size_multiplier == pytest.approx(1.15)
        assert result.regime == "NORMAL"
        assert result.n_agents == 15

    def test_split_votes_yield_low_multiplier(self):
        sizer = EntropyPositionSizer()
        votes = {f"a{i}": (i % 2 == 0) for i in range(14)}
        result = sizer.size_from_votes(votes, regime="HIGH")
        # 50/50 → entropy 1.0 → HIGH max_penalty=0.30
        assert result.size_multiplier == pytest.approx(0.30)

    def test_history_grows_per_call(self):
        sizer = EntropyPositionSizer()
        for _ in range(3):
            sizer.size_from_votes({"a": True, "b": False}, regime="NORMAL")
        assert len(sizer._history) == 3

    def test_history_capped_at_200(self):
        sizer = EntropyPositionSizer()
        for _ in range(250):
            sizer.size_from_votes({"a": True}, regime="NORMAL")
        # deque(maxlen=200)
        assert len(sizer._history) == 200


# ---------------------------------------------------------------------------
# Status & persistence
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_empty_history(self):
        sizer = EntropyPositionSizer()
        status = sizer.get_status()
        assert status["total_computations"] == 0
        assert status["avg_entropy_recent"] == 0.0
        assert status["avg_multiplier_recent"] == 1.0
        assert "regime_thresholds" in status

    def test_status_reflects_recent_history(self):
        sizer = EntropyPositionSizer()
        # 15 calls — all consensus (entropy=0.0, multiplier=1.15 in NORMAL)
        for _ in range(15):
            sizer.size_from_votes({f"a{i}": True for i in range(10)}, regime="NORMAL")
        status = sizer.get_status()
        assert status["total_computations"] == 15
        assert status["avg_entropy_recent"] == pytest.approx(0.0)
        assert status["avg_multiplier_recent"] == pytest.approx(1.15, abs=1e-3)


class TestPersistence:
    def test_save_log_writes_json(self, tmp_path):
        path = tmp_path / "entropy.json"
        sizer = EntropyPositionSizer(persistence_path=path)
        sizer.size_from_votes({"a": True, "b": False}, regime="NORMAL")
        sizer.save_log()
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert "history" in data
        assert "last_updated" in data

    def test_save_log_truncates_to_100_entries(self, tmp_path):
        path = tmp_path / "entropy.json"
        sizer = EntropyPositionSizer(persistence_path=path)
        for i in range(150):
            sizer.size_from_votes({"a": True, "b": False}, regime="NORMAL")
        sizer.save_log()
        import json
        data = json.loads(path.read_text())
        assert len(data["history"]) == 100

    def test_save_log_creates_parent_dir(self, tmp_path):
        path = tmp_path / "nested" / "deeper" / "entropy.json"
        sizer = EntropyPositionSizer(persistence_path=path)
        sizer.size_from_votes({"a": True}, regime="NORMAL")
        sizer.save_log()
        assert path.exists()

    def test_save_log_swallows_exceptions(self, tmp_path, monkeypatch):
        # Force write failure — should not raise
        path = tmp_path / "entropy.json"
        sizer = EntropyPositionSizer(persistence_path=path)
        sizer.size_from_votes({"a": True}, regime="NORMAL")

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", _boom)
        # Must not raise
        sizer.save_log()


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------

class TestResultSerialization:
    def test_to_dict_rounds_floats(self):
        result = EntropyResult(
            raw_entropy=0.123456789,
            normalized_entropy=0.987654321,
            size_multiplier=1.234567,
            regime="NORMAL",
            vote_distribution={"FOR": 5, "AGAINST": 3, "NEUTRAL": 2},
            n_agents=10,
        )
        d = result.to_dict()
        assert d["raw_entropy"] == 0.123457
        assert d["normalized_entropy"] == 0.9877
        assert d["size_multiplier"] == 1.2346
        assert d["regime"] == "NORMAL"
        assert d["vote_distribution"] == {"FOR": 5, "AGAINST": 3, "NEUTRAL": 2}
        assert d["n_agents"] == 10

    def test_default_result_serializes(self):
        d = EntropyResult().to_dict()
        assert d["raw_entropy"] == 0.0
        assert d["size_multiplier"] == 1.0
        assert d["n_agents"] == 0
