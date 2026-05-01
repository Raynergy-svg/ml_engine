"""Unit tests for ModelBanditSelector (src/scanner/automation/model_bandit.py).

Closes a zero-coverage gap surfaced by the 2026-05-01 test coverage audit.

Tests EXP3 multi-armed bandit for online ensemble model selection:
- Probability mixing: p(arm) = (1-gamma) * W[arm] / sum(W) + gamma / K
- Reward update: W[arm] *= exp(gamma * r_hat / K) where r_hat = reward / p(arm)
- Reward squashed via tanh(pnl / 50.0) to [-1, 1]
- Update clamped to [-2, 2] before exp() to prevent weight explosion
- Weights normalized so sum(W) ≈ K and floored at 0.01 (no arm starvation)
- Gamma decays by gamma_decay each update; floor at min_gamma

Determinism: tests seed numpy.random for select_model() to make sampling
reproducible.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.scanner.automation.model_bandit import (
    DEFAULT_GAMMA,
    DEFAULT_GAMMA_DECAY,
    DEFAULT_MIN_GAMMA,
    DEFAULT_MODELS,
    ModelBanditSelector,
    SelectionResult,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        assert b.models == DEFAULT_MODELS
        assert b.K == 3
        assert b.gamma == DEFAULT_GAMMA
        assert b.min_gamma == DEFAULT_MIN_GAMMA
        assert b.gamma_decay == DEFAULT_GAMMA_DECAY
        # Uniform initialization
        assert all(w == 1.0 for w in b._weights.values())
        assert b._rounds == 0

    def test_custom_models(self, tmp_path):
        b = ModelBanditSelector(
            models=["A", "B", "C", "D"],
            persistence_path=tmp_path / "bandit.json",
        )
        assert b.models == ["A", "B", "C", "D"]
        assert b.K == 4
        assert set(b._weights.keys()) == {"A", "B", "C", "D"}
        assert all(w == 1.0 for w in b._weights.values())

    def test_custom_gamma(self, tmp_path):
        b = ModelBanditSelector(
            gamma=0.25, min_gamma=0.05, gamma_decay=0.99,
            persistence_path=tmp_path / "bandit.json",
        )
        assert b.gamma == 0.25
        assert b.min_gamma == 0.05
        assert b.gamma_decay == 0.99


# ---------------------------------------------------------------------------
# Probability computation (EXP3 mixed strategy)
# ---------------------------------------------------------------------------

class TestProbabilityComputation:
    def test_uniform_weights_yield_uniform_probabilities(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        probs = b.get_model_probabilities()
        assert len(probs) == 3
        for p in probs.values():
            assert p == pytest.approx(1.0 / 3.0)
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_probabilities_sum_to_one(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        # Force asymmetric weights
        b._weights = {"TCN": 5.0, "Ridge": 1.0, "RF": 2.0}
        probs = b.get_model_probabilities()
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)

    def test_higher_weight_yields_higher_probability(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        b._weights = {"TCN": 10.0, "Ridge": 1.0, "RF": 1.0}
        probs = b.get_model_probabilities()
        assert probs["TCN"] > probs["Ridge"]
        assert probs["TCN"] > probs["RF"]

    def test_explore_floor_via_gamma(self, tmp_path):
        # gamma=0.30 → minimum probability ≥ gamma/K = 0.10 for any arm
        b = ModelBanditSelector(
            gamma=0.30,
            persistence_path=tmp_path / "bandit.json",
        )
        b._weights = {"TCN": 100.0, "Ridge": 0.01, "RF": 0.01}
        probs = b.get_model_probabilities()
        for p in probs.values():
            assert p >= 0.30 / 3.0 - 1e-9

    def test_zero_weight_sum_returns_uniform(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        # Manually zero out (degenerate); _compute_probabilities should fall back to uniform
        b._weights = {"TCN": 0.0, "Ridge": 0.0, "RF": 0.0}
        probs = b.get_model_probabilities()
        for p in probs.values():
            assert p == pytest.approx(1.0 / 3.0)

    def test_exp3_mixing_formula(self, tmp_path):
        # Verify exact formula: p(arm) = (1-gamma) * W[arm]/sum(W) + gamma/K
        b = ModelBanditSelector(
            gamma=0.10,
            persistence_path=tmp_path / "bandit.json",
        )
        b._weights = {"TCN": 4.0, "Ridge": 2.0, "RF": 2.0}
        probs = b.get_model_probabilities()
        # Expected for TCN: 0.9 * 4/8 + 0.1/3 = 0.45 + 0.0333 = 0.4833
        # Expected for Ridge/RF: 0.9 * 2/8 + 0.1/3 = 0.225 + 0.0333 = 0.2583
        # After normalization (should already be 1.0 from sum)
        expected_tcn = 0.9 * 0.5 + 0.1 / 3
        expected_other = 0.9 * 0.25 + 0.1 / 3
        total = expected_tcn + 2 * expected_other
        assert probs["TCN"] == pytest.approx(expected_tcn / total, abs=1e-6)
        assert probs["Ridge"] == pytest.approx(expected_other / total, abs=1e-6)


# ---------------------------------------------------------------------------
# select_model
# ---------------------------------------------------------------------------

class TestSelectModel:
    def test_selection_returns_valid_result(self, tmp_path):
        np.random.seed(0)
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        result = b.select_model()
        assert isinstance(result, SelectionResult)
        assert result.selected_model in b.models
        assert 0.0 < result.selection_probability <= 1.0
        assert set(result.model_weights.keys()) == set(b.models)
        assert set(result.model_probabilities.keys()) == set(b.models)

    def test_selection_increments_counters(self, tmp_path):
        np.random.seed(0)
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        for _ in range(5):
            b.select_model()
        assert b._rounds == 5
        assert sum(b._selection_counts.values()) == 5

    def test_uniform_distribution_at_init(self, tmp_path):
        # With uniform weights, large sample should converge to uniform selection
        np.random.seed(42)
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        for _ in range(3000):
            b.select_model()
        # Each arm should be ~1000 ± a few hundred
        for m in b.models:
            count = b._selection_counts[m]
            # Use generous tolerance — chi-square would be ideal but binomial bound is fine
            assert 800 <= count <= 1200, f"{m}: {count}"

    def test_dominant_arm_gets_picked_more_often(self, tmp_path):
        np.random.seed(1)
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        # Force massive weight asymmetry
        b._weights = {"TCN": 100.0, "Ridge": 1.0, "RF": 1.0}
        for _ in range(1000):
            b.select_model()
        assert b._selection_counts["TCN"] > b._selection_counts["Ridge"]
        assert b._selection_counts["TCN"] > b._selection_counts["RF"]


# ---------------------------------------------------------------------------
# update_reward
# ---------------------------------------------------------------------------

class TestUpdateReward:
    def test_unknown_model_no_op(self, tmp_path, caplog):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        before = dict(b._weights)
        b.update_reward("UNKNOWN", realized_pnl=100.0)
        assert b._weights == before

    def test_positive_reward_increases_weight(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        # Reward positive but normalize via tanh — also normalize-to-K applies
        before_tcn = b._weights["TCN"]
        b.update_reward("TCN", realized_pnl=100.0, selection_probability=0.5)
        # After update + normalization, TCN should be higher relative to others
        ratio_after = b._weights["TCN"] / b._weights["Ridge"]
        assert ratio_after > 1.0

    def test_negative_reward_decreases_weight(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        b.update_reward("TCN", realized_pnl=-100.0, selection_probability=0.5)
        # TCN should be lower than peers after a loss
        ratio_after = b._weights["TCN"] / b._weights["Ridge"]
        assert ratio_after < 1.0

    def test_reward_squashed_via_tanh(self, tmp_path):
        # Two extremely different PnLs should produce updates of similar magnitude
        # because tanh squashes both to ~1.0
        b1 = ModelBanditSelector(persistence_path=tmp_path / "b1.json")
        b2 = ModelBanditSelector(persistence_path=tmp_path / "b2.json")
        b1.update_reward("TCN", realized_pnl=200.0, selection_probability=0.5)
        b2.update_reward("TCN", realized_pnl=10000.0, selection_probability=0.5)
        # Both rewards saturate near tanh(infty)=1, so weights should be close
        assert abs(b1._weights["TCN"] - b2._weights["TCN"]) < 0.5

    def test_weights_normalized_to_sum_K(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        for _ in range(20):
            b.update_reward("TCN", realized_pnl=50.0, selection_probability=0.4)
            b.update_reward("Ridge", realized_pnl=-30.0, selection_probability=0.3)
        # sum should be ~K (3) due to normalization
        assert sum(b._weights.values()) == pytest.approx(b.K, abs=0.05)

    def test_weights_floored_at_0_01(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        # Crash one arm's weight repeatedly
        for _ in range(50):
            b.update_reward("RF", realized_pnl=-1000.0, selection_probability=0.05)
        # Should still be at least 0.01 (no starvation)
        assert b._weights["RF"] >= 0.01

    def test_gamma_decays_per_update(self, tmp_path):
        b = ModelBanditSelector(
            gamma=0.20, gamma_decay=0.95, min_gamma=0.01,
            persistence_path=tmp_path / "bandit.json",
        )
        gamma_initial = b.gamma
        b.update_reward("TCN", realized_pnl=10.0, selection_probability=0.5)
        # gamma * 0.95
        assert b.gamma == pytest.approx(gamma_initial * 0.95)

    def test_gamma_floored_at_min_gamma(self, tmp_path):
        b = ModelBanditSelector(
            gamma=0.05, min_gamma=0.04, gamma_decay=0.10,  # rapid decay
            persistence_path=tmp_path / "bandit.json",
        )
        for _ in range(10):
            b.update_reward("TCN", realized_pnl=10.0, selection_probability=0.5)
        assert b.gamma == 0.04  # Floored

    def test_total_reward_accumulates(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        b.update_reward("TCN", realized_pnl=50.0, selection_probability=0.5)
        b.update_reward("TCN", realized_pnl=100.0, selection_probability=0.5)
        # Two positive rewards → total_reward[TCN] is sum of squashed rewards
        assert b._total_reward["TCN"] > 0.0
        # tanh(50/50) + tanh(100/50) = tanh(1) + tanh(2) ≈ 0.7616 + 0.9640 ≈ 1.7256
        assert b._total_reward["TCN"] == pytest.approx(
            math.tanh(1.0) + math.tanh(2.0), abs=1e-6
        )

    def test_zero_pnl_neutral_reward(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        before = dict(b._weights)
        b.update_reward("TCN", realized_pnl=0.0, selection_probability=0.5)
        # tanh(0)=0 → r_hat=0 → exp(0)=1 → weight unchanged before normalization;
        # normalization keeps sum=K which means weights stay at 1.0
        assert b._weights["TCN"] == pytest.approx(before["TCN"])

    def test_low_selection_probability_amplifies_update(self, tmp_path):
        # r_hat = r / p; smaller p → bigger r_hat → bigger update
        b1 = ModelBanditSelector(persistence_path=tmp_path / "b1.json")
        b2 = ModelBanditSelector(persistence_path=tmp_path / "b2.json")
        b1.update_reward("TCN", realized_pnl=50.0, selection_probability=0.5)
        b2.update_reward("TCN", realized_pnl=50.0, selection_probability=0.05)
        # b2's TCN weight should be higher (bigger amplification)
        assert b2._weights["TCN"] > b1._weights["TCN"]

    def test_selection_probability_floored_at_0_01(self, tmp_path):
        # Even if caller passes p=0.0, internal floor at 0.01 prevents div-by-zero
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        # Should not raise / not produce NaN
        b.update_reward("TCN", realized_pnl=50.0, selection_probability=0.0)
        assert all(math.isfinite(w) for w in b._weights.values())

    def test_update_clamped_prevents_explosion(self, tmp_path):
        # With reward=1 (max tanh), p=0.01 floor, gamma=1 (max), K=3:
        # update = 1 * 1/0.01 / 3 = 33.3 → clamped to 2.0 → exp(2) ≈ 7.39
        b = ModelBanditSelector(
            gamma=1.0,
            persistence_path=tmp_path / "bandit.json",
        )
        b.update_reward("TCN", realized_pnl=1e9, selection_probability=0.0)
        # Weight before normalization: 1.0 * exp(2) ≈ 7.39
        # After normalization to sum=3: 7.39 / (7.39 + 1 + 1) ≈ 0.787 → 0.787 * 3 ≈ 2.36
        # Just check it didn't blow up
        assert all(math.isfinite(w) for w in b._weights.values())
        assert all(w < 10.0 for w in b._weights.values())

    def test_history_truncates_at_200(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        for _ in range(250):
            b.update_reward("TCN", realized_pnl=10.0, selection_probability=0.5)
        # Spec: when len > 200, truncate to last 100
        assert len(b._reward_history) <= 200
        # And after the truncation step it drops to 100
        # But each update only appends once, so we expect exactly 100 after the cycle
        # (250 updates → at update 201 it becomes 201 → truncated to 100 → then continues
        # appending up to 200 → truncated to 100 again, etc.)
        # Final value is between 100 and 200
        assert 100 <= len(b._reward_history) <= 200


# ---------------------------------------------------------------------------
# Recompute selection_probability when not passed
# ---------------------------------------------------------------------------

class TestImplicitSelectionProbability:
    def test_uses_current_probs_when_none_passed(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        # Symmetric init: p(TCN) = 1/3
        b.update_reward("TCN", realized_pnl=50.0)  # no selection_probability
        # r_hat = tanh(1) / (1/3) = 0.7616 * 3 = 2.285
        # update = 0.10 * 2.285 / 3 = 0.0762
        # weight = exp(0.0762) ≈ 1.079 (before normalization)
        # After normalization to sum=3: 1.079/(1.079+1+1) * 3 ≈ 1.052
        # Just check it incremented
        assert b._weights["TCN"] > 1.0


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_state_writes_json(self, tmp_path):
        path = tmp_path / "bandit.json"
        b = ModelBanditSelector(persistence_path=path)
        b.update_reward("TCN", realized_pnl=100.0, selection_probability=0.5)
        b.save_state()
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert "weights" in data
        assert "rounds" in data
        assert "gamma" in data

    def test_save_state_creates_parent_dir(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "bandit.json"
        b = ModelBanditSelector(persistence_path=path)
        b.save_state()
        assert path.exists()

    def test_save_state_swallows_exceptions(self, tmp_path, monkeypatch):
        from pathlib import Path as _P
        path = tmp_path / "bandit.json"
        b = ModelBanditSelector(persistence_path=path)

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(_P, "write_text", _boom)
        # Must not raise
        b.save_state()

    def test_load_round_trip_preserves_state(self, tmp_path):
        path = tmp_path / "bandit.json"
        b1 = ModelBanditSelector(persistence_path=path)
        b1.update_reward("TCN", realized_pnl=100.0, selection_probability=0.5)
        b1.update_reward("Ridge", realized_pnl=-50.0, selection_probability=0.3)
        weights_before = dict(b1._weights)
        rounds_before = b1._rounds  # 0 — update doesn't increment rounds
        gamma_before = b1.gamma
        b1.save_state()

        # Reload into a fresh instance
        b2 = ModelBanditSelector(persistence_path=path)
        for m in b2.models:
            assert b2._weights[m] == pytest.approx(weights_before[m], abs=1e-5)
        assert b2.gamma == pytest.approx(gamma_before, abs=1e-5)
        assert b2._rounds == rounds_before

    def test_load_ignores_corrupt_json(self, tmp_path):
        path = tmp_path / "bandit.json"
        path.write_text("{ this is not valid json")
        # Must not crash; uses fresh state
        b = ModelBanditSelector(persistence_path=path)
        assert all(w == 1.0 for w in b._weights.values())

    def test_load_ignores_non_dict(self, tmp_path):
        path = tmp_path / "bandit.json"
        path.write_text(json.dumps([1, 2, 3]))
        b = ModelBanditSelector(persistence_path=path)
        assert all(w == 1.0 for w in b._weights.values())

    def test_load_only_picks_known_models(self, tmp_path):
        path = tmp_path / "bandit.json"
        path.write_text(json.dumps({
            "weights": {"TCN": 5.0, "Ridge": 2.0, "RF": 1.0, "GhostModel": 99.0},
            "rounds": 7,
            "gamma": 0.07,
        }))
        b = ModelBanditSelector(persistence_path=path)
        assert b._weights["TCN"] == 5.0
        assert "GhostModel" not in b._weights
        assert b._rounds == 7
        assert b.gamma == pytest.approx(0.07)


# ---------------------------------------------------------------------------
# Status & observability
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_keys(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        s = b.get_status()
        for key in ("rounds", "gamma", "weights", "probabilities",
                    "selection_counts", "avg_reward"):
            assert key in s

    def test_status_avg_reward_zero_when_no_history(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        s = b.get_status()
        for m in b.models:
            assert s["avg_reward"][m] == 0.0

    def test_status_reflects_updates(self, tmp_path):
        np.random.seed(0)
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        # Drive selections
        for _ in range(20):
            b.select_model()
        b.update_reward("TCN", realized_pnl=100.0, selection_probability=0.5)
        s = b.get_status()
        assert s["rounds"] == 20
        assert sum(s["selection_counts"].values()) == 20

    def test_get_model_weights_returns_copy(self, tmp_path):
        b = ModelBanditSelector(persistence_path=tmp_path / "bandit.json")
        w = b.get_model_weights()
        w["TCN"] = 999.0
        assert b._weights["TCN"] == 1.0  # original unchanged


# ---------------------------------------------------------------------------
# SelectionResult
# ---------------------------------------------------------------------------

class TestSelectionResult:
    def test_to_dict_rounds_floats(self):
        r = SelectionResult(
            selected_model="TCN",
            is_exploration=True,
            selection_probability=0.123456789,
            model_weights={"TCN": 1.234567},
            model_probabilities={"TCN": 0.555555},
        )
        d = r.to_dict()
        assert d["selected_model"] == "TCN"
        assert d["is_exploration"] is True
        assert d["selection_probability"] == 0.123457
        assert d["model_weights"]["TCN"] == 1.234567
        assert d["model_probabilities"]["TCN"] == 0.555555

    def test_default_serializes_safely(self):
        d = SelectionResult().to_dict()
        assert d["selected_model"] == ""
        assert d["is_exploration"] is False
        assert d["selection_probability"] == 0.0
