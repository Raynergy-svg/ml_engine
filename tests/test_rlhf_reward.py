"""Tests for RLHF reward shaping (US-011)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.rl.rlhf_reward import RLHFConfig, RLHFRewardShaper


class TestRLHFRewardShaper:
    def test_init(self):
        shaper = RLHFRewardShaper()
        assert shaper.cfg.state_dim == 20
        assert not shaper._trained

    def test_add_comparison(self):
        shaper = RLHFRewardShaper(RLHFConfig(state_dim=4))
        shaper.add_comparison(np.ones(4), np.zeros(4), "A")
        assert len(shaper._comparisons) == 1

    def test_fit_insufficient_data(self):
        shaper = RLHFRewardShaper(RLHFConfig(min_comparisons=10))
        shaper.add_comparison(np.ones(4), np.zeros(4), "A")
        shaper.fit()
        assert not shaper._trained

    def test_shape_reward_fallback(self):
        shaper = RLHFRewardShaper()
        r = shaper.shape_reward(np.ones(20), base_reward=1.0)
        assert r == pytest.approx(1.0)

    def test_fit_and_shape(self):
        cfg = RLHFConfig(state_dim=4, min_comparisons=5, epochs=20)
        shaper = RLHFRewardShaper(cfg)
        np.random.seed(0)
        for i in range(20):
            sa = np.random.randn(4)
            sb = np.random.randn(4)
            # Prefer states with higher sum
            pref = "A" if sa.sum() > sb.sum() else "B"
            shaper.add_comparison(sa, sb, pref)
        shaper.fit()
        assert shaper._trained is True
        shaped = shaper.shape_reward(np.ones(4), base_reward=1.0)
        assert shaped != pytest.approx(1.0)

    def test_preference_probability(self):
        cfg = RLHFConfig(state_dim=4, min_comparisons=5, epochs=20)
        shaper = RLHFRewardShaper(cfg)
        np.random.seed(0)
        for i in range(20):
            sa = np.random.randn(4)
            sb = np.random.randn(4)
            pref = "A" if sa.sum() > sb.sum() else "B"
            shaper.add_comparison(sa, sb, pref)
        shaper.fit()
        prob = shaper.get_preference_probability(np.ones(4), np.zeros(4))
        assert 0.0 <= prob <= 1.0
