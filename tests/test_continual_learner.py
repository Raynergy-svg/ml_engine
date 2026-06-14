"""Tests for continual learner (US-018)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.autonomy.continual_learner import ContinualLearner, ContinualLearnerConfig


class TestContinualLearner:
    def test_init(self):
        cl = ContinualLearner()
        assert cl.cfg.memory_size == 500
        assert cl.get_memory_size() == 0

    def test_add_to_memory(self):
        cl = ContinualLearner(ContinualLearnerConfig(memory_size=10))
        cl.add_to_memory([{"x": i} for i in range(5)])
        assert cl.get_memory_size() == 5

    def test_memory_reservoir_sampling(self):
        cl = ContinualLearner(ContinualLearnerConfig(memory_size=5))
        cl.add_to_memory([{"x": i} for i in range(20)])
        assert cl.get_memory_size() == 5

    def test_prepare_training_batch(self):
        cl = ContinualLearner(ContinualLearnerConfig(memory_size=10, replay_ratio=0.5))
        cl.add_to_memory([{"x": i} for i in range(10)])
        new = [{"x": 100 + i} for i in range(10)]
        combined, replay = cl.prepare_training_batch(new)
        assert len(combined) > len(new)
        assert len(replay) > 0

    def test_ewc_penalty_no_history(self):
        cl = ContinualLearner()
        params = {"w": np.ones(3)}
        assert cl.compute_ewc_penalty(params) == 0.0

    def test_update_ewc_importance(self):
        cl = ContinualLearner()
        params = {"w": np.ones(3)}
        grads = {"w": np.ones(3) * 0.1}
        cl.update_ewc_importance(params, grads)
        assert cl._task_count == 1
        assert "w" in cl._ewc_importance

    def test_forgetting_metric(self):
        cl = ContinualLearner()
        assert cl.get_forgetting_metric(0.5, 0.7) == pytest.approx(0.2)
        assert cl.get_forgetting_metric(0.7, 0.5) == 0.0
