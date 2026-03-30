"""Tests for Phase 57 US-355: DriftProxyGuard.

Covers:
  - compute_proxies uses pre-penalty confidence (not post)
  - prediction_error = abs(pre - 0.5)
  - agent_disagreement = 1.0 - pre
  - is_self_referential fires when delta > threshold
  - is_self_referential False when delta small
  - Both confidence values stored correctly in DriftProxies
  - get_stats tracks self-referential count
  - compute_proxies with equal pre/post never self-referential
"""

import pytest

from src.scanner.drift_proxy_guard import DriftProxyGuard, DriftProxies, SELF_REFERENCE_THRESHOLD


class TestComputeProxies:

    def test_proxies_use_pre_penalty_confidence(self):
        """Proxies must be computed from pre_penalty, not post_penalty."""
        guard = DriftProxyGuard()
        proxies = guard.compute_proxies(pre_penalty_confidence=0.70, post_penalty_confidence=0.55)
        # prediction_error based on pre=0.70
        assert proxies.prediction_error == pytest.approx(abs(0.70 - 0.5), abs=0.001)
        assert proxies.agent_disagreement == pytest.approx(1.0 - 0.70, abs=0.001)
        assert proxies.source_confidence == pytest.approx(0.70, abs=0.001)

    def test_prediction_error_formula(self):
        guard = DriftProxyGuard()
        proxies = guard.compute_proxies(0.60, 0.60)
        assert proxies.prediction_error == pytest.approx(0.10, abs=0.001)

    def test_agent_disagreement_formula(self):
        guard = DriftProxyGuard()
        proxies = guard.compute_proxies(0.75, 0.75)
        assert proxies.agent_disagreement == pytest.approx(0.25, abs=0.001)

    def test_high_confidence_low_disagreement(self):
        guard = DriftProxyGuard()
        proxies = guard.compute_proxies(0.90, 0.90)
        assert proxies.agent_disagreement == pytest.approx(0.10, abs=0.001)

    def test_proxies_clamped_to_valid_range(self):
        """Confidence values outside 0-1 are clamped."""
        guard = DriftProxyGuard()
        proxies = guard.compute_proxies(pre_penalty_confidence=1.5, post_penalty_confidence=0.5)
        assert proxies.prediction_error == pytest.approx(abs(1.0 - 0.5), abs=0.001)


class TestIsSelfReferential:

    def test_self_referential_fires_when_delta_large(self):
        guard = DriftProxyGuard(self_reference_threshold=0.10)
        assert guard.is_self_referential(0.70, 0.55) is True  # delta=0.15 > 0.10

    def test_self_referential_false_when_delta_small(self):
        guard = DriftProxyGuard(self_reference_threshold=0.10)
        assert guard.is_self_referential(0.70, 0.65) is False  # delta=0.05 < 0.10

    def test_self_referential_exactly_at_threshold_not_triggered(self):
        """Delta == threshold → NOT self-referential (strict >)."""
        guard = DriftProxyGuard(self_reference_threshold=0.10)
        assert guard.is_self_referential(0.70, 0.60) is False  # delta=0.10 == threshold

    def test_equal_pre_post_never_self_referential(self):
        guard = DriftProxyGuard()
        assert guard.is_self_referential(0.65, 0.65) is False

    def test_compute_proxies_sets_is_self_referential_flag(self):
        guard = DriftProxyGuard(self_reference_threshold=0.10)
        proxies = guard.compute_proxies(0.72, 0.55)  # delta=0.17
        assert proxies.is_self_referential is True

    def test_compute_proxies_self_ref_false_when_small_delta(self):
        guard = DriftProxyGuard(self_reference_threshold=0.10)
        proxies = guard.compute_proxies(0.72, 0.70)  # delta=0.02
        assert proxies.is_self_referential is False


class TestGetStats:

    def test_stats_track_self_referential_count(self):
        guard = DriftProxyGuard(self_reference_threshold=0.10)
        guard.compute_proxies(0.72, 0.55)  # self-referential
        guard.compute_proxies(0.72, 0.70)  # not self-referential
        stats = guard.get_stats()
        assert stats["total_checks"] == 2
        assert stats["self_referential_count"] == 1
        assert stats["self_referential_rate"] == pytest.approx(0.5, abs=0.01)
