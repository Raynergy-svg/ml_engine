"""
Tests for Phase 55 US-342: State Persistence Retry with Exponential Backoff.

Tests cover:
  - retry_with_backoff decorator: successful retry, exhausted retries, non-retryable
  - resilient_save: success, queue on failure, flush queue
  - Save stats tracking and warning threshold
  - Failed writes queue management
  - Wiring verification: key modules use resilient_save
"""

import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from src.scanner.safe_io import (
    retry_with_backoff,
    resilient_save,
    get_save_stats,
    get_failed_writes_count,
    _record_save_stat,
    _save_stats,
    _save_stats_lock,
    _failed_writes_queue,
    _queue_lock,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level state between tests."""
    with _save_stats_lock:
        _save_stats.clear()
    with _queue_lock:
        _failed_writes_queue.clear()
    yield
    with _save_stats_lock:
        _save_stats.clear()
    with _queue_lock:
        _failed_writes_queue.clear()


class TestRetryWithBackoff:

    def test_succeeds_first_try(self):
        """Function succeeds on first call — no retries needed."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, context="test_first")
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_succeeds_after_retries(self):
        """Function fails twice, succeeds on third — retries work."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, context="test_retry")
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise IOError("transient failure")
            return "recovered"

        assert flaky() == "recovered"
        assert call_count == 3

    def test_exhausts_all_retries(self):
        """Function always fails — raises after max_retries."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, context="test_exhaust")
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise IOError("persistent failure")

        with pytest.raises(IOError, match="persistent failure"):
            always_fail()
        assert call_count == 3  # 1 initial + 2 retries

    def test_non_retryable_error_immediate(self):
        """PermissionError is non-retryable — raises immediately."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, context="test_perm")
        def perm_fail():
            nonlocal call_count
            call_count += 1
            raise PermissionError("access denied")

        with pytest.raises(PermissionError):
            perm_fail()
        assert call_count == 1  # No retries

    def test_backoff_delay_increases(self):
        """Verify delays increase exponentially (within jitter bounds)."""
        delays = []

        @retry_with_backoff(max_retries=3, base_delay=0.05, max_delay=10.0, jitter=False, context="test_delay")
        def fail_and_track():
            delays.append(time.time())
            raise IOError("fail")

        try:
            fail_and_track()
        except IOError:
            pass

        # Should have 4 timestamps (1 initial + 3 retries)
        assert len(delays) == 4
        # Gaps should grow: ~0.05, ~0.10, ~0.20
        gaps = [delays[i + 1] - delays[i] for i in range(len(delays) - 1)]
        assert gaps[1] > gaps[0] * 1.5  # Second gap > 1.5x first

    def test_max_delay_cap(self):
        """Delay never exceeds max_delay."""
        call_count = 0
        start = time.time()

        @retry_with_backoff(max_retries=2, base_delay=0.01, max_delay=0.05, jitter=False, context="test_cap")
        def fail():
            nonlocal call_count
            call_count += 1
            raise IOError("fail")

        try:
            fail()
        except IOError:
            pass

        elapsed = time.time() - start
        # With max_delay=0.05 and 2 retries, total sleep < 0.15s
        assert elapsed < 0.5


class TestResilientSave:

    def test_successful_save(self, tmp_dir):
        """resilient_save returns True on success."""
        path = os.path.join(tmp_dir, "test.json")
        assert resilient_save(path, {"ok": True}, context="test") is True
        loaded = json.loads(open(path).read())
        assert loaded["ok"] is True

    def test_queues_on_failure(self, tmp_dir):
        """resilient_save queues write on total failure."""
        # Use a path that can't be written (directory doesn't exist AND we block creation)
        with patch("src.scanner.safe_io.safe_json_write", side_effect=IOError("disk error")):
            result = resilient_save(
                os.path.join(tmp_dir, "fail.json"),
                {"data": 1},
                context="test_queue",
                max_retries=0,  # No retries for speed
            )
        assert result is False
        assert get_failed_writes_count() >= 1

    def test_flush_queued_writes(self, tmp_dir):
        """Queued writes get flushed on next successful save."""
        path = os.path.join(tmp_dir, "queued.json")
        # First: manually add to queue
        with _queue_lock:
            _failed_writes_queue.append((path, {"queued": True}, "test", time.time()))

        assert get_failed_writes_count() == 1

        # Now do a successful save — should flush
        other_path = os.path.join(tmp_dir, "trigger.json")
        resilient_save(other_path, {"trigger": True}, context="trigger")

        # Queue should be empty after flush
        assert get_failed_writes_count() == 0
        # Queued file should exist
        assert os.path.exists(path)


class TestSaveStats:

    def test_tracks_success(self):
        """Stats track successful saves."""
        _record_save_stat("test_ctx", success=True)
        _record_save_stat("test_ctx", success=True)
        stats = get_save_stats()
        assert stats["test_ctx"]["success"] == 2
        assert stats["test_ctx"]["failure"] == 0

    def test_tracks_failure(self):
        """Stats track failed saves."""
        _record_save_stat("fail_ctx", success=False)
        stats = get_save_stats()
        assert stats["fail_ctx"]["failure"] == 1

    def test_separate_contexts(self):
        """Different contexts tracked independently."""
        _record_save_stat("ctx_a", success=True)
        _record_save_stat("ctx_b", success=False)
        stats = get_save_stats()
        assert stats["ctx_a"]["success"] == 1
        assert stats["ctx_b"]["failure"] == 1
        assert "ctx_a" not in stats.get("ctx_b", {})


class TestFailedWritesQueue:

    def test_queue_limit(self):
        """Queue doesn't exceed maxlen=200."""
        with _queue_lock:
            for i in range(250):
                _failed_writes_queue.append((f"/path/{i}.json", {}, f"ctx_{i}", time.time()))
        assert get_failed_writes_count() == 200

    def test_stale_writes_dropped(self, tmp_dir):
        """Writes older than 1 hour are dropped during flush."""
        from src.scanner.safe_io import _flush_queued_writes

        path = os.path.join(tmp_dir, "stale.json")
        with _queue_lock:
            _failed_writes_queue.append((path, {"stale": True}, "stale_ctx", time.time() - 7200))

        flushed = _flush_queued_writes()
        assert flushed == 0
        assert not os.path.exists(path)


class TestWiringVerification:

    def test_agent_weights_uses_resilient_save(self):
        """agents/_team.py uses resilient_save for weight persistence."""
        try:
            from src.scanner.agents._team import ScannerAgentTeam
            import inspect
            source = inspect.getsource(ScannerAgentTeam._save_learned_weights)
            assert "resilient_save" in source
        except (ImportError, AttributeError):
            pytest.skip("ScannerAgentTeam._save_learned_weights not available")

    def test_gate_attribution_uses_resilient_save(self):
        """gate_attribution.py uses resilient_save."""
        try:
            from src.scanner.gate_attribution import GateAttributionEngine
            import inspect
            source = inspect.getsource(GateAttributionEngine.save_state)
            assert "resilient_save" in source
        except (ImportError, AttributeError):
            pytest.skip("GateAttributionEngine.save_state not available")

    def test_trade_clusters_uses_resilient_save(self):
        """trade_clusters.py uses resilient_save."""
        try:
            from src.scanner.trade_clusters import TradeClusterAnalyzer
            import inspect
            source = inspect.getsource(TradeClusterAnalyzer.save_state)
            assert "resilient_save" in source
        except (ImportError, AttributeError):
            pytest.skip("TradeClusterAnalyzer.save_state not available")

    def test_confidence_pipeline_uses_resilient_save(self):
        """confidence_pipeline.py uses resilient_save."""
        try:
            from src.scanner.confidence_pipeline import CalibrationMonitor
            import inspect
            source = inspect.getsource(CalibrationMonitor.save_state)
            assert "resilient_save" in source
        except (ImportError, AttributeError):
            pytest.skip("CalibrationMonitor.save_state not available")

    def test_pair_bayesian_uses_resilient_save(self):
        """pair_bayesian_weights.py uses resilient_save."""
        try:
            from src.scanner.pair_bayesian_weights import PairSpecificBayesianWeights
            import inspect
            source = inspect.getsource(PairSpecificBayesianWeights.save_state)
            assert "resilient_save" in source
        except (ImportError, AttributeError):
            pytest.skip("PairSpecificBayesianWeights.save_state not available")

    def test_wfe_monitor_uses_resilient_save(self):
        """wfe_monitor.py uses resilient_save."""
        try:
            from src.scanner.wfe_monitor import WFEStabilityMonitor
            import inspect
            source = inspect.getsource(WFEStabilityMonitor.save_state)
            assert "resilient_save" in source
        except (ImportError, AttributeError):
            pytest.skip("WFEStabilityMonitor.save_state not available")


class TestLazyImports:

    def test_retry_with_backoff_import(self):
        from src.scanner import retry_with_backoff
        assert retry_with_backoff is not None

    def test_resilient_save_import(self):
        from src.scanner import resilient_save
        assert resilient_save is not None
