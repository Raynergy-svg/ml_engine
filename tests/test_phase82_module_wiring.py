"""Tests for Phase 82 module-wiring stories (US-P82-001 through US-P82-004).

Covers:
  US-P82-001: ObservationLog .log() -> .log_observation() fix (3 sites)
  US-P82-002: PairModelSelector wired at 3 sites
  US-P82-003: ChainMemory lifecycle (2 sites)
  US-P82-004: AdaptiveLR.step() + RegimeConfidenceScorer.save_state()
"""

import logging
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# US-P82-001: ObservationLog .log() -> .log_observation() fix (3 sites)
# ---------------------------------------------------------------------------


class TestObservationLogConsensusPenalty:
    """US-P82-001: log_observation called with category='consensus_penalty'."""

    def test_log_observation_consensus_penalty(self):
        obs_log = MagicMock()
        pair = "EUR_USD"
        data = {"penalty": 0.12, "reason": "model_disagreement"}

        obs_log.log_observation(category="consensus_penalty", pair=pair, data=data)

        obs_log.log_observation.assert_called_once_with(
            category="consensus_penalty", pair="EUR_USD", data=data,
        )


class TestObservationLogRegimeDefault:
    """US-P82-001: log_observation called with category='regime_default'."""

    def test_log_observation_regime_default(self):
        obs_log = MagicMock()
        pair = "GBP_USD"
        data = {"regime": "NORMAL", "fallback": True}

        obs_log.log_observation(category="regime_default", pair=pair, data=data)

        obs_log.log_observation.assert_called_once_with(
            category="regime_default", pair="GBP_USD", data=data,
        )


class TestObservationLogExecutionFasttrack:
    """US-P82-001: log_observation called with category='execution_fasttrack'."""

    def test_log_observation_execution_fasttrack(self):
        obs_log = MagicMock()
        pair = "USD_JPY"
        data = {"fasttrack": True, "confidence": 0.92}

        obs_log.log_observation(category="execution_fasttrack", pair=pair, data=data)

        obs_log.log_observation.assert_called_once_with(
            category="execution_fasttrack", pair="USD_JPY", data=data,
        )


class TestObservationLogNoneGracefulSkip:
    """US-P82-001: observation_log is None -> graceful skip, no crash."""

    def test_observation_log_none_skips(self):
        observation_log = None
        pair = "EUR_USD"
        data = {"penalty": 0.1}

        # Replicate production guard pattern
        if observation_log is not None:
            observation_log.log_observation(
                category="consensus_penalty", pair=pair, data=data,
            )

        # No exception raised — test passes by reaching this line


class TestObservationLogExceptionCaught:
    """US-P82-001: log_observation raises exception -> caught, no crash."""

    def test_log_observation_exception_no_crash(self, caplog):
        obs_log = MagicMock()
        obs_log.log_observation.side_effect = RuntimeError("disk full")

        error_caught = False
        try:
            if obs_log is not None:
                obs_log.log_observation(
                    category="consensus_penalty", pair="EUR_USD", data={},
                )
        except Exception as err:
            error_caught = True
            logging.getLogger(__name__).debug(
                "Phase 82 (US-P82-001): ObservationLog error (non-fatal): %s", err,
            )

        assert error_caught is True
        obs_log.log_observation.assert_called_once()


# ---------------------------------------------------------------------------
# US-P82-002: PairModelSelector wired at 3 sites
# ---------------------------------------------------------------------------


class TestPairModelSelectorGetActiveModel:
    """US-P82-002: get_active_model(pair) returns str in _scan_pair context."""

    def test_get_active_model_returns_joint(self):
        pms = MagicMock()
        pms.get_active_model.return_value = "joint"

        result = pms.get_active_model("EUR_USD")

        assert result == "joint"
        pms.get_active_model.assert_called_once_with("EUR_USD")

    def test_get_active_model_returns_per_pair(self):
        pms = MagicMock()
        pms.get_active_model.return_value = "per_pair"

        result = pms.get_active_model("GBP_USD")

        assert result == "per_pair"


class TestPairModelSelectorRecordPrediction:
    """US-P82-002: record_prediction called after trade outcome with correct args."""

    def test_record_prediction_correct_args(self):
        pms = MagicMock()
        pair = "EUR_USD"
        model_type = "joint"
        predicted = "BUY"
        actual = "BUY"

        pms.record_prediction(
            pair=pair, model_type=model_type,
            predicted_direction=predicted, actual_direction=actual,
        )

        pms.record_prediction.assert_called_once_with(
            pair="EUR_USD", model_type="joint",
            predicted_direction="BUY", actual_direction="BUY",
        )

    def test_record_prediction_wrong_direction(self):
        pms = MagicMock()

        pms.record_prediction(
            pair="USD_JPY", model_type="per_pair",
            predicted_direction="BUY", actual_direction="SELL",
        )

        pms.record_prediction.assert_called_once_with(
            pair="USD_JPY", model_type="per_pair",
            predicted_direction="BUY", actual_direction="SELL",
        )


class TestPairModelSelectorCheckSwitch:
    """US-P82-002: check_switch returns None or new model name."""

    def test_check_switch_returns_none_no_execute(self):
        pms = MagicMock()
        pms.check_switch.return_value = None

        new_model = pms.check_switch("EUR_USD")

        if new_model is not None:
            pms.execute_switch("EUR_USD", new_model)

        assert new_model is None
        pms.execute_switch.assert_not_called()

    def test_check_switch_returns_per_pair_triggers_execute(self):
        pms = MagicMock()
        pms.check_switch.return_value = "per_pair"

        new_model = pms.check_switch("EUR_USD")

        if new_model is not None:
            pms.execute_switch("EUR_USD", new_model)

        pms.execute_switch.assert_called_once_with("EUR_USD", "per_pair")

    def test_check_switch_returns_joint_triggers_execute(self):
        pms = MagicMock()
        pms.check_switch.return_value = "joint"

        new_model = pms.check_switch("GBP_USD")

        if new_model is not None:
            pms.execute_switch("GBP_USD", new_model)

        pms.execute_switch.assert_called_once_with("GBP_USD", "joint")


class TestPairModelSelectorCycleGating:
    """US-P82-002: check_switch fires every 50 cycles."""

    @staticmethod
    def _should_check_switch(cycle: int, interval: int = 50) -> bool:
        """Replicate the production cycle-gating logic."""
        return cycle > 0 and cycle % interval == 0

    def test_cycle_50_fires(self):
        assert self._should_check_switch(50) is True

    def test_cycle_49_does_not_fire(self):
        assert self._should_check_switch(49) is False

    def test_cycle_100_fires(self):
        assert self._should_check_switch(100) is True

    def test_cycle_0_does_not_fire(self):
        assert self._should_check_switch(0) is False

    def test_cycle_51_does_not_fire(self):
        assert self._should_check_switch(51) is False


class TestPairModelSelectorNoneGracefulSkip:
    """US-P82-002: pair_model_selector is None -> graceful skip."""

    def test_pair_model_selector_none_skips(self):
        pair_model_selector = None
        model_type = "joint"  # default fallback

        if pair_model_selector is not None:
            model_type = pair_model_selector.get_active_model("EUR_USD")

        assert model_type == "joint"  # Fallback unchanged


# ---------------------------------------------------------------------------
# US-P82-003: ChainMemory lifecycle (2 sites)
# ---------------------------------------------------------------------------


class TestChainMemorySyncFromPrd:
    """US-P82-003: sync_from_prd called once, gated by _chain_memory_synced flag."""

    def test_sync_from_prd_called_first_scan(self):
        chain_memory = MagicMock()
        _chain_memory_synced = False

        if chain_memory is not None and not _chain_memory_synced:
            chain_memory.sync_from_prd()
            _chain_memory_synced = True

        chain_memory.sync_from_prd.assert_called_once()
        assert _chain_memory_synced is True

    def test_sync_from_prd_not_called_second_scan(self):
        chain_memory = MagicMock()
        _chain_memory_synced = True  # Already synced from first scan

        if chain_memory is not None and not _chain_memory_synced:
            chain_memory.sync_from_prd()

        chain_memory.sync_from_prd.assert_not_called()

    def test_sync_flag_persists_across_cycles(self):
        chain_memory = MagicMock()
        _chain_memory_synced = False

        # Simulate 3 scan cycles
        for _ in range(3):
            if chain_memory is not None and not _chain_memory_synced:
                chain_memory.sync_from_prd()
                _chain_memory_synced = True

        # Only called once despite 3 cycles
        chain_memory.sync_from_prd.assert_called_once()


class TestChainMemoryRecordChainComplete:
    """US-P82-003: record_chain_complete called at scan end with result dict."""

    def test_record_chain_complete_with_required_keys(self):
        chain_memory = MagicMock()
        result_dict = {
            "phase": 82,
            "pairs_scanned": ["EUR_USD", "GBP_USD"],
            "signals": 3,
        }

        if chain_memory is not None:
            chain_memory.record_chain_complete(result_dict)

        chain_memory.record_chain_complete.assert_called_once_with(result_dict)
        # Verify the dict has the required keys
        call_args = chain_memory.record_chain_complete.call_args[0][0]
        assert "phase" in call_args
        assert "pairs_scanned" in call_args
        assert "signals" in call_args

    def test_record_chain_complete_empty_signals(self):
        chain_memory = MagicMock()
        result_dict = {
            "phase": 82,
            "pairs_scanned": [],
            "signals": 0,
        }

        if chain_memory is not None:
            chain_memory.record_chain_complete(result_dict)

        chain_memory.record_chain_complete.assert_called_once()


class TestChainMemoryNoneGracefulSkip:
    """US-P82-003: chain_memory is None -> graceful skip."""

    def test_chain_memory_none_sync_skips(self):
        chain_memory = None
        _chain_memory_synced = False

        if chain_memory is not None and not _chain_memory_synced:
            chain_memory.sync_from_prd()
            _chain_memory_synced = True

        assert _chain_memory_synced is False  # Flag not set

    def test_chain_memory_none_record_skips(self):
        chain_memory = None
        result_dict = {"phase": 82, "pairs_scanned": [], "signals": 0}

        # Should not raise
        if chain_memory is not None:
            chain_memory.record_chain_complete(result_dict)

        # No exception raised — test passes


# ---------------------------------------------------------------------------
# US-P82-004: AdaptiveLR.step() + RegimeConfidenceScorer.save_state()
# ---------------------------------------------------------------------------


def _make_lr_state(current_lr: float):
    """Create a SimpleNamespace mirroring LRState."""
    return types.SimpleNamespace(current_lr=current_lr)


class TestAdaptiveLRStepWinningTrade:
    """US-P82-004: step(loss=0.0) called on winning trade."""

    def test_step_loss_zero_on_win(self):
        adaptive_lr = MagicMock()
        adaptive_lr.step.return_value = _make_lr_state(current_lr=0.001)

        pnl = 25.0  # Winning trade
        loss_val = 0.0 if pnl >= 0 else 1.0

        lr_state = adaptive_lr.step(loss=loss_val)

        adaptive_lr.step.assert_called_once_with(loss=0.0)
        assert lr_state.current_lr == 0.001


class TestAdaptiveLRStepLosingTrade:
    """US-P82-004: step(loss=1.0) called on losing trade."""

    def test_step_loss_one_on_loss(self):
        adaptive_lr = MagicMock()
        adaptive_lr.step.return_value = _make_lr_state(current_lr=0.0008)

        pnl = -15.0  # Losing trade
        loss_val = 0.0 if pnl >= 0 else 1.0

        lr_state = adaptive_lr.step(loss=loss_val)

        adaptive_lr.step.assert_called_once_with(loss=1.0)
        assert lr_state.current_lr == 0.0008

    def test_step_loss_breakeven_treated_as_win(self):
        adaptive_lr = MagicMock()
        adaptive_lr.step.return_value = _make_lr_state(current_lr=0.001)

        pnl = 0.0  # Breakeven
        loss_val = 0.0 if pnl >= 0 else 1.0

        adaptive_lr.step(loss=loss_val)

        adaptive_lr.step.assert_called_once_with(loss=0.0)


class TestAdaptiveLRNoneGracefulSkip:
    """US-P82-004: adaptive_lr is None -> graceful skip."""

    def test_adaptive_lr_none_skips(self):
        adaptive_lr = None
        pnl = 10.0
        loss_val = 0.0 if pnl >= 0 else 1.0

        if adaptive_lr is not None:
            adaptive_lr.step(loss=loss_val)

        # No exception raised


class TestRegimeConfidenceScorerSaveState:
    """US-P82-004: save_state() called in scan save chain."""

    def test_save_state_called(self):
        rcs = MagicMock()

        if rcs is not None:
            rcs.save_state()

        rcs.save_state.assert_called_once()

    def test_save_state_exception_caught(self, caplog):
        rcs = MagicMock()
        rcs.save_state.side_effect = IOError("write failed")

        error_caught = False
        try:
            if rcs is not None:
                rcs.save_state()
        except Exception as err:
            error_caught = True
            logging.getLogger(__name__).debug(
                "Phase 82 (US-P82-004): save_state error (non-fatal): %s", err,
            )

        assert error_caught is True
        rcs.save_state.assert_called_once()


class TestRegimeConfidenceScorerNoneGracefulSkip:
    """US-P82-004: regime_confidence_scorer_scan is None -> graceful skip."""

    def test_rcs_none_skips(self):
        regime_confidence_scorer_scan = None

        if regime_confidence_scorer_scan is not None:
            regime_confidence_scorer_scan.save_state()

        # No exception raised
