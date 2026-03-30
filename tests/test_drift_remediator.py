"""Unit tests for DriftRemediator.

Tests cover all major code paths:
1. No retrain request (SKIPPED_NO_REQUEST)
2. In cooldown (SKIPPED_COOLDOWN)
3. Current model validates (FALSE_ALARM)
4. Retrain succeeds, validation passes (REMEDIATED)
5. Retrain fails (FAILED)
6. Retrain subprocess timeout (FAILED with TIMEOUT reason)
7. Validation metrics extraction (accuracy, OOS/IS ratio)
8. Exception handling (never raises, returns error result)
"""

import json
import logging
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.scanner.automation.drift_remediator import DriftRemediationResult, DriftRemediator

logger = logging.getLogger(__name__)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for test paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "trained_data").mkdir(parents=True)
        (tmppath / "trained_data" / "models").mkdir(parents=True)
        (tmppath / "logs").mkdir(parents=True)
        yield tmppath


@pytest.fixture
def remediator(temp_dirs):
    """Create a DriftRemediator with temp paths."""
    return DriftRemediator(
        request_path=temp_dirs / "trained_data" / "retrain_request.json",
        remediation_log_path=temp_dirs / "trained_data" / "drift_remediation_log.jsonl",
        retrain_summary_path=temp_dirs / "trained_data" / "retrain_all_summary.json",
        agent_weights_path=temp_dirs / "trained_data" / "models" / "agent_weights.json",
        cooldown_hours=12,
        retrain_timeout_minutes=30,
        project_root=temp_dirs,
    )


class TestDriftRemediationResult:
    """Test DriftRemediationResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dict."""
        result = DriftRemediationResult(
            action_taken="REMEDIATED",
            success=True,
            pairs_retrained=["EUR_USD", "GBP_USD"],
            reason="Test remediation",
            duration_seconds=123.45,
            validation_metrics={"accuracy": 0.52, "oos_is_ratio": 0.65},
        )
        d = result.to_dict()
        assert d["action_taken"] == "REMEDIATED"
        assert d["success"] is True
        assert d["pairs_retrained"] == ["EUR_USD", "GBP_USD"]
        assert d["duration_seconds"] == 123.45


class TestNoRetrainRequest:
    """Test case: no retrain_request.json present."""

    def test_skipped_no_request(self, remediator):
        """When no retrain request exists, should return SKIPPED_NO_REQUEST."""
        # No request file created
        result = remediator.check_and_remediate()

        assert result.action_taken == "SKIPPED_NO_REQUEST"
        assert result.success is False
        assert "No pending retrain request" in result.reason
        assert result.pairs_retrained == []


class TestCooldownActive:
    """Test case: remediator is in cooldown period."""

    def test_skipped_cooldown(self, remediator, temp_dirs):
        """When in cooldown, should skip remediation."""
        # Create retrain request
        request = {
            "pairs": ["EUR_USD"],
            "reason": "Drift detected",
            "accuracy_snapshot": {"EUR_USD": {"accuracy": 0.45}},
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
        }
        with open(remediator.request_path, "w") as f:
            json.dump(request, f)

        # Write a recent REMEDIATION_FAILED event to trigger cooldown
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
        failed_event = {
            "timestamp": recent_time.isoformat(),
            "event_type": "REMEDIATION_FAILED",
            "details": {"reason": "Test failure"},
        }
        with open(remediator.remediation_log_path, "w") as f:
            f.write(json.dumps(failed_event) + "\n")

        result = remediator.check_and_remediate()

        assert result.action_taken == "SKIPPED_COOLDOWN"
        assert result.success is False
        assert "Cooldown period active" in result.reason


class TestFalseAlarm:
    """Test case: current model still validates (false alarm)."""

    def test_false_alarm_detection(self, remediator, temp_dirs):
        """When current model still validates, should detect false alarm."""
        # Create retrain request
        request = {
            "pairs": ["EUR_USD"],
            "reason": "Drift detected",
            "accuracy_snapshot": {"EUR_USD": {"accuracy": 0.45}},
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
        }
        with open(remediator.request_path, "w") as f:
            json.dump(request, f)

        # Create a valid retrain summary (current model passes validation)
        summary = {
            "accuracy": 0.55,
            "oos_is_ratio": 0.65,
        }
        with open(remediator.retrain_summary_path, "w") as f:
            json.dump(summary, f)

        result = remediator.check_and_remediate()

        assert result.action_taken == "FALSE_ALARM"
        assert result.success is True
        assert "Current model still validates" in result.reason
        assert result.validation_metrics.get("oos_is_ratio") == 0.65

        # Request should be cleared
        assert not remediator.request_path.exists()


class TestRemediationSuccess:
    """Test case: retrain succeeds and validates."""

    @patch("src.scanner.automation.drift_remediator.subprocess.run")
    def test_remediation_success(self, mock_run, remediator, temp_dirs):
        """When retrain subprocess succeeds and validation passes, should remediate."""
        # Create retrain request
        request = {
            "pairs": ["EUR_USD", "GBP_USD"],
            "reason": "Drift detected",
            "accuracy_snapshot": {"EUR_USD": {"accuracy": 0.45}, "GBP_USD": {"accuracy": 0.48}},
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
        }
        with open(remediator.request_path, "w") as f:
            json.dump(request, f)

        # Mock subprocess to succeed
        mock_run.return_value = MagicMock(returncode=0, stdout="Retrain successful", stderr="")

        # Create initial (failing) summary to trigger actual retrain
        failing_summary = {"accuracy": 0.48, "oos_is_ratio": 0.55}
        with open(remediator.retrain_summary_path, "w") as f:
            json.dump(failing_summary, f)

        # Create dummy script file to pass existence check
        retrain_script = temp_dirs / "scripts" / "scheduled_retrain.py"
        retrain_script.parent.mkdir(parents=True)
        retrain_script.write_text("# dummy")

        # Mock the validation to return success on second call (after retrain)
        with patch.object(remediator, "_validate_retrain_result") as mock_validate:
            mock_validate.return_value = (True, {"accuracy": 0.54, "oos_is_ratio": 0.65})

            result = remediator.check_and_remediate()

        assert result.action_taken == "REMEDIATED"
        assert result.success is True
        assert set(result.pairs_retrained) == {"EUR_USD", "GBP_USD"}
        assert "validation passed" in result.reason
        assert result.validation_metrics.get("oos_is_ratio") == 0.65

        # Request should be cleared
        assert not remediator.request_path.exists()

        # Event log should contain SUCCESS
        assert remediator.remediation_log_path.exists()


class TestRemediationFailureSubprocess:
    """Test case: retrain subprocess fails."""

    @patch("src.scanner.automation.drift_remediator.subprocess.run")
    def test_remediation_failure_subprocess(self, mock_run, remediator, temp_dirs):
        """When retrain subprocess fails, should mark as FAILED."""
        # Create retrain request
        request = {
            "pairs": ["EUR_USD"],
            "reason": "Drift detected",
            "accuracy_snapshot": {"EUR_USD": {"accuracy": 0.45}},
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
        }
        with open(remediator.request_path, "w") as f:
            json.dump(request, f)

        # Mock subprocess to fail
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Retrain failed")

        # Create dummy script file
        retrain_script = temp_dirs / "scripts" / "scheduled_retrain.py"
        retrain_script.parent.mkdir(parents=True)
        retrain_script.write_text("# dummy")

        # Create failing summary to trigger actual retrain
        failing_summary = {"accuracy": 0.48, "oos_is_ratio": 0.55}
        with open(remediator.retrain_summary_path, "w") as f:
            json.dump(failing_summary, f)

        result = remediator.check_and_remediate()

        assert result.action_taken == "FAILED"
        assert result.success is False
        assert "exited with code 1" in result.reason


class TestRemediationTimeoutSubprocess:
    """Test case: retrain subprocess times out."""

    @patch("src.scanner.automation.drift_remediator.subprocess.run")
    def test_remediation_timeout(self, mock_run, remediator, temp_dirs):
        """When retrain subprocess times out, should mark as FAILED."""
        import subprocess as subprocess_module

        # Create retrain request
        request = {
            "pairs": ["EUR_USD"],
            "reason": "Drift detected",
            "accuracy_snapshot": {"EUR_USD": {"accuracy": 0.45}},
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
        }
        with open(remediator.request_path, "w") as f:
            json.dump(request, f)

        # Mock subprocess to timeout
        mock_run.side_effect = subprocess_module.TimeoutExpired(cmd="test", timeout=30)

        # Create dummy script file
        retrain_script = temp_dirs / "scripts" / "scheduled_retrain.py"
        retrain_script.parent.mkdir(parents=True)
        retrain_script.write_text("# dummy")

        # Create failing summary to trigger actual retrain
        failing_summary = {"accuracy": 0.48, "oos_is_ratio": 0.55}
        with open(remediator.retrain_summary_path, "w") as f:
            json.dump(failing_summary, f)

        result = remediator.check_and_remediate()

        assert result.action_taken == "FAILED"
        assert result.success is False
        assert "timeout" in result.reason.lower()


class TestRemediationFailureValidation:
    """Test case: retrain succeeds but validation fails."""

    @patch("src.scanner.automation.drift_remediator.subprocess.run")
    def test_remediation_failure_validation(self, mock_run, remediator, temp_dirs):
        """When retrain succeeds but validation fails, should mark as FAILED."""
        # Create retrain request
        request = {
            "pairs": ["EUR_USD"],
            "reason": "Drift detected",
            "accuracy_snapshot": {"EUR_USD": {"accuracy": 0.45}},
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
        }
        with open(remediator.request_path, "w") as f:
            json.dump(request, f)

        # Mock subprocess to succeed
        mock_run.return_value = MagicMock(returncode=0, stdout="Retrain successful", stderr="")

        # Create dummy script file
        retrain_script = temp_dirs / "scripts" / "scheduled_retrain.py"
        retrain_script.parent.mkdir(parents=True)
        retrain_script.write_text("# dummy")

        # Create failing summary
        failing_summary = {"accuracy": 0.48, "oos_is_ratio": 0.55}
        with open(remediator.retrain_summary_path, "w") as f:
            json.dump(failing_summary, f)

        # Mock validation to fail
        with patch.object(remediator, "_validate_retrain_result") as mock_validate:
            mock_validate.return_value = (False, {"accuracy": 0.48, "oos_is_ratio": 0.55})

            result = remediator.check_and_remediate()

        assert result.action_taken == "FAILED"
        assert result.success is False
        assert "validation failed" in result.reason


class TestValidateRetrainResult:
    """Test case: validation metrics extraction."""

    def test_validate_retrain_result_passes(self, remediator):
        """Test successful retrain result validation."""
        summary = {
            "accuracy": 0.54,
            "oos_is_ratio": 0.65,
        }
        with open(remediator.retrain_summary_path, "w") as f:
            json.dump(summary, f)

        passes, metrics = remediator._validate_retrain_result()

        assert passes is True
        assert metrics["accuracy"] == 0.54
        assert metrics["oos_is_ratio"] == 0.65

    def test_validate_retrain_result_fails_accuracy(self, remediator):
        """Test retrain result validation fails on low accuracy."""
        summary = {
            "accuracy": 0.50,  # Below 0.52 threshold
            "oos_is_ratio": 0.65,
        }
        with open(remediator.retrain_summary_path, "w") as f:
            json.dump(summary, f)

        passes, metrics = remediator._validate_retrain_result()

        assert passes is False
        assert metrics["accuracy"] == 0.50

    def test_validate_retrain_result_fails_oos_is_ratio(self, remediator):
        """Test retrain result validation fails on low OOS/IS ratio."""
        summary = {
            "accuracy": 0.54,
            "oos_is_ratio": 0.55,  # Below 0.60 threshold
        }
        with open(remediator.retrain_summary_path, "w") as f:
            json.dump(summary, f)

        passes, metrics = remediator._validate_retrain_result()

        assert passes is False
        assert metrics["oos_is_ratio"] == 0.55

    def test_validate_retrain_result_no_summary(self, remediator):
        """Test validation when no summary file exists."""
        passes, metrics = remediator._validate_retrain_result()

        assert passes is False
        assert "reason" in metrics


class TestRemediationEventLogging:
    """Test case: remediation event logging to JSONL."""

    def test_write_remediation_event(self, remediator):
        """Test writing remediation events to log."""
        remediator._write_remediation_event(
            "REMEDIATION_STARTED",
            {"pairs": ["EUR_USD"]},
        )
        remediator._write_remediation_event(
            "REMEDIATION_SUCCESS",
            {"accuracy": 0.54, "oos_is_ratio": 0.65},
        )

        assert remediator.remediation_log_path.exists()

        # Read and verify events
        with open(remediator.remediation_log_path, "r") as f:
            lines = f.readlines()

        assert len(lines) >= 2
        first_event = json.loads(lines[0])
        assert first_event["event_type"] == "REMEDIATION_STARTED"

        second_event = json.loads(lines[1])
        assert second_event["event_type"] == "REMEDIATION_SUCCESS"
        assert second_event["details"]["oos_is_ratio"] == 0.65


class TestExceptionHandling:
    """Test case: exception handling (never raises)."""

    def test_exception_in_read_request(self, remediator):
        """Test that corrupted request file doesn't crash."""
        # Write corrupted JSON
        remediator.request_path.write_text("not valid json {{{")

        result = remediator.check_and_remediate()

        assert result.action_taken == "SKIPPED_NO_REQUEST"
        assert result.success is False

    def test_exception_in_check_and_remediate(self, remediator):
        """Test that unexpected exceptions are caught."""
        # Create request
        request = {
            "pairs": ["EUR_USD"],
            "reason": "Drift detected",
            "accuracy_snapshot": {"EUR_USD": {"accuracy": 0.45}},
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
        }
        with open(remediator.request_path, "w") as f:
            json.dump(request, f)

        # Mock a method to raise exception
        with patch.object(remediator, "_is_in_cooldown") as mock_cooldown:
            mock_cooldown.side_effect = RuntimeError("Unexpected error")

            result = remediator.check_and_remediate()

        assert result.action_taken == "FAILED"
        assert result.success is False
        assert "Unexpected error" in result.reason


class TestDurationTracking:
    """Test case: duration tracking."""

    @patch("src.scanner.automation.drift_remediator.subprocess.run")
    def test_duration_tracking(self, mock_run, remediator, temp_dirs):
        """Test that remediation duration is tracked."""
        import time as time_module

        # Create retrain request
        request = {
            "pairs": ["EUR_USD"],
            "reason": "Drift detected",
            "accuracy_snapshot": {"EUR_USD": {"accuracy": 0.45}},
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
        }
        with open(remediator.request_path, "w") as f:
            json.dump(request, f)

        # Mock subprocess with a small delay
        def slow_run(*args, **kwargs):
            time_module.sleep(0.1)
            return MagicMock(returncode=0, stdout="Success", stderr="")

        mock_run.side_effect = slow_run

        # Create dummy script file
        retrain_script = temp_dirs / "scripts" / "scheduled_retrain.py"
        retrain_script.parent.mkdir(parents=True)
        retrain_script.write_text("# dummy")

        # Create failing summary
        failing_summary = {"accuracy": 0.48, "oos_is_ratio": 0.55}
        with open(remediator.retrain_summary_path, "w") as f:
            json.dump(failing_summary, f)

        with patch.object(remediator, "_validate_retrain_result") as mock_validate:
            mock_validate.return_value = (True, {"accuracy": 0.54, "oos_is_ratio": 0.65})

            result = remediator.check_and_remediate()

        assert result.duration_seconds >= 0.1
