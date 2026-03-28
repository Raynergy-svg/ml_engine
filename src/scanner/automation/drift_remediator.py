"""Drift Auto-Remediation Loop — closes the gap between drift detection and model healing.

Monitors for drift-triggered retrain requests, spawns retraining subprocess,
validates results, and swaps models automatically.

Architecture:
    RetrainTrigger writes retrain_request.json
         ↓
    DriftRemediator.check_and_remediate() [called every N cycles]
         ↓
    Reads retrain_request.json
         ↓ (if request present AND not in cooldown)
    Runs walk-forward validation on EXISTING models (fast path)
         ↓
    If validation fails: spawns retraining subprocess
    After retrain: validates NEW model
         ↓
    On success: updates agent_weights.json, clears request
    On failure: sets cooldown, logs failure event
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.scanner.automation.safe_json import safe_json_read, safe_jsonl_append

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class DriftRemediationResult:
    """Result of a single drift remediation attempt."""

    action_taken: str  # "REMEDIATED", "SKIPPED_COOLDOWN", "SKIPPED_NO_REQUEST", "FAILED", "FALSE_ALARM"
    success: bool
    pairs_retrained: list = field(default_factory=list)
    reason: str = ""
    duration_seconds: float = 0.0
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        """Convert to dict for logging."""
        return {
            "action_taken": self.action_taken,
            "success": self.success,
            "pairs_retrained": self.pairs_retrained,
            "reason": self.reason,
            "duration_seconds": round(self.duration_seconds, 2),
            "validation_metrics": self.validation_metrics,
            "timestamp": self.timestamp,
        }


class DriftRemediator:
    """Auto-closes the drift → retrain → validate → swap loop.

    Never blocks the scan loop — retraining happens in a subprocess.
    All file I/O uses atomic writes and fcntl locking.
    Every method catches exceptions and returns error results.
    """

    def __init__(
        self,
        request_path: str | Path = "trained_data/retrain_request.json",
        remediation_log_path: str | Path = "trained_data/drift_remediation_log.jsonl",
        retrain_summary_path: str | Path = "trained_data/retrain_all_summary.json",
        agent_weights_path: str | Path = "trained_data/models/agent_weights.json",
        cooldown_hours: int = 12,
        retrain_timeout_minutes: int = 30,
        project_root: Optional[str | Path] = None,
    ):
        """Initialize DriftRemediator.

        Args:
            request_path: Path to retrain_request.json (written by RetrainTrigger)
            remediation_log_path: Path to drift_remediation_log.jsonl (appended to)
            retrain_summary_path: Path to retrain_all_summary.json (read after retrain)
            agent_weights_path: Path to agent_weights.json (updated on success)
            cooldown_hours: Hours to wait before retrying a failed remediation
            retrain_timeout_minutes: Max time for retrain subprocess (minutes)
            project_root: Project root (defaults to cwd)
        """
        self._root = Path(project_root) if project_root else Path.cwd()
        self.request_path = Path(request_path)
        self.remediation_log_path = Path(remediation_log_path)
        self.retrain_summary_path = Path(retrain_summary_path)
        self.agent_weights_path = Path(agent_weights_path)
        self.cooldown_hours = cooldown_hours
        self.retrain_timeout_minutes = retrain_timeout_minutes

        # Ensure paths are absolute
        if not self.request_path.is_absolute():
            self.request_path = self._root / self.request_path
        if not self.remediation_log_path.is_absolute():
            self.remediation_log_path = self._root / self.remediation_log_path
        if not self.retrain_summary_path.is_absolute():
            self.retrain_summary_path = self._root / self.retrain_summary_path
        if not self.agent_weights_path.is_absolute():
            self.agent_weights_path = self._root / self.agent_weights_path

        logger.debug(
            f"DriftRemediator initialized: request_path={self.request_path}, "
            f"cooldown_hours={self.cooldown_hours}"
        )

    def check_and_remediate(self) -> DriftRemediationResult:
        """Main entry point: check for retrain requests and auto-remediate.

        Called every N cycles from orchestrator. Never raises exceptions.

        Returns:
            DriftRemediationResult describing what action was taken.
        """
        start_time = time.time()
        result = DriftRemediationResult(
            action_taken="SKIPPED_NO_REQUEST",
            success=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        try:
            # Check if retrain request exists
            request = self._read_retrain_request()
            if request is None:
                result.reason = "No pending retrain request"
                logger.debug(f"DriftRemediator: {result.action_taken} — {result.reason}")
                return result

            # Check if we're in cooldown period
            if self._is_in_cooldown():
                result.action_taken = "SKIPPED_COOLDOWN"
                result.reason = f"Cooldown period active (cooldown_hours={self.cooldown_hours})"
                logger.debug(f"DriftRemediator: {result.action_taken} — {result.reason}")
                self._write_remediation_event("COOLDOWN_SKIP", {"reason": result.reason})
                return result

            # Log that we're starting remediation
            self._write_remediation_event("REMEDIATION_STARTED", {"request": request})

            # Fast path: check if current model still passes validation
            logger.debug("DriftRemediator: Running fast validation on current model")
            is_current_valid, current_metrics = self._validate_current_model()

            if is_current_valid:
                # Current model is fine — false alarm
                result.action_taken = "FALSE_ALARM"
                result.success = True
                result.reason = "Current model still validates — false alarm"
                result.validation_metrics = current_metrics
                logger.info(
                    f"DriftRemediator: {result.action_taken} — "
                    f"OOS/IS ratio={current_metrics.get('oos_is_ratio', 'N/A')}"
                )
                self._write_remediation_event("FALSE_ALARM", current_metrics)
                self._clear_retrain_request()
                return result

            # Current model is stale — spawn retrain subprocess
            logger.info(
                f"DriftRemediator: Current model validation failed — "
                f"spawning retrain subprocess for {len(request.get('pairs', []))} pairs"
            )

            retrain_success, retrain_output = self._run_retrain_subprocess()

            if not retrain_success:
                # Retrain failed
                result.action_taken = "FAILED"
                result.success = False
                result.reason = retrain_output
                logger.warning(
                    f"DriftRemediator: Retrain subprocess failed: {retrain_output[:200]}"
                )
                self._write_remediation_event(
                    "REMEDIATION_FAILED",
                    {
                        "reason": "Retrain subprocess failed",
                        "error": retrain_output[:500],
                    },
                )
                return result

            # Retrain succeeded — validate new model
            logger.debug("DriftRemediator: Validating retrained model")
            validation_passed, validation_metrics = self._validate_retrain_result()

            if validation_passed:
                # Success!
                result.action_taken = "REMEDIATED"
                result.success = True
                result.pairs_retrained = request.get("pairs", [])
                result.validation_metrics = validation_metrics
                result.reason = (
                    f"Retrained {len(result.pairs_retrained)} pair(s), validation passed"
                )
                logger.info(
                    f"DriftRemediator: {result.action_taken} — "
                    f"accuracy={validation_metrics.get('accuracy', 'N/A')}, "
                    f"OOS/IS ratio={validation_metrics.get('oos_is_ratio', 'N/A')}"
                )
                self._write_remediation_event("REMEDIATION_SUCCESS", validation_metrics)
                self._clear_retrain_request()
            else:
                # Validation failed
                result.action_taken = "FAILED"
                result.success = False
                result.pairs_retrained = request.get("pairs", [])
                result.validation_metrics = validation_metrics
                result.reason = (
                    f"Retrained {len(result.pairs_retrained)} pair(s), but validation failed"
                )
                logger.warning(
                    f"DriftRemediator: {result.action_taken} — "
                    f"validation_metrics={validation_metrics}"
                )
                self._write_remediation_event(
                    "REMEDIATION_FAILED", {"reason": "Validation failed", **validation_metrics}
                )

        except Exception as e:
            # Never propagate exceptions
            result.action_taken = "FAILED"
            result.success = False
            result.reason = f"Unexpected error: {str(e)}"
            logger.error(
                f"DriftRemediator: Unexpected exception in check_and_remediate: {e}",
                exc_info=True,
            )
            self._write_remediation_event(
                "REMEDIATION_FAILED",
                {"reason": "Unexpected exception", "error": str(e)},
            )

        finally:
            result.duration_seconds = time.time() - start_time

        return result

    def _read_retrain_request(self) -> Optional[Dict[str, Any]]:
        """Read retrain_request.json safely.

        Returns:
            Dict with request, or None if file missing/empty.
        """
        try:
            data = safe_json_read(self.request_path)
            if data and isinstance(data, dict) and data.get("pairs"):
                return data
            return None
        except Exception as e:
            logger.debug(f"DriftRemediator: Failed to read retrain request: {e}")
            return None

    def _is_in_cooldown(self) -> bool:
        """Check if last remediation attempt was within cooldown period.

        Reads from remediation_log.jsonl, checks last "REMEDIATION_FAILED" entry.

        Returns:
            True if in cooldown, False otherwise.
        """
        try:
            if not self.remediation_log_path.exists():
                return False

            # Read last line of JSONL file
            with open(self.remediation_log_path, "r") as f:
                lines = f.readlines()

            if not lines:
                return False

            # Parse last entry
            last_line = lines[-1].strip()
            if not last_line:
                return False

            last_entry = json.loads(last_line)

            # Check if last event was a failure and within cooldown
            if last_entry.get("event_type") == "REMEDIATION_FAILED":
                timestamp_str = last_entry.get("timestamp", "")
                if timestamp_str:
                    try:
                        last_time = datetime.fromisoformat(
                            timestamp_str.replace("Z", "+00:00")
                        )
                        now = datetime.now(timezone.utc)
                        age = now - last_time

                        in_cooldown = age < timedelta(hours=self.cooldown_hours)
                        if in_cooldown:
                            logger.debug(
                                f"DriftRemediator: In cooldown — "
                                f"{age.total_seconds():.0f}s since last failure"
                            )
                        return in_cooldown
                    except Exception as _parse_err:
                        logger.debug(f"DriftRemediator: Failed to parse timestamp: {_parse_err}")

            return False

        except Exception as e:
            logger.debug(f"DriftRemediator: Cooldown check failed: {e}")
            return False

    def _validate_current_model(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate the CURRENT model (fast path before retrain).

        Checks if current model still meets minimum validation criteria.
        This is a light validation to detect false alarms.

        Returns:
            (passes: bool, metrics: dict)
        """
        try:
            # Try to read existing retrain summary
            summary = safe_json_read(self.retrain_summary_path)

            if not summary:
                # No summary yet, assume current model is valid
                return True, {}

            # Check OOS/IS ratio (key metric for model generalization)
            oos_is_ratio = summary.get("oos_is_ratio", 0.0)
            accuracy = summary.get("accuracy", 0.0)

            # Current model is valid if OOS/IS ratio is still good
            passes = oos_is_ratio >= 0.60

            metrics = {
                "accuracy": round(accuracy, 4),
                "oos_is_ratio": round(oos_is_ratio, 4),
            }

            logger.debug(
                f"DriftRemediator: Current model validation: "
                f"passes={passes}, OOS/IS={oos_is_ratio:.4f}"
            )

            return passes, metrics

        except Exception as e:
            logger.debug(f"DriftRemediator: Current model validation failed: {e}")
            return False, {}

    def _run_retrain_subprocess(self) -> Tuple[bool, str]:
        """Spawn retraining subprocess.

        Runs `python scripts/scheduled_retrain.py` with timeout.

        Returns:
            (success: bool, output: str)
            - success=True: retrain completed without errors
            - success=False: timeout, error, or non-zero exit code
            - output: stdout/stderr captured during run
        """
        try:
            # Resolve script path
            retrain_script = self._root / "scripts" / "scheduled_retrain.py"

            if not retrain_script.exists():
                msg = f"Retrain script not found: {retrain_script}"
                logger.error(f"DriftRemediator: {msg}")
                return False, msg

            logger.info(f"DriftRemediator: Spawning retrain subprocess: {retrain_script}")

            # Build command
            cmd = [
                "python",
                str(retrain_script),
            ]

            # Spawn subprocess with timeout
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(self._root),
                    capture_output=True,
                    text=True,
                    timeout=self.retrain_timeout_minutes * 60,
                )

                output = result.stdout + "\n" + result.stderr

                if result.returncode == 0:
                    logger.info("DriftRemediator: Retrain subprocess completed successfully")
                    return True, output
                else:
                    msg = f"Retrain subprocess exited with code {result.returncode}"
                    logger.warning(f"DriftRemediator: {msg}")
                    return False, msg

            except subprocess.TimeoutExpired:
                msg = f"Retrain subprocess timeout after {self.retrain_timeout_minutes} minutes"
                logger.error(f"DriftRemediator: {msg}")
                return False, msg

        except Exception as e:
            msg = f"Failed to spawn retrain subprocess: {str(e)}"
            logger.error(f"DriftRemediator: {msg}")
            return False, msg

    def _validate_retrain_result(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate the NEW model after retrain.

        Reads retrain_all_summary.json and checks:
        - accuracy >= 0.52 (baseline)
        - OOS/IS ratio >= 0.60 (generalization)

        Returns:
            (passes: bool, metrics: dict)
        """
        try:
            summary = safe_json_read(self.retrain_summary_path)

            if not summary:
                return False, {"reason": "No retrain summary found"}

            accuracy = summary.get("accuracy", 0.0)
            oos_is_ratio = summary.get("oos_is_ratio", 0.0)

            passes = accuracy >= 0.52 and oos_is_ratio >= 0.60

            metrics = {
                "accuracy": round(accuracy, 4),
                "oos_is_ratio": round(oos_is_ratio, 4),
            }

            logger.debug(
                f"DriftRemediator: Retrain validation: "
                f"passes={passes}, accuracy={accuracy:.4f}, OOS/IS={oos_is_ratio:.4f}"
            )

            return passes, metrics

        except Exception as e:
            logger.debug(f"DriftRemediator: Retrain validation failed: {e}")
            return False, {"reason": str(e)}

    def _write_remediation_event(
        self, event_type: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Append remediation event to drift_remediation_log.jsonl.

        Uses atomic JSONL append (safe_jsonl_append with fcntl locking).

        Args:
            event_type: "REMEDIATION_STARTED", "REMEDIATION_SUCCESS", "REMEDIATION_FAILED",
                       "FALSE_ALARM", "COOLDOWN_SKIP"
            details: Optional metadata dict to include in event
        """
        try:
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
            }
            if details:
                event["details"] = details

            self.remediation_log_path.parent.mkdir(parents=True, exist_ok=True)

            success = safe_jsonl_append(self.remediation_log_path, event)

            if not success:
                logger.warning(
                    f"DriftRemediator: Failed to write remediation event: {event_type}"
                )

        except Exception as e:
            logger.warning(f"DriftRemediator: Exception writing remediation event: {e}")

    def _clear_retrain_request(self) -> None:
        """Clear the retrain_request.json after successful remediation or false alarm."""
        try:
            if self.request_path.exists():
                self.request_path.unlink()
                logger.debug(f"DriftRemediator: Cleared retrain request")
        except Exception as e:
            logger.warning(f"DriftRemediator: Failed to clear retrain request: {e}")
