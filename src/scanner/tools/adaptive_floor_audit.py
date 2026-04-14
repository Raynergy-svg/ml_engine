"""Phase 90 US-405: Adaptive floor activation audit script.

Post-Phase-89 diagnostic to confirm whether all three adaptive floor/ceiling
components are accumulating state after the cold-start observe filter fix.

Usage:
    python -m src.scanner.tools.adaptive_floor_audit

Exits with code 0 if all three components load without error.
Exits with code 1 if any component fails to load.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ── State file paths (canonical locations, matching each module's DEFAULT_STATE_PATH) ──

_CONFIDENCE_FLOOR_STATE = Path("trained_data/adaptive_floor_state.json")
_MOMENTUM_FLOOR_STATE = Path("trained_data/adaptive_momentum_floor_state.json")
_DRAWDOWN_CEILING_STATE = Path("trained_data/adaptive_drawdown_ceiling_state.json")

_STATUS_ACTIVE = "ACTIVE"
_STATUS_DORMANT = "DORMANT"
_STATUS_EXHAUSTED = "EXHAUSTED"


def _audit_confidence_floor(state_path: Path | None = None) -> Dict[str, Any]:
    """Audit AdaptiveConfidenceFloor state.

    Returns dict with: state_file_path, state_file_exists, activation_count, status.
    """
    from src.scanner.adaptive_confidence_floor import AdaptiveConfidenceFloor, DEFAULT_STATE_PATH

    path = Path(state_path) if state_path is not None else DEFAULT_STATE_PATH
    exists = path.exists()

    try:
        floor = AdaptiveConfidenceFloor(state_path=path)
        state = floor.get_state()
        activation_count = int(state.get("total_adaptations", 0))
        max_reached = bool(state.get("max_adaptations_reached", False))

        if max_reached:
            status = _STATUS_EXHAUSTED
        elif activation_count > 0:
            status = _STATUS_ACTIVE
        else:
            status = _STATUS_DORMANT
    except Exception as exc:
        logger.error("AdaptiveConfidenceFloor audit failed: %s", exc)
        raise

    return {
        "state_file_path": str(path),
        "state_file_exists": exists,
        "activation_count": activation_count,
        "status": status,
    }


def _audit_momentum_floor(state_path: Path | None = None) -> Dict[str, Any]:
    """Audit AdaptiveMomentumFloor state.

    Returns dict with: state_file_path, state_file_exists, activation_count, status.
    """
    from src.scanner.adaptive_momentum_floor import AdaptiveMomentumFloor, DEFAULT_STATE_PATH

    path = Path(state_path) if state_path is not None else DEFAULT_STATE_PATH
    exists = path.exists()

    try:
        floor = AdaptiveMomentumFloor(state_path=path)
        status_dict = floor.get_status()
        activation_count = int(status_dict.get("adaptation_count", 0))
        max_adaptations = int(status_dict.get("max_adaptations", 999))

        if activation_count >= max_adaptations:
            status = _STATUS_EXHAUSTED
        elif activation_count > 0:
            status = _STATUS_ACTIVE
        else:
            status = _STATUS_DORMANT
    except Exception as exc:
        logger.error("AdaptiveMomentumFloor audit failed: %s", exc)
        raise

    return {
        "state_file_path": str(path),
        "state_file_exists": exists,
        "activation_count": activation_count,
        "status": status,
    }


def _audit_drawdown_ceiling(state_path: Path | None = None) -> Dict[str, Any]:
    """Audit AdaptiveDrawdownCeiling state.

    Returns dict with: state_file_path, state_file_exists, activation_count, status.
    """
    from src.scanner.adaptive_drawdown_ceiling import AdaptiveDrawdownCeiling, DEFAULT_STATE_PATH

    path = Path(state_path) if state_path is not None else DEFAULT_STATE_PATH
    exists = path.exists()

    try:
        ceiling = AdaptiveDrawdownCeiling(state_path=path)
        status_dict = ceiling.get_status()
        activation_count = int(status_dict.get("adaptation_count", 0))
        max_adaptations = int(status_dict.get("max_adaptations", 999))

        if activation_count >= max_adaptations:
            status = _STATUS_EXHAUSTED
        elif activation_count > 0:
            status = _STATUS_ACTIVE
        else:
            status = _STATUS_DORMANT
    except Exception as exc:
        logger.error("AdaptiveDrawdownCeiling audit failed: %s", exc)
        raise

    return {
        "state_file_path": str(path),
        "state_file_exists": exists,
        "activation_count": activation_count,
        "status": status,
    }


def run(
    confidence_floor_path: Path | None = None,
    momentum_floor_path: Path | None = None,
    drawdown_ceiling_path: Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Run audit on all three adaptive floor/ceiling components.

    Args:
        confidence_floor_path: Override state path for AdaptiveConfidenceFloor.
        momentum_floor_path: Override state path for AdaptiveMomentumFloor.
        drawdown_ceiling_path: Override state path for AdaptiveDrawdownCeiling.

    Returns:
        Dict with keys 'confidence_floor', 'momentum_floor', 'drawdown_ceiling',
        each containing: state_file_path, state_file_exists, activation_count, status.

    Raises:
        RuntimeError: If any component fails to load (triggers exit code 1).
    """
    errors: list[str] = []
    results: Dict[str, Dict[str, Any]] = {}

    for key, audit_fn, path in [
        ("confidence_floor", _audit_confidence_floor, confidence_floor_path),
        ("momentum_floor", _audit_momentum_floor, momentum_floor_path),
        ("drawdown_ceiling", _audit_drawdown_ceiling, drawdown_ceiling_path),
    ]:
        try:
            results[key] = audit_fn(path)
        except Exception as exc:
            errors.append(f"{key}: {exc}")
            results[key] = {
                "state_file_path": str(path or "unknown"),
                "state_file_exists": False,
                "activation_count": 0,
                "status": "ERROR",
            }

    if errors:
        raise RuntimeError(f"Adaptive floor audit failed for: {', '.join(errors)}")

    return results


def _print_report(results: Dict[str, Dict[str, Any]]) -> None:
    """Print structured audit report to stdout."""
    print("\n=== Adaptive Floor Activation Audit (Phase 90 US-405) ===\n")
    for component, info in results.items():
        print(f"Component: {component}")
        print(f"  State file:       {info['state_file_path']}")
        print(f"  File exists:      {info['state_file_exists']}")
        print(f"  Activation count: {info['activation_count']}")
        print(f"  Status:           {info['status']}")
        print()
    print("=== End of Report ===\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exit_code = 0
    try:
        audit_results = run()
        _print_report(audit_results)
    except RuntimeError as err:
        logger.error("Audit failed: %s", err)
        exit_code = 1
    sys.exit(exit_code)
