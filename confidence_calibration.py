"""Shim module - re-exports from src.risk.confidence_calibration for backward compatibility."""
import warnings
warnings.warn(
    f"{__name__} is a backward-compatibility shim. "
    f"Import directly from the src package instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.risk.confidence_calibration import *  # noqa: F401, F403
from src.risk.confidence_calibration import CalibrationResult  # noqa: F401
