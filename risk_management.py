"""Shim module - re-exports from src.risk.risk_management for backward compatibility."""
import warnings
warnings.warn(
    f"{__name__} is a backward-compatibility shim. "
    f"Import directly from the src package instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.risk.risk_management import *  # noqa: F401, F403
