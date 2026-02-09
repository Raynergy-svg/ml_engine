"""Shim module - re-exports from src.utils.fx_paper for backward compatibility."""
import warnings
warnings.warn(
    f"{__name__} is a backward-compatibility shim. "
    f"Import directly from the src package instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.utils.fx_paper import *  # noqa: F401, F403
