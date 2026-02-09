"""Shim module - re-exports from src.core.modular_inference for backward compatibility."""
import warnings
warnings.warn(
    f"{__name__} is a backward-compatibility shim. "
    f"Import directly from the src package instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.core.modular_inference import *  # noqa: F401, F403
from src.core.modular_inference import ModularEnsembleInference  # noqa: F401
