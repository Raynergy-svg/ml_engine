"""Shim module - re-exports from src.data.feature_engineering for backward compatibility."""
import warnings
warnings.warn(
    f"{__name__} is a backward-compatibility shim. "
    f"Import directly from the src package instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.data.feature_engineering import *  # noqa: F401, F403
from src.data.feature_engineering import FeatureEngineering  # noqa: F401
