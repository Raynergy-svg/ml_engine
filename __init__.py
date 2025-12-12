"""Top-level package for the ML engine.

Keep imports here lightweight to avoid import-time failures when optional
modules aren't present.
"""

from .ml_engine_enhanced import EnhancedMLEngine
from .utils import setup_logging

__version__ = "1.0.0"
__all__ = [
    "EnhancedMLEngine",
    "setup_logging",
]
