#!/usr/bin/env python3
"""CLI Configuration dataclasses for ML Engine Trading Bot.

Re-exports from src.training.training_config to eliminate duplication.
All configuration dataclasses are now maintained in a single location.
"""
from src.training.training_config import (  # noqa: F401
    OandaFetchOptions,
    BuddyTrainingAdvancedOptions,
    BuddyTrainingOptions,
)

__all__ = [
    "OandaFetchOptions",
    "BuddyTrainingAdvancedOptions",
    "BuddyTrainingOptions",
]
