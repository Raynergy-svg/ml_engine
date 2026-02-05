"""CLI module for ML Engine Trading Bot.

This module provides the command-line interface for the trading bot,
including commands for training, inference, scanning, and analysis.
"""

from __future__ import annotations

# Re-export main entry point
from src.cli.entry import main

__all__ = ["main"]
