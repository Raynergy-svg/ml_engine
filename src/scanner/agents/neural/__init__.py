"""
Neural Agent Policies — learned replacements for rule-based scanner agents.

Instead of hand-coded formulas (SMA crossovers, RSI thresholds, etc.), each
specialist agent is a small neural policy network that learns from trade
outcomes via RL-style reward shaping.

Usage:
    from src.scanner.agents.neural import NeuralAgentTeam
    neural_team = NeuralAgentTeam(config)
    # Replaces ScannerAgentTeam.evaluate() — same interface, learned internals
"""

from __future__ import annotations

from .neural_agent_base import NeuralAgentBase, NeuralAgentConfig
from .policies import TrendPolicy, MomentumPolicy, MeanReversionPolicy, VolatilityPolicy
from .team_bridge import NeuralAgentTeam
from .trainer import NeuralAgentTrainer

__all__ = [
    "NeuralAgentBase",
    "NeuralAgentConfig",
    "TrendPolicy",
    "MomentumPolicy",
    "MeanReversionPolicy",
    "VolatilityPolicy",
    "NeuralAgentTeam",
    "NeuralAgentTrainer",
]
