"""Meta-parameter registry: per-agent, per-regime hyperparameter sets.

Each agent has a HyperparameterSet per regime (NORMAL, VOLATILE, TRENDING, RANGING).
These are adapted online by the OnlineBayesianAdapter based on trade outcomes.

Hyperparameters controlled:
- confidence_threshold: clip agent score below this value (range: 0.3-0.9)
- weight_decay_rate: how fast agent weight decays toward baseline (range: 0.001-0.05)
- regime_switch_sensitivity: multiplier on volatility threshold (range: 0.1-0.9)
- voting_veto_power: multiplier on agent's voting weight (range: 0.5-1.5)

Stored: trained_data/meta/agent_meta_params_{regime}.json (one file per regime)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_META_DIR = _PROJECT_ROOT / "trained_data" / "meta"

REGIMES = ["NORMAL", "VOLATILE", "TRENDING", "RANGING"]

# Hyperparameter grid — each dimension is a discrete set of candidate values
DEFAULT_GRID = {
    "confidence_threshold": [0.3, 0.45, 0.6, 0.75, 0.9],
    "weight_decay_rate": [0.001, 0.005, 0.01, 0.025, 0.05],
    "regime_switch_sensitivity": [0.1, 0.3, 0.5, 0.7, 0.9],
    "voting_veto_power": [0.5, 0.75, 1.0, 1.25, 1.5],
}

# Default (baseline) values — midpoints of each grid
DEFAULT_HYPERPARAMS = {
    "confidence_threshold": 0.6,
    "weight_decay_rate": 0.01,
    "regime_switch_sensitivity": 0.5,
    "voting_veto_power": 1.0,
}


@dataclass
class HyperparameterSet:
    """A single active hyperparameter configuration for (agent, regime)."""
    confidence_threshold: float = 0.6
    weight_decay_rate: float = 0.01
    regime_switch_sensitivity: float = 0.5
    voting_veto_power: float = 1.0
    fitness: float = 0.5           # rolling 0-1 success metric
    update_count: int = 0
    last_updated: str = ""
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_threshold": round(self.confidence_threshold, 4),
            "weight_decay_rate": round(self.weight_decay_rate, 5),
            "regime_switch_sensitivity": round(self.regime_switch_sensitivity, 4),
            "voting_veto_power": round(self.voting_veto_power, 4),
            "fitness": round(self.fitness, 4),
            "update_count": self.update_count,
            "last_updated": self.last_updated or datetime.now(timezone.utc).isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperparameterSet":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def default(cls) -> "HyperparameterSet":
        return cls(**DEFAULT_HYPERPARAMS)


class MetaParametersRegistry:
    """Manages hyperparameter sets for all agents × regimes.

    Structure: {agent_name → {regime → HyperparameterSet}}
    Persisted per-regime to allow partial loads.
    """

    def __init__(
        self,
        meta_dir: Optional[Path] = None,
        grid: Optional[Dict[str, List[float]]] = None,
    ):
        self._meta_dir = meta_dir or _META_DIR
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        self._grid = grid or DEFAULT_GRID
        # {agent_name → {regime → HyperparameterSet}}
        self._store: Dict[str, Dict[str, HyperparameterSet]] = {}
        self._load_all()

    def get(self, agent_name: str, regime: str) -> HyperparameterSet:
        """Get active hyperparams for (agent, regime). Creates default if missing."""
        regime = regime.upper()
        if regime not in REGIMES:
            regime = "NORMAL"
        if agent_name not in self._store:
            self._store[agent_name] = {}
        if regime not in self._store[agent_name]:
            self._store[agent_name][regime] = HyperparameterSet.default()
        return self._store[agent_name][regime]

    def set(self, agent_name: str, regime: str, params: HyperparameterSet) -> None:
        """Update hyperparams for (agent, regime)."""
        regime = regime.upper()
        if agent_name not in self._store:
            self._store[agent_name] = {}
        params.last_updated = datetime.now(timezone.utc).isoformat()
        params.update_count += 1
        self._store[agent_name][regime] = params

    def update_fitness(self, agent_name: str, regime: str, outcome_value: float) -> None:
        """Update rolling fitness for (agent, regime). outcome_value in [-1, +1]."""
        params = self.get(agent_name, regime)
        # Exponential moving average of normalized outcome
        normalized = (outcome_value + 1.0) / 2.0  # map [-1,1] → [0,1]
        alpha = 0.2  # EMA weight for new observation
        params.fitness = (1 - alpha) * params.fitness + alpha * normalized
        self.set(agent_name, regime, params)

    def reset_to_default(self, agent_name: str, regime: str) -> None:
        """Rollback: reset to baseline hyperparams."""
        default = HyperparameterSet.default()
        default.last_updated = datetime.now(timezone.utc).isoformat()
        if agent_name not in self._store:
            self._store[agent_name] = {}
        self._store[agent_name][regime.upper()] = default
        logger.info("MetaParams: reset %s/%s to baseline", agent_name, regime)

    def all_agents(self) -> List[str]:
        return list(self._store.keys())

    def save(self) -> None:
        """Persist all hyperparams to disk atomically (one file per regime)."""
        for regime in REGIMES:
            regime_data: Dict[str, Any] = {}
            for agent_name, regime_map in self._store.items():
                if regime in regime_map:
                    regime_data[agent_name] = regime_map[regime].to_dict()
            if not regime_data:
                continue
            path = self._meta_dir / f"agent_meta_params_{regime}.json"
            payload = json.dumps(
                {"version": "1.0", "regime": regime, "agents": regime_data},
                indent=2,
                sort_keys=True
            )
            try:
                fd, tmp = tempfile.mkstemp(dir=str(self._meta_dir), suffix=".tmp")
                os.write(fd, payload.encode())
                os.fsync(fd)
                os.close(fd)
                os.rename(tmp, str(path))
            except Exception as e:
                logger.error("MetaParams: save failed for regime %s: %s", regime, e)

    def _load_all(self) -> None:
        """Load all regime files on startup. Missing files → defaults (no crash)."""
        for regime in REGIMES:
            path = self._meta_dir / f"agent_meta_params_{regime}.json"
            if not path.exists():
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                agents = data.get("agents", {})
                for agent_name, params_dict in agents.items():
                    if agent_name not in self._store:
                        self._store[agent_name] = {}
                    self._store[agent_name][regime] = HyperparameterSet.from_dict(params_dict)
            except Exception as e:
                logger.warning("MetaParams: failed to load %s: %s — using defaults", path.name, e)
