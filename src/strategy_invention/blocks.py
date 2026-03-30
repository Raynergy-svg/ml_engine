"""Strategy building blocks — the DSL atoms.

Each BlockType has a fixed set of typed parameters with validated ranges.
Blocks are JSON-serializable. Blocks can be combined into StrategyTemplates.

Design constraints:
- All math is numpy-only
- No external API calls
- Parameter validation at construction time
- Each block has a human-readable description
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ────────────────────────────── Block Types ──────────────────────────────

class BlockCategory(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    SIZING = "sizing"
    FILTER = "filter"


class BlockType(str, Enum):
    # Entry signals
    TREND_ENTRY = "TrendFollowingEntry"
    MEAN_REVERSION_ENTRY = "MeanReversionEntry"
    MOMENTUM_ENTRY = "MomentumEntry"
    BREAKOUT_ENTRY = "BreakoutEntry"

    # Exit signals
    TIME_BASED_EXIT = "TimeBasedExit"
    PROFIT_TARGET_EXIT = "ProfitTargetExit"
    STOP_LOSS_EXIT = "StopLossExit"
    TRAILING_STOP_EXIT = "TrailingStopExit"
    CORRELATION_EXIT = "CorrelationExit"

    # Position sizing
    FIXED_SIZE = "FixedSize"
    CONFIDENCE_SCALING = "ConfidenceScaling"
    VOLATILITY_SCALING = "VolatilityScaling"

    # Filters
    RISK_FILTER = "RiskFilter"
    LIQUIDITY_FILTER = "LiquidityFilter"
    SESSION_FILTER = "SessionFilter"


# ────────────────────────────── Parameter Schemas ───────────────────────────

# For each BlockType: {param_name → (min, max, default)}
BLOCK_PARAM_SCHEMAS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    BlockType.TREND_ENTRY: {
        "timeframe": (15, 240, 60),           # minutes
        "slope_threshold": (0.0001, 0.005, 0.0015),
        "ma_period": (5, 50, 20),
    },
    BlockType.MEAN_REVERSION_ENTRY: {
        "rsi_period": (5, 30, 14),
        "overbought": (60, 85, 70),
        "oversold": (15, 40, 30),
        "band_width": (1.0, 3.0, 2.0),
    },
    BlockType.MOMENTUM_ENTRY: {
        "period": (5, 30, 14),
        "threshold": (0.001, 0.02, 0.005),
        "lookback": (3, 20, 10),
    },
    BlockType.BREAKOUT_ENTRY: {
        "window": (10, 100, 20),
        "atr_multiplier": (0.5, 3.0, 1.5),
        "volume_confirm": (0.0, 1.0, 0.5),  # 0=no confirm, 1=require confirm
    },
    BlockType.TIME_BASED_EXIT: {
        "hold_minutes": (15, 480, 120),
    },
    BlockType.PROFIT_TARGET_EXIT: {
        "target_pips": (5, 100, 25),
        "atr_factor": (0.5, 3.0, 1.5),
    },
    BlockType.STOP_LOSS_EXIT: {
        "stop_pips": (5, 80, 20),
        "atr_factor": (0.5, 2.5, 1.2),
    },
    BlockType.TRAILING_STOP_EXIT: {
        "trail_pips": (5, 60, 15),
        "atr_factor": (0.5, 2.0, 1.0),
        "activation_pips": (3, 30, 10),
    },
    BlockType.CORRELATION_EXIT: {
        "correlation_threshold": (0.3, 0.9, 0.7),
        "lookback_periods": (5, 50, 20),
    },
    BlockType.FIXED_SIZE: {
        "size_percent": (0.005, 0.05, 0.02),
    },
    BlockType.CONFIDENCE_SCALING: {
        "base_size": (0.005, 0.03, 0.01),
        "scale_factor": (1.0, 4.0, 2.0),
        "min_confidence": (0.3, 0.7, 0.5),
    },
    BlockType.VOLATILITY_SCALING: {
        "atr_multiplier": (0.5, 3.0, 1.5),
        "base_size": (0.005, 0.03, 0.01),
    },
    BlockType.RISK_FILTER: {
        "max_risk_pct": (0.01, 0.1, 0.05),
        "max_correlation": (0.3, 0.9, 0.7),
    },
    BlockType.LIQUIDITY_FILTER: {
        "min_spread_pips": (0.5, 5.0, 2.0),
        "max_spread_pips": (3.0, 20.0, 8.0),
    },
    BlockType.SESSION_FILTER: {
        "london_weight": (0.0, 1.0, 0.8),
        "ny_weight": (0.0, 1.0, 0.8),
        "tokyo_weight": (0.0, 1.0, 0.3),
    },
}

BLOCK_CATEGORIES: Dict[str, BlockCategory] = {
    BlockType.TREND_ENTRY: BlockCategory.ENTRY,
    BlockType.MEAN_REVERSION_ENTRY: BlockCategory.ENTRY,
    BlockType.MOMENTUM_ENTRY: BlockCategory.ENTRY,
    BlockType.BREAKOUT_ENTRY: BlockCategory.ENTRY,
    BlockType.TIME_BASED_EXIT: BlockCategory.EXIT,
    BlockType.PROFIT_TARGET_EXIT: BlockCategory.EXIT,
    BlockType.STOP_LOSS_EXIT: BlockCategory.EXIT,
    BlockType.TRAILING_STOP_EXIT: BlockCategory.EXIT,
    BlockType.CORRELATION_EXIT: BlockCategory.EXIT,
    BlockType.FIXED_SIZE: BlockCategory.SIZING,
    BlockType.CONFIDENCE_SCALING: BlockCategory.SIZING,
    BlockType.VOLATILITY_SCALING: BlockCategory.SIZING,
    BlockType.RISK_FILTER: BlockCategory.FILTER,
    BlockType.LIQUIDITY_FILTER: BlockCategory.FILTER,
    BlockType.SESSION_FILTER: BlockCategory.FILTER,
}


# ────────────────────────────── StrategyBlock ─────────────────────────────

@dataclass
class StrategyBlock:
    """A single DSL building block with validated params."""
    block_type: str          # BlockType value (string)
    params: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and clamp params to schema ranges on construction."""
        schema = BLOCK_PARAM_SCHEMAS.get(self.block_type, {})
        validated: Dict[str, float] = {}
        for param_name, (lo, hi, default) in schema.items():
            raw = self.params.get(param_name, default)
            validated[param_name] = float(np.clip(raw, lo, hi))
        # Fill any missing params with defaults
        for param_name, (lo, hi, default) in schema.items():
            if param_name not in validated:
                validated[param_name] = default
        self.params = validated

    @property
    def category(self) -> str:
        return BLOCK_CATEGORIES.get(self.block_type, BlockCategory.ENTRY).value

    def validate(self) -> Tuple[bool, str]:
        """Returns (is_valid, reason)."""
        if self.block_type not in BLOCK_PARAM_SCHEMAS:
            return False, f"Unknown block_type: {self.block_type}"
        schema = BLOCK_PARAM_SCHEMAS[self.block_type]
        for param_name, (lo, hi, _) in schema.items():
            val = self.params.get(param_name)
            if val is None:
                return False, f"Missing param: {param_name}"
            if not (lo <= val <= hi):
                return False, f"Param {param_name}={val} out of range [{lo}, {hi}]"
        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        return {"block_type": self.block_type, "params": {k: round(v, 6) for k, v in self.params.items()}}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyBlock":
        return cls(block_type=data["block_type"], params=data.get("params", {}))

    @classmethod
    def default_for(cls, block_type: str) -> "StrategyBlock":
        """Create a block with default parameters."""
        schema = BLOCK_PARAM_SCHEMAS.get(block_type, {})
        params = {k: v[2] for k, v in schema.items()}  # use default
        return cls(block_type=block_type, params=params)
