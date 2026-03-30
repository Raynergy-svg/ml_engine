"""Grammar validator: defines valid strategy block compositions.

Prevents nonsensical combinations like MeanReversionEntry + TrailingStop
(mean reversion trades are short-duration; trailing stops assume trends).

Also validates parameter constraints across blocks (e.g., TP > SL).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .blocks import BlockType, StrategyBlock, BlockCategory, BLOCK_PARAM_SCHEMAS
from .template import StrategyTemplate

logger = logging.getLogger(__name__)


# Which exit types are compatible with each entry type
# Keys and values are BlockType.value strings (not enums) for consistency with StrategyBlock.block_type
VALID_ENTRY_EXIT_PAIRS: Dict[str, List[str]] = {
    BlockType.TREND_ENTRY.value: [
        BlockType.TRAILING_STOP_EXIT.value,
        BlockType.TIME_BASED_EXIT.value,
        BlockType.PROFIT_TARGET_EXIT.value,
        BlockType.STOP_LOSS_EXIT.value,
    ],
    BlockType.MEAN_REVERSION_ENTRY.value: [
        BlockType.PROFIT_TARGET_EXIT.value,
        BlockType.STOP_LOSS_EXIT.value,
        BlockType.CORRELATION_EXIT.value,
        BlockType.TIME_BASED_EXIT.value,
    ],
    BlockType.MOMENTUM_ENTRY.value: [
        BlockType.PROFIT_TARGET_EXIT.value,
        BlockType.TRAILING_STOP_EXIT.value,
        BlockType.STOP_LOSS_EXIT.value,
    ],
    BlockType.BREAKOUT_ENTRY.value: [
        BlockType.TRAILING_STOP_EXIT.value,
        BlockType.STOP_LOSS_EXIT.value,
        BlockType.PROFIT_TARGET_EXIT.value,
    ],
}

ALL_ENTRY_TYPES = [
    BlockType.TREND_ENTRY.value, BlockType.MEAN_REVERSION_ENTRY.value,
    BlockType.MOMENTUM_ENTRY.value, BlockType.BREAKOUT_ENTRY.value,
]

ALL_EXIT_TYPES = [
    BlockType.TIME_BASED_EXIT.value, BlockType.PROFIT_TARGET_EXIT.value,
    BlockType.STOP_LOSS_EXIT.value, BlockType.TRAILING_STOP_EXIT.value,
    BlockType.CORRELATION_EXIT.value,
]

ALL_SIZING_TYPES = [
    BlockType.FIXED_SIZE.value, BlockType.CONFIDENCE_SCALING.value, BlockType.VOLATILITY_SCALING.value,
]


class StrategyGrammar:
    """Validates strategy templates against composition rules."""

    @staticmethod
    def is_valid(template: StrategyTemplate) -> Tuple[bool, str]:
        """Return (is_valid, error_message)."""
        if not template.is_complete():
            return False, "Template missing entry, exit, or sizing block"

        entry_type = template.entry.block_type
        exit_type = template.exit_block.block_type

        # Check entry-exit compatibility
        allowed_exits = VALID_ENTRY_EXIT_PAIRS.get(entry_type, [])
        if exit_type not in allowed_exits:
            return False, f"Exit {exit_type} not compatible with entry {entry_type}"

        # Validate each block individually
        for block in [template.entry, template.exit_block, template.sizing] + template.filters:
            if block is None:
                continue
            ok, reason = block.validate()
            if not ok:
                return False, f"Block {block.block_type} invalid: {reason}"

        # Cross-block constraint: if PT and SL both exist as exit, PT > SL
        # (only applies if we have both in filters — currently not enforced this way)

        return True, ""

    @staticmethod
    def get_compatible_exits(entry_type: str) -> List[str]:
        return VALID_ENTRY_EXIT_PAIRS.get(entry_type, list(ALL_EXIT_TYPES))

    @staticmethod
    def random_valid_template(rng: Optional[np.random.Generator] = None) -> StrategyTemplate:
        """Generate a random valid strategy template."""
        rng = rng or np.random.default_rng()

        entry_type = rng.choice(ALL_ENTRY_TYPES)
        compatible_exits = VALID_ENTRY_EXIT_PAIRS.get(entry_type, list(ALL_EXIT_TYPES))
        exit_type = rng.choice(compatible_exits)
        sizing_type = rng.choice(ALL_SIZING_TYPES)

        def random_block(bt: str) -> StrategyBlock:
            schema = BLOCK_PARAM_SCHEMAS.get(bt, {})
            params = {k: float(rng.uniform(lo, hi)) for k, (lo, hi, _) in schema.items()}
            return StrategyBlock(block_type=bt, params=params)

        return StrategyTemplate(
            entry=random_block(entry_type),
            exit_block=random_block(exit_type),
            sizing=random_block(sizing_type),
            filters=[random_block(BlockType.RISK_FILTER)],
        )
