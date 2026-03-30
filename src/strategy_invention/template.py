"""StrategyTemplate: a complete JSON-serializable strategy specification."""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .blocks import StrategyBlock

logger = logging.getLogger(__name__)


@dataclass
class BacktestStats:
    """Results from evaluating a strategy on historical data."""
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sample_size: int = 0
    fitness: float = 0.0    # Composite score: profit_factor*0.6 + sharpe*0.4

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestStats":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StrategyTemplate:
    """A complete strategy: entry + exit + sizing + filters.

    JSON-serializable. Persisted in trained_data/strategies/.
    Lifecycle: candidate → shadow → active → retired
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "unnamed"
    entry: Optional[StrategyBlock] = None
    exit_block: Optional[StrategyBlock] = None       # 'exit' is reserved in Python
    sizing: Optional[StrategyBlock] = None
    filters: List[StrategyBlock] = field(default_factory=list)

    fitness: float = 0.0
    backtest_stats: Optional[BacktestStats] = None
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    status: str = "candidate"   # candidate, shadow, active, retired
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entry": self.entry.to_dict() if self.entry else None,
            "exit_block": self.exit_block.to_dict() if self.exit_block else None,
            "sizing": self.sizing.to_dict() if self.sizing else None,
            "filters": [f.to_dict() for f in self.filters],
            "fitness": round(self.fitness, 4),
            "backtest_stats": self.backtest_stats.to_dict() if self.backtest_stats else None,
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyTemplate":
        entry = StrategyBlock.from_dict(data["entry"]) if data.get("entry") else None
        exit_block = StrategyBlock.from_dict(data["exit_block"]) if data.get("exit_block") else None
        sizing = StrategyBlock.from_dict(data["sizing"]) if data.get("sizing") else None
        filters = [StrategyBlock.from_dict(f) for f in data.get("filters", [])]
        backtest_stats = BacktestStats.from_dict(data["backtest_stats"]) if data.get("backtest_stats") else None
        return cls(
            id=data.get("id", uuid.uuid4().hex),
            name=data.get("name", "unnamed"),
            entry=entry,
            exit_block=exit_block,
            sizing=sizing,
            filters=filters,
            fitness=data.get("fitness", 0.0),
            backtest_stats=backtest_stats,
            parent_ids=data.get("parent_ids", []),
            generation=data.get("generation", 0),
            status=data.get("status", "candidate"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )

    def is_complete(self) -> bool:
        """Return True if the template has all required blocks."""
        return self.entry is not None and self.exit_block is not None and self.sizing is not None

    def summary(self) -> str:
        """One-line summary for logging."""
        entry_t = self.entry.block_type if self.entry else "None"
        exit_t = self.exit_block.block_type if self.exit_block else "None"
        sizing_t = self.sizing.block_type if self.sizing else "None"
        return f"Strategy({self.id[:8]}) {entry_t}+{exit_t}+{sizing_t} fit={self.fitness:.3f} gen={self.generation}"
