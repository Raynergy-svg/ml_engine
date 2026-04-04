"""Execution funnel — tracks every drop point from scan to execution.

Solves the "executing 6 but only 2 go through" problem by logging a single
summary line showing the full attrition: scanned → directional → gates → executed.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ExecutionFunnel:
    """Tracks trade candidates through the execution pipeline."""

    pairs_scanned: int = 0
    directional: int = 0       # Had a direction signal
    gates_passed: int = 0      # Passed confidence/momentum/risk gates
    capacity_filtered: int = 0  # After pre-scan capacity filter
    attempted: int = 0          # Submitted to broker
    executed: int = 0           # Confirmed by broker
    skipped: List[Dict[str, str]] = field(default_factory=list)

    def record_skip(self, pair: str, reason: str) -> None:
        self.skipped.append({"pair": pair, "reason": reason})

    def summary(self) -> str:
        """One-line funnel summary."""
        parts = []
        if self.pairs_scanned:
            parts.append(f"{self.pairs_scanned} scanned")
        if self.directional:
            parts.append(f"{self.directional} directional")
        if self.gates_passed:
            parts.append(f"{self.gates_passed} gates")
        if self.capacity_filtered:
            parts.append(f"{self.capacity_filtered} tradeable")
        if self.attempted:
            parts.append(f"{self.attempted} attempted")
        parts.append(f"{self.executed} executed")
        return " → ".join(parts)

    def log_summary(self) -> None:
        """Log the funnel at INFO level."""
        logger.info(
            "execution_funnel",
            summary=self.summary(),
            skipped_count=len(self.skipped),
            skipped=self.skipped[:5] if self.skipped else [],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pairs_scanned": self.pairs_scanned,
            "directional": self.directional,
            "gates_passed": self.gates_passed,
            "capacity_filtered": self.capacity_filtered,
            "attempted": self.attempted,
            "executed": self.executed,
            "skipped": self.skipped,
            "summary": self.summary(),
        }
