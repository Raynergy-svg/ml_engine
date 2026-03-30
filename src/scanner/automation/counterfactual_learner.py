"""Counterfactual learning from closed trades.

Runs counterfactual analysis on closed trades and extracts structured learnings.
Called by the RL feedback loop after each trade close.

Generates insights like: "This trade would have won if volatility had been lower —
consider tightening the ATR filter."

Learnings are written to .claude/learnings.md (appended, date-stamped).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LEARNINGS_PATH = _PROJECT_ROOT / ".claude" / "learnings.md"


class CounterfactualLearner:
    """Runs counterfactual analysis on closed trades and extracts learnings.

    Integrates with the RL feedback loop to generate structured insights.
    """

    def __init__(self, learnings_path: Optional[Path] = None):
        self.learnings_path = learnings_path or _LEARNINGS_PATH

    def analyze_closed_trade(
        self,
        trade_journal_entry: Dict[str, Any],
    ) -> Optional[str]:
        """Analyze a closed trade and return a learning string, or None.

        Args:
            trade_journal_entry: Dict with keys:
                - trade_id: str
                - pair: str
                - entry_price: float
                - exit_price: float
                - entry_features: Dict[str, float]
                - regime: str
                - pnl: float
                - closed: bool

        Returns:
            Learning string to append to learnings.md, or None if no insight.
        """
        try:
            from src.scanner.automation.causal_counterfactual import CounterfactualEngine

            trade_id = trade_journal_entry.get("trade_id", "unknown")
            pair = trade_journal_entry.get("pair", "")
            pnl = trade_journal_entry.get("pnl", 0.0)
            actual_outcome = 1.0 if pnl > 0 else 0.0

            # Build trade context for counterfactual engine
            trade_context = {
                "pair": pair,
                "features": trade_journal_entry.get("entry_features", {}),
                "regime": trade_journal_entry.get("regime", "NORMAL"),
                "outcome": actual_outcome,
            }

            # Run counterfactual analysis
            engine = CounterfactualEngine()
            results = engine.batch_analyze(trade_context)

            # Extract learnings from results
            learnings = self._extract_learnings(trade_id, pair, pnl, results)

            return learnings

        except ImportError:
            logger.debug("CounterfactualEngine not available")
            return None
        except Exception as e:
            logger.warning(f"CounterfactualLearner.analyze_closed_trade failed: {e}")
            return None

    def _extract_learnings(
        self,
        trade_id: str,
        pair: str,
        pnl: float,
        results: list,
    ) -> Optional[str]:
        """Extract high-impact learnings from counterfactual results.

        Returns:
            Learning string or None if no significant insights.
        """
        if not results:
            return None

        insights = []

        for result in results:
            delta = result.probability_delta
            scenario_name = result.scenario.scenario_name

            # Flag high-impact counterfactuals
            if abs(delta) > 0.20:  # >20% probability swing
                if delta > 0:
                    insights.append(
                        f"Trade {trade_id} ({pair}) would have had +{delta:.0%} better odds under "
                        f"{scenario_name} conditions. Consider: {result.explanation}"
                    )
                else:
                    insights.append(
                        f"Trade {trade_id} ({pair}) would have had {delta:.0%} worse odds under "
                        f"{scenario_name} conditions. Consider: {result.explanation}"
                    )

        if not insights:
            return None

        # Format as learning entry
        timestamp = datetime.now(timezone.utc).isoformat()
        learning_text = "\n".join(insights)

        return f"**{timestamp}** — Counterfactual Insights (Trade {trade_id}):\n{learning_text}"

    def append_learning(self, learning_text: str) -> None:
        """Append a learning to learnings.md.

        Args:
            learning_text: The learning string to append.
        """
        if not learning_text:
            return

        try:
            self.learnings_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.learnings_path, "a", encoding="utf-8") as f:
                f.write(learning_text + "\n\n")

            logger.debug(f"Appended learning to {self.learnings_path}")

        except Exception as e:
            logger.warning(f"Failed to append learning: {e}")
