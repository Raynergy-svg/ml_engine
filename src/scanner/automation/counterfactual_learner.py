"""Counterfactual learning from closed trades.

Runs counterfactual analysis on losing trades and persists structured insights.

Generates insights like: "This trade would have won if volatility had been lower —
consider tightening the ATR filter."

Output redirected (Angle 2', 2026-05-10):
    Previously this module appended human-readable prose to ``.claude/learnings.md``.
    That created a coordination problem with the new outcomes JSONL ledger
    (``learnings_outcomes.jsonl``), because both writers shared a file but
    used different formats. Now CounterfactualLearner writes JSON-Lines to
    a DEDICATED ``.claude/counterfactuals.jsonl`` (gitignored — counterfactual
    analysis is sensitive). The outcomes ledger covers EVERY closed trade;
    counterfactuals.jsonl covers loss-only deeper analysis.

Backward compatibility:
    The legacy ``learnings_path`` constructor argument is preserved for
    existing call sites; if no ``output_path`` is supplied it is treated as
    the JSONL output target. Existing tests in
    ``tests/test_causal_counterfactual.py`` keep passing.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_COUNTERFACTUALS_PATH = _PROJECT_ROOT / ".claude" / "counterfactuals.jsonl"


class CounterfactualLearner:
    """Runs counterfactual analysis on closed trades and persists insights.

    Integrates with the RL feedback loop. Output goes to
    ``.claude/counterfactuals.jsonl`` (one JSON line per analysis), with
    an exclusive fcntl lock so concurrent writers do not corrupt the file.
    """

    def __init__(
        self,
        learnings_path: Optional[Path] = None,
        *,
        output_path: Optional[Path] = None,
    ):
        # ``output_path`` (new, JSONL) takes precedence; ``learnings_path``
        # (legacy) is the fallback for backward compatibility with existing
        # call sites that already pass a custom path. Either is treated as
        # the JSONL target — the file extension is informational.
        chosen = output_path or learnings_path or _DEFAULT_COUNTERFACTUALS_PATH
        self.output_path = Path(chosen)
        # Keep the legacy attribute name so existing tests can still read it.
        self.learnings_path = self.output_path

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
            Learning string to append (via ``append_learning``), or None if
            no insight.
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
        """Append a counterfactual analysis as one JSONL line.

        Schema:
            {"ts_utc": "...", "kind": "counterfactual", "text": "..."}

        Atomicity:
            ``fcntl.LOCK_EX`` for the duration of the write, plus
            ``flush`` + ``fsync`` before the lock is released. Multi-process
            safe.

        Args:
            learning_text: The free-form analysis prose. (We DON'T re-parse
                it into structured fields — the analysis is the canonical
                payload, ``text`` field carries it verbatim.)
        """
        if not learning_text:
            return

        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "kind": "counterfactual",
                "text": learning_text,
            }
            line = json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"

            with open(self.output_path, "a", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            logger.debug("Appended counterfactual to %s", self.output_path)

        except (OSError, IOError) as e:
            logger.warning("Failed to append counterfactual: %s", e)
