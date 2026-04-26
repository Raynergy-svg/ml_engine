"""HomeworkGenerator — closed trade + outcome → HomeworkEntry.

Pure function. No LLM call. Runs HEURISTIC_CATALOG predicates over the trade
record, ranks matches by confidence, picks primary lesson, renders structured
markdown. Operator review is the intelligence layer; Buddy surfaces facts.

See spec §4.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from src.scanner.automation.homework.heuristics import HEURISTIC_CATALOG
from src.scanner.automation.homework.types import Heuristic, HomeworkEntry

logger = logging.getLogger(__name__)


class HomeworkGenerator:
    """Produces a HomeworkEntry for a closed trade.

    Args:
        catalog: heuristic list. Default = HEURISTIC_CATALOG.
    """

    def __init__(self, catalog: Optional[List[Heuristic]] = None) -> None:
        self.catalog = catalog if catalog is not None else HEURISTIC_CATALOG

    def generate(self, trade: Dict[str, Any], outcome: Dict[str, Any]) -> HomeworkEntry:
        trade_view = self._make_trade_view(trade, outcome)
        outcome_view = self._make_outcome_view(outcome)

        matches = self._run_heuristics(trade_view, outcome_view)
        primary = max(matches, key=lambda h: h.confidence) if matches else None

        reinforce, penalize = self._score_agents(trade, outcome)
        markdown = self._render_markdown(trade, outcome, matches, primary, reinforce, penalize)
        proposed_lesson = (
            self._render_lesson(primary, trade_view, outcome_view) if primary else "No clear pattern matched."
        )

        return HomeworkEntry(
            homework_id=str(uuid.uuid4()),
            trade_id=str(trade.get("trade_id", "?")),
            generated_at=datetime.now(timezone.utc).isoformat(),
            pair=str(trade.get("pair", "?")),
            direction=str(trade.get("direction", "?")),
            entry_price=float(trade.get("entry_price", 0.0)),
            sl_price=float(trade.get("sl_price", 0.0)),
            tp_price=float(trade.get("tp_price", 0.0)),
            rr_ratio=float(trade.get("rr_ratio", 0.0)),
            confidence=float(trade.get("confidence", 0.0)),
            weighted_vote_score=float(trade.get("weighted_vote_score", 0.0)),
            regime=str(trade.get("regime", "UNKNOWN")),
            agent_verdicts=list(trade.get("agents", [])),
            close_time=str(outcome.get("close_time", "")),
            close_price=float(outcome.get("close_price", 0.0)),
            realized_pl=float(outcome.get("realized_pl", 0.0)),
            close_reason=str(outcome.get("close_reason", "?")),
            duration_minutes=int(outcome.get("duration_minutes", 0)),
            mfe_pips=float(outcome.get("mfe_pips", 0.0)),
            mae_pips=float(outcome.get("mae_pips", 0.0)),
            analysis_markdown=markdown,
            proposed_lesson=proposed_lesson,
            confidence_in_analysis=primary.confidence if primary else 0.0,
            agents_to_reinforce=reinforce,
            agents_to_penalize=penalize,
        )

    # ---------------- internals ----------------

    def _make_trade_view(self, trade: Dict[str, Any], outcome: Dict[str, Any]) -> SimpleNamespace:
        gate = trade.get("gate_details", {}) or {}
        return SimpleNamespace(
            trade_id=trade.get("trade_id"),
            pair=trade.get("pair"),
            direction=trade.get("direction"),
            entry_price=float(trade.get("entry_price", 0.0)),
            sl_price=float(trade.get("sl_price", 0.0)),
            tp_price=float(trade.get("tp_price", 0.0)),
            sl_pips=float(trade.get("sl_pips", 0.0)),
            tp_pips=float(trade.get("tp_pips", 0.0)),
            rr_ratio=float(trade.get("rr_ratio", 0.0)),
            confidence=float(trade.get("confidence", 0.0)),
            weighted_vote_score=float(trade.get("weighted_vote_score", 0.0)),
            regime=trade.get("regime", "UNKNOWN"),
            agent_verdicts=trade.get("agents", []),
            gate_details=gate,
            adx=float(gate.get("adx", 99.0)),
            rsi=float(gate.get("rsi", 50.0)),
            atr_pips=float(gate.get("atr_pips", 10.0)),
            spread_pips=float(trade.get("spread_pips", 0.0)),
            slippage_pips=float(trade.get("slippage_pips", 0.0)),
            oldest_age_days=float(trade.get("oldest_age_days", 0.0)),
            news_window=bool(trade.get("news_window", False)),
            session=trade.get("session"),
            entry_regime=trade.get("entry_regime", trade.get("regime", "UNKNOWN")),
            correlated_open_at_entry=trade.get("correlated_open_at_entry", False),
            correlated_co_losses_30min=trade.get("correlated_co_losses_30min", 0),
            matching_recent_loss_count=trade.get("matching_recent_loss_count", 0),
            training_cluster_n_examples=trade.get("training_cluster_n_examples", 999),
            expected_hold_minutes=trade.get("expected_hold_minutes", 60),
            whipsaw_reversal_within_2atr_60min=trade.get("whipsaw_reversal_within_2atr_60min", False),
        )

    def _make_outcome_view(self, outcome: Dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            close_reason=outcome.get("close_reason", "?"),
            realized_pl=float(outcome.get("realized_pl", 0.0)),
            duration_minutes=int(outcome.get("duration_minutes", 0)),
            mfe_pips=float(outcome.get("mfe_pips", 0.0)),
            mae_pips=float(outcome.get("mae_pips", 0.0)),
            close_price=float(outcome.get("close_price", 0.0)),
            close_regime=outcome.get("close_regime", outcome.get("regime", "UNKNOWN")),
        )

    def _run_heuristics(self, trade_view: Any, outcome_view: Any) -> List[Heuristic]:
        matches: List[Heuristic] = []
        for h in self.catalog:
            try:
                if h.predicate(trade_view, outcome_view):
                    matches.append(h)
            except Exception as e:
                logger.debug("Heuristic %s raised: %s — skipped", h.id, e)
        return matches

    def _score_agents(
        self, trade: Dict[str, Any], outcome: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Reinforce agents whose passed aligned with outcome; penalize the rest."""
        won = outcome.get("close_reason") == "TP"
        reinforce: List[str] = []
        penalize: List[str] = []
        for v in trade.get("agents", []) or []:
            name = v.get("name")
            passed = v.get("passed")
            if name is None or passed is None:
                continue
            if won and passed:
                reinforce.append(name)
            elif not won and not passed:
                reinforce.append(name)
            elif won and not passed:
                penalize.append(name)
            elif not won and passed:
                penalize.append(name)
        return reinforce, penalize

    def _render_lesson(self, h: Heuristic, trade: Any, outcome: Any) -> str:
        """Best-effort template fill. Falls back to the raw template on any error."""
        try:
            sl_pips = trade.sl_pips or 1.0
            atr_pips = trade.atr_pips or 1.0
            ctx = dict(
                adx=trade.adx,
                rsi=trade.rsi,
                regime=trade.regime,
                rr_ratio=trade.rr_ratio,
                wvs=trade.weighted_vote_score,
                sl_to_atr=trade.sl_pips / atr_pips,
                trend_score=next(
                    (v.get("score", 0.0) for v in trade.agent_verdicts if v.get("name") == "trend"), 0.0
                ),
                disagree=(trade.gate_details or {}).get("model_disagreement", 0.0),
                n_passed=sum(1 for v in trade.agent_verdicts if v.get("passed")),
                duration=outcome.duration_minutes,
                expected_hold=trade.expected_hold_minutes,
                slippage=trade.slippage_pips,
                slip_pct=(abs(trade.slippage_pips) / max(sl_pips, 1.0)) * 100,
                age=trade.oldest_age_days,
                entry_regime=trade.entry_regime,
                close_regime=outcome.close_regime,
                n_co_losses=trade.correlated_co_losses_30min,
                n=trade.matching_recent_loss_count,
                mfe_to_atr=outcome.mfe_pips / atr_pips,
                mae_to_sl=outcome.mae_pips / max(sl_pips, 1.0),
                n_examples=trade.training_cluster_n_examples,
            )
            return h.lesson_template.format(**ctx)
        except Exception as e:
            logger.debug("HomeworkGenerator lesson template render error for %s: %s", h.id, e)
            return h.lesson_template

    def _render_markdown(
        self,
        trade: Dict[str, Any],
        outcome: Dict[str, Any],
        matches: List[Heuristic],
        primary: Optional[Heuristic],
        reinforce: List[str],
        penalize: List[str],
    ) -> str:
        close_label = {
            "TP": "🟢 TAKE PROFIT",
            "SL": "🔴 STOPPED OUT",
            "MANUAL": "🟡 MANUAL CLOSE",
        }.get(outcome.get("close_reason", "?"), "?")
        lines: List[str] = []
        lines.append(f"# Trade #{trade.get('trade_id')} {trade.get('pair')} {trade.get('direction')} — {close_label}")
        lines.append("")
        lines.append(f"**Outcome:** {outcome.get('realized_pl'):+.2f} | held {outcome.get('duration_minutes')}min "
                     f"| MFE {outcome.get('mfe_pips'):.1f} pips | MAE {outcome.get('mae_pips'):.1f} pips")
        lines.append("")
        lines.append("## Setup at entry")
        lines.append(f"- conf {trade.get('confidence'):.2f}  ·  WVS {trade.get('weighted_vote_score'):.2f}  ·  R:R {trade.get('rr_ratio'):.2f}  ·  regime {trade.get('regime')}")
        lines.append(f"- entry {trade.get('entry_price')}  ·  SL {trade.get('sl_price')}  ·  TP {trade.get('tp_price')}")
        gate = trade.get("gate_details", {}) or {}
        if gate:
            lines.append(f"- ADX {gate.get('adx', '?')}  ·  RSI {gate.get('rsi', '?')}  ·  ATR {gate.get('atr_pips', '?')} pips")
        lines.append("")
        lines.append("## Agent verdicts")
        lines.append("| Agent | Score | Passed | Weight |")
        lines.append("|---|---|---|---|")
        for v in (trade.get("agents") or []):
            mark = "✓" if v.get("passed") else "✗"
            lines.append(f"| {v.get('name')} | {v.get('score', 0.0):.2f} | {mark} | {v.get('weight', 0.0):.2f} |")
        lines.append("")
        lines.append("## Detected patterns")
        if matches:
            for m in sorted(matches, key=lambda h: -h.confidence):
                lines.append(f"- **{m.id} {m.name}** (conf {m.confidence:.2f}, {m.source})")
        else:
            lines.append("- (no heuristic matched)")
        lines.append("")
        if primary:
            lines.append("## Buddy's analysis")
            lines.append(self._render_lesson(primary, self._make_trade_view(trade, outcome), self._make_outcome_view(outcome)))
            lines.append("")
        lines.append("## Proposed adjustments")
        lines.append(f"- Reinforce: {', '.join(reinforce) if reinforce else '(none)'}")
        lines.append(f"- Penalize: {', '.join(penalize) if penalize else '(none)'}")
        if primary:
            lines.append(f"- Confidence in analysis: {primary.confidence:.2f}")
        return "\n".join(lines)
