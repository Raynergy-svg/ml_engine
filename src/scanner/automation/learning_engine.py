"""Trade outcome analyzer and learning extraction engine.

Automatically analyzes closed trades, extracts actionable patterns into
.claude/learnings.md, and promotes repeated patterns into trading rules.

US-003: Build trade outcome analyzer for automatic learning extraction.
US-004: Implement rule promotion from learnings (pattern -> rule).
US-010: Learnings consolidation and anti-proliferation guard.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LEARNINGS_PATH = Path(".claude/learnings.md")
LEARNINGS_ARCHIVE_PATH = Path(".claude/learnings-archive.md")
RULES_TRADING_PATH = Path(".claude/rules/trading.md")

CATEGORIES = [
    "sizing",
    "entry_timing",
    "sl_tp",
    "agent_accuracy",
    "regime",
    "pair_behavior",
    "override",
]


@dataclass
class LearningEntry:
    date: str
    category: str
    insight: str
    action: str
    source_trade_id: str = ""


class LearningEngine:
    """Analyzes closed trades and manages the learning -> rule promotion pipeline."""

    def __init__(
        self,
        learnings_path: Optional[Path] = None,
        rules_path: Optional[Path] = None,
    ):
        self.learnings_path = learnings_path or LEARNINGS_PATH
        self.rules_path = rules_path or RULES_TRADING_PATH
        # Initialize accuracy gate for per-pair accuracy tracking
        try:
            from src.scanner.automation.accuracy_gate import AccuracyGate
            self.accuracy_gate = AccuracyGate(min_accuracy=0.55, min_trades=5)
        except Exception as e:
            logger.warning(f"Failed to initialize AccuracyGate: {e}")
            self.accuracy_gate = None

    # ------------------------------------------------------------------
    # US-003: Trade outcome analysis
    # ------------------------------------------------------------------

    def analyze_trade(self, entry: Dict[str, Any]) -> List[LearningEntry]:
        """Examine a closed trade and return actionable learning entries.

        Analysis rules:
        1. SL hit and pnl_pips > 2x atr_pips -> 'sl_too_tight for {pair}'
        2. TP hit in <15min -> 'tp_could_be_wider for {pair}'
        3. Lost and uncertainty_score > 0.45 -> 'uncertainty_was_warning for {pair}'
        4. Won and weighted_vote_score > 0.7 -> 'high_consensus_works'
        5. model_disagreement > 0.3 and lost -> 'disagreement_predicted_loss'
        """
        entries: List[LearningEntry] = []
        outcome = entry.get("outcome")
        if not outcome:
            return entries

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        pair = entry.get("pair", "UNKNOWN")
        trade_id = str(entry.get("trade_id", ""))
        trade_won = outcome.get("trade_won", False)
        pnl_pips = abs(outcome.get("pnl_pips", 0))
        realized_pl = outcome.get("realized_pl", 0)

        # Extract context from the entry
        sl_pips = entry.get("sl_pips", 0) or 0
        tp_pips = entry.get("tp_pips", 0) or 0
        rr_ratio = round(tp_pips / sl_pips, 2) if sl_pips > 0 else 0.0
        atr_pips = entry.get("atr_pips", 0) or 0
        confidence = entry.get("confidence", 0)
        agents = entry.get("agents") or {}
        if not isinstance(agents, dict):
            agents = {}
        agent_reasons = agents.get("agent_reasons", [])

        # Extract agent metrics
        uncertainty_score = 0.0
        weighted_vote_score = 0.0
        model_disagreement = 0.0
        for ar in agent_reasons:
            name = ar.get("name", "")
            if name == "uncertainty":
                uncertainty_score = ar.get("score", 0)
            meta = ar.get("metadata", {})
            if "weighted_vote_score" in meta:
                weighted_vote_score = meta["weighted_vote_score"]
            if "model_disagreement" in meta:
                model_disagreement = meta["model_disagreement"]

        # Use top-level fields if available
        if not weighted_vote_score:
            weighted_vote_score = entry.get("weighted_vote_score", 0)
        if not model_disagreement:
            model_disagreement = entry.get("model_disagreement", 0)

        # Rule 1: SL too tight
        if not trade_won and atr_pips > 0 and pnl_pips > 2 * atr_pips:
            entries.append(LearningEntry(
                date=now,
                category="sl_tp",
                insight=f"sl_too_tight for {pair}: lost {pnl_pips:.1f}p > 2x ATR ({atr_pips:.1f}p)",
                action=f"Increase atr_sl_multiplier for {pair}",
                source_trade_id=trade_id,
            ))

        # Rule 1b: R:R ratio analysis
        if sl_pips > 0 and rr_ratio > 0:
            if rr_ratio < 1.2 and not trade_won:
                entries.append(LearningEntry(
                    date=now,
                    category="sl_tp",
                    insight=f"low_rr_ratio_loss for {pair}: R:R={rr_ratio:.2f} (<1.2), lost ${abs(realized_pl):.2f}",
                    action="Reject trades with R:R < 1.2 — gate is enforced but log for pattern tracking",
                    source_trade_id=trade_id,
                ))
            elif rr_ratio >= 2.0 and trade_won:
                entries.append(LearningEntry(
                    date=now,
                    category="sl_tp",
                    insight=f"high_rr_ratio_win for {pair}: R:R={rr_ratio:.2f} (>=2.0), won ${realized_pl:.2f}",
                    action=f"Favour high R:R setups for {pair} — consider widening TP target",
                    source_trade_id=trade_id,
                ))

        # Rule 2: TP hit too fast (< 15 min)
        if trade_won:
            open_time = entry.get("timestamp") or entry.get("open_time", "")
            close_time = outcome.get("close_time", "")
            if open_time and close_time:
                try:
                    t_open = datetime.fromisoformat(str(open_time).replace("Z", "+00:00"))
                    t_close = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
                    duration_min = (t_close - t_open).total_seconds() / 60
                    if duration_min < 15:
                        entries.append(LearningEntry(
                            date=now,
                            category="sl_tp",
                            insight=f"tp_could_be_wider for {pair}: TP hit in {duration_min:.0f}min",
                            action=f"Increase atr_tp_multiplier for {pair}",
                            source_trade_id=trade_id,
                        ))
                except (ValueError, TypeError):
                    pass

        # Rule 3: Uncertainty was warning
        if not trade_won and uncertainty_score > 0.45:
            entries.append(LearningEntry(
                date=now,
                category="agent_accuracy",
                insight=f"uncertainty_was_warning for {pair}: score={uncertainty_score:.2f}, lost ${abs(realized_pl):.2f}",
                action="Lower max_uncertainty_score threshold",
                source_trade_id=trade_id,
            ))

        # Rule 4: High consensus works
        if trade_won and weighted_vote_score > 0.7:
            entries.append(LearningEntry(
                date=now,
                category="agent_accuracy",
                insight=f"high_consensus_works: vote={weighted_vote_score:.2f}, won ${realized_pl:.2f}",
                action="Prefer high-consensus setups (>0.7 weighted vote)",
                source_trade_id=trade_id,
            ))

        # Rule 5: Disagreement predicted loss
        if not trade_won and model_disagreement > 0.3:
            entries.append(LearningEntry(
                date=now,
                category="agent_accuracy",
                insight=f"disagreement_predicted_loss: disagreement={model_disagreement:.2f}, lost ${abs(realized_pl):.2f}",
                action="Lower max_model_disagreement threshold",
                source_trade_id=trade_id,
            ))

        # Pair behavior: track direction accuracy
        direction = entry.get("direction", "")
        entries.append(LearningEntry(
            date=now,
            category="pair_behavior",
            insight=f"{pair} {direction} {'won' if trade_won else 'lost'}: {pnl_pips:.1f}p (conf={confidence:.0%})",
            action=f"Track {pair} directional accuracy",
            source_trade_id=trade_id,
        ))

        # Phase 24 (US-150): Exit reason linked to agent accountability
        exit_reason = outcome.get("exit_reason", "")
        if exit_reason and agent_reasons:
            # Pattern: exit_accountability — map agent verdicts to exit outcomes
            for ar in agent_reasons:
                agent_name = ar.get("name", "")
                agent_passed = ar.get("passed", False)
                if not agent_name:
                    continue

                if trade_won:
                    # Trade was profitable — agents who voted YES get credit
                    if agent_passed:
                        entries.append(LearningEntry(
                            date=now,
                            category="exit_accountability",
                            insight=(
                                f"exit_accountability_positive: {agent_name} voted YES, "
                                f"{pair} {direction} won via {exit_reason} "
                                f"(+{pnl_pips:.1f}p)"
                            ),
                            action=f"Maintain or boost {agent_name} weight",
                            source_trade_id=trade_id,
                        ))
                    else:
                        # Agent voted NO but trade was profitable
                        # Check if this was a fast-tracked trade
                        _analysis_ctx = entry.get("analysis_context", {})
                        _was_fasttrack = _analysis_ctx.get("fasttrack", False)
                        if _was_fasttrack:
                            entries.append(LearningEntry(
                                date=now,
                                category="exit_accountability",
                                insight=(
                                    f"agent_conservative_miss: {agent_name} voted NO, "
                                    f"but {pair} {direction} was fast-tracked and won "
                                    f"via {exit_reason} (+{pnl_pips:.1f}p)"
                                ),
                                action=(
                                    f"Consider reducing {agent_name} weight — "
                                    f"conservative bias costs profitable trades"
                                ),
                                source_trade_id=trade_id,
                            ))
                else:
                    # Trade lost — agents who voted NO were correct
                    if not agent_passed:
                        entries.append(LearningEntry(
                            date=now,
                            category="exit_accountability",
                            insight=(
                                f"exit_accountability_correct_reject: {agent_name} "
                                f"voted NO, {pair} {direction} lost via {exit_reason} "
                                f"(-{pnl_pips:.1f}p) — agent was right"
                            ),
                            action=f"Maintain or boost {agent_name} weight for rejection accuracy",
                            source_trade_id=trade_id,
                        ))

        # Record outcome in accuracy gate for per-pair gating (US-013)
        if self.accuracy_gate and pair and direction:
            try:
                self.accuracy_gate.record_outcome(
                    pair=pair,
                    predicted_direction=direction,
                    actual_outcome=trade_won,
                    confidence=confidence,
                )
            except Exception as e:
                logger.debug(f"Accuracy gate recording error for {pair}: {e}")

        return entries

    # ------------------------------------------------------------------
    # OVERRIDE_PATTERN: Override event learning extraction (PRD v2.2 §7)
    # ------------------------------------------------------------------

    def analyze_override(self, event: Dict[str, Any]) -> List[LearningEntry]:
        """Examine an override event and extract learning entries.

        Dual-engine logging: captures both market context (pair, regime, confidence)
        and human context (emotional state, cognitive load) from a single event.

        Analysis rules:
        1. Override lost → 'override_lost_{type}' (core pattern for promotion)
        2. Override won → 'override_won_{type}' (trader intuition was right)
        3. High confidence override lost → 'override_against_high_conf_lost'
        4. Emotional override lost → 'emotional_override_lost' (state-dependent)
        5. High cognitive load + loss → 'cognitive_overload_override_lost'
        6. Low confidence override won → 'low_conf_override_won' (Buddy was weak)

        Args:
            event: OverrideEvent as dict (from bridge/signals.py OverrideEvent.to_dict())

        Returns:
            List of LearningEntry for the override event.
        """
        entries: List[LearningEntry] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        outcome = event.get("outcome")
        if not outcome:
            return entries  # Can't learn without an outcome

        pair = event.get("pair", "UNKNOWN")
        override_type = event.get("override_type", "unknown")
        pnl_pips = event.get("pnl_pips", 0.0)
        emotional_state = event.get("emotional_state", "").lower()
        cognitive_load = event.get("cognitive_load", "").lower()
        confidence = event.get("confidence_at_time", 0.0)
        weighted_vote = event.get("weighted_vote_at_time", 0.0)
        regime = event.get("regime", "NORMAL")
        buddy_rec = event.get("buddy_recommendation", "")
        trader_action = event.get("trader_action", "")
        ts = event.get("timestamp", "")

        override_lost = outcome == "loss"
        override_won = outcome == "win"

        # Rule 1: Override lost — core pattern
        if override_lost:
            entries.append(LearningEntry(
                date=now,
                category="override",
                insight=(
                    f"override_lost_{override_type} on {pair}: "
                    f"Buddy said '{buddy_rec}', trader did '{trader_action}', "
                    f"lost {abs(pnl_pips):.1f}p (conf={confidence:.0%}, regime={regime})"
                ),
                action=f"Flag {override_type} overrides as risky — track loss rate",
                source_trade_id=ts,
            ))

        # Rule 2: Override won — trader intuition
        if override_won:
            entries.append(LearningEntry(
                date=now,
                category="override",
                insight=(
                    f"override_won_{override_type} on {pair}: "
                    f"Buddy said '{buddy_rec}', trader did '{trader_action}', "
                    f"won {pnl_pips:.1f}p (conf={confidence:.0%}, regime={regime})"
                ),
                action=f"Note: trader intuition correct on {override_type} — review Buddy's model for {pair}",
                source_trade_id=ts,
            ))

        # Rule 3: High confidence override lost (Buddy was confident, trader disagreed and lost)
        if override_lost and confidence >= 0.70:
            entries.append(LearningEntry(
                date=now,
                category="override",
                insight=(
                    f"override_against_high_conf_lost on {pair}: "
                    f"conf={confidence:.0%}, vote={weighted_vote:.2f}, "
                    f"lost {abs(pnl_pips):.1f}p"
                ),
                action="NEVER override when Buddy confidence >= 70% — high-confidence overrides lose",
                source_trade_id=ts,
            ))

        # Rule 4: Emotional override lost (negative emotional state + loss)
        negative_states = {"anxious", "stressed", "frustrated", "angry", "fearful", "panic", "revenge"}
        if override_lost and any(neg in emotional_state for neg in negative_states):
            entries.append(LearningEntry(
                date=now,
                category="override",
                insight=(
                    f"emotional_override_lost on {pair}: "
                    f"state='{emotional_state}', lost {abs(pnl_pips):.1f}p"
                ),
                action="NEVER override during negative emotional states — emotional overrides lose",
                source_trade_id=ts,
            ))

        # Rule 5: High cognitive load + loss
        high_load_terms = {"high", "overloaded", "exhausted", "overwhelmed"}
        if override_lost and any(term in cognitive_load for term in high_load_terms):
            entries.append(LearningEntry(
                date=now,
                category="override",
                insight=(
                    f"cognitive_overload_override_lost on {pair}: "
                    f"load='{cognitive_load}', lost {abs(pnl_pips):.1f}p"
                ),
                action="NEVER override when cognitive load is high — decision quality degrades",
                source_trade_id=ts,
            ))

        # Rule 6: Low confidence override won (Buddy was unsure, trader's gut was right)
        if override_won and confidence < 0.50:
            entries.append(LearningEntry(
                date=now,
                category="override",
                insight=(
                    f"low_conf_override_won on {pair}: "
                    f"conf={confidence:.0%}, won {pnl_pips:.1f}p — Buddy was weak"
                ),
                action=f"Review Buddy's model for {pair} when confidence < 50% — trader outperforms",
                source_trade_id=ts,
            ))

        return entries

    def analyze_overrides_batch(
        self, events: List[Dict[str, Any]]
    ) -> List[LearningEntry]:
        """Analyze a batch of override events and extract aggregate patterns.

        In addition to per-event analysis, computes aggregate metrics:
        - Overall override win rate
        - Win rate by override_type
        - Win rate by emotional_state

        Args:
            events: List of OverrideEvent dicts with outcomes.

        Returns:
            Combined list of per-event + aggregate learning entries.
        """
        all_entries: List[LearningEntry] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Per-event analysis
        for event in events:
            all_entries.extend(self.analyze_override(event))

        # Aggregate analysis (only if we have enough data)
        resolved = [e for e in events if e.get("outcome")]
        if len(resolved) < 3:
            return all_entries

        wins = sum(1 for e in resolved if e.get("outcome") == "win")
        total = len(resolved)
        if total == 0:  # C-4: guard against zero-division if guard threshold changes
            return all_entries
        overall_wr = wins / total

        # Aggregate: overall override win rate
        if overall_wr < 0.40 and total >= 5:
            all_entries.append(LearningEntry(
                date=now,
                category="override",
                insight=(
                    f"override_aggregate_loss_rate: {total - wins}/{total} overrides lost "
                    f"({1 - overall_wr:.0%} loss rate over {total} events)"
                ),
                action="Reduce override frequency — aggregate loss rate exceeds 60%",
                source_trade_id="aggregate",
            ))

        # By override_type
        from collections import Counter
        type_results: Dict[str, Dict[str, int]] = {}
        for e in resolved:
            otype = e.get("override_type", "unknown")
            if otype not in type_results:
                type_results[otype] = {"wins": 0, "total": 0}
            type_results[otype]["total"] += 1
            if e.get("outcome") == "win":
                type_results[otype]["wins"] += 1

        for otype, stats in type_results.items():
            if stats["total"] >= 3:
                wr = stats["wins"] / stats["total"]
                if wr < 0.35:
                    all_entries.append(LearningEntry(
                        date=now,
                        category="override",
                        insight=(
                            f"override_type_loss_rate_{otype}: "
                            f"{stats['total'] - stats['wins']}/{stats['total']} lost ({1 - wr:.0%})"
                        ),
                        action=f"STOP {otype} overrides — loss rate exceeds 65%",
                        source_trade_id="aggregate",
                    ))

        return all_entries

    def append_to_learnings(self, entries: List[LearningEntry]) -> int:
        """Append formatted entries to .claude/learnings.md with date prefix."""
        if not entries:
            return 0

        self.learnings_path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        for e in entries:
            lines.append(f"- **[{e.date}]** `{e.category}` | {e.insight} → *{e.action}*")

        existing = ""
        if self.learnings_path.exists():
            existing = self.learnings_path.read_text()

        # Append under a new section if date changed
        date_header = f"\n### Auto-extracted {entries[0].date}\n"
        if date_header.strip() not in existing:
            existing += date_header

        existing += "\n".join(lines) + "\n"
        self.learnings_path.write_text(existing)
        logger.info("Appended %d learning entries to %s", len(entries), self.learnings_path)
        return len(entries)

    # ------------------------------------------------------------------
    # US-004: Rule promotion
    # ------------------------------------------------------------------

    def check_promotions(self) -> List[str]:
        """Read learnings.md, count patterns, promote those with 3+ occurrences."""
        if not self.learnings_path.exists():
            return []

        content = self.learnings_path.read_text()
        lines = content.split("\n")

        # Extract pattern keys from learning entries
        # Pattern: `category` | pattern_key ...
        pattern_counter: Counter[str] = Counter()
        pattern_lines: Dict[str, List[int]] = {}

        for idx, line in enumerate(lines):
            match = re.search(r"`(\w+)`\s*\|\s*(\S+)", line)
            if match and "[PROMOTED]" not in line:
                category = match.group(1)
                pattern_key = match.group(2)
                full_key = f"{category}:{pattern_key}"
                pattern_counter[full_key] += 1
                pattern_lines.setdefault(full_key, []).append(idx)
            elif line.strip().startswith("-") and "[" in line and "]" in line and "PROMOTED" not in line:
                # Line looks like a learning entry but didn't match the pattern
                logger.warning(f"Unmatched learning line at {idx}: {line[:80]}")

        promoted: List[str] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for key, count in pattern_counter.items():
            if count < 3:
                continue

            category, pattern = key.split(":", 1)

            # Build promotion rule
            if "sl_too_tight" in pattern:
                pair = pattern.replace("sl_too_tight for ", "").strip()
                rule = f"Increase atr_sl_multiplier by 0.1 for {pair} (SL too tight observed {count} times)"
            elif "tp_could_be_wider" in pattern:
                pair = pattern.replace("tp_could_be_wider for ", "").strip()
                rule = f"Increase atr_tp_multiplier by 0.1 for {pair} (TP hit too fast {count} times)"
            elif "uncertainty_was_warning" in pattern:
                rule = f"Lower max_uncertainty_score by 0.02 (uncertainty predicted {count} losses)"
            elif "high_consensus_works" in pattern:
                rule = f"Prefer weighted_vote_score > 0.7 (high consensus won {count} times)"
            elif "disagreement_predicted_loss" in pattern:
                rule = f"Lower max_model_disagreement by 0.02 (disagreement predicted {count} losses)"
            else:
                rule = f"{pattern} observed {count} times — review and adapt"

            promotion_line = f"- [{now}] {category}: {rule} (promoted from {count} observations)"

            # Append to rules/trading.md
            self._append_rule(promotion_line)
            promoted.append(promotion_line)

            # Mark source learnings as [PROMOTED]
            for line_idx in pattern_lines.get(key, []):
                if line_idx < len(lines):
                    lines[line_idx] = lines[line_idx] + " [PROMOTED]"

        if promoted:
            self.learnings_path.write_text("\n".join(lines))
            logger.info("Promoted %d patterns to rules", len(promoted))

        return promoted

    def _append_rule(self, rule_line: str) -> None:
        """Append a promoted rule to .claude/rules/trading.md."""
        self.rules_path.parent.mkdir(parents=True, exist_ok=True)

        existing = ""
        if self.rules_path.exists():
            existing = self.rules_path.read_text()

        # Don't duplicate: compare normalized rule text (strip dates and counts)
        # Extract the core rule by removing date, source count, and parenthetical notes
        normalized_new = re.sub(r'\[\d{4}-\d{2}-\d{2}\]|\(.*?\)', '', rule_line).strip()
        for existing_line in existing.split("\n"):
            normalized_existing = re.sub(r'\[\d{4}-\d{2}-\d{2}\]|\(.*?\)', '', existing_line).strip()
            if normalized_new and normalized_existing == normalized_new:
                logger.debug(f"Rule already exists (similar): {rule_line[:60]}")
                return

        # Add under a Promoted Rules section
        promoted_header = "\n## Promoted Rules\n"
        if promoted_header.strip() not in existing:
            existing += promoted_header

        existing += rule_line + "\n"
        self.rules_path.write_text(existing)

    # ------------------------------------------------------------------
    # US-009: LLM deep analysis for significant losses
    # ------------------------------------------------------------------

    def deep_analyze_loss(
        self, entry: Dict[str, Any], pair_history: Optional[List[Dict[str, Any]]] = None
    ) -> List[LearningEntry]:
        """Use local 3B model to analyze a losing trade and suggest improvements.

        Only call for trades losing > $100. Uses llm_providers.llm_call() which
        prefers the local Buddy Planner 3B (Qwen2.5-3B-Instruct) — no API tokens needed.
        """
        outcome = entry.get("outcome", {})
        realized_pl = outcome.get("realized_pl", 0)
        if realized_pl >= 0 or abs(realized_pl) < 100:
            return []

        pair = entry.get("pair", "UNKNOWN")
        direction = entry.get("direction", "UNKNOWN")
        confidence = entry.get("confidence", 0)
        agents = entry.get("agents", {})
        regime = entry.get("volatility_regime", "UNKNOWN")

        system_prompt = (
            "You are Buddy Planner 3B, an FX trading analysis assistant. "
            "Analyze losing trades and suggest specific, actionable config improvements. "
            "Be concise. Return 1-3 insights in this exact format per line:\n"
            "CATEGORY: insight text -> action to take\n"
            "Categories: sizing, entry_timing, sl_tp, agent_accuracy, regime, pair_behavior"
        )

        prompt = (
            f"Analyze this losing FX trade:\n"
            f"Pair: {pair}, Direction: {direction}, Confidence: {confidence:.0%}\n"
            f"Entry: {entry.get('entry_price', 0)}, SL: {entry.get('sl_pips', 0)}p, TP: {entry.get('tp_pips', 0)}p\n"
            f"Outcome: ${realized_pl:.2f}, {outcome.get('pnl_pips', 0):.1f} pips\n"
            f"Regime: {regime}\n"
            f"Agent verdicts: {json.dumps(agents.get('agent_reasons', [])[:3], default=str)}\n"
        )

        if pair_history:
            recent = pair_history[-5:]
            history_str = ", ".join(
                f"{'W' if h.get('outcome', {}).get('trade_won') else 'L'} {h.get('outcome', {}).get('pnl_pips', 0):.0f}p"
                for h in recent if h.get("outcome")
            )
            prompt += f"Recent {pair} history: {history_str}\n"

        prompt += "\nWhat caused this loss and what config change would prevent it?"

        try:
            from llm_providers import llm_call, select_buddy_provider_name
            provider = select_buddy_provider_name()  # Prefers local 3B model
            response = llm_call(
                prompt,
                system_prompt=system_prompt,
                provider=provider,
                max_tokens=256,
                temperature=0.1,
            )
            if not response:
                return []
        except Exception as e:
            logger.warning(f"Local model deep analysis failed: {e}")
            return []

        # Parse LLM response into LearningEntry objects
        entries: List[LearningEntry] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for line in response.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            parts = line.split(":", 1)
            category = parts[0].strip().lower().replace(" ", "_")
            if category not in CATEGORIES:
                category = "pair_behavior"
            rest = parts[1].strip()
            action = ""
            if "->" in rest:
                insight_part, action = rest.split("->", 1)
                action = action.strip()
            else:
                insight_part = rest
                action = "Review and adapt"
            entries.append(LearningEntry(
                date=now,
                category=category,
                insight=f"[3B] {insight_part.strip()}",
                action=action,
                source_trade_id=str(entry.get("trade_id", "")),
            ))

        return entries[:3]  # Cap at 3 insights

    # ------------------------------------------------------------------
    # US-067: Exit reason pattern extraction
    # ------------------------------------------------------------------

    def extract_exit_reason_patterns(
        self,
        journal_path: Optional[Path] = None,
        min_trades: int = 5,
    ) -> List[LearningEntry]:
        """Extract actionable patterns from exit_reason data in the trade journal.

        Reads trade_journal_rl.json, groups by exit_reason per pair, and generates
        structured learnings like:
        - 'SL-hit rate > 60% on pair X → consider widening SL'
        - 'trailing_stop exits average 2.3x more pnl_pips than tp_hit'

        Args:
            journal_path: Path to trade journal (default: trained_data/trade_journal_rl.json)
            min_trades: Minimum trades per pair to analyze (default: 5)

        Returns:
            List of LearningEntry objects with exit reason insights.
        """
        jp = journal_path or Path("trained_data/trade_journal_rl.json")
        if not jp.exists():
            logger.debug("No trade journal found at %s", jp)
            return []

        try:
            data = json.loads(jp.read_text())
            if not isinstance(data, list):
                data = []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read trade journal: %s", e)
            return []

        # Group trades by pair → exit_reason with pnl_pips
        pair_exit_data: Dict[str, Dict[str, List[float]]] = {}
        for trade in data:
            outcome = trade.get("outcome", {})
            if not outcome:
                continue
            pair = trade.get("pair", "")
            exit_reason = outcome.get("exit_reason", "")
            pnl_pips = outcome.get("pnl_pips", 0)
            if not pair or not exit_reason:
                continue
            pair_exit_data.setdefault(pair, {}).setdefault(exit_reason, []).append(pnl_pips)

        entries: List[LearningEntry] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for pair, exit_groups in pair_exit_data.items():
            total_trades = sum(len(v) for v in exit_groups.values())
            if total_trades < min_trades:
                continue
            if total_trades == 0:  # C-5: explicit guard against zero-division
                continue

            # Compute distribution
            distribution: Dict[str, float] = {}
            avg_pnl: Dict[str, float] = {}
            for reason, pnl_list in exit_groups.items():
                distribution[reason] = len(pnl_list) / total_trades
                avg_pnl[reason] = sum(pnl_list) / len(pnl_list) if pnl_list else 0

            # Pattern 1: SL-hit rate > 60% → suggest widening SL
            sl_rate = distribution.get("sl_hit", 0)
            if sl_rate > 0.60:
                avg_sl_loss = avg_pnl.get("sl_hit", 0)
                entries.append(LearningEntry(
                    date=now,
                    category="sl_tp",
                    insight=(
                        f"exit_pattern:{pair} SL-hit rate {sl_rate:.0%} "
                        f"({int(sl_rate * total_trades)}/{total_trades} trades, "
                        f"avg loss {avg_sl_loss:.1f}p)"
                    ),
                    action=f"Consider widening SL for {pair} (atr_sl_multiplier +0.1)",
                ))

            # Pattern 2: TP-hit rate > 70% with fast exits → widen TP
            tp_rate = distribution.get("tp_hit", 0)
            if tp_rate > 0.70:
                avg_tp_gain = avg_pnl.get("tp_hit", 0)
                entries.append(LearningEntry(
                    date=now,
                    category="sl_tp",
                    insight=(
                        f"exit_pattern:{pair} TP-hit rate {tp_rate:.0%} "
                        f"({int(tp_rate * total_trades)}/{total_trades} trades, "
                        f"avg gain {avg_tp_gain:.1f}p)"
                    ),
                    action=f"Consider widening TP for {pair} — capturing more per trade",
                ))

            # Pattern 3: Trailing stop outperforms TP
            trailing_pnl = avg_pnl.get("trailing_stop", 0)
            tp_pnl = avg_pnl.get("tp_hit", 0)
            trailing_count = len(exit_groups.get("trailing_stop", []))
            if trailing_count >= 3 and tp_pnl > 0 and trailing_pnl > tp_pnl * 1.5:
                ratio = trailing_pnl / tp_pnl if tp_pnl else 0
                entries.append(LearningEntry(
                    date=now,
                    category="sl_tp",
                    insight=(
                        f"exit_pattern:{pair} trailing_stop avg {trailing_pnl:.1f}p "
                        f"vs tp_hit avg {tp_pnl:.1f}p ({ratio:.1f}x better)"
                    ),
                    action=f"Favor trailing stops over fixed TP for {pair}",
                ))

            # Pattern 4: High timeout rate → entry timing issue
            timeout_rate = distribution.get("timeout", 0)
            if timeout_rate > 0.30:
                entries.append(LearningEntry(
                    date=now,
                    category="entry_timing",
                    insight=(
                        f"exit_pattern:{pair} timeout rate {timeout_rate:.0%} "
                        f"({int(timeout_rate * total_trades)}/{total_trades} trades)"
                    ),
                    action=f"Review entry timing and session filters for {pair}",
                ))

            # Pattern 5: Breakeven stop dominance → tighten trailing params
            be_rate = distribution.get("breakeven_stop", 0)
            if be_rate > 0.40:
                avg_be_pnl = avg_pnl.get("breakeven_stop", 0)
                entries.append(LearningEntry(
                    date=now,
                    category="sl_tp",
                    insight=(
                        f"exit_pattern:{pair} breakeven_stop rate {be_rate:.0%} "
                        f"(avg {avg_be_pnl:.1f}p) — profits not running"
                    ),
                    action=f"Review trailing_stop_breakeven_pct for {pair} — may be too aggressive",
                ))

        if entries:
            logger.info(
                "Extracted %d exit reason patterns from %d pairs",
                len(entries),
                len(pair_exit_data),
            )

        return entries

    # ------------------------------------------------------------------
    # US-010: Consolidation and anti-proliferation
    # ------------------------------------------------------------------

    def audit(self) -> Dict[str, Any]:
        """Check file sizes and trigger consolidation if needed.

        Returns dict with audit results.
        """
        results: Dict[str, Any] = {"actions": []}

        # Check learnings.md line count
        if self.learnings_path.exists():
            learnings_lines = self.learnings_path.read_text().split("\n")
            if len(learnings_lines) > 30:
                self.consolidate()
                results["actions"].append(f"Consolidated learnings ({len(learnings_lines)} lines)")

        # Check rules/trading.md line count
        if self.rules_path.exists():
            rules_lines = self.rules_path.read_text().split("\n")
            if len(rules_lines) > 50:
                results["actions"].append(f"Rules file large ({len(rules_lines)} lines) — consider splitting")

        # Check config_adjustments.json
        adj_path = Path(".claude/config_adjustments.json")
        if adj_path.exists():
            try:
                adjustments = json.loads(adj_path.read_text())
                if len(adjustments) > 100:
                    # Archive entries older than 30 days
                    cutoff = datetime.now(timezone.utc).timestamp() - (30 * 86400)
                    recent = [a for a in adjustments if (_parse_ts(a.get("timestamp", "")) or 0) > cutoff]
                    archived = len(adjustments) - len(recent)
                    adj_path.write_text(json.dumps(recent, indent=2))
                    results["actions"].append(f"Archived {archived} old config adjustments")
            except Exception as e:
                logger.debug(f"Config adjustments audit error: {e}")

        if results["actions"]:
            logger.info("Audit results: %s", results["actions"])
        return results

    def consolidate(self) -> int:
        """Group learnings by category, archive promoted/old, keep active."""
        if not self.learnings_path.exists():
            return 0

        content = self.learnings_path.read_text()
        lines = content.split("\n")

        active: List[str] = []
        archived: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                active.append(line)
                continue
            if "[PROMOTED]" in line:
                archived.append(line)
            else:
                active.append(line)

        # Archive promoted entries
        if archived:
            LEARNINGS_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            archive_content = ""
            if LEARNINGS_ARCHIVE_PATH.exists():
                archive_content = LEARNINGS_ARCHIVE_PATH.read_text()
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            archive_content += f"\n### Archived {now}\n"
            archive_content += "\n".join(archived) + "\n"
            LEARNINGS_ARCHIVE_PATH.write_text(archive_content)

            self.learnings_path.write_text("\n".join(active))
            logger.info("Consolidated: archived %d promoted entries", len(archived))

        return len(archived)

    # ------------------------------------------------------------------
    # US-006: Per-pair adaptive SL/TP
    # ------------------------------------------------------------------

    def update_pair_sl_tp(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update per-pair SL/TP multipliers based on trade outcome.

        Returns updated config for the pair, or None if no change.
        """
        outcome = entry.get("outcome")
        if not outcome:
            return None

        pair = entry.get("pair", "")
        if not pair:
            return None

        # Validate pair name format (3 uppercase letters + underscore + 3 uppercase letters, e.g. EUR_USD)
        if not re.match(r"^[A-Z]{3}_[A-Z]{3}$", pair):
            logger.warning(f"Invalid pair name format: {pair} (expected XXX_YYY format)")
            return None

        config_path = Path("trained_data/models/pair_sl_tp_config.json")
        config: Dict[str, Any] = {}
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
            except Exception:
                pass

        pair_cfg = config.get(pair, {
            "atr_sl_multiplier": 1.0,
            "atr_tp_multiplier": 1.5,
            "last_updated": "",
            "sample_size": 0,
        })

        trade_won = outcome.get("trade_won", False)
        pnl_pips = abs(outcome.get("pnl_pips", 0))
        atr_pips = entry.get("atr_pips", 0) or 0

        changed = False

        if trade_won:
            # TP hit early check: if trade lasted < 30% estimated time
            open_time = entry.get("timestamp") or entry.get("open_time", "")
            close_time = outcome.get("close_time", "")
            if open_time and close_time:
                try:
                    t_open = datetime.fromisoformat(str(open_time).replace("Z", "+00:00"))
                    t_close = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
                    duration_min = (t_close - t_open).total_seconds() / 60
                    # If TP hit in under 15 min, widen TP
                    if duration_min < 15:
                        pair_cfg["atr_tp_multiplier"] = min(3.0, pair_cfg["atr_tp_multiplier"] + 0.05)
                        changed = True
                except (ValueError, TypeError):
                    pass
        else:
            # SL hit: check if price reversed to profit within 2x ATR
            if atr_pips > 0 and pnl_pips <= atr_pips * 1.5:
                # SL was reasonable, no change
                pass
            elif atr_pips > 0 and pnl_pips > 2 * atr_pips:
                # SL too tight — widen
                pair_cfg["atr_sl_multiplier"] = min(2.0, pair_cfg["atr_sl_multiplier"] + 0.05)
                changed = True

        pair_cfg["sample_size"] = pair_cfg.get("sample_size", 0) + 1
        pair_cfg["last_updated"] = datetime.now(timezone.utc).isoformat()
        config[pair] = pair_cfg

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2))

        return pair_cfg if changed else None


def _parse_ts(ts_str: str) -> Optional[float]:
    """Parse an ISO timestamp to epoch seconds, return None on failure."""
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None
