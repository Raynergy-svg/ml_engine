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
        atr_pips = entry.get("atr_pips", 0) or 0
        confidence = entry.get("confidence", 0)
        agents = entry.get("agents", {})
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

        return entries

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

        # Don't duplicate
        if rule_line.strip() in existing:
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
                    recent = [a for a in adjustments if _parse_ts(a.get("timestamp", "")) > cutoff]
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


def _parse_ts(ts_str: str) -> float:
    """Parse an ISO timestamp to epoch seconds, return 0 on failure."""
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0
