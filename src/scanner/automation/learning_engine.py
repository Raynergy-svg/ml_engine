"""Trade outcome analyzer and learning extraction engine.

Analyzes closed trades, extracts patterns into learnings.md,
promotes repeated patterns to rules, and consolidates knowledge.
Implementation targets: PRD US-003, US-004, US-009, US-010.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


class LearningEngine:
    """Extracts actionable learnings from trade outcomes.

    Reads and writes to the project-root-relative .claude/ directory.
    All paths are resolved against *project_root* so the engine works
    regardless of the caller's working directory.
    """

    LEARNINGS_FILE = ".claude/learnings.md"
    LEARNINGS_ARCHIVE = ".claude/learnings_archive.md"
    RULES_FILE = ".claude/rules/trading.md"

    # Thresholds aligned with trading.md rules
    HIGH_CONFIDENCE_THRESHOLD = 0.65
    UNCERTAINTY_WARNING = 0.45
    MODEL_DISAGREEMENT_WARNING = 0.30
    DEEP_LOSS_PIP_THRESHOLD = -10.0

    # Consolidation limits from improvement.md
    MAX_LEARNINGS_BEFORE_CONSOLIDATE = 30
    PROMOTION_OCCURRENCE_THRESHOLD = 3

    # -------------------------------------------------------------------
    # Pattern for parsing learning entries from learnings.md
    # Format: - [YYYY-MM-DD] **topic**: insight text
    # -------------------------------------------------------------------
    _ENTRY_RE = re.compile(
        r"^- \[(\d{4}-\d{2}-\d{2})\] "
        r"(?:\[PROMOTED\] )?"
        r"\*\*([^*]+)\*\*: (.+)$"
    )

    def __init__(self, project_root: str | Path | None = None) -> None:
        if project_root is None:
            # Walk up from this file to find the repo root (contains .claude/)
            here = Path(__file__).resolve()
            for parent in here.parents:
                if (parent / ".claude").is_dir():
                    project_root = parent
                    break
            else:
                project_root = Path.cwd()
        self._root = Path(project_root)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _today(self) -> str:
        """ISO date string for today in UTC."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _learnings_path(self) -> Path:
        return self._root / self.LEARNINGS_FILE

    def _archive_path(self) -> Path:
        return self._root / self.LEARNINGS_ARCHIVE

    def _rules_path(self) -> Path:
        return self._root / self.RULES_FILE

    def _read_learnings(self) -> str:
        """Return raw learnings.md content or empty string."""
        path = self._learnings_path()
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def _write_learnings(self, content: str) -> None:
        path = self._learnings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _parse_entries(self, text: str | None = None) -> list[dict]:
        """Parse all learning entries from learnings.md content.

        Returns list of dicts with keys: date, topic, text, promoted, raw_line.
        """
        if text is None:
            text = self._read_learnings()
        entries: list[dict] = []
        for line in text.splitlines():
            stripped = line.strip()
            m = self._ENTRY_RE.match(stripped)
            if m:
                entries.append({
                    "date": m.group(1),
                    "topic": m.group(2),
                    "text": m.group(3),
                    "promoted": "[PROMOTED]" in stripped,
                    "raw_line": stripped,
                })
        return entries

    def _extract_category(self, topic: str) -> str:
        """Extract the top-level category from a topic like 'pair_behavior/EUR_USD'."""
        return topic.split("/")[0]

    # ------------------------------------------------------------------
    # US-003: analyze_trade
    # ------------------------------------------------------------------

    def analyze_trade(self, trade_entry: dict) -> list[str]:
        """Analyze a closed trade and extract learning strings.

        Parameters
        ----------
        trade_entry : dict
            Must contain: pair, direction, pnl_pips, confidence,
            weighted_vote_score, volatility_regime, agent_reasons,
            sl_pips, tp_pips, entry_price, exit_price, duration_minutes.
            Optional: uncertainty_score, model_disagreement.

        Returns
        -------
        list[str]
            Formatted learning strings ready for learnings.md.
        """
        today = self._today()
        learnings: list[str] = []

        pair = trade_entry.get("pair", "UNKNOWN")
        direction = trade_entry.get("direction", "UNKNOWN")
        pnl_pips = trade_entry.get("pnl_pips", 0.0)
        confidence = trade_entry.get("confidence", 0.0)
        vote_score = trade_entry.get("weighted_vote_score", 0.0)
        regime = trade_entry.get("volatility_regime", "unknown")
        agent_reasons = trade_entry.get("agent_reasons", {})
        sl_pips = trade_entry.get("sl_pips", 0.0)
        tp_pips = trade_entry.get("tp_pips", 0.0)
        duration = trade_entry.get("duration_minutes", 0)

        is_win = pnl_pips > 0
        outcome_word = "won" if is_win else "lost"
        pnl_sign = "+" if is_win else ""
        conf_pct = round(confidence * 100) if confidence <= 1.0 else round(confidence)

        # 1. Confidence accuracy
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            qualifier = "confirms >65% threshold" if is_win else "challenges >65% threshold — high confidence did not predict success"
            learnings.append(
                f"[{today}] **confidence_accuracy**: High confidence ({conf_pct}%) "
                f"trade on {pair} resulted in {pnl_sign}{pnl_pips:.1f} pips — {qualifier}"
            )
        else:
            qualifier = "low-confidence hit succeeded — consider lowering threshold" if is_win else "low-confidence miss — threshold is protective"
            learnings.append(
                f"[{today}] **confidence_accuracy**: Low confidence ({conf_pct}%) "
                f"trade on {pair} resulted in {pnl_sign}{pnl_pips:.1f} pips — {qualifier}"
            )

        # 2. Agent consensus / weighted vote analysis
        if vote_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            consensus_note = "strong consensus delivered" if is_win else "strong consensus was WRONG"
        else:
            consensus_note = "weak consensus succeeded — consensus is not everything" if is_win else "weak consensus correctly predicted poor outcome"
        learnings.append(
            f"[{today}] **agent_accuracy**: {pair} {direction} with vote_score "
            f"{vote_score:.2f} {outcome_word} ({pnl_sign}{pnl_pips:.1f} pips) — {consensus_note}"
        )

        # 3. Regime analysis
        learnings.append(
            f"[{today}] **regime_impact/{regime}**: {pair} {direction} in "
            f"{regime} regime {outcome_word} {pnl_sign}{pnl_pips:.1f} pips "
            f"(conf={conf_pct}%, vote={vote_score:.2f})"
        )

        # 4. R:R outcome — did the planned R:R materialize?
        if sl_pips and sl_pips > 0:
            planned_rr = tp_pips / sl_pips if tp_pips else 0.0
            actual_rr = abs(pnl_pips) / sl_pips if is_win else -(abs(pnl_pips) / sl_pips)
            learnings.append(
                f"[{today}] **rr_outcome**: {pair} planned R:R={planned_rr:.2f}, "
                f"actual R:R={actual_rr:+.2f} — "
                f"{'target met' if actual_rr >= planned_rr else 'fell short of target'}"
            )

        # 5. Duration analysis
        if duration > 0:
            if duration < 30 and is_win:
                dur_note = "quick win — strong directional move"
            elif duration < 30 and not is_win:
                dur_note = "quick loss — entry timing may be off"
            elif duration > 240 and not is_win:
                dur_note = "slow grind loss — consider tighter time stops"
            else:
                dur_note = f"held {duration}min — normal duration"
            learnings.append(
                f"[{today}] **duration/{pair}**: {direction} trade lasted "
                f"{duration}min, {outcome_word} {pnl_sign}{pnl_pips:.1f} pips — {dur_note}"
            )

        # 6. Pair-level tracking
        learnings.append(
            f"[{today}] **pair_behavior/{pair}**: {pair} {direction} "
            f"signaled at conf={conf_pct}%, {outcome_word} {pnl_sign}{pnl_pips:.1f} pips"
        )

        # 7. Individual agent accuracy (identify right/wrong agents)
        if agent_reasons:
            correct_agents = []
            wrong_agents = []
            for agent_name, reason in agent_reasons.items():
                reason_lower = str(reason).lower()
                agent_bullish = "bullish" in reason_lower or "long" in reason_lower or "buy" in reason_lower or "positive" in reason_lower
                agent_bearish = "bearish" in reason_lower or "short" in reason_lower or "sell" in reason_lower or "negative" in reason_lower

                trade_bullish = direction.upper() in ("LONG", "BUY")

                if agent_bullish or agent_bearish:
                    agent_agreed_with_trade = (agent_bullish and trade_bullish) or (agent_bearish and not trade_bullish)
                    if (agent_agreed_with_trade and is_win) or (not agent_agreed_with_trade and not is_win):
                        correct_agents.append(agent_name)
                    else:
                        wrong_agents.append(agent_name)

            if correct_agents:
                learnings.append(
                    f"[{today}] **agent_detail/{pair}**: Correct agents: "
                    f"{', '.join(sorted(correct_agents))}"
                )
            if wrong_agents:
                learnings.append(
                    f"[{today}] **agent_detail/{pair}**: Wrong agents: "
                    f"{', '.join(sorted(wrong_agents))}"
                )

        # 8. SL hit detection
        if not is_win and sl_pips > 0 and abs(abs(pnl_pips) - sl_pips) < 1.0:
            learnings.append(
                f"[{today}] **sl_tp/{pair}**: Hit exact SL at "
                f"{abs(pnl_pips):.1f} pips (SL was {sl_pips:.1f}) — "
                f"SL performed as designed"
            )

        # 9. Uncertainty / disagreement warnings if present
        uncertainty = trade_entry.get("uncertainty_score")
        if uncertainty is not None and uncertainty > self.UNCERTAINTY_WARNING:
            qualifier = "uncertainty warning was correct — avoid these" if not is_win else "high uncertainty but still won — lucky?"
            learnings.append(
                f"[{today}] **uncertainty_signal**: {pair} uncertainty={uncertainty:.2f} "
                f"(>{self.UNCERTAINTY_WARNING}) {outcome_word} — {qualifier}"
            )

        disagreement = trade_entry.get("model_disagreement")
        if disagreement is not None and disagreement > self.MODEL_DISAGREEMENT_WARNING:
            qualifier = "model disagreement predicted this loss" if not is_win else "survived model disagreement — outlier"
            learnings.append(
                f"[{today}] **model_disagreement**: {pair} disagreement={disagreement:.2f} "
                f"(>{self.MODEL_DISAGREEMENT_WARNING}) — {qualifier}"
            )

        return learnings

    # ------------------------------------------------------------------
    # US-003: append_to_learnings
    # ------------------------------------------------------------------

    def append_to_learnings(self, entries: list[str]) -> None:
        """Append learning entries to learnings.md under today's date header.

        Skips entries whose text already exists in the file to prevent
        duplicates.
        """
        if not entries:
            return

        today = self._today()
        content = self._read_learnings()

        # Build set of existing entry texts for dedup
        existing_texts: set[str] = set()
        for parsed in self._parse_entries(content):
            existing_texts.add(parsed["text"])

        # Filter duplicates — compare the insight text portion only
        new_entries: list[str] = []
        for entry in entries:
            m = self._ENTRY_RE.match(entry.strip().lstrip("- "))
            if m:
                if m.group(3) not in existing_texts:
                    new_entries.append(entry)
            else:
                # Non-standard format — check full string
                if entry not in content:
                    new_entries.append(entry)

        if not new_entries:
            return

        # Find or create today's section header
        header = f"## {today}"
        formatted_entries = "\n".join(f"- {e}" for e in new_entries)

        if header in content:
            # Append under existing header (before next header or EOF)
            parts = content.split(header, 1)
            after_header = parts[1]

            # Find next section header
            next_header_match = re.search(r"\n## \d{4}-\d{2}-\d{2}", after_header)
            if next_header_match:
                insert_pos = next_header_match.start()
                after_header = (
                    after_header[:insert_pos]
                    + "\n" + formatted_entries
                    + after_header[insert_pos:]
                )
            else:
                after_header = after_header.rstrip("\n") + "\n" + formatted_entries + "\n"

            content = parts[0] + header + after_header
        else:
            # Create new section at the end
            content = content.rstrip("\n") + f"\n\n{header}\n\n{formatted_entries}\n"

        self._write_learnings(content)

    # ------------------------------------------------------------------
    # US-004: check_promotions
    # ------------------------------------------------------------------

    def check_promotions(self) -> list[dict]:
        """Find patterns mentioned 3+ times and return promotion candidates.

        Returns list of dicts: {pattern, count, suggested_rule}.
        Also marks source entries as [PROMOTED] in learnings.md.
        """
        content = self._read_learnings()
        entries = self._parse_entries(content)

        # Count occurrences of each top-level category (ignoring already promoted)
        category_counts: Counter = Counter()
        category_entries: defaultdict[str, list[dict]] = defaultdict(list)

        for entry in entries:
            if entry["promoted"]:
                continue
            cat = self._extract_category(entry["topic"])
            category_counts[cat] += 1
            category_entries[cat].append(entry)

        promotions: list[dict] = []
        promoted_lines: set[str] = set()

        for category, count in category_counts.items():
            if count < self.PROMOTION_OCCURRENCE_THRESHOLD:
                continue

            # Synthesize a rule from the entries
            cat_entries = category_entries[category]
            suggested_rule = self._synthesize_rule(category, cat_entries)

            promotions.append({
                "pattern": category,
                "count": count,
                "suggested_rule": suggested_rule,
            })

            # Collect lines to mark as promoted
            for entry in cat_entries:
                promoted_lines.add(entry["raw_line"])

        # Mark promoted lines in content
        if promoted_lines:
            for raw_line in promoted_lines:
                promoted_version = raw_line.replace(
                    f"**{self._ENTRY_RE.match(raw_line).group(2)}**",
                    f"[PROMOTED] **{self._ENTRY_RE.match(raw_line).group(2)}**",
                )
                content = content.replace(raw_line, promoted_version)
            self._write_learnings(content)

        return promotions

    def _synthesize_rule(self, category: str, entries: list[dict]) -> str:
        """Build a suggested rule string from a set of related entries."""
        today = self._today()
        count = len(entries)

        # Analyze win/loss patterns in the entries
        win_mentions = sum(
            1 for e in entries
            if any(w in e["text"].lower() for w in ("won", "success", "confirms", "correct", "hit tp"))
        )
        loss_mentions = sum(
            1 for e in entries
            if any(w in e["text"].lower() for w in ("lost", "wrong", "fail", "challenges", "hit sl"))
        )

        if category == "confidence_accuracy":
            if win_mentions > loss_mentions:
                return f"[Promoted {today}, {count} observations] High confidence trades (>65%) are net profitable — maintain threshold"
            else:
                return f"[Promoted {today}, {count} observations] High confidence does not guarantee success — add secondary confirmation"

        if category == "agent_accuracy":
            return f"[Promoted {today}, {count} observations] Agent consensus (weighted_vote_score) is a meaningful signal — weight it in execution gating"

        if category == "pair_behavior":
            # Identify pairs with consistent losses
            pair_pnl: defaultdict[str, list] = defaultdict(list)
            for e in entries:
                topic = e["topic"]
                if "/" in topic:
                    pair = topic.split("/", 1)[1]
                    if "won" in e["text"].lower() or "+" in e["text"]:
                        pair_pnl[pair].append("win")
                    elif "lost" in e["text"].lower() or "-" in e["text"]:
                        pair_pnl[pair].append("loss")

            problem_pairs = [p for p, results in pair_pnl.items() if results.count("loss") > results.count("win")]
            if problem_pairs:
                return f"[Promoted {today}, {count} observations] Pairs with net negative outcomes ({', '.join(problem_pairs)}) — reduce confidence or skip"
            return f"[Promoted {today}, {count} observations] Track per-pair win rates and bias toward historically profitable pairs"

        if category == "regime_impact":
            return f"[Promoted {today}, {count} observations] Volatility regime affects outcomes — factor regime into confidence scoring"

        if category == "rr_outcome":
            return f"[Promoted {today}, {count} observations] Monitor actual vs planned R:R — adjust SL/TP calibration if consistently falling short"

        if category == "sl_tp":
            return f"[Promoted {today}, {count} observations] SL levels are being hit precisely — SL calibration is accurate"

        if category == "duration":
            return f"[Promoted {today}, {count} observations] Trade duration correlates with outcome quality — consider time-based exit rules"

        if category == "uncertainty_signal":
            return f"[Promoted {today}, {count} observations] Uncertainty score >{self.UNCERTAINTY_WARNING} is a reliable negative signal — penalize or skip"

        if category == "model_disagreement":
            return f"[Promoted {today}, {count} observations] Model disagreement >{self.MODEL_DISAGREEMENT_WARNING} predicts losses — reduce position size or skip"

        if category == "agent_detail":
            return f"[Promoted {today}, {count} observations] Track per-agent accuracy and adjust RL weights accordingly"

        # Fallback for unknown categories
        return f"[Promoted {today}, {count} observations] Pattern '{category}' observed repeatedly — investigate and codify"

    # ------------------------------------------------------------------
    # US-010: consolidate
    # ------------------------------------------------------------------

    def consolidate(self) -> None:
        """Consolidate learnings.md when it exceeds 30 entries.

        Groups entries by category, archives old entries (keeping the most
        recent 20) to learnings_archive.md, and rewrites learnings.md with
        the retained entries organized by category.
        """
        content = self._read_learnings()
        entries = self._parse_entries(content)

        if len(entries) <= self.MAX_LEARNINGS_BEFORE_CONSOLIDATE:
            return

        # Sort by date descending to keep newest
        entries_sorted = sorted(entries, key=lambda e: e["date"], reverse=True)

        # Keep 20 most recent, archive the rest
        keep_count = 20
        to_keep = entries_sorted[:keep_count]
        to_archive = entries_sorted[keep_count:]

        # Write archive
        if to_archive:
            archive_path = self._archive_path()
            existing_archive = ""
            if archive_path.exists():
                existing_archive = archive_path.read_text(encoding="utf-8")

            if not existing_archive.strip():
                existing_archive = "# Buddy Learnings Archive\n\nArchived entries from learnings.md consolidation.\n\n---\n"

            today = self._today()
            archive_section = f"\n## Archived {today}\n\n"
            for entry in to_archive:
                prefix = "- [PROMOTED] " if entry["promoted"] else "- "
                archive_section += f"{prefix}[{entry['date']}] **{entry['topic']}**: {entry['text']}\n"

            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_text(
                existing_archive.rstrip("\n") + "\n" + archive_section,
                encoding="utf-8",
            )

        # Rebuild learnings.md grouped by category
        categories: defaultdict[str, list[dict]] = defaultdict(list)
        for entry in to_keep:
            cat = self._extract_category(entry["topic"])
            categories[cat].append(entry)

        # Preserve the file header
        header_match = re.match(r"(.*?---\n)", content, re.DOTALL)
        header = header_match.group(1) if header_match else "# Buddy Trading Learnings\n\n---\n"

        new_content = header + "\n"
        for cat in sorted(categories.keys()):
            cat_entries = sorted(categories[cat], key=lambda e: e["date"], reverse=True)
            new_content += f"## {cat}\n\n"
            for entry in cat_entries:
                prefix = "[PROMOTED] " if entry["promoted"] else ""
                new_content += f"- [{entry['date']}] {prefix}**{entry['topic']}**: {entry['text']}\n"
            new_content += "\n"

        self._write_learnings(new_content)

    # ------------------------------------------------------------------
    # US-010: audit
    # ------------------------------------------------------------------

    def audit(self) -> dict:
        """Return stats about current learnings state.

        Returns dict with: total_learnings, promoted_count, categories,
        oldest_entry, newest_entry.
        """
        entries = self._parse_entries()

        if not entries:
            return {
                "total_learnings": 0,
                "promoted_count": 0,
                "categories": [],
                "oldest_entry": None,
                "newest_entry": None,
            }

        categories: set[str] = set()
        promoted = 0
        dates: list[str] = []

        for entry in entries:
            categories.add(self._extract_category(entry["topic"]))
            if entry["promoted"]:
                promoted += 1
            dates.append(entry["date"])

        dates_sorted = sorted(dates)

        return {
            "total_learnings": len(entries),
            "promoted_count": promoted,
            "categories": sorted(categories),
            "oldest_entry": dates_sorted[0],
            "newest_entry": dates_sorted[-1],
        }

    # ------------------------------------------------------------------
    # US-009: deep_analyze_loss
    # ------------------------------------------------------------------

    def deep_analyze_loss(
        self, trade_entry: dict, pair_history: list[dict]
    ) -> list[str]:
        """Perform deep analysis on a losing trade vs pair history.

        Parameters
        ----------
        trade_entry : dict
            The losing trade with standard fields.
        pair_history : list[dict]
            Previous trades on the same pair, each with the same schema
            as trade_entry.

        Returns
        -------
        list[str]
            Detailed learning strings for the loss.
        """
        today = self._today()
        learnings: list[str] = []

        pair = trade_entry.get("pair", "UNKNOWN")
        direction = trade_entry.get("direction", "UNKNOWN")
        pnl_pips = trade_entry.get("pnl_pips", 0.0)
        confidence = trade_entry.get("confidence", 0.0)
        vote_score = trade_entry.get("weighted_vote_score", 0.0)
        regime = trade_entry.get("volatility_regime", "unknown")
        sl_pips = trade_entry.get("sl_pips", 0.0)
        duration = trade_entry.get("duration_minutes", 0)
        conf_pct = round(confidence * 100) if confidence <= 1.0 else round(confidence)

        if not pair_history:
            learnings.append(
                f"[{today}] **deep_loss/{pair}**: Lost {pnl_pips:.1f} pips on {pair} "
                f"{direction} — no prior history for comparison"
            )
            return learnings

        # 1. Direction analysis — was the pair trending opposite?
        same_dir_trades = [
            t for t in pair_history
            if t.get("direction", "").upper() == direction.upper()
        ]
        opposite_dir_trades = [
            t for t in pair_history
            if t.get("direction", "").upper() != direction.upper()
            and t.get("direction")
        ]

        same_dir_wins = sum(1 for t in same_dir_trades if t.get("pnl_pips", 0) > 0)
        same_dir_total = len(same_dir_trades)

        if same_dir_total > 0:
            win_rate = same_dir_wins / same_dir_total
            if win_rate < 0.4:
                learnings.append(
                    f"[{today}] **deep_loss/{pair}**: {pair} {direction} has "
                    f"historical win rate of {win_rate:.0%} ({same_dir_wins}/{same_dir_total}) "
                    f"— pair may be trending opposite to signal direction"
                )
            else:
                learnings.append(
                    f"[{today}] **deep_loss/{pair}**: {pair} {direction} has "
                    f"historical win rate of {win_rate:.0%} ({same_dir_wins}/{same_dir_total}) "
                    f"— this loss is against the historical trend"
                )

        # 2. Regime comparison — was this regime unusual for the pair?
        regime_trades = [
            t for t in pair_history if t.get("volatility_regime") == regime
        ]
        other_regime_trades = [
            t for t in pair_history if t.get("volatility_regime") != regime
        ]

        if regime_trades:
            regime_win_rate = (
                sum(1 for t in regime_trades if t.get("pnl_pips", 0) > 0)
                / len(regime_trades)
            )
            learnings.append(
                f"[{today}] **deep_loss/regime_{pair}**: {pair} in {regime} regime "
                f"has {regime_win_rate:.0%} win rate ({len(regime_trades)} trades) — "
                f"{'regime is problematic' if regime_win_rate < 0.5 else 'regime is usually favorable, this loss is an outlier'}"
            )
        else:
            learnings.append(
                f"[{today}] **deep_loss/regime_{pair}**: {pair} has no prior trades "
                f"in {regime} regime — unfamiliar conditions may have contributed to loss"
            )

        # 3. Confidence comparison — was this trade's confidence typical?
        hist_confidences = [
            t.get("confidence", 0) for t in pair_history if t.get("confidence")
        ]
        if hist_confidences:
            avg_conf = sum(hist_confidences) / len(hist_confidences)
            avg_conf_pct = round(avg_conf * 100) if avg_conf <= 1.0 else round(avg_conf)
            if confidence < avg_conf * 0.9:
                learnings.append(
                    f"[{today}] **deep_loss/conf_{pair}**: This trade's confidence "
                    f"({conf_pct}%) was below {pair}'s average ({avg_conf_pct}%) — "
                    f"below-average confidence on this pair predicts losses"
                )
            elif confidence > avg_conf * 1.1:
                learnings.append(
                    f"[{today}] **deep_loss/conf_{pair}**: This trade's confidence "
                    f"({conf_pct}%) was above {pair}'s average ({avg_conf_pct}%) — "
                    f"high confidence did not help, other factors dominated"
                )

        # 4. Agent disagreement analysis
        hist_vote_scores = [
            t.get("weighted_vote_score", 0)
            for t in pair_history
            if t.get("weighted_vote_score")
        ]
        if hist_vote_scores:
            avg_vote = sum(hist_vote_scores) / len(hist_vote_scores)
            if vote_score < avg_vote:
                learnings.append(
                    f"[{today}] **deep_loss/consensus_{pair}**: Vote score "
                    f"{vote_score:.2f} was below {pair}'s average {avg_vote:.2f} — "
                    f"weak consensus contributed to loss"
                )

        # 5. Duration comparison
        hist_durations = [
            t.get("duration_minutes", 0)
            for t in pair_history
            if t.get("duration_minutes")
        ]
        if hist_durations and duration > 0:
            avg_duration = sum(hist_durations) / len(hist_durations)
            if duration < avg_duration * 0.5:
                learnings.append(
                    f"[{today}] **deep_loss/duration_{pair}**: Loss came in "
                    f"{duration}min vs {avg_duration:.0f}min average — "
                    f"unusually fast loss suggests strong adverse move at entry"
                )
            elif duration > avg_duration * 2:
                learnings.append(
                    f"[{today}] **deep_loss/duration_{pair}**: Loss took "
                    f"{duration}min vs {avg_duration:.0f}min average — "
                    f"slow bleed suggests lack of momentum, consider tighter time stops"
                )

        # 6. SL adequacy — was the SL too tight compared to historical?
        hist_sl = [
            t.get("sl_pips", 0) for t in pair_history if t.get("sl_pips", 0) > 0
        ]
        if hist_sl and sl_pips > 0:
            avg_sl = sum(hist_sl) / len(hist_sl)
            if sl_pips < avg_sl * 0.8:
                learnings.append(
                    f"[{today}] **deep_loss/sl_{pair}**: SL of {sl_pips:.1f} pips "
                    f"was tighter than {pair}'s average SL ({avg_sl:.1f} pips) — "
                    f"consider widening SL for this pair"
                )

        # 7. Summary
        learnings.append(
            f"[{today}] **deep_loss_summary/{pair}**: {pair} {direction} lost "
            f"{pnl_pips:.1f} pips (conf={conf_pct}%, vote={vote_score:.2f}, "
            f"regime={regime}, duration={duration}min) — "
            f"analyzed against {len(pair_history)} historical trades"
        )

        return learnings
