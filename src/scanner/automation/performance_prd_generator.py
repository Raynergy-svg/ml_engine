"""Autonomous Ralph Trigger — Performance-Driven PRD Generation.

Reads 30-day performance data, identifies weakness patterns in pairs/agents/regimes,
generates PRD stories from those gaps, and spawns ralph.sh for execution.

This closes the loop: Buddy becomes its own product manager.

Usage:
    gen = PerformancePRDGenerator(project_root="/path/to/ml_engine")
    result = gen.run()
    print(f"Gaps found: {result.gaps_found}, Stories: {result.stories_generated}")
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.scanner.automation.background_activity import get_background_activity_tracker
from src.scanner.automation.safe_json import safe_json_read, safe_json_write

logger = logging.getLogger(__name__)


@dataclass
class PerformancePRDResult:
    """Result of a single autonomous PRD generation run."""

    run_triggered: bool
    gaps_found: int
    stories_generated: int
    ralph_spawned: bool
    reason: str
    prd_branch: str = ""
    analysis_summary: Dict[str, Any] = field(default_factory=dict)


class PerformancePRDGenerator:
    """Autonomously generates PRD stories from 30-day performance gaps."""

    def __init__(
        self,
        project_root: Optional[str] = None,
        journal_path: Optional[str] = None,
        prd_path: Optional[str] = None,
        ralph_script_path: Optional[str] = None,
        min_trades_for_analysis: int = 10,
        win_rate_gap_threshold: float = 0.45,
        check_interval_hours: int = 24,
        max_gaps_per_run: int = 5,
    ) -> None:
        """Initialize the performance PRD generator.

        Args:
            project_root: Root path of ml_engine project.
            journal_path: Path to trade_journal_rl.json (optional, inferred from project_root).
            prd_path: Path to .claude/ralph/prd.json (optional).
            ralph_script_path: Path to scripts/ralph.sh (optional).
            min_trades_for_analysis: Minimum closed trades in 30d to trigger analysis.
            win_rate_gap_threshold: Win rate below this triggers a gap story.
            check_interval_hours: Hours between autonomous runs.
            max_gaps_per_run: Maximum stories per auto-generated PRD.
        """
        self._project_root = Path(project_root) if project_root else Path.cwd()
        self._journal_path = (
            Path(journal_path)
            if journal_path
            else self._project_root / "trained_data" / "trade_journal_rl.json"
        )
        self._prd_path = (
            Path(prd_path)
            if prd_path
            else self._project_root / ".claude" / "ralph" / "prd.json"
        )
        self._ralph_script_path = (
            Path(ralph_script_path)
            if ralph_script_path
            else self._project_root / "scripts" / "ralph.sh"
        )
        self._last_run_marker = (
            self._project_root / ".claude" / "ralph" / ".last_auto_prd"
        )
        self._prd_archive_dir = self._project_root / ".claude" / "ralph" / "archive"

        self._min_trades_for_analysis = min_trades_for_analysis
        self._win_rate_gap_threshold = win_rate_gap_threshold
        self._check_interval_hours = check_interval_hours
        self._max_gaps_per_run = max_gaps_per_run

        logger.debug(
            "PerformancePRDGenerator initialized: journal=%s, prd=%s",
            self._journal_path,
            self._prd_path,
        )

    def should_run(self) -> bool:
        """Check if autonomous PRD generation should run.

        Returns True if:
        - Enough time has passed since last run (check .last_auto_prd)
        - AND enough trades have closed in the last 30 days

        Returns False on any error (fail open — don't block the system).
        """
        try:
            # Check time since last run
            if self._last_run_marker.exists():
                last_run_time = float(self._last_run_marker.read_text().strip())
                elapsed_hours = (datetime.utcnow().timestamp() - last_run_time) / 3600
                if elapsed_hours < self._check_interval_hours:
                    logger.debug(
                        "Last auto_prd run was %.1f hours ago, interval=%.1f — skipping",
                        elapsed_hours,
                        self._check_interval_hours,
                    )
                    return False
            else:
                logger.debug(".last_auto_prd not found, will create on next run")

            # Check trade count in last 30 days
            analysis = self.analyze_performance()
            if "error" in analysis:
                logger.warning("analyze_performance failed: %s", analysis.get("error"))
                return False

            trade_count = analysis.get("total_trades", 0)
            if trade_count < self._min_trades_for_analysis:
                logger.debug(
                    "Only %d trades in last 30d (min %d) — skipping",
                    trade_count,
                    self._min_trades_for_analysis,
                )
                return False

            logger.info(
                "should_run=True: %d trades in 30d, interval check passed",
                trade_count,
            )
            return True

        except Exception as e:
            logger.warning("should_run() exception: %s", e, exc_info=True)
            return False

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze trade journal for the last 30 days.

        Returns:
            {
              "total_trades": int,
              "overall_win_rate": float,
              "by_pair": { "EUR_USD": {"win_rate": float, "trade_count": int, "avg_pnl": float}, ... },
              "by_regime": { "TRENDING": {"win_rate": float, "trade_count": int}, ... },
              "by_session": { "london": {"win_rate": float, "trade_count": int}, ... },
              "worst_pairs": [...],
              "worst_regimes": [...],
              "worst_sessions": [...],
              "period_days": 30
            }
            Or on error: {"error": "reason"}
        """
        try:
            if not self._journal_path.exists():
                logger.warning("Trade journal not found: %s", self._journal_path)
                return {"error": "journal_not_found"}

            # Read journal
            journal = safe_json_read(self._journal_path, default=[])
            if not isinstance(journal, list):
                logger.warning("Trade journal is not a list")
                return {"error": "journal_invalid_format"}

            # Filter last 30 days
            now = datetime.utcnow()
            cutoff = now - timedelta(days=30)
            cutoff_ts = cutoff.timestamp()

            recent_trades: List[Dict[str, Any]] = []
            for trade in journal:
                try:
                    # Try multiple timestamp field names
                    ts = None
                    for ts_field in ["closed_at", "timestamp", "exit_time"]:
                        if ts_field in trade and trade[ts_field]:
                            if isinstance(trade[ts_field], str):
                                ts = datetime.fromisoformat(
                                    trade[ts_field].replace("Z", "+00:00")
                                ).timestamp()
                            elif isinstance(trade[ts_field], (int, float)):
                                ts = float(trade[ts_field])
                            if ts:
                                break

                    if ts and ts > cutoff_ts:
                        recent_trades.append(trade)
                except Exception as e:
                    logger.debug(
                        "Skipping trade with invalid timestamp: %s", e
                    )

            logger.info(
                "analyze_performance: %d trades in last 30 days (from %d total)",
                len(recent_trades),
                len(journal),
            )

            if not recent_trades:
                return {
                    "total_trades": 0,
                    "overall_win_rate": 0.0,
                    "by_pair": {},
                    "by_regime": {},
                    "by_session": {},
                    "worst_pairs": [],
                    "worst_regimes": [],
                    "worst_sessions": [],
                    "period_days": 30,
                }

            # Compute overall stats
            total_trades = len(recent_trades)
            wins = sum(
                1
                for t in recent_trades
                if t.get("trade_won") or t.get("pnl_pips", 0) > 0
            )
            overall_win_rate = wins / total_trades if total_trades > 0 else 0.0

            # By pair
            by_pair: Dict[str, Dict[str, Any]] = {}
            for trade in recent_trades:
                pair = trade.get("pair", "UNKNOWN")
                if pair not in by_pair:
                    by_pair[pair] = {"wins": 0, "losses": 0, "total_pnl": 0.0}
                is_win = trade.get("trade_won") or trade.get("pnl_pips", 0) > 0
                by_pair[pair]["wins" if is_win else "losses"] += 1
                by_pair[pair]["total_pnl"] += trade.get("pnl_pips", 0)

            pair_stats: Dict[str, Any] = {}
            worst_pairs_list: List[tuple[str, float]] = []
            for pair, data in by_pair.items():
                total = data["wins"] + data["losses"]
                wr = data["wins"] / total if total > 0 else 0.0
                pair_stats[pair] = {
                    "win_rate": wr,
                    "trade_count": total,
                    "avg_pnl": data["total_pnl"] / total if total > 0 else 0.0,
                }
                worst_pairs_list.append((pair, wr))

            worst_pairs = sorted(worst_pairs_list, key=lambda x: x[1])[:5]

            # By regime
            by_regime: Dict[str, Dict[str, Any]] = {}
            for trade in recent_trades:
                regime = trade.get("regime", "UNKNOWN")
                if regime not in by_regime:
                    by_regime[regime] = {"wins": 0, "losses": 0}
                is_win = trade.get("trade_won") or trade.get("pnl_pips", 0) > 0
                by_regime[regime]["wins" if is_win else "losses"] += 1

            regime_stats: Dict[str, Any] = {}
            worst_regimes_list: List[tuple[str, float]] = []
            for regime, data in by_regime.items():
                total = data["wins"] + data["losses"]
                wr = data["wins"] / total if total > 0 else 0.0
                regime_stats[regime] = {"win_rate": wr, "trade_count": total}
                worst_regimes_list.append((regime, wr))

            worst_regimes = sorted(worst_regimes_list, key=lambda x: x[1])[:5]

            # By session (infer from hour of entry)
            by_session: Dict[str, Dict[str, Any]] = {}
            session_map = {
                "london": (7, 16),  # 07:00-16:00 UTC
                "newyork": (13, 22),  # 13:00-22:00 UTC
                "sydney": (21, 6),  # 21:00 UTC-06:00 UTC
                "tokyo": (23, 8),  # 23:00 UTC-08:00 UTC
            }

            for trade in recent_trades:
                try:
                    ts_str = trade.get("entry_time", trade.get("timestamp", ""))
                    if ts_str:
                        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        hour = dt.hour
                        session = None
                        for name, (start, end) in session_map.items():
                            if start <= end:
                                if start <= hour < end:
                                    session = name
                                    break
                            else:
                                if hour >= start or hour < end:
                                    session = name
                                    break
                        if session:
                            if session not in by_session:
                                by_session[session] = {"wins": 0, "losses": 0}
                            is_win = (
                                trade.get("trade_won")
                                or trade.get("pnl_pips", 0) > 0
                            )
                            by_session[session]["wins" if is_win else "losses"] += 1
                except Exception:
                    pass

            session_stats: Dict[str, Any] = {}
            worst_sessions_list: List[tuple[str, float]] = []
            for session, data in by_session.items():
                total = data["wins"] + data["losses"]
                wr = data["wins"] / total if total > 0 else 0.0
                session_stats[session] = {"win_rate": wr, "trade_count": total}
                worst_sessions_list.append((session, wr))

            worst_sessions = sorted(worst_sessions_list, key=lambda x: x[1])[:5]

            result = {
                "total_trades": total_trades,
                "overall_win_rate": overall_win_rate,
                "by_pair": pair_stats,
                "by_regime": regime_stats,
                "by_session": session_stats,
                "worst_pairs": [p[0] for p in worst_pairs],
                "worst_regimes": [r[0] for r in worst_regimes],
                "worst_sessions": [s[0] for s in worst_sessions],
                "period_days": 30,
            }

            logger.debug(
                "Performance analysis complete: %d trades, win_rate=%.2f%%",
                total_trades,
                overall_win_rate * 100,
            )
            return result

        except Exception as e:
            logger.error("analyze_performance failed: %s", e, exc_info=True)
            return {"error": str(e)}

    def identify_gaps(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance gaps from analysis.

        Only includes gaps with:
        - win_rate < win_rate_gap_threshold
        - trade_count >= 5

        Returns list of at most 5 gaps, sorted by worst win_rate first.
        """
        gaps: List[Dict[str, Any]] = []

        if "error" in analysis:
            logger.warning("Cannot identify gaps: analysis has error")
            return gaps

        overall_wr = analysis.get("overall_win_rate", 0.5)

        # Check pair performance
        for pair, stats in analysis.get("by_pair", {}).items():
            wr = stats.get("win_rate", 0.5)
            count = stats.get("trade_count", 0)
            if wr < self._win_rate_gap_threshold and count >= 5:
                gaps.append(
                    {
                        "type": "pair_underperformance",
                        "target": pair,
                        "win_rate": wr,
                        "trade_count": count,
                        "priority": 1,
                        "suggested_action": f"Reduce or suppress {pair} signals in low-confidence regimes or when news_risk > 0.5",
                    }
                )

        # Check regime performance
        for regime, stats in analysis.get("by_regime", {}).items():
            wr = stats.get("win_rate", 0.5)
            count = stats.get("trade_count", 0)
            if wr < self._win_rate_gap_threshold and count >= 5:
                gaps.append(
                    {
                        "type": "regime_weakness",
                        "target": regime,
                        "win_rate": wr,
                        "trade_count": count,
                        "priority": 2,
                        "suggested_action": f"Add regime detection filter for {regime} and suppress entries when confidence is below 0.65",
                    }
                )

        # Check session performance
        for session, stats in analysis.get("by_session", {}).items():
            wr = stats.get("win_rate", 0.5)
            count = stats.get("trade_count", 0)
            if wr < self._win_rate_gap_threshold and count >= 5:
                gaps.append(
                    {
                        "type": "session_weakness",
                        "target": session,
                        "win_rate": wr,
                        "trade_count": count,
                        "priority": 3,
                        "suggested_action": f"Increase SL buffer or reduce position size during {session} session",
                    }
                )

        # Sort by win_rate (worst first) and take top max_gaps_per_run
        gaps = sorted(gaps, key=lambda g: g["win_rate"])
        gaps = gaps[: self._max_gaps_per_run]

        logger.info("Identified %d performance gaps", len(gaps))
        return gaps

    def generate_prd_stories(
        self, gaps: List[Dict[str, Any]], analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate PRD JSON from performance gaps.

        Returns PRD dict in the exact format of .claude/ralph/prd.json.
        """
        branch_date = datetime.utcnow().strftime("%Y-%m-%d")
        branch_name = f"ralph/auto-perf-improvement-{branch_date}"

        stories: List[Dict[str, Any]] = []
        for idx, gap in enumerate(gaps, start=1):
            story_id = f"US-PERF-{idx:03d}"
            gap_type = gap["type"]
            target = gap["target"]
            wr = gap["win_rate"]
            count = gap["trade_count"]

            if gap_type == "pair_underperformance":
                title = f"Suppress {target} entries when confidence < 0.65 (win_rate={wr:.1%})"
                description = f"Pair {target} has underperformed with {wr:.1%} win rate over {count} trades (threshold={self._win_rate_gap_threshold:.1%}). Add config rule to suppress or reduce position sizing."
                acceptance_criteria = [
                    f"Add pair_blacklist or suppress rule for {target} when confidence_score < 0.65",
                    f"Unit test verifies {target} entries are blocked/reduced in test conditions",
                    "Rule is logged when triggered in production",
                    "Syntax validation passes",
                    "Documentation updated",
                ]
            elif gap_type == "regime_weakness":
                title = f"Improve {target} regime handling (win_rate={wr:.1%})"
                description = f"Regime {target} has underperformed with {wr:.1%} win rate over {count} trades. Add tighter entry filters or increase SL for this regime."
                acceptance_criteria = [
                    f"Add regime-specific config for {target} in ScannerConfig",
                    f"Increase SL multiplier or reduce position size when regime={target}",
                    f"Unit test covers regime={target} entry suppression",
                    "Syntax validation passes",
                    f"Trade logs show {target} regime handling active",
                ]
            else:  # session_weakness
                title = f"Improve {target} session performance (win_rate={wr:.1%})"
                description = f"Session {target} has underperformed with {wr:.1%} win rate over {count} trades. Tighten entry criteria or increase risk management during this session."
                acceptance_criteria = [
                    f"Add session-specific config override for {target} in ScannerConfig",
                    f"Increase SL or reduce position size during {target} session",
                    f"Unit test covers {target} session adjustments",
                    "Syntax validation passes",
                    f"Trade logs show {target} session handling active",
                ]

            story = {
                "id": story_id,
                "title": title,
                "description": description,
                "acceptanceCriteria": acceptance_criteria,
                "priority": gap.get("priority", 1),
                "passes": False,
                "notes": f"Auto-generated from performance gap: {gap_type}={target}, win_rate={wr:.1%}, n={count} trades",
            }
            stories.append(story)

        prd = {
            "project": "ML_Engine",
            "branchName": branch_name,
            "description": f"Auto-generated from 30-day performance analysis. Win rate gaps detected in {len(gaps)} areas.",
            "userStories": stories,
        }

        logger.info("Generated %d user stories for PRD", len(stories))
        return prd

    def write_prd(self, prd: Dict[str, Any]) -> bool:
        """Write PRD to .claude/ralph/prd.json and archive.

        Writes atomically using safe_json_write, and also writes a copy to the archive.

        Args:
            prd: PRD dict to write.

        Returns:
            True if write succeeded, False otherwise.
        """
        try:
            # Ensure parent directory exists
            self._prd_path.parent.mkdir(parents=True, exist_ok=True)

            # Write main PRD atomically
            if not safe_json_write(self._prd_path, prd):
                logger.error("Failed to write PRD to %s", self._prd_path)
                return False

            logger.info("Wrote PRD to %s", self._prd_path)

            # Archive copy with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self._prd_archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = (
                self._prd_archive_dir
                / f"auto_prd_{timestamp}.json"
            )
            if not safe_json_write(archive_path, prd):
                logger.warning("Failed to write archive copy to %s", archive_path)
            else:
                logger.debug("Wrote archive copy to %s", archive_path)

            return True

        except Exception as e:
            logger.error("write_prd failed: %s", e, exc_info=True)
            return False

    def spawn_ralph(self, background: bool = True) -> bool:
        """Spawn ralph.sh as a subprocess.

        Args:
            background: If True, spawn in background (don't wait). If False, wait with 60-min timeout.

        Returns:
            True if spawn succeeded, False otherwise.
        """
        try:
            if not self._ralph_script_path.exists():
                logger.error("Ralph script not found: %s", self._ralph_script_path)
                return False

            logger.info("Spawning ralph.sh (background=%s)", background)

            if background:
                # Spawn in background with log redirection
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                log_dir = self._project_root / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"ralph_auto_{timestamp}.log"
                tracker = get_background_activity_tracker()

                with open(log_path, "w") as log_file:
                    # Use Popen to spawn in background
                    # --tool claude ensures non-interactive autonomous execution (no amp approval gate)
                    proc = subprocess.Popen(
                        ["bash", str(self._ralph_script_path), "--tool", "claude"],
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        cwd=str(self._project_root),
                    )
                activity_id = tracker.start_activity(
                    kind="ralph",
                    title="Performance PRD Ralph run",
                    source="performance_prd_generator",
                    metadata={
                        "pid": proc.pid,
                        "log_path": str(log_path),
                        "command": ["bash", str(self._ralph_script_path), "--tool", "claude"],
                    },
                )
                tracker.append_event(activity_id, "Spawned autonomous Ralph run", details={"pid": proc.pid})

                logger.info("Spawned ralph.sh in background, logs: %s", log_path)
                return True
            else:
                # Synchronous with timeout
                result = subprocess.run(
                    ["bash", str(self._ralph_script_path)],
                    cwd=str(self._project_root),
                    timeout=3600,  # 60 minutes
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    logger.info("Ralph.sh completed successfully")
                    return True
                else:
                    logger.warning(
                        "Ralph.sh exited with code %d: %s",
                        result.returncode,
                        result.stderr,
                    )
                    return False

        except subprocess.TimeoutExpired:
            logger.error("Ralph.sh timed out (60 minutes)")
            return False
        except Exception as e:
            logger.error("spawn_ralph failed: %s", e, exc_info=True)
            return False

    def run(self) -> PerformancePRDResult:
        """Execute the full autonomous PRD generation cycle.

        Returns:
            PerformancePRDResult with run details.
        """
        try:
            # Check if should run
            if not self.should_run():
                return PerformancePRDResult(
                    run_triggered=False,
                    gaps_found=0,
                    stories_generated=0,
                    ralph_spawned=False,
                    reason="Cooldown interval not elapsed or insufficient trades",
                    analysis_summary={},
                )

            # Analyze performance
            analysis = self.analyze_performance()
            if "error" in analysis:
                return PerformancePRDResult(
                    run_triggered=True,
                    gaps_found=0,
                    stories_generated=0,
                    ralph_spawned=False,
                    reason=f"Performance analysis failed: {analysis['error']}",
                    analysis_summary=analysis,
                )

            # Identify gaps
            gaps = self.identify_gaps(analysis)
            if not gaps:
                # Update marker even if no gaps
                self._last_run_marker.parent.mkdir(parents=True, exist_ok=True)
                self._last_run_marker.write_text(
                    str(datetime.utcnow().timestamp())
                )
                return PerformancePRDResult(
                    run_triggered=True,
                    gaps_found=0,
                    stories_generated=0,
                    ralph_spawned=False,
                    reason="No performance gaps detected",
                    analysis_summary=analysis,
                )

            # Generate PRD stories
            prd = self.generate_prd_stories(gaps, analysis)
            stories_count = len(prd.get("userStories", []))

            # Write PRD
            if not self.write_prd(prd):
                return PerformancePRDResult(
                    run_triggered=True,
                    gaps_found=len(gaps),
                    stories_generated=stories_count,
                    ralph_spawned=False,
                    reason="Failed to write PRD to disk",
                    prd_branch=prd.get("branchName", ""),
                    analysis_summary=analysis,
                )

            # Spawn ralph
            ralph_spawned = self.spawn_ralph(background=True)

            # Update marker
            self._last_run_marker.parent.mkdir(parents=True, exist_ok=True)
            self._last_run_marker.write_text(
                str(datetime.utcnow().timestamp())
            )

            result = PerformancePRDResult(
                run_triggered=True,
                gaps_found=len(gaps),
                stories_generated=stories_count,
                ralph_spawned=ralph_spawned,
                reason="Auto PRD generation completed successfully",
                prd_branch=prd.get("branchName", ""),
                analysis_summary=analysis,
            )

            logger.info(
                "PerformancePRDGenerator.run() complete: gaps=%d, stories=%d, ralph_spawned=%s",
                len(gaps),
                stories_count,
                ralph_spawned,
            )

            return result

        except Exception as e:
            logger.error("run() failed: %s", e, exc_info=True)
            return PerformancePRDResult(
                run_triggered=False,
                gaps_found=0,
                stories_generated=0,
                ralph_spawned=False,
                reason=f"Unexpected error: {str(e)}",
                analysis_summary={},
            )
