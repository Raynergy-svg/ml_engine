"""
Journal Pattern Analyzer — Phase 50 (US-311).

Extracts actionable patterns from historical trade journal data.
Groups trades by pair, confidence band, duration band, and exit reason.
Identifies statistically significant win-rate patterns.

Usage:
    analyzer = JournalAnalyzer()
    report = analyzer.analyze("trained_data/trade_journal_rl.json")
    analyzer.save_report(report, "trained_data/models/journal_patterns.json")
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────

@dataclass
class AnalyzerConfig:
    """Configuration for journal pattern analysis."""
    min_observations: int = 20        # Minimum trades per group for significance
    confidence_bands: int = 5         # Number of confidence bins
    duration_bands: List[int] = field(default_factory=lambda: [30, 60, 120, 240, 480, 1440])
    significance_threshold: float = 0.05  # p-value threshold for pattern significance
    min_win_rate_diff: float = 0.05   # Minimum win rate difference from baseline to report


# ── Data Structures ──────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Normalized trade record for analysis."""
    pair: str
    direction: str
    confidence: float
    entry_price: float
    exit_reason: str       # sl_hit, tp_hit, trailing, timeout
    trade_won: bool
    pnl_pips: float
    duration_minutes: float
    agent_votes: int = 0
    weighted_vote_score: float = 0.0
    regime: str = "unknown"


@dataclass
class PatternResult:
    """A discovered pattern with statistical backing."""
    group_key: str         # e.g., "pair:EUR_USD", "confidence:0.6-0.7"
    group_value: str       # The specific value
    win_rate: float        # Win rate for this group
    total_trades: int      # Number of observations
    wins: int
    losses: int
    baseline_win_rate: float  # Overall win rate for comparison
    win_rate_diff: float   # Difference from baseline
    z_score: float         # Z-score for significance
    p_value: float         # Two-tailed p-value
    is_significant: bool   # p < threshold
    avg_pnl_pips: float    # Average P/L in pips
    avg_duration_min: float  # Average trade duration
    actionable_threshold: Optional[str] = None  # e.g., "confidence >= 0.7"


@dataclass
class AnalysisReport:
    """Full analysis report with all patterns."""
    total_trades: int
    baseline_win_rate: float
    baseline_avg_pnl: float
    patterns: List[PatternResult] = field(default_factory=list)
    top_patterns: List[PatternResult] = field(default_factory=list)
    pair_breakdown: Dict[str, Dict] = field(default_factory=dict)
    confidence_breakdown: Dict[str, Dict] = field(default_factory=dict)
    duration_breakdown: Dict[str, Dict] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


# ── Core Analyzer ────────────────────────────────────────────────

class JournalAnalyzer:
    """Extracts actionable patterns from trade journal data."""

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()

    def load_journal(self, path: str) -> List[TradeRecord]:
        """Load and normalize trade journal entries."""
        try:
            with open(path, "r") as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Phase 50: Failed to load journal from {path}: {e}")
            return []

        if not isinstance(raw, list):
            logger.error(f"Phase 50: Journal is not a list: {type(raw)}")
            return []

        records = []
        for entry in raw:
            try:
                record = self._normalize_entry(entry)
                if record is not None:
                    records.append(record)
            except Exception as e:
                logger.debug(f"Phase 50: Skipping malformed entry: {e}")
                continue

        logger.info(f"Phase 50: Loaded {len(records)} trade records from {path}")
        return records

    def _normalize_entry(self, entry: Dict[str, Any]) -> Optional[TradeRecord]:
        """Normalize a raw journal entry into a TradeRecord."""
        if not isinstance(entry, dict):
            return None

        outcome = entry.get("outcome")
        if outcome is None:
            return None

        # Handle both dict and legacy string outcomes
        if isinstance(outcome, dict):
            exit_reason = outcome.get("exit_reason", "unknown")
            trade_won = outcome.get("trade_won", False)
            pnl_pips = outcome.get("pnl_pips", 0.0)
            duration = outcome.get("duration_minutes", 0.0)
        elif isinstance(outcome, str):
            trade_won = outcome.lower() == "win"
            exit_reason = "tp_hit" if trade_won else "sl_hit"
            pnl_pips = 0.0
            duration = 0.0
        else:
            return None

        confidence = entry.get("confidence", entry.get("entry_confidence", 0.0))
        if not isinstance(confidence, (int, float)) or confidence <= 0:
            return None

        agents = entry.get("agents", {})
        agent_votes = 0
        wvs = 0.0
        if isinstance(agents, dict):
            agent_votes = agents.get("agent_votes", 0)
            wvs = agents.get("weighted_vote_score", 0.0)

        # Extract regime from outcome if available
        regime = "unknown"
        if isinstance(outcome, dict):
            regime = outcome.get("regime_at_entry", "unknown")

        return TradeRecord(
            pair=entry.get("pair", "unknown"),
            direction=entry.get("direction", "unknown"),
            confidence=float(confidence),
            entry_price=float(entry.get("entry_price", 0.0)),
            exit_reason=exit_reason,
            trade_won=bool(trade_won),
            pnl_pips=float(pnl_pips) if isinstance(pnl_pips, (int, float)) else 0.0,
            duration_minutes=float(duration) if isinstance(duration, (int, float)) else 0.0,
            agent_votes=int(agent_votes) if isinstance(agent_votes, (int, float)) else 0,
            weighted_vote_score=float(wvs) if isinstance(wvs, (int, float)) else 0.0,
            regime=regime,
        )

    def analyze(self, journal_path: str) -> AnalysisReport:
        """Run full analysis on trade journal."""
        records = self.load_journal(journal_path)
        if not records:
            logger.warning("Phase 50: No records to analyze")
            return AnalysisReport(total_trades=0, baseline_win_rate=0.0, baseline_avg_pnl=0.0)

        total = len(records)
        wins = sum(1 for r in records if r.trade_won)
        baseline_wr = wins / total if total > 0 else 0.0
        baseline_pnl = sum(r.pnl_pips for r in records) / total if total > 0 else 0.0

        report = AnalysisReport(
            total_trades=total,
            baseline_win_rate=baseline_wr,
            baseline_avg_pnl=baseline_pnl,
        )

        # Analyze by each dimension
        all_patterns = []

        # 1. By pair
        pair_groups = self._group_by(records, lambda r: r.pair)
        for key, group in pair_groups.items():
            pattern = self._evaluate_group(f"pair", key, group, baseline_wr)
            if pattern:
                all_patterns.append(pattern)
                report.pair_breakdown[key] = {
                    "win_rate": pattern.win_rate,
                    "total": pattern.total_trades,
                    "avg_pnl": pattern.avg_pnl_pips,
                }

        # 2. By confidence band
        conf_groups = self._group_by_confidence_band(records)
        for key, group in conf_groups.items():
            pattern = self._evaluate_group("confidence", key, group, baseline_wr)
            if pattern:
                all_patterns.append(pattern)
                report.confidence_breakdown[key] = {
                    "win_rate": pattern.win_rate,
                    "total": pattern.total_trades,
                    "avg_pnl": pattern.avg_pnl_pips,
                }

        # 3. By duration band
        dur_groups = self._group_by_duration_band(records)
        for key, group in dur_groups.items():
            pattern = self._evaluate_group("duration", key, group, baseline_wr)
            if pattern:
                all_patterns.append(pattern)
                report.duration_breakdown[key] = {
                    "win_rate": pattern.win_rate,
                    "total": pattern.total_trades,
                    "avg_pnl": pattern.avg_pnl_pips,
                }

        # 4. By exit reason
        exit_groups = self._group_by(records, lambda r: r.exit_reason)
        for key, group in exit_groups.items():
            pattern = self._evaluate_group("exit_reason", key, group, baseline_wr)
            if pattern:
                all_patterns.append(pattern)

        # 5. By direction
        dir_groups = self._group_by(records, lambda r: r.direction)
        for key, group in dir_groups.items():
            pattern = self._evaluate_group("direction", key, group, baseline_wr)
            if pattern:
                all_patterns.append(pattern)

        report.patterns = all_patterns

        # Find top patterns (significant, positive difference, enough data)
        significant = [
            p for p in all_patterns
            if p.is_significant and p.win_rate_diff >= self.config.min_win_rate_diff
        ]
        significant.sort(key=lambda p: p.win_rate_diff, reverse=True)
        report.top_patterns = significant[:5]

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        logger.info(
            f"Phase 50: Analysis complete — {total} trades, "
            f"{len(all_patterns)} patterns, {len(report.top_patterns)} significant"
        )
        return report

    def _group_by(self, records: List[TradeRecord], key_fn) -> Dict[str, List[TradeRecord]]:
        """Group records by a key function."""
        groups: Dict[str, List[TradeRecord]] = {}
        for r in records:
            k = str(key_fn(r))
            groups.setdefault(k, []).append(r)
        return groups

    def _group_by_confidence_band(self, records: List[TradeRecord]) -> Dict[str, List[TradeRecord]]:
        """Group records into confidence bands."""
        n = self.config.confidence_bands
        confs = [r.confidence for r in records]
        if not confs:
            return {}

        min_c, max_c = min(confs), max(confs)
        band_width = (max_c - min_c) / n if max_c > min_c else 0.1

        groups: Dict[str, List[TradeRecord]] = {}
        for r in records:
            band_idx = min(int((r.confidence - min_c) / band_width), n - 1)
            low = round(min_c + band_idx * band_width, 3)
            high = round(min_c + (band_idx + 1) * band_width, 3)
            key = f"{low:.3f}-{high:.3f}"
            groups.setdefault(key, []).append(r)
        return groups

    def _group_by_duration_band(self, records: List[TradeRecord]) -> Dict[str, List[TradeRecord]]:
        """Group records into duration bands."""
        bands = self.config.duration_bands
        groups: Dict[str, List[TradeRecord]] = {}
        for r in records:
            label = f">{bands[-1]}min"
            for i, threshold in enumerate(bands):
                if r.duration_minutes <= threshold:
                    low = bands[i - 1] if i > 0 else 0
                    label = f"{low}-{threshold}min"
                    break
            groups.setdefault(label, []).append(r)
        return groups

    def _evaluate_group(
        self,
        group_key: str,
        group_value: str,
        group: List[TradeRecord],
        baseline_wr: float,
    ) -> Optional[PatternResult]:
        """Evaluate a group for statistical significance."""
        n = len(group)
        if n < self.config.min_observations:
            return None

        wins = sum(1 for r in group if r.trade_won)
        wr = wins / n
        diff = wr - baseline_wr

        # Z-test for proportion difference
        z_score, p_value = self._proportion_z_test(wr, n, baseline_wr)
        is_sig = p_value < self.config.significance_threshold

        avg_pnl = sum(r.pnl_pips for r in group) / n
        avg_dur = sum(r.duration_minutes for r in group) / n

        # Generate actionable threshold
        threshold = None
        if group_key == "confidence" and diff > 0:
            low_bound = group_value.split("-")[0]
            threshold = f"confidence >= {low_bound}"
        elif group_key == "pair" and diff > 0:
            threshold = f"prefer pair {group_value}"
        elif group_key == "duration" and diff > 0:
            threshold = f"target duration {group_value}"

        return PatternResult(
            group_key=group_key,
            group_value=group_value,
            win_rate=round(wr, 4),
            total_trades=n,
            wins=wins,
            losses=n - wins,
            baseline_win_rate=round(baseline_wr, 4),
            win_rate_diff=round(diff, 4),
            z_score=round(z_score, 4),
            p_value=round(p_value, 4),
            is_significant=is_sig,
            avg_pnl_pips=round(avg_pnl, 2),
            avg_duration_min=round(avg_dur, 1),
            actionable_threshold=threshold,
        )

    @staticmethod
    def _proportion_z_test(sample_p: float, n: int, population_p: float) -> Tuple[float, float]:
        """Two-tailed Z-test for a proportion against a known population proportion."""
        if n == 0 or population_p <= 0 or population_p >= 1:
            return 0.0, 1.0

        se = math.sqrt(population_p * (1 - population_p) / n)
        if se == 0:
            return 0.0, 1.0

        z = (sample_p - population_p) / se

        # Approximate two-tailed p-value using error function
        p_value = 2 * (1 - _normal_cdf(abs(z)))
        return z, p_value

    def _generate_recommendations(self, report: AnalysisReport) -> List[str]:
        """Generate actionable recommendations from analysis."""
        recs = []

        if not report.top_patterns:
            recs.append("No statistically significant patterns found with current data. Need more trades.")
            return recs

        for p in report.top_patterns:
            direction = "higher" if p.win_rate_diff > 0 else "lower"
            recs.append(
                f"{p.group_key}={p.group_value}: {p.win_rate:.1%} win rate "
                f"({direction} than {p.baseline_win_rate:.1%} baseline by {abs(p.win_rate_diff):.1%}, "
                f"p={p.p_value:.4f}, n={p.total_trades})"
            )

        # Confidence band recommendation
        best_conf = None
        for p in report.top_patterns:
            if p.group_key == "confidence" and p.win_rate_diff > 0:
                best_conf = p
                break
        if best_conf:
            recs.append(
                f"RECOMMENDATION: Prefer trades in confidence band {best_conf.group_value} "
                f"({best_conf.win_rate:.1%} win rate vs {best_conf.baseline_win_rate:.1%} baseline)"
            )

        return recs

    def save_report(self, report: AnalysisReport, path: str) -> bool:
        """Save analysis report atomically."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                "total_trades": report.total_trades,
                "baseline_win_rate": report.baseline_win_rate,
                "baseline_avg_pnl": report.baseline_avg_pnl,
                "patterns": [asdict(p) for p in report.patterns],
                "top_patterns": [asdict(p) for p in report.top_patterns],
                "pair_breakdown": report.pair_breakdown,
                "confidence_breakdown": report.confidence_breakdown,
                "duration_breakdown": report.duration_breakdown,
                "recommendations": report.recommendations,
            }
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp, path)
            logger.info(f"Phase 50: Saved journal patterns to {path}")
            return True
        except Exception as e:
            logger.error(f"Phase 50: Failed to save report: {e}")
            return False


# ── Helpers ──────────────────────────────────────────────────────

def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ── Factory ──────────────────────────────────────────────────────

def create_default_journal_analyzer() -> JournalAnalyzer:
    """Create a JournalAnalyzer with default config."""
    return JournalAnalyzer(AnalyzerConfig())
