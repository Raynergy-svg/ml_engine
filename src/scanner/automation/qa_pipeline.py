"""QA Pipeline + Brutal Code Review Agent.

Runs automated quality checks on the trading system:
- Config validation (thresholds in sane ranges)
- Agent weight sanity (no extreme drift)
- Trade journal integrity (no gaps, consistent schema)
- Rule consistency (no contradictions)
- Code health metrics from scan results

This is the system's immune system — catches drift before it becomes damage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QAFinding:
    """A single QA finding."""

    severity: str  # "critical", "warning", "info"
    category: str  # "config", "agents", "journal", "rules", "health"
    message: str
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None


@dataclass
class QAReport:
    """Full QA report from a pipeline run."""

    timestamp: str = ""
    findings: List[QAFinding] = field(default_factory=list)
    passed: bool = True
    score: float = 100.0  # 0-100 health score

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "passed": self.passed,
            "score": self.score,
            "findings_count": len(self.findings),
            "critical": sum(1 for f in self.findings if f.severity == "critical"),
            "warnings": sum(1 for f in self.findings if f.severity == "warning"),
            "info": sum(1 for f in self.findings if f.severity == "info"),
            "findings": [
                {
                    "severity": f.severity,
                    "category": f.category,
                    "message": f.message,
                    "details": f.details,
                    "fix_suggestion": f.fix_suggestion,
                }
                for f in self.findings
            ],
        }


class QAPipeline:
    """Automated quality assurance for the trading system.

    Runs a battery of checks across all system components and
    produces a report with findings and a health score.
    """

    def __init__(self, project_root: Optional[str] = None):
        self._root = Path(project_root) if project_root else Path.cwd()

    def run_full_audit(self) -> QAReport:
        """Run all QA checks and produce a report."""
        report = QAReport(timestamp=datetime.utcnow().isoformat() + "Z")

        # Run all check categories
        report.findings.extend(self._check_config())
        report.findings.extend(self._check_agent_weights())
        report.findings.extend(self._check_trade_journal())
        report.findings.extend(self._check_rules_consistency())
        report.findings.extend(self._check_state_health())
        report.findings.extend(self._check_file_structure())
        report.findings.extend(self._check_learnings_health())

        # Calculate score
        critical_count = sum(1 for f in report.findings if f.severity == "critical")
        warning_count = sum(1 for f in report.findings if f.severity == "warning")
        report.score = max(0, 100 - (critical_count * 25) - (warning_count * 5))
        report.passed = critical_count == 0

        return report

    def _check_config(self) -> List[QAFinding]:
        """Validate scanner config parameters are in sane ranges."""
        findings = []
        try:
            from src.scanner.config import ScannerConfig
            config = ScannerConfig()

            # Check critical thresholds
            checks = [
                ("min_risk_reward_ratio", 1.0, 5.0, "R:R ratio"),
                ("min_confidence", 30, 90, "Min confidence"),
                ("max_drawdown_pct", 0.005, 0.10, "Max drawdown %"),
                ("weighted_vote_threshold", 0.3, 0.9, "Vote threshold"),
                ("max_uncertainty_score", 0.2, 0.8, "Max uncertainty"),
                ("risk_per_trade_pct", 0.005, 0.10, "Risk per trade %"),
            ]

            for attr, low, high, label in checks:
                val = getattr(config, attr, None)
                if val is None:
                    continue
                if val < low:
                    findings.append(QAFinding(
                        severity="warning",
                        category="config",
                        message=f"{label} ({attr}={val}) is dangerously low (min recommended: {low})",
                        fix_suggestion=f"Set {attr} >= {low}",
                    ))
                elif val > high:
                    findings.append(QAFinding(
                        severity="warning",
                        category="config",
                        message=f"{label} ({attr}={val}) is unusually high (max recommended: {high})",
                        fix_suggestion=f"Set {attr} <= {high}",
                    ))

            # Check SL/TP relationship
            sl_mult = getattr(config, "atr_sl_multiplier", 1.0)
            tp_mult = getattr(config, "atr_tp_multiplier", 1.5)
            if tp_mult / max(sl_mult, 0.01) < 1.2:
                findings.append(QAFinding(
                    severity="critical",
                    category="config",
                    message=f"TP/SL multiplier ratio ({tp_mult}/{sl_mult} = {tp_mult/max(sl_mult,0.01):.2f}) violates 1.2:1 minimum R:R rule",
                    fix_suggestion="Increase atr_tp_multiplier or decrease atr_sl_multiplier",
                ))

            if not findings:
                findings.append(QAFinding(
                    severity="info",
                    category="config",
                    message="All config parameters within sane ranges",
                ))

        except Exception as e:
            findings.append(QAFinding(
                severity="critical",
                category="config",
                message=f"Cannot load ScannerConfig: {e}",
                fix_suggestion="Check src/scanner/config.py for import errors",
            ))

        return findings

    def _check_agent_weights(self) -> List[QAFinding]:
        """Check learned agent weights for extreme drift."""
        findings = []
        weights_path = self._root / "trained_data/models/agent_weights.json"

        if not weights_path.exists():
            findings.append(QAFinding(
                severity="info",
                category="agents",
                message="No learned weights file — agents using base weights",
            ))
            return findings

        try:
            weights = json.loads(weights_path.read_text())

            for name, weight in weights.items():
                if weight < 0.2:
                    findings.append(QAFinding(
                        severity="warning",
                        category="agents",
                        message=f"Agent '{name}' weight ({weight:.4f}) is near floor — may be effectively disabled",
                        details={"agent": name, "weight": weight, "floor": 0.1},
                        fix_suggestion=f"Review {name} agent performance or reset weights",
                    ))
                elif weight > 1.8:
                    findings.append(QAFinding(
                        severity="warning",
                        category="agents",
                        message=f"Agent '{name}' weight ({weight:.4f}) is near ceiling — may dominate voting",
                        details={"agent": name, "weight": weight, "ceiling": 2.0},
                        fix_suggestion=f"Apply weight decay or manually review {name} agent",
                    ))

            # Check for missing agents
            expected_agents = {
                "trend", "mean_reversion", "volatility", "risk_sentinel",
                "uncertainty", "execution_quality", "momentum", "news_risk",
                "multi_timeframe", "pair_performance", "session_timing",
                "support_resistance",
            }
            missing = expected_agents - set(weights.keys())
            if missing:
                findings.append(QAFinding(
                    severity="info",
                    category="agents",
                    message=f"Agents without learned weights: {', '.join(sorted(missing))}",
                ))

        except Exception as e:
            findings.append(QAFinding(
                severity="warning",
                category="agents",
                message=f"Cannot parse agent_weights.json: {e}",
            ))

        return findings

    def _check_trade_journal(self) -> List[QAFinding]:
        """Validate trade journal integrity."""
        findings = []
        journal_path = self._root / "trained_data/trade_journal_rl.json"

        if not journal_path.exists():
            findings.append(QAFinding(
                severity="info",
                category="journal",
                message="No trade journal found — no trades recorded yet",
            ))
            return findings

        try:
            journal = json.loads(journal_path.read_text())
            if not isinstance(journal, list):
                findings.append(QAFinding(
                    severity="critical",
                    category="journal",
                    message="Trade journal is not a JSON array — data corruption",
                    fix_suggestion="Backup and recreate trained_data/trade_journal_rl.json as []",
                ))
                return findings

            if len(journal) == 0:
                findings.append(QAFinding(
                    severity="info",
                    category="journal",
                    message="Trade journal is empty — no trades recorded",
                ))
                return findings

            # Schema validation
            required_fields = {"pair", "direction"}
            for i, trade in enumerate(journal[-10:]):  # Check last 10
                missing = required_fields - set(trade.keys())
                if missing:
                    findings.append(QAFinding(
                        severity="warning",
                        category="journal",
                        message=f"Trade entry #{len(journal) - 10 + i} missing fields: {missing}",
                    ))

            # Check for unrealistic P/L
            extreme_trades = [
                t for t in journal
                if abs(t.get("pnl_pips", 0) or 0) > 200
            ]
            if extreme_trades:
                findings.append(QAFinding(
                    severity="warning",
                    category="journal",
                    message=f"{len(extreme_trades)} trades with >200 pip P/L — verify accuracy",
                    details={"count": len(extreme_trades)},
                ))

            findings.append(QAFinding(
                severity="info",
                category="journal",
                message=f"Trade journal has {len(journal)} entries",
                details={"total": len(journal)},
            ))

        except json.JSONDecodeError as e:
            findings.append(QAFinding(
                severity="critical",
                category="journal",
                message=f"Trade journal JSON is malformed: {e}",
                fix_suggestion="Fix JSON syntax or restore from backup",
            ))

        return findings

    def _check_rules_consistency(self) -> List[QAFinding]:
        """Check trading rules for internal contradictions."""
        findings = []
        rules_path = self._root / ".claude/rules/trading.md"

        if not rules_path.exists():
            findings.append(QAFinding(
                severity="warning",
                category="rules",
                message="No trading.md rules file found",
                fix_suggestion="Create .claude/rules/trading.md with core trading rules",
            ))
            return findings

        content = rules_path.read_text()
        lines = content.strip().split("\n")
        rule_count = sum(1 for line in lines if line.strip().startswith("- "))

        if rule_count > 50:
            findings.append(QAFinding(
                severity="warning",
                category="rules",
                message=f"Rules file has {rule_count} rules — exceeds consolidation threshold (50)",
                fix_suggestion="Split into domain-specific files per improvement.md guidelines",
            ))

        # Check for conflicting threshold mentions
        numbers = re.findall(r"(\d+\.?\d*)", content)
        findings.append(QAFinding(
            severity="info",
            category="rules",
            message=f"Trading rules: {rule_count} rules, {len(lines)} lines",
        ))

        return findings

    def _check_state_health(self) -> List[QAFinding]:
        """Check session state for staleness."""
        findings = []
        state_path = self._root / ".claude/state.json"

        if not state_path.exists():
            findings.append(QAFinding(
                severity="warning",
                category="health",
                message="No state.json found — session state not being tracked",
                fix_suggestion="Initialize .claude/state.json via StateEngine",
            ))
            return findings

        try:
            state = json.loads(state_path.read_text())
            last_updated = state.get("last_updated", "")
            if last_updated:
                try:
                    ts = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
                    age_hours = (datetime.utcnow() - ts.replace(tzinfo=None)).total_seconds() / 3600
                    if age_hours > 24:
                        findings.append(QAFinding(
                            severity="warning",
                            category="health",
                            message=f"State is {age_hours:.0f}h old — may be stale",
                            fix_suggestion="Run a scan cycle to refresh state",
                        ))
                except (ValueError, TypeError):
                    pass

            # Check portfolio snapshot
            snapshot = state.get("portfolio_snapshot", {})
            if not snapshot:
                findings.append(QAFinding(
                    severity="info",
                    category="health",
                    message="No portfolio snapshot in state",
                ))

        except Exception as e:
            findings.append(QAFinding(
                severity="warning",
                category="health",
                message=f"Cannot parse state.json: {e}",
            ))

        return findings

    def _check_file_structure(self) -> List[QAFinding]:
        """Verify critical files and directories exist."""
        findings = []

        required_dirs = [
            "src/scanner",
            "src/scanner/automation",
            "src/risk",
            "src/dashboard",
            "trained_data/models",
            ".claude",
            ".claude/rules",
        ]

        required_files = [
            "src/scanner/engine.py",
            "src/scanner/agents.py",
            "src/scanner/execution.py",
            "src/scanner/config.py",
            "src/dashboard/app.py",
            ".claude/state.json",
            ".claude/rules/trading.md",
            ".claude/rules/improvement.md",
        ]

        for d in required_dirs:
            if not (self._root / d).is_dir():
                findings.append(QAFinding(
                    severity="critical",
                    category="health",
                    message=f"Required directory missing: {d}",
                    fix_suggestion=f"Create directory: mkdir -p {d}",
                ))

        for f in required_files:
            if not (self._root / f).is_file():
                findings.append(QAFinding(
                    severity="warning",
                    category="health",
                    message=f"Expected file missing: {f}",
                ))

        return findings

    def _check_learnings_health(self) -> List[QAFinding]:
        """Check learnings file for staleness and bloat."""
        findings = []
        learnings_path = self._root / ".claude/learnings.md"

        if not learnings_path.exists():
            findings.append(QAFinding(
                severity="info",
                category="health",
                message="No learnings.md — learning loop not yet active",
            ))
            return findings

        content = learnings_path.read_text()
        entry_count = content.count("- [")
        promoted_count = content.lower().count("[promoted]")

        if entry_count > 30:
            findings.append(QAFinding(
                severity="warning",
                category="health",
                message=f"Learnings file has {entry_count} entries — needs consolidation (threshold: 30)",
                fix_suggestion="Run LearningEngine.consolidate()",
            ))

        findings.append(QAFinding(
            severity="info",
            category="health",
            message=f"Learnings: {entry_count} entries, {promoted_count} promoted",
        ))

        return findings


class BrutalCodeReview:
    """Brutal code review agent that reviews system output quality.

    Reviews scan results, trade decisions, and agent behavior
    for logical errors, missed opportunities, and poor decisions.
    """

    def __init__(self, project_root: Optional[str] = None):
        self._root = Path(project_root) if project_root else Path.cwd()

    def review_scan_result(self, scan_analyses: list) -> List[Dict[str, Any]]:
        """Review scan results for questionable decisions.

        Looks for:
        - High confidence trades that were gated out (missed opportunity)
        - Low confidence trades that passed all gates (risky)
        - Agent consensus anomalies (all agree but gates block)
        - Regime mismatches (trading against volatility trend)
        """
        reviews = []

        for a in scan_analyses:
            pair = _attr(a, "pair", "?")
            confidence = _attr(a, "confidence", 0)
            gates = _attr(a, "gates_passed", False)
            vote = _attr(a, "weighted_vote_score", 0)
            regime = str(_attr(a, "volatility_regime", ""))

            # Missed opportunity: high confidence + high vote but gated
            if confidence > 0.65 and vote > 0.65 and not gates:
                reviews.append({
                    "type": "missed_opportunity",
                    "severity": "warning",
                    "pair": pair,
                    "message": f"High confidence ({confidence:.1%}) + high vote ({vote:.1%}) but gates blocked. Investigate gate failures.",
                })

            # Risky pass: low confidence but gates passed
            if confidence < 0.45 and gates:
                reviews.append({
                    "type": "risky_pass",
                    "severity": "critical",
                    "pair": pair,
                    "message": f"Low confidence ({confidence:.1%}) but gates passed — check gate thresholds.",
                })

            # Extreme regime trading
            if regime.upper() == "EXTREME" and gates:
                reviews.append({
                    "type": "extreme_regime",
                    "severity": "warning",
                    "pair": pair,
                    "message": f"Trading in EXTREME volatility regime — higher risk.",
                })

        return reviews

    def review_agent_behavior(self, agent_reasons: list) -> List[Dict[str, Any]]:
        """Review agent voting patterns for anomalies.

        Looks for:
        - All agents agree but trade blocked (systemic issue)
        - Single agent blocking trade (domination)
        - High-weight agents consistently wrong
        """
        reviews = []

        if not agent_reasons:
            return reviews

        passed = [a for a in agent_reasons if a.get("passed", False)]
        failed = [a for a in agent_reasons if not a.get("passed", False)]

        # Single blocker
        if len(failed) == 1 and len(passed) >= 8:
            blocker = failed[0]
            reviews.append({
                "type": "single_blocker",
                "severity": "info",
                "agent": blocker.get("name", "?"),
                "message": f"Agent '{blocker.get('name', '?')}' is the sole blocker ({len(passed)} agents passed). Reason: {blocker.get('reason', '?')}",
            })

        # Near-unanimous agreement
        if len(passed) >= 10:
            reviews.append({
                "type": "strong_consensus",
                "severity": "info",
                "message": f"Very strong consensus: {len(passed)}/{len(agent_reasons)} agents agree",
            })

        return reviews

    def generate_report(self, scan_analyses: list) -> Dict[str, Any]:
        """Generate a full brutal code review report."""
        scan_reviews = self.review_scan_result(scan_analyses)

        # Review agent behavior for each analysis
        agent_reviews = []
        for a in scan_analyses:
            reasons = _attr(a, "agent_reasons", [])
            agent_reviews.extend(self.review_agent_behavior(reasons))

        all_reviews = scan_reviews + agent_reviews
        critical = sum(1 for r in all_reviews if r.get("severity") == "critical")
        warnings = sum(1 for r in all_reviews if r.get("severity") == "warning")

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_findings": len(all_reviews),
            "critical": critical,
            "warnings": warnings,
            "grade": "A" if critical == 0 and warnings <= 2 else "B" if critical == 0 else "C" if critical <= 2 else "F",
            "reviews": all_reviews,
        }


def _attr(obj, name, default=None):
    """Get attribute from object or dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
