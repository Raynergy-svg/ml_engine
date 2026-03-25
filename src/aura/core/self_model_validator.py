"""Self-Model Validation Loop — Graph integrity and freshness checking.

PRD v2.2 §13 Phase 4: "Self-model validation loops"

The self-model graph accumulates nodes and edges over time from conversations,
onboarding, and bridge signals. Without validation, the graph can develop
problems:

1. STALE NODES — Nodes not updated in weeks (e.g., old TradingState that no
   longer reflects reality)
2. ORPHAN NODES — Nodes with no edges (disconnected knowledge fragments)
3. CONTRADICTORY EDGES — Conflicting relationship types between the same nodes
   (e.g., SUPPORTS and CONFLICTS_WITH simultaneously)
4. LOW CONFIDENCE ACCUMULATION — Too many low-confidence nodes degrading
   overall graph quality
5. CONVERSATION GAPS — Long periods without conversations suggest the graph
   is becoming stale

The validator runs periodically (every T3 cycle or on-demand) and produces:
- A health report (scored 0-100)
- Specific issues found with remediation suggestions
- Auto-remediation of simple issues (confidence decay, orphan flagging)

Zero-dependency — uses only the SelfModelGraph interface and Python stdlib.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Validation Thresholds ---
STALE_NODE_DAYS = 21           # Nodes not updated in 3 weeks are stale
STALE_CONVERSATION_DAYS = 7    # No conversations in 7 days = engagement gap
ORPHAN_CONFIDENCE_DECAY = 0.05 # Reduce confidence of orphan nodes each check
LOW_CONFIDENCE_THRESHOLD = 0.3 # Nodes below this are flagged
MIN_GRAPH_NODES = 3            # Minimum nodes expected (Person + Goal + TradingState)
MAX_NODES_WITHOUT_EDGES = 5    # Flag if too many orphans


@dataclass
class ValidationIssue:
    """A single issue found during graph validation."""

    severity: str       # "info", "warning", "error"
    category: str       # "stale", "orphan", "contradiction", "low_confidence", "gap"
    description: str
    node_ids: List[str] = field(default_factory=list)
    auto_remediated: bool = False
    remediation_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "node_ids": self.node_ids,
            "auto_remediated": self.auto_remediated,
            "remediation_action": self.remediation_action,
        }


@dataclass
class ValidationReport:
    """Complete validation report for the self-model graph."""

    timestamp: str
    health_score: float          # 0-100
    total_nodes: int
    total_edges: int
    total_conversations: int
    issues: List[ValidationIssue] = field(default_factory=list)
    auto_remediations: int = 0
    category_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "health_score": round(self.health_score, 1),
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "total_conversations": self.total_conversations,
            "issues": [i.to_dict() for i in self.issues],
            "auto_remediations": self.auto_remediations,
            "category_scores": {
                k: round(v, 1) for k, v in self.category_scores.items()
            },
        }

    def format_report(self) -> str:
        """Format as human-readable CLI output."""
        lines = [f"═══ Self-Model Health: {self.health_score:.0f}/100 ═══"]
        lines.append(
            f"Nodes: {self.total_nodes} | Edges: {self.total_edges} | "
            f"Conversations: {self.total_conversations}"
        )

        if self.category_scores:
            lines.append("")
            for cat, score in sorted(
                self.category_scores.items(), key=lambda x: x[1]
            ):
                bar_len = int(score / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"  {cat:20s} {bar} {score:.0f}")

        if self.issues:
            lines.append("")
            errors = [i for i in self.issues if i.severity == "error"]
            warnings = [i for i in self.issues if i.severity == "warning"]
            infos = [i for i in self.issues if i.severity == "info"]

            if errors:
                lines.append("─── Errors ───")
                for issue in errors:
                    prefix = "🔴"
                    lines.append(f"  {prefix} {issue.description}")
                    if issue.auto_remediated:
                        lines.append(f"     → Auto-fix: {issue.remediation_action}")

            if warnings:
                lines.append("─── Warnings ───")
                for issue in warnings:
                    prefix = "🟡"
                    lines.append(f"  {prefix} {issue.description}")
                    if issue.auto_remediated:
                        lines.append(f"     → Auto-fix: {issue.remediation_action}")

            if infos:
                lines.append("─── Info ───")
                for issue in infos[:5]:  # Cap info display
                    lines.append(f"  ℹ️ {issue.description}")

        if self.auto_remediations > 0:
            lines.append(f"\n✅ {self.auto_remediations} issues auto-remediated")

        return "\n".join(lines)


class SelfModelValidator:
    """Validates and maintains the self-model graph health.

    Runs a battery of checks against the graph and produces a health report.
    Can optionally auto-remediate simple issues (stale confidence decay,
    orphan flagging).

    Args:
        graph: SelfModelGraph instance to validate
        auto_remediate: Whether to automatically fix simple issues
        report_dir: Where to persist validation reports
    """

    def __init__(
        self,
        graph=None,
        auto_remediate: bool = True,
        report_dir: Optional[Path] = None,
    ):
        self._graph = graph
        self._auto_remediate = auto_remediate
        self._report_dir = report_dir or Path(".aura/validation_reports")
        self._report_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, graph=None) -> ValidationReport:
        """Run full validation against the self-model graph.

        Args:
            graph: Optional graph override (uses constructor graph if None)

        Returns:
            ValidationReport with health score and issues
        """
        g = graph or self._graph
        if g is None:
            return ValidationReport(
                timestamp=datetime.now(timezone.utc).isoformat(),
                health_score=0.0,
                total_nodes=0,
                total_edges=0,
                total_conversations=0,
                issues=[ValidationIssue(
                    severity="error",
                    category="configuration",
                    description="No graph provided for validation",
                )],
            )

        stats = g.get_stats()
        now = datetime.now(timezone.utc)

        issues: List[ValidationIssue] = []
        auto_remediations = 0

        # --- Check 1: Graph existence and minimum structure ---
        issues.extend(self._check_minimum_structure(g, stats))

        # --- Check 2: Stale nodes ---
        stale_issues, stale_fixes = self._check_stale_nodes(g, now)
        issues.extend(stale_issues)
        auto_remediations += stale_fixes

        # --- Check 3: Orphan nodes (no edges) ---
        orphan_issues, orphan_fixes = self._check_orphan_nodes(g)
        issues.extend(orphan_issues)
        auto_remediations += orphan_fixes

        # --- Check 4: Contradictory edges ---
        issues.extend(self._check_contradictory_edges(g))

        # --- Check 5: Low confidence nodes ---
        issues.extend(self._check_low_confidence(g))

        # --- Check 6: Conversation freshness ---
        issues.extend(self._check_conversation_freshness(g, now))

        # --- Check 7: Edge dangling references ---
        issues.extend(self._check_dangling_edges(g))

        # Compute health score
        category_scores = self._compute_category_scores(issues, stats)
        health_score = sum(category_scores.values()) / max(len(category_scores), 1)

        report = ValidationReport(
            timestamp=now.isoformat(),
            health_score=health_score,
            total_nodes=stats.get("total_nodes", 0),
            total_edges=stats.get("total_edges", 0),
            total_conversations=stats.get("total_conversations", 0),
            issues=issues,
            auto_remediations=auto_remediations,
            category_scores=category_scores,
        )

        # Persist report
        self._save_report(report)

        return report

    # --- Individual Checks ---

    def _check_minimum_structure(
        self, graph, stats: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Check graph has minimum expected structure."""
        issues = []
        nodes_by_type = stats.get("nodes_by_type", {})

        if stats.get("total_nodes", 0) < MIN_GRAPH_NODES:
            issues.append(ValidationIssue(
                severity="error",
                category="structure",
                description=(
                    f"Graph has only {stats.get('total_nodes', 0)} nodes "
                    f"(minimum {MIN_GRAPH_NODES} expected — need Person, Goal, TradingState)"
                ),
            ))

        if not nodes_by_type.get("person", 0):
            issues.append(ValidationIssue(
                severity="error",
                category="structure",
                description="No Person node found — onboarding may not have completed",
            ))

        if not nodes_by_type.get("goal", 0):
            issues.append(ValidationIssue(
                severity="warning",
                category="structure",
                description="No Goal nodes found — user goals should be captured during onboarding",
            ))

        return issues

    def _check_stale_nodes(
        self, graph, now: datetime
    ) -> Tuple[List[ValidationIssue], int]:
        """Check for nodes not updated recently."""
        issues = []
        fixes = 0
        cutoff = now - timedelta(days=STALE_NODE_DAYS)

        # Query all nodes and check updated_at
        try:
            all_nodes = []
            from src.aura.core.self_model import NodeType
            for node_type in NodeType:
                all_nodes.extend(graph.get_nodes_by_type(node_type))
        except Exception:
            return issues, fixes

        stale_nodes = []
        for node in all_nodes:
            try:
                updated = datetime.fromisoformat(
                    node.updated_at.replace("Z", "+00:00")
                )
                if updated < cutoff:
                    stale_nodes.append(node)
            except (ValueError, AttributeError):
                stale_nodes.append(node)  # Can't parse = treat as stale

        if stale_nodes:
            # Group by type for cleaner reporting
            by_type: Dict[str, List[str]] = {}
            for node in stale_nodes:
                t = node.node_type.value
                by_type.setdefault(t, []).append(node.id)

            for node_type, node_ids in by_type.items():
                issues.append(ValidationIssue(
                    severity="warning",
                    category="stale",
                    description=(
                        f"{len(node_ids)} stale {node_type} node(s) — "
                        f"not updated in {STALE_NODE_DAYS}+ days"
                    ),
                    node_ids=node_ids,
                ))

            # Auto-remediate: decay confidence of stale nodes
            if self._auto_remediate:
                for node in stale_nodes:
                    if node.confidence > LOW_CONFIDENCE_THRESHOLD:
                        new_conf = max(
                            LOW_CONFIDENCE_THRESHOLD,
                            node.confidence - ORPHAN_CONFIDENCE_DECAY,
                        )
                        try:
                            node.confidence = new_conf
                            graph.add_node(node)  # add_node does upsert
                            fixes += 1
                        except Exception:
                            pass

                if fixes > 0:
                    issues.append(ValidationIssue(
                        severity="info",
                        category="stale",
                        description=f"Decayed confidence on {fixes} stale nodes",
                        auto_remediated=True,
                        remediation_action="confidence -= 0.05",
                    ))

        return issues, fixes

    def _check_orphan_nodes(
        self, graph
    ) -> Tuple[List[ValidationIssue], int]:
        """Check for nodes with no edges (disconnected from graph)."""
        issues = []
        fixes = 0

        try:
            conn = graph._conn
            # Find nodes that appear in neither edges.source_id nor edges.target_id
            orphans = conn.execute("""
                SELECT id, node_type, label, confidence
                FROM nodes
                WHERE id NOT IN (SELECT source_id FROM edges)
                  AND id NOT IN (SELECT target_id FROM edges)
                  AND node_type NOT IN ('conversation')
            """).fetchall()
        except Exception:
            return issues, fixes

        if len(orphans) > MAX_NODES_WITHOUT_EDGES:
            orphan_ids = [r["id"] for r in orphans]
            issues.append(ValidationIssue(
                severity="warning",
                category="orphan",
                description=(
                    f"{len(orphans)} orphan nodes with no connections — "
                    f"these knowledge fragments are isolated"
                ),
                node_ids=orphan_ids[:10],  # Cap for display
            ))

            # Auto-remediate: flag orphans with low confidence for review
            if self._auto_remediate:
                for row in orphans:
                    if row["confidence"] > LOW_CONFIDENCE_THRESHOLD + 0.1:
                        try:
                            conn.execute(
                                "UPDATE nodes SET confidence = ? WHERE id = ?",
                                (max(LOW_CONFIDENCE_THRESHOLD, row["confidence"] - ORPHAN_CONFIDENCE_DECAY), row["id"]),
                            )
                            fixes += 1
                        except Exception:
                            pass
                if fixes:
                    conn.commit()
                    issues.append(ValidationIssue(
                        severity="info",
                        category="orphan",
                        description=f"Decayed confidence on {fixes} orphan nodes",
                        auto_remediated=True,
                        remediation_action="confidence -= 0.05 on orphans",
                    ))

        return issues, fixes

    def _check_contradictory_edges(self, graph) -> List[ValidationIssue]:
        """Check for conflicting edge types between the same node pairs."""
        issues = []

        contradictions = {
            ("supports", "conflicts_with"),
            ("influences", "blocked_by"),
        }

        try:
            conn = graph._conn
            # Get all edge type pairs between same source→target
            rows = conn.execute("""
                SELECT e1.source_id, e1.target_id, e1.edge_type AS type1, e2.edge_type AS type2
                FROM edges e1
                JOIN edges e2 ON e1.source_id = e2.source_id
                                AND e1.target_id = e2.target_id
                                AND e1.edge_type < e2.edge_type
            """).fetchall()
        except Exception:
            return issues

        for row in rows:
            pair = (row["type1"], row["type2"])
            reverse_pair = (row["type2"], row["type1"])
            if pair in contradictions or reverse_pair in contradictions:
                issues.append(ValidationIssue(
                    severity="error",
                    category="contradiction",
                    description=(
                        f"Contradictory edges between {row['source_id']} → {row['target_id']}: "
                        f"{row['type1']} AND {row['type2']}"
                    ),
                    node_ids=[row["source_id"], row["target_id"]],
                ))

        return issues

    def _check_low_confidence(self, graph) -> List[ValidationIssue]:
        """Check for accumulation of low-confidence nodes."""
        issues = []

        try:
            conn = graph._conn
            low_conf = conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE confidence < ?",
                (LOW_CONFIDENCE_THRESHOLD,),
            ).fetchone()[0]

            total = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        except Exception:
            return issues

        if total > 0 and low_conf / total > 0.3:
            issues.append(ValidationIssue(
                severity="warning",
                category="low_confidence",
                description=(
                    f"{low_conf}/{total} nodes ({low_conf/total:.0%}) have "
                    f"confidence below {LOW_CONFIDENCE_THRESHOLD:.0%} — "
                    f"graph quality may be degraded"
                ),
            ))

        return issues

    def _check_conversation_freshness(
        self, graph, now: datetime
    ) -> List[ValidationIssue]:
        """Check if there are recent conversations."""
        issues = []

        try:
            conversations = graph.get_recent_conversations(limit=1)
        except Exception:
            return issues

        if not conversations:
            issues.append(ValidationIssue(
                severity="warning",
                category="gap",
                description="No conversations recorded yet — self-model is empty",
            ))
            return issues

        latest_ts = conversations[0].get("timestamp", "")
        try:
            latest = datetime.fromisoformat(latest_ts.replace("Z", "+00:00"))
            gap = (now - latest).days
            if gap > STALE_CONVERSATION_DAYS:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="gap",
                    description=(
                        f"No conversations in {gap} days — "
                        f"self-model may be going stale"
                    ),
                ))
        except (ValueError, AttributeError):
            pass

        return issues

    def _check_dangling_edges(self, graph) -> List[ValidationIssue]:
        """Check for edges referencing non-existent nodes."""
        issues = []

        try:
            conn = graph._conn
            dangling = conn.execute("""
                SELECT e.source_id, e.target_id, e.edge_type
                FROM edges e
                LEFT JOIN nodes n1 ON e.source_id = n1.id
                LEFT JOIN nodes n2 ON e.target_id = n2.id
                WHERE n1.id IS NULL OR n2.id IS NULL
            """).fetchall()
        except Exception:
            return issues

        if dangling:
            issues.append(ValidationIssue(
                severity="error",
                category="dangling",
                description=(
                    f"{len(dangling)} edges reference non-existent nodes — "
                    f"graph has referential integrity issues"
                ),
            ))

            # Auto-remediate: remove dangling edges
            if self._auto_remediate:
                try:
                    conn = graph._conn
                    conn.execute("""
                        DELETE FROM edges
                        WHERE source_id NOT IN (SELECT id FROM nodes)
                           OR target_id NOT IN (SELECT id FROM nodes)
                    """)
                    conn.commit()
                    issues.append(ValidationIssue(
                        severity="info",
                        category="dangling",
                        description=f"Removed {len(dangling)} dangling edges",
                        auto_remediated=True,
                        remediation_action="DELETE dangling edges",
                    ))
                except Exception:
                    pass

        return issues

    # --- Scoring ---

    def _compute_category_scores(
        self,
        issues: List[ValidationIssue],
        stats: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute per-category health scores (0-100)."""
        scores: Dict[str, float] = {
            "structure": 100.0,
            "freshness": 100.0,
            "connectivity": 100.0,
            "consistency": 100.0,
            "confidence": 100.0,
        }

        for issue in issues:
            penalty = {"info": 0, "warning": 15, "error": 30}.get(
                issue.severity, 10
            )

            if issue.category in ("structure",):
                scores["structure"] = max(0, scores["structure"] - penalty)
            elif issue.category in ("stale", "gap"):
                scores["freshness"] = max(0, scores["freshness"] - penalty)
            elif issue.category in ("orphan", "dangling"):
                scores["connectivity"] = max(0, scores["connectivity"] - penalty)
            elif issue.category in ("contradiction",):
                scores["consistency"] = max(0, scores["consistency"] - penalty)
            elif issue.category in ("low_confidence",):
                scores["confidence"] = max(0, scores["confidence"] - penalty)

        return scores

    # --- Persistence ---

    def _save_report(self, report: ValidationReport) -> None:
        """Save validation report to disk."""
        try:
            report_path = self._report_dir / "latest_report.json"
            report_path.write_text(json.dumps(report.to_dict(), indent=2))

            # Also append to history
            history_path = self._report_dir / "validation_history.jsonl"
            with open(history_path, "a") as f:
                f.write(json.dumps({
                    "timestamp": report.timestamp,
                    "health_score": report.health_score,
                    "issues_count": len(report.issues),
                    "auto_remediations": report.auto_remediations,
                }) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save validation report: {e}")

    def get_latest_report(self) -> Optional[ValidationReport]:
        """Load the most recent validation report."""
        report_path = self._report_dir / "latest_report.json"
        if not report_path.exists():
            return None
        try:
            data = json.loads(report_path.read_text())
            return ValidationReport(
                timestamp=data["timestamp"],
                health_score=data["health_score"],
                total_nodes=data["total_nodes"],
                total_edges=data["total_edges"],
                total_conversations=data["total_conversations"],
                issues=[ValidationIssue(**i) for i in data.get("issues", [])],
                auto_remediations=data.get("auto_remediations", 0),
                category_scores=data.get("category_scores", {}),
            )
        except Exception:
            return None
