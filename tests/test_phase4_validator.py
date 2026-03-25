"""Tests for Phase 4: Self-Model Validation Loop.

Covers:
- SelfModelValidator with a real SelfModelGraph (in-memory SQLite)
- All 7 validation checks
- Auto-remediation behavior
- Health scoring and report generation
- Edge cases (empty graph, no graph, stale nodes, orphans, contradictions)
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# Project root on import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aura.core.self_model import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    SelfModelGraph,
)
from src.aura.core.self_model_validator import (
    LOW_CONFIDENCE_THRESHOLD,
    MAX_NODES_WITHOUT_EDGES,
    MIN_GRAPH_NODES,
    STALE_CONVERSATION_DAYS,
    STALE_NODE_DAYS,
    SelfModelValidator,
    ValidationIssue,
    ValidationReport,
)


# --- Fixtures ---


@pytest.fixture
def tmp_dir():
    """Temp directory for graph DB and validation reports."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def graph(tmp_dir):
    """Fresh SelfModelGraph with in-memory-like temp DB."""
    db_path = tmp_dir / "test_self_model.db"
    g = SelfModelGraph(db_path=db_path)
    yield g
    g.close()


@pytest.fixture
def validator(graph, tmp_dir):
    """SelfModelValidator wired to the test graph."""
    report_dir = tmp_dir / "validation_reports"
    return SelfModelValidator(
        graph=graph,
        auto_remediate=True,
        report_dir=report_dir,
    )


@pytest.fixture
def healthy_graph(graph):
    """Graph with minimum healthy structure (Person + Goal + TradingState + edges)."""
    now = datetime.now(timezone.utc).isoformat()

    # Core nodes
    person = GraphNode(
        id="person-user",
        node_type=NodeType.PERSON,
        label="Test User",
        confidence=0.9,
        created_at=now,
        updated_at=now,
    )
    goal = GraphNode(
        id="goal-profit",
        node_type=NodeType.GOAL,
        label="Consistent profitability",
        confidence=0.8,
        created_at=now,
        updated_at=now,
    )
    trading = GraphNode(
        id="ts-current",
        node_type=NodeType.TRADING_STATE,
        label="Active trading",
        confidence=0.85,
        created_at=now,
        updated_at=now,
    )
    emotion = GraphNode(
        id="emotion-focused",
        node_type=NodeType.EMOTION,
        label="Focused",
        confidence=0.7,
        created_at=now,
        updated_at=now,
    )

    for node in [person, goal, trading, emotion]:
        graph.add_node(node)

    # Connect them
    graph.add_edge(GraphEdge(
        source_id="person-user",
        target_id="goal-profit",
        edge_type=EdgeType.INFLUENCES,
    ))
    graph.add_edge(GraphEdge(
        source_id="goal-profit",
        target_id="ts-current",
        edge_type=EdgeType.RELATES_TO,
    ))
    graph.add_edge(GraphEdge(
        source_id="emotion-focused",
        target_id="ts-current",
        edge_type=EdgeType.INFLUENCES,
    ))

    # Log a recent conversation
    graph.log_conversation(
        conversation_id="conv-001",
        summary="Discussed trading plan",
        emotional_state="focused",
        topics=["trading", "risk"],
        readiness_impact=0.1,
    )

    return graph


# --- Test: No graph provided ---


def test_validate_no_graph():
    """Validator with no graph returns error report."""
    v = SelfModelValidator(graph=None, report_dir=Path(tempfile.mkdtemp()))
    report = v.validate()

    assert report.health_score == 0.0
    assert len(report.issues) == 1
    assert report.issues[0].severity == "error"
    assert report.issues[0].category == "configuration"


# --- Test: Empty graph ---


def test_validate_empty_graph(validator, graph):
    """Empty graph flags missing structure."""
    report = validator.validate()

    # Should have structure errors (no nodes, no Person, no Goal)
    errors = [i for i in report.issues if i.severity == "error"]
    assert len(errors) >= 1  # At least "too few nodes"

    categories = {i.category for i in report.issues}
    assert "structure" in categories

    # Health score dragged down by structure failures (avg across 5 categories)
    assert report.health_score < 90
    assert report.category_scores["structure"] < 50


# --- Test: Healthy graph passes ---


def test_validate_healthy_graph(validator, healthy_graph):
    """Healthy graph gets high health score with no errors."""
    report = validator.validate()

    errors = [i for i in report.issues if i.severity == "error"]
    assert len(errors) == 0, f"Unexpected errors: {[e.description for e in errors]}"

    assert report.health_score >= 80
    assert report.total_nodes >= MIN_GRAPH_NODES
    assert report.total_edges >= 1
    assert report.total_conversations >= 1


# --- Test: Stale nodes detected ---


def test_stale_nodes_detected(validator, graph):
    """Nodes not updated in 21+ days are flagged as stale."""
    old_date = (datetime.now(timezone.utc) - timedelta(days=STALE_NODE_DAYS + 5)).isoformat()

    # Add nodes with stale timestamps
    for nid, label in [("person-old", "Old Person"), ("goal-old", "Old Goal"), ("ts-old", "Old State")]:
        node = GraphNode(
            id=nid,
            node_type=NodeType.PERSON if "person" in nid else (NodeType.GOAL if "goal" in nid else NodeType.TRADING_STATE),
            label=label,
            confidence=0.8,
            created_at=old_date,
            updated_at=old_date,
        )
        graph.add_node(node)
        # Force the updated_at to stay old (add_node resets it)
        graph._conn.execute(
            "UPDATE nodes SET updated_at = ? WHERE id = ?",
            (old_date, nid),
        )
    graph._conn.commit()

    report = validator.validate()

    stale_issues = [i for i in report.issues if i.category == "stale" and i.severity == "warning"]
    assert len(stale_issues) >= 1, "Should detect stale nodes"


# --- Test: Stale node auto-remediation (confidence decay) ---


def test_stale_node_auto_remediation(graph, tmp_dir):
    """Auto-remediation decays confidence on stale nodes."""
    old_date = (datetime.now(timezone.utc) - timedelta(days=STALE_NODE_DAYS + 10)).isoformat()

    node = GraphNode(
        id="goal-stale",
        node_type=NodeType.GOAL,
        label="Stale Goal",
        confidence=0.8,
        created_at=old_date,
        updated_at=old_date,
    )
    graph.add_node(node)
    # Force old updated_at
    graph._conn.execute("UPDATE nodes SET updated_at = ? WHERE id = ?", (old_date, "goal-stale"))
    graph._conn.commit()

    v = SelfModelValidator(graph=graph, auto_remediate=True, report_dir=tmp_dir / "reports")
    report = v.validate()

    assert report.auto_remediations >= 1

    # Confidence should have been decayed
    updated = graph.get_node("goal-stale")
    assert updated.confidence < 0.8


# --- Test: Orphan nodes detected ---


def test_orphan_nodes_detected(validator, graph):
    """Nodes with no edges are flagged when count exceeds threshold."""
    now = datetime.now(timezone.utc).isoformat()

    # Create many orphan nodes (more than MAX_NODES_WITHOUT_EDGES)
    for i in range(MAX_NODES_WITHOUT_EDGES + 3):
        graph.add_node(GraphNode(
            id=f"orphan-{i}",
            node_type=NodeType.DECISION,
            label=f"Orphan Decision {i}",
            confidence=0.6,
            created_at=now,
            updated_at=now,
        ))

    report = validator.validate()

    orphan_issues = [i for i in report.issues if i.category == "orphan"]
    assert len(orphan_issues) >= 1, "Should detect excessive orphan nodes"


# --- Test: Contradictory edges ---


def test_contradictory_edges_detected(validator, graph):
    """Edges with conflicting types between same nodes flagged as errors."""
    now = datetime.now(timezone.utc).isoformat()

    # Create two nodes
    graph.add_node(GraphNode(id="a", node_type=NodeType.GOAL, label="A", created_at=now, updated_at=now))
    graph.add_node(GraphNode(id="b", node_type=NodeType.GOAL, label="B", created_at=now, updated_at=now))

    # Add contradictory edges: SUPPORTS and CONFLICTS_WITH
    graph.add_edge(GraphEdge(source_id="a", target_id="b", edge_type=EdgeType.SUPPORTS))
    graph.add_edge(GraphEdge(source_id="a", target_id="b", edge_type=EdgeType.CONFLICTS_WITH))

    report = validator.validate()

    contradiction_issues = [i for i in report.issues if i.category == "contradiction"]
    assert len(contradiction_issues) >= 1, "Should detect contradictory edges"
    assert contradiction_issues[0].severity == "error"


# --- Test: Low confidence accumulation ---


def test_low_confidence_accumulation(validator, graph):
    """Graph with >30% low-confidence nodes flagged."""
    now = datetime.now(timezone.utc).isoformat()

    # Create 10 nodes, 4 with very low confidence (40% > threshold)
    for i in range(10):
        conf = 0.1 if i < 4 else 0.8
        graph.add_node(GraphNode(
            id=f"node-{i}",
            node_type=NodeType.DECISION,
            label=f"Node {i}",
            confidence=conf,
            created_at=now,
            updated_at=now,
        ))

    report = validator.validate()

    low_conf_issues = [i for i in report.issues if i.category == "low_confidence"]
    assert len(low_conf_issues) >= 1, "Should detect low confidence accumulation"


# --- Test: Conversation freshness gap ---


def test_conversation_gap_detected(validator, graph):
    """No recent conversations flagged as staleness gap."""
    # Log a conversation with old timestamp
    old_ts = (datetime.now(timezone.utc) - timedelta(days=STALE_CONVERSATION_DAYS + 3)).isoformat()
    graph._conn.execute(
        "INSERT INTO conversations (id, timestamp, summary, emotional_state) VALUES (?, ?, ?, ?)",
        ("conv-old", old_ts, "Old conversation", "neutral"),
    )
    graph._conn.commit()

    report = validator.validate()

    gap_issues = [i for i in report.issues if i.category == "gap"]
    assert len(gap_issues) >= 1, "Should detect conversation gap"


# --- Test: Dangling edges detected and auto-remediated ---


def test_dangling_edges_remediated(graph, tmp_dir):
    """Edges referencing non-existent nodes are detected and removed."""
    now = datetime.now(timezone.utc).isoformat()

    # Create one real node
    graph.add_node(GraphNode(id="real-node", node_type=NodeType.PERSON, label="Real", created_at=now, updated_at=now))

    # Insert dangling edge (target doesn't exist)
    graph._conn.execute(
        "INSERT INTO edges (source_id, target_id, edge_type, created_at) VALUES (?, ?, ?, ?)",
        ("real-node", "ghost-node", "influences", now),
    )
    graph._conn.commit()

    v = SelfModelValidator(graph=graph, auto_remediate=True, report_dir=tmp_dir / "reports")
    report = v.validate()

    dangling_issues = [i for i in report.issues if i.category == "dangling" and i.severity == "error"]
    assert len(dangling_issues) >= 1, "Should detect dangling edges"

    # Check auto-remediation removed the dangling edge
    remediation_info = [i for i in report.issues if i.category == "dangling" and i.auto_remediated]
    assert len(remediation_info) >= 1, "Should auto-remediate dangling edges"

    # Verify edge was actually removed from DB
    remaining = graph._conn.execute("SELECT COUNT(*) FROM edges WHERE target_id = 'ghost-node'").fetchone()[0]
    assert remaining == 0, "Dangling edge should have been deleted"


# --- Test: Health score categories ---


def test_category_scores_computed(validator, healthy_graph):
    """Category scores are computed for all 5 dimensions."""
    report = validator.validate()

    expected_categories = {"structure", "freshness", "connectivity", "consistency", "confidence"}
    assert set(report.category_scores.keys()) == expected_categories

    for cat, score in report.category_scores.items():
        assert 0 <= score <= 100, f"Category {cat} score {score} out of range"


# --- Test: Report persistence ---


def test_report_saved_to_disk(validator, healthy_graph, tmp_dir):
    """Validation report is persisted to JSON."""
    report = validator.validate()

    report_path = tmp_dir / "validation_reports" / "latest_report.json"
    assert report_path.exists(), "Report should be saved"

    data = json.loads(report_path.read_text())
    assert "health_score" in data
    assert "issues" in data
    assert "category_scores" in data

    # History log should also be written
    history_path = tmp_dir / "validation_reports" / "validation_history.jsonl"
    assert history_path.exists(), "History should be appended"


# --- Test: Report formatting ---


def test_report_format_output(validator, healthy_graph):
    """format_report() produces human-readable output."""
    report = validator.validate()
    output = report.format_report()

    assert "Self-Model Health:" in output
    assert "Nodes:" in output
    assert "Edges:" in output


# --- Test: get_latest_report round-trips ---


def test_get_latest_report(validator, healthy_graph):
    """get_latest_report() loads the saved report."""
    original = validator.validate()
    loaded = validator.get_latest_report()

    assert loaded is not None
    assert loaded.health_score == original.health_score
    assert loaded.total_nodes == original.total_nodes
    assert len(loaded.issues) == len(original.issues)


# --- Test: ValidationIssue and ValidationReport serialization ---


def test_serialization_round_trip():
    """Dataclass to_dict() produces valid JSON."""
    issue = ValidationIssue(
        severity="warning",
        category="stale",
        description="Test issue",
        node_ids=["a", "b"],
        auto_remediated=True,
        remediation_action="confidence -= 0.05",
    )
    d = issue.to_dict()
    assert d["severity"] == "warning"
    assert d["node_ids"] == ["a", "b"]
    assert json.dumps(d)  # Must be JSON-serializable

    report = ValidationReport(
        timestamp="2026-03-22T00:00:00+00:00",
        health_score=85.0,
        total_nodes=10,
        total_edges=5,
        total_conversations=3,
        issues=[issue],
        auto_remediations=1,
        category_scores={"structure": 100, "freshness": 85},
    )
    rd = report.to_dict()
    assert rd["health_score"] == 85.0
    assert len(rd["issues"]) == 1
    assert json.dumps(rd)  # Must be JSON-serializable


# --- Test: Auto-remediate OFF ---


def test_auto_remediate_disabled(graph, tmp_dir):
    """With auto_remediate=False, no remediations happen."""
    old_date = (datetime.now(timezone.utc) - timedelta(days=STALE_NODE_DAYS + 10)).isoformat()

    node = GraphNode(
        id="goal-stale-noremediate",
        node_type=NodeType.GOAL,
        label="Stale Goal",
        confidence=0.8,
        created_at=old_date,
        updated_at=old_date,
    )
    graph.add_node(node)
    graph._conn.execute("UPDATE nodes SET updated_at = ? WHERE id = ?", (old_date, "goal-stale-noremediate"))
    graph._conn.commit()

    v = SelfModelValidator(graph=graph, auto_remediate=False, report_dir=tmp_dir / "reports")
    report = v.validate()

    assert report.auto_remediations == 0

    # Confidence should be unchanged
    updated = graph.get_node("goal-stale-noremediate")
    assert updated.confidence == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
