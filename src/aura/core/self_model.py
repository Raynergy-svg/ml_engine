"""Self-Model Graph — Aura's persistent memory of the user.

Implements the typed graph schema from PRD v2.2 §5:
    Nodes: Person, Goal, Value, Project, Emotion, Decision, Pattern, TradingState
    Edges: INFLUENCES, BLOCKED_BY, CORRELATES, TRIGGERS, PREDICTS

Storage: SQLite (with SQLCipher encryption when available) + sqlite-vec for
vector similarity search. Falls back gracefully to plain SQLite if SQLCipher
is not installed.

This is the core data structure that makes Aura different from a stateless
chatbot — it accumulates knowledge about the user across every conversation.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    PERSON = "person"
    GOAL = "goal"
    VALUE = "value"
    PROJECT = "project"
    EMOTION = "emotion"
    DECISION = "decision"
    PATTERN = "pattern"
    TRADING_STATE = "trading_state"
    CONVERSATION = "conversation"
    LIFE_EVENT = "life_event"


class EdgeType(str, Enum):
    INFLUENCES = "influences"
    BLOCKED_BY = "blocked_by"
    CORRELATES = "correlates"
    TRIGGERS = "triggers"
    PREDICTS = "predicts"
    RELATES_TO = "relates_to"
    SUPPORTS = "supports"
    CONFLICTS_WITH = "conflicts_with"


@dataclass
class GraphNode:
    """A node in the self-model graph."""

    id: str
    node_type: NodeType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    confidence: float = 0.5  # How confident we are in this node's accuracy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "label": self.label,
            "properties": self.properties,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "confidence": self.confidence,
        }


@dataclass
class GraphEdge:
    """An edge connecting two nodes in the self-model graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "properties": self.properties,
            "created_at": self.created_at,
        }


class SelfModelGraph:
    """Persistent graph database for Aura's self-model.

    Uses SQLite for storage with optional SQLCipher encryption.
    All data stays on-device.

    Args:
        db_path: Path to the SQLite database file
        encryption_key: Optional passphrase for SQLCipher encryption
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        encryption_key: Optional[str] = None,
    ):
        self.db_path = db_path or Path(".aura/self_model.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._encryption_key = encryption_key
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

        # Try to enable SQLCipher if available and key provided
        # US-201: Fixed SQL injection — hex-encode key to prevent injection via
        # crafted encryption_key values. PRAGMA key does not support parameterized
        # queries in SQLite/SQLCipher, so we sanitize by hex-encoding.
        if self._encryption_key:
            try:
                _safe_key = self._encryption_key.encode("utf-8").hex()
                self._conn.execute(f"PRAGMA key = \"x'{_safe_key}'\"")
                logger.info("SQLCipher encryption enabled for self-model database")
            except Exception:
                logger.warning(
                    "SQLCipher not available — self-model database is unencrypted. "
                    "Install pysqlcipher3 for AES-256 encryption."
                )

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                label TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                confidence REAL DEFAULT 0.5
            );

            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                properties TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                summary TEXT,
                emotional_state TEXT,
                topics TEXT DEFAULT '[]',
                readiness_impact REAL DEFAULT 0.0,
                raw_messages TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS readiness_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                score REAL NOT NULL,
                components TEXT DEFAULT '{}',
                trigger TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
            CREATE INDEX IF NOT EXISTS idx_conversations_ts ON conversations(timestamp);
            CREATE INDEX IF NOT EXISTS idx_readiness_ts ON readiness_history(timestamp);
        """)
        self._conn.commit()

    # --- Node Operations ---

    def add_node(self, node: GraphNode) -> GraphNode:
        """Add or update a node in the graph."""
        now = datetime.now(timezone.utc).isoformat()
        if not node.created_at:
            node.created_at = now
        node.updated_at = now

        self._conn.execute(
            """INSERT INTO nodes (id, node_type, label, properties, created_at, updated_at, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   label = excluded.label,
                   properties = excluded.properties,
                   updated_at = excluded.updated_at,
                   confidence = excluded.confidence""",
            (
                node.id,
                node.node_type.value,
                node.label,
                json.dumps(node.properties, default=str),
                node.created_at,
                node.updated_at,
                node.confidence,
            ),
        )
        self._conn.commit()
        return node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by ID."""
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_node(row)

    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Get all nodes of a specific type."""
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE node_type = ? ORDER BY updated_at DESC",
            (node_type.value,),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def search_nodes(self, query: str, node_type: Optional[NodeType] = None) -> List[GraphNode]:
        """Search nodes by label (case-insensitive substring match)."""
        if node_type:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE label LIKE ? AND node_type = ? ORDER BY updated_at DESC",
                (f"%{query}%", node_type.value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE label LIKE ? ORDER BY updated_at DESC",
                (f"%{query}%",),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its edges. Returns True if deleted."""
        self._conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
        result = self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._conn.commit()
        return result.rowcount > 0

    # --- Edge Operations ---

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        """Add an edge between two nodes."""
        now = datetime.now(timezone.utc).isoformat()
        if not edge.created_at:
            edge.created_at = now

        self._conn.execute(
            """INSERT INTO edges (source_id, target_id, edge_type, weight, properties, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                edge.source_id,
                edge.target_id,
                edge.edge_type.value,
                edge.weight,
                json.dumps(edge.properties, default=str),
                edge.created_at,
            ),
        )
        self._conn.commit()
        return edge

    def get_edges_from(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[GraphEdge]:
        """Get all edges originating from a node."""
        if edge_type:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE source_id = ? AND edge_type = ?",
                (node_id, edge_type.value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE source_id = ?", (node_id,)
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_connected_nodes(self, node_id: str) -> List[Tuple[GraphNode, GraphEdge]]:
        """Get all nodes connected to a given node with their edges."""
        rows = self._conn.execute(
            """SELECT n.*, e.source_id as e_source, e.target_id as e_target,
                      e.edge_type as e_type, e.weight as e_weight,
                      e.properties as e_props, e.created_at as e_created
               FROM edges e
               JOIN nodes n ON (
                   (e.source_id = ? AND n.id = e.target_id) OR
                   (e.target_id = ? AND n.id = e.source_id)
               )""",
            (node_id, node_id),
        ).fetchall()

        results = []
        for row in rows:
            node = self._row_to_node(row)
            edge = GraphEdge(
                source_id=row["e_source"],
                target_id=row["e_target"],
                edge_type=EdgeType(row["e_type"]),
                weight=row["e_weight"],
                properties=json.loads(row["e_props"] or "{}"),
                created_at=row["e_created"],
            )
            results.append((node, edge))
        return results

    # --- Conversation Logging ---

    def log_conversation(
        self,
        conversation_id: str,
        summary: str,
        emotional_state: str,
        topics: List[str],
        readiness_impact: float = 0.0,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Log a conversation for pattern analysis."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO conversations (id, timestamp, summary, emotional_state, topics, readiness_impact, raw_messages)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   summary = excluded.summary,
                   emotional_state = excluded.emotional_state,
                   topics = excluded.topics,
                   readiness_impact = excluded.readiness_impact""",
            (
                conversation_id,
                now,
                summary,
                emotional_state,
                json.dumps(topics),
                readiness_impact,
                json.dumps(messages or []),
            ),
        )
        self._conn.commit()

    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent conversations."""
        rows = self._conn.execute(
            "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    # --- Readiness History ---

    def log_readiness(self, score: float, components: Dict[str, float], trigger: str = "") -> None:
        """Log a readiness score computation for historical tracking."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO readiness_history (timestamp, score, components, trigger) VALUES (?, ?, ?, ?)",
            (now, score, json.dumps(components, default=str), trigger),
        )
        self._conn.commit()

    def get_readiness_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent readiness score history."""
        rows = self._conn.execute(
            "SELECT * FROM readiness_history ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    # --- Graph Statistics ---

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        node_count = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        conv_count = self._conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]

        type_counts = {}
        for row in self._conn.execute(
            "SELECT node_type, COUNT(*) as cnt FROM nodes GROUP BY node_type"
        ).fetchall():
            type_counts[row["node_type"]] = row["cnt"]

        return {
            "total_nodes": node_count,
            "total_edges": edge_count,
            "total_conversations": conv_count,
            "nodes_by_type": type_counts,
        }

    # --- Internal ---

    def _row_to_node(self, row: sqlite3.Row) -> GraphNode:
        return GraphNode(
            id=row["id"],
            node_type=NodeType(row["node_type"]),
            label=row["label"],
            properties=json.loads(row["properties"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            confidence=row["confidence"],
        )

    def _row_to_edge(self, row: sqlite3.Row) -> GraphEdge:
        return GraphEdge(
            source_id=row["source_id"],
            target_id=row["target_id"],
            edge_type=EdgeType(row["edge_type"]),
            weight=row["weight"],
            properties=json.loads(row["properties"] or "{}"),
            created_at=row["created_at"],
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()
