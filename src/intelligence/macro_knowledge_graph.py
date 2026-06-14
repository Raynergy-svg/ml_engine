"""Macro Knowledge Graph — static scaffold for structured macro reasoning.

US-002: Provides a queryable graph of macro relationships that the LLM Macro Agent
(and other intelligence modules) can traverse to reason about contagion,
regime shifts, and safe-haven flows.

Entities: central_banks, currencies, economies, commodities.
Relations: directional causal links between entities.

Persistence: JSON serialization with atomic writes.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Entity:
    """A node in the macro knowledge graph."""

    id: str
    label: str
    type: str  # e.g., "central_bank", "currency", "economy", "commodity"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            id=data["id"],
            label=data["label"],
            type=data["type"],
            attributes=data.get("attributes", {}),
        )


@dataclass
class Relation:
    """A directed edge between two entities."""

    source: str
    target: str
    relation_type: str  # e.g., "hikes_rates", "appreciates_against"
    weight: float = 1.0  # causal strength / confidence
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        return cls(
            source=data["source"],
            target=data["target"],
            relation_type=data["relation_type"],
            weight=data.get("weight", 1.0),
            attributes=data.get("attributes", {}),
        )


# ------------------------------------------------------------------
# Macro Knowledge Graph
# ------------------------------------------------------------------

class MacroKnowledgeGraph:
    """Static knowledge graph for macroeconomic reasoning.

    Example queries:
        kg.query_entity("fed")
        kg.query_relations("hikes_rates")
        kg.find_paths("fed", "usd")
    """

    DEFAULT_ENTITIES: List[Entity] = [
        # Central banks
        Entity("fed", "Federal Reserve", "central_bank", {"jurisdiction": "US"}),
        Entity("ecb", "European Central Bank", "central_bank", {"jurisdiction": "EU"}),
        Entity("boj", "Bank of Japan", "central_bank", {"jurisdiction": "JP"}),
        Entity("boe", "Bank of England", "central_bank", {"jurisdiction": "UK"}),
        Entity("snb", "Swiss National Bank", "central_bank", {"jurisdiction": "CH"}),
        Entity("rba", "Reserve Bank of Australia", "central_bank", {"jurisdiction": "AU"}),
        Entity("rbcn", "People's Bank of China", "central_bank", {"jurisdiction": "CN"}),
        Entity("boc", "Bank of Canada", "central_bank", {"jurisdiction": "CA"}),
        # Currencies
        Entity("usd", "US Dollar", "currency", {"iso": "USD"}),
        Entity("eur", "Euro", "currency", {"iso": "EUR"}),
        Entity("jpy", "Japanese Yen", "currency", {"iso": "JPY"}),
        Entity("gbp", "British Pound", "currency", {"iso": "GBP"}),
        Entity("chf", "Swiss Franc", "currency", {"iso": "CHF"}),
        Entity("aud", "Australian Dollar", "currency", {"iso": "AUD"}),
        Entity("nzd", "New Zealand Dollar", "currency", {"iso": "NZD"}),
        Entity("cad", "Canadian Dollar", "currency", {"iso": "CAD"}),
        Entity("cny", "Chinese Yuan", "currency", {"iso": "CNY"}),
        # Economies
        Entity("us_economy", "United States Economy", "economy", {"region": "North America"}),
        Entity("eu_economy", "Eurozone Economy", "economy", {"region": "Europe"}),
        Entity("jp_economy", "Japan Economy", "economy", {"region": "Asia"}),
        Entity("uk_economy", "United Kingdom Economy", "economy", {"region": "Europe"}),
        Entity("cn_economy", "China Economy", "economy", {"region": "Asia"}),
        # Commodities
        Entity("oil", "Crude Oil", "commodity", {"benchmark": "WTI"}),
        Entity("gold", "Gold", "commodity", {"benchmark": "XAU"}),
    ]

    DEFAULT_RELATIONS: List[Relation] = [
        # Central bank → currency
        Relation("fed", "usd", "issues"),
        Relation("ecb", "eur", "issues"),
        Relation("boj", "jpy", "issues"),
        Relation("boe", "gbp", "issues"),
        Relation("snb", "chf", "issues"),
        Relation("rba", "aud", "issues"),
        Relation("boc", "cad", "issues"),
        Relation("rbcn", "cny", "issues"),
        # Central bank actions → currency effects
        Relation("fed", "usd", "hikes_rates", weight=0.85),
        Relation("fed", "usd", "cuts_rates", weight=0.80),
        Relation("ecb", "eur", "hikes_rates", weight=0.85),
        Relation("ecb", "eur", "cuts_rates", weight=0.80),
        Relation("boj", "jpy", "hikes_rates", weight=0.85),
        Relation("boj", "jpy", "cuts_rates", weight=0.80),
        # Currency cross-rates
        Relation("usd", "eur", "appreciates_against", weight=0.70),
        Relation("eur", "usd", "appreciates_against", weight=0.70),
        Relation("usd", "jpy", "appreciates_against", weight=0.65),
        Relation("jpy", "usd", "appreciates_against", weight=0.65),
        # Safe-haven dynamics
        Relation("usd", "jpy", "safe_haven_flow", weight=0.75),
        Relation("usd", "chf", "safe_haven_flow", weight=0.70),
        Relation("gold", "usd", "safe_haven_flow", weight=0.60),
        # Geopolitical shock → safe haven
        Relation("geopolitical_shock", "usd", "safe_haven_flow", weight=0.90),
        Relation("geopolitical_shock", "jpy", "safe_haven_flow", weight=0.85),
        Relation("geopolitical_shock", "chf", "safe_haven_flow", weight=0.80),
        Relation("geopolitical_shock", "gold", "safe_haven_flow", weight=0.95),
        # Commodity → economy
        Relation("oil", "us_economy", "inflation_pressure", weight=0.60),
        Relation("oil", "eu_economy", "inflation_pressure", weight=0.55),
        # Economy → central bank policy
        Relation("us_economy", "fed", "demands_tightening", weight=0.65),
        Relation("eu_economy", "ecb", "demands_tightening", weight=0.60),
        Relation("jp_economy", "boj", "demands_tightening", weight=0.50),
    ]

    def __init__(
        self,
        entities: Optional[List[Entity]] = None,
        relations: Optional[List[Relation]] = None,
    ):
        self._entities: Dict[str, Entity] = {}
        self._relations: List[Relation] = []
        self._index_by_type: Dict[str, Set[str]] = {}
        self._index_by_relation: Dict[str, List[Relation]] = {}

        for e in (entities or self.DEFAULT_ENTITIES):
            self.add_entity(e)
        for r in (relations or self.DEFAULT_RELATIONS):
            self.add_relation(r)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def add_entity(self, entity: Entity) -> None:
        self._entities[entity.id] = entity
        self._index_by_type.setdefault(entity.type, set()).add(entity.id)

    def add_relation(self, relation: Relation) -> None:
        self._relations.append(relation)
        self._index_by_relation.setdefault(relation.relation_type, []).append(relation)

    def remove_entity(self, entity_id: str) -> None:
        entity = self._entities.pop(entity_id, None)
        if entity:
            self._index_by_type.get(entity.type, set()).discard(entity_id)
        self._relations = [r for r in self._relations if r.source != entity_id and r.target != entity_id]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Fetch an entity by its ID."""
        return self._entities.get(entity_id)

    def query_by_name(self, name: str) -> List[Entity]:
        """Find entities whose label contains the given name (case-insensitive)."""
        name_lower = name.lower()
        return [e for e in self._entities.values() if name_lower in e.label.lower()]

    def query_by_type(self, entity_type: str) -> List[Entity]:
        """Return all entities of a given type."""
        return [self._entities[eid] for eid in self._index_by_type.get(entity_type, [])]

    def query_relations(
        self,
        relation_type: Optional[str] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
    ) -> List[Relation]:
        """Query relations by type, source, and/or target."""
        candidates: List[Relation]
        if relation_type:
            candidates = self._index_by_relation.get(relation_type, [])
        else:
            candidates = self._relations
        results = []
        for r in candidates:
            if source and r.source != source:
                continue
            if target and r.target != target:
                continue
            results.append(r)
        return results

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
        relation_types: Optional[List[str]] = None,
    ) -> List[List[Relation]]:
        """Find all simple paths from source to target up to max_depth hops.

        Returns a list of paths, where each path is a list of Relation objects.
        """
        allowed = set(relation_types) if relation_types else None
        results: List[List[Relation]] = []
        visited: Set[str] = set()

        def _dfs(current: str, path: List[Relation], depth: int) -> None:
            if depth > max_depth:
                return
            if current == target_id and path:
                results.append(list(path))
                return
            for r in self._relations:
                if r.source != current:
                    continue
                if allowed and r.relation_type not in allowed:
                    continue
                if r.target in visited:
                    continue
                visited.add(r.target)
                path.append(r)
                _dfs(r.target, path, depth + 1)
                path.pop()
                visited.discard(r.target)

        _dfs(source_id, [], 0)
        return results

    def get_neighbors(self, entity_id: str, relation_type: Optional[str] = None) -> List[Entity]:
        """Return all entities directly connected from the given entity."""
        rels = self.query_relations(source=entity_id, relation_type=relation_type)
        return [self._entities[r.target] for r in rels if r.target in self._entities]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self._entities.values()],
            "relations": [r.to_dict() for r in self._relations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MacroKnowledgeGraph":
        entities = [Entity.from_dict(e) for e in data.get("entities", [])]
        relations = [Relation.from_dict(r) for r in data.get("relations", [])]
        return cls(entities=entities, relations=relations)

    def save(self, path: Path) -> None:
        """Serialize to JSON with atomic write."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
            os.replace(tmp, path)
            logger.debug("MacroKnowledgeGraph saved to %s", path)
        except Exception as exc:
            logger.warning("Failed to save MacroKnowledgeGraph to %s: %s", path, exc)

    @classmethod
    def load(cls, path: Path) -> "MacroKnowledgeGraph":
        """Load from JSON. Returns default graph if file missing or corrupt."""
        if not path.exists():
            logger.debug("MacroKnowledgeGraph load: %s not found, returning default", path)
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return cls.from_dict(data)
        except Exception as exc:
            logger.warning("Failed to load MacroKnowledgeGraph from %s: %s", path, exc)
            return cls()
