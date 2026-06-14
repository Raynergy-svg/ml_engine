"""Tests for Macro Knowledge Graph (US-002)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.intelligence.macro_knowledge_graph import (
    Entity,
    MacroKnowledgeGraph,
    Relation,
)


class TestEntity:

    def test_to_dict_roundtrip(self):
        e = Entity("fed", "Federal Reserve", "central_bank", {"jurisdiction": "US"})
        d = e.to_dict()
        assert d["id"] == "fed"
        assert d["type"] == "central_bank"
        e2 = Entity.from_dict(d)
        assert e2.id == e.id
        assert e2.attributes == e.attributes


class TestRelation:

    def test_to_dict_roundtrip(self):
        r = Relation("fed", "usd", "hikes_rates", weight=0.85)
        d = r.to_dict()
        assert d["source"] == "fed"
        assert d["weight"] == 0.85
        r2 = Relation.from_dict(d)
        assert r2.source == r.source
        assert r2.weight == r.weight


class TestMacroKnowledgeGraph:

    def test_default_entities_populated(self):
        kg = MacroKnowledgeGraph()
        assert len(kg._entities) > 0
        assert kg.get_entity("fed") is not None
        assert kg.get_entity("usd") is not None

    def test_default_relations_populated(self):
        kg = MacroKnowledgeGraph()
        assert len(kg._relations) > 0
        rels = kg.query_relations(relation_type="hikes_rates")
        assert len(rels) >= 3

    def test_query_by_name(self):
        kg = MacroKnowledgeGraph()
        results = kg.query_by_name("federal")
        ids = {e.id for e in results}
        assert "fed" in ids

    def test_query_by_type(self):
        kg = MacroKnowledgeGraph()
        currencies = kg.query_by_type("currency")
        ids = {c.id for c in currencies}
        assert "usd" in ids
        assert "eur" in ids
        assert "jpy" in ids

    def test_query_relations_by_type(self):
        kg = MacroKnowledgeGraph()
        rels = kg.query_relations(relation_type="safe_haven_flow")
        assert len(rels) >= 4

    def test_query_relations_by_source_and_target(self):
        kg = MacroKnowledgeGraph()
        rels = kg.query_relations(source="fed", target="usd")
        assert len(rels) >= 1

    def test_find_paths_exists(self):
        kg = MacroKnowledgeGraph()
        paths = kg.find_paths("fed", "usd", max_depth=1)
        assert len(paths) >= 1
        assert paths[0][0].source == "fed"
        assert paths[0][0].target == "usd"

    def test_find_paths_no_path(self):
        kg = MacroKnowledgeGraph()
        paths = kg.find_paths("oil", "fed", max_depth=1)
        assert len(paths) == 0

    def test_get_neighbors(self):
        kg = MacroKnowledgeGraph()
        neighbors = kg.get_neighbors("fed", relation_type="issues")
        ids = {n.id for n in neighbors}
        assert "usd" in ids

    def test_add_and_remove_entity(self):
        kg = MacroKnowledgeGraph()
        kg.add_entity(Entity("test_cb", "Test CB", "central_bank"))
        assert kg.get_entity("test_cb") is not None
        kg.remove_entity("test_cb")
        assert kg.get_entity("test_cb") is None

    def test_save_and_load(self, tmp_path: Path):
        kg = MacroKnowledgeGraph()
        path = tmp_path / "kg.json"
        kg.save(path)
        assert path.exists()
        kg2 = MacroKnowledgeGraph.load(path)
        assert len(kg2._entities) == len(kg._entities)
        assert len(kg2._relations) == len(kg._relations)

    def test_load_missing_file_returns_default(self, tmp_path: Path):
        path = tmp_path / "nonexistent.json"
        kg = MacroKnowledgeGraph.load(path)
        assert kg.get_entity("fed") is not None

    def test_load_corrupt_file_returns_default(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        kg = MacroKnowledgeGraph.load(path)
        assert kg.get_entity("fed") is not None
