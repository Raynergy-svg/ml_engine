"""Tests for Macro RAG module (US-003)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.intelligence.macro_rag import (
    EmbeddingModel,
    FlatVectorStore,
    MacroRAG,
    MacroRAGConfig,
    ParagraphChunker,
    TextChunk,
)


class TestParagraphChunker:

    def test_basic_chunking(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunker = ParagraphChunker(overlap_sentences=0)
        chunks = chunker.chunk_document(text, source_doc="test")
        assert len(chunks) == 3
        assert chunks[0].text == "Para one."
        assert chunks[1].text == "Para two."

    def test_overlap(self):
        text = "First sentence. Second sentence.\n\nThird sentence."
        chunker = ParagraphChunker(overlap_sentences=1)
        chunks = chunker.chunk_document(text, source_doc="test")
        assert len(chunks) == 2
        # Second chunk should prepend last sentence of first paragraph
        assert "Second sentence." in chunks[1].text
        assert "Third sentence." in chunks[1].text

    def test_empty_text(self):
        chunker = ParagraphChunker()
        chunks = chunker.chunk_document("", source_doc="test")
        assert chunks == []


class TestTextChunk:

    def test_roundtrip(self):
        chunk = TextChunk("abc", "Hello world.", "doc1", 0, embedding=np.ones(5, dtype=np.float32))
        d = chunk.to_dict()
        assert d["text"] == "Hello world."
        chunk2 = TextChunk.from_dict(d)
        assert chunk2.text == chunk.text
        assert np.array_equal(chunk2.embedding, chunk.embedding)


class TestEmbeddingModel:

    def test_encode_shape(self):
        model = EmbeddingModel()
        texts = ["hello", "world"]
        emb = model.encode(texts)
        assert emb.shape == (2, model.dim)
        assert emb.dtype == np.float32

    def test_deterministic_fallback(self):
        """Fallback embeddings should be deterministic for the same text."""
        model = EmbeddingModel()
        emb1 = model.encode(["same text"])[0]
        emb2 = model.encode(["same text"])[0]
        assert np.allclose(emb1, emb2)


class TestFlatVectorStore:

    def test_search(self):
        store = FlatVectorStore(dim=4)
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        chunk = TextChunk("c1", "hello", "doc", 0, embedding=emb)
        store.add([chunk])
        results = store.search(emb, top_k=1)
        assert len(results) == 1
        assert results[0][0].chunk_id == "c1"
        assert results[0][1] > 0.99

    def test_empty_store(self):
        store = FlatVectorStore(dim=4)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=5)
        assert results == []

    def test_add_rejects_bad_shape(self):
        store = FlatVectorStore(dim=4)
        bad_chunk = TextChunk("c1", "text", "doc", 0, embedding=np.ones(2, dtype=np.float32))
        store.add([bad_chunk])
        assert len(store.chunks) == 0


class TestMacroRAG:

    def test_ingest_and_retrieve(self):
        rag = MacroRAG(MacroRAGConfig(top_k=3))
        doc = "The Fed raised rates by 25bps. Inflation remains elevated. The committee expects further tightening."
        rag.ingest_document(doc, source="FOMC_2024_0319")
        results = rag.retrieve("What about inflation?")
        assert len(results) > 0
        assert all(score > -1.0 for _, score in results)

    def test_retrieve_before_ingest(self):
        rag = MacroRAG()
        results = rag.retrieve("query")
        assert results == []

    def test_retrieve_with_context(self):
        rag = MacroRAG(MacroRAGConfig(top_k=2))
        doc = "Paragraph one.\n\nParagraph two about inflation."
        rag.ingest_document(doc, source="ECB_2024_01")
        enriched = rag.retrieve_with_context("inflation")
        assert len(enriched) > 0
        assert "text" in enriched[0]
        assert "score" in enriched[0]

    def test_ingest_file(self, tmp_path: Path):
        rag = MacroRAG()
        f = tmp_path / "test.txt"
        f.write_text("The economy is growing. Unemployment is low.")
        rag.ingest_file(f, source="test")
        assert len(rag.store.chunks) > 0

    def test_save_and_load(self, tmp_path: Path):
        rag = MacroRAG(MacroRAGConfig(persist_dir=tmp_path))
        rag.ingest_document("Some macro text here.", source="doc1")
        rag.save()

        rag2 = MacroRAG(MacroRAGConfig(persist_dir=tmp_path))
        rag2.load()
        assert len(rag2.store.chunks) == len(rag.store.chunks)
