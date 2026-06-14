"""RAG module for historical macro analogies.

US-003: Retrieval-augmented generation over central-bank documents.

Ingests plaintext documents (FOMC statements, ECB minutes, BoJ minutes),
chunks them at paragraph level with overlap, embeds chunks with a lightweight
sentence-transformer, and stores them in a flat numpy index for top-k retrieval.

No external vector DB — pure numpy + optional sentence-transformers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore[assignment,misc]

# ------------------------------------------------------------------
# Chunk
# ------------------------------------------------------------------

@dataclass
class TextChunk:
    """A document chunk with metadata."""

    chunk_id: str
    text: str
    source_doc: str
    paragraph_index: int
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_doc": self.source_doc,
            "paragraph_index": self.paragraph_index,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextChunk":
        emb = None
        if data.get("embedding"):
            emb = np.array(data["embedding"], dtype=np.float32)
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            source_doc=data["source_doc"],
            paragraph_index=data["paragraph_index"],
            embedding=emb,
        )


# ------------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------------

class EmbeddingModel:
    """Lightweight embedding model wrapper with graceful fallback."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    FALLBACK_DIM = 384  # dimension of all-MiniLM-L6-v2

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model: Any = None
        self._dim = self.FALLBACK_DIM
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._model = SentenceTransformer(self.model_name)  # type: ignore[operator]
                logger.info("EmbeddingModel: loaded %s", self.model_name)
            except Exception as exc:
                logger.warning("EmbeddingModel: failed to load %s: %s", self.model_name, exc)
                self._model = None
        else:
            logger.warning("EmbeddingModel: sentence-transformers not installed; using random fallback")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        if self._model is not None:
            try:
                emb = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                return np.array(emb, dtype=np.float32)
            except Exception as exc:
                logger.warning("EmbeddingModel encode failed: %s", exc)
        # Fallback: deterministic random embeddings based on text hash
        rng = np.random.RandomState(42)
        embeddings = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rng_state = np.random.RandomState(seed)
            embeddings.append(rng_state.randn(self._dim).astype(np.float32))
        return np.stack(embeddings, axis=0)

    @property
    def dim(self) -> int:
        return self._dim


# ------------------------------------------------------------------
# Chunker
# ------------------------------------------------------------------

class ParagraphChunker:
    """Paragraph-level chunker with optional overlap."""

    def __init__(self, overlap_sentences: int = 1):
        self.overlap = overlap_sentences

    def chunk_document(self, text: str, source_doc: str) -> List[TextChunk]:
        """Split a document into paragraph chunks."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[TextChunk] = []
        for i, para in enumerate(paragraphs):
            # Simple overlap: prepend last sentence of previous paragraph
            if self.overlap > 0 and i > 0:
                prev_sentences = self._split_sentences(paragraphs[i - 1])
                overlap_text = " ".join(prev_sentences[-self.overlap:])
                para = overlap_text + " " + para
            chunk_id = hashlib.sha256(f"{source_doc}:{i}".encode()).hexdigest()[:16]
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                text=para,
                source_doc=source_doc,
                paragraph_index=i,
            ))
        return chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Very simple sentence splitting."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# ------------------------------------------------------------------
# Flat numpy vector store
# ------------------------------------------------------------------

class FlatVectorStore:
    """In-memory flat index — no external DB."""

    def __init__(self, dim: int):
        self.dim = dim
        self.chunks: List[TextChunk] = []
        self._vectors: Optional[np.ndarray] = None
        self._dirty = True

    def add(self, chunks: List[TextChunk]) -> None:
        """Add chunks (must already have embeddings)."""
        for c in chunks:
            if c.embedding is not None and c.embedding.shape[0] == self.dim:
                self.chunks.append(c)
                self._dirty = True
            else:
                logger.warning("Skipping chunk %s: bad embedding shape", c.chunk_id)

    def _build_matrix(self) -> np.ndarray:
        """Rebuild the numpy matrix from chunk embeddings."""
        if not self.chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([c.embedding for c in self.chunks], axis=0)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[TextChunk, float]]:
        """Cosine-similarity search. Returns (chunk, score) sorted descending."""
        if not self.chunks:
            return []
        matrix = self._build_matrix()
        # Cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_vec = query_embedding / query_norm
        mat_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        mat_norm[mat_norm == 0] = 1.0
        mat = matrix / mat_norm
        scores = mat @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[int(i)], float(scores[i])) for i in top_indices]

    def clear(self) -> None:
        self.chunks = []
        self._vectors = None
        self._dirty = True


# ------------------------------------------------------------------
# MacroRAG — orchestrator
# ------------------------------------------------------------------

@dataclass
class MacroRAGConfig:
    """Configuration for the Macro RAG system."""

    model_name: str = "all-MiniLM-L6-v2"
    chunk_overlap_sentences: int = 1
    top_k: int = 5
    persist_dir: Path = field(default_factory=lambda: Path(".buddy_cache/macro_rag"))


class MacroRAG:
    """Retrieval-augmented generation for historical macro analogies.

    Usage:
        rag = MacroRAG()
        rag.ingest_document(fomc_text, source="FOMC_2024_0319")
        rag.build_index()
        results = rag.retrieve("What did the Fed say about inflation in 2024?")
    """

    def __init__(self, config: Optional[MacroRAGConfig] = None):
        self.cfg = config or MacroRAGConfig()
        self.embedder = EmbeddingModel(self.cfg.model_name)
        self.chunker = ParagraphChunker(self.cfg.chunk_overlap_sentences)
        self.store = FlatVectorStore(self.embedder.dim)
        self._index_built = False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_document(self, text: str, source: str) -> None:
        """Chunk and embed a single document."""
        chunks = self.chunker.chunk_document(text, source_doc=source)
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts)
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
        self.store.add(chunks)
        self._index_built = True
        logger.debug("Ingested %d chunks from %s", len(chunks), source)

    def ingest_file(self, path: Path, source: Optional[str] = None) -> None:
        """Read a plaintext file and ingest it."""
        if not path.exists():
            logger.warning("MacroRAG: file not found %s", path)
            return
        try:
            text = path.read_text(encoding="utf-8")
            self.ingest_document(text, source or path.name)
        except Exception as exc:
            logger.warning("MacroRAG: failed to read %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[TextChunk, float]]:
        """Retrieve top-k chunks most similar to the query."""
        k = top_k or self.cfg.top_k
        if not self._index_built or not self.store.chunks:
            logger.debug("MacroRAG: retrieve called before ingestion")
            return []
        query_emb = self.embedder.encode([query])[0]
        return self.store.search(query_emb, top_k=k)

    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        context_window_chunks: int = 1,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks with neighboring context."""
        results = self.retrieve(query, top_k=top_k)
        enriched: List[Dict[str, Any]] = []
        for chunk, score in results:
            item = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_doc": chunk.source_doc,
                "paragraph_index": chunk.paragraph_index,
                "score": round(score, 4),
            }
            enriched.append(item)
        return enriched

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> None:
        """Save the index and chunks to disk."""
        save_path = path or (self.cfg.persist_dir / "index.json")
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "config": {
                    "model_name": self.cfg.model_name,
                    "chunk_overlap_sentences": self.cfg.chunk_overlap_sentences,
                    "top_k": self.cfg.top_k,
                },
                "chunks": [c.to_dict() for c in self.store.chunks],
            }
            tmp = save_path.with_suffix(save_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, save_path)
            logger.debug("MacroRAG saved to %s", save_path)
        except Exception as exc:
            logger.warning("MacroRAG save failed: %s", exc)

    def load(self, path: Optional[Path] = None) -> None:
        """Load index from disk."""
        load_path = path or (self.cfg.persist_dir / "index.json")
        if not load_path.exists():
            logger.debug("MacroRAG: no persisted index at %s", load_path)
            return
        try:
            with open(load_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            chunks = [TextChunk.from_dict(c) for c in data.get("chunks", [])]
            self.store.clear()
            self.store.add(chunks)
            self._index_built = len(chunks) > 0
            logger.info("MacroRAG loaded %d chunks from %s", len(chunks), load_path)
        except Exception as exc:
            logger.warning("MacroRAG load failed: %s", exc)
