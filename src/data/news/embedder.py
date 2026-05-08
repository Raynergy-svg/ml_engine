"""News embedders (Phase 1 stubs).

Defines the ``NewsEmbedder`` abstract contract and the ``FinBERTEmbedder``
concrete stub. The recommended embedder is ``ProsusAI/finbert`` (see design
doc §2 for cost / quality / M1-compat comparison).

Phase 1: stubs only. Calling ``embed`` raises ``NotImplementedError`` so
any drift off the phase plan fails loudly.

Phase 2 implementation will:
    - Lazy-import ``transformers.AutoTokenizer`` and ``transformers.AutoModel``.
    - Load ``ProsusAI/finbert`` via ``AutoModel.from_pretrained``; weights cached
      to the standard HF cache (``~/.cache/huggingface/``).
    - Tokenize batched input headlines (max_length=128, padding=True, truncation=True).
    - Forward pass; return mean-pooled last-hidden-state, shape (n_events, 768).
    - Use ``torch.no_grad()`` + ``model.train(False)`` for pure-inference mode
      (no dropout, no grad accumulation).
    - On M1: optionally use Metal via ``device='mps'`` if PyTorch was built with
      MPS support; otherwise CPU. Latency budget ~5-10ms/headline on CPU; ~2ms
      on Metal.

Note: docstring mentions of "inference mode" / "model.train(False)" refer to
PyTorch's training-flag toggle, not Python's code-evaluation builtin.
"""

from __future__ import annotations

import abc
from typing import List, Optional

import numpy as np

from src.data.news.source import NewsEvent


class NewsEmbedder(abc.ABC):
    """Abstract base for any text-to-vector embedder.

    Concrete implementations must:
      - Be deterministic — same input list produces same output array.
      - Return ``np.ndarray`` of shape (len(events), embedding_dim) and dtype
        ``np.float32``. Float32 keeps PCA + downstream training memory-friendly.
      - Tolerate empty input (return shape (0, embedding_dim)).
      - Document ``embedding_dim`` so feature_alignment and PCA know the
        contract upfront.
    """

    #: Embedding dimensionality. Subclasses must set this; tests + alignment
    #: code reads it for shape validation. None on the ABC; concrete classes
    #: set it in __init__ or as a class attribute.
    embedding_dim: Optional[int] = None

    @abc.abstractmethod
    def embed(self, events: List[NewsEvent]) -> np.ndarray:
        """Embed a list of news events to a dense matrix.

        Args:
            events: List of ``NewsEvent``. Empty list is allowed and returns
                an empty matrix of correct shape.

        Returns:
            ``np.ndarray`` of shape ``(len(events), self.embedding_dim)``,
            dtype ``np.float32``. Row order matches input order — the alignment
            stage relies on positional correspondence with the events list.

        Raises:
            RuntimeError: if the model fails to load or inference errors. Phase
                3+ tests use real model loads; no mocks.
        """
        raise NotImplementedError


class FinBERTEmbedder(NewsEmbedder):
    """FinBERT (``ProsusAI/finbert``) embedder, mean-pooled last-hidden-state.

    Recommended embedder per design doc §2: free, finance-domain-tuned,
    768-dim, M1-compatible via ``transformers`` (already in env).

    Phase 1: stub only. Constructor stores config; ``embed`` raises
    NotImplementedError until Phase 2 wires the actual model load + inference.

    Phase 2 implementation outline (NOT IMPLEMENTED):
        - Lazy-import transformers.AutoTokenizer / AutoModel + torch.
        - Tokenize headline texts with padding + truncation to max_length.
        - Forward pass under torch.no_grad(); model in inference mode.
        - Mean-pool last hidden state, masking pad tokens.
        - Return float32 numpy of shape (n_events, 768).
    """

    embedding_dim: int = 768  # FinBERT base; constant across versions

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 128,
    ) -> None:
        """Initialize FinBERT embedder config.

        Args:
            model_name: HuggingFace model ID. Default ``"ProsusAI/finbert"``.
            device: ``"cpu"``, ``"mps"`` (M1 Metal), or ``"cuda"``. Phase 2
                will validate availability at first ``embed()`` call.
            batch_size: Number of events per forward pass. 32 is the M1
                CPU sweet spot; raise to 128 on Metal/CUDA.
            max_length: Tokenization truncation length. 128 covers >99% of
                headlines without truncation.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        # Phase 1: instance constructible (test-verifiable type). Actual
        # model + tokenizer are lazy-loaded on first embed() call (Phase 2).
        self._model = None
        self._tokenizer = None

    def embed(self, events: List[NewsEvent]) -> np.ndarray:
        """Phase 2 implementation pending."""
        raise NotImplementedError(
            "FinBERTEmbedder.embed is a Phase 2 implementation. "
            "See docs/superpowers/plans/2026-05-08-news-macro-signal-design.md "
            "§2 (Embedding model comparison) and §6 (Sequencing) for phase "
            "boundaries."
        )
