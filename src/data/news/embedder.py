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

    def _resolve_device(self) -> str:
        """Resolve requested device against runtime availability.

        Falls back to CPU with a logged warning if requested accelerator isn't
        available. Honors .claude/rules/improvement.md "Silent Exception
        Prevention" — never silently degrade without surfacing.
        """
        import logging
        import torch

        logger = logging.getLogger(__name__)
        requested = self.device

        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning(
                "FinBERTEmbedder: device='cuda' requested but torch.cuda.is_available()=False; falling back to cpu"
            )
            return "cpu"
        if requested == "mps":
            # PyTorch 2.x: backends.mps.is_available() returns True only when
            # the build supports MPS AND the runtime can use it.
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
            logger.warning(
                "FinBERTEmbedder: device='mps' requested but unavailable; falling back to cpu"
            )
            return "cpu"
        if requested == "cpu":
            return "cpu"
        logger.warning(
            "FinBERTEmbedder: unknown device=%r; falling back to cpu", requested
        )
        return "cpu"

    def _ensure_loaded(self) -> None:
        """Lazy-load tokenizer + model on first embed() call.

        Why lazy: the constructor must stay cheap so tests / config-only paths
        don't pay for a HuggingFace download. Phase 2 plan locked this contract.
        """
        if self._model is not None and self._tokenizer is not None:
            return

        # Lazy imports — heavy deps stay out of import-time cost.
        from transformers import AutoModel, AutoTokenizer
        import torch

        resolved_device = self._resolve_device()
        # Cache resolved device so embed() doesn't re-resolve every call.
        self._resolved_device = resolved_device

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model.train(False)  # PyTorch's training-flag toggle (NOT Python builtin)
        model.to(torch.device(resolved_device))
        self._model = model

    def embed(self, events: List[NewsEvent]) -> np.ndarray:
        """Embed a list of NewsEvents to a (n, 768) float32 matrix.

        Mean-pools the last hidden state with attention-mask weighting; this
        is more robust than CLS-token pooling for short headlines (CLS tokens
        in BERT-family encoders carry less semantic signal without a [CLS]
        classification objective during pretraining).

        Empty input short-circuits to a zero-row matrix — callers iterating
        over per-bar event windows hit this constantly.
        """
        # Empty input: skip the model entirely. Contract (NewsEmbedder ABC):
        # shape (0, embedding_dim), dtype float32.
        if not events:
            return np.zeros((0, int(self.embedding_dim)), dtype=np.float32)

        self._ensure_loaded()

        import torch

        device = torch.device(self._resolved_device)
        texts: List[str] = [ev.text for ev in events]

        out_chunks: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # last_hidden_state: (batch, seq_len, 768)
                last_hidden = outputs.last_hidden_state
                # Mean-pool with attention mask. Mask shape (batch, seq_len);
                # broadcast to (batch, seq_len, 1) for multiplication.
                mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
                summed = (last_hidden * mask).sum(dim=1)  # (batch, 768)
                # Avoid div-by-zero on degenerate all-pad inputs (shouldn't
                # happen post-tokenization but defensive).
                token_counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / token_counts  # (batch, 768)
                out_chunks.append(pooled.detach().to("cpu").numpy().astype(np.float32))

        return np.concatenate(out_chunks, axis=0)
