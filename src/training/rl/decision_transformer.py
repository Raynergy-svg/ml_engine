"""Decision Transformer — offline RL via sequence modeling on trade trajectories.

Phase 1 (this commit): contract stubs only. No model, no training. Methods raise
``NotImplementedError`` so drift off the phase plan fails loudly.

See ``docs/superpowers/specs/2026-05-09-decision-transformer-design.md`` for the
full design (architecture diagram §3, trajectory schema §4, validation §6).

Phasing summary:
  - P1 (this commit): scaffolding stubs + tests
  - P2-prep: blocked on TrajectoryLoader (≥1K replayed trades on one pair)
  - P2-train: blocked on ≥5K trades (real or replayed); decision rule §6
  - P3: continuous-action extension (size head)
  - P4: CQL/IQL fallback if P2-train shows no lift

Per ``.claude/rules/improvement.md`` No-Mock Rule: nothing in this module imports
``unittest.mock``. Tests use real classes against ``tmp_path``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    # Heavy deps (torch) imported only at training time; the stub module must
    # be importable on a barebones interpreter for the test suite.
    import torch
    from torch import Tensor


# ---------------------------------------------------------------------------
# Action / token-type constants
# ---------------------------------------------------------------------------

# Discrete 3-way action space (see spec §4 "Trajectory Schema Decisions").
ACTION_NO_TRADE: int = 0
ACTION_LONG: int = 1
ACTION_SHORT: int = 2
NUM_ACTIONS: int = 3

# Token-type tags for the autoregressive (rtg, state, action) interleaving.
# DT-original recipe: each timestep emits 3 tokens in this order.
TOKEN_TYPE_RTG: int = 0
TOKEN_TYPE_STATE: int = 1
TOKEN_TYPE_ACTION: int = 2


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionTransformerConfig:
    """Hyperparameters for the Decision Transformer.

    Defaults are scoped to the "small data" regime (≤10K trades total). Values
    are derived from spec §3 and §4.

    Attributes:
        state_dim: Dimensionality of the per-step state vector ``s_t``. The
            spec fixes this at 32 (15 agent scores + confidence + 4 regime
            one-hots + drawdown + atr_pips + model_disagreement + uncertainty
            + group_momentum + spread_pips + 5 reserved slots).
        num_actions: Size of the discrete action space. Fixed at 3 (NO_TRADE,
            LONG, SHORT) for Phase 2; Phase 3 extension swaps the action head
            for a continuous Gaussian over position-size.
        context_length: Number of trade steps in one trajectory window. Each
            step emits 3 tokens (rtg, state, action), so the actual sequence
            length seen by the Transformer is ``3 * context_length``.
        d_model: Transformer hidden size. Tiny by default — small dataset.
        n_heads: Number of attention heads. ``d_model`` must be divisible.
        n_layers: Stack depth.
        dropout: Applied inside Transformer blocks.
        n_pairs: Number of distinct pairs the model is trained on. Each pair
            gets a learnable embedding concatenated to ``s_t`` (spec §4 "Pair
            conditioning").
        pair_embed_dim: Width of the pair-id embedding.
        rtg_target: Default return-to-go used at *inference time* (spec §3
            "inference: condition on rtg_target = +1.5R"). Ignored during
            training (the loader supplies real RTG values).
        rtg_clip: Symmetric clip on RTG values (R-multiples) to keep
            unbounded equity tails from dominating gradients.
    """

    state_dim: int = 32
    num_actions: int = NUM_ACTIONS
    context_length: int = 32
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    n_pairs: int = 8
    pair_embed_dim: int = 4
    rtg_target: float = 1.5
    rtg_clip: float = 10.0

    def __post_init__(self) -> None:
        # Lightweight invariants — these run at construction and are
        # exercised by the test suite. Keeps misconfiguration loud.
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads})"
            )
        if self.num_actions != NUM_ACTIONS:
            # Phase 3 will relax this; Phase 2 is hard-wired to 3-way discrete.
            raise ValueError(
                f"Phase 2 requires num_actions=={NUM_ACTIONS}; got "
                f"{self.num_actions}. See spec §7 (P3 extension)."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {self.dropout}")
        if self.context_length < 1:
            raise ValueError(
                f"context_length must be >= 1; got {self.context_length}"
            )


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class DecisionTransformer:
    """Causal Transformer over (rtg, state, action) trajectories.

    Phase 1: stub. Construction succeeds (so the trainer can be wired and
    tested for shape contracts), but ``forward`` and ``predict_action`` raise
    ``NotImplementedError``. The torch-dependent layers are NOT created in
    ``__init__`` — the stub deliberately defers ``import torch`` until
    ``_build_layers`` is called from a future Phase-2 implementation.

    Phase 2 (``forward``): standard DT recipe. Embed each (rtg, state, action)
    tuple into 3 d_model-wide tokens, sum with a learnable positional embedding
    over the 3K-long sequence, run a causal Transformer, project the *state*
    token positions to action logits. Loss is cross-entropy on actions.

    Phase 2 (``predict_action``): condition on ``config.rtg_target``, run one
    forward pass on the most recent ``context_length`` trade tuples, return
    ``argmax`` over action logits at the next slot.

    Phase 3 (deferred): replace the action softmax with a continuous Gaussian
    head over position-size in [0, 0.05] NAV-fraction.
    """

    def __init__(self, config: DecisionTransformerConfig) -> None:
        self.config = config
        # Layers are NOT constructed until Phase 2; keeping the stub
        # importable without torch is intentional (CI runs on barebones).
        self._layers_built: bool = False

    # ------------------------------------------------------------------
    # Phase 2 entry points
    # ------------------------------------------------------------------

    def _build_layers(self) -> None:
        """Construct the torch ``nn.Module`` graph (state/action/rtg embeds,
        causal Transformer blocks, action head). Called once on first
        ``forward``. Phase 2 fills this in."""
        raise NotImplementedError(
            "DecisionTransformer._build_layers: Phase 2 work. "
            "See spec §3 architecture diagram."
        )

    def forward(
        self,
        rtg: "Tensor",
        states: "Tensor",
        actions: "Tensor",
        pair_ids: "Tensor",
        timesteps: Optional["Tensor"] = None,
    ) -> "Tensor":
        """Run a forward pass.

        Args:
            rtg: ``(B, K, 1)`` — return-to-go per step (R-multiples, clipped).
            states: ``(B, K, state_dim)`` — per-step state vector.
            actions: ``(B, K)`` int64 — past actions (training: ground-truth;
                inference: previously-predicted, with the *next* slot left
                blank for the prediction).
            pair_ids: ``(B,)`` int64 — pair index for the whole trajectory.
            timesteps: optional ``(B, K)`` int64 — absolute step index used
                for the temporal positional embedding. If ``None``, uses
                ``arange(K)`` per row.

        Returns:
            ``(B, K, num_actions)`` action logits at every step.

        Raises:
            NotImplementedError: Phase 2 work.
        """
        raise NotImplementedError(
            "DecisionTransformer.forward: Phase 2 work. "
            "See spec §3 (architecture) + §4 (trajectory schema)."
        )

    def predict_action(
        self,
        recent_rtg: "Tensor",
        recent_states: "Tensor",
        recent_actions: "Tensor",
        pair_id: int,
        rtg_target: Optional[float] = None,
    ) -> int:
        """Predict the next discrete action conditional on ``rtg_target``.

        Args:
            recent_rtg: ``(K-1, 1)`` — past return-to-go.
            recent_states: ``(K, state_dim)`` — past states + current state.
            recent_actions: ``(K-1,)`` int64 — past actions.
            pair_id: scalar pair index.
            rtg_target: optional override; defaults to ``config.rtg_target``.

        Returns:
            One of ``ACTION_NO_TRADE | ACTION_LONG | ACTION_SHORT``.

        Raises:
            NotImplementedError: Phase 2 work.

        Note:
            This is the *training-time* prediction. The runtime safety filter
            (``OfflineRLTrainer._apply_runtime_safety_filter``) must wrap any
            production use of this method to enforce ``.claude/rules/trading.md``
            vetos (R:R, trend, MR, staleness). See spec §6 catastrophic guard.
        """
        raise NotImplementedError(
            "DecisionTransformer.predict_action: Phase 2 work. "
            "See spec §6 (validation + runtime safety guard)."
        )

    # ------------------------------------------------------------------
    # Persistence (Phase 2; stubs surface the API for the trainer)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save state dict + config to ``path``. Phase 2 work."""
        raise NotImplementedError(
            "DecisionTransformer.save: Phase 2 work. "
            "Will use torch.save({'config': asdict(config), "
            "'state_dict': model.state_dict()}, path)."
        )

    @classmethod
    def load(cls, path: str) -> "DecisionTransformer":
        """Restore from a ``save()`` artifact. Phase 2 work."""
        raise NotImplementedError(
            "DecisionTransformer.load: Phase 2 work."
        )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "ACTION_NO_TRADE",
    "ACTION_LONG",
    "ACTION_SHORT",
    "NUM_ACTIONS",
    "TOKEN_TYPE_RTG",
    "TOKEN_TYPE_STATE",
    "TOKEN_TYPE_ACTION",
    "DecisionTransformerConfig",
    "DecisionTransformer",
]
