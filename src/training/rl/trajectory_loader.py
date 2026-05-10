"""Trade-journal → trajectory tensor pipeline for the Decision Transformer.

Phase 1 (this commit): contract stubs only. Reads ``trained_data/trade_journal_rl.json``
and ``trained_data/virtual_trades.jsonl`` are NOT performed. The dataclass +
class signatures are the contract; tests verify shape promises by way of
documented attribute types.

Schema decisions live in the spec, not here:
    docs/superpowers/specs/2026-05-09-decision-transformer-design.md §4

Per ``.claude/rules/improvement.md`` JSON Safety Gates: the Phase-2
implementation MUST wrap every JSON read in try/except with graceful fallback
to an empty list, and validate the trade-record schema before consuming. Stub
docstrings flag this where relevant.

Per the No-Mock Rule: tests for this loader use real ``tmp_path`` fixtures
with hand-crafted JSON files. No ``unittest.mock``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Trajectory dataclass
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """One contiguous trade trajectory for a single pair.

    Phase 2 will populate these arrays from the journal. Phase 1 leaves the
    fields type-annotated so the trainer can wire the contract.

    Attributes:
        pair: e.g. "EUR_USD". One trajectory belongs to exactly one pair.
        pair_id: integer index assigned by ``TrajectoryLoader._pair_to_id``.
        states: ``(T, state_dim)`` float32 — per-step state vector. State
            schema is fixed by the spec (§4 "Trajectory Schema Decisions"):
            15 agent scores + confidence + 4 regime one-hots + drawdown +
            atr_pips + model_disagreement + uncertainty + group_momentum +
            spread_pips + 5 reserved slots = 32-dim.
        actions: ``(T,)`` int64 — discrete action taken at each step. Values
            in ``{ACTION_NO_TRADE=0, ACTION_LONG=1, ACTION_SHORT=2}``.
        rewards: ``(T,)`` float32 — R-multiples, clipped to ``[-1.0, +5.0]``.
            ``r_t = realized_pl / sl_risk_dollars`` for trades; ``r_t = 0``
            for NO_TRADE slots inserted from ``virtual_trades.jsonl``.
        returns_to_go: ``(T,)`` float32 — ``Σ_{i≥t} r_i`` over the bounded
            horizon (default: end-of-trajectory or +50 trades, whichever
            comes first). Clipped to ``[-rtg_clip, +rtg_clip]``.
        timestamps: ``(T,)`` int64 — UTC unix-seconds at decision time. Used
            only for sorting / debugging; not fed to the model.
    """

    pair: str
    pair_id: int
    states: "NDArray[np.float32]"
    actions: "NDArray[np.int64]"
    rewards: "NDArray[np.float32]"
    returns_to_go: "NDArray[np.float32]"
    timestamps: "NDArray[np.int64]"

    def __len__(self) -> int:
        """Number of decision steps in this trajectory."""
        # Phase 2 will return ``self.actions.shape[0]``; the stub leaves the
        # method present so the trainer's signature checks pass.
        raise NotImplementedError(
            "Trajectory.__len__: Phase 2 work. "
            "Will return self.actions.shape[0]."
        )


# ---------------------------------------------------------------------------
# Loader config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryLoaderConfig:
    """Configuration for the trajectory loader.

    Attributes:
        journal_path: Path to ``trained_data/trade_journal_rl.json``. Closed
            real trades only — populated by ``ExecutionManager`` on close.
        virtual_trades_path: Path to ``trained_data/virtual_trades.jsonl``.
            Gate-rejected scans; supplies NO_TRADE supervision when
            ``include_no_trade=True``.
        context_length: K in the spec. Used by ``windowize`` to chunk
            trajectories into fixed-length training windows.
        reward_clip_min: Lower clip on per-step R-multiples.
        reward_clip_max: Upper clip. Spec §4 default = +5.0.
        rtg_horizon: Bounded horizon for return-to-go computation. None means
            "end of trajectory"; an int means "min(end, t+horizon)".
        rtg_clip: Symmetric clip on return-to-go.
        include_no_trade: If True, insert NO_TRADE steps at scan timestamps
            from ``virtual_trades.jsonl``. If False, the trajectory contains
            only realized trades. Spec §4 recommends True for richer
            supervision; False available as ablation.
        min_trajectory_length: Trajectories shorter than this are dropped
            (one trade per pair is not a sequence).
    """

    journal_path: Path = field(
        default_factory=lambda: Path("trained_data/trade_journal_rl.json")
    )
    virtual_trades_path: Path = field(
        default_factory=lambda: Path("trained_data/virtual_trades.jsonl")
    )
    context_length: int = 32
    reward_clip_min: float = -1.0
    reward_clip_max: float = 5.0
    rtg_horizon: Optional[int] = 50
    rtg_clip: float = 10.0
    include_no_trade: bool = True
    min_trajectory_length: int = 4

    def __post_init__(self) -> None:
        if self.context_length < 1:
            raise ValueError(
                f"context_length must be >= 1; got {self.context_length}"
            )
        if self.reward_clip_min >= self.reward_clip_max:
            raise ValueError(
                f"reward_clip_min ({self.reward_clip_min}) must be < "
                f"reward_clip_max ({self.reward_clip_max})"
            )
        if self.rtg_clip <= 0:
            raise ValueError(f"rtg_clip must be > 0; got {self.rtg_clip}")
        if self.min_trajectory_length < 1:
            raise ValueError(
                "min_trajectory_length must be >= 1; got "
                f"{self.min_trajectory_length}"
            )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TrajectoryLoader:
    """Load + window the trade journal into Decision-Transformer batches.

    Phase 1 stubs only. Construction succeeds (so the trainer can be unit-
    tested for wiring), but ``load`` and ``windowize`` raise.

    Phase 2 contract:

      1. ``load()``:
         - Read ``journal_path`` (JSON list) with try/except → empty list on
           parse error.
         - If ``include_no_trade``: read ``virtual_trades_path`` (JSONL) line-
           by-line, skipping malformed lines.
         - Group by ``pair``. Sort each group by entry timestamp.
         - For each trade record, extract the 32-dim state vector per spec §4.
           State extraction is keyed on existing journal fields:
             * 15 agent scores from ``trade["agents"]["agent_reasons"][i]["score"]``
             * confidence from ``trade["confidence"]``
             * regime one-hot from ``trade["regime"]["regime"]`` (LOW/MED/HIGH/EXTREME)
             * drawdown_pct from ``trade["regime"]["account_drawdown"]`` (or 0.0)
             * atr_pips from ``trade["regime"]["atr_pips"]``
             * model_disagreement from ``trade["regime"]["model_disagreement"]``
             * uncertainty from ``trade["regime"]["uncertainty_score"]``
             * group_momentum from ``trade["group_momentum"]``
             * spread_pips from ``trade["spread_pips"]``
             * 5 reserved slots: zeros in P2; reused by P3 news/macro fusion.
         - Map ``trade["direction"]`` ∈ {"long", "short"} to ACTION_LONG /
           ACTION_SHORT; virtual-trade rows always map to ACTION_NO_TRADE.
         - Compute ``r_t = realized_pl / sl_risk_dollars`` from
           ``trade["outcome"]["realized_pl"]`` and the SL distance in
           ``trade["sl_pips"] * trade["tick_value"] * trade["lots"]``.
         - Clip rewards per config; compute return-to-go with horizon.
         - Drop trajectories below ``min_trajectory_length``.
         - Return ``List[Trajectory]``.

      2. ``windowize(trajectories)``: chunk each trajectory into overlapping
         windows of length ``context_length`` (stride=1). Yield batches of
         tensors matching the DT.forward signature (rtg, states, actions,
         pair_ids, timesteps).

      3. ``train_holdout_split(trajectories, holdout_frac)``: chronological
         split per pair (NOT random shuffle) to prevent lookahead leakage.
         Embargo of 5 trades between train and holdout (spec §6).

    JSON safety per ``.claude/rules/improvement.md``: every read is wrapped
    in try/except with graceful fallback. Empty journal returns ``[]``, not
    a crash.
    """

    def __init__(self, config: TrajectoryLoaderConfig) -> None:
        self.config = config
        self._pair_to_id: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Phase 2 entry points
    # ------------------------------------------------------------------

    def load(self) -> List[Trajectory]:
        """Read the journal + virtual-trades, group by pair, build trajectories.

        Returns:
            List of ``Trajectory`` objects, one per pair with at least
            ``min_trajectory_length`` decisions.

        Raises:
            NotImplementedError: Phase 2 work.

        Phase 2 implementation must:
          - Wrap JSON reads in try/except (rule: JSON Safety Gates).
          - Validate ``trade["outcome"]`` exists before consuming (rule: never
            trust JSON without schema validation in production paths).
          - Log every dropped record with reason (corrupted, missing field,
            negative SL distance) — never silently skip.
        """
        raise NotImplementedError(
            "TrajectoryLoader.load: Phase 2 work. "
            "See spec §4 + this docstring for the extraction contract."
        )

    def windowize(
        self, trajectories: Sequence[Trajectory]
    ) -> Iterable[Tuple["NDArray", "NDArray", "NDArray", "NDArray", "NDArray"]]:
        """Chunk trajectories into fixed-length training windows.

        Yields tuples of ``(rtg, states, actions, pair_ids, timesteps)`` ready
        to be torch-converted by the trainer. Stride=1 windowing maximizes
        sample count on small data; the trainer can subsample if needed.

        Raises:
            NotImplementedError: Phase 2 work.
        """
        raise NotImplementedError(
            "TrajectoryLoader.windowize: Phase 2 work."
        )

    def train_holdout_split(
        self,
        trajectories: Sequence[Trajectory],
        holdout_frac: float = 0.2,
        embargo: int = 5,
    ) -> Tuple[List[Trajectory], List[Trajectory]]:
        """Chronologically split each trajectory; embargo prevents leakage.

        Args:
            trajectories: from ``load()``.
            holdout_frac: fraction of *each* trajectory's tail used for
                holdout. Per-pair fractional split preserves coverage.
            embargo: number of trade steps removed from the train tail to
                prevent label leakage across the boundary (spec §6).

        Returns:
            ``(train_trajectories, holdout_trajectories)``.

        Raises:
            NotImplementedError: Phase 2 work.
        """
        raise NotImplementedError(
            "TrajectoryLoader.train_holdout_split: Phase 2 work."
        )

    # ------------------------------------------------------------------
    # Pair-id assignment (Phase 2 will populate during load)
    # ------------------------------------------------------------------

    def get_pair_id(self, pair: str) -> int:
        """Return the integer pair-id, assigning a new one if unseen.

        Phase 2 invariant: ids are assigned in load-order, persisted to the
        config artifact at save time, and restored at inference time so the
        runtime maps the same ``pair`` string to the same embedding row as
        during training.
        """
        raise NotImplementedError(
            "TrajectoryLoader.get_pair_id: Phase 2 work."
        )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "Trajectory",
    "TrajectoryLoaderConfig",
    "TrajectoryLoader",
]
