"""Offline RL position sizer using IQL (Implicit Q-Learning) on logged trade trajectories.

Trains on historical trade_journal_rl.json data without live exploration,
producing a position-sizing policy that can complement or replace the
online PPO sizer (position_sizer.py).  Controlled by the
``offline_sizer_enabled`` config flag -- defaults to ``false`` so the
existing PPO path stays active.

Requires: d3rlpy >= 2.8.0 (PyTorch backend).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InsufficientTrajectoriesError(Exception):
    """Raised when journal has fewer than MIN_TRAJECTORIES completed trades."""


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

# Features extracted from each journal entry, in fixed order.
# Missing fields default to 0.0 so partial entries are never skipped.
_STATE_FIELDS: List[Tuple[str, ...]] = [
    # Top-level agent/gate scores
    ("confidence_score",),
    ("weighted_vote_score",),
    ("uncertainty_score",),
    ("trend_strength",),
    ("volatility_percentile",),
    ("momentum_score",),
    # Gate pass booleans (cast to float)
    ("gates", "confidence_gate"),
    ("gates", "momentum_gate"),
    ("gates", "risk_gate"),
    # Regime features
    ("regime", "regime_id"),
    ("regime", "volatility_regime"),
    # Risk metrics
    ("atr",),
    ("spread_pips",),
    # Model outputs
    ("model_confidence",),
    ("direction_probability",),
]

STATE_DIM: int = len(_STATE_FIELDS)


def _deep_get(d: dict, keys: Tuple[str, ...], default: float = 0.0) -> float:
    """Safely traverse nested dict keys, returning *default* on any miss."""
    current: Any = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k)
        if current is None:
            return default
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def extract_state(entry: dict) -> np.ndarray:
    """Extract a fixed-size feature vector from a single journal entry.

    Returns:
        np.ndarray of shape ``(STATE_DIM,)`` with dtype ``float32``.
    """
    return np.array(
        [_deep_get(entry, keys) for keys in _STATE_FIELDS],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OfflineSizerIQL:
    """Offline RL position sizer using d3rlpy IQL on trade_journal_rl.json.

    Trains on completed-trade trajectories stored in the trade journal,
    learning a position-sizing policy from (state, action, reward) tuples
    without any live market interaction.

    Parameters
    ----------
    journal_path : Path or None
        Path to the trade journal JSON.  Defaults to
        ``trained_data/trade_journal_rl.json``.
    model_path : Path or None
        Path where the trained model is persisted.  Defaults to
        ``trained_data/models/offline_sizer_iql.d3``.
    """

    MIN_TRAJECTORIES: int = 200
    DEFAULT_EPOCHS: int = 100
    MODEL_PATH: Path = Path("trained_data/models/offline_sizer_iql.d3")
    JOURNAL_PATH: Path = Path("trained_data/trade_journal_rl.json")

    # Shaped-reward coefficients (aligned with PostTradeLoop)
    SLIP_WEIGHT: float = 1.0
    MAE_WEIGHT: float = 0.5

    # Action normalisation: max forex micro-lot units for [0, 1] scaling
    MAX_UNITS: float = 100_000.0

    def __init__(
        self,
        journal_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
    ) -> None:
        self.journal_path = Path(journal_path) if journal_path else self.JOURNAL_PATH
        self.model_path = Path(model_path) if model_path else self.MODEL_PATH
        self._iql: Any = None  # Lazy-loaded d3rlpy IQL instance
        self._dataset: Any = None  # Cached MDPDataset for rebuild on load

    # ------------------------------------------------------------------
    # Journal loading
    # ------------------------------------------------------------------

    def _load_journal(self) -> List[dict]:
        """Read trade journal and return entries with an ``outcome`` dict.

        Raises
        ------
        InsufficientTrajectoriesError
            If fewer than :pyattr:`MIN_TRAJECTORIES` completed trades exist.
        FileNotFoundError
            If the journal file does not exist.
        """
        if not self.journal_path.exists():
            raise FileNotFoundError(
                f"Trade journal not found at {self.journal_path}"
            )

        try:
            raw_text = self.journal_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error(
                "_load_journal: failed to read %s: %s", self.journal_path, exc
            )
            raise

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error(
                "_load_journal: invalid JSON in %s: %s", self.journal_path, exc
            )
            raise

        if not isinstance(data, list):
            logger.error(
                "_load_journal: expected list, got %s", type(data).__name__
            )
            raise ValueError(
                f"Journal root must be a JSON array, got {type(data).__name__}"
            )

        completed = [
            entry for entry in data
            if isinstance(entry, dict) and isinstance(entry.get("outcome"), dict)
        ]

        if len(completed) < self.MIN_TRAJECTORIES:
            raise InsufficientTrajectoriesError(
                f"Only {len(completed)} completed trades in journal "
                f"(minimum {self.MIN_TRAJECTORIES} required for offline IQL training)"
            )

        logger.info(
            "_load_journal: %d completed trades loaded from %s",
            len(completed),
            self.journal_path,
        )
        return completed

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _extract_action(self, entry: dict) -> float:
        """Extract normalised action (position size) from a journal entry.

        Looks for ``units``, ``lots``, or ``position_size`` in the entry
        and normalises to [0, 1] range.
        """
        for key in ("units", "lots", "position_size"):
            val = entry.get(key)
            if val is not None:
                try:
                    raw = float(val)
                    return np.clip(abs(raw) / self.MAX_UNITS, 0.0, 1.0)
                except (TypeError, ValueError):
                    continue

        # Fallback: check inside outcome
        outcome = entry.get("outcome", {})
        for key in ("units", "lots", "position_size"):
            val = outcome.get(key)
            if val is not None:
                try:
                    raw = float(val)
                    return np.clip(abs(raw) / self.MAX_UNITS, 0.0, 1.0)
                except (TypeError, ValueError):
                    continue

        return 0.0

    def _extract_reward(self, entry: dict) -> float:
        """Compute shaped reward from an entry's outcome dict.

        Reward = pnl_pips - slippage_pips * SLIP_WEIGHT - mae_pips * MAE_WEIGHT

        Falls back to raw ``pnl`` or ``pnl_pips`` if shaping fields are
        absent.
        """
        outcome = entry.get("outcome", {})

        pnl_pips = 0.0
        for key in ("pnl_pips", "pnl", "profit_pips"):
            val = outcome.get(key)
            if val is not None:
                try:
                    pnl_pips = float(val)
                    break
                except (TypeError, ValueError):
                    continue

        slip = 0.0
        for key in ("slippage_pips", "slippage"):
            val = outcome.get(key)
            if val is not None:
                try:
                    slip = float(val)
                    break
                except (TypeError, ValueError):
                    continue

        mae = 0.0
        for key in ("mae_pips", "mae"):
            val = outcome.get(key)
            if val is not None:
                try:
                    mae = float(val)
                    break
                except (TypeError, ValueError):
                    continue

        return pnl_pips - slip * self.SLIP_WEIGHT - mae * self.MAE_WEIGHT

    def _build_dataset(self, journal: List[dict]) -> Any:
        """Transform journal entries into a d3rlpy MDPDataset.

        Each journal entry becomes one (S, A, R) transition.  The terminal
        flag is set to ``True`` for the last entry (single-episode framing)
        and at day boundaries when ``timestamp`` is available.

        Returns
        -------
        d3rlpy.dataset.MDPDataset
        """
        try:
            from d3rlpy.dataset import MDPDataset
        except ImportError as exc:
            logger.error(
                "_build_dataset: d3rlpy not installed: %s", exc
            )
            raise ImportError(
                "d3rlpy >= 2.8.0 is required for offline IQL. "
                "Install with: pip install d3rlpy>=2.8.0"
            ) from exc

        n = len(journal)
        observations = np.zeros((n, STATE_DIM), dtype=np.float32)
        actions = np.zeros((n, 1), dtype=np.float32)
        rewards = np.zeros(n, dtype=np.float32)
        terminals = np.zeros(n, dtype=np.float32)

        for i, entry in enumerate(journal):
            observations[i] = extract_state(entry)
            actions[i, 0] = self._extract_action(entry)
            rewards[i] = self._extract_reward(entry)

        # Terminal: mark last entry as terminal; also mark day boundaries
        terminals[-1] = 1.0
        for i in range(n - 1):
            ts_cur = journal[i].get("timestamp", "")
            ts_nxt = journal[i + 1].get("timestamp", "")
            if ts_cur and ts_nxt:
                # Different date => episode boundary
                day_cur = str(ts_cur)[:10]
                day_nxt = str(ts_nxt)[:10]
                if day_cur != day_nxt:
                    terminals[i] = 1.0

        # Normalise observations per-feature to [0, 1] (min-max)
        obs_min = observations.min(axis=0)
        obs_max = observations.max(axis=0)
        obs_range = obs_max - obs_min
        obs_range[obs_range < 1e-8] = 1.0  # avoid division by zero
        observations = (observations - obs_min) / obs_range

        # Store normalisation params for inference
        self._obs_min = obs_min
        self._obs_max = obs_max
        self._obs_range = obs_range

        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )

        logger.info(
            "_build_dataset: observations=%s, actions=%s, "
            "reward_range=[%.3f, %.3f], episodes=%d",
            observations.shape,
            actions.shape,
            float(rewards.min()),
            float(rewards.max()),
            int(terminals.sum()),
        )

        return dataset

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        n_epochs: Optional[int] = None,
        dry_run: bool = False,
    ) -> Path:
        """Train the IQL model on journal data.

        Parameters
        ----------
        n_epochs : int or None
            Number of training epochs.  Defaults to ``DEFAULT_EPOCHS``.
        dry_run : bool
            If ``True``, build the dataset and print stats without training.

        Returns
        -------
        Path
            Path to the saved model file.

        Raises
        ------
        InsufficientTrajectoriesError
            If the journal has fewer than ``MIN_TRAJECTORIES`` trades.
        ImportError
            If d3rlpy is not installed.
        """
        try:
            from d3rlpy.algos import IQLConfig
        except ImportError as exc:
            logger.error("train: d3rlpy not installed: %s", exc)
            raise ImportError(
                "d3rlpy >= 2.8.0 is required for offline IQL. "
                "Install with: pip install d3rlpy>=2.8.0"
            ) from exc

        epochs = n_epochs if n_epochs is not None else self.DEFAULT_EPOCHS

        journal = self._load_journal()
        dataset = self._build_dataset(journal)
        self._dataset = dataset

        n_transitions = dataset.transition_count
        obs_shape = (n_transitions, STATE_DIM)
        action_shape = (n_transitions, 1)

        if dry_run:
            logger.info(
                "train [DRY RUN]: observations=%s, actions=%s, "
                "transitions=%d, planned_epochs=%d",
                obs_shape,
                action_shape,
                n_transitions,
                epochs,
            )
            print(f"[DRY RUN] Dataset stats:")
            print(f"  observations shape : {obs_shape}")
            print(f"  actions shape      : {action_shape}")
            print(f"  transitions        : {n_transitions}")
            print(f"  planned epochs     : {epochs}")
            return self.model_path

        # Build IQL on CPU (practice account, no GPU needed)
        iql = IQLConfig(
            batch_size=256,
            gamma=0.99,
            expectile=0.7,
            weight_temp=3.0,
            max_weight=100.0,
        ).create(device="cpu:0")

        steps_per_epoch = max(n_transitions, 1000)
        total_steps = epochs * steps_per_epoch

        logger.info(
            "train: starting IQL fit, n_steps=%d, steps_per_epoch=%d",
            total_steps,
            steps_per_epoch,
        )

        results = iql.fit(
            dataset,
            n_steps=total_steps,
            n_steps_per_epoch=steps_per_epoch,
            show_progress=True,
        )

        # Persist model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        iql.save(str(self.model_path))
        self._iql = iql

        # Also save normalisation parameters alongside model
        norm_path = self.model_path.with_suffix(".norm.npz")
        np.savez(
            str(norm_path),
            obs_min=self._obs_min,
            obs_max=self._obs_max,
            obs_range=self._obs_range,
        )

        # Log training summary
        final_metrics: Dict[str, float] = {}
        if results:
            _, final_metrics = results[-1]
        critic_loss = final_metrics.get("critic_loss", float("nan"))
        actor_loss = final_metrics.get("actor_loss", float("nan"))

        logger.info(
            "train: complete. transitions=%d, obs_dim=%d, action_dim=1, "
            "final_critic_loss=%.4f, final_actor_loss=%.4f, model=%s",
            n_transitions,
            STATE_DIM,
            critic_loss,
            actor_loss,
            self.model_path,
        )

        return self.model_path

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the persisted IQL model if not already in memory.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        RuntimeError
            If loading fails for any other reason.
        """
        if self._iql is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No trained offline sizer model at {self.model_path}. "
                "Run train() first."
            )

        try:
            import d3rlpy
            self._iql = d3rlpy.load_learnable(
                str(self.model_path), device="cpu:0"
            )
        except Exception as exc:
            logger.error(
                "_ensure_loaded: failed to load model from %s: %s",
                self.model_path,
                exc,
            )
            raise RuntimeError(
                f"Failed to load offline sizer model from {self.model_path}: {exc}"
            ) from exc

        # Load normalisation parameters
        norm_path = self.model_path.with_suffix(".norm.npz")
        if norm_path.exists():
            try:
                data = np.load(str(norm_path))
                self._obs_min = data["obs_min"]
                self._obs_max = data["obs_max"]
                self._obs_range = data["obs_range"]
            except Exception as exc:
                logger.warning(
                    "_ensure_loaded: failed to load norm params from %s: %s",
                    norm_path,
                    exc,
                )
                self._obs_min = None
                self._obs_max = None
                self._obs_range = None
        else:
            self._obs_min = None
            self._obs_max = None
            self._obs_range = None

        logger.info("_ensure_loaded: model loaded from %s", self.model_path)

    def _normalise_state(self, state: np.ndarray) -> np.ndarray:
        """Apply min-max normalisation using training-time statistics."""
        if self._obs_min is not None and self._obs_range is not None:
            return (state - self._obs_min) / self._obs_range
        return state

    def predict(self, state: np.ndarray, max_lots: float = 1.0) -> float:
        """Predict position size for a given state vector.

        Parameters
        ----------
        state : np.ndarray
            Feature vector of shape ``(STATE_DIM,)`` or ``(1, STATE_DIM)``.
        max_lots : float
            Upper bound for the returned lot size.  The raw IQL output is
            clamped to ``[0, max_lots]``.

        Returns
        -------
        float
            Recommended position size in lots.
        """
        self._ensure_loaded()

        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state.reshape(1, -1)

        state = self._normalise_state(state)

        raw_action = self._iql.predict(state)[0]
        # d3rlpy returns shape (1,) for continuous actions
        if hasattr(raw_action, "__len__"):
            raw_action = float(raw_action[0])
        else:
            raw_action = float(raw_action)

        # De-normalise from [0, 1] to lots and clamp
        lots = float(np.clip(raw_action * max_lots, 0.0, max_lots))

        logger.debug(
            "predict: raw_action=%.4f, clamped_lots=%.4f", raw_action, lots
        )
        return lots


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def train_offline_sizer(
    journal_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    n_epochs: int = 100,
    dry_run: bool = False,
) -> Path:
    """Train an offline IQL position sizer and return the model path.

    Convenience wrapper matching the ``train_rl_position_sizer(...)``
    pattern from the online PPO module.

    Parameters
    ----------
    journal_path : Path or None
        Path to trade journal.  Defaults to ``OfflineSizerIQL.JOURNAL_PATH``.
    output_path : Path or None
        Where to save the model.  Defaults to ``OfflineSizerIQL.MODEL_PATH``.
    n_epochs : int
        Training epochs.
    dry_run : bool
        If ``True``, only print dataset stats without training.

    Returns
    -------
    Path
        Path to the saved (or would-be-saved) model file.
    """
    sizer = OfflineSizerIQL(
        journal_path=journal_path,
        model_path=output_path,
    )
    return sizer.train(n_epochs=n_epochs, dry_run=dry_run)


def load_offline_sizer(path: Optional[Path] = None) -> OfflineSizerIQL:
    """Load a trained offline IQL sizer from disk.

    Parameters
    ----------
    path : Path or None
        Model path.  Defaults to ``OfflineSizerIQL.MODEL_PATH``.

    Returns
    -------
    OfflineSizerIQL
        Sizer instance with the model loaded and ready for ``predict()``.
    """
    sizer = OfflineSizerIQL(model_path=path)
    sizer._ensure_loaded()
    return sizer
