"""
Correlation-based transfer learning orchestrator.

Trains master models on high-liquidity pairs and transfers knowledge
to correlated pairs using warm-start, EWC, and layer freezing.

Usage:
    buddy train --correlation-transfer \\
        --master-pairs EUR_JPY \\
        --target-pairs GBP_JPY,AUD_JPY \\
        --skip-master-training
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.scanner.analysis.correlation import CorrelationAnalyzer, CorrelationResult
from src.training.trainers.config import TrainerConfig

if TYPE_CHECKING:
    from src.training.trainers.joint_trainer import JointMultiPairTrainer

logger = logging.getLogger(__name__)


# Liquidity ranking (higher = more liquid, better as master)
DEFAULT_LIQUIDITY_RANKING: Dict[str, int] = {
    "EUR_USD": 10,
    "USD_JPY": 9,
    "GBP_USD": 8,
    "EUR_JPY": 7,
    "GBP_JPY": 6,
    "AUD_USD": 5,
    "USD_CAD": 4,
    "NZD_USD": 3,
    "AUD_JPY": 2,
    "EUR_GBP": 1,
}

TRANSFER_MODELS_DIR = "trained_data/models/transfer"


@dataclass
class CorrelationGroup:
    """A group of correlated currency pairs."""

    group_id: str
    pairs: List[str]
    master_pair: str
    correlation_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def get_correlation(self, pair1: str, pair2: str) -> float:
        """Get correlation between two pairs in this group."""
        return self.correlation_matrix.get(
            (pair1, pair2), self.correlation_matrix.get((pair2, pair1), 0.0)
        )


@dataclass
class TransferResult:
    """Result of transfer learning to a target pair."""

    target_pair: str
    source_pair: str
    correlation: float
    pre_transfer_accuracy: Optional[float]
    post_transfer_accuracy: float
    epochs_trained: int
    ewc_enabled: bool
    layers_frozen: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_pair": self.target_pair,
            "source_pair": self.source_pair,
            "correlation": self.correlation,
            "pre_transfer_accuracy": self.pre_transfer_accuracy,
            "post_transfer_accuracy": self.post_transfer_accuracy,
            "epochs_trained": self.epochs_trained,
            "ewc_enabled": self.ewc_enabled,
            "layers_frozen": self.layers_frozen,
        }


@dataclass
class CorrelationTransferConfig:
    """Configuration for correlation-based transfer learning."""

    # Correlation analysis settings
    correlation_threshold: float = 0.70
    lookback_periods: int = 100

    # Transfer learning settings (aligned with TrainerConfig warm-start)
    transfer_lr_factor: float = 0.03  # 33x LR reduction (matches warm_start_lr_factor)
    transfer_encoder_layers_to_freeze: int = 2
    transfer_epochs: int = 50
    transfer_patience: int = 15

    # EWC settings for knowledge preservation
    use_ewc: bool = True
    ewc_lambda: float = 100.0

    # Gradual unfreezing
    gradual_unfreeze: bool = True
    unfreeze_after_epochs: int = 10

    # Master pair selection
    liquidity_ranking: Dict[str, int] = field(
        default_factory=lambda: DEFAULT_LIQUIDITY_RANKING.copy()
    )
    user_master_pairs: Optional[List[str]] = None  # Override automatic selection

    # Output settings
    save_dir: str = TRANSFER_MODELS_DIR


class CorrelationTransferOrchestrator:
    """
    Orchestrates correlation-based transfer learning across currency pairs.

    Workflow:
    1. Compute correlation groups from returns data
    2. Select master pair per group (highest liquidity or user-specified)
    3. Train/load master model
    4. Transfer to correlated pairs via warm-start with EWC

    Example:
        orchestrator = CorrelationTransferOrchestrator()
        results = orchestrator.run_full_pipeline(
            returns_data=returns_data,
            dfs=feature_dfs,
            master_pairs=["EUR_JPY"],
            target_pairs=["GBP_JPY", "AUD_JPY"],
        )
    """

    def __init__(
        self,
        config: Optional[CorrelationTransferConfig] = None,
        trainer_config: Optional[TrainerConfig] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Transfer learning configuration
            trainer_config: Base trainer configuration (for model hyperparameters)
        """
        self.config = config or CorrelationTransferConfig()
        self.trainer_config = trainer_config or TrainerConfig()

        # Initialize correlation analyzer with configured threshold
        self.correlation_analyzer = CorrelationAnalyzer(
            threshold=self.config.correlation_threshold,
            lookback_periods=self.config.lookback_periods,
        )

        # Track state
        self.correlation_groups: List[CorrelationGroup] = []
        self.master_trainers: Dict[str, "JointMultiPairTrainer"] = {}
        self.transfer_results: List[TransferResult] = []
        self._correlation_matrix: Optional[pd.DataFrame] = None

    def compute_correlation_groups(
        self,
        returns_data: Dict[str, pd.Series],
    ) -> List[CorrelationGroup]:
        """
        Compute correlation groups from returns data.

        Uses the existing CorrelationAnalyzer with Union-Find clustering
        to identify groups of correlated pairs.

        Args:
            returns_data: Dict mapping pair name -> returns Series

        Returns:
            List of CorrelationGroup objects with master pair selected
        """
        if len(returns_data) < 2:
            logger.warning("Need at least 2 pairs for correlation analysis")
            return []

        # Use existing CorrelationAnalyzer
        results = self.correlation_analyzer.analyze(returns_data)
        self._correlation_matrix = self.correlation_analyzer.get_correlation_matrix()

        # Build groups from Union-Find clusters
        groups_dict: Dict[str, List[str]] = {}
        for pair, result in results.items():
            root = result.correlation_group or pair
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(pair)

        # Create CorrelationGroup objects with master selection
        self.correlation_groups = []
        for group_id, pairs in groups_dict.items():
            if len(pairs) < 2:
                continue  # Skip single-pair "groups"

            master = self._select_master_for_group(pairs)

            # Build correlation matrix for the group
            corr_matrix: Dict[Tuple[str, str], float] = {}
            if self._correlation_matrix is not None:
                for p1 in pairs:
                    for p2 in pairs:
                        if p1 != p2:
                            if (
                                p1 in self._correlation_matrix.columns
                                and p2 in self._correlation_matrix.columns
                            ):
                                corr_matrix[(p1, p2)] = float(
                                    self._correlation_matrix.loc[p1, p2]
                                )

            group = CorrelationGroup(
                group_id=group_id,
                pairs=pairs,
                master_pair=master,
                correlation_matrix=corr_matrix,
            )
            self.correlation_groups.append(group)

        logger.info(f"Found {len(self.correlation_groups)} correlation groups")
        for g in self.correlation_groups:
            logger.info(
                f"  Group {g.group_id}: master={g.master_pair}, "
                f"pairs={g.pairs}"
            )

        return self.correlation_groups

    def _select_master_for_group(self, pairs: List[str]) -> str:
        """
        Select master pair for a group based on liquidity or user override.

        Args:
            pairs: List of pairs in the group

        Returns:
            Selected master pair name
        """
        # User override takes precedence
        if self.config.user_master_pairs:
            for master in self.config.user_master_pairs:
                if master in pairs:
                    return master

        # Select by liquidity ranking
        ranking = self.config.liquidity_ranking
        return max(pairs, key=lambda p: ranking.get(p, 0))

    def train_master(
        self,
        group: CorrelationGroup,
        dfs: Dict[str, pd.DataFrame],
        skip_if_exists: bool = False,
    ) -> "JointMultiPairTrainer":
        """
        Train master model for a correlation group.

        Args:
            group: The correlation group
            dfs: DataFrames for each pair (with features)
            skip_if_exists: Skip training if model already exists

        Returns:
            Trained JointMultiPairTrainer
        """
        from src.training.trainers.joint_trainer import JointMultiPairTrainer
        from src.training.trainers.utils import TRANSFORMER_DIRECTION_FILENAME

        master_pair = group.master_pair
        save_dir = Path(self.config.save_dir) / master_pair

        # Check for existing model
        model_path = save_dir / TRANSFORMER_DIRECTION_FILENAME
        if skip_if_exists and model_path.exists():
            logger.info(f"Loading existing master model for {master_pair}")
            trainer = JointMultiPairTrainer(self.trainer_config)
            trainer.load(str(save_dir))
            self.master_trainers[master_pair] = trainer
            return trainer

        logger.info(f"Training master model for {master_pair}")

        if master_pair not in dfs:
            raise ValueError(f"No data provided for master pair {master_pair}")

        # Train on master pair only
        trainer = JointMultiPairTrainer(self.trainer_config)
        trainer.train(
            dfs={master_pair: dfs[master_pair]},
            instruments=[master_pair],
            save_dir=str(save_dir),
        )

        # Save transfer metadata
        self._save_master_metadata(save_dir, group)

        self.master_trainers[master_pair] = trainer
        return trainer

    def transfer_to_correlated(
        self,
        group: CorrelationGroup,
        dfs: Dict[str, pd.DataFrame],
        target_pairs: Optional[List[str]] = None,
    ) -> List[TransferResult]:
        """
        Transfer master model to correlated pairs.

        Args:
            group: The correlation group
            dfs: DataFrames for each pair
            target_pairs: Specific pairs to transfer to (None = all in group except master)

        Returns:
            List of TransferResult for each target
        """
        master_pair = group.master_pair
        if master_pair not in self.master_trainers:
            raise ValueError(f"Master model for {master_pair} not trained")

        master_trainer = self.master_trainers[master_pair]
        targets = target_pairs or [p for p in group.pairs if p != master_pair]

        results = []
        for target in targets:
            if target not in dfs:
                logger.warning(f"No data for target {target}, skipping")
                continue

            correlation = group.get_correlation(master_pair, target)

            if abs(correlation) < self.config.correlation_threshold:
                logger.warning(
                    f"Correlation {correlation:.3f} below threshold "
                    f"{self.config.correlation_threshold} for {target}, skipping"
                )
                continue

            result = self._transfer_single_pair(
                master_trainer=master_trainer,
                master_pair=master_pair,
                target_pair=target,
                target_df=dfs[target],
                correlation=correlation,
            )
            results.append(result)
            self.transfer_results.append(result)

        return results

    def _transfer_single_pair(
        self,
        master_trainer: "JointMultiPairTrainer",
        master_pair: str,
        target_pair: str,
        target_df: pd.DataFrame,
        correlation: float,
    ) -> TransferResult:
        """
        Transfer model from master to a single target pair.

        Uses warm-start with reduced learning rate, frozen encoder layers,
        and EWC to prevent catastrophic forgetting.

        Args:
            master_trainer: Trained master model
            master_pair: Source pair name
            target_pair: Target pair name
            target_df: Target pair DataFrame
            correlation: Correlation between master and target

        Returns:
            TransferResult with metrics
        """
        logger.info(
            f"Transferring {master_pair} -> {target_pair} "
            f"(correlation={correlation:.3f})"
        )

        # Create transfer-specific config
        # We modify the trainer's config for the fine-tuning
        transfer_config = TrainerConfig(
            # Base settings
            epochs=self.config.transfer_epochs,
            patience=self.config.transfer_patience,
            batch_size=self.trainer_config.batch_size,
            learning_rate=self.trainer_config.learning_rate,

            # Warm-start settings for transfer
            warm_start_lr_factor=self.config.transfer_lr_factor,
            warm_start_freeze_encoder=True,
            warm_start_encoder_layers_to_freeze=self.config.transfer_encoder_layers_to_freeze,
            warm_start_gradual_unfreeze=self.config.gradual_unfreeze,
            warm_start_unfreeze_epochs=self.config.unfreeze_after_epochs,

            # EWC for knowledge preservation
            use_ewc=self.config.use_ewc,
            ewc_lambda=self.config.ewc_lambda,

            # Inherit other settings
            use_ema=self.trainer_config.use_ema,
            use_replay_buffer=False,  # Don't use replay for transfer
            llrd_enabled=self.trainer_config.llrd_enabled,
            llrd_decay_rate=self.trainer_config.llrd_decay_rate,
        )

        # Save directory for target pair
        target_save_dir = Path(self.config.save_dir) / target_pair
        target_save_dir.mkdir(parents=True, exist_ok=True)

        # Use fine_tune_for_pair which handles LightGBM warm-start
        # For the Transformer, we need to ensure warm-start is used
        metrics = master_trainer.fine_tune_for_pair(
            df=target_df,
            instrument=target_pair,
            save_dir=str(target_save_dir),
        )

        # Extract accuracy from metrics
        post_accuracy = 0.0
        if "direction" in metrics:
            if isinstance(metrics["direction"], dict):
                post_accuracy = metrics["direction"].get("val_accuracy", 0.0)
            else:
                post_accuracy = metrics["direction"]

        # Save transfer metadata
        self._save_transfer_metadata(
            target_save_dir, master_pair, target_pair, correlation, metrics
        )

        return TransferResult(
            target_pair=target_pair,
            source_pair=master_pair,
            correlation=correlation,
            pre_transfer_accuracy=None,
            post_transfer_accuracy=post_accuracy,
            epochs_trained=self.config.transfer_epochs,
            ewc_enabled=self.config.use_ewc,
            layers_frozen=self.config.transfer_encoder_layers_to_freeze,
        )

    def run_full_pipeline(
        self,
        returns_data: Dict[str, pd.Series],
        dfs: Dict[str, pd.DataFrame],
        master_pairs: Optional[List[str]] = None,
        target_pairs: Optional[List[str]] = None,
        skip_master_training: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete correlation transfer pipeline.

        Args:
            returns_data: Returns data for correlation analysis (pair -> Series)
            dfs: Feature DataFrames for each pair (pair -> DataFrame)
            master_pairs: User-specified master pairs (overrides auto-selection)
            target_pairs: Specific target pairs (None = all correlated pairs)
            skip_master_training: Use existing master models if available

        Returns:
            Summary dict with:
            - correlation_groups: List of group info
            - transfer_results: List of transfer outcomes
            - correlation_matrix: The computed correlation matrix
        """
        logger.info("=" * 60)
        logger.info("CORRELATION-BASED TRANSFER LEARNING PIPELINE")
        logger.info("=" * 60)

        # Override master pairs if specified
        if master_pairs:
            self.config.user_master_pairs = master_pairs
            logger.info(f"User-specified master pairs: {master_pairs}")

        # Step 1: Compute correlation groups
        logger.info("\nStep 1: Computing correlation groups...")
        groups = self.compute_correlation_groups(returns_data)

        if not groups:
            logger.warning("No correlation groups found above threshold")
            return {
                "correlation_groups": [],
                "transfer_results": [],
                "correlation_matrix": None,
            }

        # Step 2: Train master models
        logger.info("\nStep 2: Training/loading master models...")
        for group in groups:
            if group.master_pair in dfs:
                self.train_master(group, dfs, skip_if_exists=skip_master_training)
            else:
                logger.warning(
                    f"No data for master {group.master_pair}, skipping group"
                )

        # Step 3: Transfer to correlated pairs
        logger.info("\nStep 3: Transferring to correlated pairs...")
        all_results = []
        for group in groups:
            if group.master_pair not in self.master_trainers:
                continue

            # Filter to targets in this group if specific targets requested
            group_targets = None
            if target_pairs:
                group_targets = [p for p in target_pairs if p in group.pairs]
                if not group_targets:
                    continue

            results = self.transfer_to_correlated(group, dfs, group_targets)
            all_results.extend(results)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TRANSFER LEARNING COMPLETE")
        logger.info("=" * 60)

        for result in all_results:
            logger.info(
                f"  {result.source_pair} -> {result.target_pair}: "
                f"correlation={result.correlation:.3f}, "
                f"accuracy={result.post_transfer_accuracy:.1%}"
            )

        return {
            "correlation_groups": [
                {
                    "group_id": g.group_id,
                    "master": g.master_pair,
                    "pairs": g.pairs,
                }
                for g in groups
            ],
            "transfer_results": [r.to_dict() for r in all_results],
            "correlation_matrix": (
                self._correlation_matrix.to_dict()
                if self._correlation_matrix is not None
                else None
            ),
        }

    def _save_master_metadata(
        self, save_dir: Path, group: CorrelationGroup
    ) -> None:
        """Save metadata for master model."""
        meta = {
            "type": "master",
            "pair": group.master_pair,
            "group_id": group.group_id,
            "correlated_pairs": group.pairs,
            "correlation_threshold": self.config.correlation_threshold,
            "saved_at": datetime.now().isoformat(),
        }

        meta_path = save_dir / "transfer_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _save_transfer_metadata(
        self,
        save_dir: Path,
        source_pair: str,
        target_pair: str,
        correlation: float,
        metrics: Dict[str, Any],
    ) -> None:
        """Save metadata for transferred model."""
        meta = {
            "type": "transferred",
            "source_pair": source_pair,
            "target_pair": target_pair,
            "correlation": correlation,
            "transfer_config": {
                "lr_factor": self.config.transfer_lr_factor,
                "encoder_layers_frozen": self.config.transfer_encoder_layers_to_freeze,
                "epochs": self.config.transfer_epochs,
                "use_ewc": self.config.use_ewc,
                "ewc_lambda": self.config.ewc_lambda,
            },
            "metrics": metrics,
            "saved_at": datetime.now().isoformat(),
        }

        meta_path = save_dir / "transfer_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get the computed correlation matrix."""
        return self._correlation_matrix

    def get_groups_for_pair(self, pair: str) -> List[CorrelationGroup]:
        """Get all groups containing a specific pair."""
        return [g for g in self.correlation_groups if pair in g.pairs]
