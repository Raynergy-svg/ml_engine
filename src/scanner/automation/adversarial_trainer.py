"""Adversarial robustness training for ensemble models.

US-089: Generate adversarial market scenarios (stop-hunting, liquidity
traps, correlation breaks) and retrain ensemble on combined original +
adversarial data. Track robustness score (adversarial vs clean accuracy).
Based on 2025 adversarial DRL research showing 94% baseline retention
with 90% attack defense.

Three adversarial scenario types:
1. stop_hunt — Volatility spike that hunts stop losses before reversing
2. spoof — Fake liquidity/momentum that reverses (liquidity trap)
3. correlation_trap — Correlated pairs temporarily decouple

Training loop:
- Generate adversarial perturbations of training data
- Combine original + adversarial into augmented dataset
- Train model on augmented data
- Measure clean accuracy, adversarial accuracy, robustness ratio
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ROBUSTNESS_PATH = _PROJECT_ROOT / "trained_data" / "adversarial_robustness.json"


@dataclass
class RobustnessScore:
    """Result of robustness evaluation."""

    clean_accuracy: float = 0.0
    adversarial_accuracy: float = 0.0
    robustness_ratio: float = 0.0   # adv_acc / clean_acc
    scenario_scores: Dict[str, float] = field(default_factory=dict)
    n_clean_samples: int = 0
    n_adversarial_samples: int = 0
    training_rounds: int = 0

    def to_dict(self) -> dict:
        return {
            "clean_accuracy": round(self.clean_accuracy, 4),
            "adversarial_accuracy": round(self.adversarial_accuracy, 4),
            "robustness_ratio": round(self.robustness_ratio, 4),
            "scenario_scores": {
                k: round(v, 4) for k, v in self.scenario_scores.items()
            },
            "n_clean_samples": self.n_clean_samples,
            "n_adversarial_samples": self.n_adversarial_samples,
            "training_rounds": self.training_rounds,
        }


@dataclass
class AdversarialBatch:
    """A batch of adversarial examples."""

    X: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    scenario_type: str = ""
    epsilon: float = 0.0
    n_samples: int = 0


class AdversarialTrainer:
    """Generate adversarial scenarios and train robust ensemble models.

    Creates three types of market adversarial perturbations:
    - stop_hunt: Volatility spikes that trigger stops before reversal
    - spoof: Fake momentum signals that reverse (liquidity traps)
    - correlation_trap: Features that normally correlate suddenly decouple

    The trainer augments training data with these scenarios and measures
    how well the model maintains accuracy on both clean and adversarial
    inputs.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        adversarial_ratio: float = 0.30,
        persistence_path: Optional[Path] = None,
    ):
        """
        Args:
            epsilon: Perturbation magnitude (fraction of feature range)
            adversarial_ratio: Fraction of adversarial samples to add
            persistence_path: Where to save robustness scores
        """
        self.epsilon = epsilon
        self.adversarial_ratio = adversarial_ratio
        self.persistence_path = persistence_path or _ROBUSTNESS_PATH

        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_adversarial(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epsilon: Optional[float] = None,
    ) -> Dict[str, AdversarialBatch]:
        """Generate adversarial perturbations of training data.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            epsilon: Perturbation magnitude (uses default if None)

        Returns:
            Dict of scenario_type -> AdversarialBatch
        """
        eps = epsilon or self.epsilon
        n_samples = len(X_train)
        n_adv = max(1, int(n_samples * self.adversarial_ratio))

        batches: Dict[str, AdversarialBatch] = {}

        # Scenario 1: Stop hunt (volatility spike → reversal)
        batches["stop_hunt"] = self._generate_stop_hunt(
            X_train, y_train, n_adv, eps
        )

        # Scenario 2: Spoof (fake momentum reversal)
        batches["spoof"] = self._generate_spoof(
            X_train, y_train, n_adv, eps
        )

        # Scenario 3: Correlation trap (feature decoupling)
        batches["correlation_trap"] = self._generate_correlation_trap(
            X_train, y_train, n_adv, eps
        )

        total = sum(b.n_samples for b in batches.values())
        logger.info(
            f"AdversarialTrainer: Generated {total} adversarial samples "
            f"across {len(batches)} scenarios (eps={eps:.3f})"
        )

        return batches

    def compute_robustness_score(
        self,
        model: Any,
        X_clean: np.ndarray,
        X_adversarial: np.ndarray,
        y: np.ndarray,
    ) -> RobustnessScore:
        """Compute robustness metrics for a model.

        Args:
            model: Sklearn-compatible model with predict() method
            X_clean: Clean test features
            X_adversarial: Adversarial test features (same labels)
            y: True labels

        Returns:
            RobustnessScore with clean/adversarial accuracy and ratio
        """
        result = RobustnessScore(
            n_clean_samples=len(X_clean),
            n_adversarial_samples=len(X_adversarial),
        )

        try:
            # Clean accuracy
            clean_preds = model.predict(X_clean)
            result.clean_accuracy = float(np.mean(clean_preds == y[:len(clean_preds)]))

            # Adversarial accuracy
            if len(X_adversarial) > 0:
                adv_preds = model.predict(X_adversarial)
                y_adv = y[:len(adv_preds)]
                result.adversarial_accuracy = float(np.mean(adv_preds == y_adv))

            # Robustness ratio
            if result.clean_accuracy > 0:
                result.robustness_ratio = (
                    result.adversarial_accuracy / result.clean_accuracy
                )
            else:
                result.robustness_ratio = 0.0

            logger.info(
                f"AdversarialTrainer: Robustness — "
                f"clean={result.clean_accuracy:.3f}, "
                f"adversarial={result.adversarial_accuracy:.3f}, "
                f"ratio={result.robustness_ratio:.3f}"
            )
        except Exception as e:
            logger.warning(f"AdversarialTrainer: Robustness eval failed: {e}")

        return result

    def train_robust(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_rounds: int = 3,
    ) -> Tuple[Any, RobustnessScore]:
        """Iteratively train model on clean + adversarial data.

        Each round:
        1. Generate adversarial examples from current training data
        2. Combine clean + adversarial
        3. Retrain model on combined data
        4. Measure robustness

        Args:
            model: Sklearn-compatible model with fit() and predict()
            X_train: Original training features
            y_train: Original training labels
            n_rounds: Number of adversarial training rounds

        Returns:
            (trained_model, final_robustness_score)
        """
        best_score = RobustnessScore()
        current_X = X_train.copy()
        current_y = y_train.copy()

        for round_idx in range(n_rounds):
            # Generate adversarial examples
            adv_batches = self.generate_adversarial(
                current_X, current_y,
                epsilon=self.epsilon * (1.0 + round_idx * 0.2),  # Slightly increase eps each round
            )

            # Combine clean + adversarial
            adv_X_list = [current_X]
            adv_y_list = [current_y]
            for batch in adv_batches.values():
                if batch.n_samples > 0:
                    adv_X_list.append(batch.X)
                    adv_y_list.append(batch.y)

            combined_X = np.vstack(adv_X_list)
            combined_y = np.concatenate(adv_y_list)

            # Shuffle
            idx = np.random.permutation(len(combined_X))
            combined_X = combined_X[idx]
            combined_y = combined_y[idx]

            # Retrain
            try:
                model.fit(combined_X, combined_y)
            except Exception as e:
                logger.warning(
                    f"AdversarialTrainer: Round {round_idx + 1} training failed: {e}"
                )
                break

            # Evaluate
            # Use last batch's adversarial data for evaluation
            all_adv_X = np.vstack([b.X for b in adv_batches.values() if b.n_samples > 0])
            all_adv_y = np.concatenate([b.y for b in adv_batches.values() if b.n_samples > 0])

            score = self.compute_robustness_score(
                model, current_X, all_adv_X, current_y
            )
            score.training_rounds = round_idx + 1

            # Per-scenario scores
            for scenario, batch in adv_batches.items():
                if batch.n_samples > 0:
                    try:
                        preds = model.predict(batch.X)
                        score.scenario_scores[scenario] = float(
                            np.mean(preds == batch.y[:len(preds)])
                        )
                    except Exception:
                        score.scenario_scores[scenario] = 0.0

            logger.info(
                f"AdversarialTrainer: Round {round_idx + 1}/{n_rounds} — "
                f"clean={score.clean_accuracy:.3f}, "
                f"adv={score.adversarial_accuracy:.3f}, "
                f"ratio={score.robustness_ratio:.3f}"
            )

            best_score = score

        # Persist results
        self._history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "score": best_score.to_dict(),
        })
        self.save_scores()

        return model, best_score

    # ------------------------------------------------------------------
    # Adversarial scenario generators
    # ------------------------------------------------------------------

    def _generate_stop_hunt(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        epsilon: float,
    ) -> AdversarialBatch:
        """Stop hunt: spike volatility features, flip labels.

        Simulates stop-loss hunting where volatility spikes trigger stops
        before the market reverses to the original direction.
        """
        n = min(n_samples, len(X))
        indices = np.random.choice(len(X), n, replace=False)

        X_adv = X[indices].copy()
        y_adv = y[indices].copy()

        # Spike volatility-related features (typically cols with high variance)
        feature_stds = np.std(X, axis=0)
        vol_cols = np.argsort(feature_stds)[-3:]  # Top 3 highest variance features

        for col in vol_cols:
            # Sharp spike followed by mean-reversion signal
            X_adv[:, col] += np.random.uniform(
                epsilon * 2, epsilon * 5, size=n
            ) * feature_stds[col]

        # Flip labels (the stop hunt reverses)
        y_adv = 1 - y_adv if y_adv.max() <= 1 else y_adv

        return AdversarialBatch(
            X=X_adv,
            y=y_adv,
            scenario_type="stop_hunt",
            epsilon=epsilon,
            n_samples=n,
        )

    def _generate_spoof(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        epsilon: float,
    ) -> AdversarialBatch:
        """Spoof: inflate momentum features, then reverse.

        Simulates fake momentum (spoofing/liquidity trap) where momentum
        indicators look strong but the move is fake.
        """
        n = min(n_samples, len(X))
        indices = np.random.choice(len(X), n, replace=False)

        X_adv = X[indices].copy()
        y_adv = y[indices].copy()

        # Amplify features that correlate with momentum
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)

        # Perturb all features slightly in the "confident" direction
        for col in range(X_adv.shape[1]):
            if feature_stds[col] > 1e-10:
                # Make features look more confident than they should be
                X_adv[:, col] += np.random.normal(
                    epsilon * 1.5, epsilon * 0.5, size=n
                ) * feature_stds[col]

        # Keep same labels but flip some (spoof tricks the model)
        flip_mask = np.random.random(n) < 0.6
        y_adv[flip_mask] = 1 - y_adv[flip_mask] if y_adv.max() <= 1 else y_adv[flip_mask]

        return AdversarialBatch(
            X=X_adv,
            y=y_adv,
            scenario_type="spoof",
            epsilon=epsilon,
            n_samples=n,
        )

    def _generate_correlation_trap(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        epsilon: float,
    ) -> AdversarialBatch:
        """Correlation trap: decouple normally correlated features.

        Simulates correlation breakdown where features that usually move
        together suddenly diverge (e.g., during liquidity crisis).
        """
        n = min(n_samples, len(X))
        indices = np.random.choice(len(X), n, replace=False)

        X_adv = X[indices].copy()
        y_adv = y[indices].copy()

        n_features = X_adv.shape[1]
        if n_features < 2:
            return AdversarialBatch(
                X=X_adv, y=y_adv,
                scenario_type="correlation_trap",
                epsilon=epsilon, n_samples=n,
            )

        # Find correlated feature pairs
        try:
            corr_matrix = np.corrcoef(X.T)
            # Find top correlated pairs (excluding diagonal)
            np.fill_diagonal(corr_matrix, 0)
            flat_idx = np.argsort(np.abs(corr_matrix).ravel())[-6:]
            pairs = [(idx // n_features, idx % n_features) for idx in flat_idx]
        except Exception:
            # Fallback: just perturb adjacent features in opposite directions
            pairs = [(i, i + 1) for i in range(0, min(6, n_features - 1), 2)]

        feature_stds = np.std(X, axis=0)

        for col_a, col_b in pairs:
            if col_a >= n_features or col_b >= n_features:
                continue
            if feature_stds[col_a] < 1e-10 or feature_stds[col_b] < 1e-10:
                continue

            # Move correlated features in opposite directions (decorrelate)
            X_adv[:, col_a] += np.random.normal(
                epsilon * 2, epsilon, size=n
            ) * feature_stds[col_a]
            X_adv[:, col_b] -= np.random.normal(
                epsilon * 2, epsilon, size=n
            ) * feature_stds[col_b]

        return AdversarialBatch(
            X=X_adv,
            y=y_adv,
            scenario_type="correlation_trap",
            epsilon=epsilon,
            n_samples=n,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_scores(self) -> None:
        """Persist robustness score history."""
        try:
            data = {
                "version": 1,
                "history": self._history[-20:],  # Keep last 20
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"AdversarialTrainer: Failed to save scores: {e}")

    def load_scores(self) -> List[Dict[str, Any]]:
        """Load persisted robustness score history."""
        if not self.persistence_path.exists():
            return []
        try:
            data = json.loads(self.persistence_path.read_text())
            self._history = data.get("history", [])
            return self._history
        except Exception as e:
            logger.debug(f"AdversarialTrainer: Failed to load scores: {e}")
            return []

    def get_latest_score(self) -> Optional[RobustnessScore]:
        """Get the most recent robustness score."""
        if not self._history:
            self.load_scores()
        if not self._history:
            return None
        latest = self._history[-1].get("score", {})
        return RobustnessScore(
            clean_accuracy=latest.get("clean_accuracy", 0.0),
            adversarial_accuracy=latest.get("adversarial_accuracy", 0.0),
            robustness_ratio=latest.get("robustness_ratio", 0.0),
            scenario_scores=latest.get("scenario_scores", {}),
            n_clean_samples=latest.get("n_clean_samples", 0),
            n_adversarial_samples=latest.get("n_adversarial_samples", 0),
            training_rounds=latest.get("training_rounds", 0),
        )
