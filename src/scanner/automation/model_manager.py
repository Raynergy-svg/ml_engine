"""
Model Version Manager — A/B testing and safe model promotion.

Tracks model versions, runs shadow tests (candidate vs incumbent),
and auto-promotes when candidate proves superior.

Features:
- Version registration with training metadata
- Shadow test comparison (incumbent vs candidate on live scans)
- Automatic promotion when candidate > incumbent + threshold
- Safe rollback to previous versions
- Corruption-resistant JSONL logging with file locking
"""

from __future__ import annotations

import fcntl
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CandidateModelLoader:
    """Lazy-loads candidate model for true A/B comparison.

    Features:
    - Lazy loading: only loads on first prediction request
    - Graceful failure: candidate unavailability never crashes scanner
    - Memory-safe: can be unloaded when A/B testing disabled
    - Silent fallback: missing candidate file is handled silently

    Example:
        >>> loader = CandidateModelLoader()
        >>> pred = loader.get_prediction(features)
        >>> if pred:
        >>>     print(f"Candidate predicted: {pred}")
    """

    def __init__(self, candidate_path: str = "trained_data/models/buddy_tf_candidate.keras"):
        """Initialize candidate model loader.

        Args:
            candidate_path: Path to candidate model file (.keras format).
                If file doesn't exist, predictions will return None.
        """
        self._model = None
        self._candidate_path = Path(candidate_path)
        self._loaded = False

    def get_prediction(self, features) -> Optional[str]:
        """Run inference on candidate model.

        Args:
            features: Feature array for the model (format must match incumbent)

        Returns:
            Prediction string: "LONG", "SHORT", "HOLD", or None on failure.
            Returns None if:
            - Candidate model file doesn't exist
            - Model failed to load
            - Prediction failed
        """
        if not self._loaded:
            self._load()

        if self._model is None:
            return None  # Candidate not available

        try:
            # Import TensorFlow lazily to avoid conflicts with other ML frameworks
            import tensorflow as tf

            # Reshape features if needed for the model
            if len(features.shape) == 1:
                features_batch = features.reshape(1, -1)
            else:
                features_batch = features

            # Run inference
            predictions = self._model.predict(features_batch, verbose=0)

            # Interpret predictions as direction (LONG/SHORT/HOLD)
            # Assuming model outputs probability: >0.55 = LONG, <0.45 = SHORT, else HOLD
            prob = float(predictions[0][0]) if predictions.ndim > 1 else float(predictions[0])

            if prob > 0.55:
                return "LONG"
            elif prob < 0.45:
                return "SHORT"
            else:
                return "HOLD"

        except Exception as e:
            logger.debug(f"Candidate prediction failed: {e}")
            return None

    def _load(self) -> None:
        """Lazy load candidate model. Only called once via get_prediction.

        Handles missing file and load errors gracefully. Sets self._model to
        the loaded model or None if unavailable. Always sets self._loaded=True
        to prevent repeated load attempts.
        """
        self._loaded = True

        # Silently return if candidate file doesn't exist
        if not self._candidate_path.exists():
            logger.debug(f"Candidate model not found: {self._candidate_path}")
            return

        try:
            import tensorflow as tf

            # Load .keras model
            self._model = tf.keras.models.load_model(str(self._candidate_path))
            logger.debug(f"Candidate model loaded: {self._candidate_path}")

        except Exception as e:
            logger.warning(f"Failed to load candidate model: {e}")
            self._model = None

    def unload(self) -> None:
        """Free memory when shadow testing disabled.

        Call this when enable_true_ab_testing is disabled to release
        the loaded model from memory.
        """
        self._model = None
        self._loaded = False
        logger.debug("Candidate model unloaded")


@dataclass
class ModelVersion:
    """Metadata for a model version."""

    version_tag: str  # e.g. "v1.0.0", "v1.1.0-candidate"
    model_path: str  # path to model directory or file
    registered_at: str  # ISO timestamp
    accuracy: float = 0.0  # validation accuracy at training time
    shadow_scans: int = 0  # number of shadow-test scans completed
    shadow_correct: int = 0  # correct predictions during shadow testing
    shadow_accuracy: float = 0.0
    is_production: bool = False
    is_candidate: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class ShadowTestResult:
    """Result of a single shadow test comparison."""

    pair: str
    incumbent_direction: str  # LONG/SHORT/HOLD
    candidate_direction: str
    actual_movement: float  # actual price change after N candles
    incumbent_correct: bool
    candidate_correct: bool
    timestamp: str


class ModelManager:
    """Manages model versions, shadow testing, and safe promotion."""

    def __init__(
        self,
        models_dir: str = "trained_data/models",
        version_history_path: str = "trained_data/models/version_history.jsonl",
        min_shadow_scans: int = 50,
        min_improvement: float = 0.02,  # 2% improvement threshold
    ):
        """Initialize ModelManager.

        Args:
            models_dir: Directory containing model files
            version_history_path: Path to JSONL version history log
            min_shadow_scans: Minimum shadow test scans before promotion decision
            min_improvement: Minimum accuracy improvement to promote (0-1 scale)
        """
        self.models_dir = Path(models_dir)
        self.version_history_path = Path(version_history_path)
        self.min_shadow_scans = min_shadow_scans
        self.min_improvement = min_improvement

        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.version_history_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self._versions: dict[str, ModelVersion] = {}
        self._shadow_results: list[ShadowTestResult] = []

        # Load existing versions from history
        self._load_versions()

    def _load_versions(self) -> None:
        """Load version history from JSONL file.

        Handles missing/corrupted files gracefully.
        """
        if not self.version_history_path.exists():
            logger.debug(f"Version history not found at {self.version_history_path}")
            return

        try:
            with open(self.version_history_path, "r") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        version_tag = data.get("version_tag")
                        if version_tag:
                            # Reconstruct ModelVersion from JSON
                            version = ModelVersion(
                                version_tag=version_tag,
                                model_path=data.get("model_path", ""),
                                registered_at=data.get("registered_at", ""),
                                accuracy=float(data.get("accuracy", 0.0)),
                                shadow_scans=int(data.get("shadow_scans", 0)),
                                shadow_correct=int(data.get("shadow_correct", 0)),
                                shadow_accuracy=float(data.get("shadow_accuracy", 0.0)),
                                is_production=bool(data.get("is_production", False)),
                                is_candidate=bool(data.get("is_candidate", False)),
                                metadata=data.get("metadata", {}),
                            )
                            self._versions[version_tag] = version
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Corrupted line {line_num} in version history: {e}")
                        continue

            logger.debug(f"Loaded {len(self._versions)} model versions from history")
        except Exception as e:
            logger.error(f"Error loading version history: {e}")

    def _safe_write_jsonl(self, path: Path, data: dict) -> bool:
        """Write a single JSONL record with file locking.

        Args:
            path: Path to JSONL file
            data: Dictionary to append as JSON line

        Returns:
            True if successful, False otherwise
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(data, default=str) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except Exception as e:
            logger.error(f"Error writing to {path}: {e}")
            return False

    def register_version(
        self,
        version_tag: str,
        model_path: str,
        accuracy: float = 0.0,
        metadata: dict = None,
        is_candidate: bool = False,
    ) -> Optional[ModelVersion]:
        """Register a new model version.

        Args:
            version_tag: Unique version identifier (e.g., "v1.0.0")
            model_path: Path to model file or directory
            accuracy: Validation accuracy (0-1 scale)
            metadata: Optional metadata dict (training info, etc.)
            is_candidate: Mark as candidate for shadow testing

        Returns:
            ModelVersion instance, or None if registration failed
        """
        if version_tag in self._versions:
            logger.warning(f"Version {version_tag} already registered")
            return self._versions[version_tag]

        # Validate model path exists
        model_file = Path(model_path)
        if not model_file.exists():
            logger.error(f"Model path does not exist: {model_path}")
            return None

        now = datetime.now().isoformat()
        version = ModelVersion(
            version_tag=version_tag,
            model_path=str(model_path),
            registered_at=now,
            accuracy=float(accuracy),
            is_candidate=is_candidate,
            is_production=False,
            metadata=metadata or {},
        )

        # Store in memory
        self._versions[version_tag] = version

        # Persist to history
        record = {
            **asdict(version),
            "action": "registered",
            "logged_at": now,
        }
        success = self._safe_write_jsonl(self.version_history_path, record)

        if success:
            logger.info(
                f"Registered model version {version_tag} "
                f"(accuracy={accuracy:.4f}, is_candidate={is_candidate})"
            )
            return version
        else:
            # Remove from memory if write failed
            del self._versions[version_tag]
            return None

    def record_shadow_result(
        self,
        pair: str,
        incumbent_direction: str,
        candidate_direction: Optional[str],
        actual_movement: float,
    ) -> bool:
        """Record a single shadow test comparison.

        Handles both true A/B testing (both predictions) and incumbent-only
        shadow testing (candidate_direction = None).

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            incumbent_direction: Production model prediction (LONG/SHORT/HOLD)
            candidate_direction: Candidate model prediction (LONG/SHORT/HOLD),
                or None if candidate unavailable (incumbent-only test)
            actual_movement: Actual price movement after N candles (0=down, 1=up)

        Returns:
            True if recorded successfully
        """
        # Normalize actual_movement to the LONG/SHORT/HOLD label space used by model
        # predictions. Using UP/DOWN here would cause predictions to never match.
        # actual_movement: 1.0 = price rose → LONG was correct
        #                  0.0 = price fell → SHORT was correct
        #                  ~0.5 = no significant move → HOLD
        if actual_movement > 0.55:
            correct_direction = "LONG"
        elif actual_movement < 0.45:
            correct_direction = "SHORT"
        else:
            correct_direction = "HOLD"

        incumbent_correct = incumbent_direction == correct_direction
        candidate_correct = candidate_direction == correct_direction

        result = ShadowTestResult(
            pair=pair,
            incumbent_direction=incumbent_direction,
            candidate_direction=candidate_direction,
            actual_movement=float(actual_movement),
            incumbent_correct=incumbent_correct,
            candidate_correct=candidate_correct,
            timestamp=datetime.now().isoformat(),
        )

        self._shadow_results.append(result)

        # Update candidate version stats if it exists
        # CRITICAL: Atomic update with bounds checking to prevent division by zero
        # and ensure accuracy stays in [0.0, 1.0]
        candidate = self._get_candidate()
        if candidate:
            candidate.shadow_scans += 1
            if candidate_correct:
                candidate.shadow_correct += 1
            # Atomic calculation with bounds check
            candidate.shadow_accuracy = min(1.0, max(0.0, candidate.shadow_correct / max(1, candidate.shadow_scans)))

        # Persist to history
        record = {
            "action": "shadow_test",
            **asdict(result),
            "logged_at": datetime.now().isoformat(),
        }
        success = self._safe_write_jsonl(self.version_history_path, record)

        if success and candidate_correct and not incumbent_correct:
            logger.debug(f"Shadow test (CANDIDATE WIN): {pair} candidate={candidate_direction} actual={correct_direction}")
        elif success and incumbent_correct and not candidate_correct:
            logger.debug(f"Shadow test (INCUMBENT WIN): {pair} incumbent={incumbent_direction} actual={correct_direction}")

        return success

    def score_from_trade_outcome(self, trade_entry: dict) -> bool:
        """Score a closed trade's direction against the stored shadow predictions log.

        Finds the most recent unscored shadow prediction for this pair from
        shadow_predictions.jsonl (written by _run_shadow_tests during each scan).
        Marks it scored and records a shadow result for both incumbent and candidate
        (if a candidate prediction was stored).

        Args:
            trade_entry: A closed trade from trade_journal_rl.json with keys:
                pair, direction, outcome (dict with realized_pl or closed bool)

        Returns:
            True if a shadow prediction was found and scored, False otherwise.
        """
        import json as _json
        from pathlib import Path as _Path

        pair = trade_entry.get("pair")
        direction = trade_entry.get("direction")  # "LONG" or "SHORT"
        outcome = trade_entry.get("outcome", {})

        if not pair or not direction:
            return False

        # Determine correctness from trade outcome
        realized_pl = outcome.get("realized_pl", 0.0)
        trade_won = realized_pl > 0

        # actual_movement: 1.0 = LONG correct, 0.0 = SHORT correct
        if direction == "LONG":
            actual_movement = 1.0 if trade_won else 0.0
        elif direction == "SHORT":
            actual_movement = 0.0 if trade_won else 1.0
        else:
            return False  # Can't score HOLD direction

        # Look up the most recent unscored prediction for this pair
        shadow_log = _Path("trained_data/models/shadow_predictions.jsonl")
        if not shadow_log.exists():
            return False

        try:
            lines = shadow_log.read_text().splitlines()
            records = [_json.loads(l) for l in lines if l.strip()]
        except Exception as e:
            logger.debug(f"score_from_trade_outcome: failed to read shadow log: {e}")
            return False

        # Find the most recent unscored entry for this pair with matching direction
        target_idx = None
        for i in range(len(records) - 1, -1, -1):
            r = records[i]
            if (
                r.get("pair") == pair
                and r.get("incumbent_direction") == direction
                and not r.get("scored")
            ):
                target_idx = i
                break

        if target_idx is None:
            logger.debug(f"score_from_trade_outcome: no unscored prediction for {pair}/{direction}")
            return False

        # Mark as scored
        records[target_idx]["scored"] = True
        records[target_idx]["actual_outcome"] = "win" if trade_won else "loss"
        records[target_idx]["incumbent_correct"] = trade_won  # direction matched

        candidate_direction = records[target_idx].get("candidate_direction")

        # Update shadow log
        try:
            shadow_log.write_text("\n".join(_json.dumps(r) for r in records) + "\n")
        except Exception as e:
            logger.debug(f"score_from_trade_outcome: failed to update shadow log: {e}")

        # Record in ModelManager only if we have a candidate prediction to compare
        if candidate_direction is not None:
            return self.record_shadow_result(
                pair=pair,
                incumbent_direction=direction,
                candidate_direction=candidate_direction,
                actual_movement=actual_movement,
            )

        # Incumbent-only scoring: log event without candidate comparison
        self._safe_write_jsonl(
            self.version_history_path,
            {
                "action": "incumbent_scored",
                "pair": pair,
                "direction": direction,
                "trade_won": trade_won,
                "incumbent_correct": trade_won,
                "candidate_direction": None,
                "logged_at": __import__("datetime").datetime.now().isoformat(),
            },
        )
        return True

    def check_promotion(self) -> Optional[str]:
        """Check if candidate should be promoted to production.

        Returns:
            version_tag if promotion warranted, None otherwise
        """
        candidate = self._get_candidate()
        if not candidate:
            return None

        # Not enough data yet
        if candidate.shadow_scans < self.min_shadow_scans:
            logger.debug(
                f"Candidate has {candidate.shadow_scans}/{self.min_shadow_scans} shadow scans; "
                "not enough data for promotion decision"
            )
            return None

        incumbent = self._get_production()
        if not incumbent:
            logger.warning("No production model found; cannot evaluate promotion")
            return None

        improvement = candidate.shadow_accuracy - incumbent.shadow_accuracy

        if improvement > self.min_improvement:
            logger.info(
                f"✓ PROMOTION CANDIDATE: {candidate.version_tag} "
                f"({candidate.shadow_accuracy:.4f}) > "
                f"{incumbent.version_tag} ({incumbent.shadow_accuracy:.4f}) "
                f"by {improvement:.4f}"
            )
            return candidate.version_tag

        elif improvement < -self.min_improvement:
            logger.warning(
                f"✗ CANDIDATE INFERIOR: {candidate.version_tag} "
                f"({candidate.shadow_accuracy:.4f}) < "
                f"{incumbent.version_tag} ({incumbent.shadow_accuracy:.4f}) "
                f"by {-improvement:.4f}; candidate should be retired"
            )
            return None

        else:
            logger.debug(
                f"Candidate within margin: {candidate.shadow_accuracy:.4f} vs "
                f"incumbent {incumbent.shadow_accuracy:.4f} "
                f"(threshold: {self.min_improvement:.4f})"
            )
            return None

    def check_promotion_with_significance(
        self,
        min_delta: float = 0.05,
        min_samples: int = 50,
        alpha: float = 0.05,
    ) -> dict:
        """Check if candidate should be promoted using statistical significance.

        Uses Fisher exact test (or chi-squared fallback) to ensure the observed
        improvement is statistically significant, not just noise.

        Args:
            min_delta: Minimum accuracy improvement required (0-1)
            min_samples: Minimum shadow predictions needed
            alpha: Significance level (p-value threshold)

        Returns:
            Dict with: should_promote, incumbent_acc, candidate_acc, p_value,
            sample_size, effect_size, reason
        """
        result = {
            "should_promote": False,
            "incumbent_acc": 0.0,
            "candidate_acc": 0.0,
            "p_value": 1.0,
            "sample_size": 0,
            "effect_size": 0.0,
            "reason": "",
        }

        # Gather shadow results where both models made predictions
        paired = [
            r for r in self._shadow_results
            if r.candidate_correct is not None and r.incumbent_correct is not None
        ]
        n = len(paired)
        result["sample_size"] = n

        if n < min_samples:
            result["reason"] = f"Insufficient data: {n}/{min_samples} paired predictions"
            return result

        inc_correct = sum(1 for r in paired if r.incumbent_correct)
        cand_correct = sum(1 for r in paired if r.candidate_correct)

        inc_acc = inc_correct / n
        cand_acc = cand_correct / n
        delta = cand_acc - inc_acc

        result["incumbent_acc"] = round(inc_acc, 4)
        result["candidate_acc"] = round(cand_acc, 4)
        result["effect_size"] = round(delta, 4)

        if delta < min_delta:
            result["reason"] = (
                f"Improvement {delta:.4f} below threshold {min_delta:.4f} "
                f"(candidate={cand_acc:.4f}, incumbent={inc_acc:.4f})"
            )
            return result

        # Statistical test: 2x2 contingency table
        # [[inc_correct, inc_wrong], [cand_correct, cand_wrong]]
        table = [
            [inc_correct, n - inc_correct],
            [cand_correct, n - cand_correct],
        ]

        p_value = 1.0
        try:
            from scipy.stats import fisher_exact
            _, p_value = fisher_exact(table, alternative="less")
        except ImportError:
            # Fallback: chi-squared approximation
            try:
                from scipy.stats import chi2_contingency
                _, p_value, _, _ = chi2_contingency(table, correction=True)
            except ImportError:
                # No scipy — use normal approximation for proportions
                import math
                p_pooled = (inc_correct + cand_correct) / (2 * n)
                if 0 < p_pooled < 1:
                    se = math.sqrt(2 * p_pooled * (1 - p_pooled) / n)
                    if se > 0:
                        z = delta / se
                        # Approximate one-tailed p-value from z-score
                        p_value = 0.5 * math.erfc(z / math.sqrt(2))

        result["p_value"] = round(float(p_value), 6)

        if p_value < alpha:
            result["should_promote"] = True
            result["reason"] = (
                f"PROMOTE: candidate +{delta:.4f} accuracy (p={p_value:.4f} < {alpha}), "
                f"n={n}, candidate={cand_acc:.4f} vs incumbent={inc_acc:.4f}"
            )
            logger.info(f"✓ A/B promotion test PASSED: {result['reason']}")
        else:
            result["reason"] = (
                f"Not significant: delta={delta:.4f} but p={p_value:.4f} >= {alpha}, n={n}"
            )
            logger.info(f"✗ A/B promotion test: {result['reason']}")

        return result

    def auto_promote_if_ready(self) -> bool:
        """Run significance test and auto-promote if criteria met.

        Returns:
            True if promotion happened, False otherwise
        """
        sig_result = self.check_promotion_with_significance()
        if not sig_result["should_promote"]:
            return False

        candidate = self._get_candidate()
        if not candidate:
            return False

        # Log promotion decision
        try:
            from src.scanner.automation.safe_json import safe_jsonl_append
            safe_jsonl_append(
                self.version_history_path,
                {
                    "event": "auto_promotion",
                    "candidate": candidate.version_tag,
                    "p_value": sig_result["p_value"],
                    "effect_size": sig_result["effect_size"],
                    "sample_size": sig_result["sample_size"],
                    "incumbent_acc": sig_result["incumbent_acc"],
                    "candidate_acc": sig_result["candidate_acc"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception:
            pass

        return self.promote(candidate.version_tag)

    def promote(self, version_tag: str) -> bool:
        """Promote a candidate to production.

        Backs up current production model, copies candidate to production path.

        Args:
            version_tag: Version tag to promote

        Returns:
            True on success, False on failure
        """
        if version_tag not in self._versions:
            logger.error(f"Version {version_tag} not found in registry")
            return False

        candidate = self._versions[version_tag]
        candidate_path = Path(candidate.model_path)

        if not candidate_path.exists():
            logger.error(f"Candidate model not found at {candidate_path}")
            return False

        # Find current production model
        incumbent = self._get_production()
        production_path = self.models_dir / "buddy_tf.keras"

        try:
            # 1. Backup current production if it exists
            if incumbent and production_path.exists():
                archive_dir = self.models_dir / "archive"
                archive_dir.mkdir(parents=True, exist_ok=True)

                # Use timestamp-based backup name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = archive_dir / f"buddy_tf_backup_{incumbent.version_tag}_{timestamp}.keras"

                shutil.copy2(production_path, backup_path)
                logger.info(f"Backed up incumbent to {backup_path}")

                # Update incumbent record
                incumbent.is_production = False
                backup_record = {
                    "action": "backup",
                    "version_tag": incumbent.version_tag,
                    "backup_path": str(backup_path),
                    "logged_at": datetime.now().isoformat(),
                }
                self._safe_write_jsonl(self.version_history_path, backup_record)

            # 2. Copy candidate to production path
            if candidate_path.is_file():
                shutil.copy2(candidate_path, production_path)
            elif candidate_path.is_dir():
                if production_path.exists():
                    shutil.rmtree(production_path)
                shutil.copytree(candidate_path, production_path)

            logger.info(f"Copied {candidate_path} to {production_path}")

            # 3. Update version records
            # CRITICAL: Clear is_production on ALL other versions before setting new one
            for v in self._versions.values():
                if v.version_tag != version_tag:
                    v.is_production = False

            candidate.is_production = True
            candidate.is_candidate = False

            promotion_record = {
                "action": "promotion",
                "candidate_version": version_tag,
                "promoted_at": datetime.now().isoformat(),
                "candidate_accuracy": candidate.shadow_accuracy,
                "incumbent_accuracy": incumbent.shadow_accuracy if incumbent else 0.0,
            }
            self._safe_write_jsonl(self.version_history_path, promotion_record)

            logger.info(
                f"✓ PROMOTED {version_tag} to production "
                f"(shadow accuracy: {candidate.shadow_accuracy:.4f})"
            )
            return True

        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            return False

    def rollback(self, version_tag: str) -> bool:
        """Rollback to a previous model version.

        Args:
            version_tag: Version tag to rollback to

        Returns:
            True on success, False on failure
        """
        if version_tag not in self._versions:
            logger.error(f"Version {version_tag} not found in registry")
            return False

        target = self._versions[version_tag]
        target_path = Path(target.model_path)

        if not target_path.exists():
            logger.error(f"Target model not found at {target_path}")
            return False

        production_path = self.models_dir / "buddy_tf.keras"

        try:
            # If target is already at production path, no need to copy
            if target_path.resolve() == production_path.resolve():
                logger.info(f"Target model is already at production path; skipping copy")
                target.is_production = True
                rollback_record = {
                    "action": "rollback",
                    "rollback_to_version": version_tag,
                    "rolled_back_at": datetime.now().isoformat(),
                }
                self._safe_write_jsonl(self.version_history_path, rollback_record)
                return True

            # Backup current production
            incumbent = self._get_production()
            if incumbent and production_path.exists():
                archive_dir = self.models_dir / "archive"
                archive_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = archive_dir / f"buddy_tf_backup_{incumbent.version_tag}_{timestamp}.keras"

                shutil.copy2(production_path, backup_path)
                # CRITICAL: Only update incumbent after file copy succeeds
                incumbent.is_production = False

                backup_record = {
                    "action": "backup_before_rollback",
                    "version_tag": incumbent.version_tag,
                    "backup_path": str(backup_path),
                    "logged_at": datetime.now().isoformat(),
                }
                self._safe_write_jsonl(self.version_history_path, backup_record)

            # CRITICAL: Copy target to production FIRST, then update state only on success
            if target_path.is_file():
                shutil.copy2(target_path, production_path)
            elif target_path.is_dir():
                if production_path.exists():
                    shutil.rmtree(production_path)
                shutil.copytree(target_path, production_path)

            # Only update state after file copy succeeds
            target.is_production = True

            rollback_record = {
                "action": "rollback",
                "rollback_to_version": version_tag,
                "rolled_back_at": datetime.now().isoformat(),
            }
            self._safe_write_jsonl(self.version_history_path, rollback_record)

            logger.info(f"✓ ROLLED BACK to {version_tag}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def get_status(self) -> dict:
        """Get current model status summary.

        Returns:
            Dict with production/candidate versions and metrics
        """
        production = self._get_production()
        candidate = self._get_candidate()

        status = {
            "production": None,
            "candidate": None,
            "total_versions": len(self._versions),
            "shadow_test_progress": None,
        }

        if production:
            status["production"] = {
                "version_tag": production.version_tag,
                "accuracy": production.accuracy,
                "shadow_accuracy": production.shadow_accuracy,
                "shadow_scans": production.shadow_scans,
                "registered_at": production.registered_at,
            }

        if candidate:
            status["candidate"] = {
                "version_tag": candidate.version_tag,
                "accuracy": candidate.accuracy,
                "shadow_accuracy": candidate.shadow_accuracy,
                "shadow_scans": candidate.shadow_scans,
                "registered_at": candidate.registered_at,
            }

            if production:
                improvement = candidate.shadow_accuracy - production.shadow_accuracy
                progress = min(1.0, candidate.shadow_scans / self.min_shadow_scans)
                promotion_ready = (
                    candidate.shadow_scans >= self.min_shadow_scans
                    and improvement > self.min_improvement
                )

                status["shadow_test_progress"] = {
                    "scans_completed": candidate.shadow_scans,
                    "scans_required": self.min_shadow_scans,
                    "progress_pct": round(progress * 100, 1),
                    "improvement_vs_incumbent": round(improvement, 4),
                    "improvement_threshold": self.min_improvement,
                    "promotion_ready": promotion_ready,
                }

        return status

    def _get_production(self) -> Optional[ModelVersion]:
        """Get the currently active production model."""
        for version in self._versions.values():
            if version.is_production:
                return version
        return None

    def _get_candidate(self) -> Optional[ModelVersion]:
        """Get the currently active candidate model."""
        for version in self._versions.values():
            if version.is_candidate:
                return version
        return None
