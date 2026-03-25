"""
Tests for ModelManager — Model A/B testing and version control.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scanner.automation.model_manager import ModelManager, ModelVersion, ShadowTestResult


@pytest.fixture
def temp_models_dir():
    """Create temporary models directory with dummy model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy model files
        (models_dir / "buddy_tf.keras").write_text("dummy production model")
        (models_dir / "buddy_tf_candidate.keras").write_text("dummy candidate model")

        yield models_dir


@pytest.fixture
def model_manager(temp_models_dir):
    """Create ModelManager with temporary directory."""
    history_path = temp_models_dir / "version_history.jsonl"
    return ModelManager(
        models_dir=str(temp_models_dir),
        version_history_path=str(history_path),
        min_shadow_scans=10,
        min_improvement=0.05,
    )


class TestModelVersion:
    """Tests for ModelVersion dataclass."""

    def test_model_version_creation(self):
        """Test creating a ModelVersion."""
        version = ModelVersion(
            version_tag="v1.0.0",
            model_path="/path/to/model.keras",
            registered_at="2026-01-01T00:00:00",
            accuracy=0.85,
            is_production=True,
        )

        assert version.version_tag == "v1.0.0"
        assert version.accuracy == 0.85
        assert version.is_production is True
        assert version.shadow_scans == 0
        assert version.shadow_accuracy == 0.0

    def test_model_version_with_metadata(self):
        """Test ModelVersion with custom metadata."""
        version = ModelVersion(
            version_tag="v1.1.0",
            model_path="/path/to/model.keras",
            registered_at="2026-01-02T00:00:00",
            metadata={"trainer": "alice", "epoch": 10},
        )

        assert version.metadata["trainer"] == "alice"
        assert version.metadata["epoch"] == 10


class TestModelManagerRegistration:
    """Tests for version registration."""

    def test_register_version(self, model_manager, temp_models_dir):
        """Test registering a new version."""
        version = model_manager.register_version(
            version_tag="v1.0.0",
            model_path=str(temp_models_dir / "buddy_tf.keras"),
            accuracy=0.80,
            is_candidate=False,
        )

        assert version is not None
        assert version.version_tag == "v1.0.0"
        assert version.accuracy == 0.80
        assert version.is_production is False

    def test_register_candidate(self, model_manager, temp_models_dir):
        """Test registering a candidate model."""
        version = model_manager.register_version(
            version_tag="v1.1.0",
            model_path=str(temp_models_dir / "buddy_tf_candidate.keras"),
            accuracy=0.82,
            is_candidate=True,
        )

        assert version is not None
        assert version.is_candidate is True

    def test_register_duplicate_fails_gracefully(self, model_manager, temp_models_dir):
        """Test that registering same version twice returns cached version."""
        version1 = model_manager.register_version(
            version_tag="v1.0.0",
            model_path=str(temp_models_dir / "buddy_tf.keras"),
            accuracy=0.80,
        )

        version2 = model_manager.register_version(
            version_tag="v1.0.0",
            model_path=str(temp_models_dir / "buddy_tf.keras"),
            accuracy=0.80,
        )

        assert version1 is version2

    def test_register_nonexistent_model_fails(self, model_manager):
        """Test that registering nonexistent model path fails."""
        version = model_manager.register_version(
            version_tag="v1.0.0",
            model_path="/nonexistent/model.keras",
            accuracy=0.80,
        )

        assert version is None

    def test_version_persisted_to_jsonl(self, model_manager, temp_models_dir):
        """Test that registered version is persisted to JSONL."""
        model_manager.register_version(
            version_tag="v1.0.0",
            model_path=str(temp_models_dir / "buddy_tf.keras"),
            accuracy=0.80,
        )

        # Read JSONL
        history_path = model_manager.version_history_path
        assert history_path.exists()

        lines = history_path.read_text().strip().split("\n")
        assert len(lines) > 0

        record = json.loads(lines[0])
        assert record["version_tag"] == "v1.0.0"
        assert record["action"] == "registered"


class TestShadowTesting:
    """Tests for shadow test recording."""

    def test_record_shadow_result(self, model_manager, temp_models_dir):
        """Test recording a shadow test result."""
        # Register versions first
        model_manager.register_version(
            "v1.0.0",
            str(temp_models_dir / "buddy_tf.keras"),
            is_candidate=False,
        )
        model_manager.register_version(
            "v1.1.0",
            str(temp_models_dir / "buddy_tf_candidate.keras"),
            is_candidate=True,
        )

        # Record shadow result
        success = model_manager.record_shadow_result(
            pair="EUR_USD",
            incumbent_direction="LONG",
            candidate_direction="LONG",
            actual_movement=0.8,  # Up
        )

        assert success is True

        # Check candidate stats updated
        candidate = model_manager._get_candidate()
        assert candidate.shadow_scans == 1
        assert candidate.shadow_correct == 1
        assert candidate.shadow_accuracy == 1.0

    def test_shadow_results_accuracy_calculation(self, model_manager, temp_models_dir):
        """Test that shadow accuracy is correctly calculated."""
        model_manager.register_version(
            "v1.0.0",
            str(temp_models_dir / "buddy_tf.keras"),
            is_candidate=False,
        )
        model_manager.register_version(
            "v1.1.0",
            str(temp_models_dir / "buddy_tf_candidate.keras"),
            is_candidate=True,
        )

        # Record 4 results: 3 correct, 1 incorrect
        model_manager.record_shadow_result("EUR_USD", "LONG", "LONG", 0.8)  # UP, both correct
        model_manager.record_shadow_result("GBP_USD", "LONG", "LONG", 0.3)  # DOWN, both wrong
        model_manager.record_shadow_result("USD_JPY", "SHORT", "SHORT", 0.2)  # DOWN, both correct
        model_manager.record_shadow_result("AUD_USD", "LONG", "SHORT", 0.5)  # HOLD, neither correct

        candidate = model_manager._get_candidate()
        assert candidate.shadow_scans == 4
        # Count: scan 1 (correct), scan 3 (correct) = 2 correct
        assert candidate.shadow_correct == 2
        assert candidate.shadow_accuracy == 0.5


class TestPromotion:
    """Tests for model promotion logic."""

    def test_check_promotion_not_ready(self, model_manager, temp_models_dir):
        """Test that promotion is blocked until min scans reached."""
        model_manager.register_version(
            "v1.0.0",
            str(temp_models_dir / "buddy_tf.keras"),
            accuracy=0.80,
            is_candidate=False,
        )
        model_manager.register_version(
            "v1.1.0",
            str(temp_models_dir / "buddy_tf_candidate.keras"),
            accuracy=0.85,
            is_candidate=True,
        )

        # Only 1 shadow result (need 10)
        model_manager.record_shadow_result("EUR_USD", "LONG", "LONG", 0.8)

        promotion = model_manager.check_promotion()
        assert promotion is None  # Not ready yet

    def test_check_promotion_insufficient_improvement(self, model_manager, temp_models_dir):
        """Test that promotion is blocked if improvement is below threshold."""
        model_manager.register_version(
            "v1.0.0",
            str(temp_models_dir / "buddy_tf.keras"),
            accuracy=0.80,
            is_candidate=False,
        )
        candidate = model_manager.register_version(
            "v1.1.0",
            str(temp_models_dir / "buddy_tf_candidate.keras"),
            accuracy=0.81,
            is_candidate=True,
        )

        # Set production accuracy via shadow testing
        incumbent = model_manager._get_production()
        incumbent.shadow_scans = 10
        incumbent.shadow_correct = 8
        incumbent.shadow_accuracy = 0.80

        # Set candidate to only 3% improvement (need 5%)
        candidate.shadow_scans = 10
        candidate.shadow_correct = 8  # 80% accuracy (only +0.01)
        candidate.shadow_accuracy = 0.81

        promotion = model_manager.check_promotion()
        assert promotion is None

    def test_check_promotion_ready(self, model_manager, temp_models_dir):
        """Test that promotion is ready when criteria met."""
        model_manager.register_version(
            "v1.0.0",
            str(temp_models_dir / "buddy_tf.keras"),
            accuracy=0.80,
            is_candidate=False,
        )
        candidate = model_manager.register_version(
            "v1.1.0",
            str(temp_models_dir / "buddy_tf_candidate.keras"),
            accuracy=0.88,
            is_candidate=True,
        )

        # Set production accuracy
        incumbent = model_manager._get_production()
        incumbent.shadow_scans = 10
        incumbent.shadow_correct = 8
        incumbent.shadow_accuracy = 0.80

        # Set candidate to 7% improvement (meets 5% threshold)
        candidate.shadow_scans = 10
        candidate.shadow_correct = 9
        candidate.shadow_accuracy = 0.87  # 7% better than incumbent

        promotion = model_manager.check_promotion()
        assert promotion == "v1.1.0"

    def test_promote_creates_backup(self, model_manager, temp_models_dir):
        """Test that promotion creates backup of incumbent."""
        # Register and set as production
        model_manager.register_version(
            "v1.0.0",
            str(temp_models_dir / "buddy_tf.keras"),
            is_candidate=False,
        )
        incumbent = model_manager._get_production()
        incumbent.is_production = True

        # Register candidate
        model_manager.register_version(
            "v1.1.0",
            str(temp_models_dir / "buddy_tf_candidate.keras"),
            is_candidate=True,
        )
        candidate = model_manager._get_candidate()
        candidate.shadow_scans = 10
        candidate.shadow_correct = 10
        candidate.shadow_accuracy = 0.90

        # Promote
        success = model_manager.promote("v1.1.0")
        assert success is True

        # Check backup was created
        archive_dir = temp_models_dir / "archive"
        assert archive_dir.exists()
        backups = list(archive_dir.glob("buddy_tf_backup_v1.0.0_*"))
        assert len(backups) == 1

    def test_promote_no_production_fails(self, model_manager, temp_models_dir):
        """Test that promotion fails if model path invalid."""
        model_manager.register_version(
            "v1.0.0",
            str(temp_models_dir / "nonexistent.keras"),
            is_candidate=False,
        )

        # This will fail during promotion
        success = model_manager.promote("v1.0.0")
        assert success is False


class TestRollback:
    """Tests for model rollback."""

    def test_rollback_to_previous_version(self, model_manager, temp_models_dir):
        """Test rolling back to a previous version."""
        model_manager.register_version(
            "v1.0.0",
            str(temp_models_dir / "buddy_tf.keras"),
            is_candidate=False,
        )
        v1_0 = model_manager._get_production()
        v1_0.is_production = True

        # Promote to v1.1.0
        model_manager.register_version(
            "v1.1.0",
            str(temp_models_dir / "buddy_tf_candidate.keras"),
            is_candidate=False,
        )
        v1_1 = model_manager._versions["v1.1.0"]
        v1_1.is_production = True
        v1_0.is_production = False

        # Rollback to v1.0.0
        success = model_manager.rollback("v1.0.0")
        assert success is True

        # Check production is now v1.0.0
        current = model_manager._get_production()
        assert current.version_tag == "v1.0.0"


class TestStatus:
    """Tests for status summary."""

    def test_get_status_with_production_and_candidate(self, model_manager, temp_models_dir):
        """Test status summary with both production and candidate."""
        model_manager.register_version(
            "v1.0.0",
            str(temp_models_dir / "buddy_tf.keras"),
            accuracy=0.80,
            is_candidate=False,
        )
        incumbent = model_manager._get_production()
        incumbent.is_production = True
        incumbent.shadow_accuracy = 0.80
        incumbent.shadow_scans = 50

        model_manager.register_version(
            "v1.1.0",
            str(temp_models_dir / "buddy_tf_candidate.keras"),
            accuracy=0.85,
            is_candidate=True,
        )
        candidate = model_manager._get_candidate()
        candidate.shadow_accuracy = 0.83
        candidate.shadow_scans = 30

        status = model_manager.get_status()

        assert status["production"]["version_tag"] == "v1.0.0"
        assert status["candidate"]["version_tag"] == "v1.1.0"
        assert status["shadow_test_progress"]["improvement_vs_incumbent"] == pytest.approx(0.03)
        assert status["shadow_test_progress"]["progress_pct"] == pytest.approx(60.0)


class TestFileOperations:
    """Tests for file I/O and locking."""

    def test_load_versions_from_jsonl(self, temp_models_dir):
        """Test loading versions from existing JSONL."""
        history_path = temp_models_dir / "version_history.jsonl"

        # Write test data
        records = [
            {"version_tag": "v1.0.0", "registered_at": "2026-01-01", "action": "registered"},
            {"version_tag": "v1.1.0", "registered_at": "2026-01-02", "action": "registered"},
        ]
        for record in records:
            history_path.write_text(history_path.read_text() + json.dumps(record) + "\n", "a")

        # Load manager
        mm = ModelManager(
            models_dir=str(temp_models_dir),
            version_history_path=str(history_path),
        )

        assert len(mm._versions) == 2
        assert "v1.0.0" in mm._versions
        assert "v1.1.0" in mm._versions

    def test_safe_write_jsonl_creates_parent_dirs(self, temp_models_dir):
        """Test that safe_write_jsonl creates parent directories."""
        mm = ModelManager(models_dir=str(temp_models_dir))

        nested_path = temp_models_dir / "level1" / "level2" / "test.jsonl"
        success = mm._safe_write_jsonl(nested_path, {"test": "data"})

        assert success is True
        assert nested_path.exists()
        assert "test" in nested_path.read_text()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
