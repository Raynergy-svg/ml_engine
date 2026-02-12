"""
Tests for model registry and versioning system.

Tests verify that:
- Model metadata can be created and saved
- Versions can be listed and compared
- Best models can be retrieved by metric
- Model promotion workflow works
- Git commit tracking works
- Dataset hashing works
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.utils.model_registry import (
    ModelMetadata,
    ModelRegistry,
    create_model_metadata_from_training,
)


class TestModelMetadata:
    """Test ModelMetadata class."""
    
    def test_create_metadata(self):
        """Test creating basic metadata."""
        metadata = ModelMetadata(
            model_name="test_model",
            version="1.0.0",
            hyperparameters={"lr": 0.001},
            metrics={"val_loss": 0.45},
        )
        
        assert metadata.model_name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.hyperparameters == {"lr": 0.001}
        assert metadata.metrics == {"val_loss": 0.45}
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = ModelMetadata(
            model_name="test",
            version="1.0.0",
        )
        
        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data["model_name"] == "test"
        assert data["version"] == "1.0.0"
    
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "model_name": "test",
            "version": "1.0.0",
            "timestamp": "2024-01-01T00:00:00",
            "hyperparameters": {},
            "metrics": {},
            "dataset_hash": None,
            "git_commit": None,
            "environment": {},
            "description": "",
            "tags": {},
            "parent_version": None,
        }
        
        metadata = ModelMetadata.from_dict(data)
        assert metadata.model_name == "test"
        assert metadata.version == "1.0.0"
    
    def test_metadata_json_serialization(self):
        """Test JSON serialization."""
        metadata = ModelMetadata(
            model_name="test",
            version="1.0.0",
            hyperparameters={"lr": 0.001},
            metrics={"loss": 0.5},
        )
        
        # To JSON
        json_str = metadata.to_json()
        assert isinstance(json_str, str)
        
        # From JSON
        metadata2 = ModelMetadata.from_json(json_str)
        assert metadata2.model_name == metadata.model_name
        assert metadata2.version == metadata.version
        assert metadata2.hyperparameters == metadata.hyperparameters


class TestModelRegistry:
    """Test ModelRegistry class."""
    
    def test_registry_creation(self):
        """Test creating a registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_path=tmpdir)
            assert registry.registry_path == Path(tmpdir)
            assert registry.metadata_dir.exists()
    
    def test_register_model(self):
        """Test registering a model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_path=tmpdir)
            
            # Create dummy model file
            model_path = Path(tmpdir) / "model.keras"
            model_path.write_text("dummy model")
            
            metadata = ModelMetadata(
                model_name="test_model",
                version="1.0.0",
                hyperparameters={"lr": 0.001},
                metrics={"val_loss": 0.45},
            )
            
            registry.register_model(metadata, str(model_path))
            
            # Check metadata was saved
            saved_metadata = registry.get_metadata("test_model", "1.0.0")
            assert saved_metadata is not None
            assert saved_metadata.model_name == "test_model"
            assert saved_metadata.version == "1.0.0"
    
    def test_list_versions(self):
        """Test listing model versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_path=tmpdir)
            model_path = Path(tmpdir) / "model.keras"
            model_path.write_text("dummy")
            
            # Register multiple versions
            for version in ["1.0.0", "1.1.0", "2.0.0"]:
                metadata = ModelMetadata(
                    model_name="test_model",
                    version=version,
                    metrics={"val_loss": 0.5},
                )
                registry.register_model(metadata, str(model_path))
            
            # List versions
            versions = registry.list_versions("test_model")
            assert len(versions) == 3
            
            # Should be sorted by timestamp (newest first)
            version_strings = [v.version for v in versions]
            assert "2.0.0" in version_strings
            assert "1.1.0" in version_strings
            assert "1.0.0" in version_strings
    
    def test_get_best_model_minimize(self):
        """Test getting best model (minimize metric)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_path=tmpdir)
            model_path = Path(tmpdir) / "model.keras"
            model_path.write_text("dummy")
            
            # Register models with different losses
            versions_data = [
                ("1.0.0", 0.5),
                ("1.1.0", 0.3),  # Best
                ("1.2.0", 0.6),
            ]
            
            for version, loss in versions_data:
                metadata = ModelMetadata(
                    model_name="test_model",
                    version=version,
                    metrics={"val_loss": loss},
                )
                registry.register_model(metadata, str(model_path))
            
            # Get best model
            best = registry.get_best_model("test_model", "val_loss")
            assert best is not None
            assert best.version == "1.1.0"
            assert best.metrics["val_loss"] == 0.3
    
    def test_get_best_model_maximize(self):
        """Test getting best model (maximize metric)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_path=tmpdir)
            model_path = Path(tmpdir) / "model.keras"
            model_path.write_text("dummy")
            
            # Register models with different accuracies
            versions_data = [
                ("1.0.0", 0.75),
                ("1.1.0", 0.85),  # Best
                ("1.2.0", 0.70),
            ]
            
            for version, acc in versions_data:
                metadata = ModelMetadata(
                    model_name="test_model",
                    version=version,
                    metrics={"accuracy": acc},
                )
                registry.register_model(metadata, str(model_path))
            
            # Get best model
            best = registry.get_best_model("test_model", "accuracy", higher_is_better=True)
            assert best is not None
            assert best.version == "1.1.0"
            assert best.metrics["accuracy"] == 0.85
    
    def test_compare_versions(self):
        """Test comparing two model versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_path=tmpdir)
            model_path = Path(tmpdir) / "model.keras"
            model_path.write_text("dummy")
            
            # Register two versions
            metadata1 = ModelMetadata(
                model_name="test_model",
                version="1.0.0",
                hyperparameters={"lr": 0.001, "dropout": 0.2},
                metrics={"val_loss": 0.5, "accuracy": 0.75},
            )
            registry.register_model(metadata1, str(model_path))
            
            metadata2 = ModelMetadata(
                model_name="test_model",
                version="2.0.0",
                hyperparameters={"lr": 0.0005, "dropout": 0.3},
                metrics={"val_loss": 0.4, "accuracy": 0.80},
            )
            registry.register_model(metadata2, str(model_path))
            
            # Compare
            comparison = registry.compare_versions("test_model", "1.0.0", "2.0.0")
            
            assert "hyperparameter_changes" in comparison
            assert "metric_changes" in comparison
            
            # Check hyperparameter changes
            assert "lr" in comparison["hyperparameter_changes"]
            assert comparison["hyperparameter_changes"]["lr"]["v1"] == 0.001
            assert comparison["hyperparameter_changes"]["lr"]["v2"] == 0.0005
            
            # Check metric changes
            assert "val_loss" in comparison["metric_changes"]
            assert comparison["metric_changes"]["val_loss"]["diff"] < 0  # Improved
    
    def test_promote_model(self):
        """Test model promotion workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_path=tmpdir)
            model_path = Path(tmpdir) / "model.keras"
            model_path.write_text("dummy")
            
            # Register model in dev stage
            metadata = ModelMetadata(
                model_name="test_model",
                version="1.0.0",
                metrics={"val_loss": 0.4},
                tags={"stage": "dev"},
            )
            registry.register_model(metadata, str(model_path), stage="dev")
            
            # Promote to staging
            registry.promote_model("test_model", "1.0.0", "dev", "staging")
            
            # Check updated metadata
            updated = registry.get_metadata("test_model", "1.0.0")
            assert updated.tags["stage"] == "staging"
            assert updated.tags["promoted_from"] == "dev"
            assert "promoted_at" in updated.tags
    
    def test_git_commit_tracking(self):
        """Test git commit hash tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock git command
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="abc123def456\n",
                    returncode=0,
                )
                
                registry = ModelRegistry(registry_path=tmpdir)
                commit = registry._get_git_commit()
                
                assert commit == "abc123def456"
    
    def test_dataset_hashing(self):
        """Test dataset hash computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_path=tmpdir)
            
            # Create dummy dataset
            dataset_path = Path(tmpdir) / "data.csv"
            dataset_path.write_text("a,b,c\n1,2,3\n")
            
            # Compute hash
            hash1 = registry._compute_dataset_hash(str(dataset_path))
            assert hash1 is not None
            assert len(hash1) == 64  # SHA-256 hex digest
            
            # Same file should give same hash
            hash2 = registry._compute_dataset_hash(str(dataset_path))
            assert hash1 == hash2
            
            # Different content should give different hash
            dataset_path.write_text("x,y,z\n4,5,6\n")
            hash3 = registry._compute_dataset_hash(str(dataset_path))
            assert hash3 != hash1


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_create_metadata_from_training(self):
        """Test creating metadata from training config."""
        # Mock trainer config
        class MockConfig:
            def __init__(self):
                self.d_model = 32
                self.num_heads = 4
                self.learning_rate = 0.001
        
        config = MockConfig()
        metrics = {"val_loss": 0.45, "accuracy": 0.78}
        
        metadata = create_model_metadata_from_training(
            model_name="transformer",
            version="1.0.0",
            trainer_config=config,
            metrics=metrics,
            description="Test model",
        )
        
        assert metadata.model_name == "transformer"
        assert metadata.version == "1.0.0"
        assert metadata.hyperparameters["d_model"] == 32
        assert metadata.hyperparameters["num_heads"] == 4
        assert metadata.metrics == metrics
        assert metadata.description == "Test model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
