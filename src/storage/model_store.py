import pickle
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)


class ModelStore:
    def __init__(self, base_path: str = None):
        self.base_path = Path(
            base_path or "/Users/davidcertan/Documents/ml_engine/models"
        )
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.base_path / "index.json"
        self._load_index()

    def _load_index(self):
        """Load model index from disk"""
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                self.index = json.load(f)
        else:
            self.index = {}
            self._save_index()

    def _save_index(self):
        """Save model index to disk"""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)

    def save_model(
        self, model_id: str, model: Any, metadata: Dict[str, Any] = None
    ) -> bool:
        """Save model with metadata and versioning"""
        try:
            model_dir = self.base_path / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            model_path = model_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Compute checksum
            checksum = self._compute_checksum(model_path)

            # Prepare metadata
            full_metadata = {
                "model_id": model_id,
                "saved_at": datetime.utcnow().isoformat(),
                "checksum": checksum,
                "size_bytes": model_path.stat().st_size,
                **(metadata or {}),
            }

            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(full_metadata, f, indent=2)

            # Update index
            self.index[model_id] = {"path": str(model_dir), "metadata": full_metadata}
            self._save_index()

            logger.info(f"Model {model_id} saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {str(e)}")
            return False

    def load_model(
        self, model_id: str
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """Load model and its metadata"""
        try:
            if model_id not in self.index:
                logger.warning(f"Model {model_id} not found in index")
                return None, None

            model_dir = Path(self.index[model_id]["path"])
            model_path = model_dir / "model.pkl"
            metadata_path = model_dir / "metadata.json"

            # Verify checksum
            stored_checksum = self.index[model_id]["metadata"].get("checksum")
            current_checksum = self._compute_checksum(model_path)

            if stored_checksum and stored_checksum != current_checksum:
                logger.error(f"Checksum mismatch for model {model_id}")
                return None, None

            # Load model
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            logger.info(f"Model {model_id} loaded successfully")
            return model, metadata

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            return None, None

    def delete_model(self, model_id: str) -> bool:
        """Delete a model and its metadata"""
        try:
            if model_id not in self.index:
                return False

            model_dir = Path(self.index[model_id]["path"])
            if model_dir.exists():
                shutil.rmtree(model_dir)

            del self.index[model_id]
            self._save_index()

            logger.info(f"Model {model_id} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {str(e)}")
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        models = []
        for model_id, info in self.index.items():
            models.append(
                {
                    "model_id": model_id,
                    "saved_at": info["metadata"].get("saved_at"),
                    "size_bytes": info["metadata"].get("size_bytes"),
                    "config": info["metadata"].get("config", {}),
                }
            )
        return models

    def count_models(self) -> int:
        """Count total number of models"""
        return len(self.index)

    def create_version(self, model_id: str, new_version: str) -> bool:
        """Create a versioned copy of a model"""
        try:
            model, metadata = self.load_model(model_id)
            if model is None:
                return False

            versioned_id = f"{model_id}_v{new_version}"
            metadata["version"] = new_version
            metadata["parent_model"] = model_id

            return self.save_model(versioned_id, model, metadata)

        except Exception as e:
            logger.error(f"Failed to create version: {str(e)}")
            return False

    def export_model(self, model_id: str, export_path: str) -> bool:
        """Export model to external location"""
        try:
            if model_id not in self.index:
                return False

            model_dir = Path(self.index[model_id]["path"])
            export_dir = Path(export_path)

            shutil.copytree(model_dir, export_dir, dirs_exist_ok=True)
            logger.info(f"Model {model_id} exported to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export model: {str(e)}")
            return False

    @staticmethod
    def _compute_checksum(file_path: Path) -> str:
        """Compute SHA256 checksum of file"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
