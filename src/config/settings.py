from pydantic import BaseSettings, Field
from typing import Optional, List
from pathlib import Path
import os


class Settings(BaseSettings):
    # Application
    app_name: str = "ML Engine"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", env="ENV")
    debug: bool = Field(default=True, env="DEBUG")

    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")

    # Paths
    base_path: Path = Field(default=Path("/Users/davidcertan/Documents/ml_engine"))
    models_path: Path = Field(default=None)
    data_path: Path = Field(default=None)
    logs_path: Path = Field(default=None)

    # Model defaults
    default_model_type: str = "linear"
    default_epochs: int = 10
    default_batch_size: int = 32
    default_learning_rate: float = 0.01

    # Inference
    inference_batch_size: int = 32
    inference_workers: int = 4
    model_cache_size: int = 10

    # Monitoring
    enable_metrics: bool = True
    metrics_retention_hours: int = 24
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Storage
    max_model_size_mb: int = 500
    model_retention_days: int = 30
    enable_versioning: bool = True

    # Security
    enable_auth: bool = False
    api_key: Optional[str] = Field(default=None, env="API_KEY")

    # Database (for future use)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set derived paths
        if self.models_path is None:
            self.models_path = self.base_path / "models"
        if self.data_path is None:
            self.data_path = self.base_path / "data"
        if self.logs_path is None:
            self.logs_path = self.base_path / "logs"

        # Create directories
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

    def get_model_config(self) -> dict:
        """Get default model configuration"""
        return {
            "model_type": self.default_model_type,
            "epochs": self.default_epochs,
            "batch_size": self.default_batch_size,
            "learning_rate": self.default_learning_rate,
        }

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"


# Global settings instance
settings = Settings()
