# train.py
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import optuna

from preprocess import preprocess_data
from utils import setup_logging

logger = setup_logging(log_file="my_app.log")
log_dir = Path.home() / ".trading_bot_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir=str(log_dir))

# ---------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------


class StockDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for stock market data sequences."""

    def __init__(
        self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 60
    ):
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        seq = torch.FloatTensor(self.features[idx])
        target = torch.FloatTensor([self.targets[idx]])
        return seq, target


# ---------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------


class StockPredictor(nn.Module):
    """Enhanced LSTM-based predictor with residual connections."""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        lstm_out_size = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size),
            nn.Tanh(),
            nn.Linear(lstm_out_size, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        return self.fc(attended)


# Additional models like AttentiveLSTM, TransformerPredictor,
# TCNPredictor could be defined similarly.

# ---------------------------------------------------------------------
# Enhanced MLEngine class
# ---------------------------------------------------------------------


class EnhancedMLEngine:
    """
    Enhanced ML engine for trading bot training,
    prediction, and monitoring.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.sequence_length = config.get("sequence_length", 20)
        self.architecture = config.get("architecture", "lstm")
        self.model = None
        self._init_model()
        self.best_loss = float("inf")
        self.patience = config.get("patience", 10)
        self.training_history = {
            "losses": [],
            "val_losses": [],
            "epoch_times": [],
            "timestamps": [],
            "metrics": {
                "r2_scores": [],
                "mse_scores": [],
            },
        }
        self.data_dir = Path(config.get("data_dir", "./trained_data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.data_dir / "models"
        self.model_dir.mkdir(exist_ok=True)
        self.viz_dir = self.data_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        self.pin_memory = config.get("pin_memory", False)
        self.data_file = self.data_dir / "training_history.npz"
        self.load_training_history()
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.scaler = GradScaler()
            self.pin_memory = True
            torch.cuda.empty_cache()
        self.train_losses = []
        self.val_losses = []
        self.epoch_times = []
        self.recent_predictions = []
        self.recent_actuals = []
        self.performance_thresholds = {
            "min_r2_score": 0.6,
            "max_mse": 0.1,
            "performance_window": 100,
        }
        self.is_training = False
        self.should_stop = False
        self.writer = writer

    def _init_model(self) -> None:
        """Initialize the model based on chosen architecture."""
        if self.architecture == "lstm":
            self.model = StockPredictor(
                input_size=self.config.get("input_features", 10),
                hidden_size=self.config.get("hidden_size", 64),
                num_layers=self.config.get("num_layers", 2),
                dropout=self.config.get("dropout", 0.2),
            ).to(self.device)
        # (Other architectures could be added here.)
        lr = self.config.get("learning_rate", 1e-3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        if torch.cuda.device_count() > 1:
            # Optionally wrap model with DistributedDataParallel.
            pass
        self.model = torch.compile(self.model)

    def load_training_history(self) -> None:
        try:
            if self.data_file.exists():
                data = np.load(self.data_file, allow_pickle=True)
                self.training_history = {
                    "data": data["data"].tolist(),
                    "targets": data["targets"].tolist(),
                    "losses": data["losses"].tolist(),
                    "timestamps": data["timestamps"].tolist(),
                }
                num_samples = len(self.training_history["data"])
                logger.info(f"Loaded {num_samples} historical training samples")
        except Exception as e:
            logger.error(f"Error loading training history: {e}")

    def save_training_history(self) -> None:
        try:
            history_data = {
                "data": np.array(self.training_history.get("data", [])),
                "targets": np.array(self.training_history.get("targets", [])),
                "losses": np.array(self.training_history.get("losses", [])),
                "val_losses": np.array(self.training_history.get("val_losses", [])),
                "timestamps": np.array(self.training_history.get("timestamps", [])),
                "metrics": {
                    "r2_scores": np.array(
                        self.training_history["metrics"].get("r2_scores", [])
                    ),
                    "mse_scores": np.array(
                        self.training_history["metrics"].get("mse_scores", [])
                    ),
                },
            }
            np.savez(self.data_file, **history_data)
            logger.info(f"Training history saved to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving training history: {e}")

    def preprocess_data(self, data: pd.DataFrame) -> tuple:
        return preprocess_data(data)

    def create_features(self, df):
        # Implement your feature creation logic here.
        # This is a placeholder implementation.
        return df

    # Ensure a private alias exists.
    _create_features = create_features

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 500,
        batch_size: int = 32,
        validation_split: float = 0.1,
    ) -> None:
        if self.is_training:
            logger.warning("Training is already in progress.")
            return
        self.is_training = True
        total_samples = len(X_train)
        val_size = int(total_samples * validation_split)
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train, y_train = X_train[:-val_size], y_train[:-val_size]

        train_dataset = StockDataset(X_train, y_train, self.sequence_length)
        val_dataset = StockDataset(X_val, y_val, self.sequence_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=mp.cpu_count(),
            prefetch_factor=2,
            pin_memory=self.pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=mp.cpu_count(),
            prefetch_factor=2,
            pin_memory=self.pin_memory,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )
        best_val_loss = float("inf")
        no_improve_epochs = 0
        logger.info(f"Starting training for {epochs} epochs...")
        # Use tensorboard writer for logging
        for epoch in range(epochs):
            if self.should_stop:
                break
            running_loss, batch_count = 0.0, 0
            for i, (X_batch, y_batch) in enumerate(train_loader):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                if torch.cuda.is_available():
                    from torch.cuda.amp import autocast

                    with autocast():
                        outputs = self.model(X_batch)
                        loss = self._custom_loss(outputs.squeeze(), y_batch.squeeze())
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(X_batch)
                    loss = self._custom_loss(outputs.squeeze(), y_batch.squeeze())
                    loss.backward()
                    self.optimizer.step()
                running_loss += loss.item()
                batch_count += 1
            train_loss = running_loss / batch_count
            val_loss = self._validate(val_loader)
            scheduler.step()
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.training_history["losses"].append(train_loss)
            self.training_history["val_losses"].append(val_loss)
            timestamp = datetime.now().isoformat()
            self.training_history["timestamps"].append(timestamp)
        self.save_training_history()
        final_model_path = self.model_dir / "model.pth"
        self.save_model(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")
        self.is_training = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self._custom_loss(outputs.squeeze(), y_batch.squeeze())
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def _custom_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        mse_loss = self.criterion(predictions, targets)
        pred_direction = torch.sign(predictions[1:] - targets[:-1])
        target_direction = torch.sign(targets[1:] - targets[:-1])
        direction_loss = torch.mean(torch.abs(pred_direction - target_direction))
        alpha = self.config.get("loss_alpha", 0.1)
        return mse_loss + alpha * direction_loss

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        try:
            ckpt_dir = self.model_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": val_loss,
                "config": self.config,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "timestamp": datetime.now().isoformat(),
            }
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, ckpt_path)
            logger.info(
                f"Checkpoint saved at epoch {epoch} with val loss {val_loss:.4f}"
            )
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def save_model(self, path: str) -> None:
        try:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state": self.scaler.__getstate__(),
                "config": self.config,
                "epoch": len(self.train_losses),
                "loss": (self.train_losses[-1] if self.train_losses else float("inf")),
                "training_info": {
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses,
                    "timestamp": datetime.now().isoformat(),
                },
            }
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_checkpoint(self, path: str) -> None:
        # Load checkpoint from file and restore
        # model and optimizer states
        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Optionally restore additional states here if needed
        print("Checkpoint loaded successfully.")


def run_hyperparameter_tuning(data: pd.DataFrame, n_trials: int = 20) -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization."""

    def objective(trial: optuna.trial.Trial) -> float:
        from train import auto_configure

        config = auto_configure()
        config.update(
            {
                "hidden_size": trial.suggest_int("hidden_size", 32, 256),
                "num_layers": trial.suggest_int("num_layers", 1, 4),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-2),
                "architecture": trial.suggest_categorical(
                    "architecture", ["lstm", "attention_lstm"]
                ),
            }
        )

        engine = EnhancedMLEngine(config)
        X, y = engine.preprocess_data(data)
        if X is None or len(X) < 50:
            raise optuna.TrialPruned()
        split_idx = int(0.8 * len(X))
        engine.train(X[:split_idx], y[:split_idx], epochs=10)
        val_dataset = StockDataset(X[split_idx:], y[split_idx:], engine.sequence_length)
        val_loss = engine._validate(DataLoader(val_dataset))
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def auto_configure() -> Dict[str, Any]:
    """
    Automatically configure hyperparameters based on system resources.
    """
    import psutil

    config = {}
    cpu_count = mp.cpu_count()
    config["num_workers"] = max(1, cpu_count - 1)
    total_mem = psutil.virtual_memory().total / (1024**3)
    logger.info(f"Total system memory: {total_mem:.2f} GB")
    config["system_memory"] = total_mem
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_props = torch.cuda.get_device_properties(device)
        total_gpu_mem = gpu_props.total_memory / (1024**3)
        logger.info(f"GPU detected with total memory: {total_gpu_mem:.2f} GB")
        if total_gpu_mem >= 12:
            config["batch_size"] = 128
        elif total_gpu_mem >= 8:
            config["batch_size"] = 64
        else:
            config["batch_size"] = 32
        config["pin_memory"] = True
        config["use_mixed_precision"] = True
        config["use_compile"] = hasattr(torch, "compile")
    else:
        config["batch_size"] = 16
        config["pin_memory"] = False
        config["use_mixed_precision"] = False
        config["use_compile"] = False
    config.update(
        {
            "learning_rate": 1e-3,
            "input_features": 7,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "sequence_length": 60,
            "patience": 10,
            "min_delta": 0.001,
            "epochs": 100,
        }
    )
    return config
