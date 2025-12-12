"""
Enhanced training script with comprehensive features.
Demonstrates best practices for ML model training.
"""

import os
import argparse
import copy
import contextlib
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import our enhanced modules
from data_loader import MarketDataLoader
from models_enhanced import (
    StockPredictor,
    AttentiveLSTM,
    GRUPredictor,
    TransformerPredictor,
)
from evaluation import ModelEvaluator, Backtester, generate_simple_strategy_signals
from utils import setup_logging, load_config

# Configure logging
logger = setup_logging(log_file="training.log")


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for regression tasks.
    Adds noise to targets to prevent overfitting and improve generalization.
    """
    
    def __init__(self, base_loss=nn.MSELoss(), smoothing: float = 0.1):
        """
        Args:
            base_loss: Base loss function (MSELoss, HuberLoss, etc.)
            smoothing: Amount of label smoothing (0 = no smoothing, higher = more smoothing)
        """
        super(LabelSmoothingLoss, self).__init__()
        self.base_loss = base_loss
        self.smoothing = smoothing
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Apply label smoothing by adding small random noise to targets.
        """
        if self.smoothing > 0 and self.training:
            # Add small Gaussian noise to targets for label smoothing
            # Use robust scaling factor to avoid issues with small std
            reduce_dims = tuple(range(targets.dim()))
            target_scale = torch.std(targets, dim=reduce_dims, unbiased=False) + 1e-8
            noise = torch.randn_like(targets) * self.smoothing * target_scale
            smoothed_targets = targets + noise
            return self.base_loss(predictions, smoothed_targets)
        else:
            return self.base_loss(predictions, targets)


class EnhancedTrainer:
    """Enhanced trainer with modern best practices."""

    def __init__(self, config: dict):
        """Initialize trainer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize components
        self.data_loader = MarketDataLoader(config)
        self.evaluator = ModelEvaluator(config)

        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = (
            torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None
        )

        # Create directories
        self.create_directories()

    def create_directories(self):
        """Create necessary directories."""
        dirs = [
            "trained_data/models",
            "trained_data/logs",
            "trained_data/visualizations",
            "trained_data/checkpoints",
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self, checkpoint_path: str, input_size: int):
        """
        Load checkpoint to resume training.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file (.pth)
            input_size (int): Number of input features for rebuilding the model
            
        Returns:
            tuple: (start_epoch, best_val_loss) where:
                - start_epoch (int): The epoch number to resume from
                - best_val_loss (float): The best validation loss from checkpoint
                
        Raises:
            FileNotFoundError: If checkpoint file does not exist at the specified path
            RuntimeError: If checkpoint is corrupted or incompatible with current config
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        
        # Build model with same architecture
        model_type = self.config.get("architecture", "attention_lstm")
        self.model = self.build_model(model_type, input_size)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Setup optimizer
        learning_rate = self.config.get("learning_rate", 0.001)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )
        
        # Load scheduler state if available
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Return checkpoint info
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        
        logger.info(f"Resuming from epoch {start_epoch} with val_loss {best_val_loss:.6f}")
        
        return start_epoch, best_val_loss

    def build_model(self, model_type: str, input_size: int) -> nn.Module:
        """Build model based on type."""
        model_config = self.config.get("model", {})
        hidden_size = model_config.get("hidden_size", 128)
        num_layers = model_config.get("num_layers", 3)
        dropout = model_config.get("dropout", 0.2)

        if model_type == "lstm":
            model = StockPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif model_type == "attention_lstm":
            model = AttentiveLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=model_config.get("num_heads", 4),
                dropout=dropout,
            )
        elif model_type == "gru":
            model = GRUPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif model_type == "transformer":
            model = TransformerPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=model_config.get("num_heads", 8),
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model.to(self.device)

    def _get_autocast_context(self):
        """Return an autocast context manager when AMP is enabled, otherwise a no-op context."""
        if self.scaler is None:
            return contextlib.nullcontext()
        return torch.cuda.amp.autocast()

    def _compute_loss(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        criterion: nn.Module,
        accumulation_steps: int,
    ) -> Tuple[torch.Tensor, float]:
        """Compute normalized loss for backprop plus raw loss value for logging."""
        with self._get_autocast_context():
            predictions = self.model(x_batch)
            raw_loss = criterion(predictions, y_batch.unsqueeze(1))
            loss = raw_loss / accumulation_steps
        return loss, float(raw_loss.detach().item())

    def _backward(self, loss: torch.Tensor) -> None:
        """Backpropagate using AMP scaler when enabled."""
        if self.scaler is None:
            loss.backward()
        else:
            self.scaler.scale(loss).backward()

    def _optimizer_step(self, max_grad_norm: float) -> None:
        """Clip gradients and step optimizer using AMP scaler when enabled."""
        if self.scaler is None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            return

        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def _maybe_step_onecycle(self, use_step_scheduler: bool) -> None:
        """Step OneCycleLR scheduler after an optimizer step when configured."""
        if use_step_scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self.scheduler.step()

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        use_step_scheduler: bool = False,
    ) -> float:
        """Train for one epoch with improved gradient handling and accumulation support."""
        self.model.train()
        total_loss = 0.0

        max_grad_norm = self.config.get("grad_clip_norm", 1.0)
        accumulation_steps = self.config.get("gradient_accumulation_steps", 1)

        self.optimizer.zero_grad()

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            loss, raw_loss_value = self._compute_loss(
                x_batch, y_batch, criterion, accumulation_steps
            )
            self._backward(loss)
            total_loss += raw_loss_value

            should_step = (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader)
            if should_step:
                self._optimizer_step(max_grad_norm)
                self._maybe_step_onecycle(use_step_scheduler)

        return total_loss / len(train_loader)

    def validate(
        self, val_loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(x_batch)
                loss = criterion(predictions, y_batch.unsqueeze(1))

                total_loss += loss.item()

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()

        return avg_loss, predictions, targets

    def _create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training/validation DataLoaders."""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))

        batch_size = self.config.get("batch_size", 32)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False,
        )
        return train_loader, val_loader

    def _setup_optimizer_from_scratch(self) -> float:
        """Configure optimizer and return the learning rate used."""
        learning_rate = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 1e-5)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        logger.info(
            f"Using AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}"
        )
        return learning_rate

    def _setup_scheduler_from_scratch(
        self,
        scheduler_type: str,
        num_epochs: int,
        train_loader: DataLoader,
        learning_rate: float,
    ) -> None:
        """Configure learning rate scheduler."""
        if scheduler_type == "cosine":
            warmup_steps = self.config.get("warmup_steps", 1000)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=warmup_steps, T_mult=2, eta_min=1e-6
            )
            logger.info(
                f"Using CosineAnnealingWarmRestarts scheduler with warmup_steps={warmup_steps}"
            )
            return

        if scheduler_type == "onecycle":
            max_lr = self.config.get("max_lr", learning_rate * 10)
            total_steps = num_epochs * len(train_loader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=max_lr, total_steps=total_steps
            )
            logger.info(f"Using OneCycleLR scheduler with max_lr={max_lr}")
            return

        if scheduler_type == "exponential":
            gamma = self.config.get("lr_decay_gamma", 0.95)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=gamma
            )
            logger.info(f"Using ExponentialLR scheduler with gamma={gamma}")
            return

        if scheduler_type == "step":
            step_size = self.config.get("lr_step_size", 30)
            gamma = self.config.get("lr_decay_gamma", 0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
            logger.info(
                f"Using StepLR scheduler with step_size={step_size}, gamma={gamma}"
            )
            return

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )
        logger.info("Using ReduceLROnPlateau scheduler")

    def _initialize_training_from_scratch(
        self, input_size: int, num_epochs: int, train_loader: DataLoader
    ) -> None:
        """Build model and configure optimizer/scheduler when not resuming from a checkpoint."""
        model_type = self.config.get("architecture", "attention_lstm")
        self.model = self.build_model(model_type, input_size)

        learning_rate = self._setup_optimizer_from_scratch()
        scheduler_type = self.config.get("learning_rate_scheduler", "plateau")
        self._setup_scheduler_from_scratch(
            scheduler_type=scheduler_type,
            num_epochs=num_epochs,
            train_loader=train_loader,
            learning_rate=learning_rate,
        )

    def _build_training_criterion(self) -> nn.Module:
        """Create the loss function (optionally with label smoothing)."""
        loss_type = self.config.get("loss_type", "mse")
        label_smoothing = self.config.get("label_smoothing", 0.0)

        if loss_type == "huber" or self.config.get("use_huber_loss", False):
            delta = self.config.get("huber_delta", 1.0)
            base_criterion = nn.HuberLoss(delta=delta)
            logger.info(f"Using Huber loss with delta={delta}")
        elif loss_type == "smooth_l1":
            base_criterion = nn.SmoothL1Loss()
            logger.info("Using Smooth L1 loss")
        else:
            base_criterion = nn.MSELoss()
            logger.info("Using MSE loss")

        if label_smoothing > 0:
            logger.info(f"Applied label smoothing with smoothing={label_smoothing}")
            return LabelSmoothingLoss(base_criterion, smoothing=label_smoothing)

        return base_criterion

    def _step_epoch_scheduler(self, val_loss: float) -> None:
        """Step the LR scheduler when it is epoch-based (OneCycleLR is stepped per batch)."""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
            return

        if isinstance(
            self.scheduler,
            (
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                torch.optim.lr_scheduler.ExponentialLR,
                torch.optim.lr_scheduler.StepLR,
            ),
        ):
            self.scheduler.step()

    def _maybe_save_best_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        metrics: dict,
        best_val_loss: float,
        patience_counter: int,
    ) -> Tuple[float, int]:
        """Save best checkpoint and update early-stopping counter."""
        if val_loss >= best_val_loss:
            return best_val_loss, patience_counter + 1

        checkpoint_path = "trained_data/models/best_model.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "metrics": metrics,
            },
            checkpoint_path,
        )
        logger.info(f"Saved best model with val_loss: {val_loss:.6f}")
        return val_loss, 0

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        num_epochs: int = 100,
        resume_from: str = None,
    ):
        """
        Complete training pipeline.
        
        Args:
            X_train (np.ndarray): Training features of shape (n_samples, sequence_length, n_features)
            y_train (np.ndarray): Training targets of shape (n_samples,)
            x_val (np.ndarray): Validation features of shape (n_samples, sequence_length, n_features)
            y_val (np.ndarray): Validation targets of shape (n_samples,)
            num_epochs (int): Number of epochs to train. When starting from scratch, this is the
                total number of epochs. When resuming from a checkpoint, this represents 
                ADDITIONAL epochs beyond the checkpoint's epoch. Default: 100
            resume_from (str, optional): Path to checkpoint file (.pth) to resume training from.
                If None, training starts from scratch. Default: None
            
        Returns:
            tuple: (train_losses, val_losses) - Lists of training and validation losses per epoch
            
        Note:
            When resume_from is specified, num_epochs represents ADDITIONAL epochs to train,
            not total epochs. For example, resuming from epoch 50 with num_epochs=100 will
            train until epoch 150 (50 + 100 additional epochs).
            
        Example:
            # Train from scratch for 100 epochs
            trainer.train(X_train, y_train, x_val, y_val, num_epochs=100)
            
            # Resume from checkpoint and train 50 more epochs
            trainer.train(X_train, y_train, x_val, y_val, num_epochs=50, 
                         resume_from='trained_data/models/best_model.pth')
        """
        logger.info("Starting training...")

        train_loader, val_loader = self._create_dataloaders(X_train, y_train, x_val, y_val)
        input_size = X_train.shape[2]

        start_epoch = 0
        best_val_loss = float("inf")
        if resume_from:
            start_epoch, best_val_loss = self.load_checkpoint(resume_from, input_size)
        else:
            self._initialize_training_from_scratch(input_size, num_epochs, train_loader)

        criterion = self._build_training_criterion()

        patience = self.config.get("early_stopping_patience", 20)
        patience_counter = 0

        train_losses = []
        val_losses = []

        scheduler_type = self.config.get("learning_rate_scheduler", "plateau")
        use_step_scheduler = scheduler_type == "onecycle"

        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_loss = self.train_epoch(train_loader, criterion, use_step_scheduler)
            train_losses.append(train_loss)

            val_loss, val_predictions, val_targets = self.validate(val_loader, criterion)
            val_losses.append(val_loss)

            self._step_epoch_scheduler(val_loss)

            metrics = self.evaluator.evaluate(val_targets, val_predictions, prefix="val_")

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{start_epoch + num_epochs} - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"Val RMSE: {metrics['val_rmse']:.6f}, "
                    f"Val R2: {metrics['val_r2']:.6f}"
                )

            best_val_loss, patience_counter = self._maybe_save_best_checkpoint(
                epoch=epoch,
                val_loss=val_loss,
                metrics=metrics,
                best_val_loss=best_val_loss,
                patience_counter=patience_counter,
            )

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        logger.info("Training complete!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")

        return train_losses, val_losses

    def evaluate_on_test(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model on test set."""
        logger.info("Evaluating on test set...")

        # Load best model
        checkpoint = torch.load(
            "trained_data/models/best_model.pth",
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Make predictions
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                predictions = self.model(x_batch)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y_batch.numpy())

        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()

        # Calculate metrics
        metrics = self.evaluator.evaluate(targets, predictions, prefix="test_")
        self.evaluator.print_metrics(metrics)

        # Plot results
        self.evaluator.plot_predictions(
            targets,
            predictions,
            title="Test Set: Predictions vs Actual",
            save_path="trained_data/visualizations/test_predictions.png",
        )

        return predictions, targets, metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train stock prediction model")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="market_data/TSLA_data.csv",
        help="Path to data file",
    )
    parser.add_argument(
        "--text-data",
        type=str,
        default=None,
        help=(
            "Optional path to dated text CSV (columns: date,text) to add "
            "text-derived features"
        ),
    )
    parser.add_argument(
        "--text-date-col",
        type=str,
        default="date",
        help="Date column name in --text-data CSV",
    )
    parser.add_argument(
        "--text-text-col",
        type=str,
        default="text",
        help="Text column name in --text-data CSV",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )

    args = parser.parse_args()

    # Load config
    config = copy.deepcopy(load_config(args.config))

    # Initialize trainer
    trainer = EnhancedTrainer(config)

    # Load and preprocess data
    logger.info(f"Loading data from {args.data}")
    df = trainer.data_loader.load_csv(args.data)

    if args.text_data:
        logger.info("Loading and merging text data: %s", args.text_data)
        text_cfg = config.get("text")
        if not isinstance(text_cfg, dict):
            config["text"] = {}
        config["text"]["enabled"] = True
        config["text"]["text_column"] = args.text_text_col

        text_features_df = trainer.data_loader.load_text_csv(
            args.text_data,
            date_column=args.text_date_col,
            text_column=args.text_text_col,
        )
        df = trainer.data_loader.merge_text_features(df, text_features_df)

    logger.info("Preprocessing data...")
    x_train, y_train, x_val, y_val, x_test, y_test = trainer.data_loader.preprocess(
        df,
        add_features=True,
        scaler_type="standard",
        sequence_length=config.get("data", {}).get("sequence_length", 60),
        test_size=0.2,
        validation_size=0.1,
    )

    # Save scaler
    trainer.data_loader.save_scaler("trained_data/models/scaler.pkl")

    # Train model
    _, _ = trainer.train(
        x_train, y_train, x_val, y_val, num_epochs=args.epochs, resume_from=args.resume
    )

    # Evaluate on test set
    predictions, targets, _ = trainer.evaluate_on_test(x_test, y_test)

    # Run backtest
    logger.info("Running backtest...")
    backtester = Backtester(initial_capital=10000.0)
    signals = generate_simple_strategy_signals(predictions, targets)
    backtest_metrics = backtester.run_backtest(targets, signals)

    logger.info("Backtest Results:")
    for key, value in backtest_metrics.items():
        logger.info(f"{key}: {value:.2f}")

    logger.info("All done!")


if __name__ == "__main__":
    main()
