"""
Enhanced training script with comprehensive features.
Demonstrates best practices for ML model training.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
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
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
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

    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module, use_step_scheduler: bool = False) -> float:
        """Train for one epoch with improved gradient handling."""
        self.model.train()
        total_loss = 0
        
        # Improved gradient clipping value
        max_grad_norm = self.config.get("grad_clip_norm", 1.0)

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training for faster computation and lower memory
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch.unsqueeze(1))

                self.scaler.scale(loss).backward()

                # Adaptive gradient clipping for better training stability
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch.unsqueeze(1))
                loss.backward()

                # Adaptive gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

                self.optimizer.step()
            
            # Step scheduler if using OneCycleLR
            if use_step_scheduler and hasattr(self.scheduler, 'step') and isinstance(
                self.scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                self.scheduler.step()

            total_loss += loss.item()

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
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch.unsqueeze(1))

                total_loss += loss.item()

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()

        return avg_loss, predictions, targets

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_epochs: int = 100,
        resume_from: str = None,
    ):
        """
        Complete training pipeline.
        
        Args:
            X_train (np.ndarray): Training features of shape (n_samples, sequence_length, n_features)
            y_train (np.ndarray): Training targets of shape (n_samples,)
            X_val (np.ndarray): Validation features of shape (n_samples, sequence_length, n_features)
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
            trainer.train(X_train, y_train, X_val, y_val, num_epochs=100)
            
            # Resume from checkpoint and train 50 more epochs
            trainer.train(X_train, y_train, X_val, y_val, num_epochs=50, 
                         resume_from='trained_data/models/best_model.pth')
        """
        logger.info("Starting training...")

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

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

        # Get input size
        input_size = X_train.shape[2]
        
        # Check if resuming from checkpoint
        start_epoch = 0
        best_val_loss = float("inf")
        
        if resume_from:
            start_epoch, best_val_loss = self.load_checkpoint(resume_from, input_size)
        else:
            # Build model from scratch
            model_type = self.config.get("architecture", "attention_lstm")
            self.model = self.build_model(model_type, input_size)

            # Setup optimizer with improved parameters
            learning_rate = self.config.get("learning_rate", 0.001)
            weight_decay = self.config.get("weight_decay", 1e-5)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            logger.info(f"Using AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}")

            # Setup scheduler with warmup support
            scheduler_type = self.config.get("learning_rate_scheduler", "plateau")
            if scheduler_type == "cosine":
                # Cosine annealing with warmup
                warmup_steps = self.config.get("warmup_steps", 1000)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=warmup_steps, T_mult=2, eta_min=1e-6
                )
                logger.info(f"Using CosineAnnealingWarmRestarts scheduler with warmup_steps={warmup_steps}")
            elif scheduler_type == "onecycle":
                # One cycle learning rate policy
                max_lr = self.config.get("max_lr", learning_rate * 10)
                total_steps = num_epochs * len(train_loader)
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer, max_lr=max_lr, total_steps=total_steps
                )
                logger.info(f"Using OneCycleLR scheduler with max_lr={max_lr}")
            else:
                # Default: ReduceLROnPlateau
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
                )
                logger.info("Using ReduceLROnPlateau scheduler")

        # Loss function - Use Huber loss for better robustness to outliers
        loss_type = self.config.get("loss_type", "mse")
        if loss_type == "huber" or self.config.get("use_huber_loss", False):
            # Huber loss is more robust to outliers than MSE
            delta = self.config.get("huber_delta", 1.0)
            criterion = nn.HuberLoss(delta=delta)
            logger.info(f"Using Huber loss with delta={delta}")
        elif loss_type == "smooth_l1":
            # Smooth L1 loss (similar to Huber but with different formulation)
            criterion = nn.SmoothL1Loss()
            logger.info("Using Smooth L1 loss")
        else:
            criterion = nn.MSELoss()
            logger.info("Using MSE loss")

        # Training loop
        patience = self.config.get("early_stopping_patience", 20)
        patience_counter = 0

        train_losses = []
        val_losses = []
        
        # Check if using step-based scheduler (OneCycleLR)
        scheduler_type = self.config.get("learning_rate_scheduler", "plateau")
        use_step_scheduler = scheduler_type == "onecycle"

        for epoch in range(start_epoch, start_epoch + num_epochs):
            # Train with improved scheduling
            train_loss = self.train_epoch(train_loader, criterion, use_step_scheduler)
            train_losses.append(train_loss)

            # Validate
            val_loss, val_predictions, val_targets = self.validate(
                val_loader, criterion
            )
            val_losses.append(val_loss)

            # Update scheduler based on type
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                self.scheduler.step(epoch)
            # OneCycleLR is stepped in train_epoch

            # Calculate metrics
            metrics = self.evaluator.evaluate(
                val_targets, val_predictions, prefix="val_"
            )

            # Logging
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{start_epoch + num_epochs} - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"Val RMSE: {metrics['val_rmse']:.6f}, "
                    f"Val R2: {metrics['val_r2']:.6f}"
                )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

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
            else:
                patience_counter += 1

            # Early stopping
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
        checkpoint = torch.load("trained_data/models/best_model.pth")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Make predictions
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                predictions = self.model(X_batch)
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
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize trainer
    trainer = EnhancedTrainer(config)

    # Load and preprocess data
    logger.info(f"Loading data from {args.data}")
    df = trainer.data_loader.load_csv(args.data)

    logger.info("Preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.data_loader.preprocess(
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
    train_losses, val_losses = trainer.train(
        X_train, y_train, X_val, y_val, num_epochs=args.epochs, resume_from=args.resume
    )

    # Evaluate on test set
    predictions, targets, metrics = trainer.evaluate_on_test(X_test, y_test)

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
