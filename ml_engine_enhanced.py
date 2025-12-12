"""
Optimized ML Engine with improved CPU and GPU performance.
Includes enhanced training loops, mixed precision, and memory optimization.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import r2_score, mean_squared_error

# Import optimized modules
from memory_manager_enhanced import MemoryManager, mixed_precision_context
from data_processing_optimized import create_optimized_dataloader
from models_enhanced import (
    StockPredictor,
    AttentiveLSTM,
    GRUPredictor,
    TransformerPredictor
)


# Configure logging
logger = logging.getLogger(__name__)


class EnhancedMLEngine:
    """
    Enhanced ML Engine for stock market prediction with optimized CPU/GPU performance.
    
    Features:
    - Automatic mixed precision training
    - Dynamic batch sizing
    - Memory-efficient operations
    - Gradient accumulation
    - Advanced learning rate scheduling
    - Model architecture selection
    - Ensemble methods
    """

    BEST_MODEL_FILENAME = "best_model.pth"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ML Engine with configuration.
        
        Args:
            config: Configuration dictionary with model and training parameters
        """
        self.config = config
        
        # Set device
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            device=self.device,
            threshold_mb=config.get("memory", {}).get("threshold_mb", 1000),
            log_level=config.get("memory", {}).get("log_level", "INFO"),
            proactive_cleanup=config.get("memory", {}).get("proactive_cleanup", True)
        )
        
        # Set paths
        self.model_dir = Path(config.get("model_dir", "./models"))
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Initialize model
        self.model = self._create_model()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler(enabled=config.get("mixed_precision", True) and self.device == "cuda")
        
        # Set training parameters
        self.batch_size = config.get("batch_size", 32)
        self.accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.mixed_precision = config.get("mixed_precision", True) and self.device == "cuda"
        self.clip_grad_norm = config.get("clip_grad_norm", 1.0)
        self.early_stopping_patience = config.get("early_stopping_patience", 10)
        
        # Auto-resume configuration
        self.auto_resume = config.get("auto_resume", True)
        self.start_epoch = 0
        
        logger.info(f"EnhancedMLEngine initialized with {self.model.__class__.__name__} model")
        
    def _create_model(self) -> nn.Module:
        """
        Create model based on configuration.
        
        Returns:
            PyTorch model
        """
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "lstm")

        base_kwargs = {
            "input_size": model_config.get("input_size", 7),
            "hidden_size": model_config.get("hidden_size", 128),
            "num_layers": model_config.get("num_layers", 3),
            "dropout": model_config.get("dropout", 0.2),
            "bidirectional": model_config.get("bidirectional", False),
        }

        if model_type == "ensemble":
            model = self._create_ensemble_model(model_config, base_kwargs)
        else:
            model = self._create_single_model(model_type, model_config, base_kwargs)

        # Load pre-trained weights if specified
        pretrained_path = model_config.get("pretrained_path")
        if pretrained_path:
            self._load_pretrained_weights(model, pretrained_path)

        return model

    def _create_single_model(
        self,
        model_type: str,
        model_config: Dict[str, Any],
        base_kwargs: Dict[str, Any]
    ) -> nn.Module:
        builders = {
            "lstm": lambda: StockPredictor(**base_kwargs),
            "attention_lstm": lambda: AttentiveLSTM(
                **base_kwargs,
                num_heads=model_config.get("num_heads", 4),
                use_flash_attention=model_config.get("use_flash_attention", True)
            ),
            "gru": lambda: GRUPredictor(**base_kwargs),
            "transformer": lambda: TransformerPredictor(
                **base_kwargs,
                num_heads=model_config.get("num_heads", 8),
                use_flash_attention=model_config.get("use_flash_attention", True),
                positional_encoding=model_config.get("positional_encoding", "learned")
            ),
        }

        try:
            return builders[model_type]()
        except KeyError as e:
            raise ValueError(f"Unknown model type: {model_type}") from e

    def _create_ensemble_model(
        self,
        model_config: Dict[str, Any],
        base_kwargs: Dict[str, Any]
    ) -> nn.Module:
        base_models = []
        for base_config in model_config.get("base_models", []):
            base_type = base_config.get("type", "lstm")

            per_model_kwargs = dict(base_kwargs)
            for k in ("input_size", "hidden_size", "num_layers", "dropout", "bidirectional"):
                if k in base_config:
                    per_model_kwargs[k] = base_config[k]

            base_models.append(self._create_single_model(base_type, base_config, per_model_kwargs))

        # If no base models specified, create default set
        if not base_models:
            base_models = [
                self._create_single_model("lstm", {}, base_kwargs),
                self._create_single_model("attention_lstm", {}, base_kwargs),
                self._create_single_model("gru", {}, base_kwargs),
            ]

        # EnsemblePredictor not yet implemented; fallback to first base model
        if base_models:
            logger.warning("EnsemblePredictor not yet implemented, using first base model instead")
            return base_models[0]

        return StockPredictor(**base_kwargs)

    def _load_pretrained_weights(self, model: nn.Module, pretrained_path: str) -> None:
        try:
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict)
            logger.info(f"Loaded pre-trained weights from {pretrained_path}")
        except Exception as e:
            logger.warning(f"Failed to load pre-trained weights: {e}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer based on configuration.
        
        Returns:
            PyTorch optimizer
        """
        optimizer_config = self.config.get("optimizer", {})
        if optimizer_config is None:
            optimizer_config = {}
        if isinstance(optimizer_config, str):
            optimizer_config = {"type": optimizer_config}

        optimizer_type = optimizer_config.get("type", "adam")
        lr = float(
            optimizer_config.get(
                "learning_rate",
                self.config.get("learning_rate", self.config.get("model", {}).get("learning_rate", 0.001)),
            )
        )
        weight_decay = optimizer_config.get(
            "weight_decay",
            self.config.get("model", {}).get("optimizer", {}).get("weight_decay", 0.0),
        )

        # Keep a stable reference for LR warmup regardless of how config is structured
        self.base_lr = lr
        
        # Create optimizer based on type
        if optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get("betas", (0.9, 0.999))
            )
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get("betas", (0.9, 0.999))
            )
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=optimizer_config.get("momentum", 0.9),
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "rmsprop":
            return optim.RMSprop(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=optimizer_config.get("momentum", 0.0)
            )
        else:
            logger.warning(f"Unknown optimizer type: {optimizer_type}, using Adam")
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler based on configuration.
        
        Returns:
            PyTorch learning rate scheduler
        """
        scheduler_config = self.config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type")
        
        if not scheduler_type:
            return None
        
        # Create scheduler based on type
        if scheduler_type.lower() == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 5),
                verbose=True
            )
        elif scheduler_type.lower() == "cosine":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config.get("t_0", 10),
                T_mult=scheduler_config.get("t_mult", 1),
                eta_min=scheduler_config.get("eta_min", 1e-6)
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, not using scheduler")
            return None
    
    def _prepare_data(
        self, 
        features: Union[List, np.ndarray, torch.Tensor],
        targets: Union[List, np.ndarray, torch.Tensor],
        validation_split: float = 0.2,
        shuffle: bool = True
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare data loaders for training and validation.
        
        Args:
            features: Input features
            targets: Target values
            validation_split: Fraction of data to use for validation
            shuffle: Whether to shuffle the data
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert to numpy arrays if needed
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)
        if isinstance(targets, list):
            targets = np.array(targets, dtype=np.float32)
        
        # Convert to tensors if needed
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        if isinstance(targets, np.ndarray):
            targets = torch.tensor(targets, dtype=torch.float32)
        
        # Ensure targets are 2D
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)
        
        # Split data into train and validation sets
        if validation_split > 0:
            # Calculate split index
            split_idx = int(len(features) * (1 - validation_split))
            
            if shuffle:
                # Generate random indices
                indices = torch.randperm(len(features))
                train_indices = indices[:split_idx]
                val_indices = indices[split_idx:]
                
                # Split data
                train_features = features[train_indices]
                train_targets = targets[train_indices]
                val_features = features[val_indices]
                val_targets = targets[val_indices]
            else:
                # Split data sequentially
                train_features = features[:split_idx]
                train_targets = targets[:split_idx]
                val_features = features[split_idx:]
                val_targets = targets[split_idx:]
            
            # Create datasets
            train_dataset = TensorDataset(train_features, train_targets)
            val_dataset = TensorDataset(val_features, val_targets)
            
            # Create data loaders with optimized settings
            train_loader = create_optimized_dataloader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.config.get("num_workers"),
                pin_memory=self.config.get("pin_memory"),
                device=self.device
            )
            
            val_loader = create_optimized_dataloader(
                val_dataset,
                batch_size=self.batch_size * 2,  # Larger batch size for validation
                shuffle=False,
                num_workers=self.config.get("num_workers"),
                pin_memory=self.config.get("pin_memory"),
                device=self.device
            )
            
            return train_loader, val_loader
        else:
            # No validation split - return only training loader
            train_dataset = TensorDataset(features, targets)
            train_loader = create_optimized_dataloader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.config.get("num_workers"),
                pin_memory=self.config.get("pin_memory"),
                device=self.device
            )
            
            return train_loader, None
    
    def _init_weights(self, module: nn.Module):
        """
        Initialize model weights using Xavier/He initialization (PyTorch best practices).
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            # Use Xavier uniform for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
            # Use orthogonal initialization for recurrent layers (PyTorch recommendation)
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def _get_loss_function(self) -> nn.Module:
        """
        Get loss function based on configuration.
        
        Returns:
            Loss function module
        """
        loss_type = str(self.config.get("loss_type", "mse") or "mse").lower()

        if loss_type == "mse":
            return nn.MSELoss()
        if loss_type == "mae":
            return nn.L1Loss()
        if loss_type == "huber":
            # Huber loss is more robust to outliers (PyTorch best practice)
            delta = self.config.get("huber_delta", 1.0)
            return nn.HuberLoss(delta=delta)
        if loss_type == "smooth_l1":
            return nn.SmoothL1Loss()
        logger.warning(f"Unknown loss type: {loss_type}, using MSE")
        return nn.MSELoss()

    def _ensure_float_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32)
        return x

    def _match_target_shape(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        if targets.shape != outputs.shape:
            if targets.numel() == outputs.numel():
                targets = targets.view_as(outputs)
            else:
                raise ValueError(f"Target shape {targets.shape} does not match output shape {outputs.shape}")

        return targets

    def _build_train_val_loaders(
        self,
        train_features: Union[np.ndarray, torch.Tensor],
        train_targets: Union[np.ndarray, torch.Tensor],
        val_features: Optional[Union[np.ndarray, torch.Tensor]],
        val_targets: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        if val_features is None or val_targets is None:
            return self._prepare_data(
                train_features,
                train_targets,
                validation_split=self.config.get("validation_split", 0.2),
            )

        train_loader = self._prepare_data(train_features, train_targets, validation_split=0.0)[0]

        val_features = self._ensure_float_tensor(val_features)
        val_targets = self._ensure_float_tensor(val_targets)
        if isinstance(val_targets, torch.Tensor) and val_targets.dim() == 1:
            val_targets = val_targets.unsqueeze(1)

        val_dataset = TensorDataset(val_features, val_targets)
        val_loader = create_optimized_dataloader(
            val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.config.get("num_workers"),
            pin_memory=self.config.get("pin_memory"),
            device=self.device,
        )
        return train_loader, val_loader

    def _should_step_optimizer(self, batch_idx: int) -> bool:
        return (batch_idx + 1) % self.accumulation_steps == 0

    def _backward(self, loss: torch.Tensor) -> None:
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self) -> None:
        if self.mixed_precision:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        if self.mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

    def _apply_lr_warmup(self, global_step: int, warmup_steps: int) -> None:
        if warmup_steps <= 0 or global_step >= warmup_steps:
            return

        lr_scale = min(1.0, float(global_step + 1) / warmup_steps)
        base_lr = float(getattr(self, "base_lr", self.optimizer.param_groups[0]["lr"]))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = base_lr * lr_scale

    def _train_one_epoch(
        self,
        epoch: int,
        train_loader: DataLoader,
        criterion: nn.Module,
        warmup_steps: int,
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        steps = 0

        for batch_idx, (batch_features, batch_targets) in enumerate(train_loader):
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)

            with mixed_precision_context(self.mixed_precision):
                outputs = self.model(batch_features)
                batch_targets = self._match_target_shape(batch_targets, outputs)
                loss = criterion(outputs, batch_targets) / self.accumulation_steps

            self._backward(loss)

            if self._should_step_optimizer(batch_idx):
                self._optimizer_step()
                global_step = epoch * len(train_loader) + batch_idx
                self._apply_lr_warmup(global_step, warmup_steps)

            epoch_loss += loss.item() * self.accumulation_steps
            steps += 1

        return epoch_loss / max(steps, 1)

    def _validate_one_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        epoch_loss = 0.0
        steps = 0

        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                outputs = self.model(batch_features)
                batch_targets = self._match_target_shape(batch_targets, outputs)

                loss = criterion(outputs, batch_targets)
                epoch_loss += loss.item()
                steps += 1

        return epoch_loss / max(steps, 1)

    def _step_scheduler(self, val_loss: float) -> None:
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _maybe_resume_from_checkpoint(self, checkpoint_path: Path) -> bool:
        if not self.auto_resume or not checkpoint_path.exists():
            return False
        
        try:
            logger.info(f"Auto-resuming from checkpoint: {checkpoint_path}")
            self.load_model(self.BEST_MODEL_FILENAME)
            logger.info(
                f"Resumed from epoch {self.start_epoch} with "
                f"best_val_loss={self.best_val_loss:.6f}"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to resume from checkpoint: {e}. Starting fresh.")
            return False

    def _initialize_fresh_training_state(self) -> None:
        logger.info("Initializing model weights from scratch")
        self.model.apply(self._init_weights)
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

    def _train_epochs(
        self,
        start_epoch: int,
        total_epochs: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: nn.Module,
        warmup_steps: int,
        best_val_loss: float,
        train_losses: List[float],
        val_losses: List[float],
    ) -> Tuple[List[float], List[float], float, int]:
        epochs_without_improvement = 0
        completed_epoch = start_epoch

        for epoch in range(start_epoch, total_epochs):
            completed_epoch = epoch + 1

            avg_train_loss = self._train_one_epoch(epoch, train_loader, criterion, warmup_steps)
            train_losses.append(avg_train_loss)

            if val_loader is None:
                logger.info(f"Epoch {epoch+1}/{total_epochs} - Train Loss: {avg_train_loss:.6f}")
                continue

            avg_val_loss = self._validate_one_epoch(val_loader, criterion)
            val_losses.append(avg_val_loss)

            self._step_scheduler(avg_val_loss)

            if avg_val_loss < best_val_loss:
                improvement = best_val_loss - avg_val_loss
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                self.save_model(self.BEST_MODEL_FILENAME)
                logger.info(f"✓ New best model saved (improved by {improvement:.6f})")
            else:
                epochs_without_improvement += 1

            logger.info(
                f"Epoch {epoch+1}/{total_epochs} - "
                f"Train Loss: {avg_train_loss:.6f}, "
                f"Val Loss: {avg_val_loss:.6f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            if epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        return train_losses, val_losses, best_val_loss, completed_epoch

    def train(
        self,
        train_features: Union[np.ndarray, torch.Tensor],
        train_targets: Union[np.ndarray, torch.Tensor],
        val_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
        val_targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
        epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the model with enhanced training loop using PyTorch best practices.
        
        Features:
        - Automatic checkpoint resumption
        - Gradient clipping
        - Learning rate warmup
        - Mixed precision training
        - Early stopping
        - Model checkpointing
        
        Args:
            train_features: Training features
            train_targets: Training targets
            val_features: Validation features (optional)
            val_targets: Validation targets (optional)
            epochs: Number of epochs (overrides config)
            
        Returns:
            Dictionary with training history
        """
        checkpoint_path = self.model_dir / self.BEST_MODEL_FILENAME
        resumed = self._maybe_resume_from_checkpoint(checkpoint_path)
        if not resumed:
            self._initialize_fresh_training_state()

        num_epochs = epochs if epochs is not None else self.config.get("epochs", 100)
        train_loader, val_loader = self._build_train_val_loaders(
            train_features, train_targets, val_features, val_targets
        )

        criterion = self._get_loss_function()
        warmup_steps = int(self.config.get("warmup_steps", 1000) or 0)

        best_val_loss = self.best_val_loss
        train_losses: List[float] = self.train_losses.copy()
        val_losses: List[float] = self.val_losses.copy()

        total_epochs = self.start_epoch + num_epochs
        logger.info(
            f"Training from epoch {self.start_epoch + 1} to {total_epochs} "
            f"(+{num_epochs} epochs)"
        )
        logger.info(f"Using {criterion.__class__.__name__} loss function")
        logger.info(f"Mixed precision: {self.mixed_precision}, Gradient clipping: {self.clip_grad_norm}")
        if resumed:
            logger.info(f"Continuing from best_val_loss={best_val_loss:.6f}")

        train_losses, val_losses, best_val_loss, completed_epoch = self._train_epochs(
            start_epoch=self.start_epoch,
            total_epochs=total_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            warmup_steps=warmup_steps,
            best_val_loss=best_val_loss,
            train_losses=train_losses,
            val_losses=val_losses,
        )

        # Update engine state for next training session
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.best_val_loss = best_val_loss
        self.start_epoch = completed_epoch

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses),
            "total_epochs": total_epochs,
            "resumed": resumed
        }
    
    def evaluate(
        self, 
        features: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            features: Test features
            targets: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        # Prepare data
        test_loader, _ = self._prepare_data(features, targets, validation_split=0.0)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_targets.numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics (PyTorch best practice for evaluation)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(targets - predictions))
        r2 = r2_score(targets, predictions)
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def predict(
        self,
        features: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            features: Input features
            
        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        
        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        
        # Move to device
        features = features.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(features)
        
        return predictions.cpu().numpy()
    
    def save_model(self, filepath: str):
        """
        Save model state including training progress.
        
        Args:
            filepath: Path to save model
        """
        save_path = self.model_dir / filepath
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'start_epoch': self.start_epoch
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, filepath: str):
        """
        Load model state including training progress.
        
        Args:
            filepath: Path to load model from
        """
        load_path = self.model_dir / filepath
        
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.start_epoch = checkpoint.get('start_epoch', 0)
        
        logger.info(f"Model loaded from {load_path}")
