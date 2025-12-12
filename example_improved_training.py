"""
Example script demonstrating improved ML engine with PyTorch best practices.
This script shows how to use all the new features for optimal model training.
"""

import numpy as np
import torch

from ml_engine_enhanced import EnhancedMLEngine
from data_processing import prepare_sequences, augment_data
from training_utils import ModelEnsemble

# Set random seeds for reproducibility (PyTorch best practice)
torch.manual_seed(42)
rng = np.random.default_rng(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def create_sample_data(n_samples=1000, n_features=7, rng=None, seed=42):
    """Create sample time series data for demonstration."""
    rng = rng if rng is not None else np.random.default_rng(seed)

    # Generate synthetic stock data
    time = np.arange(n_samples)
    trend = 0.01 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 252)  # Annual cycle
    noise = rng.normal(0, 2, n_samples)
    
    prices = 100 + trend + seasonality + noise
    
    # Create OHLCV-like features
    features = np.zeros((n_samples, n_features))
    features[:, 0] = prices  # close
    features[:, 1] = prices * 1.01  # high
    features[:, 2] = prices * 0.99  # low
    features[:, 3] = prices * 0.995  # open
    features[:, 4] = rng.uniform(1e6, 1e7, n_samples)  # volume
    features[:, 5] = np.gradient(prices)  # price change
    features[:, 6] = np.abs(features[:, 1] - features[:, 2])  # high-low range
    
    return features


def main():
    """Main training example with all improvements."""
    
    print("=" * 80)
    print("Enhanced ML Engine Training Example")
    print("Using PyTorch Best Practices")
    print("=" * 80)
    
    # 1. Configuration with optimized settings
    config = {
        "model": {
            "type": "attention_lstm",  # or "transformer", "lstm", "gru"
            "input_size": 7,
            "hidden_size": 128,
            "num_layers": 3,
            "dropout": 0.2,
            "num_heads": 4,
            "use_flash_attention": True,  # PyTorch 2.0+ feature
        },
        "optimizer": "adamw",  # AdamW recommended for transformers
        "learning_rate": 0.001,
        "weight_decay": 0.01,  # L2 regularization
        "batch_size": 32,
        "epochs": 100,
        "validation_split": 0.2,
        "loss_type": "huber",  # Robust to outliers
        "huber_delta": 1.0,
        "clip_grad_norm": 1.0,  # Gradient clipping
        "mixed_precision": True,  # Faster training on GPU
        "early_stopping_patience": 15,
        "warmup_steps": 1000,  # Learning rate warmup
        "model_dir": "./trained_data/models",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 4,
        "pin_memory": True,
        "enable_data_augmentation": False,
    }
    
    print(f"\nUsing device: {config['device']}")
    print(f"Model architecture: {config['model']['type']}")
    print(f"Loss function: {config['loss_type']}")
    
    print("Generating sample data...")
    raw_data = create_sample_data(n_samples=2000, n_features=7, rng=rng)
    
    # 3. Prepare sequences
    print("Preparing sequences...")
    sequences, targets, _ = prepare_sequences(
        df=None,  # Would use actual DataFrame in practice
        sequence_length=60,
        target_column="close",
        scale_features=True,
        scale_target=True
    )
    
    # For this example, use the raw data directly
    n_samples = len(raw_data) - 60
    sequences = np.array([raw_data[i:i+60] for i in range(n_samples)])
    targets = np.array([raw_data[i+60, 0] for i in range(n_samples)])
    
    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")
    # 4. Optional: Data augmentation
    if config.get("enable_data_augmentation", False):
        print("\nApplying data augmentation...")
        sequences, targets = augment_data(
            sequences, targets,
            noise_level=0.01,
            augmentation_factor=2
        )
        print(f"Augmented to {len(sequences)} samples")
        print(f"Augmented to {len(sequences)} samples")
    
    # 5. Split data
    split_idx = int(len(sequences) * 0.8)
    train_features = sequences[:split_idx]
    train_targets = targets[:split_idx]
    test_features = sequences[split_idx:]
    test_targets = targets[split_idx:]
    
    print(f"\nTrain set: {len(train_features)} samples")
    print(f"Test set: {len(test_features)} samples")
    
    # 6. Initialize and train model
    print("\n" + "-" * 80)
    print("Initializing ML Engine...")
    engine = EnhancedMLEngine(config)
    
    print("\nStarting training with:")
    print("  - Xavier/Orthogonal weight initialization")
    print("  - Gradient clipping (max norm: 1.0)")
    print("  - Learning rate warmup (1000 steps)")
    print("  - Mixed precision training")
    print("  - Early stopping (patience: 15)")
    print("  - Huber loss (robust to outliers)")
    
    # Train the model
    history = engine.train(
        train_features=train_features,
        train_targets=train_targets,
        epochs=config["epochs"]
    )
    
    print("\n" + "-" * 80)
    print("Training completed!")
    print(f"Epochs trained: {history['epochs_trained']}")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Final train loss: {history['train_losses'][-1]:.6f}")
    if history['val_losses']:
        print(f"Final val loss: {history['val_losses'][-1]:.6f}")
    
    # 7. Evaluate on test set
    print("\n" + "-" * 80)
    print("Evaluating on test set...")
    metrics = engine.evaluate(test_features, test_targets)
    
    print("\nTest Set Metrics:")
    print(f"  MSE:      {metrics['mse']:.6f}")
    print(f"  RMSE:     {metrics['rmse']:.6f}")
    print(f"  MAE:      {metrics['mae']:.6f}")
    print(f"  R² Score: {metrics['r2_score']:.6f}")
    
    # 8. Make predictions
    print("\n" + "-" * 80)
    print("Making predictions on test set...")
    predictions = engine.predict(test_features[:10])
    
    print("\nSample Predictions vs Actual:")
    print("Idx | Predicted | Actual   | Error")
    print("-" * 40)
    for i in range(min(10, len(predictions))):
        pred = predictions[i, 0]
        actual = test_targets[i]
        error = abs(pred - actual)
        print(f"{i:3d} | {pred:9.4f} | {actual:8.4f} | {error:6.4f}")
    
    # 9. Save model
    print("\n" + "-" * 80)
    print("Saving trained model...")
    engine.save_model("improved_model.pth")
    print("Model saved successfully!")
    
    # 10. Demonstrate ensemble (optional)
    print("\n" + "-" * 80)
    print("Training ensemble models for improved predictions...")
    
    # Train multiple models with different configurations
    models = []
    for i, model_type in enumerate(["lstm", "attention_lstm"]):
        print(f"\nTraining model {i+1}/2: {model_type}")
        config_copy = config.copy()
        config_copy["model"] = config["model"].copy()
        config_copy["model"]["type"] = model_type
        config_copy["epochs"] = 50  # Fewer epochs for demo
        
        engine_i = EnhancedMLEngine(config_copy)
        engine_i.train(train_features, train_targets)
        models.append(engine_i.model)
    
    # Create ensemble
    ensemble = ModelEnsemble(models, ensemble_method='average')
    
    # Evaluate ensemble
    test_tensor = torch.tensor(test_features[:100], dtype=torch.float32)
    _ = ensemble.predict(test_tensor)
    
    print("\nEnsemble predictions completed")
    print(f"Prediction diversity: {ensemble.evaluate_diversity(test_tensor):.6f}")
    
    print("\n" + "=" * 80)
    print("Training example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
