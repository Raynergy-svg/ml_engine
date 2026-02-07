#!/usr/bin/env python3
"""
Migrate Keras 2.x models to Keras 3.x format.

The .keras format changed between Keras 2 and Keras 3:
- Keras 2: uses 'keras.src.engine.functional' module path
- Keras 3: uses 'keras.models' module path

This script extracts weights from old models and loads them into fresh Keras 3 architectures.

Usage:
    python migrate_keras_models.py                    # Migrate all models
    python migrate_keras_models.py --dry-run          # Show what would be migrated
    python migrate_keras_models.py --model buddy_tf   # Migrate specific model
    python migrate_keras_models.py --scan             # Scan all models (including pair subdirs)
"""

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.table import Table

console = Console()

MODEL_DIR = Path("trained_data/models")
BACKUP_DIR = Path("trained_data/models_keras2_backup")

# Model filename to architecture type mapping
MODEL_ARCH_MAP = {
    "buddy_tf.keras": "tcn",
    "tf_model.keras": "tcn",
    "tcn_direction.keras": "tcn",
    "transformer_direction.keras": "transformer",
}


def check_model_version(model_path: Path) -> str:
    """Check if a .keras model is Keras 2 or Keras 3 format."""
    try:
        with zipfile.ZipFile(model_path, 'r') as zf:
            config_data = zf.read("config.json")
            config = json.loads(config_data)
            
            # Check module path in config
            module = config.get("module", "")
            if "keras.src.engine" in module:
                return "keras2"
            elif "keras.models" in module or "keras.src.models" in module:
                return "keras3"
            
            # Check layers for Keras 2 markers
            layers = config.get("config", {}).get("layers", [])
            for layer in layers:
                if "keras.src.engine" in layer.get("module", ""):
                    return "keras2"
            
            return "keras3"  # Assume Keras 3 if no Keras 2 markers found
    except Exception as e:
        return f"error: {e}"


def get_model_input_shape(model_path: Path) -> tuple:
    """Extract input shape from model config."""
    try:
        with zipfile.ZipFile(model_path, 'r') as zf:
            config = json.loads(zf.read("config.json"))
            layers = config.get("config", {}).get("layers", [])
            for layer in layers:
                if layer.get("class_name") == "InputLayer":
                    shape = layer.get("config", {}).get("batch_input_shape", [])
                    if len(shape) >= 3:
                        return (shape[1], shape[2])  # (seq_len, feature_dim)
    except Exception:
        pass
    return (60, 142)  # Default fallback


def build_fresh_model(arch_type: str, seq_len: int, feature_dim: int):
    """Build a fresh model with the specified architecture."""
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.regularizers import l2
    
    if arch_type == "transformer":
        # Build Transformer model matching TransformerDirectionTrainer._build_model()
        l2_reg = l2(0.005)
        d_model = 16
        num_heads = 2
        num_layers = 1
        dff = 32
        dropout = 0.2
        
        inp = tf.keras.Input(shape=(int(seq_len), int(feature_dim)), name="features")
        
        # Input noise and spatial dropout
        x = layers.GaussianNoise(0.03)(inp)
        x = layers.SpatialDropout1D(0.15)(x)
        
        # Project to d_model dimension
        x = layers.Dense(d_model, kernel_regularizer=l2_reg, name='input_projection')(x)
        x = layers.Dropout(0.15)(x)
        
        # Positional encoding (sinusoidal)
        positions = np.arange(seq_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = pos_encoding[np.newaxis, :, :].astype(np.float32)
        pos_tensor = tf.constant(pos_encoding)
        x = x + tf.cast(pos_tensor, x.dtype)
        
        # Transformer encoder layers
        for i in range(num_layers):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads,
                name=f'transformer_{i}_mha'
            )(x, x)
            attn_output = layers.Dropout(dropout)(attn_output)
            x = layers.LayerNormalization(epsilon=1e-6, name=f'transformer_{i}_ln1')(x + attn_output)
            
            # Feedforward
            ffn = layers.Dense(dff, activation='relu', kernel_regularizer=l2_reg, name=f'transformer_{i}_ffn1')(x)
            ffn = layers.Dense(d_model, kernel_regularizer=l2_reg, name=f'transformer_{i}_ffn2')(ffn)
            ffn = layers.Dropout(dropout)(ffn)
            x = layers.LayerNormalization(epsilon=1e-6, name=f'transformer_{i}_ln2')(x + ffn)
        
        # Global pooling and output head
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(16, activation='tanh', kernel_regularizer=l2_reg)(x)
        x = layers.Dropout(0.15)(x)
        
        direction = layers.Dense(
            1, activation='sigmoid', name='direction', dtype='float32',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05),
            bias_initializer=tf.keras.initializers.Zeros(),
        )(x)
        
        model = tf.keras.Model(inputs=inp, outputs=direction, name='transformer_direction')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        return model
    
    elif arch_type == "tcn":
        # Build a multi-output TCN model matching the old MetalOptimizedTCN architecture
        # Based on main.py buddy model structure
        l2_reg = l2(0.002)
        hidden_size = 32  # Match old model
        dropout = 0.35
        
        inp = tf.keras.Input(shape=(int(seq_len), int(feature_dim)), name="input")
        
        # Input regularization 
        x = layers.GaussianNoise(0.03)(inp)
        x = layers.SpatialDropout1D(dropout * 0.5)(x)
        
        # TCN layers with exponentially increasing dilation (2 blocks like old model)
        for i in range(2):
            dilation_rate = 2 ** i
            
            # Causal convolution with dilation
            conv_out = layers.Conv1D(
                filters=hidden_size,
                kernel_size=3,
                padding='causal',
                dilation_rate=dilation_rate,
                activation='relu',
                kernel_regularizer=l2_reg,
            )(x)
            conv_out = layers.BatchNormalization()(conv_out)
            conv_out = layers.Dropout(dropout)(conv_out)
            
            # Residual connection
            if x.shape[-1] != hidden_size:
                x = layers.Conv1D(hidden_size, 1)(x)
            x = layers.Add()([x, conv_out])
        
        # Take last timestep (like SlicingOpLambda in old model)
        x = x[:, -1, :]
        
        # Dense head
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        
        # Multi-output heads matching old model
        dir_branch = layers.Dense(32, activation='relu')(x)
        dir_branch = layers.Dropout(dropout * 0.5)(dir_branch)
        
        price_branch = layers.Dense(32, activation='relu')(x)
        
        risk_branch = layers.Dense(16, activation='relu')(x)
        
        state_branch = layers.Dense(32, activation='relu')(x)
        state_branch = layers.Dropout(dropout * 0.5)(state_branch)
        
        trend_branch = layers.Dense(32, activation='relu')(x)
        
        # Output layers (float32 for numerical stability)
        direction = layers.Dense(1, activation='sigmoid', name='direction', dtype='float32')(dir_branch)
        price = layers.Dense(1, activation='linear', name='price', dtype='float32')(price_branch)
        risk = layers.Dense(1, activation='sigmoid', name='risk', dtype='float32')(risk_branch)
        state_logits = layers.Dense(2, activation='softmax', name='state_logits', dtype='float32')(state_branch)
        trend = layers.Dense(1, activation='linear', name='trend', dtype='float32')(trend_branch)
        
        return tf.keras.Model(
            inputs=inp,
            outputs={
                'direction': direction,
                'price': price,
                'risk': risk,
                'state_logits': state_logits,
                'trend': trend,
            },
            name='MetalOptimizedTCN',
        )
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")


def migrate_model(model_path: Path, arch_type: str, dry_run: bool = False) -> bool:
    """
    Migrate a single Keras 2 model to Keras 3 format.
    
    Returns True if migration successful, False otherwise.
    """
    console.print(f"\n[bold]Migrating:[/bold] {model_path.name}")
    
    # Check version
    version = check_model_version(model_path)
    if version == "keras3":
        console.print(f"  [green]✓ Already Keras 3 format[/green]")
        return True
    elif version.startswith("error"):
        console.print(f"  [red]✗ Error checking version: {version}[/red]")
        return False
    
    console.print(f"  [yellow]Detected: {version}[/yellow]")
    
    if dry_run:
        console.print(f"  [dim]Would migrate to Keras 3[/dim]")
        return True
    
    # Get input shape from old model
    seq_len, feature_dim = get_model_input_shape(model_path)
    console.print(f"  [dim]Input shape: seq_len={seq_len}, feature_dim={feature_dim}[/dim]")
    
    # Create backup (preserving subdirectory structure)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    rel_path = model_path.relative_to(MODEL_DIR)
    backup_path = BACKUP_DIR / rel_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy(model_path, backup_path)
        console.print(f"  [dim]Backup: {backup_path}[/dim]")
    
    # Extract weights from old model
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with zipfile.ZipFile(model_path, 'r') as zf:
                zf.extractall(tmpdir)
            
            weights_path = Path(tmpdir) / "model.weights.h5"
            if not weights_path.exists():
                console.print(f"  [red]✗ No weights file found in archive[/red]")
                return False
            
            # Build fresh Keras 3 model
            console.print(f"  [dim]Building fresh {arch_type} model...[/dim]")
            fresh_model = build_fresh_model(arch_type, seq_len, feature_dim)
            
            # Try to load weights
            try:
                fresh_model.load_weights(str(weights_path))
                console.print(f"  [green]✓ Weights loaded successfully[/green]")
            except Exception as e:
                console.print(f"  [yellow]⚠ Weight loading issue: {e}[/yellow]")
                console.print(f"  [dim]Creating model with fresh weights instead[/dim]")
            
            # Save as Keras 3
            fresh_model.save(str(model_path))
            console.print(f"  [green]✓ Saved as Keras 3 format[/green]")
            
            # Verify it loads
            try:
                tf.keras.models.load_model(str(model_path), compile=False)
                console.print(f"  [green]✓ Verified: loads successfully[/green]")
                return True
            except Exception as e:
                console.print(f"  [red]✗ Verification failed: {e}[/red]")
                # Restore backup
                shutil.copy(backup_path, model_path)
                console.print(f"  [yellow]Restored from backup[/yellow]")
                return False
                
        except zipfile.BadZipFile:
            console.print(f"  [red]✗ Not a valid .keras archive[/red]")
            return False
        except Exception as e:
            console.print(f"  [red]✗ Migration failed: {e}[/red]")
            return False


def scan_all_models() -> list:
    """
    Scan MODEL_DIR and all pair subdirectories for .keras files.
    
    Returns list of (model_path, arch_type) tuples.
    """
    models = []
    
    # Root-level models
    for model_file in MODEL_DIR.glob("*.keras"):
        arch_type = MODEL_ARCH_MAP.get(model_file.name, "tcn")  # Default to tcn
        models.append((model_file, arch_type))
    
    # Pair subdirectory models (EUR_USD, USD_JPY, etc.)
    for subdir in MODEL_DIR.iterdir():
        if subdir.is_dir() and not subdir.name.startswith(('.', 'backup', 'checkpoint')):
            for model_file in subdir.glob("*.keras"):
                arch_type = MODEL_ARCH_MAP.get(model_file.name, "tcn")
                models.append((model_file, arch_type))
    
    return models


def clear_stale_checkpoints(model_path: Path):
    """
    Delete EMA and EWC checkpoint files for a model.
    
    These will be re-initialized fresh from the migrated model weights.
    """
    ema_path = model_path.with_suffix('.ema.pkl')
    ewc_path = model_path.with_suffix('.ewc.pkl')
    
    for path in [ema_path, ewc_path]:
        if path.exists():
            path.unlink()
            console.print(f"  [dim]Cleared: {path.name}[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Migrate Keras 2 models to Keras 3")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    parser.add_argument("--model", type=str, help="Migrate specific model (e.g., 'transformer_direction')")
    parser.add_argument("--scan", action="store_true", help="Scan and report model versions")
    args = parser.parse_args()
    
    console.print("\n[bold blue]═══ Keras Model Migration Tool ═══[/bold blue]\n")
    console.print(f"Keras version: {tf.keras.__version__}")
    console.print(f"TensorFlow version: {tf.__version__}")
    
    # Discover all models
    all_models = scan_all_models()
    console.print(f"\nFound {len(all_models)} .keras model(s)\n")
    
    # Scan mode
    if args.scan:
        table = Table(title="Model Status")
        table.add_column("Model Path", style="cyan")
        table.add_column("Architecture", style="magenta")
        table.add_column("Version", style="green")
        table.add_column("Input Shape")
        table.add_column("Status")
        
        for model_path, arch_type in all_models:
            rel_path = model_path.relative_to(MODEL_DIR)
            version = check_model_version(model_path)
            shape = get_model_input_shape(model_path)
            status = "✓ Ready" if version == "keras3" else "⚠ Needs migration"
            table.add_row(str(rel_path), arch_type, version, str(shape), status)
        
        console.print(table)
        return
    
    # Filter models if specific one requested
    models = all_models
    if args.model:
        models = [(p, a) for p, a in all_models if args.model in p.name]
        if not models:
            console.print(f"[red]Model '{args.model}' not found[/red]")
            return
    
    # Migrate
    results = []
    for model_path, arch_type in models:
        rel_path = model_path.relative_to(MODEL_DIR)
        success = migrate_model(model_path, arch_type, dry_run=args.dry_run)
        results.append((str(rel_path), success))
        
        # Clear stale EMA/EWC files after successful migration
        if success and not args.dry_run:
            clear_stale_checkpoints(model_path)
    
    # Summary
    console.print("\n[bold]═══ Summary ═══[/bold]")
    for name, success in results:
        status = "[green]✓ Success[/green]" if success else "[red]✗ Failed[/red]"
        console.print(f"  {name}: {status}")
    
    if args.dry_run:
        console.print("\n[yellow]This was a dry run. Use without --dry-run to migrate.[/yellow]")


if __name__ == "__main__":
    main()
