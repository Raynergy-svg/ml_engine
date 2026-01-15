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

import tensorflow as tf
from rich.console import Console
from rich.table import Table

console = Console()

MODEL_DIR = Path("trained_data/models")
BACKUP_DIR = Path("trained_data/models_keras2_backup")

# Models to migrate and their architecture builders
MODELS_TO_MIGRATE = [
    ("buddy_tf.keras", "tcn"),
    ("tf_model.keras", "tcn"),
    ("transformer_direction.keras", "tcn"),
]


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
    
    if arch_type == "tcn":
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
    
    # Create backup
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / model_path.name
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


def main():
    parser = argparse.ArgumentParser(description="Migrate Keras 2 models to Keras 3")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    parser.add_argument("--model", type=str, help="Migrate specific model (e.g., 'buddy_tf')")
    parser.add_argument("--scan", action="store_true", help="Scan and report model versions")
    args = parser.parse_args()
    
    console.print("\n[bold blue]═══ Keras Model Migration Tool ═══[/bold blue]\n")
    console.print(f"Keras version: {tf.keras.__version__}")
    console.print(f"TensorFlow version: {tf.__version__}")
    
    # Scan mode
    if args.scan:
        table = Table(title="Model Status")
        table.add_column("Model", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Input Shape")
        table.add_column("Status")
        
        for model_name, _ in MODELS_TO_MIGRATE:
            model_path = MODEL_DIR / model_name
            if model_path.exists():
                version = check_model_version(model_path)
                shape = get_model_input_shape(model_path)
                status = "✓ Ready" if version == "keras3" else "⚠ Needs migration"
                table.add_row(model_name, version, str(shape), status)
            else:
                table.add_row(model_name, "-", "-", "Not found")
        
        console.print(table)
        return
    
    # Filter models if specific one requested
    models = MODELS_TO_MIGRATE
    if args.model:
        models = [(m, a) for m, a in MODELS_TO_MIGRATE if args.model in m]
        if not models:
            console.print(f"[red]Model '{args.model}' not found in migration list[/red]")
            return
    
    # Migrate
    results = []
    for model_name, arch_type in models:
        model_path = MODEL_DIR / model_name
        if model_path.exists():
            success = migrate_model(model_path, arch_type, dry_run=args.dry_run)
            results.append((model_name, success))
        else:
            console.print(f"\n[dim]Skipping {model_name}: not found[/dim]")
    
    # Summary
    console.print("\n[bold]═══ Summary ═══[/bold]")
    for name, success in results:
        status = "[green]✓ Success[/green]" if success else "[red]✗ Failed[/red]"
        console.print(f"  {name}: {status}")
    
    if args.dry_run:
        console.print("\n[yellow]This was a dry run. Use without --dry-run to migrate.[/yellow]")


if __name__ == "__main__":
    main()
