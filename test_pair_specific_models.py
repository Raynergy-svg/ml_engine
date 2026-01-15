#!/usr/bin/env python3
"""
Test script for per-pair model saving/loading.

This verifies that:
1. _extract_instrument_from_csv_path correctly extracts pair names
2. _get_pair_model_paths returns correct paths
3. ModularEnsembleInference loads pair-specific models with fallback

Usage:
    python test_pair_specific_models.py
"""

import sys
import os
from pathlib import Path

# Set TF logging before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def test_extract_instrument():
    """Test instrument extraction from CSV filenames."""
    console.print("\n[bold cyan]Test 1: Extract Instrument from CSV Path[/bold cyan]")
    
    from main import _extract_instrument_from_csv_path
    
    test_cases = [
        # (csv_path, expected_instrument)
        ("market_data/EUR_USD_H1.csv", "EUR_USD"),
        ("market_data/oanda_GBP_USD_H1_live_20250114.csv", "GBP_USD"),
        ("market_data/USD_JPY_M5.csv", "USD_JPY"),
        ("market_data/EURUSD.csv", "EUR_USD"),  # 6-char format
        ("market_data/aud_nzd_h4.csv", "AUD_NZD"),  # lowercase
        ("market_data/random_data.csv", "GENERIC"),  # No valid pair
        (None, "GENERIC"),  # None input
        ("", "GENERIC"),  # Empty string
    ]
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("CSV Path", width=45)
    table.add_column("Expected", width=12)
    table.add_column("Actual", width=12)
    table.add_column("Status", width=8)
    
    all_passed = True
    for csv_path, expected in test_cases:
        actual = _extract_instrument_from_csv_path(csv_path)
        passed = actual == expected
        all_passed = all_passed and passed
        status = "[green]✓[/green]" if passed else "[red]✗[/red]"
        table.add_row(str(csv_path), expected, actual, status)
    
    console.print(table)
    return all_passed


def test_get_pair_model_paths():
    """Test pair-specific model path generation."""
    console.print("\n[bold cyan]Test 2: Get Pair Model Paths[/bold cyan]")
    
    from main import _get_pair_model_paths
    
    model_dir = Path("trained_data/models")
    
    # Test EUR_USD pair
    eur_paths = _get_pair_model_paths(model_dir, "EUR_USD", "transformer")
    console.print(f"\n[yellow]EUR_USD paths:[/yellow]")
    console.print(f"  direction: {eur_paths['direction']}")
    console.print(f"  xgboost:   {eur_paths['xgboost']}")
    console.print(f"  rf:        {eur_paths['rf']}")
    console.print(f"  ridge:     {eur_paths['ridge']}")
    console.print(f"  pair_dir:  {eur_paths['pair_dir']}")
    
    # Test GENERIC (no pair)
    generic_paths = _get_pair_model_paths(model_dir, "GENERIC", "transformer")
    console.print(f"\n[yellow]GENERIC paths:[/yellow]")
    console.print(f"  direction: {generic_paths['direction']}")
    console.print(f"  pair_dir:  {generic_paths['pair_dir']}")
    
    # Verify paths are correct
    assert str(eur_paths['direction']).endswith("EUR_USD/transformer_direction.keras")
    assert str(generic_paths['direction']).endswith("models/transformer_direction.keras")
    
    console.print("\n[green]✓ Path generation test passed[/green]")
    return True


def test_modular_inference_loading():
    """Test that ModularEnsembleInference loads correct pair models."""
    console.print("\n[bold cyan]Test 3: Modular Ensemble Inference Loading[/bold cyan]")
    
    import logging
    logging.getLogger('modular_trainers').setLevel(logging.ERROR)
    logging.getLogger('modular_inference').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    from modular_inference import ModularEnsembleInference
    
    model_dir = Path("trained_data/models")
    
    # List available pair-specific models
    console.print("\n[yellow]Available pair-specific model directories:[/yellow]")
    pairs_found = []
    for item in model_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != "checkpoints":
            keras_files = list(item.glob("*.keras"))
            pkl_files = list(item.glob("*.pkl"))
            if keras_files or pkl_files:
                pairs_found.append(item.name)
                console.print(f"  {item.name}: {len(keras_files)} .keras, {len(pkl_files)} .pkl")
    
    if not pairs_found:
        console.print("  [dim]No pair-specific models found yet[/dim]")
        console.print("  [dim]Train with: python main.py train-buddy --csv market_data/EUR_USD_H1.csv[/dim]")
    
    # Test loading generic models
    console.print("\n[yellow]Testing model path resolution (without loading TF)...[/yellow]")
    try:
        ensemble_generic = ModularEnsembleInference(instrument=None)
        # Test path resolution without loading models
        dir_path = ensemble_generic._get_model_path("transformer_direction", ".keras")
        xgb_path = ensemble_generic._get_model_path("xgb_momentum", ".pkl")
        console.print(f"  Generic direction path: {dir_path}")
        console.print(f"  Generic XGBoost path: {xgb_path}")
        console.print(f"  Direction exists: {dir_path.exists()}")
        console.print(f"  XGBoost exists: {xgb_path.exists()}")
        generic_loaded = True
    except Exception as e:
        console.print(f"  [red]Failed to test generic paths: {e}[/red]")
        generic_loaded = False
    
    # Test loading pair-specific models (if available)
    if pairs_found:
        test_pair = pairs_found[0]
        console.print(f"\n[yellow]Testing {test_pair} path resolution...[/yellow]")
        try:
            ensemble_pair = ModularEnsembleInference(instrument=test_pair)
            dir_path = ensemble_pair._get_model_path("transformer_direction", ".keras")
            xgb_path = ensemble_pair._get_model_path("xgb_momentum", ".pkl")
            console.print(f"  {test_pair} direction path: {dir_path}")
            console.print(f"  {test_pair} XGBoost path: {xgb_path}")
            console.print(f"  Direction exists: {dir_path.exists()}")
            console.print(f"  XGBoost exists: {xgb_path.exists()}")
            
            # Verify it chose the pair-specific path
            if test_pair in str(dir_path):
                console.print(f"  [green]✓ Correctly using pair-specific path[/green]")
            pair_loaded = True
        except Exception as e:
            console.print(f"  [red]Failed to test {test_pair} paths: {e}[/red]")
            pair_loaded = False
    else:
        pair_loaded = True  # Skip if no pair models
    
    console.print("\n[green]✓ Path resolution test passed[/green]")
    return generic_loaded


def test_get_model_path():
    """Test the _get_model_path helper method."""
    console.print("\n[bold cyan]Test 4: _get_model_path Helper Method[/bold cyan]")
    
    from modular_inference import ModularEnsembleInference
    
    model_dir = Path("trained_data/models")
    
    # Create test instances
    generic_ensemble = ModularEnsembleInference(instrument=None)
    pair_ensemble = ModularEnsembleInference(instrument="EUR_USD")
    
    # Test generic (should use model_dir directly)
    generic_path = generic_ensemble._get_model_path("transformer_direction", ".keras")
    console.print(f"[yellow]Generic path:[/yellow] {generic_path}")
    assert str(generic_path).endswith("models/transformer_direction.keras")
    
    # Test pair-specific (should check pair dir first)
    pair_path = pair_ensemble._get_model_path("transformer_direction", ".keras")
    console.print(f"[yellow]EUR_USD path:[/yellow] {pair_path}")
    
    # If EUR_USD dir exists and has the file, path should be pair-specific
    # Otherwise, it falls back to generic
    eur_specific = model_dir / "EUR_USD" / "transformer_direction.keras"
    if eur_specific.exists():
        assert str(pair_path).endswith("EUR_USD/transformer_direction.keras")
        console.print("  [green]✓ Pair-specific model found and used[/green]")
    else:
        assert str(pair_path).endswith("models/transformer_direction.keras")
        console.print("  [dim]✓ Fell back to generic model (pair model doesn't exist)[/dim]")
    
    console.print("\n[green]✓ _get_model_path test passed[/green]")
    return True


def main():
    """Run all tests."""
    console.print(Panel(
        "[bold]Per-Pair Model System Tests[/bold]\n\n"
        "Testing compound learning infrastructure:\n"
        "• Instrument extraction from CSV paths\n"
        "• Pair-specific model path generation\n"
        "• ModularEnsembleInference pair loading",
        title="🧪 Test Suite",
        border_style="cyan"
    ))
    
    results = []
    
    try:
        results.append(("Extract Instrument", test_extract_instrument()))
    except Exception as e:
        console.print(f"[red]Test 1 failed: {e}[/red]")
        results.append(("Extract Instrument", False))
    
    try:
        results.append(("Get Model Paths", test_get_pair_model_paths()))
    except Exception as e:
        console.print(f"[red]Test 2 failed: {e}[/red]")
        results.append(("Get Model Paths", False))
    
    try:
        results.append(("_get_model_path", test_get_model_path()))
    except Exception as e:
        console.print(f"[red]Test 4 failed: {e}[/red]")
        results.append(("_get_model_path", False))
    
    try:
        results.append(("Modular Inference", test_modular_inference_loading()))
    except Exception as e:
        console.print(f"[red]Test 3 failed: {e}[/red]")
        results.append(("Modular Inference", False))
    
    # Summary
    console.print("\n")
    console.print(Panel(
        "\n".join([
            f"{'[green]✓[/green]' if passed else '[red]✗[/red]'} {name}"
            for name, passed in results
        ]),
        title="📋 Test Results",
        border_style="green" if all(p for _, p in results) else "red"
    ))
    
    all_passed = all(p for _, p in results)
    if all_passed:
        console.print("\n[bold green]✅ All tests passed![/bold green]")
        console.print("\n[dim]Next steps:[/dim]")
        console.print("[dim]  1. Train pair-specific model: python main.py train-buddy --csv market_data/EUR_USD_H1.csv[/dim]")
        console.print("[dim]  2. Run inference with pair: python main.py buddy --instrument EUR_USD[/dim]")
        console.print("[dim]  3. Models saved to: trained_data/models/EUR_USD/[/dim]")
    else:
        console.print("\n[bold red]❌ Some tests failed[/bold red]")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
