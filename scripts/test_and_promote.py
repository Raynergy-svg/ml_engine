#!/usr/bin/env python3
"""
Test both candidate and production models, then promote the better one.
"""
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def load_meta(path):
    """Load model metadata."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def test_model(meta_path, model_name):
    """Extract test metrics from metadata."""
    meta = load_meta(meta_path)
    if not meta:
        return None
    
    return {
        "name": model_name,
        "accuracy": meta.get("val_direction_accuracy", 0) * 100,
        "combined_score": meta.get("val_combined_score", 0),
        "confidence": meta.get("val_avg_confidence", 0),
        "epochs": meta.get("trained_epochs", 0),
        "best_epoch": meta.get("best_epoch", 0),
        "path": str(meta_path),
    }


def compare_and_promote():
    """Compare models and promote the better one."""
    models_dir = Path("trained_data/models")
    
    candidate_meta = models_dir / "buddy_tf_candidate.meta.json"
    production_meta = models_dir / "buddy_tf.meta.json"
    
    console.print("\n[bold cyan]🔍 TESTING AND COMPARING MODELS[/bold cyan]\n")
    
    # Load both models
    candidate = test_model(candidate_meta, "Candidate")
    production = test_model(production_meta, "Production")
    
    if not candidate and not production:
        console.print("[red]No models found![/red]")
        return
    
    # Display comparison table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    if candidate:
        table.add_column("Candidate", justify="right")
    if production:
        table.add_column("Production", justify="right")
    
    if candidate and production:
        table.add_row(
            "Direction Accuracy",
            f"{candidate['accuracy']:.2f}%",
            f"{production['accuracy']:.2f}%"
        )
        table.add_row(
            "Combined Score",
            f"{candidate['combined_score']:.4f}",
            f"{production['combined_score']:.4f}"
        )
        table.add_row(
            "Avg Confidence",
            f"{candidate['confidence']:.4f}",
            f"{production['confidence']:.4f}"
        )
        table.add_row(
            "Trained Epochs",
            str(candidate['epochs']),
            str(production['epochs'])
        )
        table.add_row(
            "Best Epoch",
            str(candidate['best_epoch']),
            str(production['best_epoch'])
        )
        
        console.print(table)
        console.print()
        
        # Determine winner
        candidate_score = candidate['accuracy'] * 0.6 + candidate['combined_score'] * 100 * 0.4
        production_score = production['accuracy'] * 0.6 + production['combined_score'] * 100 * 0.4
        
        if candidate_score > production_score:
            winner = candidate
            console.print("[green]✓ Candidate model is better![/green]")
            console.print(f"  Score: {candidate_score:.2f} vs {production_score:.2f}")
        else:
            winner = production
            console.print("[green]✓ Production model is better![/green]")
            console.print(f"  Score: {production_score:.2f} vs {candidate_score:.2f}")
        
        # If production is better, we should promote it (copy to candidate)
        if winner['name'] == "Production":
            console.print("\n[cyan]Promoting production model to candidate...[/cyan]")
            
            # Copy production model files to candidate
            import shutil
            
            prod_model = models_dir / "buddy_tf.keras"
            cand_model = models_dir / "buddy_tf_candidate.keras"
            
            if prod_model.exists():
                shutil.copy2(prod_model, cand_model)
                console.print(f"[green]✓ Copied {prod_model.name} → {cand_model.name}[/green]")
            
            # Copy metadata
            shutil.copy2(production_meta, candidate_meta)
            console.print(f"[green]✓ Copied {production_meta.name} → {candidate_meta.name}[/green]")
            
            console.print("\n[bold green]✅ Production model promoted to candidate![/bold green]")
            console.print("[dim]Buddy will now use the better model.[/dim]\n")
        else:
            console.print("\n[green]Candidate model is already the best - no promotion needed.[/green]\n")
    
    elif candidate:
        console.print("[yellow]Only candidate model found.[/yellow]")
        console.print(f"  Accuracy: {candidate['accuracy']:.2f}%")
        console.print(f"  Combined Score: {candidate['combined_score']:.4f}\n")
    
    elif production:
        console.print("[yellow]Only production model found.[/yellow]")
        console.print(f"  Accuracy: {production['accuracy']:.2f}%")
        console.print(f"  Combined Score: {production['combined_score']:.4f}\n")


if __name__ == "__main__":
    compare_and_promote()
