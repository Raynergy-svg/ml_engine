#!/usr/bin/env python3
"""
Standalone RL Position Sizer Training Script

This script runs in a separate process to avoid TensorFlow/PyTorch GPU conflicts.
It expects training data to be passed via a .npz file.

Usage:
    python train_rl_standalone.py --data rl_train_data.npz --timesteps 100000
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Rich for beautiful output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def create_progress_callback(total_timesteps: int, progress: Progress, task_id):
    """
    Create a proper SB3 BaseCallback for tracking progress.
    
    Must be defined inside function to import BaseCallback lazily.
    """
    from stable_baselines3.common.callbacks import BaseCallback
    
    class PPOProgressCallback(BaseCallback):
        """Callback to track PPO training progress with Rich progress bar."""
        
        def __init__(self, total_steps: int, prog: Progress, tid, verbose=0):
            super().__init__(verbose)
            self.total_steps = total_steps
            self.prog = prog
            self.tid = tid
            self.last_update = 0
            self.update_interval = max(500, total_steps // 100)
        
        def _on_step(self) -> bool:
            """Called after each step."""
            if self.num_timesteps - self.last_update >= self.update_interval:
                self.prog.update(self.tid, completed=self.num_timesteps)
                self.last_update = self.num_timesteps
            return True
        
        def _on_training_end(self) -> None:
            """Called at the end of training."""
            self.prog.update(self.tid, completed=self.total_steps)
    
    return PPOProgressCallback(total_timesteps, progress, task_id)


def main():
    parser = argparse.ArgumentParser(description="Train RL Position Sizer (standalone)")
    parser.add_argument("--data", required=True, help="Path to training data .npz file")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # ═══════════════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════════════
    console.print()
    console.print(Panel.fit(
        "[bold cyan]RL Position Sizer Training[/bold cyan]\n"
        "[dim]PPO Agent • Subprocess Mode[/dim]",
        border_style="cyan"
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHECKLIST TABLE
    # ═══════════════════════════════════════════════════════════════════════
    steps = [
        ["⏳", "Load training data", ""],
        ["⏳", "Import RL components", ""],
        ["⏳", "Configure PPO agent", ""],
        ["⏳", "Train PPO agent", ""],
        ["⏳", "Save model", ""],
        ["⏳", "Test inference", ""],
    ]
    
    def print_checklist():
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("", width=2)
        table.add_column("", width=25)
        table.add_column("", style="dim")
        for s in steps:
            clr = "green" if s[0] == "✓" else ("yellow" if s[0] == "⏳" else "red")
            table.add_row(f"[{clr}]{s[0]}[/{clr}]", s[1], s[2])
        console.print(table)
    
    stats = {}
    train_time = 0
    
    try:
        # Step 1: Load Data
        data = np.load(args.data)
        features = data['features']
        predictions = data['predictions']
        prices = data['prices']
        steps[0] = ["✓", "Load training data", f"{len(features):,} samples"]
        
        # Step 2: Import RL
        from rl_position_sizing import RLPositionSizer, RLConfig, GYM_AVAILABLE, SB3_AVAILABLE
        
        if not GYM_AVAILABLE or not SB3_AVAILABLE:
            steps[1] = ["✗", "Import RL components", "Missing deps"]
            print_checklist()
            sys.exit(1)
        steps[1] = ["✓", "Import RL components", "gymnasium + SB3"]
        
        # Step 3: Configure
        config = RLConfig(
            total_timesteps=args.timesteps,
            max_drawdown_pct=0.10,
            max_position_pct=0.05,
        )
        sizer = RLPositionSizer(config)
        steps[2] = ["✓", "Configure PPO agent", f"{args.timesteps:,} steps"]
        
        # Print checklist before training
        console.print()
        print_checklist()
        console.print()
        
        # Step 4: Train with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]Training PPO[/cyan]"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("", total=args.timesteps)
            callback = create_progress_callback(args.timesteps, progress, task)
            
            train_start = time.time()
            stats = sizer.train(
                features=features,
                ensemble_predictions=predictions,
                prices=prices,
                verbose=0,
                callback=callback,
            )
            train_time = time.time() - train_start
            progress.update(task, completed=args.timesteps)
        
        steps[3] = ["✓", "Train PPO agent", f"{train_time:.1f}s"]
        
        # Step 5: Save
        model_dir = Path("trained_data/models")
        meta = {
            "trained_at": datetime.now().isoformat(),
            "timesteps": args.timesteps,
            "training_time_seconds": train_time,
            "training_samples": len(features),
            "stats": stats,
            "config": {
                "total_timesteps": config.total_timesteps,
                "max_drawdown_pct": config.max_drawdown_pct,
                "max_position_pct": config.max_position_pct,
            },
        }
        meta_path = model_dir / "rl_position_sizer.meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        steps[4] = ["✓", "Save model", "rl_position_sizer.zip"]
        
        # Step 6: Test
        test_features = features[-10:].mean(axis=0)
        test_predictions = np.array([0.65, 0.7])
        position_size = sizer.get_position_size(
            features=test_features,
            ensemble_prediction=test_predictions,
            account_equity=10000.0,
        )
        steps[5] = ["✓", "Test inference", f"${position_size:,.0f} / $10k"]
        
        # ═══════════════════════════════════════════════════════════════════
        # RESULTS
        # ═══════════════════════════════════════════════════════════════════
        console.print()
        
        results = Table(title="[bold]Results[/bold]", box=box.ROUNDED, border_style="green")
        results.add_column("Metric", style="bold")
        results.add_column("Value", justify="right")
        results.add_row("Training Time", f"{train_time:.1f}s")
        results.add_row("Timesteps", f"{args.timesteps:,}")
        results.add_row("Samples", f"{len(features):,}")
        results.add_row("Final Equity", f"${stats.get('final_equity', 10000):,.2f}")
        results.add_row("Trades", f"{stats.get('total_trades', 0):,}")
        results.add_row("Win Rate", f"{stats.get('win_rate', 0):.1%}")
        results.add_row("Max Drawdown", f"{stats.get('max_drawdown', 0):.1%}")
        console.print(results)
        
        console.print()
        console.print("[bold green]✓ Complete[/bold green] — Model saved to trained_data/models/")
        console.print()
        
    except Exception as e:
        console.print()
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
