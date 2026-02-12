"""
Training display utilities using Rich for clean output.

This module contains:
- TrainingDisplay: Clean, professional training output using Rich library
"""

from __future__ import annotations


class TrainingDisplay:
    """Clean, professional training output using Rich."""

    def __init__(self, model_name: str = "Model"):
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        self.console = Console()
        self.model_name = model_name
        self._Table = Table
        self._Panel = Panel
        self._Progress = Progress
        self._SpinnerColumn = SpinnerColumn
        self._TextColumn = TextColumn
        self._BarColumn = BarColumn
        self._TaskProgressColumn = TaskProgressColumn

    def show_config(self, config: dict):
        """Display configuration as a clean table."""
        table = self._Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="dim")
        table.add_column("Value", style="cyan")
        for k, v in config.items():
            table.add_row(str(k), str(v))
        self.console.print(self._Panel(table, title=f"[bold]{self.model_name}[/bold]", border_style="blue"))

    def show_summary(self, metrics: dict, title: str = "Results"):
        """Display training results summary."""
        table = self._Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="green")
        for k, v in metrics.items():
            if isinstance(v, float):
                table.add_row(str(k), f"{v:.4f}")
            else:
                table.add_row(str(k), str(v))
        self.console.print(self._Panel(table, title=f"[bold]{title}[/bold]", border_style="green"))

    def status(self, message: str, style: str = ""):
        """Print a status message."""
        self.console.print(f"  {message}", style=style)

    def warn(self, message: str):
        """Print a warning message."""
        self.console.print(f"  [yellow]⚠ {message}[/yellow]")

    def error(self, message: str):
        """Print an error message."""
        self.console.print(f"  [red]✗ {message}[/red]")

    def success(self, message: str):
        """Print a success message."""
        self.console.print(f"  [green]✓ {message}[/green]")
