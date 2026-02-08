#!/usr/bin/env python3
"""Click-based monitoring command for the Buddy trading bot."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import click
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.io_utils import DEFAULT_CONFIG_PATH, console
from src.utils import load_config
from src.utils.monitoring import (
    Alert,
    AlertLevel,
    ModelDriftMetrics,
    MonitoringSystem,
    create_monitoring_report,
)

_FALLBACK_CONFIG: Dict[str, Any] = {"paths": {"log_dir": "trained_data/logs"}}
try:
    from yaml import YAMLError
except ModuleNotFoundError:
    YAMLError: Type[Exception] | None = None


@contextmanager
def _silence_logging() -> Iterable[None]:
    """Temporarily silence logging output for pristine CLI rendering."""
    previous_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_level)


def _format_timestamp(timestamp: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format ISO timestamps for display."""
    if not timestamp:
        return "N/A"
    normalized = str(timestamp).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).strftime(fmt)
    except ValueError:
        return str(timestamp)


def _alert_label(level: AlertLevel) -> Tuple[str, str]:
    """Return formatted label and color for alert levels."""
    if level == AlertLevel.CRITICAL:
        return "CRITICAL", "red"
    if level == AlertLevel.WARNING:
        return "WARNING", "yellow"
    return "INFO", "cyan"


def _safe_load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration or fall back to a minimal default."""
    try:
        return load_config(config_path)
    except (FileNotFoundError, ModuleNotFoundError, ValueError, TypeError, OSError):
        return dict(_FALLBACK_CONFIG)
    except Exception as exc:
        if YAMLError is not None and isinstance(exc, YAMLError):
            return dict(_FALLBACK_CONFIG)
        raise


def _build_monitor(config_path: str) -> Optional[MonitoringSystem]:
    """Create a MonitoringSystem instance with safe fallbacks."""
    config = _safe_load_config(config_path)
    try:
        return MonitoringSystem(config)
    except (OSError, ValueError, TypeError):
        try:
            return MonitoringSystem(dict(_FALLBACK_CONFIG))
        except (OSError, ValueError, TypeError):
            return None


def _render_alert_summary(alerts: Dict[str, int]) -> Panel:
    """Render the alert summary section."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Level", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("Critical", str(alerts.get("critical", 0)), style="red")
    table.add_row("Warning", str(alerts.get("warning", 0)), style="yellow")
    table.add_row("Info", str(alerts.get("info", 0)), style="cyan")
    table.add_row("Total", str(alerts.get("total", 0)), style="bold")
    return Panel(table, title="Alert Summary", border_style="yellow")


def _render_recent_alerts(alerts: Sequence[Alert]) -> Panel:
    """Render the recent alerts section."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Time", style="dim", width=19)
    table.add_column("Level", style="bold", width=9)
    table.add_column("Category", style="bold")
    table.add_column("Message", overflow="fold")

    if not alerts:
        table.add_row("—", "—", "—", "No recent alerts.")
        return Panel(table, title="Recent Alerts", border_style="blue")

    for alert in alerts:
        label, color = _alert_label(alert.level)
        table.add_row(
            _format_timestamp(alert.timestamp, fmt="%H:%M:%S"),
            f"[{color}]{label}[/{color}]",
            alert.category.replace("_", " ").title(),
            alert.message,
        )
    return Panel(table, title="Recent Alerts", border_style="blue")


def _render_model_health(recent_drift: Sequence[Dict[str, Any]]) -> Panel:
    """Render the model health section."""
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    if recent_drift:
        latest = recent_drift[-1]
        status_label = "Drift Detected" if latest.get("alert_triggered") else "Healthy"
        status_color = "red" if latest.get("alert_triggered") else "green"
        table.add_row("Status", f"[{status_color}]{status_label}[/{status_color}]")
        table.add_row(
            "Confidence Mean",
            f"{latest.get('confidence_mean', 0.0):.3f}",
        )
        table.add_row(
            "Drift Score",
            f"{latest.get('drift_score', 0.0):.3f}",
        )
    else:
        table.add_row("Status", "No drift data")
        table.add_row("Confidence Mean", "N/A")
        table.add_row("Drift Score", "N/A")

    return Panel(table, title="Model Health", border_style="cyan")


def _render_performance_baseline(baseline: Optional[Dict[str, Any]]) -> Panel:
    """Render the performance baseline section."""
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    if baseline and baseline.get("win_rate") is not None:
        win_rate = float(baseline.get("win_rate", 0.0))
        sharpe = float(baseline.get("sharpe_ratio", 0.0))
        table.add_row("Win Rate", f"{win_rate:.1%}")
        table.add_row("Sharpe Ratio", f"{sharpe:.2f}")
    else:
        table.add_row("Win Rate", "N/A")
        table.add_row("Sharpe Ratio", "N/A")

    return Panel(table, title="Performance Baseline", border_style="green")


def _render_drift_history(history: Sequence[ModelDriftMetrics], limit: int) -> None:
    """Render the model drift history table."""
    if not history:
        console.print(Panel("No drift history available.", title="Model Drift"))
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Timestamp", style="dim", width=19)
    table.add_column("Confidence", justify="right")
    table.add_column("Drift Score", justify="right")
    table.add_column("Status", justify="center")

    for metrics in history[-limit:]:
        status_label = "Alert" if metrics.alert_triggered else "OK"
        status_color = "red" if metrics.alert_triggered else "green"
        table.add_row(
            _format_timestamp(metrics.timestamp, fmt="%Y-%m-%d %H:%M"),
            f"{metrics.confidence_mean:.3f}",
            f"{metrics.feature_drift_score:.3f}",
            f"[{status_color}]{status_label}[/{status_color}]",
        )

    console.print(Panel(table, title="Model Drift History", border_style="cyan"))


def _render_alert_details(alerts: Sequence[Alert]) -> None:
    """Render detailed categorized alerts."""
    if not alerts:
        console.print(
            Panel("No alerts available.", title="Alerts", border_style="green")
        )
        return

    grouped: Dict[str, List[Alert]] = defaultdict(list)
    for alert in alerts:
        grouped[alert.category].append(alert)

    for category, category_alerts in sorted(grouped.items()):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Time", style="dim", width=19)
        table.add_column("Level", style="bold", width=9)
        table.add_column("Message", overflow="fold")
        for alert in category_alerts:
            label, color = _alert_label(alert.level)
            table.add_row(
                _format_timestamp(alert.timestamp),
                f"[{color}]{label}[/{color}]",
                alert.message,
            )

        title = category.replace("_", " ").title()
        console.print(Panel(table, title=title, border_style="yellow"))


def _render_dashboard(monitor: MonitoringSystem) -> None:
    """Render the default monitoring dashboard."""
    data = monitor.get_dashboard_data()
    last_updated = _format_timestamp(data.get("timestamp", ""))
    header = Text("FX Trading Bot Monitoring", style="bold cyan")
    console.print(Panel(header, subtitle=f"Last Updated: {last_updated}"))

    alert_summary = _render_alert_summary(data.get("alerts", {}))
    model_health = _render_model_health(data.get("recent_drift", []))
    console.print(Columns([alert_summary, model_health]))

    recent_alerts = _render_recent_alerts(monitor.get_alerts()[-5:])
    baseline = _render_performance_baseline(data.get("baseline_metrics"))
    console.print(Columns([recent_alerts, baseline]))


def run_monitoring_dashboard(
    config_path: str,
    *,
    show_alerts: bool = False,
    show_drift: bool = False,
    generate_report: bool = False,
    drift_limit: int = 20,
) -> None:
    """Run the monitoring dashboard with robust error handling."""
    with _silence_logging():
        try:
            monitor = _build_monitor(config_path)
            if monitor is None:
                console.print(Panel("Monitoring data unavailable.", title="Monitoring"))
                return

            if generate_report:
                report_path = create_monitoring_report(monitor)
                console.print(
                    Panel(
                        f"Report saved to: {report_path}",
                        title="Monitoring Report",
                        border_style="green",
                    )
                )
                return

            if show_alerts:
                _render_alert_details(monitor.get_alerts())
                return

            if show_drift:
                _render_drift_history(monitor.drift_history, drift_limit)
                return

            _render_dashboard(monitor)
        except (OSError, ValueError, TypeError, RuntimeError):
            console.print(Panel("Monitoring data unavailable.", title="Monitoring"))


@click.group(help="Buddy monitoring commands.")
def buddy() -> None:
    """Buddy CLI command group."""


@buddy.command("monitor")
@click.option(
    "--config",
    "config_path",
    default=DEFAULT_CONFIG_PATH,
    show_default=True,
    help="Path to the config file.",
)
@click.option("--alerts", "show_alerts", is_flag=True, help="Show detailed alerts.")
@click.option("--drift", "show_drift", is_flag=True, help="Show drift history.")
@click.option("--report", "generate_report", is_flag=True, help="Generate full report.")
@click.option(
    "--limit",
    "drift_limit",
    type=int,
    default=20,
    show_default=True,
    help="Limit drift history entries.",
)
def monitor(
    config_path: str,
    show_alerts: bool,
    show_drift: bool,
    generate_report: bool,
    drift_limit: int,
) -> None:
    """Display monitoring data for the trading system."""
    run_monitoring_dashboard(
        config_path,
        show_alerts=show_alerts,
        show_drift=show_drift,
        generate_report=generate_report,
        drift_limit=drift_limit,
    )


def main() -> None:
    """Entry point for the Buddy monitoring CLI."""
    buddy()


__all__ = ["buddy", "main", "monitor", "run_monitoring_dashboard"]
