#!/usr/bin/env python3
"""Click-based monitoring command for the Buddy trading bot."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
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


def _fetch_oanda_account_data() -> Optional[Dict[str, Any]]:
    """Fetch live OANDA account data with error handling."""
    try:
        from src.utils.oanda_practice import OandaPracticeClient
        client = OandaPracticeClient.from_env()
        summary = client.get_account_summary()
        trades = client.get_trades(state="CLOSED", count=100)
        open_trades = client.get_trades(state="OPEN", count=50)
        return {
            "account": summary.get("account", {}),
            "closed_trades": trades.get("trades", []),
            "open_trades": open_trades.get("trades", []),
        }
    except Exception:
        return None


def _calculate_trade_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate performance metrics from closed trades."""
    if not trades:
        return {"win_rate": None, "total_pnl": 0, "total_trades": 0}

    total_trades = len(trades)
    winning = 0
    total_pnl = 0.0
    returns = []

    for trade in trades:
        pnl = float(trade.get("realizedPL", 0))
        total_pnl += pnl
        if pnl > 0:
            winning += 1
        # Calculate return percentage for Sharpe
        initial = abs(float(trade.get("initialUnits", 1)))
        price = float(trade.get("price", 1))
        if initial > 0 and price > 0:
            trade_value = initial * price
            if trade_value > 0:
                returns.append(pnl / trade_value)

    win_rate = winning / total_trades if total_trades > 0 else 0

    # Simple Sharpe approximation (annualized assuming daily trades)
    import numpy as np
    sharpe = 0.0
    if len(returns) > 1:
        returns_arr = np.array(returns)
        mean_ret = np.mean(returns_arr)
        std_ret = np.std(returns_arr)
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(252)  # Annualized

    return {
        "win_rate": win_rate,
        "sharpe_ratio": sharpe,
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "winning_trades": winning,
        "losing_trades": total_trades - winning,
    }


def _render_oanda_account(account: Dict[str, Any]) -> Panel:
    """Render OANDA account summary panel."""
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    balance = float(account.get("balance", 0))
    nav = float(account.get("NAV", balance))
    unrealized_pl = float(account.get("unrealizedPL", 0))
    margin_used = float(account.get("marginUsed", 0))
    margin_avail = float(account.get("marginAvailable", 0))
    open_trades = int(account.get("openTradeCount", 0))

    # Color for P&L
    pl_color = "green" if unrealized_pl >= 0 else "red"

    table.add_row("Balance", f"${balance:,.2f}")
    table.add_row("NAV", f"${nav:,.2f}")
    table.add_row("Unrealized P&L", f"[{pl_color}]${unrealized_pl:+,.2f}[/{pl_color}]")
    table.add_row("Margin Used", f"${margin_used:,.2f}")
    table.add_row("Margin Available", f"${margin_avail:,.2f}")
    table.add_row("Open Trades", str(open_trades))

    return Panel(table, title="OANDA Account", border_style="cyan")


def _render_trade_performance(metrics: Dict[str, Any]) -> Panel:
    """Render trade performance metrics panel."""
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    win_rate = metrics.get("win_rate")
    sharpe = metrics.get("sharpe_ratio", 0)
    total_pnl = metrics.get("total_pnl", 0)
    total_trades = metrics.get("total_trades", 0)
    winning = metrics.get("winning_trades", 0)
    losing = metrics.get("losing_trades", 0)

    pl_color = "green" if total_pnl >= 0 else "red"

    if win_rate is not None:
        wr_color = "green" if win_rate >= 0.5 else "yellow" if win_rate >= 0.4 else "red"
        table.add_row("Win Rate", f"[{wr_color}]{win_rate:.1%}[/{wr_color}]")
    else:
        table.add_row("Win Rate", "N/A")

    table.add_row("Sharpe Ratio", f"{sharpe:.2f}" if sharpe else "N/A")
    table.add_row("Total P&L", f"[{pl_color}]${total_pnl:+,.2f}[/{pl_color}]")
    table.add_row("Total Trades", str(total_trades))
    table.add_row("Wins / Losses", f"{winning} / {losing}")

    return Panel(table, title="Trade Performance (Last 100)", border_style="green")


def _render_open_positions(trades: List[Dict[str, Any]]) -> Panel:
    """Render open positions table."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Instrument", style="bold")
    table.add_column("Side", justify="center")
    table.add_column("Units", justify="right")
    table.add_column("Entry Price", justify="right")
    table.add_column("Current P&L", justify="right")

    if not trades:
        table.add_row("—", "—", "—", "—", "No open positions")
    else:
        for trade in trades[:10]:  # Show max 10
            instrument = trade.get("instrument", "???")
            units = int(trade.get("currentUnits", 0))
            side = "LONG" if units > 0 else "SHORT"
            side_color = "green" if units > 0 else "red"
            price = float(trade.get("price", 0))
            unrealized = float(trade.get("unrealizedPL", 0))
            pl_color = "green" if unrealized >= 0 else "red"

            table.add_row(
                instrument,
                f"[{side_color}]{side}[/{side_color}]",
                str(abs(units)),
                f"{price:.5f}",
                f"[{pl_color}]${unrealized:+.2f}[/{pl_color}]",
            )

    return Panel(table, title="Open Positions", border_style="yellow")


def _maybe_save_baseline(monitor: MonitoringSystem, metrics: Dict[str, Any]) -> None:
    """Save baseline metrics from OANDA data if none exist."""
    from datetime import date
    from src.utils.monitoring import PerformanceMetrics

    if monitor.baseline_metrics is not None:
        return  # Already have baseline

    if metrics.get("win_rate") is None:
        return  # No valid metrics

    try:
        baseline = PerformanceMetrics(
            date=date.today().isoformat(),
            total_pnl=float(metrics.get("total_pnl", 0)),
            total_trades=int(metrics.get("total_trades", 0)),
            winning_trades=int(metrics.get("winning_trades", 0)),
            losing_trades=int(metrics.get("losing_trades", 0)),
            win_rate=float(metrics.get("win_rate", 0)),
            sharpe_ratio=float(metrics.get("sharpe_ratio", 0)),
            max_drawdown=0.0,  # Would need trade history to calculate
            avg_trade_duration_hours=0.0,
            largest_win=0.0,
            largest_loss=0.0,
        )
        monitor.save_baseline_metrics(baseline)
        monitor.baseline_metrics = baseline
    except Exception:
        pass  # Silently fail


def _render_dashboard(monitor: MonitoringSystem) -> None:
    """Render the default monitoring dashboard with OANDA data."""
    data = monitor.get_dashboard_data()
    last_updated = _format_timestamp(data.get("timestamp", ""))
    header = Text("FX Trading Bot Monitoring", style="bold cyan")
    console.print(Panel(header, subtitle=f"Last Updated: {last_updated}"))

    # Fetch OANDA data
    oanda_data = _fetch_oanda_account_data()

    if oanda_data:
        # Row 1: Account + Trade Performance
        account_panel = _render_oanda_account(oanda_data["account"])
        trade_metrics = _calculate_trade_metrics(oanda_data["closed_trades"])
        performance_panel = _render_trade_performance(trade_metrics)
        console.print(Columns([account_panel, performance_panel]))

        # Auto-save baseline metrics if none exist
        _maybe_save_baseline(monitor, trade_metrics)

        # Row 2: Open Positions
        if oanda_data["open_trades"]:
            console.print(_render_open_positions(oanda_data["open_trades"]))
    else:
        # Fallback to original dashboard if OANDA unavailable
        console.print(Panel("[yellow]OANDA connection unavailable - showing cached data[/yellow]"))
        alert_summary = _render_alert_summary(data.get("alerts", {}))
        model_health = _render_model_health(data.get("recent_drift", []))
        console.print(Columns([alert_summary, model_health]))

        baseline = _render_performance_baseline(data.get("baseline_metrics"))
        console.print(baseline)

    # Row 3: Model Health + Alerts (always show)
    alert_summary = _render_alert_summary(data.get("alerts", {}))
    model_health = _render_model_health(data.get("recent_drift", []))
    console.print(Columns([alert_summary, model_health]))


def run_monitoring_dashboard(
    config_path: str,
    *,
    show_alerts: bool = False,
    show_drift: bool = False,
    generate_report: bool = False,
    drift_limit: int = 20,
    training_mode: bool = False,
    show_model: Optional[str] = None,
    replay_file: Optional[str] = None,
) -> None:
    """Run the monitoring dashboard with robust error handling."""
    # Training visualization mode
    if training_mode:
        console.print(Panel(
            "[bold cyan]Training Monitor Mode[/bold cyan]\n\n"
            "This mode shows live training visualization.\n"
            "Start training with: [yellow]buddy train -i EUR_USD[/yellow]\n\n"
            "The monitor will automatically display:\n"
            "  - Model architecture\n"
            "  - Loss curves (sparklines)\n"
            "  - Metrics with trend indicators\n"
            "  - Prediction flow visualization",
            title="Training Monitor",
            border_style="cyan",
        ))
        return

    # Show model architecture
    if show_model:
        from cli.training_monitor import render_model_architecture
        render_model_architecture(show_model, config_path)
        return

    # Replay training history
    if replay_file:
        from cli.training_monitor import replay_training_log
        replay_training_log(replay_file)
        return

    # Standard OANDA monitoring dashboard
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
