#!/usr/bin/env python3
"""
Enhanced ML Engine Trading Bot CLI

A command-line interface for the ML trading engine that handles:
- Command processing and orchestration
- Interactive dashboard visualization
- Configuration management
- Training/evaluation progress display
- Real-time monitoring
"""

import sys
import time
import logging
import argparse
from typing import Dict, Any

from pathlib import Path

import torch
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel

from ai_assistant import ai_assistant
from ml_engine_enhanced import EnhancedMLEngine
from mr_engine import MREngine
from neural_network_integrator_enhanced import NeuralNetworkIntegrator
from reasoning_enhanced import ReasoningEngine
from utils import setup_logging, load_config

# Constants
DEFAULT_MESSAGE_FORMAT = "Epoch {epoch} completed"
TIMESTAMP_FORMAT = "%H:%M:%S"
TABLE_HEADER_STYLE = "bold magenta"

# Initialize console and logging
console = Console()
logger = setup_logging(log_file="cli.log")


def _configure_predict_output(verbose: bool) -> None:
    """Reduce log/warning noise for interactive predict runs."""
    if verbose:
        return

    import warnings

    # Silence torch GradScaler deprecation warning during prediction.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*GradScaler\(args\.\.\.\) is deprecated\..*",
    )

    # Keep CLI output clean by muting INFO logs from internal modules.
    logging.getLogger().setLevel(logging.WARNING)
    for name in (
        "utils",
        "neural_network_integrator_enhanced",
        "ml_engine_enhanced",
        "memory_manager_enhanced",
        "reasoning_enhanced",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _build_integrated_engines(config: Dict[str, Any]) -> NeuralNetworkIntegrator:
    """Create and wire ML/MT/MR engines into the neural integrator."""
    integrator_config = {
        "device": config.get("device", "cpu"),
        "use_attention": False,
        "use_dynamic_weights": True,
    }
    integrator = NeuralNetworkIntegrator(integrator_config)

    ml_engine = EnhancedMLEngine(config)

    class _TorchPredictAdapter:
        def __init__(self, module: torch.nn.Module, device: str, kind: str):
            self._module = module.to(device)
            self._device = device
            self._kind = kind

        def predict(self, features):
            if features is None:
                raise ValueError(f"Missing features for {self._kind}")
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            features = features.to(self._device)
            self._module.eval()
            with torch.no_grad():
                out = self._module(features)
            # Handle multi-output modules
            if isinstance(out, (tuple, list)):
                out = out[0]
            return {"prediction": out.detach().cpu().numpy(), "uncertainty": 0.2}

    def _build_mt_model() -> torch.nn.Module:
        try:
            from mt_engine import MultiTaskModel

            return MultiTaskModel(input_size=11)
        except Exception:
            class _MTModel(torch.nn.Module):
                def __init__(self, input_size: int = 11, hidden_size: int = 32):
                    super().__init__()
                    self._lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
                    self._head = torch.nn.Linear(hidden_size, 1)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    out, _ = self._lstm(x)
                    return self._head(out[:, -1, :])

            return _MTModel(input_size=11)

    mt_model = _build_mt_model()
    mr_model = MREngine(input_size=11)
    mt_engine = _TorchPredictAdapter(mt_model, integrator.device, "mt")
    mr_engine = _TorchPredictAdapter(mr_model, integrator.device, "mr")

    reasoning_engine = ReasoningEngine(config.get("reasoning", {}))

    integrator.set_engines(
        ml_engine=ml_engine,
        mt_engine=mt_engine,
        mr_engine=mr_engine,
        reasoning_engine=reasoning_engine,
    )
    return integrator


def integrated_predict_once(config: Dict[str, Any]) -> Dict[str, Any]:
    """Smoke: run one integrated prediction across ML/MT/MR."""
    integrator = _build_integrated_engines(config)

    ml_features = torch.zeros((2, 5, int(config.get("model", {}).get("input_size", 7))), dtype=torch.float32)
    mt_features = torch.zeros((2, 5, 11), dtype=torch.float32)
    mr_features = torch.zeros((2, 5, 11), dtype=torch.float32)

    return integrator.predict(
        {"ml_features": ml_features, "mt_features": mt_features, "mr_features": mr_features}
    )


def compute_slm_indicators(market_data_df):
    """Compute a small subset of SLM technical indicators without network calls."""
    try:
        from SLM.stock_slm import (
            calculate_sma,
            calculate_ema,
            calculate_rsi,
            calculate_macd,
            calculate_bollinger_bands,
        )
    except Exception as e:  # pragma: no cover
        raise ImportError(f"Failed to import SLM indicators: {e}") from e

    df_for_slm = market_data_df
    # SLM indicator functions expect title-case columns like 'Close'.
    if hasattr(market_data_df, "columns") and "Close" not in market_data_df.columns:
        rename_map = {}
        for col in market_data_df.columns:
            key = str(col).strip().lower()
            if key == "close":
                rename_map[col] = "Close"
            elif key == "open":
                rename_map[col] = "Open"
            elif key == "high":
                rename_map[col] = "High"
            elif key == "low":
                rename_map[col] = "Low"
            elif key == "volume":
                rename_map[col] = "Volume"
        if rename_map:
            df_for_slm = market_data_df.rename(columns=rename_map)

    sma_20 = calculate_sma(df_for_slm, 20)
    ema_20 = calculate_ema(df_for_slm, 20)
    rsi_14 = calculate_rsi(df_for_slm, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(df_for_slm)
    bb_upper, bb_lower = calculate_bollinger_bands(df_for_slm, 20, 2)

    return {
        "sma_20": sma_20,
        "ema_20": ema_20,
        "rsi_14": rsi_14,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
    }


def _pick_default_market_csv(config: Dict[str, Any]) -> str:
    data_dir = (
        config.get("data", {}).get("data_dir")
        or config.get("paths", {}).get("data_dir")
        or config.get("DATA_DIR")
        or config.get("data_dir")
        or "market_data"
    )
    candidates = sorted(Path(data_dir).glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No .csv files found under: {data_dir}")
    return str(candidates[0])


def _normalize_market_dataframe(df):
    """Return a copy with standardized OHLCV column names: open/high/low/close/volume."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("market_data_df must be a pandas DataFrame")

    out = df.copy()
    rename_map = {}
    for col in out.columns:
        key = str(col).strip().lower()
        if key in {"open", "o"}:
            rename_map[col] = "open"
        elif key in {"high", "h"}:
            rename_map[col] = "high"
        elif key in {"low", "l"}:
            rename_map[col] = "low"
        elif key in {"close", "c", "adj close", "adj_close", "adjclose"}:
            rename_map[col] = "close"
        elif key in {"volume", "vol", "v"}:
            rename_map[col] = "volume"

    out = out.rename(columns=rename_map)
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Market DataFrame missing required columns: {missing}")
    return out


def build_integrated_feature_tensors(
    market_data_df, config: Dict[str, Any], *, allow_pad: bool = False
) -> Dict[str, torch.Tensor]:
    """Build feature tensors for ML/MT/MR using OHLCV + SLM indicators."""
    import pandas as pd
    import numpy as np

    df = _normalize_market_dataframe(market_data_df)

    seq_len = int(
        config.get("data", {}).get("sequence_length")
        or config.get("sequence_length")
        or 60
    )
    if len(df) < seq_len:
        if not allow_pad:
            raise ValueError(f"Need at least {seq_len} rows; got {len(df)}")
        pad_rows = seq_len - len(df)
        first_row = df.iloc[[0]].copy()
        df = pd.concat([pd.concat([first_row] * pad_rows, ignore_index=True), df], ignore_index=True)

    indicators = compute_slm_indicators(df)
    feature_frame = df[["open", "high", "low", "close", "volume"]].copy()
    feature_frame["sma_20"] = indicators["sma_20"]
    feature_frame["ema_20"] = indicators["ema_20"]
    feature_frame["rsi_14"] = indicators["rsi_14"]
    feature_frame["macd_line"] = indicators["macd_line"]
    feature_frame["bb_upper"] = indicators["bb_upper"]
    feature_frame["bb_lower"] = indicators["bb_lower"]

    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.ffill().bfill().fillna(0.0)
    window = feature_frame.tail(seq_len)

    ml_input_size = int(
        config.get("model", {}).get("input_size")
        or config.get("input_features")
        or config.get("input_size")
        or 7
    )

    base_cols = ["open", "high", "low", "close", "volume"]
    extra_cols = ["rsi_14", "sma_20", "ema_20", "macd_line", "bb_upper", "bb_lower"]
    ml_cols = (base_cols + extra_cols)[:ml_input_size]
    if len(ml_cols) < ml_input_size:
        for i in range(ml_input_size - len(ml_cols)):
            pad_name = f"ml_pad_{i}"
            window[pad_name] = 0.0
            ml_cols.append(pad_name)

    mt_mr_cols = base_cols + ["sma_20", "ema_20", "rsi_14", "macd_line", "bb_upper", "bb_lower"]

    ml_arr = window[ml_cols].to_numpy(dtype=np.float32, copy=True)
    mt_mr_arr = window[mt_mr_cols].to_numpy(dtype=np.float32, copy=True)

    ml_tensor = torch.tensor(ml_arr, dtype=torch.float32).unsqueeze(0)
    mt_tensor = torch.tensor(mt_mr_arr, dtype=torch.float32).unsqueeze(0)
    mr_tensor = torch.tensor(mt_mr_arr, dtype=torch.float32).unsqueeze(0)

    return {"ml_features": ml_tensor, "mt_features": mt_tensor, "mr_features": mr_tensor}


def integrated_predict(config_path: str, csv_path: str | None = None) -> None:
    """Run a unified prediction: CSV -> SLM indicators -> ML/MT/MR -> integrator -> reasoning."""
    import pandas as pd

    config = load_config(config_path)
    if csv_path is None:
        csv_path = _pick_default_market_csv(config)

    df = pd.read_csv(csv_path)
    features = build_integrated_feature_tensors(df, config)
    integrator = _build_integrated_engines(config)
    result = integrator.predict(features)

    pred = result.get("prediction")
    uncertainty = result.get("uncertainty")
    weights = result.get("weights")
    reasoning = result.get("reasoning")

    console.print(f"[bold blue]Integrated predict[/bold blue] CSV={csv_path}")
    console.print(f"[bold yellow]Prediction:[/bold yellow] {pred}")
    console.print(f"[bold yellow]Uncertainty:[/bold yellow] {uncertainty}")
    if weights is not None:
        console.print(f"[cyan]Weights (ML/MT/MR):[/cyan] {weights}")
    if isinstance(reasoning, dict) and reasoning.get("insights"):
        console.print("[bold green]Reasoning insights:[/bold green]")
        for insight in reasoning["insights"]:
            console.print(f"- {insight}")


def _fetch_live_market_data(ticker: str, period: str, interval: str):
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Live data fetch requires `yfinance`.") from e

    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if data is None or data.empty:
        raise RuntimeError(f"No live data returned for {ticker} ({period}, {interval})")
    # Keep the index as a column (Date/Datetime) for display/debugging.
    return data.reset_index()


def _df_last_value(df, candidates: tuple[str, ...]):
    for col in candidates:
        if col in df.columns:
            return df[col].iloc[-1]
    return None


def _print_last_bar(df) -> None:
    last_close = _df_last_value(df, ("Close", "close"))
    last_ts = _df_last_value(df, ("Datetime", "Date", "date", "datetime"))
    if last_ts is None and last_close is None:
        return
    console.print(f"[dim]Last bar:[/dim] {last_ts}  close={last_close}")


def _print_reasoning_insights(reasoning) -> None:
    if not isinstance(reasoning, dict):
        return
    insights = reasoning.get("insights")
    if not insights:
        return
    console.print("[bold green]Reasoning insights:[/bold green]")
    for insight in insights:
        console.print(f"- {insight}")


def _integrated_predict_live_once(
    *,
    ticker: str,
    period: str,
    interval: str,
    config: Dict[str, Any],
    integrator: NeuralNetworkIntegrator,
) -> None:
    df = _fetch_live_market_data(ticker, period=period, interval=interval)
    features = build_integrated_feature_tensors(df, config, allow_pad=True)
    result = integrator.predict(features)

    console.print(f"[bold blue]Live predict[/bold blue] {ticker} ({period}, {interval})")
    _print_last_bar(df)
    console.print(f"[bold yellow]Prediction:[/bold yellow] {result.get('prediction')}")
    console.print(f"[bold yellow]Uncertainty:[/bold yellow] {result.get('uncertainty')}")

    weights = result.get("weights")
    if weights is not None:
        console.print(f"[cyan]Weights (ML/MT/MR):[/cyan] {weights}")

    _print_reasoning_insights(result.get("reasoning"))


def integrated_predict_live(
    config_path: str,
    ticker: str,
    period: str = "5d",
    interval: str = "1h",
    watch_seconds: int | None = None,
) -> None:
    """Run the unified prediction using live-ish OHLCV from yfinance."""
    config = load_config(config_path)
    integrator = _build_integrated_engines(config)

    _integrated_predict_live_once(
        ticker=ticker,
        period=period,
        interval=interval,
        config=config,
        integrator=integrator,
    )

    if watch_seconds is None:
        return

    interval_seconds = max(1, int(watch_seconds))
    while True:  # pragma: no cover
        time.sleep(interval_seconds)
        _integrated_predict_live_once(
            ticker=ticker,
            period=period,
            interval=interval,
            config=config,
            integrator=integrator,
        )


def fx_paper_trade(
    config_path: str,
    instrument: str,
    granularity: str = "M5",
    candles: int = 300,
    execute: bool = False,
    equity: float = 10_000.0,
    risk_per_trade_pct: float = 0.005,
) -> None:
    """Paper trade forex on OANDA PRACTICE using a simple setup rule.

    This is intentionally conservative and defaults to DRY-RUN (no order placed).
    """
    _ = load_config(config_path)

    from oanda_practice import OandaPracticeClient
    from fx_paper import candles_to_ohlcv_df, atr, setup_signal, RiskRules, position_size_units

    client = OandaPracticeClient.from_env()
    resp = client.get_candles(instrument, granularity=granularity, count=candles)
    df = candles_to_ohlcv_df(resp)

    signal = setup_signal(df)
    last_close = float(df["close"].iloc[-1])
    atr_value = atr(df, period=14)

    rules = RiskRules(equity=equity, risk_per_trade_pct=risk_per_trade_pct)
    stop_distance = rules.atr_stop_mult * atr_value
    tp_distance = rules.rr_take_profit * stop_distance

    if signal == "hold":
        console.print(f"FX setup: HOLD  {instrument} {granularity}  close={last_close:.5f}  ATR14={atr_value:.5f}")
        return

    direction = 1 if signal == "buy" else -1
    stop_price = last_close - direction * stop_distance
    tp_price = last_close + direction * tp_distance

    units = position_size_units(
        instrument=instrument,
        equity=rules.equity,
        risk_per_trade_pct=rules.risk_per_trade_pct,
        stop_distance_price=abs(last_close - stop_price),
        price=last_close,
    )
    units = int(units * direction)

    console.print(
        f"FX setup: {signal.upper()}  {instrument} {granularity}  close={last_close:.5f}  units={units}  SL={stop_price:.5f}  TP={tp_price:.5f}"
    )

    if not execute:
        console.print("[dim]Dry-run (no order). Pass --execute to place a PRACTICE order.[/dim]")
        return

    result = client.create_market_order(
        instrument=instrument,
        units=units,
        stop_loss_price=stop_price,
        take_profit_price=tp_price,
    )
    console.print("[bold green]Order submitted (PRACTICE).[/bold green]")
    console.print(result)


def generate_dashboard(
    epoch: int,
    total_epochs: int,
    latency: int,
    train_loss: float,
    val_loss: float,
    lr_delta: float,
    api_logs: list,
    config: Dict[str, Any],
) -> Layout:
    """Generate the rich dashboard layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", size=8),
        Layout(name="footer", size=8),
    )
    metrics_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE, expand=True)
    metrics_table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    metrics_table.add_column("Value", style="green")
    metrics_table.add_row("Epoch", f"{epoch}/{total_epochs}")
    metrics_table.add_row("Training Loss", f"{train_loss:.6f}")
    metrics_table.add_row("Validation Loss", f"{val_loss:.6f}")
    metrics_table.add_row("API Latency", f"{latency}ms")
    lr_color = "green" if lr_delta >= 0 else "red"
    metrics_table.add_row("LR Change", f"[{lr_color}]{lr_delta:+.2f}%[/{lr_color}]")
    metrics_table.add_row(
        "Device", f"{config.get('hardware', {}).get('device', 'cpu')}"
    )
    status = (
        "[bold green]Active[/bold green]"
        if epoch < total_epochs
        else "[bold yellow]Complete[/bold yellow]"
    )
    metrics_table.add_row("Status", status)
    layout["header"].update(
        Panel(metrics_table, title="[bold]Training Metrics[/bold]", border_style="blue")
    )
    body_layout = Layout()
    body_layout.split_row(
        Layout(name="progress", ratio=1), Layout(name="logs", ratio=2)
    )
    progress_pct = (epoch / total_epochs * 100) if total_epochs > 0 else 0
    bar = "█" * int(progress_pct / 2) + "░" * (50 - int(progress_pct / 2))
    eta = (total_epochs - epoch) * latency / 1000
    progress_text = f"{progress_pct:.1f}%\n{bar}\nETA: {eta:.1f}s"
    body_layout["progress"].update(
        Panel(progress_text, title="[bold]Progress[/bold]", border_style="green")
    )
    current_time = time.strftime("%H:%M:%S")
    if not api_logs:
        api_logs.append(f"[grey]{current_time}[/grey] Starting training...")
    formatted_logs = [
        log if log.startswith("[") else f"[grey]{current_time}[/grey] {log}"
        for log in api_logs[-5:]
    ]
    logs_content = "\n".join(formatted_logs)
    body_layout["logs"].update(
        Panel(logs_content, title="[bold]Recent Updates[/bold]", border_style="yellow")
    )
    layout["body"].update(body_layout)
    config_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE, expand=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    model_config = config.get("model", {})
    config_table.add_row("Learning Rate", f"{model_config.get('learning_rate', 'N/A')}")
    config_table.add_row("Batch Size", f"{model_config.get('batch_size', 'N/A')}")
    config_table.add_row("Architecture", f"{model_config.get('architecture', 'N/A')}")
    config_table.add_row("Hidden Size", f"{model_config.get('hidden_size', 'N/A')}")
    config_table.add_row("Num Layers", f"{model_config.get('num_layers', 'N/A')}")
    config_table.add_row("Dropout", f"{model_config.get('dropout', 'N/A')}")
    optimizer_config = model_config.get("optimizer", {})
    config_table.add_row("Optimizer", f"{optimizer_config.get('type', 'Adam')}")
    logs_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE, expand=True)
    logs_table.add_column("Latest Log", style="green")
    latest_log = api_logs[-1] if api_logs else "No logs available"
    logs_table.add_row(latest_log)
    footer_layout = Layout()
    footer_layout.split_row(
        Panel(config_table, title="[bold]Configuration[/bold]", border_style="blue"),
        Panel(logs_table, title="[bold]Log Info[/bold]", border_style="blue"),
    )
    layout["footer"].update(footer_layout)
    return layout


# CLI Command Implementations
def train_model(config_path: str, choose_csv: bool = False) -> None:
    """Run model training with live progress dashboard.
    
    Args:
        config_path: Path to configuration file
        choose_csv: Whether to choose CSV file interactively
        
    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If training fails
    """
    try:
        # Load and validate configuration
        config = load_config(config_path)
        
        # Validate training configuration
        if "training" not in config:
            raise ValueError("Configuration missing 'training' section")
        if "epochs" not in config["training"]:
            raise ValueError("Configuration missing 'training.epochs'")
        
        console.print("[bold blue]Initializing ML Engine...[/bold blue]")
        
        # Load training data
        from data_loader import MarketDataLoader
        data_loader = MarketDataLoader(config)
        
        console.print("[bold blue]Loading market data...[/bold blue]")
        
        # Get data directory from config
        data_dir = config.get("data", {}).get("data_dir", "market_data/")
        
        # Load multiple tickers from the data directory
        data_dict = data_loader.load_multiple_tickers(data_dir)
        
        if not data_dict:
            raise ValueError(f"No data files found in {data_dir}")
        
        # Combine all ticker data
        df = data_loader.combine_ticker_data(data_dict, method="concat")
        console.print(f"[cyan]Loaded {len(data_dict)} tickers, {len(df)} total rows[/cyan]")
        
        # Preprocess the data
        console.print("[bold blue]Preprocessing data...[/bold blue]")
        X_train, y_train, X_val, y_val, X_test, y_test = data_loader.preprocess(
            df,
            add_features=True,
            scaler_type="standard",
            sequence_length=config.get("data", {}).get("sequence_length", 60),
            test_size=0.2,
        )
        
        console.print(f"[bold green]Data prepared: {len(X_train)} training samples, {len(X_val)} validation samples[/bold green]")
        
        # Update config with correct input size based on actual features
        input_size = X_train.shape[-1]  # Get feature dimension
        if "model" not in config:
            config["model"] = {}
        config["model"]["input_size"] = input_size
        console.print(f"[cyan]Model input size set to: {input_size}[/cyan]")
        
        engine = EnhancedMLEngine(config)

        api_logs = [f"[blue]{time.strftime('%H:%M:%S')}[/blue] Starting training"]
        
        # Get epochs from config (not nested in 'training')
        epochs = config.get("epochs", 100)
        console.print(f"[bold green]Training for {epochs} epochs[/bold green]")
        
        # Train the model and get results
        console.print("[bold green]Starting training...[/bold green]")
        result = engine.train(X_train, y_train, X_val, y_val, epochs=epochs)
        
        console.print("[bold green]Training completed successfully![/bold green]")
        console.print(f"[cyan]Total epochs trained: {result.get('total_epochs', epochs)}[/cyan]")
        console.print(f"[cyan]Best validation loss: {result['best_val_loss']:.6f}[/cyan]")
        console.print(f"[cyan]Resumed from checkpoint: {result.get('resumed', False)}[/cyan]")
                
    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found: {e}[/red]")
        raise
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logging.error(f"Unexpected error in train_model: {e}", exc_info=True)
        raise


def evaluate_model(config_path: str) -> None:
    """Run model evaluation with results dashboard.
    
    Args:
        config_path: Path to configuration file
        
    Raises:
        RuntimeError: If evaluation fails
    """
    try:
        console.print("[bold blue]Loading configuration and model...[/bold blue]")
        config = load_config(config_path)
        engine = EnhancedMLEngine(config)

        api_logs = [f"[blue]{time.strftime('%H:%M:%S')}[/blue] Starting evaluation"]
        
        with Live(generate_dashboard(0, 10, 200, 0.0, 0.0, 0.0, api_logs, config)) as live:
            try:
                console.print("[bold green]Evaluating model...[/bold green]")
                metrics = engine.evaluate_model()
                
                # Display evaluation results
                api_logs.append("[green]Evaluation completed[/green]")
                if metrics:
                    for key, value in metrics.items():
                        api_logs.append(f"[cyan]{key}: {value:.4f}[/cyan]")
                
                live.update(generate_dashboard(10, 10, 0, 0.0, 0.0, 0.0, api_logs, config))
                console.print("[bold green]Evaluation completed successfully![/bold green]")
                
            except Exception as e:
                console.print(f"[red]Evaluation error: {e}[/red]")
                logging.error(f"Evaluation failed: {e}", exc_info=True)
                raise RuntimeError(f"Evaluation failed: {e}")
                
    except Exception as e:
        console.print(f"[red]Failed to evaluate model: {e}[/red]")
        logging.error(f"Model evaluation error: {e}", exc_info=True)
        raise


# New CLI command implementations
def visualize_dashboard(config_path: str) -> None:
    """Display dashboard visualizations using the visualizer module."""
    import visualizer

    config = load_config(config_path)
    console.print("[bold blue]Launching visualizations...[/bold blue]")
    visualizer.display_dashboard(
        config
    )  # Assumes implementation exists in visualizer.py


def openai_tune(config_path: str) -> None:
    """Execute OpenAI-based auto-tuning using the openai_integration module."""
    try:
        import openai_integration
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "openai_integration requires the `openai` package; install it to use openai-tune."
        ) from e

    current_config = load_config(config_path)

    # Only set credentials when we actually use OpenAI integration.
    openai_integration.set_openai_credentials(current_config)

    # Gather current metrics from the engine
    engine = EnhancedMLEngine(current_config)
    train_loss = (
        engine.train_losses[-1]
        if hasattr(engine, "train_losses") and engine.train_losses
        else None
    )
    val_loss = (
        engine.val_losses[-1]
        if hasattr(engine, "val_losses") and engine.val_losses
        else None
    )
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": current_config.get("model", {}).get("learning_rate", 0.001),
        "epochs_completed": len(engine.train_losses)
        if hasattr(engine, "train_losses")
        else 0,
        "device": current_config.get("hardware", {}).get("device", "cpu"),
        "batch_size": current_config.get("model", {}).get("batch_size", 32),
    }

    # Call the integration function with metrics and config in proper order
    result = openai_integration.query_for_auto_configuration(metrics, current_config)

    if result:
        _ = openai_integration.report_config_changes(current_config, result)
        console.print(
            "[bold green]Configuration updated successfully and written to config.yaml[/bold green]"
        )
        console.print(f"[cyan]Updates Applied:[/cyan] {result}")
    else:
        console.print(
            "[bold red]Failed to get configuration updates from OpenAI[/bold red]"
        )


def predict_price(config_path: str) -> None:
    """Use ML engine to predict prices given a dataset.
    
    Args:
        config_path: Path to configuration file
        
    Raises:
        RuntimeError: If prediction fails
    """
    try:
        console.print("[bold blue]Loading model for prediction...[/bold blue]")
        config = load_config(config_path)
        engine = EnhancedMLEngine(config)
        
        console.print("[bold green]Generating prediction...[/bold green]")
        prediction = engine.predict_price()
        
        console.print(f"[bold yellow]Predicted Price: {prediction}[/bold yellow]")
        logging.info(f"Price prediction completed: {prediction}")
        
    except AttributeError as e:
        console.print(f"[red]Method not available: {e}[/red]")
        logging.error(f"predict_price method error: {e}")
        raise RuntimeError(f"Prediction method not available: {e}")
    except Exception as e:
        console.print(f"[red]Prediction failed: {e}[/red]")
        logging.error(f"Price prediction error: {e}", exc_info=True)
        raise


def realtime_loop(config_path: str) -> None:
    """
    Run the ML engine in a real-time loop for continuous inference.
    """
    config = load_config(config_path)
    engine = EnhancedMLEngine(config)
    engine.run_realtime_loop()
    console.print("[bold yellow]Realtime Loop command executed[/bold yellow]")


def tune_model(config_path: str) -> None:
    """
    Perform advanced hyperparameter tuning using ML engine methods.
    """
    config = load_config(config_path)
    engine = EnhancedMLEngine(config)
    engine.tune_hyperparameters()
    console.print("[bold yellow]Tune Model command executed[/bold yellow]")


def profile_pipeline(config_path: str) -> None:
    """
    Profile the ML pipeline for bottlenecks.
    """
    config = load_config(config_path)
    engine = EnhancedMLEngine(config)
    engine.profile_pipeline()
    console.print("[bold yellow]Profile Pipeline command executed[/bold yellow]")


def run_ai_assistant(config_path: str) -> None:
    config = load_config(config_path)  # removed Path() wrapper for proper string input
    # Instantiate the AI assistant using the class from ml_engine
    assistant_instance = ai_assistant(config)
    console.print("[bold blue]Launching AI Assistant...[/bold blue]")
    query = input("Enter your query: ")
    response = assistant_instance.process_query(query)
    console.print(f"[bold green]Response:[/bold green] {response}")

    # Optionally use ML Engine's AI assistant for simulation tasks
    engine_instance = EnhancedMLEngine(config)
    simulation_output = engine_instance.ai_assistant.process_query("simulate")
    console.print(
        f"[bold blue]AI Assistant Simulation Output:[/bold blue]\n{simulation_output}"
    )


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="ML Engine Trading Bot CLI")
    parser.add_argument(
        "command",
        nargs="?",
        default="predict",
        choices=[
            "train-model",
            "evaluate-model",
            "predict-price",
            "realtime-loop",
            "tune-model",
            "profile-pipeline",
            "visualize",  # New command
            "openai-tune",  # New command
            "ai-assistant",  # New sub-command
            "integrated-predict",
            "predict",
            "fx",
            "fx-paper",
        ],
        help="Command to execute",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="./config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--csv",
        "-f",
        default=None,
        help="Path to market CSV file (used by integrated-predict)",
    )
    parser.add_argument(
        "--ticker",
        "-t",
        default=None,
        help="Ticker symbol to fetch live-ish data via yfinance (used by predict/integrated-predict)",
    )
    parser.add_argument(
        "--period",
        "-p",
        default="5d",
        help="yfinance period (used with --ticker), e.g. 1d,5d,1mo",
    )
    parser.add_argument(
        "--interval",
        "-i",
        default="1h",
        help="yfinance interval (used with --ticker), e.g. 1m,5m,1h,1d",
    )
    parser.add_argument(
        "--watch-seconds",
        "-w",
        default=None,
        type=int,
        help="If set with --ticker, rerun live predict every N seconds",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed logs (default: quiet output)",
    )

    parser.add_argument(
        "--instrument",
        "-I",
        default="EUR_USD",
        help="OANDA instrument, e.g. EUR_USD (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--granularity",
        "-g",
        default="M5",
        help="OANDA candle granularity, e.g. M5, M15, H1 (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--candles",
        "-n",
        type=int,
        default=300,
        help="How many candles to fetch (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=10_000.0,
        help="Paper equity for sizing (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--risk",
        "-r",
        type=float,
        default=0.005,
        help="Risk per trade fraction (e.g. 0.005 = 0.5%) (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--execute",
        "-x",
        action="store_true",
        help="Actually place a PRACTICE order on OANDA (default: dry-run)",
    )
    # ... add more CLI arguments as needed ...

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    command_map = {
        "train-model": train_model,
        "evaluate-model": evaluate_model,
        "predict-price": predict_price,  # New
        "realtime-loop": realtime_loop,  # New
        "tune-model": tune_model,  # New
        "profile-pipeline": profile_pipeline,  # New
        "visualize": visualize_dashboard,  # New mapping
        "openai-tune": openai_tune,  # New mapping
        "ai-assistant": run_ai_assistant,  # updated mapping
    }

    try:
        if args.command in {"integrated-predict", "predict"}:
            _configure_predict_output(args.verbose)
            if args.ticker:
                integrated_predict_live(
                    args.config,
                    ticker=args.ticker,
                    period=args.period,
                    interval=args.interval,
                    watch_seconds=args.watch_seconds,
                )
            else:
                integrated_predict(args.config, args.csv)
            return

        if args.command in {"fx", "fx-paper"}:
            fx_paper_trade(
                args.config,
                instrument=args.instrument,
                granularity=args.granularity,
                candles=args.candles,
                execute=args.execute,
                equity=args.equity,
                risk_per_trade_pct=args.risk,
            )
            return
        command_map[args.command](args.config)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logging.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
