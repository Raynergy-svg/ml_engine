#!/usr/bin/env python3
"""Interactive wizard for ML Engine Trading Bot CLI."""
from __future__ import annotations

from typing import Any, Dict

from cli.io_utils import (
    console,
    _parse_bool_answer,
    _parse_int_answer,
    _parse_float_answer,
    _normalize_instrument,
    DEFAULT_CONFIG_PATH,
)


def launch_buddy_repl_from_wizard(
    config_path: str,
    *,
    checkpoint_path: str | None = None,
    instrument: str = "USD_JPY",
    granularity: str = "M5",
    candles: int = 2000,
    execute: bool = True,
    all_features: bool = False,
    verbose: bool = False,
    assistant_name: str = "Buddy",
) -> None:
    """Launch the interactive Buddy REPL (lazy-imports heavy deps).

    This is a thin helper intended to be called from the interactive wizard
    to avoid importing `unified_talk` at module import time.
    """
    try:
        from src.utils.unified_talk import run_unified_talk, OandaSettings  # type: ignore
    except Exception as e:  # pragma: no cover - runtime guard
        console.print(
            f"[red]Could not import Buddy REPL (missing optional dependency): {e}[/red]"
        )
        raise

    # Build OandaSettings (some variants may have different constructor signatures).
    try:
        oanda_settings = OandaSettings(instrument=instrument, granularity=granularity, candles=int(candles), execute=bool(execute))
    except Exception:
        # Fallback simple namespace for compatibility
        class _S:  # simple compatibility shim
            pass

        oanda_settings = _S()
        setattr(oanda_settings, "instrument", instrument)
        setattr(oanda_settings, "granularity", granularity)
        setattr(oanda_settings, "candles", int(candles))
        setattr(oanda_settings, "execute", bool(execute))

    # Call the REPL entrypoint from unified_talk. Keep keyword args explicit.
    run_unified_talk(
        config_path,
        checkpoint_path=checkpoint_path,
        csv_path=None,
        ticker=None,
        period="5d",
        interval="1h",
        oanda=True,
        oanda_settings=oanda_settings,
        all_features=bool(all_features),
        verbose=bool(verbose),
        assistant_name=assistant_name,
    )


def _buddy_wizard_ask(prompt: str) -> str:
    try:
        return str(console.input(prompt) or "")
    except EOFError:
        return ""


def _buddy_wizard_collect_settings(*, default_config: str) -> dict[str, Any]:
    cfg_path = (_buddy_wizard_ask(f"Config path [{default_config}]: ") or "").strip() or default_config
    instrument = _normalize_instrument(_buddy_wizard_ask("Instrument [USD_JPY]: ") or "USD_JPY")
    granularity = (_buddy_wizard_ask("Granularity [M5]: ") or "M5").strip() or "M5"

    candles = _parse_int_answer(
        _buddy_wizard_ask("Candles lookback [2000]: "),
        default=2000,
    )
    if int(candles) < 300:
        candles = 300

    equity = _parse_float_answer(
        _buddy_wizard_ask("Equity for sizing (PRACTICE NAV fallback) [10000]: "),
        default=10_000.0,
    )

    # Risk is used as a fraction internally (0.005 = 0.5%). Ask as percent for humans.
    risk_pct = _parse_float_answer(_buddy_wizard_ask("Risk per trade (%) [0.5]: "), default=0.5)
    risk_frac = max(0.0, float(risk_pct) / 100.0)

    execute = _parse_bool_answer(_buddy_wizard_ask("Place PRACTICE orders? (Y/n): "), default=True)

    force_execute = False
    if execute:
        force_execute = _parse_bool_answer(
            _buddy_wizard_ask("Bypass training gate (>=62% dir acc)? (Y/n): "),
            default=True,
        )

    loop = _parse_bool_answer(_buddy_wizard_ask("Run continuously (loop)? (y/N): "), default=False)
    max_trades = 1
    if loop:
        max_trades = _parse_int_answer(_buddy_wizard_ask("Max executed trades (0 = unlimited) [1]: "), default=1)
        max_trades = max(0, int(max_trades))

    return {
        "cfg_path": cfg_path,
        "instrument": instrument,
        "granularity": granularity,
        "candles": int(candles),
        "equity": float(equity),
        "risk_frac": float(risk_frac),
        "execute": bool(execute),
        "force_execute": bool(force_execute),
        "loop": bool(loop),
        "max_trades": int(max_trades),
    }


def _buddy_wizard_confirm_settings(s: dict[str, Any]) -> bool:
    console.print(
        f"\n[dim]Summary[/dim]: instrument={s['instrument']} granularity={s['granularity']} candles={int(s['candles'])} "
        f"equity={float(s['equity']):.2f} risk={float(s['risk_frac'])*100:.3f}% execute={bool(s['execute'])} "
        f"force_execute={bool(s['force_execute'])} loop={bool(s['loop'])}"
    )
    if _parse_bool_answer(_buddy_wizard_ask("Proceed? (Y/n): "), default=True):
        return True
    console.print("[dim]Cancelled.[/dim]")
    return False


def _buddy_wizard_select_mode() -> int:
    # Offer a short mode menu to make intentions explicit.
    console.print(
        "\nChoose mode:\n 1) Single-shot inference: run one prediction and exit\n"
        " 2) Loop trading: run continuously and execute trades\n"
        " 3) Launch Buddy REPL: interactive trading commands (buy/sell/close/etc.)\n"
        " 4) Train Buddy: run training workflow"
    )
    choice = (_buddy_wizard_ask("Mode [1]: ") or "1").strip() or "1"
    try:
        return int(choice)
    except Exception:
        return 1


def _buddy_wizard_dispatch_mode(mode: int, s: dict[str, Any]) -> None:
    # Lazy import to avoid circular dependency
    from cli.commands import buddy, buddy_loop
    from cli.training import train_buddy

    if mode == 3:
        # Launch the interactive REPL that exposes Buddy's trade commands.
        launch_buddy_repl_from_wizard(
            s["cfg_path"],
            checkpoint_path=None,
            instrument=s["instrument"],
            granularity=s["granularity"],
            candles=int(s["candles"]),
            execute=bool(s["execute"]),
            all_features=False,
            verbose=False,
            assistant_name="Buddy",
        )
        return

    if mode == 4:
        # Kick off training flow
        train_buddy(s["cfg_path"])
        return

    # Mode 2 = loop, else single-shot
    if mode == 2 or bool(s["loop"]):
        buddy_loop(
            s["cfg_path"],
            instrument=s["instrument"],
            granularity=s["granularity"],
            candles=int(s["candles"]),
            execute=bool(s["execute"]),
            force_execute=bool(s["force_execute"]),
            equity=float(s["equity"]),
            risk_per_trade_pct=float(s["risk_frac"]),
            all_features=False,
            verbose=False,
            max_trades=int(s["max_trades"]),
        )
        return

    buddy(
        s["cfg_path"],
        instrument=s["instrument"],
        granularity=s["granularity"],
        candles=int(s["candles"]),
        execute=bool(s["execute"]),
        force_execute=bool(s["force_execute"]),
        equity=float(s["equity"]),
        risk_per_trade_pct=float(s["risk_frac"]),
        all_features=False,
        verbose=False,
    )


def _buddy_interactive_wizard(*, default_config: str = DEFAULT_CONFIG_PATH) -> None:
    console.print("[bold]Buddy interactive[/bold] (press Enter for defaults)")
    settings = _buddy_wizard_collect_settings(default_config=default_config)
    if not _buddy_wizard_confirm_settings(settings):
        return
    _buddy_wizard_dispatch_mode(_buddy_wizard_select_mode(), settings)
