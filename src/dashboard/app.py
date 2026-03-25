"""Buddy Trading Dashboard — FastAPI control center.

Provides a web UI for scanning, trading, agent configuration, training,
watch mode, account management, and monitoring.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env from project root
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

app = FastAPI(title="Buddy Trading Dashboard", version="2.0.0")

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

CONFIG_OVERRIDES_PATH = ".claude/config_overrides.json"

# ── State ──────────────────────────────────────────────────────────────────
_scanner_instance = None
_scan_running = False
_last_scan_results: List[Dict] = []
_watch_running = False
_watch_thread: Optional[threading.Thread] = None
_continuous_scanner = None
_training_running = False
_training_status: Dict[str, Any] = {"status": "idle", "model": None, "progress": None}


# ── Config Helpers ─────────────────────────────────────────────────────────

def _load_json(path: str) -> Any:
    p = PROJECT_ROOT / path
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_json(path: str, data: Any) -> None:
    p = PROJECT_ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def _load_text(path: str) -> str:
    p = PROJECT_ROOT / path
    if p.exists():
        return p.read_text()
    return ""


def _load_config_overrides() -> Dict[str, Any]:
    """Load saved config overrides from disk."""
    return _load_json(CONFIG_OVERRIDES_PATH) or {}


def _apply_overrides_to_config(config: Any, overrides: Optional[Dict] = None) -> Any:
    """Apply saved UI config overrides to a ScannerConfig instance.

    This is the CRITICAL fix: overrides from the UI are loaded and applied
    to every ScannerConfig created for scanning/execution.
    """
    ovr = overrides if overrides is not None else _load_config_overrides()
    if not ovr:
        return config

    for key, value in ovr.items():
        if key.startswith("_"):
            continue
        if hasattr(config, key):
            field_type = type(getattr(config, key))
            try:
                if field_type == bool:
                    setattr(config, key, bool(value))
                elif field_type == int:
                    setattr(config, key, int(value))
                elif field_type == float:
                    setattr(config, key, float(value))
                elif field_type == str:
                    setattr(config, key, str(value))
                elif field_type == list and isinstance(value, list):
                    setattr(config, key, value)
                elif field_type == dict and isinstance(value, dict):
                    setattr(config, key, value)
                else:
                    setattr(config, key, value)
            except (ValueError, TypeError) as e:
                logger.warning(f"Config override '{key}' type mismatch: {e}")
    return config


def _make_scanner_config(profile: str = "balanced", granularity: str = "H1",
                         force: bool = False) -> Any:
    """Create a ScannerConfig with profile + UI overrides applied."""
    from src.scanner.config import ScannerConfig
    config = ScannerConfig(profile=profile)
    config.apply_profile(profile)
    config.granularity = granularity
    if force:
        config.enable_session_filter = False
    _apply_overrides_to_config(config)
    return config


def _make_execution_config() -> Any:
    """Create an ExecutionConfig with UI overrides applied.

    Maps ScannerConfig field names to ExecutionConfig field names.
    """
    from src.scanner.execution import ExecutionConfig
    exec_config = ExecutionConfig()
    overrides = _load_config_overrides()
    if not overrides:
        return exec_config

    # Direct mappings between ScannerConfig and ExecutionConfig fields
    direct_fields = [
        "account_equity", "risk_per_trade_pct", "leverage",
        "position_sizing_enabled", "aggressive_mode",
        "regime_scaling_enabled", "aggressive_scale_high_vol",
        "aggressive_scale_extreme_vol", "aggressive_min_meta_confidence",
        "atr_sl_multiplier", "atr_tp_multiplier",
        "min_sl_pips", "max_sl_pips", "min_tp_pips", "max_tp_pips",
        "high_prob_threshold", "high_prob_tp_bonus",
        "max_order_attempts", "rejection_downsize_factor",
        "high_slippage_alert_pips", "fallback_tp_pips", "fallback_atr_pips",
        "min_lot_size", "max_lot_size",
        "trailing_stop_breakeven_pct", "trailing_stop_lock_pct",
    ]
    # Map daily_trade_limit -> max_trades_per_day
    if "daily_trade_limit" in overrides:
        exec_config.max_trades_per_day = int(overrides["daily_trade_limit"])

    for key in direct_fields:
        if key in overrides and hasattr(exec_config, key):
            field_type = type(getattr(exec_config, key))
            try:
                setattr(exec_config, key, field_type(overrides[key]))
            except (ValueError, TypeError):
                pass
    return exec_config


def _get_scanner():
    global _scanner_instance
    if _scanner_instance is None:
        try:
            from src.scanner.engine import Scanner
            config = _make_scanner_config()
            _scanner_instance = Scanner(config)
        except Exception as e:
            print(f"Scanner init failed: {e}")
    return _scanner_instance


# ── Pages ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── API: Scanner ───────────────────────────────────────────────────────────

@app.post("/api/scan")
async def run_scan(request: Request, background_tasks: BackgroundTasks):
    """Run scan and optionally auto-execute trades that pass all gates.

    Body params:
        profile: str (balanced/conservative/aggressive/smart)
        pairs: list[str] (optional, default: all configured pairs)
        granularity: str (H1, M15, etc.)
        force: bool (skip session filter)
        auto_execute: bool (default True — execute tradeable setups after scan)
    """
    global _scan_running, _last_scan_results
    if _scan_running:
        return JSONResponse({"status": "already_running"})

    body = await request.json()
    profile = body.get("profile", "balanced")
    pairs = body.get("pairs", [])
    granularity = body.get("granularity", "H1")
    force = bool(body.get("force", False))
    auto_execute = bool(body.get("auto_execute", True))

    def _do_scan():
        global _scan_running, _last_scan_results
        _scan_running = True
        try:
            from src.scanner.engine import Scanner

            config = _make_scanner_config(profile, granularity, force)
            scanner = Scanner(config)
            result = scanner.scan(
                pairs=pairs or None,
            )
            _last_scan_results = []
            for a in (result.analyses if result else []):
                entry = {
                    "pair": a.pair,
                    "direction": a.direction,
                    "confidence": round(float(a.confidence), 4),
                    "gates_passed": bool(a.gates_passed),
                    "weighted_vote_score": round(float(getattr(a, "weighted_vote_score", 0)), 4),
                    "atr_pips": round(float(a.atr_pips), 1),
                    "volatility_regime": str(getattr(a, "volatility_regime", "UNKNOWN")),
                    "agent_reasons": getattr(a, "agent_reasons", []),
                    "why_trade": getattr(a, "why_trade", []),
                    "why_no_trade": getattr(a, "why_no_trade", []),
                    "error": a.error,
                    # Fields for trade execution
                    "current_price": round(float(getattr(a, "current_price", 0)), 5),
                    "atr": round(float(getattr(a, "atr", 0)), 6),
                    "is_tradeable": bool(getattr(a, "is_tradeable", False)),
                    "sl_pips": round(float(getattr(a, "sl_pips", 0)), 1),
                    "tp_pips": round(float(getattr(a, "tp_pips", 0)), 1),
                    # Execution result (populated if auto_execute)
                    "executed": False,
                    "exec_result": None,
                }
                _last_scan_results.append(entry)

            # ── Auto-execute tradeable setups ──────────────────────────
            if auto_execute:
                _auto_execute_scan_results()

        except Exception as e:
            logger.error(f"Scan failed: {e}", exc_info=True)
            _last_scan_results = [{"error": str(e)}]
        finally:
            _scan_running = False

    background_tasks.add_task(_do_scan)
    return JSONResponse({"status": "started"})


def _auto_execute_scan_results():
    """Execute trades for all scan results that pass gates.

    Called after scan completes when auto_execute is True.
    Respects all trading rules: R:R gate, correlation filter, daily limits.
    """
    global _last_scan_results
    tradeable = [r for r in _last_scan_results
                 if r.get("gates_passed") and r.get("is_tradeable")
                 and r.get("current_price", 0) > 0
                 and r.get("direction") in ("LONG", "SHORT")]

    if not tradeable:
        logger.info("Auto-execute: no tradeable setups found")
        return

    try:
        from src.scanner.execution import ExecutionManager
        exec_config = _make_execution_config()
        em = ExecutionManager(config=exec_config)

        # Check daily limit
        _, trades_today, trades_remaining = em.get_account_status()
        if trades_remaining <= 0:
            logger.warning(f"Auto-execute: daily limit reached ({trades_today} trades)")
            for r in tradeable:
                r["exec_result"] = {"error": "Daily trade limit reached"}
            return

        for r in tradeable[:trades_remaining]:
            try:
                result = em.execute_trade(
                    pair=r["pair"],
                    direction=r["direction"],
                    confidence=r["confidence"],
                    current_price=r["current_price"],
                    atr=r["atr"],
                    sl_pips=r.get("sl_pips") or None,
                    tp_pips=r.get("tp_pips") or None,
                )
                r["executed"] = result.success
                r["exec_result"] = {
                    "success": result.success,
                    "trade_id": result.trade_id,
                    "fill_price": result.fill_price,
                    "lots": result.lots,
                    "sl_pips": result.sl_pips,
                    "tp_pips": result.tp_pips,
                    "error": result.error,
                    "regime_name": result.regime_name,
                }
                if result.success:
                    logger.info(f"Auto-executed {r['direction']} {r['pair']} → ID {result.trade_id}")
                else:
                    logger.warning(f"Auto-execute failed for {r['pair']}: {result.error}")
            except Exception as e:
                logger.error(f"Auto-execute error for {r['pair']}: {e}")
                r["exec_result"] = {"error": str(e)}
    except Exception as e:
        logger.error(f"Auto-execute init error: {e}", exc_info=True)


@app.get("/api/scan/results")
async def scan_results():
    return JSONResponse({
        "running": _scan_running,
        "results": _last_scan_results,
    })


# ── API: Config ────────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    try:
        from src.scanner.config import ScannerConfig
        config = ScannerConfig()
        # Apply any saved overrides so the UI shows current state
        _apply_overrides_to_config(config)
        fields = {}
        for f in config.__dataclass_fields__:
            if f.startswith("_"):
                continue
            val = getattr(config, f)
            if isinstance(val, (str, int, float, bool)):
                fields[f] = val
            elif isinstance(val, (list, dict)):
                fields[f] = val
        return JSONResponse(fields)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/config")
async def update_config(request: Request):
    body = await request.json()
    # Merge with existing overrides (don't clobber unset fields)
    existing = _load_config_overrides()
    existing.update(body)
    _save_json(CONFIG_OVERRIDES_PATH, existing)
    # Reset cached scanner so next scan picks up new config
    global _scanner_instance
    _scanner_instance = None
    return JSONResponse({"status": "saved", "fields": len(body)})


@app.delete("/api/config")
async def reset_config():
    """Reset all config overrides to defaults."""
    try:
        p = PROJECT_ROOT / CONFIG_OVERRIDES_PATH
        if p.exists():
            # Write empty JSON instead of deleting (avoids permission errors)
            p.write_text("{}")
        global _scanner_instance
        _scanner_instance = None
        return JSONResponse({"status": "reset"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── API: Agents ────────────────────────────────────────────────────────────

@app.get("/api/agents")
async def get_agents():
    try:
        from src.scanner.agents import ScannerAgentTeam
        from src.scanner.config import ScannerConfig

        config = ScannerConfig()
        _apply_overrides_to_config(config)
        team = ScannerAgentTeam(config)

        agents = []
        for name, base_weight in team._BASE_WEIGHTS.items():
            learned = team._learned_weights.get(name)
            enabled_attr = f"enable_{name}_agent"
            enabled = getattr(config, enabled_attr, None)
            agents.append({
                "name": name,
                "base_weight": base_weight,
                "learned_weight": learned,
                "effective_weight": learned if learned is not None else base_weight,
                "enabled": enabled,
                "config_flag": enabled_attr,
            })
        return JSONResponse(agents)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[/api/agents] ERROR: {e}\n{tb}")
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)


@app.post("/api/agents/weights")
async def update_agent_weights(request: Request):
    body = await request.json()
    weights = body.get("weights", {})
    if weights:
        _save_json("trained_data/models/agent_weights.json", weights)
    return JSONResponse({"status": "saved"})


@app.post("/api/agents/reset-weights")
async def reset_agent_weights():
    path = PROJECT_ROOT / "trained_data/models/agent_weights.json"
    try:
        if path.exists():
            # Write default weights instead of deleting (avoids permission errors)
            import json
            default_weights = {name: 1.0 for name in [
                "trend", "mean_reversion", "volatility", "risk_sentinel",
                "uncertainty", "execution_quality", "momentum", "news_risk",
                "multi_timeframe", "pair_performance", "session_timing",
                "support_resistance"
            ]}
            path.write_text(json.dumps(default_weights, indent=2))
        return JSONResponse({"status": "reset"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── API: Account ───────────────────────────────────────────────────────────

@app.get("/api/account")
async def get_account():
    """Get OANDA account info: NAV, margin, P/L, trades today."""
    try:
        from src.scanner.execution import ExecutionManager
        exec_config = _make_execution_config()
        em = ExecutionManager(config=exec_config)
        nav, trades_today, trades_remaining = em.get_account_status()

        # Get fuller account details from OANDA
        account_detail = {}
        if em._init_oanda_client():
            try:
                result = em._retry_oanda(em._oanda.get_account_summary)
                acct = result.get("account", {})
                account_detail = {
                    "balance": float(acct.get("balance", 0)),
                    "unrealized_pl": float(acct.get("unrealizedPL", 0)),
                    "realized_pl": float(acct.get("pl", 0)),
                    "margin_used": float(acct.get("marginUsed", 0)),
                    "margin_available": float(acct.get("marginAvailable", 0)),
                    "open_trade_count": int(acct.get("openTradeCount", 0)),
                    "open_position_count": int(acct.get("openPositionCount", 0)),
                    "financing": float(acct.get("financing", 0)),
                }
            except Exception as e:
                account_detail = {"error": str(e)}

        return JSONResponse({
            "nav": nav,
            "trades_today": trades_today,
            "trades_remaining": trades_remaining,
            "max_trades_per_day": exec_config.max_trades_per_day,
            **account_detail,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── API: Trades ────────────────────────────────────────────────────────────

@app.get("/api/trades/open")
async def get_open_trades():
    try:
        from src.scanner.execution import ExecutionManager
        em = ExecutionManager(config=_make_execution_config())
        statuses = em.monitor_open_trades()
        return JSONResponse(statuses or [])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/trades/execute")
async def execute_trade(request: Request):
    """Execute a trade from scan results or manual input.

    Expects JSON body:
        pair: str, direction: str (LONG/SHORT), confidence: float,
        current_price: float, atr: float,
        sl_pips: float (optional), tp_pips: float (optional)
    """
    try:
        from src.scanner.execution import ExecutionManager
        body = await request.json()

        pair = body.get("pair", "")
        direction = body.get("direction", "")
        confidence = float(body.get("confidence", 0))
        current_price = float(body.get("current_price", 0))
        atr = float(body.get("atr", 0))
        sl_pips = body.get("sl_pips")
        tp_pips = body.get("tp_pips")

        if not pair or not direction:
            return JSONResponse({"error": "pair and direction required"}, status_code=400)
        if direction not in ("LONG", "SHORT"):
            return JSONResponse({"error": "direction must be LONG or SHORT"}, status_code=400)
        if current_price <= 0:
            return JSONResponse({"error": "current_price must be > 0"}, status_code=400)

        exec_config = _make_execution_config()
        em = ExecutionManager(config=exec_config)

        result = em.execute_trade(
            pair=pair,
            direction=direction,
            confidence=confidence,
            current_price=current_price,
            atr=atr,
            sl_pips=float(sl_pips) if sl_pips is not None else None,
            tp_pips=float(tp_pips) if tp_pips is not None else None,
        )

        return JSONResponse({
            "success": result.success,
            "trade_id": result.trade_id,
            "fill_price": result.fill_price,
            "units": result.units,
            "lots": result.lots,
            "sl_price": result.sl_price,
            "tp_price": result.tp_price,
            "sl_pips": result.sl_pips,
            "tp_pips": result.tp_pips,
            "error": result.error,
            "regime_name": result.regime_name,
            "regime_scale": result.regime_scale,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/trades/close")
async def close_trade(request: Request):
    """Close a specific trade by trade_id."""
    try:
        body = await request.json()
        trade_id = body.get("trade_id", "")
        if not trade_id:
            return JSONResponse({"error": "trade_id required"}, status_code=400)

        from src.utils.oanda_practice import OandaPracticeClient
        client = OandaPracticeClient.from_env()
        result = client.close_trade(trade_id=trade_id)
        return JSONResponse({"status": "closed", "result": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/trades/close-all")
async def close_all_trades():
    """Close ALL open trades."""
    try:
        from src.utils.oanda_practice import OandaPracticeClient
        import requests as req_lib

        token = os.getenv("OANDA_API_TOKEN", "")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")
        base = "https://api-fxpractice.oanda.com"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        resp = req_lib.get(f"{base}/v3/accounts/{acct}/openTrades", headers=headers, timeout=10)
        trades = resp.json().get("trades", [])

        closed = []
        errors = []
        client = OandaPracticeClient.from_env()
        for t in trades:
            tid = t.get("id")
            try:
                client.close_trade(trade_id=tid)
                closed.append(tid)
            except Exception as e:
                errors.append({"trade_id": tid, "error": str(e)})

        return JSONResponse({"closed": closed, "errors": errors})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/trades/modify")
async def modify_trade(request: Request):
    """Modify SL/TP on an open trade.

    Expects JSON: trade_id, sl_price (optional), tp_price (optional)
    """
    try:
        body = await request.json()
        trade_id = body.get("trade_id", "")
        sl_price = body.get("sl_price")
        tp_price = body.get("tp_price")

        if not trade_id:
            return JSONResponse({"error": "trade_id required"}, status_code=400)
        if sl_price is None and tp_price is None:
            return JSONResponse({"error": "sl_price or tp_price required"}, status_code=400)

        import requests as req_lib
        token = os.getenv("OANDA_API_TOKEN", "")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")
        base = "https://api-fxpractice.oanda.com"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        # Modify SL
        if sl_price is not None:
            payload = {"stopLoss": {"price": str(round(float(sl_price), 5))}}
            resp = req_lib.put(
                f"{base}/v3/accounts/{acct}/trades/{trade_id}/orders",
                headers=headers, json=payload, timeout=10
            )
            if resp.status_code >= 400:
                return JSONResponse({"error": f"SL modify failed: {resp.text}"}, status_code=resp.status_code)

        # Modify TP
        if tp_price is not None:
            payload = {"takeProfit": {"price": str(round(float(tp_price), 5))}}
            resp = req_lib.put(
                f"{base}/v3/accounts/{acct}/trades/{trade_id}/orders",
                headers=headers, json=payload, timeout=10
            )
            if resp.status_code >= 400:
                return JSONResponse({"error": f"TP modify failed: {resp.text}"}, status_code=resp.status_code)

        return JSONResponse({"status": "modified", "trade_id": trade_id})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/trades/journal")
async def get_trade_journal():
    for path in ["trained_data/trade_journal.json", "trained_data/trade_journal_rl.json"]:
        journal = _load_json(path)
        if isinstance(journal, list) and journal:
            return JSONResponse(journal[-50:])
    return JSONResponse([])


# ── API: Watch Mode ────────────────────────────────────────────────────────

@app.post("/api/watch/start")
async def start_watch(request: Request):
    """Start continuous scanning (watch mode)."""
    global _watch_running, _watch_thread, _continuous_scanner

    if _watch_running:
        return JSONResponse({"status": "already_running"})

    body = await request.json()
    profile = body.get("profile", "smart")
    granularity = body.get("granularity", "H1")
    interval = int(body.get("interval_minutes", 5))
    auto_execute = bool(body.get("auto_execute", False))

    def _run_watch():
        global _watch_running, _continuous_scanner, _last_scan_results
        _watch_running = True
        try:
            from src.scanner.engine import Scanner
            from src.scanner.automation.continuous import ContinuousScanner, ContinuousConfig

            config = _make_scanner_config(profile, granularity)
            scanner = Scanner(config)

            cont_config = ContinuousConfig(
                interval_minutes=interval,
                auto_execute=auto_execute,
                granularity=granularity,
            )
            _continuous_scanner = ContinuousScanner(scanner, cont_config)

            def on_scan_complete(result):
                global _last_scan_results
                _last_scan_results = []
                for a in (result.analyses if result else []):
                    _last_scan_results.append({
                        "pair": a.pair,
                        "direction": a.direction,
                        "confidence": round(float(a.confidence), 4),
                        "gates_passed": bool(a.gates_passed),
                        "weighted_vote_score": round(float(getattr(a, "weighted_vote_score", 0)), 4),
                        "atr_pips": round(float(a.atr_pips), 1),
                        "volatility_regime": str(getattr(a, "volatility_regime", "UNKNOWN")),
                        "is_tradeable": bool(getattr(a, "is_tradeable", False)),
                        "error": a.error,
                    })

            _continuous_scanner.run(
                auto_execute=auto_execute,
                on_scan_complete=on_scan_complete,
            )
        except Exception as e:
            logger.error(f"Watch mode error: {e}")
        finally:
            _watch_running = False
            _continuous_scanner = None

    _watch_thread = threading.Thread(target=_run_watch, daemon=True)
    _watch_thread.start()
    return JSONResponse({"status": "started", "interval_minutes": interval, "auto_execute": auto_execute})


@app.post("/api/watch/stop")
async def stop_watch():
    """Stop continuous scanning."""
    global _watch_running, _continuous_scanner
    if not _watch_running:
        return JSONResponse({"status": "not_running"})

    if _continuous_scanner:
        _continuous_scanner.stop()
    _watch_running = False
    return JSONResponse({"status": "stopping"})


@app.get("/api/watch/status")
async def watch_status():
    """Get watch mode status."""
    scan_count = 0
    if _continuous_scanner:
        scan_count = getattr(_continuous_scanner, "_scan_count", 0)
    return JSONResponse({
        "running": _watch_running,
        "scan_count": scan_count,
    })


# ── API: Training ──────────────────────────────────────────────────────────

@app.get("/api/training/status")
async def training_status():
    """Get current training status."""
    return JSONResponse(_training_status)


@app.post("/api/training/train-all")
async def train_all_models(request: Request, background_tasks: BackgroundTasks):
    """Train all ensemble models (Transformer, TCN, XGBoost, RF, Ridge, HistGB)."""
    global _training_running, _training_status

    if _training_running:
        return JSONResponse({"status": "already_running"})

    body = await request.json()
    warm_start = bool(body.get("warm_start", False))
    use_transformer = bool(body.get("use_transformer", True))
    train_histgb = bool(body.get("train_histgb", True))

    def _do_train():
        global _training_running, _training_status
        _training_running = True
        _training_status = {"status": "running", "model": "all", "started": time.time()}
        try:
            from src.training.trainers.train_all import train_all_modular
            from src.training.trainers.config import TrainerConfig
            from src.training.trainers.utils import PRODUCTION_MODELS_DIR

            # Load data
            _training_status["model"] = "loading_data"
            from src.training.data_loader import load_all_modular_data
            data = load_all_modular_data()

            _training_status["model"] = "training"
            trainers = train_all_modular(
                data=data,
                config=TrainerConfig(),
                save_dir=PRODUCTION_MODELS_DIR,
                use_transformer=use_transformer,
                warm_start=warm_start,
                train_histgb=train_histgb,
            )
            _training_status = {
                "status": "completed",
                "model": "all",
                "trained": list(trainers.keys()),
                "completed": time.time(),
            }
        except Exception as e:
            _training_status = {
                "status": "error",
                "model": "all",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        finally:
            _training_running = False

    background_tasks.add_task(_do_train)
    return JSONResponse({"status": "started"})


@app.post("/api/training/train-single")
async def train_single_model(request: Request, background_tasks: BackgroundTasks):
    """Train a single model by name.

    Expects JSON: model (str) — one of: transformer, tcn, xgboost, rf, ridge, histgb,
                  lightgbm_regime, lightgbm_momentum, lightgbm_risk, joint
    """
    global _training_running, _training_status

    if _training_running:
        return JSONResponse({"status": "already_running"})

    body = await request.json()
    model_name = body.get("model", "")

    valid_models = [
        "transformer", "tcn", "xgboost", "rf", "ridge", "histgb",
        "lightgbm_regime", "lightgbm_momentum", "lightgbm_risk",
        "transformer_regime", "joint",
    ]
    if model_name not in valid_models:
        return JSONResponse(
            {"error": f"Unknown model '{model_name}'. Valid: {valid_models}"},
            status_code=400
        )

    def _do_train_single():
        global _training_running, _training_status
        _training_running = True
        _training_status = {"status": "running", "model": model_name, "started": time.time()}
        try:
            from src.training.trainers.config import TrainerConfig
            from src.training.trainers.utils import PRODUCTION_MODELS_DIR

            config = TrainerConfig()
            save_dir = Path(PRODUCTION_MODELS_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Load data
            _training_status["progress"] = "loading_data"
            from src.training.data_loader import load_all_modular_data
            data = load_all_modular_data()

            _training_status["progress"] = "training"

            trainer = None
            if model_name == "transformer":
                from src.training.trainers.transformer_trainer import TransformerDirectionTrainer
                trainer = TransformerDirectionTrainer(config)
                d = data.get("direction", data.get("tcn"))
                trainer.train(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
                trainer.save(str(save_dir / "transformer_direction.keras"))

            elif model_name == "tcn":
                from src.training.trainers.tcn_trainer import TCNTrainer
                trainer = TCNTrainer(config)
                d = data.get("direction", data.get("tcn"))
                trainer.train(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
                trainer.save(str(save_dir / "tcn_direction.keras"))

            elif model_name == "xgboost":
                from src.training.trainers.xgboost_trainer import XGBoostTrainer
                trainer = XGBoostTrainer(config)
                d = data["xgboost"]
                trainer.train(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
                trainer.save(str(save_dir / "xgboost_momentum.json"))

            elif model_name == "rf":
                from src.training.trainers.random_forest_trainer import RandomForestTrainer
                trainer = RandomForestTrainer(config)
                d = data["rf"]
                trainer.train(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
                trainer.save(str(save_dir / "random_forest_drawdown.pkl"))

            elif model_name == "ridge":
                from src.training.trainers.ridge_trainer import RidgeTrainer
                trainer = RidgeTrainer(config)
                d = data["ridge"]
                trainer.train(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
                trainer.save(str(save_dir / "ridge_confidence.pkl"))

            elif model_name == "histgb":
                from src.training.trainers.histgb_trainer import HistGradientBoostingDirectionTrainer
                trainer = HistGradientBoostingDirectionTrainer(config)
                d = data.get("direction", data.get("tcn"))
                trainer.train(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
                trainer.save(str(save_dir / "histgb_direction.pkl"))

            elif model_name == "transformer_regime":
                from src.training.trainers.transformer_regime_trainer import TransformerRegimeTrainer
                trainer = TransformerRegimeTrainer(config)
                d = data.get("regime")
                if d is None:
                    raise ValueError("No regime data available — load with use_regime=True")
                trainer.train(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
                trainer.save(str(save_dir / "transformer_regime.keras"))

            elif model_name == "lightgbm_regime":
                from src.training.trainers.lightgbm_trainers import RegimeLGBMTrainer
                trainer = RegimeLGBMTrainer(config)
                d = data.get("direction", data.get("tcn"))
                trainer.train(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
                trainer.save(str(save_dir / "lgbm_regime.txt"))

            elif model_name == "lightgbm_momentum":
                from src.training.trainers.lightgbm_trainers import LightGBMMomentumTrainer
                trainer = LightGBMMomentumTrainer(config)
                d = data.get("xgboost", data.get("direction"))
                trainer.train(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
                trainer.save(str(save_dir / "lgbm_momentum.txt"))

            elif model_name == "lightgbm_risk":
                from src.training.trainers.lightgbm_trainers import LightGBMRiskTrainer
                trainer = LightGBMRiskTrainer(config)
                d = data.get("rf", data.get("direction"))
                trainer.train(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
                trainer.save(str(save_dir / "lgbm_risk.txt"))

            elif model_name == "joint":
                from src.training.trainers.joint_trainer import JointMultiPairTrainer
                jt = JointMultiPairTrainer()
                jt.train()
                trainer = jt  # Has different interface

            _training_status = {
                "status": "completed",
                "model": model_name,
                "completed": time.time(),
            }
        except Exception as e:
            _training_status = {
                "status": "error",
                "model": model_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        finally:
            _training_running = False

    background_tasks.add_task(_do_train_single)
    return JSONResponse({"status": "started", "model": model_name})


@app.get("/api/training/models")
async def list_models():
    """List available trained models and their file info."""
    models_dir = PROJECT_ROOT / "trained_data" / "models"
    joint_dir = models_dir / "joint"

    model_files = []
    for d in [models_dir, joint_dir]:
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix in (".keras", ".pkl", ".json", ".txt", ".h5"):
                    model_files.append({
                        "name": f.stem,
                        "path": str(f.relative_to(PROJECT_ROOT)),
                        "size_kb": round(f.stat().st_size / 1024, 1),
                        "modified": f.stat().st_mtime,
                        "directory": "joint" if "joint" in str(f) else "standard",
                    })

    model_files.sort(key=lambda x: x["modified"], reverse=True)
    return JSONResponse(model_files)


# ── API: Learning & State ──────────────────────────────────────────────────

@app.get("/api/state")
async def get_state():
    return JSONResponse(_load_json(".claude/state.json"))


@app.get("/api/learnings")
async def get_learnings():
    return JSONResponse({"content": _load_text(".claude/learnings.md")})


@app.get("/api/rules")
async def get_rules():
    return JSONResponse({"content": _load_text(".claude/rules/trading.md")})


@app.get("/api/config-adjustments")
async def get_config_adjustments():
    return JSONResponse(_load_json(".claude/config_adjustments.json"))


# ── API: Pairs ─────────────────────────────────────────────────────────────

@app.get("/api/pairs")
async def get_pairs():
    try:
        from src.scanner.config import DEFAULT_PAIRS
        return JSONResponse(DEFAULT_PAIRS)
    except Exception:
        return JSONResponse([
            "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
            "AUD_USD", "NZD_USD", "USD_CAD", "EUR_GBP",
            "EUR_JPY", "GBP_JPY",
        ])


# ── API: PRD ───────────────────────────────────────────────────────────────

@app.get("/api/prd")
async def get_prd():
    return JSONResponse(_load_json(".claude/ralph/prd.json"))


# ── API: Orchestrator ──────────────────────────────────────────────────────

_orchestrator = None


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        try:
            from src.scanner.automation.orchestrator import Orchestrator
            _orchestrator = Orchestrator(project_root=str(PROJECT_ROOT))
        except Exception as e:
            print(f"Orchestrator init failed: {e}")
    return _orchestrator


@app.post("/api/orchestrate")
async def run_orchestration_cycle(request: Request, background_tasks: BackgroundTasks):
    """Run a full scan->trade->learn->tune cycle."""
    global _scan_running
    if _scan_running:
        return JSONResponse({"status": "already_running"})

    body = await request.json()
    profile = body.get("profile", "balanced")
    granularity = body.get("granularity", "H1")

    def _do_cycle():
        global _scan_running, _last_scan_results
        _scan_running = True
        try:
            orch = _get_orchestrator()
            if orch:
                result = orch.run_cycle(profile=profile, granularity=granularity)
                _last_scan_results = [result.to_dict()]
            else:
                _last_scan_results = [{"error": "Orchestrator not available"}]
        except Exception as e:
            _last_scan_results = [{"error": str(e)}]
        finally:
            _scan_running = False

    background_tasks.add_task(_do_cycle)
    return JSONResponse({"status": "started"})


@app.get("/api/system-status")
async def get_system_status():
    """Get comprehensive system status including all automation modules."""
    try:
        orch = _get_orchestrator()
        if orch:
            status = orch.get_system_status()
            # Inject watch mode and training status
            status["watch_mode"] = {"running": _watch_running}
            status["training"] = _training_status
            return JSONResponse(status)
        return JSONResponse({
            "error": "Orchestrator not available",
            "watch_mode": {"running": _watch_running},
            "training": _training_status,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/improvement-report")
async def get_improvement_report():
    """Get human-readable improvement report."""
    try:
        orch = _get_orchestrator()
        if orch:
            return JSONResponse({"report": orch.get_improvement_report()})
        return JSONResponse({"report": "Orchestrator not available"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── API: QA Pipeline ──────────────────────────────────────────────────────

@app.get("/api/qa/audit")
async def run_qa_audit():
    """Run full QA audit and return report."""
    try:
        from src.scanner.automation.qa_pipeline import QAPipeline
        qa = QAPipeline(project_root=str(PROJECT_ROOT))
        report = qa.run_full_audit()
        return JSONResponse(report.to_dict())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/qa/review")
async def run_code_review():
    """Run brutal code review on last scan results."""
    try:
        from src.scanner.automation.qa_pipeline import BrutalCodeReview
        reviewer = BrutalCodeReview(project_root=str(PROJECT_ROOT))
        report = reviewer.generate_report(_last_scan_results)
        return JSONResponse(report)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── API: Observations ─────────────────────────────────────────────────────

@app.get("/api/observations")
async def get_observations():
    """Get recent market observations."""
    try:
        from src.scanner.automation.observation_log import ObservationLog
        obs = ObservationLog()
        return JSONResponse(obs.get_recent(limit=30))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── API: Improvement Trend ────────────────────────────────────────────────

@app.get("/api/improvement-trend")
async def get_improvement_trend():
    """Get improvement metrics trend."""
    try:
        from src.scanner.automation.improvement_tracker import ImprovementTracker
        tracker = ImprovementTracker()
        return JSONResponse(tracker.get_trend())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
