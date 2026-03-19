"""Buddy Trading Dashboard — FastAPI control center.

Provides a web UI for scanning, trading, agent configuration, and monitoring.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

app = FastAPI(title="Buddy Trading Dashboard", version="1.0.0")

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# ── State ──────────────────────────────────────────────────────────────────
_scanner_instance = None
_scan_running = False
_last_scan_results: List[Dict] = []
_watch_running = False


def _get_scanner():
    global _scanner_instance
    if _scanner_instance is None:
        try:
            from src.scanner.config import ScannerConfig
            from src.scanner.engine import Scanner
            _scanner_instance = Scanner(ScannerConfig())
        except Exception as e:
            print(f"Scanner init failed: {e}")
    return _scanner_instance


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


# ── Pages ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── API: Scanner ───────────────────────────────────────────────────────────

@app.post("/api/scan")
async def run_scan(request: Request, background_tasks: BackgroundTasks):
    global _scan_running, _last_scan_results
    if _scan_running:
        return JSONResponse({"status": "already_running"})

    body = await request.json()
    profile = body.get("profile", "balanced")
    pairs = body.get("pairs", [])
    granularity = body.get("granularity", "H1")

    def _do_scan():
        global _scan_running, _last_scan_results
        _scan_running = True
        try:
            from src.scanner.config import ScannerConfig
            from src.scanner.engine import Scanner

            config = ScannerConfig(profile=profile)
            scanner = Scanner(config)
            result = scanner.scan(
                pairs=pairs or None,
                granularity=granularity,
            )
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
                    "agent_reasons": getattr(a, "agent_reasons", []),
                    "why_trade": getattr(a, "why_trade", []),
                    "why_no_trade": getattr(a, "why_no_trade", []),
                    "error": a.error,
                })
        except Exception as e:
            _last_scan_results = [{"error": str(e)}]
        finally:
            _scan_running = False

    background_tasks.add_task(_do_scan)
    return JSONResponse({"status": "started"})


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
        # Extract all fields as dict
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
    # Store config overrides
    _save_json(".claude/config_overrides.json", body)
    return JSONResponse({"status": "saved", "fields": len(body)})


# ── API: Agents ────────────────────────────────────────────────────────────

@app.get("/api/agents")
async def get_agents():
    try:
        from src.scanner.agents import ScannerAgentTeam
        from src.scanner.config import ScannerConfig

        config = ScannerConfig()
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
        return JSONResponse({"error": str(e)}, status_code=500)


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
    if path.exists():
        path.unlink()
    return JSONResponse({"status": "reset"})


# ── API: Trades ────────────────────────────────────────────────────────────

@app.get("/api/trades/open")
async def get_open_trades():
    try:
        from src.scanner.execution import ExecutionManager
        em = ExecutionManager()
        statuses = em.monitor_open_trades()
        return JSONResponse(statuses or [])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/trades/journal")
async def get_trade_journal():
    journal = _load_json("trained_data/trade_journal_rl.json")
    if isinstance(journal, list):
        return JSONResponse(journal[-50:])  # Last 50 trades
    return JSONResponse([])


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
    """Run a full scan→trade→learn→tune cycle."""
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
            return JSONResponse(orch.get_system_status())
        return JSONResponse({"error": "Orchestrator not available"})
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
        obs = ObservationLog(project_root=str(PROJECT_ROOT))
        return JSONResponse(obs.get_recent(limit=30))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── API: Improvement Trend ────────────────────────────────────────────────

@app.get("/api/improvement-trend")
async def get_improvement_trend():
    """Get improvement metrics trend."""
    try:
        from src.scanner.automation.improvement_tracker import ImprovementTracker
        tracker = ImprovementTracker(project_root=str(PROJECT_ROOT))
        return JSONResponse(tracker.get_trend())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
