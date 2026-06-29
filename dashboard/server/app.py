"""AXIOM live-data layer — FastAPI READ-ONLY API + SSE stream.

Serves the bot's state-file data and read-only OANDA v20 practice views to the
Next.js dashboard. It NEVER places, modifies, or closes an order: the only OANDA
access is through ``ReadOnlyOandaClient`` (allowlist proxy) and the only mutating
imports from the bot are deliberately absent.

Run from the repo root:
    uvicorn dashboard.server.app:app --port 8888 --reload
or:
    python dashboard/server/app.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is importable (src.*) regardless of launch cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import re  # noqa: E402

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

from dashboard.server import data_sources as ds  # noqa: E402
from dashboard.server.safety import build_readonly_client  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("axiom.app")

@asynccontextmanager
async def lifespan(app_: FastAPI):
    # Built once; None if token missing/stale (live views degrade honestly).
    app_.state.client = build_readonly_client()
    logger.info("AXIOM data layer up. OANDA practice client: %s",
                "connected" if app_.state.client else "NOT connected (file-backed views still live)")
    yield


app = FastAPI(title="AXIOM data layer", version="1.0",
              description="Read-only terminal API for the Buddy trading engine", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    # Local-only dev tool: allow any localhost port (Next dev may auto-assign one).
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=False,
    allow_methods=["GET"],          # READ-ONLY: GET only, no POST/PUT/DELETE
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Tiny TTL cache so live OANDA reads aren't hammered (daily trend is stable).
# --------------------------------------------------------------------------- #
class _TTL:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def get(self, key: str, ttl: float):
        hit = self._store.get(key)
        if hit and (time.monotonic() - hit[0]) < ttl:
            return hit[1]
        return None

    def put(self, key: str, value: Any) -> Any:
        self._store[key] = (time.monotonic(), value)
        return value


_cache = _TTL()


def _cache_if_live(key: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Cache only successful live reads. A failed (connected:False) response is
    returned uncached so the next poll retries immediately instead of serving a
    stale failure for the full TTL — the UI self-heals after a transient 401."""
    if result.get("connected"):
        _cache.put(key, result)
    return result


def _client():
    return getattr(app.state, "client", None)


# --------------------------------------------------------------------------- #
# REST — file-backed (no network)
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": "AXIOM data layer", "read_only": True, "environment": "practice",
            "endpoints": ["/api/account", "/api/status", "/api/trades", "/api/equity",
                          "/api/strategy", "/api/sentiment", "/api/tier7", "/api/prices",
                          "/api/candles/{instrument}", "/api/instruments", "/api/stream"]}


# --------------------------------------------------------------------------- #
# Phase 2 control router — mounted ONLY when explicitly enabled. Default OFF =>
# not mounted => every /api/control/* route 404s. DO NOT enable until the separate
# safety verifier PASSes and the operator confirms the action set (CONTROL_DESIGN.md).
# --------------------------------------------------------------------------- #
_CONTROL_ENABLED = os.environ.get("AXIOM_CONTROL_ENABLED", "").lower() in ("1", "true", "yes")
if _CONTROL_ENABLED:
    from dashboard.server.control import router as control_router
    app.include_router(control_router)
    logger.warning("AXIOM CONTROL ROUTER MOUNTED (write path ACTIVE) — AXIOM_CONTROL_ENABLED is set.")
else:
    logger.info("AXIOM control router NOT mounted (read-only). Set AXIOM_CONTROL_ENABLED=1 to enable.")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "oanda_connected": _client() is not None,
            "environment": "practice", "control_enabled": _CONTROL_ENABLED}


@app.get("/api/account")
def account() -> Dict[str, Any]:
    return ds.read_account()


@app.get("/api/status")
def status() -> Dict[str, Any]:
    return ds.read_status()


@app.get("/api/trades")
def trades(limit: int = Query(100, ge=1, le=2000)) -> Dict[str, Any]:
    return ds.read_trades(limit=limit)


@app.get("/api/equity")
def equity() -> Dict[str, Any]:
    return ds.read_equity()


@app.get("/api/sentiment")
def sentiment() -> Dict[str, Any]:
    return ds.read_sentiment()


@app.get("/api/equity_sleeve")
def equity_sleeve() -> Dict[str, Any]:
    return ds.read_equity_sleeve()


@app.get("/api/tier7")
def tier7() -> Dict[str, Any]:
    return ds.read_tier7()


@app.get("/api/instruments")
def instruments() -> Dict[str, Any]:
    acct = ds.read_account()
    held = {p.get("instrument") for p in acct.get("positions", [])}
    return {"universe": ds.FX_MAJORS,
            "held": sorted([i for i in held if i]),
            "items": [{"instrument": i, "held": i in held} for i in ds.FX_MAJORS]}


# --------------------------------------------------------------------------- #
# REST — live read-only OANDA (cached)
# --------------------------------------------------------------------------- #
@app.get("/api/prices")
def prices(instruments: Optional[str] = None) -> Dict[str, Any]:
    insts: List[str] = instruments.split(",") if instruments else ds.FX_MAJORS
    key = "prices:" + ",".join(insts)
    cached = _cache.get(key, ttl=2.0)
    if cached is not None:
        return cached
    return _cache_if_live(key, ds.live_prices(_client(), insts))


# OANDA instrument names: BASE_QUOTE, alphanumerics each side (FX majors, metals,
# index/commodity CFDs). Validate before it flows into the v20 URL path (no
# injection / probing of other GET endpoints, even though the verb is read-only).
_INSTRUMENT_RE = re.compile(r"^[A-Z0-9]{2,7}_[A-Z0-9]{2,7}$")
_GRAN_OK = {"S5", "S10", "S30", "M1", "M2", "M4", "M5", "M10", "M15", "M30",
            "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D", "W", "M"}


@app.get("/api/candles/{instrument}")
def candles(instrument: str, granularity: str = "D",
            count: int = Query(300, ge=30, le=5000), sma: int = Query(100, ge=2, le=400)) -> Dict[str, Any]:
    if not _INSTRUMENT_RE.match(instrument):
        raise HTTPException(status_code=422, detail="invalid instrument format")
    if granularity not in _GRAN_OK:
        raise HTTPException(status_code=422, detail="invalid granularity")
    key = f"candles:{instrument}:{granularity}:{count}:{sma}"
    cached = _cache.get(key, ttl=30.0)
    if cached is not None:
        return cached
    return _cache_if_live(key, ds.live_candles(_client(), instrument, granularity=granularity,
                                               count=count, sma_window=sma))


@app.get("/api/strategy")
def strategy(sma: int = Query(100, ge=2, le=400), granularity: str = "D") -> Dict[str, Any]:
    key = f"strategy:{granularity}:{sma}"
    cached = _cache.get(key, ttl=60.0)
    if cached is not None:
        return cached
    trend = ds.live_trend(_client(), sma_window=sma, granularity=granularity)
    trend["status"] = ds.read_status()
    return _cache_if_live(key, trend)


# --------------------------------------------------------------------------- #
# SSE — push file-backed snapshot frequently + live prices periodically
# --------------------------------------------------------------------------- #
async def _event_stream():
    tick = 0
    last_prices: Dict[str, Any] = {"connected": False, "prices": {}}
    try:
        while True:
            payload: Dict[str, Any] = {
                "ts": time.time(),
                "account": ds.read_account(),     # cheap file reads
                "status": ds.read_status(),
            }
            # Refresh prices every ~4s (network) without blocking the event loop.
            if tick % 2 == 0:
                last_prices = await asyncio.to_thread(ds.live_prices, _client(), ds.FX_MAJORS)
            payload["prices"] = last_prices
            yield f"data: {json.dumps(payload)}\n\n"
            tick += 1
            await asyncio.sleep(2.0)
    except asyncio.CancelledError:  # client disconnected
        return


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.server.app:app", host="127.0.0.1", port=8888, reload=False)
