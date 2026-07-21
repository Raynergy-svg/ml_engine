import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


def pytest_configure(config) -> None:  # noqa: ARG001
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# ── US-516: shared e2e fixtures ────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EVIDENCE_PATH = _REPO_ROOT / ".claude" / "ralph" / "reports" / "phase91_e2e_evidence.jsonl"


@pytest.fixture
def oanda_mock():
    """httpretty-backed OANDA REST mock with orders/trades/close endpoints."""
    import httpretty
    from httpretty import core as _hp_core

    # Workaround: urllib3>=2 calls sock.shutdown() during connection cleanup.
    # httpretty's fakesock.__getattr__ raises UnmockedError on unknown attrs.
    # Patch to return a no-op for shutdown/close so registered mocks succeed.
    _orig_getattr = _hp_core.fakesock.socket.__getattr__

    def _patched_getattr(self, name):
        if name in ("shutdown", "close"):
            return lambda *a, **kw: None
        return _orig_getattr(self, name)

    _hp_core.fakesock.socket.__getattr__ = _patched_getattr

    base = "https://api-fxpractice.oanda.com"
    acct = "test-acct"
    os.environ["OANDA_API_TOKEN"] = "test-token"
    os.environ["OANDA_ACCOUNT_ID"] = acct
    os.environ["OANDA_API_URL"] = base

    calls: list[dict] = []

    def _record(req, uri, hdrs, status=200, body=None):
        calls.append(
            {
                "method": req.method,
                "uri": uri,
                "body": req.body.decode("utf-8", "ignore") if req.body else "",
            }
        )
        return (status, hdrs, json.dumps(body or {"status": "ok"}))

    httpretty.enable(allow_net_connect=False, verbose=False)
    try:
        httpretty.register_uri(
            httpretty.POST,
            f"{base}/v3/accounts/{acct}/orders",
            body=lambda req, uri, hdrs: _record(
                req,
                uri,
                hdrs,
                body={
                    "orderFillTransaction": {
                        "id": "999",
                        "tradeOpened": {"tradeID": "T-NEW"},
                    }
                },
            ),
        )
        httpretty.register_uri(
            httpretty.GET,
            f"{base}/v3/accounts/{acct}/openTrades",
            body=lambda req, uri, hdrs: _record(req, uri, hdrs, body={"trades": []}),
        )
        # PUT close: register a wildcard prefix on the trades path. httpretty
        # matches a registered URI against the request's full URI by prefix
        # only when last_request_index is set; using the explicit per-trade
        # path here works for the test trade IDs we exercise.
        for tid in ("T-CLOSE-1", "T-NEW", "T-A", "T-B"):
            httpretty.register_uri(
                httpretty.PUT,
                f"{base}/v3/accounts/{acct}/trades/{tid}/close",
                body=lambda req, uri, hdrs: _record(
                    req, uri, hdrs, body={"orderCreateTransaction": {"id": "1001"}}
                ),
            )
        yield {"base": base, "account": acct, "calls": calls}
    finally:
        httpretty.disable()
        httpretty.reset()
        _hp_core.fakesock.socket.__getattr__ = _orig_getattr


@pytest.fixture
def tui_pilot():
    """Factory yielding (app, async_pilot_ctx) for a fresh BuddyApp.

    Usage:
        app, ctx = tui_pilot()
        async with ctx as pilot:
            await pilot.press("k")
    """
    from src.tui.app import BuddyApp

    def _factory(size=(120, 40)):
        app = BuddyApp(live=False)
        return app, app.run_test(size=size)

    return _factory


@pytest.fixture
def evidence_log(request):
    """Append per-test evidence row to phase91_e2e_evidence.jsonl."""
    _EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record: dict = {
        "name": request.node.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    yield record
    record["duration_ms"] = round((time.time() - started) * 1000, 2)
    record.setdefault("status", "passed")
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    with _EVIDENCE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Production-ledger write guard (2026-07-21)
#
# When DecisionRecord capture was added to AutonomousLoop, every existing
# control-loop test that did NOT set ``decision_ledger_path`` silently began
# writing real rows into trained_data/learning/decision_ledger.jsonl. 22 fake
# AAPL/MSFT decisions and 4 broker-link rows reached the PRODUCTION evidence
# ledger before it was noticed.
#
# That is the third instance of the same class this session (global os.environ
# mutation un-skipping a network test; a test writing the live .claude/state.json).
# The pattern: a module-level DEFAULT path is correct for production and lethal
# in tests, and every new writer inherits the hazard.
#
# So the guard is central rather than per-fixture: redirect the module-level
# defaults for the whole test session. A test that forgets to inject a tmp path
# lands in tmp, not in the evidence store. Individual tests that DO inject a
# path are unaffected.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def _guard_production_ledgers(tmp_path_factory):
    """Point every learning-ledger default at a throwaway dir for the session."""
    sandbox = tmp_path_factory.mktemp("ledger_guard")
    patched: list = []

    targets = [
        ("src.learning.decision_ledger", "DECISION_LEDGER_PATH", "decision_ledger.jsonl"),
        ("src.learning.outcome_ledger", "LEDGER_PATH", "outcome_ledger.jsonl"),
        ("src.learning.execution_context_builder", "SUBMISSION_GATE_PATH", "submission_gate.json"),
    ]
    for module_name, attr, filename in targets:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - optional deps must not break collection
            continue
        if hasattr(module, attr):
            patched.append((module, attr, getattr(module, attr)))
            setattr(module, attr, sandbox / filename)

    yield sandbox

    for module, attr, original in patched:
        setattr(module, attr, original)
