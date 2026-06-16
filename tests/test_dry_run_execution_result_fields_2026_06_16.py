"""Regression test for the US-013 dry-run ExecutionResult field bug (2026-06-16).

Bug: the US-013 dry-run branch in ``Scanner.execute_trades`` constructed
``ExecutionResult(success=..., trade_id=..., pair=..., direction=...,
lots=..., entry_price=..., mode=...)``. ``ExecutionResult``
(``src/scanner/execution.py``) is a plain ``@dataclass`` whose only fields are
``success / trade_id / fill_price / units / lots / sl_price / tp_price /
sl_pips / tp_pips / risk_pct / confidence_level / error / regime_* /
fill_status / slippage_pips``. There is NO ``pair``, ``direction``,
``entry_price`` or ``mode`` field, so the construction raised
``TypeError: __init__() got an unexpected keyword argument 'pair'`` on the
FIRST tradeable pair — taking down every scan cycle that exercised the
dry-run path (the headline "evaluate-without-ordering" safety feature was
non-functional).

Fix: construct ``ExecutionResult`` with valid fields only
(``fill_price`` / ``fill_status="DRY_RUN"``); pair/direction/entry/mode are
recorded in the ``dry_run_journal.jsonl`` entry, which already captures them.

No mocks: a real ``ScannerConfig``, a real ``PairAnalysis``, and a real
``ExecutionManager`` subclass (``_NavStubExecutionManager``) whose only
override is ``fetch_live_nav`` returning a fixed positive float so the
pre-flight NAV check passes without a live OANDA connection. Real disk via
``tmp_path``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.engine import Scanner
from src.scanner.config import ScannerConfig
from src.scanner.execution import ExecutionConfig, ExecutionManager
from src.scanner.results import PairAnalysis


class _NavStubExecutionManager(ExecutionManager):
    """Real ExecutionManager with a deterministic NAV so the execute_trades
    pre-flight passes without contacting the broker. Only fetch_live_nav is
    overridden — everything else is the genuine class. The dry-run branch
    returns before any order submission, so no broker call is ever reached.
    """

    def fetch_live_nav(self) -> float:  # type: ignore[override]
        return 100_000.0


def _make_tradeable_analysis() -> PairAnalysis:
    """Build a PairAnalysis that passes Scanner.execute_trades' tradeable
    filter (``is_tradeable`` True + ``agent_passed`` True). Uses the real
    ``is_tradeable`` setter override (``_force_tradeable_override``) shipped
    2026-05-10, which still enforces direction in {LONG, SHORT} and error is
    None.
    """
    a = PairAnalysis(pair="EUR_USD", direction="LONG", confidence=0.72)
    a.agent_passed = True
    a.error = None
    a.current_price = 1.0800
    a.atr = 0.0010
    a.recommended_lots = 1.5
    a.is_tradeable = True  # setter sets _force_tradeable_override
    assert a.is_tradeable is True  # fixture sanity
    return a


def test_execute_trades_dry_run_returns_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The dry-run branch must construct ExecutionResult and return cleanly.

    Pre-fix: ``ExecutionResult(pair=..., direction=..., entry_price=...,
    mode=...)`` raised TypeError on the first tradeable pair.
    """
    monkeypatch.chdir(tmp_path)

    cfg = ScannerConfig()
    cfg.enable_execution = True
    scanner = Scanner(config=cfg)

    # Inject a real (non-mock) executor so _init_executor() short-circuits and
    # the NAV pre-flight passes without a live broker.
    scanner._executor = _NavStubExecutionManager(config=ExecutionConfig())

    analysis = _make_tradeable_analysis()

    # The load-bearing assertion: this must NOT raise.
    results = scanner.execute_trades(analyses=[analysis], mode="dry_run")

    assert len(results) == 1
    r = results[0]
    assert r.success is True
    assert r.trade_id.startswith("DRYRUN-EUR_USD-")
    assert r.fill_price == pytest.approx(1.0800)
    assert r.fill_status == "DRY_RUN"
    assert r.lots == pytest.approx(1.5)

    # The dry-run journal entry carries pair/direction/mode (the fields that
    # do NOT belong on the dataclass).
    journal = tmp_path / "trained_data" / "dry_run_journal.jsonl"
    assert journal.exists()
    lines = [ln for ln in journal.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["pair"] == "EUR_USD"
    assert entry["direction"] == "LONG"
    assert entry["mode"] == "dry_run"
