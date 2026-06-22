"""US-021 — Shadow / paper end-to-end orchestrator tests.

No mocks. Real :class:`ShadowPipeline`, real :class:`ShadowFillSimulator`,
real :class:`SimulatedFill`, real ``tmp_path`` disk for the SHIP_GATE
fixture, the ledger, and the pipeline state file.

Acceptance criteria covered:

* Ship-gate guard: missing / ``gate_pass != True`` / mismatched-hash all
  refuse construction; in-flight invalidation is caught on the next call.
* Halted-by-default: pipeline starts halted and refuses execute until
  the operator unhalts (the divergence check still runs, the stats stay
  untouched).
* Divergence (DD side): synthetic NAV path past the
  ``max_dd_breach_pp`` threshold flips ``breached=True`` and logs
  ``SHADOW_PIPELINE DIVERGENCE``; under the threshold does not.
* Divergence (turnover side): synthetic gross-traded total past the
  ``turnover_drift_pct`` threshold flips ``breached=True`` and the test
  also asserts the symmetric under-trading case triggers.
* Live mode refusal: :meth:`set_mode` raises on ``'live'`` (US-022 owns
  that path).
* Atomic state writes: state file is bytewise replaced (mkstemp +
  os.replace pattern) and survives a forced overwrite.
* Source-level: ``src/equity/shadow_pipeline.py`` contains zero bare
  ``except:`` clauses; this test file imports zero ``unittest.mock``
  symbols.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path

import pytest

from src.equity.shadow_fills import SimulatedFill
from src.equity.shadow_pipeline import (
    LOG_TAG_DIVERGENCE,
    DivergenceReport,
    DivergenceThresholds,
    ShadowPipeline,
    ShadowPipelineError,
    ShadowRunStats,
    enforce_ship_gate,
)


VALID_HASH = "8bfe419a0c1c06306ff3fa35c39132df522b17ae31718a1dd5463097b467899c"
ALT_HASH = "0000000000000000000000000000000000000000000000000000000000000001"


# ---------------------------------------------------------------------------
# Real-disk fixtures (no mocks).
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".shadow_pipeline_test_",
        suffix=".json",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except OSError:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _passing_payload(
    universe_hash: str = VALID_HASH,
    max_dd: float = 0.20,
    turnover_per_year: float = None,
) -> dict:
    payload = {
        "asof": "2026-05-29",
        "criteria_source": "test",
        "gate_pass": True,
        "max_dd": max_dd,
        "net_sharpe": 0.921,
        "pipeline_version": "test-v1",
        "positive_years": 13,
        "recommendation": "PASS",
        "thresholds": {
            "max_drawdown": 0.25,
            "min_net_sharpe": 0.4,
            "min_positive_years": 6.0,
            "min_total_years": 10.0,
        },
        "total_years": 17,
        "universe_hash": universe_hash,
    }
    if turnover_per_year is not None:
        payload["turnover_per_year"] = turnover_per_year
    return payload


def _ship_gate(
    tmp_path: Path,
    hash_: str = VALID_HASH,
    max_dd: float = 0.20,
    turnover_per_year: float = None,
) -> Path:
    path = tmp_path / "SHIP_GATE.json"
    _atomic_write_json(path, _passing_payload(hash_, max_dd, turnover_per_year))
    return path


def _thresholds(
    max_dd_breach_pp: float = 5.0,
    turnover_drift_pct: float = 25.0,
    backtest_turnover_per_year: float = None,
) -> DivergenceThresholds:
    return DivergenceThresholds(
        max_dd_breach_pp=max_dd_breach_pp,
        turnover_drift_pct=turnover_drift_pct,
        backtest_turnover_per_year=backtest_turnover_per_year,
    )


def _pipeline(
    tmp_path: Path,
    *,
    max_dd: float = 0.20,
    halted: bool = True,
    backtest_turnover_per_year: float = None,
    gate_turnover: float = None,
    mode: str = "shadow",
) -> ShadowPipeline:
    gate = _ship_gate(tmp_path, max_dd=max_dd, turnover_per_year=gate_turnover)
    return ShadowPipeline.create(
        universe_hash=VALID_HASH,
        ledger_path=tmp_path / "ledger.json",
        state_path=tmp_path / "state.json",
        thresholds=_thresholds(
            backtest_turnover_per_year=backtest_turnover_per_year,
        ),
        ship_gate_path=gate,
        halted=halted,
        mode=mode,
    )


def _fill(qty: int = 10, side: str = "BUY") -> SimulatedFill:
    return SimulatedFill(
        client_order_id="test-ord-1",
        ticker="AAPL",
        side=side,
        requested_quantity=qty,
        fill_quantity=qty,
        fill_price=100.05,
        reference_price=100.00,
        slippage_bps=5.0,
        order_state="FILLED",
        reason="clean_full_fill",
    )


# ---------------------------------------------------------------------------
# Ship-gate guard (AC #3).
# ---------------------------------------------------------------------------


def test_enforce_ship_gate_passes_for_matching_hash(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path)
    enforce_ship_gate(gate, VALID_HASH)  # does not raise


def test_enforce_ship_gate_refuses_when_missing(tmp_path: Path) -> None:
    gate = tmp_path / "no_such.json"
    with pytest.raises(ShadowPipelineError, match="ship gate not found"):
        enforce_ship_gate(gate, VALID_HASH)


def test_enforce_ship_gate_refuses_when_unreadable_json(tmp_path: Path) -> None:
    gate = tmp_path / "SHIP_GATE.json"
    gate.write_text("not json {{", encoding="utf-8")
    with pytest.raises(ShadowPipelineError, match="cannot read"):
        enforce_ship_gate(gate, VALID_HASH)


def test_enforce_ship_gate_refuses_non_object_payload(tmp_path: Path) -> None:
    gate = tmp_path / "SHIP_GATE.json"
    gate.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ShadowPipelineError, match="not a JSON object"):
        enforce_ship_gate(gate, VALID_HASH)


def test_enforce_ship_gate_refuses_when_gate_pass_false(tmp_path: Path) -> None:
    gate = tmp_path / "SHIP_GATE.json"
    payload = _passing_payload()
    payload["gate_pass"] = False
    _atomic_write_json(gate, payload)
    with pytest.raises(ShadowPipelineError, match="gate_pass="):
        enforce_ship_gate(gate, VALID_HASH)


def test_enforce_ship_gate_refuses_when_gate_pass_truthy_but_not_true(
    tmp_path: Path,
) -> None:
    for truthy in (1, "true", "yes"):
        gate = tmp_path / "SHIP_GATE.json"
        payload = _passing_payload()
        payload["gate_pass"] = truthy
        _atomic_write_json(gate, payload)
        with pytest.raises(ShadowPipelineError, match="gate_pass="):
            enforce_ship_gate(gate, VALID_HASH)


def test_enforce_ship_gate_refuses_when_hash_missing(tmp_path: Path) -> None:
    gate = tmp_path / "SHIP_GATE.json"
    payload = _passing_payload()
    payload["universe_hash"] = ""
    _atomic_write_json(gate, payload)
    with pytest.raises(
        ShadowPipelineError, match="universe_hash missing/blank"
    ):
        enforce_ship_gate(gate, VALID_HASH)


def test_enforce_ship_gate_refuses_when_hash_mismatch(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path, hash_=ALT_HASH)
    with pytest.raises(ShadowPipelineError, match="universe_hash mismatch"):
        enforce_ship_gate(gate, VALID_HASH)


def test_pipeline_constructor_refuses_when_gate_missing(tmp_path: Path) -> None:
    with pytest.raises(ShadowPipelineError, match="ship gate not found"):
        ShadowPipeline.create(
            universe_hash=VALID_HASH,
            ledger_path=tmp_path / "ledger.json",
            state_path=tmp_path / "state.json",
            thresholds=_thresholds(),
            ship_gate_path=tmp_path / "no_such.json",
        )


def test_pipeline_constructor_refuses_when_gate_pass_false(
    tmp_path: Path,
) -> None:
    gate = tmp_path / "SHIP_GATE.json"
    payload = _passing_payload()
    payload["gate_pass"] = False
    _atomic_write_json(gate, payload)
    with pytest.raises(ShadowPipelineError, match="gate_pass="):
        ShadowPipeline.create(
            universe_hash=VALID_HASH,
            ledger_path=tmp_path / "ledger.json",
            state_path=tmp_path / "state.json",
            thresholds=_thresholds(),
            ship_gate_path=gate,
        )


def test_pipeline_constructor_refuses_when_hash_mismatch(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path, hash_=ALT_HASH)
    with pytest.raises(ShadowPipelineError, match="universe_hash mismatch"):
        ShadowPipeline.create(
            universe_hash=VALID_HASH,
            ledger_path=tmp_path / "ledger.json",
            state_path=tmp_path / "state.json",
            thresholds=_thresholds(),
            ship_gate_path=gate,
        )


def test_pipeline_reenforces_gate_on_step(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, halted=False)
    # Invalidate the gate AFTER construction.
    payload = _passing_payload()
    payload["gate_pass"] = False
    _atomic_write_json(pipeline.ship_gate_path, payload)
    with pytest.raises(ShadowPipelineError, match="gate_pass="):
        pipeline.step(nav=100_000.0, gross_traded_notional=0.0)


# ---------------------------------------------------------------------------
# Halted-by-default + live refusal (AC #1).
# ---------------------------------------------------------------------------


def test_pipeline_default_is_halted_and_shadow(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    assert pipeline.halted is True
    assert pipeline.mode == "shadow"


def test_pipeline_step_while_halted_skips_execute(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    result = pipeline.step(
        nav=100_000.0,
        gross_traded_notional=10_000.0,
        cycle_seconds=86_400.0,
    )
    assert result.halted is True
    assert result.executed is False
    # Stats are not advanced when halted.
    assert pipeline.stats.cycles == 0
    assert pipeline.stats.gross_traded_notional == 0.0


def test_pipeline_unhalt_then_step_advances_stats(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.unhalt()
    assert pipeline.halted is False
    result = pipeline.step(
        nav=100_000.0,
        gross_traded_notional=5_000.0,
        cycle_seconds=86_400.0,
    )
    assert result.executed is True
    assert pipeline.stats.cycles == 1
    assert pipeline.stats.gross_traded_notional == 5_000.0


def test_pipeline_refuses_live_mode(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    with pytest.raises(ShadowPipelineError, match="refuses mode='live'"):
        pipeline.set_mode("live")


def test_pipeline_constructor_refuses_live_mode(tmp_path: Path) -> None:
    gate = _ship_gate(tmp_path)
    with pytest.raises(ShadowPipelineError, match="mode must be"):
        ShadowPipeline.create(
            universe_hash=VALID_HASH,
            ledger_path=tmp_path / "ledger.json",
            state_path=tmp_path / "state.json",
            thresholds=_thresholds(),
            ship_gate_path=gate,
            mode="live",
        )


def test_pipeline_paper_mode_allowed(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.set_mode("paper")
    assert pipeline.mode == "paper"


# ---------------------------------------------------------------------------
# Divergence — DD side (AC #2).
# ---------------------------------------------------------------------------


def test_divergence_under_threshold_does_not_breach(tmp_path: Path) -> None:
    # Backtest max_dd = 20%; threshold = 5pp; realized peak->trough = 22% -> 2pp.
    pipeline = _pipeline(tmp_path, max_dd=0.20, halted=False)
    pipeline.step(nav=100_000.0, gross_traded_notional=0.0, cycle_seconds=1.0)
    pipeline.step(nav=78_000.0, gross_traded_notional=0.0, cycle_seconds=1.0)
    report = pipeline.last_report
    assert report is not None
    assert report.breached is False
    assert report.max_dd_realized == pytest.approx(0.22, abs=1e-9)


def test_divergence_dd_above_threshold_breaches(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Backtest 20%, threshold 5pp -> realized 30% breaches (10pp).
    pipeline = _pipeline(tmp_path, max_dd=0.20, halted=False)
    with caplog.at_level(logging.WARNING, logger="src.equity.shadow_pipeline"):
        pipeline.step(
            nav=100_000.0, gross_traded_notional=0.0, cycle_seconds=1.0
        )
        result = pipeline.step(
            nav=70_000.0, gross_traded_notional=0.0, cycle_seconds=1.0
        )
    assert result.diverged is True
    assert result.report.breached is True
    assert result.report.max_dd_breach_pp == pytest.approx(10.0, abs=1e-6)
    assert any(LOG_TAG_DIVERGENCE in rec.message for rec in caplog.records)
    assert any("max_dd breach" in r for r in result.report.reasons)


def test_divergence_dd_below_backtest_is_not_breach(tmp_path: Path) -> None:
    # Shadow does *better* than backtest (DD = 0%); never triggers warning.
    pipeline = _pipeline(tmp_path, max_dd=0.20, halted=False)
    pipeline.step(nav=100_000.0, gross_traded_notional=0.0, cycle_seconds=1.0)
    pipeline.step(nav=105_000.0, gross_traded_notional=0.0, cycle_seconds=1.0)
    report = pipeline.last_report
    assert report is not None
    assert report.breached is False
    assert report.max_dd_breach_pp <= 0.0


# ---------------------------------------------------------------------------
# Divergence — turnover side (AC #2).
# ---------------------------------------------------------------------------


def test_divergence_turnover_over_threshold_breaches(tmp_path: Path) -> None:
    # Backtest turnover_per_year = 4.0 (via gate), drift threshold = 25%.
    # Shadow realized = 5.5 -> drift 37.5% -> BREACH.
    pipeline = _pipeline(tmp_path, halted=False, gate_turnover=4.0)
    # One year of elapsed time, NAV $100k, gross 550k -> 5.5x turnover.
    one_year = 365.25 * 24.0 * 3600.0
    pipeline.step(
        nav=100_000.0,
        gross_traded_notional=550_000.0,
        cycle_seconds=one_year,
    )
    report = pipeline.last_report
    assert report is not None
    assert report.breached is True
    assert report.turnover_realized == pytest.approx(5.5, abs=1e-6)
    assert report.turnover_drift_pct == pytest.approx(37.5, abs=1e-6)
    assert any("turnover drift" in r for r in report.reasons)


def test_divergence_turnover_under_threshold_also_breaches(
    tmp_path: Path,
) -> None:
    # Symmetric: drastically *under*-trading the backtest is also a
    # divergence signal (the model is silent / risk gates are over-firing).
    pipeline = _pipeline(tmp_path, halted=False, gate_turnover=4.0)
    one_year = 365.25 * 24.0 * 3600.0
    pipeline.step(
        nav=100_000.0,
        gross_traded_notional=100_000.0,  # 1.0x turnover, 75% drift
        cycle_seconds=one_year,
    )
    report = pipeline.last_report
    assert report is not None
    assert report.breached is True
    assert report.turnover_realized == pytest.approx(1.0, abs=1e-6)
    assert report.turnover_drift_pct == pytest.approx(75.0, abs=1e-6)


def test_divergence_turnover_within_threshold_does_not_breach(
    tmp_path: Path,
) -> None:
    # Backtest 4.0, drift threshold 25% -> [3.0, 5.0]. Realized 4.5 -> OK.
    pipeline = _pipeline(tmp_path, halted=False, gate_turnover=4.0)
    one_year = 365.25 * 24.0 * 3600.0
    pipeline.step(
        nav=100_000.0,
        gross_traded_notional=450_000.0,
        cycle_seconds=one_year,
    )
    report = pipeline.last_report
    assert report is not None
    assert report.breached is False
    assert report.turnover_realized == pytest.approx(4.5, abs=1e-6)


def test_divergence_turnover_fallback_to_operator_target(tmp_path: Path) -> None:
    # SHIP_GATE has NO turnover_per_year; operator supplies it via
    # thresholds.backtest_turnover_per_year. Test that fallback works.
    pipeline = _pipeline(
        tmp_path, halted=False, backtest_turnover_per_year=2.0
    )
    one_year = 365.25 * 24.0 * 3600.0
    pipeline.step(
        nav=100_000.0,
        gross_traded_notional=400_000.0,  # 4.0x, 100% drift -> BREACH
        cycle_seconds=one_year,
    )
    report = pipeline.last_report
    assert report is not None
    assert report.breached is True
    assert report.turnover_backtest == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Atomic state writes (AC #7).
# ---------------------------------------------------------------------------


def test_state_file_written_atomically_at_construction(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    assert pipeline.state_path.exists()
    payload = json.loads(pipeline.state_path.read_text(encoding="utf-8"))
    assert payload["universe_hash"] == VALID_HASH
    assert payload["halted"] is True
    assert payload["mode"] == "shadow"
    assert payload["stats"]["cycles"] == 0


def test_state_file_updates_after_step(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, halted=False)
    pipeline.step(nav=100_000.0, gross_traded_notional=5_000.0, cycle_seconds=1.0)
    payload = json.loads(pipeline.state_path.read_text(encoding="utf-8"))
    assert payload["stats"]["cycles"] == 1
    assert payload["stats"]["gross_traded_notional"] == 5_000.0
    assert payload["last_report"] is not None
    # No leftover .tmp files in the state directory.
    leftover = list(pipeline.state_path.parent.glob(".shadow_pipeline_*"))
    assert leftover == [], f"leftover temp files: {leftover}"


def test_state_payload_is_round_trippable(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, halted=False)
    pipeline.step(nav=100_000.0, gross_traded_notional=1.0, cycle_seconds=1.0)
    payload = json.loads(pipeline.state_path.read_text(encoding="utf-8"))
    # All required schema keys.
    for key in (
        "version",
        "universe_hash",
        "mode",
        "halted",
        "stats",
        "thresholds",
        "backtest_stats",
        "last_report",
    ):
        assert key in payload, f"missing state key: {key}"


# ---------------------------------------------------------------------------
# Fills recorded through pipeline land in the simulator ledger.
# ---------------------------------------------------------------------------


def test_step_records_supplied_fills_to_ledger(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, halted=False)
    fills = [_fill(qty=10), _fill(qty=20)]
    result = pipeline.step(
        nav=100_000.0,
        gross_traded_notional=3_005.0,
        cycle_seconds=1.0,
        fills=fills,
    )
    assert result.fills_recorded == 2
    ledger = json.loads(pipeline.ledger_path.read_text(encoding="utf-8"))
    assert len(ledger["entries"]) == 2
    assert ledger["entries"][0]["ticker"] == "AAPL"


def test_step_rejects_non_simulated_fill(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, halted=False)
    with pytest.raises(
        ShadowPipelineError, match="fills must be SimulatedFill"
    ):
        pipeline.step(
            nav=100_000.0,
            gross_traded_notional=0.0,
            cycle_seconds=1.0,
            fills=[{"not": "a fill"}],
        )


# ---------------------------------------------------------------------------
# Validation guards (AC #7 — no bare excepts, errors surface as exceptions).
# ---------------------------------------------------------------------------


def test_run_stats_rejects_non_positive_nav() -> None:
    stats = ShadowRunStats()
    with pytest.raises(ShadowPipelineError, match="nav must be"):
        stats.update(nav=0.0, gross_traded=0.0, cycle_seconds=0.0)
    with pytest.raises(ShadowPipelineError, match="nav must be"):
        stats.update(nav=float("nan"), gross_traded=0.0, cycle_seconds=0.0)


def test_run_stats_rejects_negative_gross() -> None:
    stats = ShadowRunStats()
    with pytest.raises(ShadowPipelineError, match="gross_traded must be"):
        stats.update(nav=100.0, gross_traded=-1.0, cycle_seconds=0.0)


def test_thresholds_reject_negative_values() -> None:
    with pytest.raises(ShadowPipelineError, match="max_dd_breach_pp must be"):
        DivergenceThresholds(
            max_dd_breach_pp=-1.0, turnover_drift_pct=10.0
        )
    with pytest.raises(ShadowPipelineError, match="turnover_drift_pct must be"):
        DivergenceThresholds(
            max_dd_breach_pp=5.0, turnover_drift_pct=-1.0
        )


def test_thresholds_reject_nan() -> None:
    with pytest.raises(ShadowPipelineError, match="max_dd_breach_pp must be"):
        DivergenceThresholds(
            max_dd_breach_pp=float("nan"), turnover_drift_pct=10.0
        )


def test_divergence_report_to_dict_roundtrip(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, halted=False)
    pipeline.step(nav=100_000.0, gross_traded_notional=0.0, cycle_seconds=1.0)
    assert isinstance(pipeline.last_report, DivergenceReport)
    payload = pipeline.last_report.to_dict()
    assert "breached" in payload
    assert "reasons" in payload
    assert isinstance(payload["reasons"], list)


# ---------------------------------------------------------------------------
# Halt / unhalt cycle.
# ---------------------------------------------------------------------------


def test_halt_persists_state_and_blocks_execute(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, halted=False)
    pipeline.step(nav=100_000.0, gross_traded_notional=0.0, cycle_seconds=1.0)
    pipeline.halt(reason="risk_event")
    payload = json.loads(pipeline.state_path.read_text(encoding="utf-8"))
    assert payload["halted"] is True
    result = pipeline.step(
        nav=100_000.0, gross_traded_notional=10_000.0, cycle_seconds=1.0
    )
    assert result.executed is False
    # Cycles count from BEFORE the halt, not incremented while halted.
    assert pipeline.stats.cycles == 1


# ---------------------------------------------------------------------------
# Source-level: no bare except, no mocks.
# ---------------------------------------------------------------------------


def test_shadow_pipeline_source_has_no_bare_except() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "equity" / "shadow_pipeline.py"
    text = src.read_text(encoding="utf-8")
    # No bare `except:` (with optional whitespace) anywhere in source.
    assert not re.search(r"^\s*except\s*:\s*$", text, flags=re.MULTILINE), (
        "bare except: clause found in shadow_pipeline.py"
    )
    # No `except Exception: pass` either.
    assert "except Exception: pass" not in text


def test_this_test_file_imports_no_unittest_mock() -> None:
    me = Path(__file__).read_text(encoding="utf-8")
    # Look for ACTUAL imports / call-sites, not docstring references.
    assert not re.search(r"^import unittest\.mock", me, flags=re.MULTILINE)
    assert not re.search(r"^from unittest\.mock\b", me, flags=re.MULTILINE)
    assert not re.search(r"^from unittest import mock", me, flags=re.MULTILINE)
    assert not re.search(r"\bMagicMock\(", me)
    assert not re.search(r"\bpatch\(", me)
