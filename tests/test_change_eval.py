"""Tests for ChangeEvalHarness — verifies the four-layer scorecard
composes correctly with mock dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.scanner.automation.change_eval import ChangeEvalHarness
from src.scanner.automation.meta_types import (
    ChangeKind,
    ChangePackage,
    Proposal,
)


@pytest.fixture
def tmp_log(tmp_path):
    return tmp_path / "scorecards.jsonl"


@pytest.fixture
def regime_fixture(tmp_path):
    p = tmp_path / "regimes.json"
    p.write_text(json.dumps({"windows": [
        {"name": "trending_test", "pair": "EUR_USD", "days": 5, "regime": "trending"},
        {"name": "ranging_test", "pair": "EUR_USD", "days": 5, "regime": "ranging"},
    ]}))
    return p


def _pkg_with_delta() -> ChangePackage:
    pkg = ChangePackage(kind=ChangeKind.CONFIG)
    pkg.proposal = Proposal(config_delta={"min_confidence": {"old": 0.5, "new": 0.55}})
    return pkg


class _FakeReplay:
    """Fake ReplayValidator that returns deterministic numbers per pair."""

    def compare_configs(self, cfg_a, cfg_b, pair, candles):
        return {
            "config_a": {"total_pnl": 5.0, "sharpe": 0.4, "win_rate": 0.55, "trades": []},
            "config_b": {
                "total_pnl": 6.0,
                "sharpe": 0.5,
                "win_rate": 0.6,
                "trades": [
                    {"pnl_pips": 10},
                    {"pnl_pips": -5},
                    {"pnl_pips": 12},
                ],
            },
        }

    def replay_scan(self, pair, candles, cfg, config_label="default"):
        return SimpleNamespace(
            to_dict=lambda: {"total_pnl": -1.0, "sharpe": 0.1, "win_rate": 0.55, "trades": []}
        )


class _FakeQA:
    def run_full_audit(self):
        return SimpleNamespace(
            score=92.0,
            findings=[SimpleNamespace(severity="info", category="config", message="ok")],
        )


def _candles_ok(pair, days):
    return [{"close": 1.0 + i * 0.001} for i in range(200)]


def _candles_empty(pair, days):
    return []


def _pytest_pass(target):
    return {"passed": True, "exit_code": 0, "failed": [], "stdout_tail": "ok"}


def _pytest_fail(target):
    return {"passed": False, "exit_code": 1, "failed": ["tests/x.py::test_y"], "stdout_tail": "x"}


def test_all_four_subscores_populated(tmp_log, regime_fixture):
    harness = ChangeEvalHarness(
        replay_validator=_FakeReplay(),
        qa_pipeline=_FakeQA(),
        candle_loader=_candles_ok,
        pytest_runner=_pytest_pass,
        regime_windows_path=regime_fixture,
        scorecard_log_path=tmp_log,
        baseline_config={"min_confidence": 0.5},
        pairs=["EUR_USD"],
    )
    card = harness.score(_pkg_with_delta())
    assert card.software_correctness is not None
    assert card.trading_quality is not None
    assert card.governance_quality is not None
    assert card.regime_robustness is not None


def test_software_failure_propagates(tmp_log, regime_fixture):
    harness = ChangeEvalHarness(
        replay_validator=_FakeReplay(),
        qa_pipeline=_FakeQA(),
        candle_loader=_candles_ok,
        pytest_runner=_pytest_fail,
        regime_windows_path=regime_fixture,
        scorecard_log_path=tmp_log,
        baseline_config={"min_confidence": 0.5},
        pairs=["EUR_USD"],
    )
    card = harness.score(_pkg_with_delta())
    assert card.software_correctness is not None
    assert card.software_correctness.passed is False
    assert card.all_passed() is False


def test_trading_skipped_when_no_replay(tmp_log, regime_fixture):
    harness = ChangeEvalHarness(
        replay_validator=None,
        qa_pipeline=_FakeQA(),
        candle_loader=_candles_ok,
        pytest_runner=_pytest_pass,
        regime_windows_path=regime_fixture,
        scorecard_log_path=tmp_log,
    )
    card = harness.score(_pkg_with_delta())
    assert card.trading_quality is not None
    assert card.trading_quality.details.get("skipped") is True


def test_persists_scorecard_jsonl(tmp_log, regime_fixture):
    harness = ChangeEvalHarness(
        replay_validator=_FakeReplay(),
        qa_pipeline=_FakeQA(),
        candle_loader=_candles_ok,
        pytest_runner=_pytest_pass,
        regime_windows_path=regime_fixture,
        scorecard_log_path=tmp_log,
        baseline_config={"min_confidence": 0.5},
        pairs=["EUR_USD"],
    )
    pkg = _pkg_with_delta()
    harness.score(pkg)
    assert tmp_log.exists()
    line = tmp_log.read_text().strip().splitlines()[-1]
    record = json.loads(line)
    assert record["change_id"] == pkg.change_id
    assert "scorecard" in record


def test_governance_critical_finding_fails_subscore(tmp_log, regime_fixture):
    class _CriticalQA:
        def run_full_audit(self):
            return SimpleNamespace(
                score=10.0,
                findings=[SimpleNamespace(severity="critical", category="config", message="bad")],
            )
    harness = ChangeEvalHarness(
        replay_validator=_FakeReplay(),
        qa_pipeline=_CriticalQA(),
        candle_loader=_candles_ok,
        pytest_runner=_pytest_pass,
        regime_windows_path=regime_fixture,
        scorecard_log_path=tmp_log,
    )
    card = harness.score(_pkg_with_delta())
    assert card.governance_quality is not None
    assert card.governance_quality.passed is False


def test_regime_window_floor_breach_fails_subscore(tmp_log, regime_fixture):
    class _LosingReplay(_FakeReplay):
        def replay_scan(self, pair, candles, cfg, config_label="default"):
            return SimpleNamespace(
                to_dict=lambda: {"total_pnl": -10.0, "sharpe": -0.5, "win_rate": 0.2, "trades": []}
            )
    harness = ChangeEvalHarness(
        replay_validator=_LosingReplay(),
        qa_pipeline=_FakeQA(),
        candle_loader=_candles_ok,
        pytest_runner=_pytest_pass,
        regime_windows_path=regime_fixture,
        scorecard_log_path=tmp_log,
    )
    card = harness.score(_pkg_with_delta())
    assert card.regime_robustness is not None
    assert card.regime_robustness.passed is False


def test_empty_candles_skips_per_pair(tmp_log, regime_fixture):
    harness = ChangeEvalHarness(
        replay_validator=_FakeReplay(),
        qa_pipeline=_FakeQA(),
        candle_loader=_candles_empty,
        pytest_runner=_pytest_pass,
        regime_windows_path=regime_fixture,
        scorecard_log_path=tmp_log,
        pairs=["EUR_USD"],
    )
    card = harness.score(_pkg_with_delta())
    assert card.trading_quality is not None
    assert card.trading_quality.details["per_pair"][0]["skipped"] is True
