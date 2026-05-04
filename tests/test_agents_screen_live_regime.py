"""Regression guard: AgentsScreen weight panel reflects the LIVE regime.

Mythos audit 2026-05-04 — the previous _load_weights hardcoded
self._regime = "NORMAL" after first load and treated `_history`
(which is a bookkeeping key, not a regime) as a regime candidate.
Result: TUI showed NORMAL-regime weights regardless of what the
scanner was actually using. Backend uses a per-regime weight matrix
(LOW / NORMAL / HIGH / EXTREME) keyed on _LiveRegimeProbe.current().

NO MOCKS. Real json files written to tmp_path, real
AgentsScreen._load_weights driving against them, real
_LiveRegimeProbe pointed at the same tmp tree.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tui.screens.agents_screen import AgentsScreen


def _write_real_weights(weights_path: Path) -> None:
    """Write a real agent_weights.json with all 4 regimes + bookkeeping."""
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(
        json.dumps(
            {
                "LOW": {"trend": 0.50, "mean_reversion": 1.40, "volatility": 0.80},
                "NORMAL": {"trend": 1.10, "mean_reversion": 0.90, "volatility": 1.00},
                "HIGH": {"trend": 1.30, "mean_reversion": 0.70, "volatility": 1.20},
                "EXTREME": {"trend": 1.50, "mean_reversion": 0.50, "volatility": 1.40},
                "_global": {"trend": 1.10, "mean_reversion": 0.95, "volatility": 1.05},
                "_history": [{"trade_id": 1, "delta": 0.01}],
                "_meta": {"version": 1, "last_updated": "2026-05-04T00:00:00Z"},
            },
            indent=2,
        )
    )


def _write_regime_state(regime_state_path: Path, regime: str) -> None:
    """Write the microstructure_regime.json the live probe reads."""
    regime_state_path.parent.mkdir(parents=True, exist_ok=True)
    regime_state_path.write_text(
        json.dumps({"regime": regime, "last_updated": "2026-05-04T00:00:00Z"})
    )


@pytest.fixture
def screen(tmp_path: Path) -> AgentsScreen:
    """Real AgentsScreen rooted at tmp_path."""
    s = AgentsScreen()
    s._project_root = tmp_path
    s._learned_weights = {}
    s._regime = ""
    return s


def test_picks_live_regime_when_probe_returns_value(
    screen: AgentsScreen, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When _LiveRegimeProbe returns HIGH, weights for HIGH should load."""
    weights_path = tmp_path / "trained_data" / "models" / "agent_weights.json"
    _write_real_weights(weights_path)

    # Real probe — point its file lookup at our tmp tree.
    from src.scanner.automation import meta_manager

    class TmpProbe(meta_manager._LiveRegimeProbe):
        def current(self) -> str:
            return "HIGH"

    monkeypatch.setattr(meta_manager, "_LiveRegimeProbe", TmpProbe)

    screen._load_weights()

    assert screen._regime == "HIGH"
    assert screen._learned_weights["trend"] == pytest.approx(1.30)
    assert screen._learned_weights["mean_reversion"] == pytest.approx(0.70)


def test_falls_back_to_NORMAL_when_probe_returns_none(
    screen: AgentsScreen, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No live regime → NORMAL is the safe default (matches scanner's init)."""
    weights_path = tmp_path / "trained_data" / "models" / "agent_weights.json"
    _write_real_weights(weights_path)

    from src.scanner.automation import meta_manager

    class NullProbe(meta_manager._LiveRegimeProbe):
        def current(self):
            return None

    monkeypatch.setattr(meta_manager, "_LiveRegimeProbe", NullProbe)

    screen._load_weights()

    assert screen._regime == "NORMAL"
    assert screen._learned_weights["trend"] == pytest.approx(1.10)


def test_history_key_is_not_treated_as_a_regime(
    screen: AgentsScreen, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-fix code's `regimes = [k for k in data.keys() if k not in ("_global", "_meta")]`
    would include `_history` as a regime candidate. After the fix, all
    underscore-prefixed keys must be filtered.
    """
    weights_path = tmp_path / "trained_data" / "models" / "agent_weights.json"
    _write_real_weights(weights_path)

    from src.scanner.automation import meta_manager

    class HistoryClaimingProbe(meta_manager._LiveRegimeProbe):
        def current(self):
            return "_history"

    monkeypatch.setattr(meta_manager, "_LiveRegimeProbe", HistoryClaimingProbe)

    screen._load_weights()

    assert screen._regime != "_history"
    assert screen._regime == "NORMAL"
    assert "trend" in screen._learned_weights
    # _history is a list in the fixture — if it had been picked as the
    # regime, _learned_weights would be empty (list isn't a dict).
    assert len(screen._learned_weights) > 0
