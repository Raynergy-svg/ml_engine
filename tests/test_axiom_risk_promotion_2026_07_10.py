from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.axiom_training.risk_runtime import (
    CHAMPION_MODEL_REL,
    MODEL_REL,
    PEAK_NAV_REL,
    PROMOTION_REL,
    REPORT_REL,
    evaluate_runtime_risk,
    promote_candidate,
    read_promotion_status,
    scale_target_units,
    validate_candidate,
)


def _write_candidate(root, now: datetime) -> None:
    trained_at = now.isoformat()
    model = {
        "model_type": "axiom_risk_controller_v1",
        "features": [
            "trailing_volatility", "drawdown", "loss_streak",
            "portfolio_concentration", "exposure_resolved_fraction",
        ],
        "forbidden_features": ["direction", "future_return", "strategy_identity"],
        "params": {
            "target_vol_multiplier": 1.25,
            "drawdown_soft": 0.06,
            "drawdown_hard": 0.14,
            "loss_streak_decay": 0.55,
            "min_scale": 0.05,
            "max_scale": 0.85,
            "concentration_soft": 0.35,
            "concentration_hard": 0.65,
            "unresolved_exposure_scale": 0.50,
        },
        "target_vol_by_strategy": {"oanda_fx_trend": 0.01},
        "trained_at": trained_at,
        "runtime_allowed": False,
        "shadow_only": True,
    }
    report = {
        "status": "passed",
        "passing_model": True,
        "reasons": [],
        "direction_training": False,
        "trained_at": trained_at,
        "stress_results": {
            "returns_x1": {"passed": True},
            "returns_x1.5": {"passed": True},
        },
        "portfolio_exposure_verification": {"passed": True},
        "adversarial_stress": {"passed": True, "battery_version": "test-v1"},
        "behavioral_invariants": {"passed": True},
    }
    for path, payload in ((root / MODEL_REL, model), (root / REPORT_REL, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_promotion_is_practice_only_and_hash_pinned(tmp_path) -> None:
    now = datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc)
    _write_candidate(tmp_path, now)
    assert validate_candidate(root=tmp_path, now=now).valid is True

    with pytest.raises(ValueError, match="practice-only"):
        promote_candidate(root=tmp_path, environment="live", actor="test", now=now)
    assert not (tmp_path / PROMOTION_REL).exists()

    state = promote_candidate(
        root=tmp_path, environment="practice", actor="test", now=now,
    )
    assert state["enabled"] is True
    assert state["runtime_allowed"] is True
    assert state["shadow_only"] is False
    assert state["de_risk_only"] is True
    assert state["model_path"] == str(CHAMPION_MODEL_REL)
    assert state["model_sha256"]
    assert read_promotion_status(root=tmp_path)["valid"] is True


def test_promoted_runtime_uses_outcomes_and_tamper_fails_closed(tmp_path) -> None:
    now = datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc)
    _write_candidate(tmp_path, now)
    promote_candidate(root=tmp_path, environment="practice", actor="test", now=now)
    feedback = tmp_path / "logs" / "trade_feedback.jsonl"
    feedback.parent.mkdir(parents=True, exist_ok=True)
    feedback.write_text("\n".join(json.dumps({"shaped_reward": value}) for value in [20, -30, -25]) + "\n")
    peak = tmp_path / PEAK_NAV_REL
    peak.parent.mkdir(parents=True, exist_ok=True)
    peak.write_text(json.dumps({"peak_nav": 100_000}), encoding="utf-8")
    exposure = tmp_path / "trained_data" / "hedge" / "exposure_history.jsonl"
    exposure.parent.mkdir(parents=True, exist_ok=True)
    exposure.write_text(json.dumps({
        "open_trade_count": 2,
        "skipped_instruments": [],
        "net_correlation_buckets_notional_home": {"a": 50, "b": -50},
    }) + "\n", encoding="utf-8")

    decision = evaluate_runtime_risk(
        root=tmp_path, environment="practice", nav=95_000, now=now,
    )
    assert decision.promotion_enabled is True
    assert decision.valid is True
    assert 0.0 < decision.scale <= 0.85
    assert decision.drawdown == pytest.approx(0.05)
    assert decision.loss_streak == 2
    assert decision.portfolio_concentration == pytest.approx(0.5)
    assert decision.exposure_resolved_fraction == pytest.approx(1.0)
    assert read_promotion_status(root=tmp_path)["consumed_by_live_sizing"] is True

    # Replacing a future candidate does not disturb the promoted champion.
    candidate_path = tmp_path / MODEL_REL
    candidate_path.write_text(candidate_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    still_live = evaluate_runtime_risk(
        root=tmp_path, environment="practice", nav=95_000, now=now,
    )
    assert still_live.valid is True

    # Tampering with the immutable champion fails closed.
    model_path = tmp_path / CHAMPION_MODEL_REL
    model_path.write_text(model_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    refused = evaluate_runtime_risk(
        root=tmp_path, environment="practice", nav=95_000, now=now,
    )
    assert refused.valid is False
    assert refused.scale == 0.0
    assert "promoted_model_hash_mismatch" in refused.reason


def test_scale_target_units_can_only_reduce() -> None:
    want = {"EUR_USD": 10_001, "USD_JPY": 8_000, "OFF": 0}
    assert scale_target_units(want, 0.85) == {
        "EUR_USD": 8_500, "USD_JPY": 6_800, "OFF": 0,
    }
    assert scale_target_units(want, 1.5) == {"EUR_USD": 0, "USD_JPY": 0, "OFF": 0}


def test_promotion_rejects_challenger_that_exceeds_frozen_risk_floor(tmp_path) -> None:
    now = datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc)
    _write_candidate(tmp_path, now)
    # Establish a conservative champion first.
    promote_candidate(root=tmp_path, environment="practice", actor="test", now=now)

    candidate_path = tmp_path / MODEL_REL
    challenger = json.loads(candidate_path.read_text())
    challenger["params"]["min_scale"] = 0.10
    candidate_path.write_text(json.dumps(challenger), encoding="utf-8")

    with pytest.raises(ValueError, match="minimum_scale_exceeds_safety_cap"):
        promote_candidate(root=tmp_path, environment="practice", actor="test", now=now)

    champion = json.loads((tmp_path / CHAMPION_MODEL_REL).read_text())
    assert champion["params"]["min_scale"] == 0.05
