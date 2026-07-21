from __future__ import annotations

import json
import pickle

from src.axiom_training.policy_advisor import advise_and_append, backfill_policy_advice, build_policy_advice


class _FakeActionModel:
    classes_ = ["observe"]

    def predict(self, features):
        return ["observe" for _ in features]

    def predict_proba(self, features):
        return [[0.91] for _ in features]


class _InvalidActionModel:
    classes_ = ["promote_model_candidate"]

    def predict(self, features):
        return ["promote_model_candidate" for _ in features]

    def predict_proba(self, features):
        return [[0.88] for _ in features]


def _cycle(**overrides):
    base = {
        "cycle_id": "cycle-policy-advice",
        "ts": "2026-07-09T16:30:00+00:00",
        "actor": "test-agent",
        "cli_available": True,
        "autonomy_enabled": False,
        "observations": {
            "health_check": {"ok": True, "data": {"running": True}},
            "gate_health": {"ok": True, "data": {"gate": "PASS", "hard_no_ok": True}},
        },
        "proposed_actions": [
            {"action": "observe", "params": {}, "rationale": "keep watching", "tier": "operational"}
        ],
    }
    base.update(overrides)
    return base


def _write_bundle(tmp_path, model):
    model_path = tmp_path / "action_policy_model.pkl"
    report_path = tmp_path / "action_policy_report.json"
    with model_path.open("wb") as fh:
        pickle.dump({
            "model": model,
            "training_source": "test",
            "label_counts": {"observe": 3},
            "domain_counts": {"operations": 3},
        }, fh)
    report_path.write_text(json.dumps({
        "status": "trained", "replay_pass": True, "stress_pass": True,
    }), encoding="utf-8")
    return model_path, report_path


def test_policy_advisor_reports_agreement_without_runtime_authority(tmp_path) -> None:
    model_path, report_path = _write_bundle(tmp_path, _FakeActionModel())

    advice = build_policy_advice(_cycle(), model_path=model_path, report_path=report_path)

    assert advice.status == "agreement"
    assert advice.policy_mode == "shadow"
    assert advice.runtime_consumes_model is False
    assert advice.agreement_actions == ["observe"]
    assert advice.suggestions[0].predicted_action == "observe"
    assert advice.suggestions[0].confidence == 0.91


def test_policy_advisor_append_is_idempotent_by_cycle(tmp_path) -> None:
    model_path, report_path = _write_bundle(tmp_path, _FakeActionModel())
    advice_path = tmp_path / "policy_advice.jsonl"

    first = advise_and_append(
        _cycle(), model_path=model_path, report_path=report_path, advice_path=advice_path,
    )
    second = advise_and_append(
        _cycle(), model_path=model_path, report_path=report_path, advice_path=advice_path,
    )

    assert first["written"] is True
    assert second["written"] is False
    assert len(advice_path.read_text(encoding="utf-8").splitlines()) == 1


def test_policy_advisor_backfill_counts_statuses_and_existing_rows(tmp_path) -> None:
    model_path, report_path = _write_bundle(tmp_path, _FakeActionModel())
    advice_path = tmp_path / "policy_advice.jsonl"

    first = backfill_policy_advice(
        [_cycle(), _cycle(cycle_id="cycle-policy-advice-2")],
        model_path=model_path,
        report_path=report_path,
        advice_path=advice_path,
    )
    second = backfill_policy_advice(
        [_cycle(), _cycle(cycle_id="cycle-policy-advice-2")],
        model_path=model_path,
        report_path=report_path,
        advice_path=advice_path,
    )

    assert first["written"] == 2
    assert first["by_status"] == {"agreement": 2}
    assert second["written"] == 0
    assert second["skipped_existing"] == 2


def test_policy_advisor_flags_invalid_domain_prediction(tmp_path) -> None:
    model_path, report_path = _write_bundle(tmp_path, _InvalidActionModel())

    advice = build_policy_advice(_cycle(), model_path=model_path, report_path=report_path)

    assert advice.status == "invalid_prediction"
    assert advice.invalid_predictions == ["promote_model_candidate"]
    assert advice.suggestions[0].valid_for_domain is False
