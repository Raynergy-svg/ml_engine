"""AXIOM readout for the market-closed continual-learning loop (item 5).

Read-only, honest-empty-state discipline matching read_crypto_momentum /
read_track_b: no history file yet must render a zero-cycle state, never a
fabricated result.
"""
import json

from dashboard.server import data_sources as ds


def test_read_learning_loop_honest_empty_state_when_never_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "_LEARNING_LOOP_HISTORY_PATH", tmp_path / "history.jsonl")
    monkeypatch.setattr(ds, "_LEARNING_LOOP_STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(ds, "_LEARNING_LOOP_RETRAIN_REQUESTS_DIR", tmp_path / "retrain_requests")

    result = ds.read_learning_loop()

    assert result["has_run"] is False
    assert result["last_cycle"] is None
    assert result["recent_cycles"] == []
    assert result["pending_retrain_markers"] == 0
    assert "directional" in result["scope"]
    # Honest shadow disclosure: nothing consumes the model in live sizing yet.
    assert result["consumed_by_live_sizing"] is False


def test_read_learning_loop_surfaces_pending_markers_and_last_cycle(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "_LEARNING_LOOP_HISTORY_PATH", tmp_path / "history.jsonl")
    monkeypatch.setattr(ds, "_LEARNING_LOOP_STATUS_PATH", tmp_path / "status.json")
    retrain_dir = tmp_path / "retrain_requests"
    retrain_dir.mkdir()
    monkeypatch.setattr(ds, "_LEARNING_LOOP_RETRAIN_REQUESTS_DIR", retrain_dir)

    (retrain_dir / "rl_sizer_1.json").write_text("{}")
    (retrain_dir / "rl_sizer_2.json").write_text("{}")

    history_rows = [
        {"timestamp": "2026-07-01T00:00:00Z", "decision": "rejected", "incumbent_brier": 0.2, "candidate_brier": 0.25},
        {"timestamp": "2026-07-02T00:00:00Z", "decision": "accepted", "incumbent_brier": 0.2, "candidate_brier": 0.12},
    ]
    (tmp_path / "history.jsonl").write_text(
        "\n".join(json.dumps(r) for r in history_rows) + "\n"
    )
    (tmp_path / "status.json").write_text(json.dumps({
        "decision": "accepted", "summary": "offline learning cycle: accepted",
    }))

    result = ds.read_learning_loop()

    assert result["has_run"] is True
    assert result["pending_retrain_markers"] == 2
    assert result["last_cycle"]["decision"] == "accepted"
    assert len(result["recent_cycles"]) == 2
    # Most recent cycle first.
    assert result["recent_cycles"][0]["timestamp"] == "2026-07-02T00:00:00Z"
