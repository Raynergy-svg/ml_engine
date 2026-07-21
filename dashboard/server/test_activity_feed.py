from pathlib import Path

from dashboard.server import data_sources as ds


def test_activity_feed_tails_fixed_logs(monkeypatch, tmp_path):
    log_rel = Path("trained_data/axiom/tier7_loop.out")
    log_abs = tmp_path / log_rel
    log_abs.parent.mkdir(parents=True)
    log_abs.write_text("\n".join(f"line {i}" for i in range(12)), encoding="utf-8")

    monkeypatch.setattr(ds, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ds, "ACTIVITY_LOG_FEEDS", [{"id": "tier7", "title": "Tier 7", "path": log_rel}])

    out = ds.read_activity(line_limit=4)

    assert out["log_feeds"][0]["path"] == str(log_rel)
    assert out["log_feeds"][0]["exists"] is True
    assert out["log_feeds"][0]["lines"] == ["line 8", "line 9", "line 10", "line 11"]


def test_activity_feed_degrades_when_log_missing(monkeypatch, tmp_path):
    missing_rel = Path("trained_data/axiom/missing.out")
    monkeypatch.setattr(ds, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ds, "ACTIVITY_LOG_FEEDS", [{"id": "missing", "title": "Missing", "path": missing_rel}])

    out = ds.read_activity(line_limit=4)

    feed = out["log_feeds"][0]
    assert feed["exists"] is False
    assert feed["lines"] == []
    assert feed["age_s"] is None
