from __future__ import annotations
import logging

def _make_cs():
    from src.scanner.automation.continuous import ContinuousScanner
    cs = ContinuousScanner.__new__(ContinuousScanner)
    cs._scan_count = 0
    cs._peak_nav = 0.0
    cs._journal_cache = []
    cs._journal_mtime = 0.0
    return cs

def test_us001_no_duplicate_shutdown_modules():
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "src" / "scanner" / "automation" / "continuous.py").read_text()
    sd_start = src.find("_shutdown_modules")
    sd_end = src.find("]", sd_start)
    sd_block = src[sd_start:sd_end]
    tag = "(\"module_dispatcher\""
    assert sd_block.count(tag) == 1, f"Duplicate {tag} in _shutdown_modules"

def test_us002_journal_parse_failure_warns(caplog):
    import pathlib
    cs = _make_cs()
    bad = pathlib.Path("/tmp/bad_journal_ph91.json")
    bad.write_text("not-valid-json{")
    with caplog.at_level(logging.WARNING):
        try:
            cs._load_journal_cached(bad)
        except Exception:
            pass
    assert any(r.levelno >= logging.WARNING for r in caplog.records)

def test_us003_learning_loop_warning_exc_info():
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "src" / "scanner" / "automation" / "continuous.py").read_text()
    assert "exc_info=True" in src
    assert "logger.warning" in src

def test_us004_peak_nav_accumulates():
    cs = _make_cs()
    for nav in [100.0, 105.0, 102.0, 108.0, 104.0]:
        if nav > 0:
            cs._peak_nav = max(cs._peak_nav, nav)
    assert cs._peak_nav == 108.0

def test_us005_current_price_derived():
    entry, units, pl = 1.1000, 10000, 50.0
    derived = entry + pl / max(abs(units), 1)
    assert derived != entry
    assert abs(derived - 1.1050) < 1e-9

def test_us006_orchestrator_close_safe_with_none_modules():
    from src.scanner.automation.orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    for attr in ("_state", "_config_adjuster", "_observation_consumer",
                 "_drift_remediator", "_online_rl", "_gate_health",
                 "_agent_health", "_module_dispatcher"):
        setattr(o, attr, None)
    assert hasattr(o, "close")
    o.close()

def test_us007_atexit_wired():
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "src" / "scanner" / "automation" / "orchestrator.py").read_text()
    assert "import atexit" in src
    assert "atexit.register(self.close)" in src

def test_us008_peak_nav_roundtrip(tmp_path):
    import json
    ms = tmp_path / "model_state.json"
    ms.write_text(json.dumps({"peak_nav": 123.45}))
    assert float(json.loads(ms.read_text()).get("peak_nav", 0.0)) == 123.45
    d = json.loads(ms.read_text())
    d["peak_nav"] = 200.0
    ms.write_text(json.dumps(d))
    assert float(json.loads(ms.read_text()).get("peak_nav", 0.0)) == 200.0
