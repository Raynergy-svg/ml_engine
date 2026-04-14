"""Phase 97 tests: zero f-string loggers in orchestrator + module_dispatcher."""
from pathlib import Path


def test_us001_zero_fstring_loggers_orchestrator():
    src = Path("src/scanner/automation/orchestrator.py").read_text()
    hits = [l.strip() for l in src.splitlines() if "logger." in l and "(f" + chr(34) in l]
    assert hits == []


def test_us001_rl_sync_percent():
    src = Path("src/scanner/automation/orchestrator.py").read_text()
    assert 'logger.info("RL sync result: %s", sync_result)' in src


def test_us001_archive_percent():
    src = Path("src/scanner/automation/orchestrator.py").read_text()
    assert 'logger.info("US-154: Archived %d old learnings (kept %d)", len(_archive), len(_keep))' in src


def test_us001_prune_percent():
    src = Path("src/scanner/automation/orchestrator.py").read_text()
    assert 'logger.info("US-154: Pruned %d old config adjustments (kept %d)", _removed, len(_pruned))' in src


def test_us002_zero_fstring_loggers_dispatcher():
    src = Path("src/scanner/automation/module_dispatcher.py").read_text()
    hits = [l.strip() for l in src.splitlines() if "logger." in l and "(f" + chr(34) in l]
    assert hits == []


def test_us002_dispatcher_failed_percent():
    src = Path("src/scanner/automation/module_dispatcher.py").read_text()
    assert 'logger.debug("ModuleDispatcher: %s failed: %s", name, e)' in src


def test_us002_dispatcher_save_percent():
    src = Path("src/scanner/automation/module_dispatcher.py").read_text()
    assert 'logger.debug("ModuleDispatcher: Failed to save: %s", e)' in src
