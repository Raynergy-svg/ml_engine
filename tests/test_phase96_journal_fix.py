"""Phase 96 tests: journal mtime fix + zero info f-strings."""
import logging
import re
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_us001_except_block_has_mtime_update():
    src = Path("src/scanner/automation/continuous.py").read_text()
    needle = "self._journal_mtime = current_mtime  # US-001"
    assert needle in src


def test_us001_both_paths_update_mtime():
    src = Path("src/scanner/automation/continuous.py").read_text()
    m = re.search(r"def _load_journal_cached.*?(?=\n    def |\Z)", src, re.DOTALL)
    assert m
    count = m.group(0).count("self._journal_mtime = current_mtime")
    assert count >= 2, f"Expected >=2 mtime updates, got {count}"


def test_us001_simulate_parse_failure_sets_mtime():
    src = Path("src/scanner/automation/continuous.py").read_text()
    m = re.search(r"    def _load_journal_cached\(self.*?(?=\n    def |\Z)", src, re.DOTALL)
    assert m
    label = "p96sim"
    hdr = (
        "import json as _json\n"
        + "import logging\n"
        + "logger=logging.getLogger(" + repr(label) + ")\n"
        + "class _CS:\n    _journal_cache=[]\n    _journal_mtime=0.0\n"
    )
    body = ""
    for line in m.group(0).splitlines():
        body += line + "\n"
    ns: dict = {}
    exec(hdr + body, ns)
    obj = ns["_CS"]()
    with tempfile.TemporaryDirectory() as td:
        journal = Path(td) / "trade_journal_rl.json"
        journal.write_text("NOT_VALID_JSON{")
        orig = journal.stat().st_mtime
        with patch.object(logging.getLogger(label), "warning"):
            obj._load_journal_cached(journal)
        assert obj._journal_mtime == orig


def test_us002_zero_info_fstrings():
    src = Path("src/scanner/automation/continuous.py").read_text()
    hits = [l.strip() for l in src.splitlines() if "logger.info(f" in l]
    assert hits == []


def test_us002_shutdown_percent_style():
    src = Path("src/scanner/automation/continuous.py").read_text()
    assert 'logger.info("Shutdown: %s persisted via %s()", mod_label, _method)' in src


def test_us002_config_tuned_percent_style():
    src = Path("src/scanner/automation/continuous.py").read_text()
    assert 'logger.info("Config tuned: %d adjustments applied", len(config_adjustments))' in src


def test_us002_model_promoted_percent_style():
    src = Path("src/scanner/automation/continuous.py").read_text()
    assert 'logger.info("Model %s promoted to production", promotion_version)' in src
