from __future__ import annotations

def test_us001_inline_peak_nav_save_removed():
    from pathlib import Path
    src = Path(__file__).parent.parent.joinpath(*['src', 'scanner', 'automation', 'continuous.py']).read_text()
    assert 'import json as _json_pn2' not in src, 'inline peak_nav save block still present'
    assert 'self._save_peak_nav()' in src, '_save_peak_nav() call missing from shutdown path'

def test_us002_no_fstring_warning_error_in_continuous():
    from pathlib import Path
    src = Path(__file__).parent.parent.joinpath(*['src', 'scanner', 'automation', 'continuous.py']).read_text()
    bad = [l for l in src.splitlines() if 'logger.warning(f' in l or 'logger.error(f' in l]
    assert not bad, 'f-string loggers remain: ' + str(len(bad))

def test_us003_save_peak_nav_method_callable():
    from src.scanner.automation.continuous import ContinuousScanner
    cs = ContinuousScanner.__new__(ContinuousScanner)
    cs._peak_nav = 99.0
    assert hasattr(cs, '_save_peak_nav')
