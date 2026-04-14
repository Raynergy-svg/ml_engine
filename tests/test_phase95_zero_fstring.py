from __future__ import annotations

def test_zero_debug_fstrings_continuous():
    from pathlib import Path
    src = Path(__file__).parent.parent.joinpath(*['src', 'scanner', 'automation', 'continuous.py']).read_text()
    bad = [l for l in src.splitlines() if 'logger.debug(f' in l]
    assert not bad, 'f-string debug loggers in continuous.py: ' + str(len(bad))

def test_zero_debug_fstrings_orchestrator():
    from pathlib import Path
    src = Path(__file__).parent.parent.joinpath(*['src', 'scanner', 'automation', 'orchestrator.py']).read_text()
    bad = [l for l in src.splitlines() if 'logger.debug(f' in l]
    assert not bad, 'f-string debug loggers in orchestrator.py: ' + str(len(bad))
