from __future__ import annotations

def test_us001_orchestrator_debug_fstrings_reduced():
    from pathlib import Path
    src = Path(__file__).parent.parent.joinpath(*['src', 'scanner', 'automation', 'orchestrator.py']).read_text()
    remaining = [l for l in src.splitlines() if 'logger.debug(f' in l]
    assert len(remaining) <= 5, str(len(remaining))

def test_us002_continuous_debug_fstrings_reduced():
    from pathlib import Path
    src = Path(__file__).parent.parent.joinpath(*['src', 'scanner', 'automation', 'continuous.py']).read_text()
    remaining = [l for l in src.splitlines() if 'logger.debug(f' in l]
    assert len(remaining) <= 30, str(len(remaining))

def test_us003_init_errors_are_warnings():
    from pathlib import Path
    src = Path(__file__).parent.parent.joinpath(*['src', 'scanner', 'automation', 'continuous.py']).read_text()
    frags = ['Portfolio optimizer initialization error', 'Online RL updater initialization error', 'Module dispatcher initialization error']
    for frag in frags:
        hits = [l for l in src.splitlines() if frag in l]
        assert hits, frag + ' not found'
        for l in hits:
            assert 'logger.warning' in l
